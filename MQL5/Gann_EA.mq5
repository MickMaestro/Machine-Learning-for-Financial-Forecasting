//+------------------------------------------------------------------+
//|                                                     Gann_EA.mq5 |
//|                                  Copyright 2025, Your Company   |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Your Company"
#property link      "https://www.mql5.com"
#property version   "1.00"

#include <Trade\Trade.mqh>

//--- Input parameters
input group "=== Gann Settings ==="
input ENUM_TIMEFRAMES TimeFrame = PERIOD_CURRENT;     // Time frame
input int             AvgPeriod = 36;                 // Average period
input double          Filter = 0;                     // Filter to use (<=0 for no filter)
input bool            Interpolate = true;             // Interpolate mtf data ?

input group "=== Trading Settings ==="
input bool            UseFixedLots = true;            // Use fixed lots (false = use percentage of balance)
input double          FixedLotSize = 0.1;            // Fixed lot size
input double          RiskPercent = 2.0;             // Risk percentage of account balance
input int             MagicNumber = 123456;           // Magic number for trades
input string          TradeComment = "Gann EA";       // Trade comment

input group "=== Trailing Stop Settings ==="
input bool            UseTrailingStop = true;        // Use trailing stop
input int             TrailingStopPoints = 50;       // Trailing stop distance in points

//--- Global variables
CTrade trade;
int gann_handle = INVALID_HANDLE;
double gann_fup[], gann_fdn[], gann_mid[], gann_sup[], gann_supc[];

// Variables to prevent re-entries
datetime last_buy_time = 0;
datetime last_sell_time = 0;
double last_signal = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    // Initialize trade object
    trade.SetExpertMagicNumber(MagicNumber);
    trade.SetMarginMode();
    trade.SetTypeFillingBySymbol(_Symbol);
    
    // Create indicator handle
    gann_handle = iCustom(_Symbol, TimeFrame, "Gann_high-low_activator_2", 
                         TimeFrame, AvgPeriod, 1, Filter, 0, false, true, true, false, false, false, Interpolate);
    
    if(gann_handle == INVALID_HANDLE)
    {
        Print("Error creating Gann indicator handle: ", GetLastError());
        return(INIT_FAILED);
    }
    
    // Set buffer arrays
    ArraySetAsSeries(gann_fup, true);
    ArraySetAsSeries(gann_fdn, true);
    ArraySetAsSeries(gann_mid, true);
    ArraySetAsSeries(gann_sup, true);
    ArraySetAsSeries(gann_supc, true);
    
    Print("Gann EA initialised successfully");
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    if(gann_handle != INVALID_HANDLE)
        IndicatorRelease(gann_handle);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    // Copy indicator data
    if(!CopyIndicatorData())
        return;
    
    // Check for position closure first (when signal changes against us)
    CheckForPositionClosure();
    
    // Check for new trade signals only if no position is open
    if(PositionsTotal() == 0)
        CheckForTradeSignals();
    
    // Update trailing stops
    if(UseTrailingStop)
        UpdateTrailingStops();
}

//+------------------------------------------------------------------+
//| Copy indicator data                                               |
//+------------------------------------------------------------------+
bool CopyIndicatorData()
{
    // Copy more bars to analyze trend changes properly
    if(CopyBuffer(gann_handle, 0, 0, 5, gann_fup) <= 0 ||
       CopyBuffer(gann_handle, 1, 0, 5, gann_fdn) <= 0 ||
       CopyBuffer(gann_handle, 2, 0, 5, gann_mid) <= 0 ||
       CopyBuffer(gann_handle, 3, 0, 5, gann_sup) <= 0 ||
       CopyBuffer(gann_handle, 4, 0, 5, gann_supc) <= 0)
    {
        Print("Error copying indicator data: ", GetLastError());
        return false;
    }
    return true;
}

//+------------------------------------------------------------------+
//| Check for position closure when signal changes against us        |
//+------------------------------------------------------------------+
void CheckForPositionClosure()
{
    if(PositionsTotal() == 0)
        return;
    
    if(ArraySize(gann_supc) < 1)
        return;
    
    double current_signal = gann_supc[0];
    
    // Check each position
    for(int i = 0; i < PositionsTotal(); i++)
    {
        if(!PositionSelectByTicket(PositionGetTicket(i)))
            continue;
        
        if(PositionGetString(POSITION_SYMBOL) != _Symbol)
            continue;
        
        if(PositionGetInteger(POSITION_MAGIC) != MagicNumber)
            continue;
        
        ENUM_POSITION_TYPE pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
        ulong ticket = PositionGetInteger(POSITION_TICKET);
        
        bool should_close = false;
        
        // Close buy trade if signal becomes bearish (2)
        if(pos_type == POSITION_TYPE_BUY && current_signal == 2)
        {
            should_close = true;
            Print("Closing buy trade - signal changed to bearish");
        }
        // Close sell trade if signal becomes bullish (1)
        else if(pos_type == POSITION_TYPE_SELL && current_signal == 1)
        {
            should_close = true;
            Print("Closing sell trade - signal changed to bullish");
        }
        
        if(should_close)
        {
            if(trade.PositionClose(ticket))
            {
                Print("Position ", ticket, " closed successfully due to signal change");
            }
            else
            {
                Print("Error closing position ", ticket, ": ", trade.ResultRetcode());
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Check for trade signals                                          |
//+------------------------------------------------------------------+
void CheckForTradeSignals()
{
    // Need at least 4 bars to check for fresh trend changes
    if(ArraySize(gann_supc) < 4)
        return;
    
    // Analyze signals: [0] = current, [1] = previous, [2] = 2 bars ago, [3] = 3 bars ago
    double signal_0 = gann_supc[0]; // Current bar (confirmed)
    double signal_1 = gann_supc[1]; // Previous bar
    double signal_2 = gann_supc[2]; // 2 bars ago
    double signal_3 = gann_supc[3]; // 3 bars ago
    
    // Get current time for preventing re-entries
    datetime current_time = TimeCurrent();
    
    // Detect FRESH BULLISH trend change:
    // Current signal = 1 (bullish) AND previous signals were bearish (2) for at least 2 bars
    bool fresh_bullish_trend = (signal_0 == 1) && (signal_1 == 2) && (signal_2 == 2);
    
    // Detect FRESH BEARISH trend change:
    // Current signal = 2 (bearish) AND previous signals were bullish (1) for at least 2 bars  
    bool fresh_bearish_trend = (signal_0 == 2) && (signal_1 == 1) && (signal_2 == 1);
    
    // Enter BUY on fresh bullish trend change
    if(fresh_bullish_trend)
    {
        // Only open buy if we haven't opened a buy recently (prevent re-entries)
        if(current_time - last_buy_time > PeriodSeconds(PERIOD_H1)) // Wait at least 1 hour
        {
            Print("Fresh bullish trend detected - entering BUY trade");
            OpenBuyTrade();
            last_buy_time = current_time;
        }
        else
        {
            Print("Fresh bullish trend detected but buy re-entry prevention active");
        }
    }
    
    // Enter SELL on fresh bearish trend change
    if(fresh_bearish_trend)
    {
        // Only open sell if we haven't opened a sell recently (prevent re-entries)
        if(current_time - last_sell_time > PeriodSeconds(PERIOD_H1)) // Wait at least 1 hour
        {
            Print("Fresh bearish trend detected - entering SELL trade");
            OpenSellTrade();
            last_sell_time = current_time;
        }
        else
        {
            Print("Fresh bearish trend detected but sell re-entry prevention active");
        }
    }
    
    // Store last signal for reference
    last_signal = signal_0;
}

//+------------------------------------------------------------------+
//| Open buy trade                                                   |
//+------------------------------------------------------------------+
void OpenBuyTrade()
{
    double lot_size = CalculateLotSize();
    double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double sl = gann_fdn[0]; // Set stop loss at lower Gann line
    double tp = 0; // No take profit, use trailing stop
    
    if(trade.Buy(lot_size, _Symbol, ask, sl, tp, TradeComment))
    {
        Print("Buy order opened at ", ask, " with lot size ", lot_size);
    }
    else
    {
        Print("Error opening buy order: ", trade.ResultRetcode());
    }
}

//+------------------------------------------------------------------+
//| Open sell trade                                                  |
//+------------------------------------------------------------------+
void OpenSellTrade()
{
    double lot_size = CalculateLotSize();
    double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double sl = gann_fup[0]; // Set stop loss at upper Gann line
    double tp = 0; // No take profit, use trailing stop
    
    if(trade.Sell(lot_size, _Symbol, bid, sl, tp, TradeComment))
    {
        Print("Sell order opened at ", bid, " with lot size ", lot_size);
    }
    else
    {
        Print("Error opening sell order: ", trade.ResultRetcode());
    }
}

//+------------------------------------------------------------------+
//| Calculate lot size based on user settings                        |
//+------------------------------------------------------------------+
double CalculateLotSize()
{
    double lot_size;
    
    if(UseFixedLots)
    {
        lot_size = FixedLotSize;
    }
    else
    {
        // Calculate lot size based on percentage of account balance
        double account_balance = AccountInfoDouble(ACCOUNT_BALANCE);
        double risk_amount = account_balance * RiskPercent / 100.0;
        
        // Calculate lot size based on risk amount and stop loss distance
        double sl_distance = MathAbs(gann_fup[0] - gann_fdn[0]);
        double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
        double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
        
        if(sl_distance > 0 && tick_value > 0 && tick_size > 0)
        {
            double sl_points = sl_distance / tick_size;
            lot_size = risk_amount / (sl_points * tick_value);
        }
        else
        {
            lot_size = FixedLotSize; // Fallback to fixed lot size
        }
    }
    
    // Normalize lot size
    double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double max_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    double lot_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
    
    lot_size = MathMax(min_lot, MathMin(max_lot, lot_size));
    lot_size = NormalizeDouble(lot_size / lot_step, 0) * lot_step;
    
    return lot_size;
}

//+------------------------------------------------------------------+
//| Update trailing stops                                            |
//+------------------------------------------------------------------+
void UpdateTrailingStops()
{
    for(int i = 0; i < PositionsTotal(); i++)
    {
        if(!PositionSelectByTicket(PositionGetTicket(i)))
            continue;
        
        if(PositionGetString(POSITION_SYMBOL) != _Symbol)
            continue;
        
        if(PositionGetInteger(POSITION_MAGIC) != MagicNumber)
            continue;
        
        ENUM_POSITION_TYPE pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
        double pos_open_price = PositionGetDouble(POSITION_PRICE_OPEN);
        double pos_sl = PositionGetDouble(POSITION_SL);
        double pos_tp = PositionGetDouble(POSITION_TP);
        ulong ticket = PositionGetInteger(POSITION_TICKET);
        
        double new_sl = 0;
        bool modify_needed = false;
        
        if(pos_type == POSITION_TYPE_BUY)
        {
            // For buy positions, trail stop loss based on lower Gann line
            new_sl = gann_fdn[0];
            
            // Only move stop loss up (never down)
            if(new_sl > pos_sl + TrailingStopPoints * _Point || pos_sl == 0)
            {
                modify_needed = true;
            }
        }
        else if(pos_type == POSITION_TYPE_SELL)
        {
            // For sell positions, trail stop loss based on upper Gann line
            new_sl = gann_fup[0];
            
            // Only move stop loss down (never up)
            if(new_sl < pos_sl - TrailingStopPoints * _Point || pos_sl == 0)
            {
                modify_needed = true;
            }
        }
        
        if(modify_needed)
        {
            if(trade.PositionModify(ticket, new_sl, pos_tp))
            {
                Print("Trailing stop updated for ticket ", ticket, " to ", new_sl);
            }
            else
            {
                Print("Error updating trailing stop for ticket ", ticket, ": ", trade.ResultRetcode());
            }
        }
    }
}

