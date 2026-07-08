//+------------------------------------------------------------------+
//|                                                  Gann Only EA.mq5|
//|                                             Copyright 2025, IBK. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright      "Copyright 2025, Ibukunoluwa Micahel Adebanjo"
#property description    "Trades based on two Moving Averages and Gann Moving Average"
#property strict

// Trade class
#include <Trade/Trade.mqh>
CTrade   Trade;

enum RM
  {
   Fix_lots,               // Fixed Lots
   Risk_per_Balance,       // Percentage of Balance
  };

// EA general settings
input group "EA General Settings"
input int                  MagicNumber                   =  449001;                                      // Expert Magic Number
input string               TradeComment                  =  "";                             // Expert Trades Comment
input int                  MaxSlippage                   =  5;                                           // Max. Slippage (Points)

// Position management settings
input group "Position Management Settings"
input RM                   RiskMethod                    =  Fix_lots;                                    // Risk Method
input double               FixLots                       =  0.01;                                        // Fixed Lot Size
input double               TradeRisk                     =  1;                                           // Risk % per Trade
input bool                 AvoidFridayTrades             =  true;                                        // Avoid Trading on Fridays

// Gann indicator settings
input group "Gann Indicator Settings"
input ENUM_TIMEFRAMES      TimeFrame                     =  PERIOD_CURRENT;                             // Time frame
input int                  AvgPeriod                     =  36;                                          // Average period
input int                  AvgType                       =  1;                                           // Average method (0=SMA, 1=EMA, 2=SMMA, 3=LWMA)
input double               Filter                        =  0;                                           // Filter to use (<=0 for no filter)
input int                  FilterOn                      =  0;                                           // Filter on (0=prices, 1=averages, 2=all)

// EA variables
int      Space                =  20;
int      PipValue             =  0;
bool     IsNewBar             =  false;

// Performance tracking variables for CSV generation
double totalProfit;
double totalLoss;
double maxDrawdown;
double currentDrawdown;
double peakBalance;
int totalTrades;
int winningTrades;
int losingTrades;
double initialBalance;
datetime backTestStartTime;

// Gann Indicator Handle and Buffers
double   GannSignal[];                       // Dynamic array to hold the Gann signal buffer values
double   GannLine[];                         // Dynamic array to hold the Gann line buffer values
double   GannUp[];                           // Dynamic array to hold the Gann upper zone buffer values
double   GannDown[];                         // Dynamic array to hold the Gann lower zone buffer values
int      GannHandle;                         // Handle for Gann Indicator

// EMA Filter Handle and Buffers
double   EMA1[];                             // Dynamic arrays to hold the EMA 1 buffer values for each bars
int      EMA1Handle;                         // Handle for EMA 1 Indicator
double   EMA2[];                             // Dynamic arrays to hold the EMA 2 buffer values for each bars
int      EMA2Handle;                         // Handle for EMA 2 Indicator

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   IsNewBar       =  NewBar();

   // Initialize performance tracking variables
   totalProfit = 0.0;
   totalLoss = 0.0;
   maxDrawdown = 0.0;
   currentDrawdown = 0.0;
   peakBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   initialBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   totalTrades = 0;
   winningTrades = 0;
   losingTrades = 0;
   backTestStartTime = TimeCurrent();

   // Get handle for Gann indicator with parameters
   GannHandle  =  iCustom(Symbol(), PERIOD_CURRENT, "Gann_high-low_activator_2",
                         TimeFrame, AvgPeriod, AvgType, Filter, FilterOn,
                         false, true, false, false, false, false, true);

   if(GannHandle < 0)
     {
      Alert("Error Creating Handles for Gann indicator - error: ", GetLastError(), "!!");
      return(INIT_FAILED);
     }

   ArraySetAsSeries(GannSignal, true);
   ArraySetAsSeries(GannLine, true);
   ArraySetAsSeries(GannUp, true);
   ArraySetAsSeries(GannDown, true);

   ArrayInitialize(GannSignal, EMPTY_VALUE);
   ArrayInitialize(GannLine, EMPTY_VALUE);
   ArrayInitialize(GannUp, EMPTY_VALUE);
   ArrayInitialize(GannDown, EMPTY_VALUE);

   // Get handle for EMA 1 indicator (200 EMA of MEDIAN price)
   EMA1Handle  =  iMA(Symbol(), PERIOD_CURRENT, 200, 0, MODE_EMA, PRICE_MEDIAN);

   if(EMA1Handle < 0)
     {
      Alert("Error Creating Handles for EMA 1 indicator - error: ", GetLastError(), "!!");
      return(INIT_FAILED);
     }

   // Get handle for EMA 2 indicator (200 EMA of LOW price)
   EMA2Handle  =  iMA(Symbol(), PERIOD_CURRENT, 200, 0, MODE_EMA, PRICE_LOW);

   if(EMA2Handle < 0)
     {
      Alert("Error Creating Handles for EMA 2 indicator - error: ", GetLastError(), "!!");
      return(INIT_FAILED);
     }

   ArraySetAsSeries(EMA1, true);
   ArraySetAsSeries(EMA2, true);

   ArrayInitialize(EMA1, EMPTY_VALUE);
   ArrayInitialize(EMA2, EMPTY_VALUE);

   // Reading the indicator values
   RefreshGann(3);
   RefreshEMA(3);

   Trade.SetExpertMagicNumber(MagicNumber);
   Trade.SetDeviationInPoints(MaxSlippage);

//---
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   // Generate CSV file with backtest results before cleanup
   GenerateBacktestCSV();

   IndicatorRelease(GannHandle);
   IndicatorRelease(EMA1Handle);
   IndicatorRelease(EMA2Handle);
   Comment("");
   ChartRedraw(0);
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   // Refresh Gann data on every tick for trailing stops
   RefreshGann(3);
   RefreshEMA(3);

   // Update performance tracking
   UpdatePerformanceMetrics();

   // Trail stop losses for existing positions
   TrailStopLoss();

   // Signal candles are investigated after being closed
   IsNewBar    =  NewBar();

   if(IsNewBar)
     {
      // Close open trades based on Gann color change (trend reversal)
      if(PositionCounter(0) > 0 && GannColorChanged() && GannSignal[0] == 2)
         CloseOpenTrades(0);

      if(PositionCounter(1) > 0 && GannColorChanged() && GannSignal[0] == 1)
         CloseOpenTrades(1);

      // Take new positions based on Gann trend change + EMA filter (avoid Fridays if enabled)
      if(PositionCounter(2) == 0 && !IsFridayTradeFilter() && GannTrendChangeEntrySignal(1) == "Buy")
         TakePosition(0);

      if(PositionCounter(2) == 0 && !IsFridayTradeFilter() && GannTrendChangeEntrySignal(1) == "Sell")
         TakePosition(1);
     }
  }

//+------------------------------------------------------------------+
//| Open Positions Counter                                           |
//+------------------------------------------------------------------+
int PositionCounter(int type)
  {
   int  _count   =  0;

   if(type == 0)
     {
      for(int i=PositionsTotal()-1; i>=0; i--)
        {
         ulong   _ticket  =  PositionGetTicket(i);
         if(_ticket > 0)
           {
            if(PositionGetString(POSITION_SYMBOL) == Symbol() && PositionGetInteger(POSITION_MAGIC) == MagicNumber)
              {
               if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
                  _count++;
              }
           }
        }
     }

   else
      if(type == 1)
        {
         for(int i=PositionsTotal()-1; i>=0; i--)
           {
            ulong   _ticket  =  PositionGetTicket(i);
            if(_ticket > 0)
              {
               if(PositionGetString(POSITION_SYMBOL) == Symbol() && PositionGetInteger(POSITION_MAGIC) == MagicNumber)
                 {
                  if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL)
                     _count++;
                 }
              }
           }
        }

      else
        {
         for(int i=PositionsTotal()-1; i>=0; i--)
           {
            ulong   _ticket  =  PositionGetTicket(i);
            if(_ticket > 0)
              {
               if(PositionGetString(POSITION_SYMBOL) == Symbol() && PositionGetInteger(POSITION_MAGIC) == MagicNumber)
                  _count++;
              }
           }
        }

   return _count;
  }

//+------------------------------------------------------------------+
//| Close Open Positions                                             |
//+------------------------------------------------------------------+
void CloseOpenTrades(int type)
  {
   if(type == 0)
     {
      for(int i=PositionsTotal()-1; i>=0; i--)
        {
         ulong _pos_ticket   = PositionGetTicket(i);
         if(_pos_ticket>0)
           {
            if(PositionGetString(POSITION_SYMBOL) == Symbol() && PositionGetInteger(POSITION_MAGIC) == MagicNumber && PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
               Trade.PositionClose(_pos_ticket);
           }
        }
     }

   else
      if(type == 1)
        {
         for(int i=PositionsTotal()-1; i>=0; i--)
           {
            ulong _pos_ticket   = PositionGetTicket(i);
            if(_pos_ticket>0)
              {
               if(PositionGetString(POSITION_SYMBOL) == Symbol() && PositionGetInteger(POSITION_MAGIC) == MagicNumber && PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL)
                  Trade.PositionClose(_pos_ticket);
              }
           }
        }

      else
        {
         for(int i=PositionsTotal()-1; i>=0; i--)
           {
            ulong _pos_ticket   = PositionGetTicket(i);
            if(_pos_ticket>0)
              {
               if(PositionGetString(POSITION_SYMBOL) == Symbol() && PositionGetInteger(POSITION_MAGIC) == MagicNumber)
                  Trade.PositionClose(_pos_ticket);
              }
           }
        }
  }

//+------------------------------------------------------------------+
//| New Bar First Tick Control                                       |
//+------------------------------------------------------------------+
bool NewBar()
   {
    static  datetime previous_time  =  0;
    datetime         current_time   =  iTime(Symbol(), Period(), 0);
    if(previous_time != current_time)
      {
       previous_time =  current_time;
       return (true);
      }
      
    return(false);
   }

//+------------------------------------------------------------------+
//| Refresh EMA Indicators Values                                    |
//+------------------------------------------------------------------+
void RefreshEMA(int copy)
   {
    ResetLastError();

    // Get EMA 1 signals data from handle
    int  ma1Count       =  CopyBuffer(EMA1Handle, 0, 0, copy, EMA1);
        
    if(ma1Count < 0)
      {
       int  ma1Error    =  GetLastError();
       Print("EMA 1 indicator handle failed to copy data!! Error code: ", ma1Error);
      }

    // Get EMA 2 signals data from handle
    int  ma2Count       =  CopyBuffer(EMA2Handle, 0, 0, copy, EMA2);
        
    if(ma2Count < 0)
      {
       int  ma2Error    =  GetLastError();
       Print("EMA 2 indicator handle failed to copy data!! Error code: ", ma2Error);
      }
   }

//+------------------------------------------------------------------+
//| EMA Filter Check - Now checks where trend change occurred       |
//+------------------------------------------------------------------+
bool EMAFilter(int bar, string direction)
   {
    // Check if we have valid EMA data
    if(ArraySize(EMA1) <= bar || ArraySize(EMA2) <= bar)
       return false;
    
    // Get the Gann line value where the trend change occurred (previous bar)
    double trendChangePrice = GannLine[bar];
    double ema1 = EMA1[bar];
    double ema2 = EMA2[bar];
    
    // Validate data
    if(trendChangePrice <= 0 || ema1 == EMPTY_VALUE || ema2 == EMPTY_VALUE)
       return false;
    
    if(direction == "Buy")
      {
       // For BUY: trend change must have occurred above the MEDIAN EMA
       return (trendChangePrice > ema1);
      }
    else if(direction == "Sell")
      {
       // For SELL: trend change must have occurred below the LOW EMA
       return (trendChangePrice < ema2);
      }
    
    return false;
   }

//+------------------------------------------------------------------+
//| Refresh Gann Indicator Values                                    |
//+------------------------------------------------------------------+
void RefreshGann(int copy)
   {
    ResetLastError();

    // Get Gann signal data from handle (buffer 4 - color/signal buffer)
    int  gannSignalCount = CopyBuffer(GannHandle, 4, 0, copy, GannSignal);
        
    if(gannSignalCount < 0)
      {
       int  gannSignalError = GetLastError();
       Print("Gann signal buffer failed to copy data!! Error code: ", gannSignalError);
      }

    // Get Gann line data from handle (buffer 3 - the actual line values)
    int  gannLineCount = CopyBuffer(GannHandle, 3, 0, copy, GannLine);
        
    if(gannLineCount < 0)
      {
       int  gannLineError = GetLastError();
       Print("Gann line buffer failed to copy data!! Error code: ", gannLineError);
      }

    // Get Gann upper zone data from handle (buffer 0 - upper zone)
    int  gannUpCount = CopyBuffer(GannHandle, 0, 0, copy, GannUp);
        
    if(gannUpCount < 0)
      {
       int  gannUpError = GetLastError();
       Print("Gann upper zone buffer failed to copy data!! Error code: ", gannUpError);
      }

    // Get Gann lower zone data from handle (buffer 1 - lower zone)
    int  gannDownCount = CopyBuffer(GannHandle, 1, 0, copy, GannDown);
        
    if(gannDownCount < 0)
      {
       int  gannDownError = GetLastError();
       Print("Gann lower zone buffer failed to copy data!! Error code: ", gannDownError);
      }
   }

//+------------------------------------------------------------------+
//| Gann Trend Change Entry Signal (SIMPLIFIED - No Gray Zone)      |
//+------------------------------------------------------------------+
string GannTrendChangeEntrySignal(int bar)
   {
    // Check if we have valid data for current and previous bars
    if(ArraySize(GannSignal) <= bar+1)
       return "Neutral";
    
    double currentGannSignal = GannSignal[bar];
    double previousGannSignal = GannSignal[bar+1];
    
    // Validate data
    if(currentGannSignal == EMPTY_VALUE || previousGannSignal == EMPTY_VALUE)
       return "Neutral";
    
    // Buy signal: Gann changed from bearish (2) to bullish (1) AND trend change occurred above MEDIAN EMA
    if(previousGannSignal == 2 && currentGannSignal == 1 && EMAFilter(bar, "Buy"))
      {
       Print("BUY Signal: Gann changed from bearish to bullish above MEDIAN EMA");
       return "Buy";
      }

    // Sell signal: Gann changed from bullish (1) to bearish (2) AND trend change occurred below LOW EMA
    if(previousGannSignal == 1 && currentGannSignal == 2 && EMAFilter(bar, "Sell"))
      {
       Print("SELL Signal: Gann changed from bullish to bearish below LOW EMA");
       return "Sell";
      }

    return "Neutral";
   }

//+------------------------------------------------------------------+
//| Check if Today is Friday                                         |
//+------------------------------------------------------------------+
bool IsFridayTradeFilter()
   {
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);
    return (dt.day_of_week == 5); // 5 = Friday (0=Sunday, 1=Monday, ..., 6=Saturday)
   }
bool GannColorChanged()
   {
    // Check if we have valid data for current and previous bar
    if(ArraySize(GannSignal) <= 1)
       return false;
       
    // Return true if color changed from previous bar to current bar
    return (GannSignal[0] != GannSignal[1]);
   }

//+------------------------------------------------------------------+
//| Trail Stop Loss Based on Gann Lines                             |
//+------------------------------------------------------------------+
void TrailStopLoss()
   {
    // Check if we have valid Gann line data
    if(ArraySize(GannLine) <= 0)
       return;
       
    double currentGannLine = GannLine[0];
    
    // Validate Gann line value
    if(currentGannLine <= 0.0 || currentGannLine == EMPTY_VALUE)
       return;
    
    // Loop through all positions
    for(int i = PositionsTotal()-1; i >= 0; i--)
      {
       ulong ticket = PositionGetTicket(i);
       if(ticket > 0)
         {
          if(PositionGetString(POSITION_SYMBOL) == Symbol() && 
             PositionGetInteger(POSITION_MAGIC) == MagicNumber)
            {
             double currentSL = PositionGetDouble(POSITION_SL);
             double newSL = NormalizeDouble(currentGannLine, (int)SymbolInfoInteger(Symbol(), SYMBOL_DIGITS));
             
             // For BUY positions
             if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
               {
                // Trail stop loss up if new Gann line is higher than current SL
                if(newSL > currentSL)
                  {
                   if(!Trade.PositionModify(ticket, newSL, PositionGetDouble(POSITION_TP)))
                     {
                      Print("Error modifying BUY position SL. Error: ", GetLastError());
                     }
                   else
                     {
                      Print("BUY position SL trailed from ", currentSL, " to ", newSL);
                     }
                  }
               }
             
             // For SELL positions  
             else if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL)
               {
                // Trail stop loss down if new Gann line is lower than current SL
                if(newSL < currentSL || currentSL == 0.0)
                  {
                   if(!Trade.PositionModify(ticket, newSL, PositionGetDouble(POSITION_TP)))
                     {
                      Print("Error modifying SELL position SL. Error: ", GetLastError());
                     }
                   else
                     {
                      Print("SELL position SL trailed from ", currentSL, " to ", newSL);
                     }
                  }
               }
            }
         }
      }
   }

//+------------------------------------------------------------------+
//| SL Calculator - Uses Gann Line from Previous Candle             |
//+------------------------------------------------------------------+
double SLCalculator(int type)
   {
    double  _sl = 0.0;
    
    // Check if we have valid Gann line data
    if(ArraySize(GannLine) <= 1)
      {
       Print("Error: Not enough Gann line data for SL calculation");
       return 0.0;
      }
    
    // Use the Gann line value from the previous candle (index 1) as stop loss
    _sl = GannLine[1];
    
    // Validate the stop loss value
    if(_sl <= 0.0 || _sl == EMPTY_VALUE)
      {
       Print("Error: Invalid Gann line value for SL: ", _sl);
       return 0.0;
      }
    
    _sl = NormalizeDouble(_sl, (int) SymbolInfoInteger(Symbol(), SYMBOL_DIGITS));
    return _sl;
   }

//+------------------------------------------------------------------+
//| Lot Size Calculator                                              |
//+------------------------------------------------------------------+
double LotSizeCalculator(double _sl, double entryPrice)
   {
    double  _lot  =  0.0;
    
    if(RiskMethod == Fix_lots)
       _lot =  FixLots;
    
    else
      {
       double  _lot_min          =  SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MIN);
       double  _lot_max          =  SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MAX);
       double  _lot_step         =  SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_STEP);
       double  _lot_tick_size    =  SymbolInfoDouble(Symbol(), SYMBOL_TRADE_TICK_SIZE);
       double  _lot_tick_value   =  SymbolInfoDouble(Symbol(), SYMBOL_TRADE_TICK_VALUE);
       double  _balance_value    =  AccountInfoDouble(ACCOUNT_BALANCE);

       double  _risk_value       =  _balance_value* TradeRisk/ 100;
       double  _loss_order       =  MathAbs(entryPrice- _sl)* _lot_tick_value/ _lot_tick_size;
     
       _lot =  MathMin(_lot_max, MathMax(_lot_min, MathRound(_risk_value/ _loss_order/ _lot_step)* _lot_step));       
      }
    
    NormalizeDouble(_lot, 2);
    return _lot;
   }

//+------------------------------------------------------------------+
//| Take Position                                                    |
//+------------------------------------------------------------------+
bool TakePosition(int type)
   {
    if(type == 0)
      {
       double     _price   =  SymbolInfoDouble(Symbol(), SYMBOL_ASK);
       double     _sl      =  SLCalculator(0);
       double     _tp      =  0;
       
       // Validate stop loss before calculating lot size
       if(_sl <= 0.0 || _sl >= _price)
         {
          Print("Invalid stop loss for BUY trade. SL: ", _sl, " Price: ", _price);
          return false;
         }
       
       double     _lot     =  LotSizeCalculator(_sl, _price);

       if(Trade.Buy(_lot, Symbol(), _price, _sl, _tp, TradeComment))
          return true;
      }

    if(type == 1)
      {       
       double     _price   =  SymbolInfoDouble(Symbol(), SYMBOL_BID);
       double     _sl      =  SLCalculator(1);
       double     _tp      =  0;
       
       // Validate stop loss before calculating lot size
       if(_sl <= 0.0 || _sl <= _price)
         {
          Print("Invalid stop loss for SELL trade. SL: ", _sl, " Price: ", _price);
          return false;
         }
       
       double     _lot     =  LotSizeCalculator(_sl, _price);

       if(Trade.Sell(_lot, Symbol(), _price, _sl, _tp, TradeComment))
          return true;
      }
    
    Print("EA tried to open the trade @", TimeCurrent(), " but one of the position parameters is incorrect! Check trade parameters.");
    return false;
   }

//+------------------------------------------------------------------+
//| Update performance tracking on each tick                        |
//+------------------------------------------------------------------+
void UpdatePerformanceMetrics()
  {
   double currentBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   double currentEquity = AccountInfoDouble(ACCOUNT_EQUITY);

   // Update peak balance and drawdown
   if(currentEquity > peakBalance)
     {
      peakBalance = currentEquity;
     }

   currentDrawdown = (peakBalance - currentEquity) / peakBalance * 100.0;
   if(currentDrawdown > maxDrawdown)
     {
      maxDrawdown = currentDrawdown;
     }

   // Count closed trades and their P&L
   static int lastHistoryTotal = 0;
   int currentHistoryTotal = HistoryDealsTotal();

   if(currentHistoryTotal > lastHistoryTotal)
     {
      // New deal(s) closed - update statistics
      for(int i = lastHistoryTotal; i < currentHistoryTotal; i++)
        {
         ulong ticket = HistoryDealGetTicket(i);
         if(ticket > 0)
           {
            string symbol = HistoryDealGetString(ticket, DEAL_SYMBOL);
            long magic = HistoryDealGetInteger(ticket, DEAL_MAGIC);
            long type = HistoryDealGetInteger(ticket, DEAL_TYPE);

            if(symbol == Symbol() && magic == MagicNumber &&
               (type == DEAL_TYPE_BUY || type == DEAL_TYPE_SELL))
              {

               double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT);
               if(profit != 0) // Exclude entry trades (profit = 0)
                 {
                  totalTrades++;
                  if(profit > 0)
                    {
                     totalProfit += profit;
                     winningTrades++;
                    }
                  else
                    {
                     totalLoss += MathAbs(profit);
                     losingTrades++;
                    }
                 }
              }
           }
        }
      lastHistoryTotal = currentHistoryTotal;
     }
  }

//+------------------------------------------------------------------+
//| Generate CSV file with backtest results                         |
//+------------------------------------------------------------------+
void GenerateBacktestCSV()
  {
   // Extract symbol and timeframe
   string symbol = Symbol();
   string timeframe = TimeFrameToString(Period());

   // Create filename with "Baseline" instead of ML algorithm
   string filename = StringFormat("%s_%s_Baseline_BacktestResults.csv", symbol, timeframe);

   // Calculate final metrics
   double netProfitLoss = totalProfit - totalLoss;
   double winRate = totalTrades > 0 ? (double)winningTrades / totalTrades * 100.0 : 0.0;
   double sharpeRatio = CalculateSharpeRatio(netProfitLoss);

   // Save to Common folder to avoid Agent sandbox - accessible from main terminal
   string subfolder = "BacktestResults\\";
   string fullPath = subfolder + filename;

   // Open file for writing in COMMON folder (FILE_COMMON flag)
   int fileHandle = FileOpen(fullPath, FILE_WRITE | FILE_CSV | FILE_REWRITE | FILE_COMMON);
   if(fileHandle != INVALID_HANDLE)
     {
      // Write CSV header
      FileWrite(fileHandle, "Metric", "Value");

      // Write performance metrics
      FileWrite(fileHandle, "Symbol", symbol);
      FileWrite(fileHandle, "Timeframe", timeframe);
      FileWrite(fileHandle, "Strategy", "Baseline");
      FileWrite(fileHandle, "Net Profit/Loss", DoubleToString(netProfitLoss, 2));
      FileWrite(fileHandle, "Max Drawdown (%)", DoubleToString(maxDrawdown, 2));
      FileWrite(fileHandle, "Win Rate (%)", DoubleToString(winRate, 2));
      FileWrite(fileHandle, "Sharpe Ratio", DoubleToString(sharpeRatio, 4));
      FileWrite(fileHandle, "Total Trades", IntegerToString(totalTrades));
      FileWrite(fileHandle, "Winning Trades", IntegerToString(winningTrades));
      FileWrite(fileHandle, "Losing Trades", IntegerToString(losingTrades));
      FileWrite(fileHandle, "Total Profit", DoubleToString(totalProfit, 2));
      FileWrite(fileHandle, "Total Loss", DoubleToString(totalLoss, 2));
      FileWrite(fileHandle, "Initial Balance", DoubleToString(initialBalance, 2));
      FileWrite(fileHandle, "Final Balance", DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2));

      FileClose(fileHandle);
      Print("Baseline backtest results saved to Common Files: ", subfolder, filename);
     }
   else
     {
      Print("Failed to create CSV file: ", fullPath);
     }
  }

//+------------------------------------------------------------------+
//| Convert timeframe enum to string                                |
//+------------------------------------------------------------------+
string TimeFrameToString(ENUM_TIMEFRAMES timeframe)
  {
   // Handle PERIOD_CURRENT by getting the actual chart timeframe
   if(timeframe == PERIOD_CURRENT)
     {
      timeframe = Period();
     }

   switch(timeframe)
     {
      case PERIOD_M1:  return "M1";
      case PERIOD_M5:  return "M5";
      case PERIOD_M15: return "M15";
      case PERIOD_M30: return "M30";
      case PERIOD_H1:  return "H1";
      case PERIOD_H4:  return "H4";
      case PERIOD_D1:  return "D1";
      case PERIOD_W1:  return "W1";
      case PERIOD_MN1: return "MN1";
      default: return "TF" + IntegerToString(timeframe);
     }
  }

//+------------------------------------------------------------------+
//| Calculate Sharpe ratio (simplified)                             |
//+------------------------------------------------------------------+
double CalculateSharpeRatio(double netProfit)
  {
   if(totalTrades < 2) return 0.0;

   // Simplified Sharpe calculation
   double avgProfit = netProfit / totalTrades;
   double riskFreeRate = 0.02; // Assume 2% risk-free rate

   // Calculate standard deviation of returns (simplified)
   double variance = 0.0;
   double avgReturn = avgProfit;

   // This is a simplified calculation - in real implementation,
   // you'd track individual trade returns for proper standard deviation
   if(totalTrades > 0)
     {
      variance = MathPow(totalProfit / totalTrades - totalLoss / totalTrades, 2);
     }

   double stdDev = MathSqrt(variance);
   if(stdDev == 0) return 0.0;

   return (avgReturn - riskFreeRate) / stdDev;
  }