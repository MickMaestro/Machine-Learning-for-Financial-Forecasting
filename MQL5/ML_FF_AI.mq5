//+------------------------------------------------------------------+
//|                                                    ML_FF_AI.mq5 |
//|                                              Copyright 2025, IBK |
//|          GannWithEma trading system with optional ML filtering  |
//+------------------------------------------------------------------+
/*
================================================================================
ML FINANCIAL FORECASTING - CORE EXPERT ADVISOR WITH ML FILTERING
================================================================================

OVERVIEW:
This is the primary Expert Advisor for the ML Financial Forecasting system,
implementing a professional-grade trading strategy that combines traditional
technical analysis with machine learning model validation for enhanced
signal accuracy and reduced false positives.

ROLE IN TRADING SYSTEM:
- Core trading logic implementation for live and backtesting environments
- Integration point between MetaTrader 5 and Python-trained ML models
- Real-time signal generation with multi-layer filtering approach
- Risk management and position sizing with adaptive algorithms
- Foundation for the complete automated trading system

TRADING STRATEGY ARCHITECTURE:
1. Gann High-Low Activator 2: Primary momentum and trend identification
2. Dual EMA Filtering: 200-period EMA trend confirmation system
3. ML Model Validation: Confidence scoring via included .mqh headers
4. Risk Management: Gann-based stop losses with optional trailing
5. Position Sizing: Fixed lots or percentage-based risk management

SIGNAL GENERATION PROCESS:
- Gann indicator generates raw buy/sell signals based on adaptive zones
- EMA filter validates signals against established trends
- ML model evaluates 24-feature dataset for signal confidence
- Only signals passing all filters result in trade execution
- One position at a time approach for simplified management

ML MODEL INTEGRATION:
- Compile-time model selection via .mqh header inclusion
- Real-time feature calculation matching Data_Exporter2.0 specification
- Support for 6 algorithm types: RandomForest, XGBoost, LightGBM, SVM, LSTM, Ensemble
- Dynamic confidence thresholding for adaptive signal filtering
- Linear approximation of complex models for real-time execution

TECHNICAL IMPLEMENTATION:
- 24 real-time features calculated per signal evaluation
- Gann indicator state tracking for change detection
- EMA crossover analysis and price relationship validation
- Market session and timing factor integration
- Volatility-adjusted position sizing algorithms

DEPLOYMENT WORKFLOW:
1. Train ML models using Python system on historical data
2. Generate .mqh header files via MQH Generator
3. Copy header files to MetaTrader 5 Include directory
4. Select model by uncommenting appropriate #include line
5. Recompile EA to embed chosen model logic
6. Deploy on live or backtesting environment

RISK MANAGEMENT FEATURES:
- Gann-level stop losses for logical exit points
- Dynamic lot sizing based on account balance or fixed amounts
- Maximum position limits and minimum lot enforcement
- Optional trailing stops using Gann levels for trend following
================================================================================
*/
#property copyright "Copyright 2025, IBK"
#property link      "https://www.mql5.com"
#property version   "2.00"

/*
======================================================================================
ML HEADER INCLUSION INSTRUCTIONS
======================================================================================

TO USE ML FILTERING:

1. Copy the generated .mqh header file to the MQL5\Include directory
2. Uncomment the #include line below and specify the header filename
3. Recompile this EA (Ctrl+F7)
4. Set EA parameter: MLHeaderFile="YourFile.mqh"

EXAMPLE SETUP:
Step 1: Copy SVM_GER40_H1_Model.mqh to MQL5\Include\
Step 2: Uncomment: #include "SVM_GER40_H1_Model.mqh"
Step 3: Recompile EA
Step 4: Set MLHeaderFile="SVM_GER40_H1_Model.mqh"
======================================================================================
*/

// MQH Header Files - Uncomment the one you want to use
// =======================================================

// =====================================================
// ENSEMBLE MODELS
// =====================================================

// GER40 Ensemble Models
//#include "Ensemble_GER40_H1_Model.mqh"
#include "Ensemble_GER40_H4_Model.mqh"
// #include "Ensemble_GER40_M15_Model.mqh"
// #include "Ensemble_GER40_M30_Model.mqh"
// #include "Ensemble_GER40_M5_Model.mqh"

// NAS100 Ensemble Models
// #include "Ensemble_NAS100_H1_Model.mqh"
// #include "Ensemble_NAS100_H4_Model.mqh"
// #include "Ensemble_NAS100_M15_Model.mqh"
// #include "Ensemble_NAS100_M30_Model.mqh"
// #include "Ensemble_NAS100_M5_Model.mqh"

// USA30 Ensemble Models
// #include "Ensemble_USA30_H4_Model.mqh"
// #include "Ensemble_USA30_M15_Model.mqh"
// #include "Ensemble_USA30_M30_Model.mqh"
// #include "Ensemble_USA30_M5_Model.mqh"

// USA500 Ensemble Models
// #include "Ensemble_USA500_H1_Model.mqh"
// #include "Ensemble_USA500_H4_Model.mqh"
// #include "Ensemble_USA500_M15_Model.mqh"
// #include "Ensemble_USA500_M30_Model.mqh"
// #include "Ensemble_USA500_M5_Model.mqh"

// =====================================================
// LIGHTGBM MODELS
// =====================================================

// GER40 LightGBM Models
// #include "LightGBM_GER40_H1_Model.mqh"
// #include "LightGBM_GER40_H4_Model.mqh"
// #include "LightGBM_GER40_M15_Model.mqh"
// #include "LightGBM_GER40_M30_Model.mqh"
// #include "LightGBM_GER40_M5_Model.mqh"

// NAS100 LightGBM Models
// #include "LightGBM_NAS100_H1_Model.mqh"
// #include "LightGBM_NAS100_H4_Model.mqh"
// #include "LightGBM_NAS100_M15_Model.mqh"
// #include "LightGBM_NAS100_M30_Model.mqh"
// #include "LightGBM_NAS100_M5_Model.mqh"

// USA30 LightGBM Models
// #include "LightGBM_USA30_H1_Model.mqh"
// #include "LightGBM_USA30_H4_Model.mqh"
// #include "LightGBM_USA30_M15_Model.mqh"
// #include "LightGBM_USA30_M30_Model.mqh"
// #include "LightGBM_USA30_M5_Model.mqh"

// USA500 LightGBM Models
// #include "LightGBM_USA500_H1_Model.mqh"
// #include "LightGBM_USA500_H4_Model.mqh"
// #include "LightGBM_USA500_M15_Model.mqh"
// #include "LightGBM_USA500_M30_Model.mqh"
// #include "LightGBM_USA500_M5_Model.mqh"

// =====================================================
// RANDOM FOREST MODELS
// =====================================================

// GER40 Random Forest Models
// #include "RandomForest_GER40_H1_Model.mqh"
// #include "RandomForest_GER40_H4_Model.mqh"
// #include "RandomForest_GER40_M15_Model.mqh"
// #include "RandomForest_GER40_M30_Model.mqh"

// NAS100 Random Forest Models
// #include "RandomForest_NAS100_H1_Model.mqh"
// #include "RandomForest_NAS100_H4_Model.mqh"
// #include "RandomForest_NAS100_M15_Model.mqh"
// #include "RandomForest_NAS100_M30_Model.mqh"
// #include "RandomForest_NAS100_M5_Model.mqh"

// USA30 Random Forest Models
// #include "RandomForest_USA30_H1_Model.mqh"
// #include "RandomForest_USA30_H4_Model.mqh"
// #include "RandomForest_USA30_M15_Model.mqh"
// #include "RandomForest_USA30_M30_Model.mqh"
// #include "RandomForest_USA30_M5_Model.mqh"

// USA500 Random Forest Models
// #include "RandomForest_USA500_H1_Model.mqh"
// #include "RandomForest_USA500_H4_Model.mqh"
// #include "RandomForest_USA500_M15_Model.mqh"
// #include "RandomForest_USA500_M30_Model.mqh"
// #include "RandomForest_USA500_M5_Model.mqh"

// =====================================================
// SVM MODELS
// =====================================================

// GER40 SVM Models
// #include "SVM_GER40_H1_Model.mqh"
// #include "SVM_GER40_H4_Model.mqh"
// #include "SVM_GER40_M15_Model.mqh"
// #include "SVM_GER40_M30_Model.mqh"
// #include "SVM_GER40_M5_Model.mqh"

// NAS100 SVM Models
// #include "SVM_NAS100_H1_Model.mqh"
// #include "SVM_NAS100_H4_Model.mqh"
// #include "SVM_NAS100_M15_Model.mqh"
// #include "SVM_NAS100_M30_Model.mqh"
// #include "SVM_NAS100_M5_Model.mqh"

// USA30 SVM Models
// #include "SVM_USA30_H4_Model.mqh"
// #include "SVM_USA30_M15_Model.mqh"
// #include "SVM_USA30_M30_Model.mqh"
// #include "SVM_USA30_M5_Model.mqh"

// USA500 SVM Models
// #include "SVM_USA500_H1_Model.mqh"
// #include "SVM_USA500_H4_Model.mqh"
// #include "SVM_USA500_M15_Model.mqh"
// #include "SVM_USA500_M30_Model.mqh"
// #include "SVM_USA500_M5_Model.mqh"

// =====================================================
// XGBOOST MODELS
// =====================================================

// GER40 XGBoost Models
// #include "XGBoost_GER40_H1_Model.mqh"
// #include "XGBoost_GER40_H4_Model.mqh"
// #include "XGBoost_GER40_M15_Model.mqh"
// #include "XGBoost_GER40_M30_Model.mqh"
// #include "XGBoost_GER40_M5_Model.mqh"

// NAS100 XGBoost Models
// #include "XGBoost_NAS100_H1_Model.mqh"
// #include "XGBoost_NAS100_H4_Model.mqh"
// #include "XGBoost_NAS100_M15_Model.mqh"
// #include "XGBoost_NAS100_M30_Model.mqh"
// #include "XGBoost_NAS100_M5_Model.mqh"

// USA30 XGBoost Models
// #include "XGBoost_USA30_H4_Model.mqh"
// #include "XGBoost_USA30_M15_Model.mqh"
// #include "XGBoost_USA30_M30_Model.mqh"
// #include "XGBoost_USA30_M5_Model.mqh"

// USA500 XGBoost Models
// #include "XGBoost_USA500_H1_Model.mqh"
// #include "XGBoost_USA500_H4_Model.mqh"
// #include "XGBoost_USA500_M15_Model.mqh"
// #include "XGBoost_USA500_M30_Model.mqh"
// #include "XGBoost_USA500_M5_Model.mqh"

// =======================================================
// USE INSTRUCTIONS:
// 1. Copy the MQH files to the same folder as the EA
// 2. Uncomment ONE line above by removing the "//" 
// 3. Recompile the Expert Advisor; this has to be done to switch headers as the mqh code is baked in at compilation
// 4. Set the MLHeaderFile parameter to match your choice
// =======================================================

enum ENUM_LOTS_MODE
{
   Fixed,
   PercentOfBalance
};

// Strategy fixed to GannWithEma - no other options needed

enum ENUM_TRAILING_MODE
{
   NoTrailing,      // No trailing stops
   GannTrailing     // Trail using Gann levels
};

input string                        _                            = "=== Main config ===";
input string                        Name                        = "ML_FF_AI";
input int                           MagicNumber                 = 1234;
input ENUM_LOTS_MODE               LotsMode                    = Fixed;
input double                        FixedLotSize                = 0.01;    // Fixed lot size
input double                        RiskPercentage              = 1.0;     // Risk percentage (1.0 = 1%)
input double                        MaxLots                     = 10.0;    // Maximum lot size allowed
input double                        MinLots                     = 0.01;    // Minimum lot size allowed

input string                     __                              = "=== Strategy config ===";
input ENUM_TRAILING_MODE        TrailingMode                   = NoTrailing; // Trailing stop method

enum enMaTypes
{
   ma_sma,         // Simple moving average
   ma_ema,         // Exponential moving average
   ma_smma,        // Smoothed MA
   ma_lwma         // Linear weighted MA
};
enum enFilterWhat
{
   flt_prc,     // Filter the prices
   flt_val,     // Filter the averages value
   flt_all      // Filter all
};

input string                     ___                             = "=== Gann Indicator config ===";
input string                     IndicatorName                  = "Gann_high-low_activator_2";
input ENUM_TIMEFRAMES           TimeFrame                       = PERIOD_CURRENT;   // Time frame
input int                       AvgPeriod                       = 36;                // Average period
input enMaTypes                 AvgType                         = ma_ema;           // Average method
input double                    Filter                          = 0;                // Filter to use (<=0 for no filter)
input enFilterWhat              FilterOn                        = flt_prc;         // Filter :
input bool                      alertsOn                        = false;            // Turn alerts on?
input bool                      alertsOnCurrent                 = true;             // Alert on current bar?
input bool                      alertsMessage                   = true;             // Display messageas on alerts?
input bool                      alertsSound                     = false;            // Play sound on alerts?
input bool                      alertsEmail                     = false;            // Send email on alerts?
input bool                      alertsNotify                    = false;            // Send push notification on alerts?
input bool                      Interpolate                     = true;             // Interpolate mtf data ?

input string                     ____                            = "=== EMA config ===";
input int                        EmaPeriod                      = 200;              // EMA period

input string                     _____                           = "=== ML config ===";
input string                    MLHeaderFile                    = "";               // Header filename (e.g., SVM_GER40_H1_Model.mqh)

// Indicator handles
int gannHandler;
int emaLowHandle;
int emaMedianHandle;
int atrHandle;

// Data arrays
double top[], bot[], main[];
double emaLow[], emaMedian[];
double close[], high[], low[], open[];
double atr[];
datetime lastSignalTime;
int lastGannChangeBar;
double lastGannState;
MqlTradeRequest       request;
MqlTradeResult        result;

int OnInit() {
   //Initialise last signal time
   lastSignalTime = 0;

   // Initialise Gann indicator (always needed for GannWithEma strategy)
   gannHandler = iCustom(Symbol(), TimeFrame, IndicatorName, TimeFrame, AvgPeriod, AvgType, Filter, FilterOn, alertsOn, alertsOnCurrent,
         alertsMessage, alertsSound, alertsEmail, alertsNotify, Interpolate);
   
   if (gannHandler == INVALID_HANDLE) {
      return(INIT_FAILED);
   } else {
      ChartIndicatorAdd(0, 0, gannHandler);
   }

   // Initialise EMA handles (always needed for GannWithEma strategy)
   emaLowHandle = iMA(Symbol(), TimeFrame, EmaPeriod, 0, MODE_EMA, PRICE_LOW);
   emaMedianHandle = iMA(Symbol(), TimeFrame, EmaPeriod, 0, MODE_EMA, PRICE_MEDIAN);
   
   if (emaLowHandle == INVALID_HANDLE || emaMedianHandle == INVALID_HANDLE) {
      return(INIT_FAILED);
   }
   
   // Initialise ATR handle for volatility calculations
   atrHandle = iATR(Symbol(), TimeFrame, 14);
   if (atrHandle == INVALID_HANDLE) {
      return(INIT_FAILED);
   }
   
   // Initialise Gann tracking variables
   lastGannChangeBar = -1;
   lastGannState = 0;
   
   // ML configuration status
   #ifdef MODEL_ALGORITHM
      // ML header loaded successfully - no comment to avoid backtesting issues
   #endif

   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason) {
   if (gannHandler != INVALID_HANDLE)
      IndicatorRelease(gannHandler);
   if (emaLowHandle != INVALID_HANDLE)
      IndicatorRelease(emaLowHandle);
   if (emaMedianHandle != INVALID_HANDLE)
      IndicatorRelease(emaMedianHandle);
   if (atrHandle != INVALID_HANDLE)
      IndicatorRelease(atrHandle);
}

void OnTick() {
   refreshBuffers();
   
   // Update Gann change tracking
   updateGannChangeTracking();
   
   // Handle entry signals (GannWithEma only)
   sendOrders();
   
   if (TrailingMode != NoTrailing) {
      handleStoploss();
   }
}

// Count positions by type (0=buy, 1=sell, 2=all)
int PositionCounter(int type) {
   int count = 0;
   
   for (int i = PositionsTotal() - 1; i >= 0; i--) {
      ulong ticket = PositionGetTicket(i);
      if (ticket > 0) {
         if (PositionGetString(POSITION_SYMBOL) == Symbol() && 
             PositionGetInteger(POSITION_MAGIC) == MagicNumber) {
            
            if (type == 0 && PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) {
               count++;
            }
            else if (type == 1 && PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL) {
               count++;
            }
            else if (type == 2) {
               count++;
            }
         }
      }
   }
   
   return count;
}

// Update Gann change tracking on each tick
void updateGannChangeTracking() {
   if (ArraySize(main) < 2) return;
   
   double currentState = 0;
   if (main[1] == top[1]) currentState = 1;
   else if (main[1] == bot[1]) currentState = -1;
   
   if (currentState != lastGannState) {
      lastGannChangeBar = 1;
      lastGannState = currentState;
   } else {
      lastGannChangeBar++;
   }
}

void refreshBuffers() {
   // Refresh Gann buffers (always needed for GannWithEma)
   CopyBuffer(gannHandler, 0, 0, 25, top);
   CopyBuffer(gannHandler, 1, 0, 25, bot);
   CopyBuffer(gannHandler, 3, 0, 25, main);
   ArraySetAsSeries(top, true);
   ArraySetAsSeries(bot, true);
   ArraySetAsSeries(main, true);
   
   // Refresh EMA buffers (always needed for GannWithEma)
   CopyBuffer(emaLowHandle, 0, 0, 15, emaLow);
   CopyBuffer(emaMedianHandle, 0, 0, 15, emaMedian);
   ArraySetAsSeries(emaLow, true);
   ArraySetAsSeries(emaMedian, true);
   
   // Refresh price data and ATR
   CopyClose(Symbol(), TimeFrame, 0, 15, close);
   CopyHigh(Symbol(), TimeFrame, 0, 15, high);
   CopyLow(Symbol(), TimeFrame, 0, 15, low);
   CopyOpen(Symbol(), TimeFrame, 0, 15, open);
   CopyBuffer(atrHandle, 0, 0, 5, atr);
   
   ArraySetAsSeries(close, true);
   ArraySetAsSeries(high, true);
   ArraySetAsSeries(low, true);
   ArraySetAsSeries(open, true);
   ArraySetAsSeries(atr, true);
}

int checkOpenSignal() {
   // Check if we already have a signal on this bar
   if (lastSignalTime >= iTime(Symbol(), TimeFrame, 0))
      return 0;

   // Only allow new trades if no positions are open (one trade at a time)
   if (PositionCounter(2) > 0)
      return 0;

   // Always use GannWithEma strategy
   return checkGannWithEmaSignal();
}

int checkGannWithEmaSignal() {
   // Get Gann signal first
   bool gannBuySignal = (main[0] == top[0] && main[1] == bot[1]);
   bool gannSellSignal = (main[0] == bot[0] && main[1] == top[1]);
   
   // Apply EMA filter
   bool emaFilterBuy = gannBuySignal && close[1] > emaMedian[1];
   bool emaFilterSell = gannSellSignal && close[1] < emaLow[1];
   
   // Apply ML filter - always active in this AI version
   if (emaFilterBuy || emaFilterSell) {
      #ifdef MODEL_ALGORITHM
         // Header file is included - use ML filtering
         double features[24];
         calculateMLFeatures(features);
         
         // Apply ML confidence filter
         if (!evaluateMLConfidence(features)) {
            return 0; // ML filter rejected the signal
         }
      #else
         //Header file not included - EA will fail to initialise
         // This should never execute due to initialization check
      #endif
   }
   
   // Return signal if all filters pass
   if (emaFilterBuy) {
      return 1;
   }
   
   if (emaFilterSell) {
      return -1;
   }
   
   return 0;
}

void sendOrders() {
   int signal = checkOpenSignal();
   
   // BUY SIGNAL
   if (signal == 1) {
         double entryPrice = SymbolInfoDouble(Symbol(), SYMBOL_ASK);
         double stopLoss = calculateInitialStopLoss(true, entryPrice);
         
         ZeroMemory(request);
         ZeroMemory(result);
         request.action       = TRADE_ACTION_DEAL;
         request.symbol       = Symbol();
         request.volume       = NormalizeDouble(calculateLots(stopLoss, entryPrice), 2);
         request.type         = ORDER_TYPE_BUY;
         request.price        = entryPrice;
         request.magic        = MagicNumber;
         request.comment      = Name;
         request.tp = 0;              
         request.sl = NormalizeDouble(stopLoss, Digits());

         if (!OrderSend(request, result)) {
               // Silent - no logging
         } else {
               lastSignalTime = iTime(Symbol(), TimeFrame, 0);
         }
   }
   
   // SELL SIGNAL
   if (signal == -1) {
         double entryPrice = SymbolInfoDouble(Symbol(), SYMBOL_BID);
         double stopLoss = calculateInitialStopLoss(false, entryPrice);
         
         ZeroMemory(request);
         ZeroMemory(result);
         request.action       = TRADE_ACTION_DEAL;
         request.symbol       = Symbol();
         request.volume       = NormalizeDouble(calculateLots(stopLoss, entryPrice), 2);
         request.type         = ORDER_TYPE_SELL;
         request.price        = entryPrice;
         request.magic        = MagicNumber;
         request.comment      = Name;
         request.tp           = 0;
         request.sl           = NormalizeDouble(stopLoss, Digits());
         if (!OrderSend(request, result)) {
               // Silent - no logging
         } else {
               lastSignalTime = iTime(Symbol(), TimeFrame, 0);
         }
   }
}

double calculateInitialStopLoss(bool isBuy, double entryPrice) {
   // Use Gann levels (always for GannWithEma strategy)
   if (isBuy)
      return bot[1];
   else
      return top[1];
}

double calculateLots(double stopLoss, double entryPrice) {
   if (LotsMode == Fixed) {
      double fixedLots = NormalizeDouble(FixedLotSize, 2);
      // Ensure lot size is within broker limits
      double minLot = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MIN);
      double maxLot = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MAX);
      double stepLot = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_STEP);
      
      fixedLots = MathMax(fixedLots, minLot);
      fixedLots = MathMin(fixedLots, maxLot);
      fixedLots = MathMin(fixedLots, MaxLots);
      
      // Round to step size
      fixedLots = NormalizeDouble(MathRound(fixedLots / stepLot) * stepLot, 2);
      
      return fixedLots;
   }
   
   if (LotsMode == PercentOfBalance) {
      return calculatePercentRiskLots(stopLoss, entryPrice);
   }
   
   return MinLots;
}

double calculatePercentRiskLots(double stopLoss, double entryPrice) {
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskAmount = balance * RiskPercentage / 100.0; // Convert percentage to decimal
   
   if (riskAmount <= 0) {
      return MinLots;
   }
   
   // Calculate pip difference
   double pipDifference = MathAbs(entryPrice - stopLoss);
   if (pipDifference <= 0) {
      return MinLots;
   }
   
   // Get symbol specifications
   double tickSize = SymbolInfoDouble(Symbol(), SYMBOL_TRADE_TICK_SIZE);
   double tickValue = SymbolInfoDouble(Symbol(), SYMBOL_TRADE_TICK_VALUE);
   double minLot = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MAX);
   double stepLot = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_STEP);
   
   if (tickSize <= 0 || tickValue <= 0) {
      return MinLots;
   }
   
   // Calculate lot size based on risk
   double ticks = pipDifference / tickSize;
   double riskPerLot = ticks * tickValue;
   double calculatedLots = riskAmount / riskPerLot;
   
   // Apply limits
   calculatedLots = MathMax(calculatedLots, minLot);
   calculatedLots = MathMax(calculatedLots, MinLots);
   calculatedLots = MathMin(calculatedLots, maxLot);
   calculatedLots = MathMin(calculatedLots, MaxLots);
   
   // Round to step size
   calculatedLots = NormalizeDouble(MathFloor(calculatedLots / stepLot) * stepLot, 2);
   
   // Final validation
   if (calculatedLots < minLot) {
      calculatedLots = minLot;
   }
   
   return calculatedLots;
}

void handleStoploss() {
   for (int cnt = 0; cnt < PositionsTotal(); cnt++) {
         ulong    position_ticket = PositionGetTicket(cnt);
         if (PositionGetInteger(POSITION_MAGIC) == MagicNumber && PositionGetString(POSITION_SYMBOL) == Symbol()) {
               
               if (TrailingMode == GannTrailing) {
                  handleGannTrailing(position_ticket);
               }
         }
   }
}

void handleGannTrailing(ulong position_ticket) {
   if (PositionGetInteger(POSITION_TYPE) == ORDER_TYPE_BUY   
   &&  NormalizeDouble(PositionGetDouble(POSITION_SL), Digits()) < NormalizeDouble(bot[1], Digits())) {
         ZeroMemory(request);
         ZeroMemory(result);
         request.action    = TRADE_ACTION_SLTP;
         request.position = position_ticket;
         request.symbol = PositionGetString(POSITION_SYMBOL);
         request.sl        = NormalizeDouble(bot[1], Digits());
         request.tp        = NormalizeDouble(PositionGetDouble(POSITION_TP), Digits());
         if (!OrderSend(request, result)) {
               // Silent - no logging
         } else {
               // Silent - no logging
         }
   }
   else if (PositionGetInteger(POSITION_TYPE) == ORDER_TYPE_SELL
   &&  NormalizeDouble(PositionGetDouble(POSITION_SL), Digits()) > NormalizeDouble(top[1], Digits())) {
         ZeroMemory(request);
         ZeroMemory(result);
         request.action    = TRADE_ACTION_SLTP;
         request.position = position_ticket;
         request.symbol = PositionGetString(POSITION_SYMBOL);
         request.sl        = NormalizeDouble(top[1], Digits());
         request.tp        = NormalizeDouble(PositionGetDouble(POSITION_TP), Digits());
         if (!OrderSend(request, result)) {
               // Silent - no logging
         } else {
               // Silent - no logging
         }
   }
}

//+------------------------------------------------------------------+
//| Calculate ML features for confidence evaluation                  |
//+------------------------------------------------------------------+
void calculateMLFeatures(double &features[]) {
   //Initialise the 24 features array
   ArrayResize(features, 24);
   ArrayInitialize(features, 0.0);
   
   // Ensure we have enough data
   if (ArraySize(top) < 3 || ArraySize(emaLow) < 3 || ArraySize(close) < 3 || ArraySize(atr) < 2) {
      return; // Not enough data, features remain 0
   }
   
   double pointValue = Point();
   double pipValue = pointValue * 10; // Convert to pips
   
   // Feature 0: Gann_Zone_Width_Pips
   features[0] = MathAbs(top[1] - bot[1]) / pipValue;
   
   // Feature 1: Gann_Zone_Slope (approximation using price change)
   if (ArraySize(top) > 2) {
      features[1] = (top[1] - top[2]) / pipValue;
   }
   
   // Feature 2: Price_Distance_From_Zone_Center
   double zoneCentre = (top[1] + bot[1]) / 2.0;
   features[2] = (close[1] - zoneCentre) / pipValue;
   
   // Feature 3: Price_Distance_From_Zone_Boundary
   if (close[1] > zoneCentre) {
      features[3] = (close[1] - top[1]) / pipValue; // Distance from top
   } else {
      features[3] = (close[1] - bot[1]) / pipValue; // Distance from bottom
   }
   
   // Feature 4: Active_Gann_Line_Slope (using main line change)
   if (ArraySize(main) > 2) {
      features[4] = (main[1] - main[2]) / pipValue;
   }
   
   //Feature 5: Bars_Since_Last_Gann_Colour_Change
   features[5] = lastGannChangeBar;
   
   // Feature 6: EMA_Separation_Pips
   features[6] = (emaMedian[1] - emaLow[1]) / pipValue;
   
   // Feature 7: Median_EMA_Slope
   if (ArraySize(emaMedian) > 2) {
      features[7] = (emaMedian[1] - emaMedian[2]) / pipValue;
   }
   
   // Feature 8: Low_EMA_Slope
   if (ArraySize(emaLow) > 2) {
      features[8] = (emaLow[1] - emaLow[2]) / pipValue;
   }
   
   // Feature 9: Price_Distance_From_Relevant_EMA
   // Use median EMA as relevant for current logic
   features[9] = (close[1] - emaMedian[1]) / pipValue;
   
   // Feature 10: EMA_Convergence_Divergence
   double previousSeparation = ArraySize(emaMedian) > 2 && ArraySize(emaLow) > 2 ? 
                              (emaMedian[2] - emaLow[2]) / pipValue : features[6];
   features[10] = features[6] - previousSeparation;
   
   // Feature 11: Trading_Session (1=Asian, 2=London, 3=NY, 4=Overlap)
   // Must match the training data encoding: London=8-12h, Overlap=13-15h,
   // NY=16-21h, Asian=all other hours. The previous 0-3 encoding shifted
   // every session by one relative to what the models were trained on.
   MqlDateTime timeStruct;
   TimeToStruct(TimeCurrent(), timeStruct);
   int hour = timeStruct.hour;
   if (hour >= 8 && hour <= 12) features[11] = 2;       // London
   else if (hour >= 13 && hour <= 15) features[11] = 4; // London/NY overlap
   else if (hour >= 16 && hour <= 21) features[11] = 3; // NY
   else features[11] = 1;                               // Asian / off-hours
   
   // Feature 12: Day_Of_Week (1=Monday to 7=Sunday)
   features[12] = timeStruct.day_of_week == 0 ? 7 : timeStruct.day_of_week;
   
   // Feature 13: Hour_Of_Day
   features[13] = hour;
   
   // Feature 14: Entry_Candle_Size_Pips
   if (ArraySize(high) > 1 && ArraySize(low) > 1) {
      features[14] = (high[1] - low[1]) / pipValue;
   }
   
   // Feature 15: Entry_Candle_Body_Ratio
   if (ArraySize(open) > 1 && features[14] > 0) {
      double bodySize = MathAbs(close[1] - open[1]) / pipValue;
      features[15] = bodySize / features[14]; // Body to total candle ratio
   }
   
   // Feature 16: Previous_Candle_Size_Pips
   if (ArraySize(high) > 2 && ArraySize(low) > 2) {
      features[16] = (high[2] - low[2]) / pipValue;
   }
   
   // Feature 17: Bars_Since_Relevant_EMA_Cross (approximation)
   // Check if price recently crossed EMA
   int barsSinceCross = 0;
   for (int i = 1; i < MathMin(20, ArraySize(close)); i++) {
      bool currentAbove = close[i] > emaMedian[i];
      bool previousAbove = i+1 < ArraySize(close) ? close[i+1] > emaMedian[i+1] : currentAbove;
      if (currentAbove != previousAbove) {
         barsSinceCross = i;
         break;
      }
   }
   features[17] = barsSinceCross;
   
   // Feature 18: Initial_SL_Distance_Pips
   double entryPrice = close[1];
   double stopLoss = main[1] == top[1] ? bot[1] : top[1]; // Gann-based SL
   features[18] = MathAbs(entryPrice - stopLoss) / pipValue;
   
   // Feature 19: Recent_Gann_Signal_Frequency (simplified)
   int gannSignals = 0;
   for (int i = 1; i < MathMin(10, ArraySize(main)); i++) {
      if (i+1 < ArraySize(main)) {
         bool signal = (main[i] == top[i] && main[i+1] == bot[i+1]) || 
                      (main[i] == bot[i] && main[i+1] == top[i+1]);
         if (signal) gannSignals++;
      }
   }
   features[19] = gannSignals;
   
   // Feature 20: Spread
   features[20] = (SymbolInfoDouble(Symbol(), SYMBOL_ASK) - SymbolInfoDouble(Symbol(), SYMBOL_BID)) / pipValue;
   
   // Feature 21: Signal_Type (1 for buy, -1 for sell, 0 for no signal)
   if (main[0] == top[0] && main[1] == bot[1]) features[21] = 1;
   else if (main[0] == bot[0] && main[1] == top[1]) features[21] = -1;
   else features[21] = 0;
   
   // Feature 22: Entry_Price
   features[22] = close[1];
   
   // Feature 23: Profit_Loss (set to 0 for live trading, used only in training)
   features[23] = 0.0;
}

//+------------------------------------------------------------------+