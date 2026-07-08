//+------------------------------------------------------------------+
//|                                               ML_FF_AI_CSV.mq5 |
//|                                              Copyright 2025, IBK |
//|          GannWithEma trading system with ML filtering + CSV     |
//+------------------------------------------------------------------+
/*
================================================================================
ML FINANCIAL FORECASTING - EXPERT ADVISOR WITH CSV BACKTESTING RESULTS
================================================================================

OVERVIEW:
This Expert Advisor implements a sophisticated trading system that combines
Gann High-Low Activator signals with dual EMA filtering and machine learning
model validation. It includes comprehensive backtesting result generation
with CSV export functionality for performance analysis.

ROLE IN TRADING SYSTEM:
- Real-time signal generation using Gann High-Low Activator 2 indicator
- EMA-based trend filtering (200-period EMA on low and median prices)
- ML model confidence scoring for signal validation
- Automated CSV generation of backtesting performance metrics
- Risk management with Gann-based stop losses and position sizing

TRADING STRATEGY COMPONENTS:
1. Gann High-Low Activator 2: Primary signal generation based on color changes
2. Dual EMA Filter: 200-period EMA trend validation (median and low prices)
   - Buy: Gann changes from red to blue ABOVE median EMA
   - Sell: Gann changes from blue to red BELOW low EMA
3. ML Confidence Filter: Uses pre-trained models via included .mqh headers
4. Risk Management: Dynamic lot sizing and Gann-level stop losses
5. Performance Tracking: Real-time metrics collection for CSV export

ML INTEGRATION:
- Supports multiple algorithms: RandomForest, XGBoost, LightGBM, SVM, LSTM, Ensemble
- Real-time feature calculation matching Data_Exporter2.0 format (24 features)
- Model switching via .mqh header inclusion (compile-time selection)
- Confidence-based signal filtering to improve win rates

CSV BACKTESTING FEATURES:
- Automatic performance metrics calculation during backtesting
- CSV file generation with comprehensive trading statistics
- Metrics include: Net P&L, Max Drawdown, Win Rate, Sharpe Ratio, Trade Count
- Files saved to Common folder to avoid Agent sandbox limitations
- Filename format: Symbol_Timeframe_Algorithm_BacktestResults.csv

CONFIGURATION:
- Supports fixed lot size or percentage-based position sizing
- Optional Gann trailing stops for trade management
- Configurable ML model selection via .mqh header inclusion
- Comprehensive input parameters for strategy optimisation

DEPLOYMENT:
- Copy .mqh model files to MQL5/Include directory
- Uncomment desired model header and recompile
- Run backtests to generate CSV performance reports
- Use CSV data for model comparison and strategy optimisation
================================================================================
*/
#property copyright "Copyright 2025, IBK"
#property link      "https://www.mql5.com"
#property version   "2.01"

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
// GER40 MODELS
// =====================================================

// GER40 H1 Models
// #include "Ensemble_GER40_H1_Model.mqh"
// #include "LightGBM_GER40_H1_Model.mqh"
// #include "RandomForest_GER40_H1_Model.mqh"
// #include "SVM_GER40_H1_Model.mqh"
// #include "XGBoost_GER40_H1_Model.mqh"

// GER40 H4 Models
//#include "Ensemble_GER40_H4_Model.mqh"
// #include "LightGBM_GER40_H4_Model.mqh"
// #include "RandomForest_GER40_H4_Model.mqh"
// #include "SVM_GER40_H4_Model.mqh"
// #include "XGBoost_GER40_H4_Model.mqh"

// GER40 M5 Models
// #include "Ensemble_GER40_M5_Model.mqh"
// #include "LightGBM_GER40_M5_Model.mqh"
// #include "SVM_GER40_M5_Model.mqh"
// #include "XGBoost_GER40_M5_Model.mqh"

// GER40 M15 Models
// #include "Ensemble_GER40_M15_Model.mqh"
// #include "LightGBM_GER40_M15_Model.mqh"
// #include "RandomForest_GER40_M15_Model.mqh"
// #include "SVM_GER40_M15_Model.mqh"
// #include "XGBoost_GER40_M15_Model.mqh"

// GER40 M30 Models
// #include "Ensemble_GER40_M30_Model.mqh"
// #include "LightGBM_GER40_M30_Model.mqh"
// #include "RandomForest_GER40_M30_Model.mqh"
// #include "SVM_GER40_M30_Model.mqh"
// #include "XGBoost_GER40_M30_Model.mqh"

// =====================================================
// NAS100 MODELS
// =====================================================

// NAS100 H1 Models
// #include "Ensemble_NAS100_H1_Model.mqh"
// #include "LightGBM_NAS100_H1_Model.mqh"
// #include "RandomForest_NAS100_H1_Model.mqh"
// #include "SVM_NAS100_H1_Model.mqh"
// #include "XGBoost_NAS100_H1_Model.mqh"

// NAS100 H4 Models
// #include "Ensemble_NAS100_H4_Model.mqh"
// #include "LightGBM_NAS100_H4_Model.mqh"
// #include "RandomForest_NAS100_H4_Model.mqh"
// #include "SVM_NAS100_H4_Model.mqh"
// #include "XGBoost_NAS100_H4_Model.mqh"

// NAS100 M5 Models
// #include "Ensemble_NAS100_M5_Model.mqh"
// #include "LightGBM_NAS100_M5_Model.mqh"
// #include "RandomForest_NAS100_M5_Model.mqh"
// #include "SVM_NAS100_M5_Model.mqh"
// #include "XGBoost_NAS100_M5_Model.mqh"

// NAS100 M15 Models
// #include "Ensemble_NAS100_M15_Model.mqh"
// #include "LightGBM_NAS100_M15_Model.mqh"
// #include "RandomForest_NAS100_M15_Model.mqh"
// #include "SVM_NAS100_M15_Model.mqh"
// #include "XGBoost_NAS100_M15_Model.mqh"

// NAS100 M30 Models
// #include "Ensemble_NAS100_M30_Model.mqh"
// #include "LightGBM_NAS100_M30_Model.mqh"
// #include "RandomForest_NAS100_M30_Model.mqh"
// #include "SVM_NAS100_M30_Model.mqh"
// #include "XGBoost_NAS100_M30_Model.mqh"

// =====================================================
// USA30 MODELS
// =====================================================

// USA30 H4 Models
// #include "Ensemble_USA30_H4_Model.mqh"
// #include "LightGBM_USA30_H4_Model.mqh"
// #include "RandomForest_USA30_H4_Model.mqh"
// #include "SVM_USA30_H4_Model.mqh"
// #include "XGBoost_USA30_H4_Model.mqh"

// USA30 M5 Models
// #include "Ensemble_USA30_M5_Model.mqh"
// #include "LightGBM_USA30_M5_Model.mqh"
// #include "RandomForest_USA30_M5_Model.mqh"
// #include "SVM_USA30_M5_Model.mqh"
// #include "XGBoost_USA30_M5_Model.mqh"

// USA30 M15 Models
// #include "Ensemble_USA30_M15_Model.mqh"
// #include "LightGBM_USA30_M15_Model.mqh"
// #include "RandomForest_USA30_M15_Model.mqh"
// #include "SVM_USA30_M15_Model.mqh"
// #include "XGBoost_USA30_M15_Model.mqh"

// USA30 M30 Models
// #include "Ensemble_USA30_M30_Model.mqh"
// #include "LightGBM_USA30_M30_Model.mqh"
// #include "RandomForest_USA30_M30_Model.mqh"
// #include "SVM_USA30_M30_Model.mqh"
// #include "XGBoost_USA30_M30_Model.mqh"

// =====================================================
// USA500 MODELS
// =====================================================

// USA500 H1 Models
// #include "Ensemble_USA500_H1_Model.mqh"
// #include "LightGBM_USA500_H1_Model.mqh"
// #include "RandomForest_USA500_H1_Model.mqh"
// #include "SVM_USA500_H1_Model.mqh"
// #include "XGBoost_USA500_H1_Model.mqh"

// USA500 H4 Models
// #include "Ensemble_USA500_H4_Model.mqh"
// #include "LightGBM_USA500_H4_Model.mqh"
// #include "RandomForest_USA500_H4_Model.mqh"
// #include "SVM_USA500_H4_Model.mqh"
// #include "XGBoost_USA500_H4_Model.mqh"

// USA500 M5 Models
// #include "Ensemble_USA500_M5_Model.mqh"
// #include "LightGBM_USA500_M5_Model.mqh"
// #include "RandomForest_USA500_M5_Model.mqh"
// #include "SVM_USA500_M5_Model.mqh"
// #include "XGBoost_USA500_M5_Model.mqh"

// USA500 M15 Models
// #include "Ensemble_USA500_M15_Model.mqh"
// #include "LightGBM_USA500_M15_Model.mqh"
// #include "RandomForest_USA500_M15_Model.mqh"
// #include "SVM_USA500_M15_Model.mqh"
// #include "XGBoost_USA500_M15_Model.mqh"

// USA500 M30 Models
// #include "Ensemble_USA500_M30_Model.mqh"
// #include "LightGBM_USA500_M30_Model.mqh"
// #include "RandomForest_USA500_M30_Model.mqh"
// #include "SVM_USA500_M30_Model.mqh"
// #include "XGBoost_USA500_M30_Model.mqh"

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
input string                        Name                        = "ML_FF_AI_CSV";
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

// Data arrays
double top[], bot[], main[];
double emaLow[], emaMedian[];
double close[], high[], low[], open[];
datetime lastSignalTime;
int lastGannChangeBar;
double lastGannState;
MqlTradeRequest       request;
MqlTradeResult        result;

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

int OnInit() {
   //Initialise performance tracking variables
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
   // Generate CSV file with backtest results before cleanup
   GenerateBacktestCSV();
   
   if (gannHandler != INVALID_HANDLE)
      IndicatorRelease(gannHandler);
   if (emaLowHandle != INVALID_HANDLE)
      IndicatorRelease(emaLowHandle);
   if (emaMedianHandle != INVALID_HANDLE)
      IndicatorRelease(emaMedianHandle);
}

void OnTick() {
   refreshBuffers();
   
   // Update Gann change tracking
   updateGannChangeTracking();
   
   // Update performance tracking
   UpdatePerformanceMetrics();
   
   // Handle entry signals (GannWithEma only)
   sendOrders();
   
   if (TrailingMode != NoTrailing) {
      handleStoploss();
   }
}

// Update performance tracking on each tick
void UpdatePerformanceMetrics() {
   double currentBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   double currentEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   
   // Update peak balance and drawdown
   if (currentEquity > peakBalance) {
      peakBalance = currentEquity;
   }
   
   currentDrawdown = (peakBalance - currentEquity) / peakBalance * 100.0;
   if (currentDrawdown > maxDrawdown) {
      maxDrawdown = currentDrawdown;
   }
   
   // Count closed trades and their P&L
   static int lastHistoryTotal = 0;
   int currentHistoryTotal = HistoryDealsTotal();
   
   if (currentHistoryTotal > lastHistoryTotal) {
      // New deal(s) closed - update statistics
      for (int i = lastHistoryTotal; i < currentHistoryTotal; i++) {
         ulong ticket = HistoryDealGetTicket(i);
         if (ticket > 0) {
            string symbol = HistoryDealGetString(ticket, DEAL_SYMBOL);
            long magic = HistoryDealGetInteger(ticket, DEAL_MAGIC);
            long type = HistoryDealGetInteger(ticket, DEAL_TYPE);
            
            if (symbol == Symbol() && magic == MagicNumber && 
               (type == DEAL_TYPE_BUY || type == DEAL_TYPE_SELL)) {
               
               double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT);
               if (profit != 0) { // Exclude entry trades (profit = 0)
                  totalTrades++;
                  if (profit > 0) {
                     totalProfit += profit;
                     winningTrades++;
                  } else {
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

// Generate CSV file with backtest results
void GenerateBacktestCSV() {
   // Extract symbol, timeframe, and ML algorithm from the included header
   string symbol = Symbol();
   string timeframe = TimeFrameToString(Period());  // Use Period() to get actual chart timeframe
   string algorithm = GetActiveMLAlgorithm();
   
   // Create filename
   string filename = StringFormat("%s_%s_%s_BacktestResults.csv", symbol, timeframe, algorithm);
   
   // Calculate final metrics
   double netProfitLoss = totalProfit - totalLoss;
   double winRate = totalTrades > 0 ? (double)winningTrades / totalTrades * 100.0 : 0.0;
   double sharpeRatio = CalculateSharpeRatio(netProfitLoss);
   
   // Save to Common folder to avoid Agent sandbox - accessible from main terminal
   string subfolder = "BacktestResults\\";
   string fullPath = subfolder + filename;
   
   // Open file for writing in COMMON folder (FILE_COMMON flag)
   int fileHandle = FileOpen(fullPath, FILE_WRITE | FILE_CSV | FILE_REWRITE | FILE_COMMON);
   if (fileHandle != INVALID_HANDLE) {
      // Write CSV header
      FileWrite(fileHandle, "Metric", "Value");
      
      // Write performance metrics
      FileWrite(fileHandle, "Symbol", symbol);
      FileWrite(fileHandle, "Timeframe", timeframe);
      FileWrite(fileHandle, "ML Algorithm", algorithm);
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
      Print("Backtest results saved to Common Files: ", subfolder, filename);
   } else {
      Print("Failed to create CSV file: ", fullPath);
   }
}

// Convert timeframe enum to string
string TimeFrameToString(ENUM_TIMEFRAMES timeframe) {
   // Handle PERIOD_CURRENT by getting the actual chart timeframe
   if (timeframe == PERIOD_CURRENT) {
      timeframe = Period();
   }
   
   switch(timeframe) {
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

// Extract ML algorithm from the active header file
string GetActiveMLAlgorithm() {
   #ifdef MODEL_ALGORITHM
      string algorithm = MODEL_ALGORITHM;
      // Clean up the algorithm name if it contains underscores
      StringReplace(algorithm, "_", " ");
      return algorithm;
   #else
      return "NoML";
   #endif
}

// Calculate Sharpe ratio (simplified)
double CalculateSharpeRatio(double netProfit) {
   if (totalTrades < 2) return 0.0;
   
   // Simplified Sharpe calculation
   double avgProfit = netProfit / totalTrades;
   double riskFreeRate = 0.02; // Assume 2% risk-free rate
   
   // Calculate standard deviation of returns (simplified)
   double variance = 0.0;
   double avgReturn = avgProfit;
   
   // This is a simplified calculation - in real implementation,
   // you'd track individual trade returns for proper standard deviation
   if (totalTrades > 0) {
      variance = MathPow(totalProfit / totalTrades - totalLoss / totalTrades, 2);
   }
   
   double stdDev = MathSqrt(variance);
   if (stdDev == 0) return 0.0;
   
   return (avgReturn - riskFreeRate) / stdDev;
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
   
   // Refresh price data
   CopyClose(Symbol(), TimeFrame, 0, 15, close);
   CopyHigh(Symbol(), TimeFrame, 0, 15, high);
   CopyLow(Symbol(), TimeFrame, 0, 15, low);
   CopyOpen(Symbol(), TimeFrame, 0, 15, open);

   ArraySetAsSeries(close, true);
   ArraySetAsSeries(high, true);
   ArraySetAsSeries(low, true);
   ArraySetAsSeries(open, true);
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
   // Check for Gann trend change first (using Gann signal buffer)
   // Buffer 4 contains signal: 1 = bullish (blue), 2 = bearish (red)

   // Get Gann signal buffer values
   double GannSignal[];
   ArraySetAsSeries(GannSignal, true);
   CopyBuffer(gannHandler, 4, 0, 3, GannSignal);

   // Check if we have valid data
   if (ArraySize(GannSignal) < 2) return 0;

   double currentGannSignal = GannSignal[0];
   double previousGannSignal = GannSignal[1];

   // Validate data
   if (currentGannSignal == EMPTY_VALUE || previousGannSignal == EMPTY_VALUE)
      return 0;

   // Check for trend change
   bool gannBuySignal = (previousGannSignal == 2 && currentGannSignal == 1);  // Changed from bearish to bullish
   bool gannSellSignal = (previousGannSignal == 1 && currentGannSignal == 2); // Changed from bullish to bearish

   // Apply EMA filter - check where the trend change occurred
   // Use the Gann line value where the change occurred (main[1])
   double trendChangePrice = main[1];
   bool emaFilterBuy = gannBuySignal && (trendChangePrice > emaMedian[1]); // Above MEDIAN EMA
   bool emaFilterSell = gannSellSignal && (trendChangePrice < emaLow[1]);   // Below LOW EMA

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
   if (ArraySize(top) < 3 || ArraySize(emaLow) < 3 || ArraySize(close) < 3) {
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
   // NY=16-21h, Asian=all other hours.
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