//+------------------------------------------------------------------+
//|                                                  ML_FF_ONNX.mq5 |
//|                                              Copyright 2026, IBK |
//|     GannWithEma trading system with native ONNX ML filtering     |
//+------------------------------------------------------------------+
/*
================================================================================
ML FINANCIAL FORECASTING - ONNX EXPERT ADVISOR
================================================================================

OVERVIEW:
Same GannWithEma trading strategy as ML_FF_AI.mq5, but the ML confidence
filter runs the ACTUAL trained model via MetaTrader 5's native ONNX runtime
instead of a linear approximation baked into a .mqh header.

WHAT CHANGED VS ML_FF_AI.mq5:
- No #include of generated model headers, no recompile to switch models.
  The model is a .onnx file loaded at EA start; swap the file, restart EA.
- The ONNX graph contains BOTH the StandardScaler and the Random Forest,
  exported as one sklearn Pipeline by ONNX_UPDATE/train_export_onnx.py.
  The EA feeds RAW feature values; scaling happens inside the graph, so the
  training-vs-live preprocessing mismatch that plagued the header system is
  structurally impossible.
- Fail-closed: if inference fails for any reason the signal is NOT traded.
  (The old header system failed open and passed everything.)

MODEL CONTRACT (must match model_meta.json of the loaded model):
  input  "float_input"   float32 [1, 21]
      inputs 0-19 : base Data_Exporter2.0 features = EA features[] slots 0-19
      input  20   : Signal_Type                    = EA features[] slot 21
      (Spread - EA slot 20 - is NOT a model input: the historical training
       data never recorded it.)
  output "probabilities" float32 [1, 2]
      probabilities[0][1] = P(signal is profitable)

FEATURE CALCULATION:
calculateMLFeatures() is byte-for-byte the same as the fixed ML_FF_AI.mq5,
including the Trading_Session encoding recovered from the training data:
  1=Asian/off-hours, 2=London(8-12h), 4=Overlap(13-15h), 3=NY(16-21h).
Any change here must be mirrored in the Python feature pipeline, otherwise
every deployed model is silently invalidated.

DEPLOYMENT:
1. Train + export:  python ONNX_UPDATE/train_export_onnx.py --csv <data.csv>
                        --symbol GER40 --timeframe H1
2. Copy RandomForest_GER40_H1.onnx to <MT5 data folder>\MQL5\Files\ML\
3. Compile this EA, set inputs:
       OnnxModelFile       = "RandomForest_GER40_H1.onnx"
       ConfidenceThreshold = value from model_meta.json
4. Validate in the Strategy Tester before any live use.
================================================================================
*/
#property copyright "Copyright 2026, IBK"
#property link      "https://www.mql5.com"
#property version   "1.00"

enum ENUM_LOTS_MODE
{
   Fixed,
   PercentOfBalance
};

enum ENUM_TRAILING_MODE
{
   NoTrailing,      // No trailing stops
   GannTrailing     // Trail using Gann levels
};

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

input string                        _                            = "=== Main config ===";
input string                        Name                        = "ML_FF_ONNX";
input int                           MagicNumber                 = 1234;
input ENUM_LOTS_MODE               LotsMode                    = Fixed;
input double                        FixedLotSize                = 0.01;    // Fixed lot size
input double                        RiskPercentage              = 1.0;     // Risk percentage (1.0 = 1%)
input double                        MaxLots                     = 10.0;    // Maximum lot size allowed
input double                        MinLots                     = 0.01;    // Minimum lot size allowed

input string                     __                              = "=== Strategy config ===";
input ENUM_TRAILING_MODE        TrailingMode                   = NoTrailing; // Trailing stop method

input string                     ___                             = "=== Gann Indicator config ===";
input string                     IndicatorName                  = "Gann_high-low_activator_2";
input ENUM_TIMEFRAMES           TimeFrame                       = PERIOD_CURRENT;   // Time frame
input int                       AvgPeriod                       = 36;               // Average period
input enMaTypes                 AvgType                         = ma_ema;           // Average method
input double                    Filter                          = 0;                // Filter to use (<=0 for no filter)
input enFilterWhat              FilterOn                        = flt_prc;          // Filter :
input bool                      alertsOn                        = false;            // Turn alerts on?
input bool                      alertsOnCurrent                 = true;             // Alert on current bar?
input bool                      alertsMessage                   = true;             // Display messages on alerts?
input bool                      alertsSound                     = false;            // Play sound on alerts?
input bool                      alertsEmail                     = false;            // Send email on alerts?
input bool                      alertsNotify                    = false;            // Send push notification on alerts?
input bool                      Interpolate                     = true;             // Interpolate mtf data ?

input string                     ____                            = "=== EMA config ===";
input int                        EmaPeriod                      = 200;              // EMA period

input string                     _____                           = "=== ONNX ML config ===";
input string                    OnnxModelFile                   = "RandomForest_GER40_H1.onnx"; // Model file in MQL5\Files\ML\
input double                    ConfidenceThreshold             = 0.50;             // Min P(profitable) to trade (see model_meta.json)
input bool                      LogMLDecisions                  = true;             // Print each ML accept/reject decision

// --- ONNX model contract -------------------------------------------------
#define N_MODEL_INPUTS 21
// Model input i is filled from features[ FEATURE_MAP[i] ].
// Inputs 0-19 = EA slots 0-19 (base features); input 20 = slot 21 (Signal_Type).
static const int FEATURE_MAP[N_MODEL_INPUTS] =
   {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19, 21};

// Indicator handles
int gannHandler;
int emaLowHandle;
int emaMedianHandle;
int atrHandle;

// ONNX handle
long onnxHandle = INVALID_HANDLE;

// Data arrays
double top[], bot[], main[];
double emaLow[], emaMedian[];
double close[], high[], low[], open[];
double atr[];
datetime lastSignalTime;
datetime lastMLLogTime;
int lastGannChangeBar;
double lastGannState;
MqlTradeRequest       request;
MqlTradeResult        result;

int OnInit() {
   //Initialise last signal time
   lastSignalTime = 0;
   lastMLLogTime = 0;

   // Initialise Gann indicator (always needed for GannWithEma strategy)
   gannHandler = iCustom(Symbol(), TimeFrame, IndicatorName, TimeFrame, AvgPeriod, AvgType, Filter, FilterOn, alertsOn, alertsOnCurrent,
         alertsMessage, alertsSound, alertsEmail, alertsNotify, Interpolate);

   if (gannHandler == INVALID_HANDLE) {
      Print("Failed to load Gann indicator: ", IndicatorName);
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

   // ------------------------------------------------------------------
   // Load the ONNX model from MQL5\Files\ML\<OnnxModelFile>
   // ------------------------------------------------------------------
   string modelPath = "ML\\" + OnnxModelFile;
   onnxHandle = OnnxCreate(modelPath, ONNX_DEFAULT);
   if (onnxHandle == INVALID_HANDLE) {
      Print("ERROR: failed to load ONNX model '", modelPath,
            "'. Copy the .onnx file to <MT5 data folder>\\MQL5\\Files\\ML\\ . ",
            "Last error: ", GetLastError());
      return(INIT_FAILED);
   }

   // Declare fixed shapes: one signal in, one probability pair out
   const long inputShape[]  = {1, N_MODEL_INPUTS};
   const long outputShape[] = {1, 2};
   if (!OnnxSetInputShape(onnxHandle, 0, inputShape)) {
      Print("ERROR: OnnxSetInputShape failed: ", GetLastError());
      OnnxRelease(onnxHandle);
      onnxHandle = INVALID_HANDLE;
      return(INIT_FAILED);
   }
   if (!OnnxSetOutputShape(onnxHandle, 0, outputShape)) {
      Print("ERROR: OnnxSetOutputShape failed: ", GetLastError());
      OnnxRelease(onnxHandle);
      onnxHandle = INVALID_HANDLE;
      return(INIT_FAILED);
   }

   Print("ONNX model loaded: ", OnnxModelFile,
         " | inputs: ", N_MODEL_INPUTS,
         " | confidence threshold: ", DoubleToString(ConfidenceThreshold, 2));

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
   if (onnxHandle != INVALID_HANDLE) {
      OnnxRelease(onnxHandle);
      onnxHandle = INVALID_HANDLE;
   }
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

   // Apply ONNX ML confidence filter
   if (emaFilterBuy || emaFilterSell) {
      double confidence = getOnnxConfidence();

      // Fail closed: if inference failed, do NOT trade the signal.
      if (confidence < 0.0) {
         logMLDecision("ONNX inference FAILED - signal skipped (fail-closed)", 0.0);
         return 0;
      }

      if (confidence < ConfidenceThreshold) {
         logMLDecision("ML filter REJECTED signal", confidence);
         return 0;
      }

      logMLDecision("ML filter ACCEPTED signal", confidence);
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

// Log ML decisions at most once per bar to keep the journal readable
void logMLDecision(string message, double confidence) {
   if (!LogMLDecisions) return;
   datetime barTime = iTime(Symbol(), TimeFrame, 0);
   if (lastMLLogTime == barTime) return;
   lastMLLogTime = barTime;
   Print(message, " | confidence=", DoubleToString(confidence, 4),
         " | threshold=", DoubleToString(ConfidenceThreshold, 2));
}

//+------------------------------------------------------------------+
//| Run the ONNX model on the current signal's features               |
//| Returns P(profitable) in [0,1], or -1.0 on any inference failure  |
//+------------------------------------------------------------------+
double getOnnxConfidence() {
   if (onnxHandle == INVALID_HANDLE)
      return -1.0;

   double features[24];
   calculateMLFeatures(features);

   // Map the EA's 24-slot feature array onto the model's 21 inputs.
   // The graph embeds the StandardScaler, so RAW values go in here.
   matrixf X;
   X.Init(1, N_MODEL_INPUTS);
   for (int i = 0; i < N_MODEL_INPUTS; i++) {
      X[0][i] = (float)features[FEATURE_MAP[i]];
   }

   // Output: probabilities float[1][2]; column 1 = P(profitable)
   matrixf probabilities;
   if (!OnnxRun(onnxHandle, ONNX_DEFAULT, X, probabilities)) {
      Print("OnnxRun failed: ", GetLastError());
      return -1.0;
   }

   if (probabilities.Rows() < 1 || probabilities.Cols() < 2)
      return -1.0;

   double p = (double)probabilities[0][1];
   if (!MathIsValidNumber(p) || p < 0.0 || p > 1.0)
      return -1.0;

   return p;
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
//| IDENTICAL to the fixed ML_FF_AI.mq5 - keep them in sync.         |
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

   // Feature 20: Spread (NOT a model input - kept for slot compatibility)
   features[20] = (SymbolInfoDouble(Symbol(), SYMBOL_ASK) - SymbolInfoDouble(Symbol(), SYMBOL_BID)) / pipValue;

   // Feature 21: Signal_Type (1 for buy, -1 for sell, 0 for no signal)
   if (main[0] == top[0] && main[1] == bot[1]) features[21] = 1;
   else if (main[0] == bot[0] && main[1] == top[1]) features[21] = -1;
   else features[21] = 0;

   // Feature 22: Entry_Price (not a model input)
   features[22] = close[1];

   // Feature 23: Profit_Loss (unknown at trade time - never a model input)
   features[23] = 0.0;
}

//+------------------------------------------------------------------+
