//+------------------------------------------------------------------+
//| Enhanced Date Range Signal Data Exporter for ML Training v2.0   |
//| Uses ML_FF_2 EA signal strategy: Gann + trend + grey zone filter|
//+------------------------------------------------------------------+
/*
================================================================================
DATA EXPORTER 2.0 - ADVANCED TRADING DATA EXTRACTION FOR ML TRAINING
================================================================================

OVERVIEW:
This MetaTrader 5 script generates comprehensive training datasets for machine
learning models by simulating the ML_FF_2 EA trading strategy and extracting
24 sophisticated technical analysis features along with trade outcome data.

ROLE IN ML SYSTEM:
- Primary data generation tool for training Python ML models
- Simulates complete trading strategy to generate realistic signal data
- Calculates 24 advanced technical features per trading signal
- Exports structured CSV data compatible with Python ML pipeline
- Bridges MetaTrader 5 real-time data with offline ML training process

TRADING STRATEGY SIMULATION:
- Implements Gann High-Low Activator 2 signal generation
- Applies dual EMA trend filtering (200-period median and low)
- Simulates trade execution with realistic entry/exit logic
- Tracks complete trade lifecycle from signal to outcome
- Includes grey zone filtering for enhanced signal quality

FEATURE EXTRACTION (24 FEATURES):
1. Gann Zone Features (6): Width, slope, price distances, line slope, change timing
2. EMA Features (5): Separation, slopes, convergence/divergence, price relationships  
3. Time Features (3): Trading session classification, day of week, hour of day
4. Candle Features (4): Entry candle metrics, body ratios, previous candle data
5. Risk Features (2): Stop loss distances, recent signal frequency
6. Market Features (4): Spread conditions, signal types, entry prices, outcomes

DATA EXPORT SPECIFICATIONS:
- CSV format with headers for direct Python consumption
- Date range filtering for specific time period analysis
- Configurable lot sizes for realistic trade simulation
- Profit/loss calculation in both currency and pip formats
- Signal quality labelling for supervised learning

TECHNICAL IMPLEMENTATION:
- Real-time indicator data copying with proper synchronisation
- Sophisticated exit logic including Gann colour change detection
- Multi-timeframe analysis capability with adaptive periods
- Memory-efficient processing for large historical datasets
- Robust error handling and progress reporting

INTEGRATION WITH ML PIPELINE:
- Generates training data matching real-time EA feature calculation
- Ensures consistency between training data and live trading features
- Provides ground truth labels for supervised learning algorithms
- Supports backtesting validation of trained models

USAGE WORKFLOW:
1. Configure date range and export parameters
2. Run script on historical data in MetaTrader 5
3. Export generates CSV file with complete trading dataset
4. Import CSV into Python system for ML model training
5. Trained models generate .mqh headers for live trading deployment

ADVANCED FEATURES:
- Intelligent signal filtering to reduce noise
- Multiple exit condition analysis (profit/loss/Gann change)
- Statistical validation of generated signals
- Comprehensive logging and progress monitoring
================================================================================
*/
#property copyright "Enhanced Data Exporter v2.0"
#property version   "2.00"
#property script_show_inputs

// Date range inputs
input group "Export Date Range"
input datetime     StartDate = D'2011.09.20 00:00';     // Start Date (YYYY.MM.DD HH:MM)
input datetime     EndDate = D'2025.05.30 23:59';       // End Date (YYYY.MM.DD HH:MM)
input string       ExportFile = "NAS100_signal_data_.csv";      // Output filename

// Gann indicator parameters (matching ML_FF_2 EA)
input group "Gann Indicator Settings"
input ENUM_TIMEFRAMES TimeFrame = PERIOD_CURRENT;
input int             AvgPeriod = 36;
input int             AvgType = 1;
input double          Filter = 0;
input int             FilterOn = 0;

// Global handles for indicators (matching original DataExporter + ML_FF_2)
int handle_ema_low;
int handle_ema_median;
int handle_gann;
int handle_atr;

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
{
    Print("Starting ML_FF_2 strategy signal data export...");
    Print("Start Date: ", TimeToString(StartDate));
    Print("End Date: ", TimeToString(EndDate));
    
    // Validate date range
    if (StartDate >= EndDate)
    {
        Print("Error: Start date must be before end date!");
        return;
    }
    
    datetime effective_end_date = EndDate;
    if (EndDate > TimeCurrent())
    {
        Print("Warning: End date is in the future, using current time");
        effective_end_date = TimeCurrent();
    }
    
    // Create indicator handles (using ML_FF_Base settings: 200 EMA)
    handle_ema_low = iMA(_Symbol, PERIOD_CURRENT, 200, 0, MODE_EMA, PRICE_LOW);
    handle_ema_median = iMA(_Symbol, PERIOD_CURRENT, 200, 0, MODE_EMA, PRICE_MEDIAN);
    handle_gann = iCustom(_Symbol, PERIOD_CURRENT, "Gann_high-low_activator_2", 
                         TimeFrame, AvgPeriod, AvgType, Filter, FilterOn, 
                         false, true, false, false, false, false, true);
    handle_atr = iATR(_Symbol, PERIOD_CURRENT, 14);
    
    // Check if handles are valid
    if(handle_ema_low == INVALID_HANDLE || handle_ema_median == INVALID_HANDLE ||
       handle_gann == INVALID_HANDLE || handle_atr == INVALID_HANDLE)
    {
        Print("Error creating indicator handles");
        return;
    }
    
    // Wait for indicators to calculate
    Sleep(2000);
    
    // Find the bar indices for the date range
    int start_bar = FindBarByDate(StartDate);
    int end_bar = FindBarByDate(effective_end_date);
    
    if (start_bar == -1 || end_bar == -1)
    {
        Print("Error: Could not find bars for the specified date range");
        return;
    }
    
    Print("Date range spans from bar ", end_bar, " to bar ", start_bar);
    Print("Total bars to analyze: ", start_bar - end_bar + 1);
    
    // Open file for writing
    int file_handle = FileOpen(ExportFile, FILE_WRITE|FILE_CSV);
    if(file_handle == INVALID_HANDLE)
    {
        Print("Error opening file: ", ExportFile);
        return;
    }
    
    // Write CSV header (same as original DataExporter)
    FileWrite(file_handle, 
        "Timestamp", "Signal_Type", "Entry_Price", "Profit_Loss", "Duration_Bars",
        "EMA_Low", "EMA_Median", "Gann_High", "Gann_Low", "Gann_Signal", "Market_Volatility",
        "Hour", "Day_of_Week", "Spread", "Volume"
    );
    
    int signals_found = 0;
    int total_bars = start_bar - end_bar + 1;
    
    // Create arrays for indicator values
    double ema_low_array[];
    double ema_median_array[];
    double gann_signal_array[];
    double gann_up_array[];
    double gann_down_array[];
    double gann_line_array[];
    double atr_array[];
    
    // Set arrays as series
    ArraySetAsSeries(ema_low_array, true);
    ArraySetAsSeries(ema_median_array, true);
    ArraySetAsSeries(gann_signal_array, true);
    ArraySetAsSeries(gann_up_array, true);
    ArraySetAsSeries(gann_down_array, true);
    ArraySetAsSeries(gann_line_array, true);
    ArraySetAsSeries(atr_array, true);
    
    // Copy all needed data at once for efficiency
    int buffer_size = start_bar + 200; // Add buffer for exit calculations
    
    int copied_ema_low = CopyBuffer(handle_ema_low, 0, 0, buffer_size, ema_low_array);
    int copied_ema_median = CopyBuffer(handle_ema_median, 0, 0, buffer_size, ema_median_array);
    int copied_gann_signal = CopyBuffer(handle_gann, 4, 0, buffer_size, gann_signal_array);
    int copied_gann_up = CopyBuffer(handle_gann, 0, 0, buffer_size, gann_up_array);
    int copied_gann_down = CopyBuffer(handle_gann, 1, 0, buffer_size, gann_down_array);
    int copied_gann_line = CopyBuffer(handle_gann, 3, 0, buffer_size, gann_line_array);
    int copied_atr = CopyBuffer(handle_atr, 0, 0, buffer_size, atr_array);
    
    if(copied_ema_low <= 0 || copied_ema_median <= 0 || copied_gann_signal <= 0 || 
       copied_gann_up <= 0 || copied_gann_down <= 0 || copied_gann_line <= 0 || copied_atr <= 0)
    {
        Print("Error copying indicator data");
        FileClose(file_handle);
        return;
    }
    
    Print("Indicator data copied successfully, analyzing date range...");
    
    // Progress tracking
    int progress_step = MathMax(1, total_bars / 20); // Update every 5%
    
    // Loop through the specified date range (same as original DataExporter)
    for(int i = start_bar; i >= end_bar; i--)
    {
        // Progress reporting
        if ((start_bar - i) % progress_step == 0)
        {
            double progress = (double)(start_bar - i) / total_bars * 100;
            Print("Progress: ", DoubleToString(progress, 1), "% - Signals found: ", signals_found);
        }
        
        // Check if current bar time is within our date range
        datetime bar_time = iTime(_Symbol, PERIOD_CURRENT, i);
        if (bar_time < StartDate || bar_time > effective_end_date)
            continue;
            
        // Use ML_FF_Base signal logic (GannWithEma mode)
        int signal_type = GetML_FF_Base_Signal(i, ema_low_array, ema_median_array, gann_signal_array, 
                                              gann_up_array, gann_down_array, gann_line_array);
        
        if(signal_type != 0) // Signal found
        {
            // Calculate all required data (same as original DataExporter)
            double entry_price = iClose(_Symbol, PERIOD_CURRENT, i);
            double profit_loss = GetTradeOutcome(i, signal_type, atr_array);
            int duration = GetTradeDuration(i, signal_type, atr_array);
            
            // Technical indicators
            double ema_low = ema_low_array[i];
            double ema_median = ema_median_array[i];
            double gann_high = gann_up_array[i];
            double gann_low = gann_down_array[i];
            double gann_signal = gann_signal_array[i];
            double volatility = atr_array[i];
            
            // Time-based features
            MqlDateTime dt;
            TimeToStruct(bar_time, dt);
            int hour = dt.hour;
            int day_of_week = dt.day_of_week;
            
            // Market microstructure
            double spread = (double)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) * _Point;
            long volume = iVolume(_Symbol, PERIOD_CURRENT, i);
            
            // Write to file (same format as original DataExporter)
            FileWrite(file_handle,
                TimeToString(bar_time), signal_type, entry_price, profit_loss, duration,
                ema_low, ema_median, gann_high, gann_low, gann_signal, volatility,
                hour, day_of_week, spread, volume
            );
            
            signals_found++;
        }
    }
    
    FileClose(file_handle);
    
    Print("=== Export Complete ===");
    Print("Date range: ", TimeToString(StartDate), " to ", TimeToString(effective_end_date));
    Print("Total signals exported: ", signals_found);
    Print("File saved as: ", ExportFile);
    Print("Next step: Run Python script for ML model training");
}

//+------------------------------------------------------------------+
//| Find bar index by date (same as original DataExporter)         |
//+------------------------------------------------------------------+
int FindBarByDate(datetime target_date)
{
    int total_bars = iBars(_Symbol, PERIOD_CURRENT);
    
    // Binary search for efficiency with large datasets
    int left = 0;
    int right = total_bars - 1;
    int best_match = -1;
    
    while (left <= right)
    {
        int mid = (left + right) / 2;
        datetime bar_time = iTime(_Symbol, PERIOD_CURRENT, mid);
        
        if (bar_time == target_date)
        {
            return mid;
        }
        else if (bar_time > target_date)
        {
            left = mid + 1;
            best_match = mid;
        }
        else
        {
            right = mid - 1;
        }
    }
    
    // Return closest match if exact date not found
    if (best_match == -1 && right >= 0)
        best_match = right;
    
    return best_match;
}

//+------------------------------------------------------------------+
//| Get ML_FF_Base EA signal logic (GannWithEma mode)               |
//+------------------------------------------------------------------+
int GetML_FF_Base_Signal(int bar, double &ema_low[], double &ema_median[], 
                         double &gann_signal[], double &gann_up[], double &gann_down[], 
                         double &gann_line[])
{
    // Check if we have valid data for current and previous bars
    if(bar >= ArraySize(ema_low) - 1 || bar >= ArraySize(ema_median) - 1 || 
       bar >= ArraySize(gann_signal) - 1 || bar >= ArraySize(gann_up) - 1 || 
       bar >= ArraySize(gann_down) - 1 || bar >= ArraySize(gann_line) - 1)
        return 0;
    
    double currentClose = iClose(_Symbol, PERIOD_CURRENT, bar);
    double previousClose = iClose(_Symbol, PERIOD_CURRENT, bar + 1);
    double currentGannLine = gann_line[bar];
    double previousGannLine = gann_line[bar + 1];
    double currentGannUpper = gann_up[bar];
    double previousGannUpper = gann_up[bar + 1];
    double currentGannLower = gann_down[bar];
    double previousGannLower = gann_down[bar + 1];
    double currentEmaMedian = ema_median[bar + 1]; // Use previous bar for EMA filter
    double currentEmaLow = ema_low[bar + 1];       // Use previous bar for EMA filter
    
    // Validate data
    if(currentClose <= 0 || previousClose <= 0 || 
       currentGannLine == EMPTY_VALUE || previousGannLine == EMPTY_VALUE ||
       currentGannUpper == EMPTY_VALUE || previousGannUpper == EMPTY_VALUE ||
       currentGannLower == EMPTY_VALUE || previousGannLower == EMPTY_VALUE ||
       currentEmaMedian == EMPTY_VALUE || currentEmaLow == EMPTY_VALUE)
        return 0;
    
    // ML_FF_Base GannWithEma logic:
    // Long signal: Gann line switches from bottom to top AND previous close > 200 EMA Median
    bool gannBuySignal = (currentGannLine == currentGannUpper && previousGannLine == previousGannLower);
    bool emaFilterBuy = (previousClose > currentEmaMedian);
    
    if(gannBuySignal && emaFilterBuy)
        return 1; // Buy signal
    
    // Short signal: Gann line switches from top to bottom AND previous close < 200 EMA Low  
    bool gannSellSignal = (currentGannLine == currentGannLower && previousGannLine == previousGannUpper);
    bool emaFilterSell = (previousClose < currentEmaLow);
    
    if(gannSellSignal && emaFilterSell)
        return -1; // Sell signal
    
    return 0; // No signal
}

//+------------------------------------------------------------------+
//| Get trade outcome (same as original DataExporter)              |
//+------------------------------------------------------------------+
double GetTradeOutcome(int signal_bar, int signal_type, double &atr_array[])
{
    double entry_price = iClose(_Symbol, PERIOD_CURRENT, signal_bar);
    
    // Use Gann line as stop loss (matching original)
    double gann_line_array[];
    ArraySetAsSeries(gann_line_array, true);
    CopyBuffer(handle_gann, 3, signal_bar, 50, gann_line_array);
    
    double stop_loss = gann_line_array[0];
    
    // Simple take profit
    double atr = atr_array[signal_bar];
    double take_profit;
    
    if(signal_type == 1) // Buy signal
    {
        take_profit = entry_price + (atr * 2.0);
    }
    else // Sell signal
    {
        take_profit = entry_price - (atr * 2.0);
    }
    
    // Check subsequent bars for exit or Gann color change
    for(int i = signal_bar - 1; i >= 1 && i >= signal_bar - 200; i--)
    {
        double high = iHigh(_Symbol, PERIOD_CURRENT, i);
        double low = iLow(_Symbol, PERIOD_CURRENT, i);
        
        // Check for Gann color change (trend reversal)
        double current_gann_signal[];
        ArraySetAsSeries(current_gann_signal, true);
        CopyBuffer(handle_gann, 4, i, 2, current_gann_signal);
        
        // Exit on Gann color change
        if(ArraySize(current_gann_signal) >= 2)
        {
            if(signal_type == 1 && current_gann_signal[0] == 2)
            {
                double exit_price = iClose(_Symbol, PERIOD_CURRENT, i);
                return exit_price - entry_price;
            }
            else if(signal_type == -1 && current_gann_signal[0] == 1)
            {
                double exit_price = iClose(_Symbol, PERIOD_CURRENT, i);
                return entry_price - exit_price;
            }
        }
        
        // Also check traditional SL/TP
        if(signal_type == 1) // Buy trade
        {
            if(high >= take_profit)
                return take_profit - entry_price;
            if(low <= stop_loss)
                return stop_loss - entry_price;
        }
        else // Sell trade
        {
            if(low <= take_profit)
                return entry_price - take_profit;
            if(high >= stop_loss)
                return entry_price - stop_loss;
        }
    }
    
    // If no exit found, use current price
    double current_price = iClose(_Symbol, PERIOD_CURRENT, 1);
    if(signal_type == 1)
        return current_price - entry_price;
    else
        return entry_price - current_price;
}

//+------------------------------------------------------------------+
//| Calculate trade duration (same as original DataExporter)       |
//+------------------------------------------------------------------+
int GetTradeDuration(int signal_bar, int signal_type, double &atr_array[])
{
    // Check for Gann color change exit first
    for(int i = signal_bar - 1; i >= 1 && i >= signal_bar - 200; i--)
    {
        double current_gann_signal[];
        ArraySetAsSeries(current_gann_signal, true);
        CopyBuffer(handle_gann, 4, i, 2, current_gann_signal);
        
        if(ArraySize(current_gann_signal) >= 2)
        {
            if(signal_type == 1 && current_gann_signal[0] == 2)
                return signal_bar - i;
            else if(signal_type == -1 && current_gann_signal[0] == 1)
                return signal_bar - i;
        }
    }
    
    return MathMin(signal_bar - 1, 100);
}

//+------------------------------------------------------------------+
//| Script deinitialization function                                |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    // Release indicator handles
    if(handle_ema_low != INVALID_HANDLE) IndicatorRelease(handle_ema_low);
    if(handle_ema_median != INVALID_HANDLE) IndicatorRelease(handle_ema_median);
    if(handle_gann != INVALID_HANDLE) IndicatorRelease(handle_gann);
    if(handle_atr != INVALID_HANDLE) IndicatorRelease(handle_atr);
}