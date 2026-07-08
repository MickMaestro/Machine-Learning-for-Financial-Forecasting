'''
================================================================================
FEATURE ENGINEERING - DATA_EXPORTER2.0 COMPATIBLE
================================================================================

OVERVIEW:
This module processes and validates the 24 pre-calculated trading features from
the Data_Exporter2.0 format. Instead of calculating features from raw price data,
it works with sophisticated technical analysis features already computed by the
MetaTrader 5 Data_Exporter2.0 system.

ROLE IN CODEBASE:
- Validates presence of all 24 required Data_Exporter2.0 features
- Creates additional derived features from base feature set
- Generates interaction features for improved model performance
- Implements feature completeness checking and quality assurance
- Provides standardised feature engineering pipeline for all trainers

DATA_EXPORTER2.0 FEATURES (24 BASE FEATURES):
1. Gann Zone Features (6): Zone width, slope, price distances, line slope, change timing
2. EMA Features (5): Separation, slopes, convergence/divergence, price distances
3. Time Features (3): Trading session, day of week, hour of day
4. Candle Features (4): Entry candle size/ratio, previous candle size, EMA cross timing
5. Risk Features (2): Stop loss distance, recent signal frequency
6. Market Features (4): Spread, signal type, entry price, profit/loss

DERIVED FEATURES CREATED:
- Trading session binary indicators (Asian, London, NY, Overlap)
- Day-of-week patterns (Monday, Friday, Midweek indicators) 
- Market timing features (market hours, late night indicators)
- Signal direction features (buy/sell signal indicators)
- Risk-based features (tight/wide stop classifications)
- Activity-based features (high/low signal activity periods)

INTERACTION FEATURES:
- Gann-EMA alignment scoring
- Favourable time condition combinations
- Momentum alignment across indicators
- Fresh signal detection logic

INTEGRATION POINTS:
- Used by all ML trainers for consistent feature processing
- Provides feature validation before training begins
- Generates signal quality labels for supervised learning
- Ensures compatibility with MetaTrader 5 real-time feature calculation
================================================================================
'''

import pandas as pd
import numpy as np
from src.utils.logging_utils import get_logger

logger = get_logger('feature_engineering')

class FeatureEngineer:
    '''Feature engineering for Data_Exporter2.0 pre-calculated features'''
    
    def __init__(self):
        self.feature_columns = []
        self.derived_features = []
    
    def validate_data_exporter_2_features(self, df):
        '''Validate that all required Data_Exporter2.0 features are present'''
        required_features = self.get_data_exporter_2_features()
        missing_features = [feat for feat in required_features if feat not in df.columns]
        
        if missing_features:
            logger.error(f"Missing required features: {missing_features}")
            return False, missing_features
        
        logger.info("All 24 Data_Exporter2.0 features are present")
        return True, []
    
    def create_additional_derived_features(self, df):
        '''Create additional derived features from the 24 base features'''
        logger.info("Creating additional derived features from Data_Exporter2.0 features")
        
        # Trading session binary features (derived from Trading_Session)
        if 'Trading_Session' in df.columns:
            df['Is_Asian_Session'] = (df['Trading_Session'] == 1).astype(int)
            df['Is_London_Session'] = (df['Trading_Session'] == 2).astype(int) 
            df['Is_NY_Session'] = (df['Trading_Session'] == 3).astype(int)
            df['Is_Overlap_Session'] = (df['Trading_Session'] == 4).astype(int)
            
        # Day of week features
        if 'Day_Of_Week' in df.columns:
            df['Is_Monday'] = (df['Day_Of_Week'] == 1).astype(int)
            df['Is_Friday'] = (df['Day_Of_Week'] == 5).astype(int)
            df['Is_Midweek'] = (df['Day_Of_Week'].isin([2, 3, 4])).astype(int)
            
        # Hour-based features
        if 'Hour_Of_Day' in df.columns:
            df['Is_Market_Open_Hours'] = ((df['Hour_Of_Day'] >= 8) & (df['Hour_Of_Day'] <= 17)).astype(int)
            df['Is_Late_Night'] = ((df['Hour_Of_Day'] >= 22) | (df['Hour_Of_Day'] <= 5)).astype(int)
            
        # Signal type features
        if 'Signal_Type' in df.columns:
            df['Is_Buy_Signal'] = (df['Signal_Type'] == 1).astype(int)
            df['Is_Sell_Signal'] = (df['Signal_Type'] == -1).astype(int)
            
        # Gann zone features combinations
        if all(col in df.columns for col in ['Gann_Zone_Width_Pips', 'Price_Distance_From_Zone_Center']):
            # Relative position within zone
            df['Relative_Position_In_Zone'] = np.where(
                df['Gann_Zone_Width_Pips'] > 0,
                df['Price_Distance_From_Zone_Center'] / df['Gann_Zone_Width_Pips'],
                0
            )
            
        # EMA features combinations
        if all(col in df.columns for col in ['Median_EMA_Slope', 'Low_EMA_Slope']):
            df['EMA_Slopes_Alignment'] = (
                (df['Median_EMA_Slope'] > 0) & (df['Low_EMA_Slope'] > 0)
            ).astype(int) - (
                (df['Median_EMA_Slope'] < 0) & (df['Low_EMA_Slope'] < 0)
            ).astype(int)
            
        # Candle strength features
        if all(col in df.columns for col in ['Entry_Candle_Size_Pips', 'Previous_Candle_Size_Pips']):
            df['Candle_Size_Ratio'] = np.where(
                df['Previous_Candle_Size_Pips'] > 0,
                df['Entry_Candle_Size_Pips'] / df['Previous_Candle_Size_Pips'],
                1.0
            )
            
        # Risk-reward features
        if 'Initial_SL_Distance_Pips' in df.columns:
            df['Is_Tight_Stop'] = (df['Initial_SL_Distance_Pips'] < df['Initial_SL_Distance_Pips'].quantile(0.25)).astype(int)
            df['Is_Wide_Stop'] = (df['Initial_SL_Distance_Pips'] > df['Initial_SL_Distance_Pips'].quantile(0.75)).astype(int)
            
        # Market activity features
        if 'Recent_Gann_Signal_Frequency' in df.columns:
            df['High_Signal_Activity'] = (df['Recent_Gann_Signal_Frequency'] > df['Recent_Gann_Signal_Frequency'].quantile(0.75)).astype(int)
            df['Low_Signal_Activity'] = (df['Recent_Gann_Signal_Frequency'] < df['Recent_Gann_Signal_Frequency'].quantile(0.25)).astype(int)
            
        # Spread-based features
        if 'Spread' in df.columns:
            df['High_Spread'] = (df['Spread'] > df['Spread'].quantile(0.75)).astype(int)
            df['Low_Spread'] = (df['Spread'] < df['Spread'].quantile(0.25)).astype(int)
            
        # Define derived feature columns
        self.derived_features = [
            'Is_Asian_Session', 'Is_London_Session', 'Is_NY_Session', 'Is_Overlap_Session',
            'Is_Monday', 'Is_Friday', 'Is_Midweek',
            'Is_Market_Open_Hours', 'Is_Late_Night',
            'Is_Buy_Signal', 'Is_Sell_Signal',
            'Relative_Position_In_Zone', 'EMA_Slopes_Alignment', 'Candle_Size_Ratio',
            'Is_Tight_Stop', 'Is_Wide_Stop',
            'High_Signal_Activity', 'Low_Signal_Activity',
            'High_Spread', 'Low_Spread'
        ]
        
        # Only keep derived features that were actually created
        self.derived_features = [feat for feat in self.derived_features if feat in df.columns]
        
        logger.info(f"Created {len(self.derived_features)} additional derived features")
        return df
    
    def create_interaction_features(self, df):
        '''Create interaction features from Data_Exporter2.0 features'''
        logger.info("Creating interaction features")
        
        # Gann-EMA alignment
        if all(col in df.columns for col in ['Price_Distance_From_Zone_Boundary', 'Price_Distance_From_Relevant_EMA', 'Signal_Type']):
            df['Gann_EMA_Alignment'] = 0
            
            # For buy signals: both distances should be positive (price above levels)
            buy_mask = (df['Signal_Type'] == 1)
            aligned_buy = (df['Price_Distance_From_Zone_Boundary'] > 0) & (df['Price_Distance_From_Relevant_EMA'] > 0)
            df.loc[buy_mask & aligned_buy, 'Gann_EMA_Alignment'] = 1
            
            # For sell signals: both distances should be negative (price below levels)  
            sell_mask = (df['Signal_Type'] == -1)
            aligned_sell = (df['Price_Distance_From_Zone_Boundary'] < 0) & (df['Price_Distance_From_Relevant_EMA'] < 0)
            df.loc[sell_mask & aligned_sell, 'Gann_EMA_Alignment'] = 1
            
        # Time-based signal quality
        if all(col in df.columns for col in ['Trading_Session', 'Day_Of_Week', 'Entry_Candle_Body_Ratio']):
            df['Favorable_Time_Conditions'] = (
                (df['Trading_Session'].isin([2, 3, 4])) &  # London, NY, or Overlap
                (df['Day_Of_Week'].isin([1, 2, 3, 4, 5])) &  # Weekdays
                (df['Entry_Candle_Body_Ratio'] > 0.5)  # Strong candles
            ).astype(int)
            
        # Momentum alignment
        if all(col in df.columns for col in ['Active_Gann_Line_Slope', 'Median_EMA_Slope', 'Signal_Type']):
            df['Momentum_Alignment'] = 0
            
            buy_momentum = (df['Signal_Type'] == 1) & (df['Active_Gann_Line_Slope'] > 0) & (df['Median_EMA_Slope'] > 0)
            sell_momentum = (df['Signal_Type'] == -1) & (df['Active_Gann_Line_Slope'] < 0) & (df['Median_EMA_Slope'] < 0)
            
            df.loc[buy_momentum | sell_momentum, 'Momentum_Alignment'] = 1
            
        # Fresh signal conditions
        if all(col in df.columns for col in ['Bars_Since_Last_Gann_Color_Change', 'Bars_Since_Relevant_EMA_Cross']):
            df['Fresh_Signal'] = (
                (df['Bars_Since_Last_Gann_Color_Change'] <= 5) &  # Recent Gann change
                (df['Bars_Since_Relevant_EMA_Cross'] <= 10)  # Recent EMA cross
            ).astype(int)
            
        interaction_features = ['Gann_EMA_Alignment', 'Favorable_Time_Conditions', 'Momentum_Alignment', 'Fresh_Signal']
        interaction_features = [feat for feat in interaction_features if feat in df.columns]
        
        logger.info(f"Created {len(interaction_features)} interaction features")
        return df
    
    def engineer_all_features(self, df):
        '''Process all Data_Exporter2.0 features and create derived features'''
        logger.info("Starting Data_Exporter2.0 feature engineering")
        
        # Validate that we have the required base features
        is_valid, missing = self.validate_data_exporter_2_features(df)
        if not is_valid:
            raise ValueError(f"Cannot proceed with feature engineering. Missing features: {missing}")
        
        # The 24 base features are already calculated by Data_Exporter2.0
        base_features = self.get_data_exporter_2_features()
        
        # Create additional derived features
        df = self.create_additional_derived_features(df)
        
        # Create interaction features
        df = self.create_interaction_features(df)
        
        # Combine all feature columns
        self.feature_columns = base_features + self.derived_features
        
        # Only keep features that actually exist in the dataframe
        self.feature_columns = [feat for feat in self.feature_columns if feat in df.columns]
        
        logger.info(f"Feature engineering complete:")
        logger.info(f"  - Base features (Data_Exporter2.0): {len(base_features)}")
        logger.info(f"  - Derived features: {len(self.derived_features)}")
        logger.info(f"  - Total features: {len(self.feature_columns)}")
        
        return df
    
    def get_feature_columns(self):
        '''Get list of all feature columns (base + derived)'''
        return self.feature_columns
    
    def get_data_exporter_2_features(self):
        '''Get list of the 24 base Data_Exporter2.0 features'''
        return [
            # Gann Zone Features (6 features)
            'Gann_Zone_Width_Pips', 'Gann_Zone_Slope', 'Price_Distance_From_Zone_Center',
            'Price_Distance_From_Zone_Boundary', 'Active_Gann_Line_Slope', 'Bars_Since_Last_Gann_Color_Change',
            
            # EMA Features (5 features) 
            'EMA_Separation_Pips', 'Median_EMA_Slope', 'Low_EMA_Slope', 'Price_Distance_From_Relevant_EMA',
            'EMA_Convergence_Divergence',
            
            # Time-based Features (3 features)
            'Trading_Session', 'Day_Of_Week', 'Hour_Of_Day',
            
            # Candle Pattern Features (4 features)
            'Entry_Candle_Size_Pips', 'Entry_Candle_Body_Ratio', 'Previous_Candle_Size_Pips',
            'Bars_Since_Relevant_EMA_Cross',
            
            # Risk Management Features (2 features)
            'Initial_SL_Distance_Pips', 'Recent_Gann_Signal_Frequency'
        ]
    
    def get_derived_features(self):
        '''Get list of derived features created by this module'''
        return self.derived_features

def create_signal_labels(df, min_profit_points=0.0):
    '''Create signal quality labels based on profit/loss performance.

    Profit_Loss from the MQL5 exporter is expressed in raw price units
    (points) of the traded instrument, NOT forex pips. For indices such as
    GER40/NAS100 a value of 50.0 means 50 index points. The previous
    implementation multiplied by 10,000 (a forex pip conversion), which
    inflated every value so much that the tiered thresholds collapsed to a
    simple sign check. The labelling is now an explicit sign check with an
    optional dead-zone (min_profit_points) so the intent is honest and the
    behaviour is identical across instruments.
    '''
    logger.info("Creating signal quality labels")

    # Debug: Show profit/loss statistics (raw price units)
    profit_stats = df['Profit_Loss'].describe()
    logger.info(f"Profit/Loss statistics (price points): {dict(profit_stats)}")

    df['Signal_Quality'] = (df['Profit_Loss'] > min_profit_points).astype(int)
    
    # Log distribution
    quality_dist = df['Signal_Quality'].value_counts()
    logger.info(f"Signal quality distribution: {dict(quality_dist)}")
    
    return df

def validate_feature_completeness(df, required_coverage=0.95):
    '''Validate that features have sufficient data coverage'''
    issues = []
    
    feature_engineer = FeatureEngineer()
    all_features = feature_engineer.get_data_exporter_2_features()
    
    for feature in all_features:
        if feature in df.columns:
            coverage = 1 - (df[feature].isnull().sum() / len(df))
            if coverage < required_coverage:
                issues.append(f"Feature '{feature}' has {coverage:.1%} coverage (< {required_coverage:.1%})")
        else:
            issues.append(f"Missing feature: {feature}")
    
    return len(issues) == 0, issues