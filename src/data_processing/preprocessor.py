"""
Data Preprocessor for Financial ML - Data_Exporter2.0 Compatible
Updated to handle Data_Exporter2.0 format with 24 pre-calculated features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.utils.logging_utils import get_logger

logger = get_logger('preprocessor')

class DataPreprocessor:
    """Data preprocessing for financial forecasting"""
    
    def __init__(self):
        self.scaler = None
        self.feature_stats = {}
    
    def clean_data(self, df, remove_duplicates=True, fill_missing=True):
        """Clean the data"""
        logger.info("Starting data cleaning")
        
        original_shape = df.shape
        
        # Remove duplicates
        if remove_duplicates:
            df = df.drop_duplicates()
            
        # Handle missing values
        if fill_missing:
            # Fill numeric columns with median
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if df[col].isnull().sum() > 0:
                    df[col] = df[col].fillna(df[col].median())
            
            # Fill categorical columns with mode
            categorical_columns = df.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if df[col].isnull().sum() > 0:
                    mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                    df[col] = df[col].fillna(mode_value)
        
        # Convert data types for core columns
        if 'Signal_Type' in df.columns:
            df['Signal_Type'] = pd.to_numeric(df['Signal_Type'], errors='coerce')
        
        if 'Entry_Price' in df.columns:
            df['Entry_Price'] = pd.to_numeric(df['Entry_Price'], errors='coerce')
        
        if 'Profit_Loss' in df.columns:
            df['Profit_Loss'] = pd.to_numeric(df['Profit_Loss'], errors='coerce')
            
        # Convert Data_Exporter2.0 specific features to proper types
        self._convert_data_exporter_2_features(df)
        
        logger.info(f"Data cleaned: {original_shape} -> {df.shape}")
        return df
    
    def prepare_timestamps(self, df):
        """Prepare timestamp column"""
        timestamp_columns = ['Timestamp', 'Time', 'Date', 'DateTime']
        
        for col in timestamp_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    if col != 'Timestamp':
                        df['Timestamp'] = df[col]
                    break
                except Exception:
                    continue
        
        # Sort by timestamp if available
        if 'Timestamp' in df.columns:
            df = df.sort_values('Timestamp').reset_index(drop=True)
        
        return df
    
    def _convert_data_exporter_2_features(self, df):
        """Convert Data_Exporter2.0 features to appropriate data types"""
        
        # Integer features that should be whole numbers
        integer_features = [
            'Bars_Since_Last_Gann_Color_Change', 'EMA_Convergence_Divergence', 
            'Trading_Session', 'Day_Of_Week', 'Hour_Of_Day',
            'Bars_Since_Relevant_EMA_Cross', 'Recent_Gann_Signal_Frequency'
        ]
        
        # Float features (pips, slopes, ratios, etc.)
        float_features = [
            'Gann_Zone_Width_Pips', 'Gann_Zone_Slope', 'Price_Distance_From_Zone_Center',
            'Price_Distance_From_Zone_Boundary', 'Active_Gann_Line_Slope',
            'EMA_Separation_Pips', 'Median_EMA_Slope', 'Low_EMA_Slope', 
            'Price_Distance_From_Relevant_EMA', 'Entry_Candle_Size_Pips',
            'Entry_Candle_Body_Ratio', 'Previous_Candle_Size_Pips',
            'Initial_SL_Distance_Pips', 'Spread'
        ]
        
        # Convert integer features
        for col in integer_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        
        # Convert float features
        for col in float_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Validate ranges for specific features
        if 'Trading_Session' in df.columns:
            # Should be 1-4 (Asian=1, London=2, NY=3, Overlap=4)
            df.loc[~df['Trading_Session'].between(1, 4), 'Trading_Session'] = np.nan
            
        if 'Day_Of_Week' in df.columns:
            # Should be 1-7 (Monday=1 to Sunday=7, but trading is Mon-Fri)
            df.loc[~df['Day_Of_Week'].between(1, 7), 'Day_Of_Week'] = np.nan
            
        if 'Hour_Of_Day' in df.columns:
            # Should be 0-23
            df.loc[~df['Hour_Of_Day'].between(0, 23), 'Hour_Of_Day'] = np.nan
            
        if 'Entry_Candle_Body_Ratio' in df.columns:
            # Should be 0-1 (ratio of body to total candle range)
            df.loc[~df['Entry_Candle_Body_Ratio'].between(0, 1), 'Entry_Candle_Body_Ratio'] = np.nan
    
    def validate_data_exporter_2_format(self, df):
        """Validate Data_Exporter2.0 specific format and features"""
        issues = []
        
        # Expected 24 features
        expected_features = [
            'Gann_Zone_Width_Pips', 'Gann_Zone_Slope', 'Price_Distance_From_Zone_Center',
            'Price_Distance_From_Zone_Boundary', 'Active_Gann_Line_Slope', 'Bars_Since_Last_Gann_Color_Change',
            'EMA_Separation_Pips', 'Median_EMA_Slope', 'Low_EMA_Slope', 'Price_Distance_From_Relevant_EMA',
            'EMA_Convergence_Divergence', 'Trading_Session', 'Day_Of_Week', 'Hour_Of_Day',
            'Entry_Candle_Size_Pips', 'Entry_Candle_Body_Ratio', 'Previous_Candle_Size_Pips',
            'Bars_Since_Relevant_EMA_Cross', 'Initial_SL_Distance_Pips', 'Recent_Gann_Signal_Frequency'
        ]
        
        missing_features = [feat for feat in expected_features if feat not in df.columns]
        if missing_features:
            issues.append(f"Missing Data_Exporter2.0 features: {missing_features}")
        
        # Check for old format columns that shouldn't exist
        old_format_cols = ['EMA_Low', 'EMA_Median', 'Gann_High', 'Gann_Low', 'Market_Volatility', 'Volume', 'Duration_Bars']
        found_old_cols = [col for col in old_format_cols if col in df.columns]
        if found_old_cols:
            issues.append(f"Found old format columns (should not exist in Data_Exporter2.0): {found_old_cols}")
        
        # Validate feature value ranges
        if 'Trading_Session' in df.columns:
            invalid_sessions = df[~df['Trading_Session'].between(1, 4)]['Trading_Session'].dropna()
            if len(invalid_sessions) > 0:
                issues.append(f"Invalid Trading_Session values (should be 1-4): {invalid_sessions.unique()}")
        
        if 'Entry_Candle_Body_Ratio' in df.columns:
            invalid_ratios = df[~df['Entry_Candle_Body_Ratio'].between(0, 1)]['Entry_Candle_Body_Ratio'].dropna()
            if len(invalid_ratios) > 0:
                issues.append(f"Invalid Entry_Candle_Body_Ratio values (should be 0-1): found {len(invalid_ratios)} invalid values")
        
        return len(issues) == 0, issues
    
    def scale_features(self, X_train, X_test=None, method='standard'):
        """Scale features"""
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def validate_data(self, df):
        """Validate data quality"""
        issues = []
        
        # Check for required columns (Data_Exporter2.0 format)
        required_columns = [
            # Core signal data
            'Timestamp', 'Signal_Type', 'Entry_Price', 'Profit_Loss',
            # Key features from Data_Exporter2.0 (subset of 24 features for validation)
            'Gann_Zone_Width_Pips', 'EMA_Separation_Pips', 'Trading_Session',
            'Day_Of_Week', 'Hour_Of_Day', 'Entry_Candle_Size_Pips', 'Spread'
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
        
        # Check data size
        if len(df) < 100:
            issues.append("Dataset too small (< 100 rows)")
        
        # Check for excessive missing values
        missing_counts = df.isnull().sum()
        critical_missing = missing_counts[missing_counts > len(df) * 0.5]
        
        if not critical_missing.empty:
            issues.append(f"High missing values in columns: {critical_missing.index.tolist()}")
        
        return len(issues) == 0, issues
    
    def remove_outliers(self, df, columns=None, method='iqr', threshold=3):
        """Remove outliers from specified columns with Data_Exporter2.0 awareness"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        original_len = len(df)
        
        # Skip outlier removal for certain categorical/discrete features
        protected_columns = [
            'Signal_Type', 'Trading_Session', 'Day_Of_Week', 'Hour_Of_Day',
            'Bars_Since_Last_Gann_Color_Change', 'Bars_Since_Relevant_EMA_Cross',
            'Recent_Gann_Signal_Frequency', 'EMA_Convergence_Divergence'
        ]
        
        for col in columns:
            if col not in df.columns or col in protected_columns:
                continue
                
            # Use different thresholds for different types of features
            if col in ['Entry_Candle_Body_Ratio']:
                # Body ratio should be 0-1, use stricter bounds
                df = df[(df[col] >= 0) & (df[col] <= 1)]
            elif 'Pips' in col:
                # For pip-based features, use more lenient outlier detection
                if method == 'iqr':
                    Q1 = df[col].quantile(0.10)  # More lenient quantiles
                    Q3 = df[col].quantile(0.90)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 2.0 * IQR  # More lenient multiplier
                    upper_bound = Q3 + 2.0 * IQR
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                elif method == 'zscore':
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    df = df[z_scores < (threshold * 1.5)]  # More lenient for pips
            else:
                # Standard outlier removal for other features
                if method == 'iqr':
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                    
                elif method == 'zscore':
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    df = df[z_scores < threshold]
        
        logger.info(f"Outlier removal: {original_len} -> {len(df)} rows")
        return df
    
    def balance_classes(self, X, y, method='undersample'):
        """Balance classes in the dataset"""
        from collections import Counter
        
        original_distribution = Counter(y)
        logger.info(f"Original class distribution: {original_distribution}")
        
        if method == 'undersample':
            # Undersample majority class
            min_class_size = min(original_distribution.values())
            
            indices_to_keep = []
            for class_label in original_distribution.keys():
                class_indices = np.where(y == class_label)[0]
                selected_indices = np.random.choice(class_indices, min_class_size, replace=False)
                indices_to_keep.extend(selected_indices)
            
            indices_to_keep = sorted(indices_to_keep)
            X_balanced = X.iloc[indices_to_keep]
            y_balanced = y.iloc[indices_to_keep]
            
        elif method == 'oversample':
            # Simple random oversampling
            max_class_size = max(original_distribution.values())
            
            X_balanced = []
            y_balanced = []
            
            for class_label in original_distribution.keys():
                class_indices = np.where(y == class_label)[0]
                current_size = len(class_indices)
                
                if current_size < max_class_size:
                    # Oversample this class
                    additional_samples = max_class_size - current_size
                    additional_indices = np.random.choice(class_indices, additional_samples, replace=True)
                    all_indices = np.concatenate([class_indices, additional_indices])
                else:
                    all_indices = class_indices
                
                X_balanced.extend(X.iloc[all_indices].values.tolist())
                y_balanced.extend(y.iloc[all_indices].values.tolist())
            
            X_balanced = pd.DataFrame(X_balanced, columns=X.columns)
            y_balanced = pd.Series(y_balanced)
        
        new_distribution = Counter(y_balanced)
        logger.info(f"Balanced class distribution: {new_distribution}")
        
        return X_balanced, y_balanced
    
    def create_time_splits(self, df, test_size=0.2):
        """Create time-based train/test splits"""
        total_samples = len(df)
        split_point = int(total_samples * (1 - test_size))
        
        train_df = df.iloc[:split_point].copy()
        test_df = df.iloc[split_point:].copy()
        
        logger.info(f"Time-based split: Train {len(train_df)}, Test {len(test_df)}")
        return train_df, test_df
    
    def get_feature_stats(self, X, feature_columns):
        """Calculate feature statistics"""
        self.feature_stats = {}
        
        for col in feature_columns:
            if col in X.columns:
                self.feature_stats[col] = {
                    'mean': X[col].mean(),
                    'std': X[col].std(),
                    'min': X[col].min(),
                    'max': X[col].max(),
                    'median': X[col].median(),
                    'missing_count': X[col].isnull().sum()
                }
        
        return self.feature_stats
    
    def get_data_exporter_2_features(self):
        """Get list of all 24 Data_Exporter2.0 feature columns"""
        return [
            'Gann_Zone_Width_Pips', 'Gann_Zone_Slope', 'Price_Distance_From_Zone_Center',
            'Price_Distance_From_Zone_Boundary', 'Active_Gann_Line_Slope', 'Bars_Since_Last_Gann_Color_Change',
            'EMA_Separation_Pips', 'Median_EMA_Slope', 'Low_EMA_Slope', 'Price_Distance_From_Relevant_EMA',
            'EMA_Convergence_Divergence', 'Trading_Session', 'Day_Of_Week', 'Hour_Of_Day',
            'Entry_Candle_Size_Pips', 'Entry_Candle_Body_Ratio', 'Previous_Candle_Size_Pips',
            'Bars_Since_Relevant_EMA_Cross', 'Initial_SL_Distance_Pips', 'Recent_Gann_Signal_Frequency'
        ]
    
    def get_core_columns(self):
        """Get list of core data columns (non-feature columns)"""
        return ['Timestamp', 'Signal_Type', 'Entry_Price', 'Profit_Loss', 'Spread']
