"""
Improved CSV Data Loader for Financial ML System - Data_Exporter2.0 Compatible
This module provides robust CSV loading functionality with better error handling
and data validation. Updated to handle Data_Exporter2.0 format with 24 pre-calculated
features from the Features.txt specification.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import chardet
# Suppress specific pandas warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')

class RobustCSVLoader:
    def __init__(self):
        self.df = None
        self.encoding = None
        self.separator = None
        
    def detect_encoding(self, file_path):
        """Detect file encoding using chardet"""
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # Read first 10KB
            result = chardet.detect(raw_data)
            return result['encoding']
    
    def detect_separator(self, file_path, encoding):
        """Detect the most likely separator"""
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            first_lines = [f.readline() for _ in range(5)]
        
        # Count occurrences of common separators
        separators = [',', '\t', ';', '|']
        sep_counts = {}
        
        for sep in separators:
            counts = [line.count(sep) for line in first_lines if line]
            if counts and all(c == counts[0] for c in counts) and counts[0] > 0:
                sep_counts[sep] = counts[0]
        
        if sep_counts:
            return max(sep_counts, key=sep_counts.get)
        return ','  # Default to comma
    
    def clean_dataframe(self, df):
        """Clean and standardize the dataframe"""
        # Remove BOM and special characters from column names
        df.columns = [col.strip().replace('\ufeff', '').replace('ÿþ', '') for col in df.columns]
        
        # Remove completely empty rows and columns
        df = df.dropna(how='all')
        df = df.dropna(axis=1, how='all')
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        return df
    
    def load_csv_intelligent(self, file_path):
        """
        Intelligently load CSV with automatic detection of encoding and separator
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            tuple: (success: bool, dataframe: pd.DataFrame or None, info: dict)
        """
        info = {
            'encoding': None,
            'separator': None,
            'rows': 0,
            'columns': 0,
            'column_names': [],
            'errors': []
        }
        
        try:
            # Convert to Path object
            file_path = Path(file_path)
            
            if not file_path.exists():
                info['errors'].append(f"File not found: {file_path}")
                return False, None, info
            
            # Detect encoding
            try:
                encoding = self.detect_encoding(file_path)
                info['encoding'] = encoding
            except Exception as e:
                info['errors'].append(f"Encoding detection failed: {str(e)}")
                encoding = 'utf-8'  # Fallback
            
            # Detect separator
            try:
                separator = self.detect_separator(file_path, encoding)
                info['separator'] = separator
            except Exception as e:
                info['errors'].append(f"Separator detection failed: {str(e)}")
                separator = ','  # Fallback
            
            # Try loading with detected parameters
            try:
                df = pd.read_csv(
                    file_path,
                    encoding=encoding,
                    sep=separator,
                    on_bad_lines='skip',
                    low_memory=False
                )
                
                # Clean the dataframe
                df = self.clean_dataframe(df)
                
                # Update info
                info['rows'] = len(df)
                info['columns'] = len(df.columns)
                info['column_names'] = list(df.columns)
                
                self.df = df
                self.encoding = encoding
                self.separator = separator
                
                return True, df, info
                
            except Exception as e:
                info['errors'].append(f"Primary load failed: {str(e)}")
                
                # Try alternative encodings
                alternative_encodings = ['utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
                for alt_encoding in alternative_encodings:
                    if alt_encoding == encoding:
                        continue
                        
                    try:
                        df = pd.read_csv(
                            file_path,
                            encoding=alt_encoding,
                            sep=separator,
                            on_bad_lines='skip',
                            low_memory=False
                        )
                        
                        df = self.clean_dataframe(df)
                        
                        info['encoding'] = alt_encoding
                        info['rows'] = len(df)
                        info['columns'] = len(df.columns)
                        info['column_names'] = list(df.columns)
                        info['errors'].append(f"Loaded with alternative encoding: {alt_encoding}")
                        
                        self.df = df
                        self.encoding = alt_encoding
                        self.separator = separator
                        
                        return True, df, info
                        
                    except Exception:
                        continue
                
                # If all attempts failed
                info['errors'].append("All loading attempts failed")
                return False, None, info
                
        except Exception as e:
            info['errors'].append(f"Unexpected error: {str(e)}")
            return False, None, info
    
    def validate_financial_data(self, df, required_columns=None):
        """
        Validate that the dataframe contains expected financial data columns
        
        Args:
            df: pandas DataFrame
            required_columns: list of required column names
            
        Returns:
            tuple: (is_valid: bool, missing_columns: list, validation_report: dict)
        """
        if required_columns is None:
            # Default expected columns for Data_Exporter2.0 format (24 features + core data)
            required_columns = [
                # Core signal data
                'Timestamp', 'Signal_Type', 'Entry_Price', 'Profit_Loss',
                # Features.txt features (24 features)
                'Gann_Zone_Width_Pips', 'Gann_Zone_Slope', 'Price_Distance_From_Zone_Center',
                'Price_Distance_From_Zone_Boundary', 'Active_Gann_Line_Slope', 'Bars_Since_Last_Gann_Color_Change',
                'EMA_Separation_Pips', 'Median_EMA_Slope', 'Low_EMA_Slope', 'Price_Distance_From_Relevant_EMA',
                'EMA_Convergence_Divergence', 'Trading_Session', 'Day_Of_Week', 'Hour_Of_Day',
                'Entry_Candle_Size_Pips', 'Entry_Candle_Body_Ratio', 'Previous_Candle_Size_Pips',
                'Bars_Since_Relevant_EMA_Cross', 'Initial_SL_Distance_Pips', 'Recent_Gann_Signal_Frequency',
                # Essential market data
                'Spread'
            ]
        
        validation_report = {
            'has_required_columns': False,
            'missing_columns': [],
            'data_types': {},
            'null_counts': {},
            'unique_counts': {},
            'numeric_columns': [],
            'datetime_columns': [],
            'warnings': []
        }
        
        # Check for required columns (case-insensitive)
        df_columns_lower = [col.lower() for col in df.columns]
        missing_columns = []
        
        for req_col in required_columns:
            if req_col.lower() not in df_columns_lower:
                missing_columns.append(req_col)
        
        validation_report['missing_columns'] = missing_columns
        validation_report['has_required_columns'] = len(missing_columns) == 0
        
        # Analyse data types and quality
        for col in df.columns:
            validation_report['data_types'][col] = str(df[col].dtype)
            validation_report['null_counts'][col] = df[col].isnull().sum()
            validation_report['unique_counts'][col] = df[col].nunique()
            
            # Check if numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                validation_report['numeric_columns'].append(col)
            
            # Check if could be datetime
            if 'time' in col.lower() or 'date' in col.lower():
                try:
                    pd.to_datetime(df[col], errors='coerce')
                    validation_report['datetime_columns'].append(col)
                except (ValueError, TypeError):
                    pass
        
        # Add warnings for data quality issues
        for col, null_count in validation_report['null_counts'].items():
            if null_count > len(df) * 0.1:  # More than 10% nulls
                validation_report['warnings'].append(f"Column '{col}' has {null_count} null values ({null_count/len(df)*100:.1f}%)")
        
        is_valid = validation_report['has_required_columns'] and len(validation_report['warnings']) == 0
        
        return is_valid, missing_columns, validation_report
    
    def validate_data_exporter_2_format(self, df):
        """
        Validate that the dataframe matches Data_Exporter2.0 format exactly
        
        Args:
            df: pandas DataFrame
            
        Returns:
            tuple: (is_valid: bool, issues: list, feature_report: dict)
        """
        issues = []
        feature_report = {
            'has_all_24_features': False,
            'missing_features': [],
            'extra_columns': [],
            'feature_data_quality': {}
        }
        
        # Expected 24 features from Features.txt specification
        expected_features = [
            'Gann_Zone_Width_Pips', 'Gann_Zone_Slope', 'Price_Distance_From_Zone_Center',
            'Price_Distance_From_Zone_Boundary', 'Active_Gann_Line_Slope', 'Bars_Since_Last_Gann_Color_Change',
            'EMA_Separation_Pips', 'Median_EMA_Slope', 'Low_EMA_Slope', 'Price_Distance_From_Relevant_EMA',
            'EMA_Convergence_Divergence', 'Trading_Session', 'Day_Of_Week', 'Hour_Of_Day',
            'Entry_Candle_Size_Pips', 'Entry_Candle_Body_Ratio', 'Previous_Candle_Size_Pips',
            'Bars_Since_Relevant_EMA_Cross', 'Initial_SL_Distance_Pips', 'Recent_Gann_Signal_Frequency'
        ]
        
        # Core data columns
        core_columns = ['Timestamp', 'Signal_Type', 'Entry_Price', 'Profit_Loss', 'Spread']
        
        # Check for missing features
        missing_features = [feat for feat in expected_features if feat not in df.columns]
        missing_core = [col for col in core_columns if col not in df.columns]
        
        feature_report['missing_features'] = missing_features + missing_core
        feature_report['has_all_24_features'] = len(missing_features) == 0
        
        # Check for unexpected extra columns (old format indicators)
        old_format_indicators = ['EMA_Low', 'EMA_Median', 'Gann_High', 'Gann_Low', 'Market_Volatility', 'Volume', 'Duration_Bars']
        extra_columns = [col for col in df.columns if col in old_format_indicators]
        feature_report['extra_columns'] = extra_columns
        
        if missing_features:
            issues.append(f"Missing {len(missing_features)} required features: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")
        
        if missing_core:
            issues.append(f"Missing core columns: {missing_core}")
            
        if extra_columns:
            issues.append(f"Found old format columns (should not exist in Data_Exporter2.0): {extra_columns}")
        
        # Validate feature data quality
        for feature in expected_features:
            if feature in df.columns:
                null_count = df[feature].isnull().sum()
                infinite_count = np.isinf(df[feature]).sum() if pd.api.types.is_numeric_dtype(df[feature]) else 0
                
                feature_report['feature_data_quality'][feature] = {
                    'null_count': null_count,
                    'infinite_count': infinite_count,
                    'data_type': str(df[feature].dtype)
                }
                
                if null_count > len(df) * 0.05:  # More than 5% nulls is concerning for features
                    issues.append(f"Feature '{feature}' has {null_count} null values ({null_count/len(df)*100:.1f}%)")
                
                if infinite_count > 0:
                    issues.append(f"Feature '{feature}' has {infinite_count} infinite values")
        
        is_valid = len(issues) == 0
        return is_valid, issues, feature_report

# Integration with your existing TimeBasedSignalTrainer class
def load_and_prepare_data_improved(self):
    """
    Improved version of load_and_prepare_data using RobustCSVLoader
    This method can replace the existing one in your TimeBasedSignalTrainer class
    """
    print("Loading signal data with improved CSV loader...")
    
    loader = RobustCSVLoader()
    success, df, load_info = loader.load_csv_intelligent(self.csv_file)
    
    if not success:
        print(f"Failed to load CSV file: {self.csv_file}")
        for error in load_info['errors']:
            print(f"  - {error}")
        return False
    
    print(f"Successfully loaded CSV with:")
    print(f"  - Encoding: {load_info['encoding']}")
    print(f"  - Separator: '{load_info['separator']}'")
    print(f"  - Rows: {load_info['rows']}")
    print(f"  - Columns: {load_info['columns']}")
    
    # Validate financial data
    is_valid, missing_cols, validation = loader.validate_financial_data(df)
    
    if not is_valid:
        print("Data validation failed:")
        if missing_cols:
            print(f"  - Missing required columns: {missing_cols}")
        for warning in validation['warnings']:
            print(f"  - Warning: {warning}")
    
    # Apply column mapping for Data_Exporter2.0 format variations
    column_mapping = {
        # Timestamp variations
        'T': 'Timestamp',
        'Time': 'Timestamp', 
        'Date': 'Timestamp',
        'DateTime': 'Timestamp',
        'timestamp': 'Timestamp',
        
        # Signal Type variations  
        'SignalType': 'Signal_Type',
        'signal_type': 'Signal_Type',
        'Type': 'Signal_Type',
        
        # Entry Price variations
        'EntryPrice': 'Entry_Price',
        'entry_price': 'Entry_Price',
        'Price': 'Entry_Price',
        
        # Profit/Loss variations
        'ProfitLoss': 'Profit_Loss',
        'profit_loss': 'Profit_Loss',
        'PnL': 'Profit_Loss',
        'P&L': 'Profit_Loss',
        
        # Hour variations (Data_Exporter2.0 uses Hour_Of_Day)
        'Hour': 'Hour_Of_Day',
        'hour': 'Hour_Of_Day',
        'HourOfDay': 'Hour_Of_Day',
        
        # Day of Week variations  
        'DayOfWeek': 'Day_Of_Week',
        'day_of_week': 'Day_Of_Week',
        'DOW': 'Day_Of_Week'
    }
    
    # Apply mapping
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns and old_name != new_name:
            df = df.rename(columns={old_name: new_name})
    
    # Convert timestamp to datetime with error handling
    if 'Timestamp' in df.columns:
        try:
            # Try multiple datetime formats
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            
            # Remove rows with invalid timestamps
            invalid_timestamps = df['Timestamp'].isnull().sum()
            if invalid_timestamps > 0:
                print(f"Removing {invalid_timestamps} rows with invalid timestamps")
                df = df.dropna(subset=['Timestamp'])
            
        except Exception as e:
            print(f"Error converting timestamp: {e}")
            return False
    else:
        print("Error: No timestamp column found")
        return False
    
    # Sort by timestamp
    df = df.sort_values('Timestamp').reset_index(drop=True)
    
    # Assign to instance
    self.df = df
    
    print(f"\nData successfully loaded and cleaned:")
    print(f"Final shape: {self.df.shape}")
    print(f"Columns: {list(self.df.columns)}")
    print(f"Time range: {self.df['Timestamp'].min()} to {self.df['Timestamp'].max()}")
    
    return True

# Example usage
if __name__ == "__main__":
    # Test the improved CSV loader
    loader = RobustCSVLoader()
    
    # Load your CSV file from Data_Exporter2.0
    success, df, info = loader.load_csv_intelligent('NAS100_signal_data_v2.csv')
    
    if success:
        print("\nCSV loaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"\nFirst 5 rows:")
        print(df.head())
        
        # Validate as financial data
        is_valid, missing, validation = loader.validate_financial_data(df)
        print(f"\nValidation result: {'PASSED' if is_valid else 'FAILED'}")
        
        if validation['warnings']:
            print("\nWarnings:")
            for warning in validation['warnings']:
                print(f"  - {warning}")
    else:
        print("\nFailed to load CSV")
        print("Errors:")
        for error in info['errors']:
            print(f"  - {error}")
