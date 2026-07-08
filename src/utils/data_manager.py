"""
Data Manager for ML Financial Forecasting
========================================
Handles data loading, preprocessing, and management
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import warnings
# Suppress specific pandas warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')

class DataManager:
    """Data management class for financial forecasting"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.raw_data_dir = os.path.join(data_dir, 'signal')
        
        # Create directories if they don't exist
        os.makedirs(self.raw_data_dir, exist_ok=True)
        
        # Data storage
        self.data = None
        self.data_info = {}
        self.preprocessing_history = []
        
    def load_csv(self, file_path, **kwargs):
        """Load CSV file with automatic encoding and separator detection"""
        # Try encodings in order of likelihood
        encodings = ['utf-8', 'utf-8-sig', 'utf-16', 'utf-16-le', 'utf-16-be', 
                     'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            if encoding is None:
                continue
                
            try:
                # Try different separators
                for sep in [',', '\t', ';', '|']:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding, sep=sep, 
                                       on_bad_lines='skip', low_memory=False, **kwargs)
                        
                        # Check if we got a valid dataframe with multiple columns
                        if len(df.columns) > 1:
                            # Clean column names
                            df.columns = [col.strip().replace('\ufeff', '').replace('ÿþ', '') 
                                        for col in df.columns]
                            
                            # Remove empty rows/columns
                            df = df.dropna(how='all').dropna(axis=1, how='all')
                            
                            # Store the data
                            self.data = df
                            self.data_info = {
                                'source': file_path,
                                'encoding': encoding,
                                'separator': sep,
                                'loaded_at': datetime.now(),
                                'shape': df.shape,
                                'columns': list(df.columns)
                            }
                            
                            print(f"Successfully loaded {file_path}")
                            print(f"  Encoding: {encoding}")
                            print(f"  Separator: '{sep}'")
                            print(f"  Shape: {df.shape}")
                            
                            # Validate if it looks like financial data
                            self._validate_financial_columns(df)
                            
                            return True
                            
                    except pd.errors.ParserError:
                        continue
                    except Exception as e:
                        if "codec" not in str(e).lower():
                            print(f"Parser error with {encoding}, {sep}: {e}")
                        continue
                        
            except Exception as e:
                if "codec" in str(e).lower() or "decode" in str(e).lower():
                    continue
                else:
                    print(f"Unexpected error with {encoding}: {e}")
        
        raise ValueError(f"Could not load CSV file: {file_path}. "
                        f"Tried encodings: {encodings} with separators: [',', '\\t', ';', '|']")

    def _validate_financial_columns(self, df):
        """Helper method to validate and map financial columns"""
        # Common column name variations to standard names
        column_mapping = {
            # Timestamp variations
            'T': 'Timestamp', 'Time': 'Timestamp', 'Date': 'Timestamp',
            'DateTime': 'Timestamp', 'timestamp': 'Timestamp',
            
            # Signal Type variations  
            'SignalType': 'Signal_Type', 'signal_type': 'Signal_Type',
            'Type': 'Signal_Type',
            
            # Entry Price variations
            'EntryPrice': 'Entry_Price', 'entry_price': 'Entry_Price',
            'Price': 'Entry_Price',
            
            # Profit/Loss variations
            'ProfitLoss': 'Profit_Loss', 'profit_loss': 'Profit_Loss',
            'PnL': 'Profit_Loss', 'P&L': 'Profit_Loss',
            
            # Data_Exporter2.0 variations
            'Hour': 'Hour_Of_Day', 'DayOfWeek': 'Day_Of_Week',
            'day_of_week': 'Day_Of_Week', 'hour': 'Hour_Of_Day'
        }
        
        # Apply mapping
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and old_name != new_name:
                df.rename(columns={old_name: new_name}, inplace=True)
                print(f"  Renamed column: {old_name} -> {new_name}")
        
        # Update column list in data_info
        if hasattr(self, 'data_info'):
            self.data_info['columns'] = list(df.columns)
        
        # Check for essential financial columns (Data_Exporter2.0 format)
        essential_columns = ['Timestamp', 'Signal_Type', 'Entry_Price', 'Profit_Loss', 'Trading_Session', 'Gann_Zone_Width_Pips']
        found_columns = [col for col in essential_columns if col in df.columns]
        missing_columns = [col for col in essential_columns if col not in df.columns]
        
        if found_columns:
            print(f"  Found financial columns: {found_columns}")
        if missing_columns:
            print(f"  Warning: Missing expected columns: {missing_columns}")
    
    def load_multiple_csv(self, file_paths, combine_method='concat'):
        """Load multiple CSV files"""
        dataframes = []
        
        for file_path in file_paths:
            try:
                temp_manager = DataManager()
                temp_manager.load_csv(file_path)
                dataframes.append(temp_manager.data)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        if not dataframes:
            raise ValueError("No valid CSV files could be loaded")
        
        if combine_method == 'concat':
            self.data = pd.concat(dataframes, ignore_index=True)
        elif combine_method == 'merge':
            # Merge on common columns (requires all files to have same structure)
            self.data = dataframes[0]
            for df in dataframes[1:]:
                self.data = pd.merge(self.data, df, how='outer')
        
        self.data_info = {
            'source': 'multiple_files',
            'files': file_paths,
            'combine_method': combine_method,
            'loaded_at': datetime.now(),
            'shape': self.data.shape,
            'columns': list(self.data.columns)
        }
        
        print(f"Successfully loaded and combined {len(dataframes)} files")
        return True
    
    def validate_data(self):
        """Validate loaded data for financial forecasting"""
        if self.data is None:
            return False, "No data loaded"
        
        issues = []
        
        # Check for required columns
        required_columns = ['Signal_Type', 'Entry_Price', 'Profit_Loss']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
        
        # Check for timestamp column
        timestamp_columns = ['Timestamp', 'Time', 'Date', 'DateTime']
        has_timestamp = any(col in self.data.columns for col in timestamp_columns)
        
        if not has_timestamp:
            issues.append("No timestamp column found")
        
        # Check data types
        if 'Signal_Type' in self.data.columns:
            if not pd.api.types.is_numeric_dtype(self.data['Signal_Type']):
                issues.append("Signal_Type should be numeric")
        
        if 'Entry_Price' in self.data.columns:
            if not pd.api.types.is_numeric_dtype(self.data['Entry_Price']):
                issues.append("Entry_Price should be numeric")
        
        # Check for missing values
        missing_counts = self.data.isnull().sum()
        critical_missing = missing_counts[missing_counts > len(self.data) * 0.5]
        
        if not critical_missing.empty:
            issues.append(f"High missing values in columns: {critical_missing.index.tolist()}")
        
        # Check for duplicates
        duplicate_count = self.data.duplicated().sum()
        if duplicate_count > 0:
            issues.append(f"Found {duplicate_count} duplicate rows")
        
        # Check data size
        if len(self.data) < 100:
            issues.append("Dataset too small (< 100 rows)")
        
        return len(issues) == 0, issues
    
    def clean_data(self, remove_duplicates=True, fill_missing=True):
        """Clean the loaded data"""
        if self.data is None:
            raise ValueError("No data loaded")
        
        original_shape = self.data.shape
        cleaning_steps = []
        
        # Remove duplicates
        if remove_duplicates:
            before_count = len(self.data)
            self.data = self.data.drop_duplicates()
            after_count = len(self.data)
            if before_count != after_count:
                cleaning_steps.append(f"Removed {before_count - after_count} duplicate rows")
        
        # Handle missing values
        if fill_missing:
            missing_before = self.data.isnull().sum().sum()
            
            # Fill numeric columns with median
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if self.data[col].isnull().sum() > 0:
                    self.data[col] = self.data[col].fillna(self.data[col].median())
            
            # Fill categorical columns with mode
            categorical_columns = self.data.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if self.data[col].isnull().sum() > 0:
                    mode_value = self.data[col].mode().iloc[0] if not self.data[col].mode().empty else 'Unknown'
                    self.data[col] = self.data[col].fillna(mode_value)
            
            missing_after = self.data.isnull().sum().sum()
            if missing_before != missing_after:
                cleaning_steps.append(f"Filled {missing_before - missing_after} missing values")
        
        # Convert data types
        if 'Signal_Type' in self.data.columns:
            self.data['Signal_Type'] = pd.to_numeric(self.data['Signal_Type'], errors='coerce')
        
        if 'Entry_Price' in self.data.columns:
            self.data['Entry_Price'] = pd.to_numeric(self.data['Entry_Price'], errors='coerce')
        
        if 'Profit_Loss' in self.data.columns:
            self.data['Profit_Loss'] = pd.to_numeric(self.data['Profit_Loss'], errors='coerce')
        
        # Handle timestamp
        timestamp_columns = ['Timestamp', 'Time', 'Date', 'DateTime']
        for col in timestamp_columns:
            if col in self.data.columns:
                try:
                    self.data[col] = pd.to_datetime(self.data[col])
                    if col != 'Timestamp':
                        self.data['Timestamp'] = self.data[col]
                    break
                except Exception:
                    continue
        
        # Record cleaning history
        self.preprocessing_history.append({
            'step': 'data_cleaning',
            'timestamp': datetime.now(),
            'original_shape': original_shape,
            'final_shape': self.data.shape,
            'cleaning_steps': cleaning_steps
        })
        
        print(f"Data cleaning completed: {original_shape} -> {self.data.shape}")
        return cleaning_steps
    
    def get_data_summary(self):
        """Get comprehensive data summary"""
        if self.data is None:
            return "No data loaded"
        
        summary = {
            'basic_info': {
                'shape': self.data.shape,
                'memory_usage': f"{self.data.memory_usage().sum() / 1024:.2f} KB",
                'columns': list(self.data.columns)
            },
            'data_types': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'numeric_summary': {},
            'categorical_summary': {}
        }
        
        # Numeric columns summary
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            summary['numeric_summary'][col] = {
                'mean': self.data[col].mean(),
                'std': self.data[col].std(),
                'min': self.data[col].min(),
                'max': self.data[col].max(),
                'median': self.data[col].median()
            }
        
        # Categorical columns summary
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            summary['categorical_summary'][col] = {
                'unique_values': self.data[col].nunique(),
                'top_values': self.data[col].value_counts().head(5).to_dict()
            }
        
        return summary
    
    def export_data(self, file_path, format='csv'):
        """Export data to file"""
        if self.data is None:
            raise ValueError("No data to export")
        
        if format.lower() == 'csv':
            self.data.to_csv(file_path, index=False)
        elif format.lower() == 'excel':
            self.data.to_excel(file_path, index=False)
        elif format.lower() == 'json':
            self.data.to_json(file_path, orient='records', indent=2)
        elif format.lower() == 'parquet':
            self.data.to_parquet(file_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Data exported to {file_path}")
    
    def get_data(self):
        """Get the current data"""
        return self.data
    
    def has_data(self):
        """Check if data is loaded"""
        return self.data is not None
    
    def get_column_info(self):
        """Get detailed column information"""
        if self.data is None:
            return None
        
        column_info = []
        
        for col in self.data.columns:
            info = {
                'name': col,
                'type': str(self.data[col].dtype),
                'non_null_count': self.data[col].notna().sum(),
                'null_count': self.data[col].isnull().sum(),
                'unique_values': self.data[col].nunique(),
                'sample_values': self.data[col].head(3).tolist()
            }
            
            if pd.api.types.is_numeric_dtype(self.data[col]):
                info.update({
                    'min': self.data[col].min(),
                    'max': self.data[col].max(),
                    'mean': self.data[col].mean(),
                    'std': self.data[col].std()
                })
            
            column_info.append(info)
        
        return column_info
    
    def filter_data(self, conditions):
        """Filter data based on conditions"""
        if self.data is None:
            raise ValueError("No data loaded")
        
        filtered_data = self.data.copy()
        
        for column, operator, value in conditions:
            if column not in filtered_data.columns:
                continue
            
            if operator == '==':
                filtered_data = filtered_data[filtered_data[column] == value]
            elif operator == '!=':
                filtered_data = filtered_data[filtered_data[column] != value]
            elif operator == '>':
                filtered_data = filtered_data[filtered_data[column] > value]
            elif operator == '<':
                filtered_data = filtered_data[filtered_data[column] < value]
            elif operator == '>=':
                filtered_data = filtered_data[filtered_data[column] >= value]
            elif operator == '<=':
                filtered_data = filtered_data[filtered_data[column] <= value]
            elif operator == 'contains':
                filtered_data = filtered_data[filtered_data[column].str.contains(value, na=False)]
        
        return filtered_data
    
    def get_date_range(self):
        """Get date range of the data"""
        if self.data is None:
            return None
        
        timestamp_columns = ['Timestamp', 'Time', 'Date', 'DateTime']
        
        for col in timestamp_columns:
            if col in self.data.columns:
                try:
                    dates = pd.to_datetime(self.data[col])
                    return {
                        'start_date': dates.min(),
                        'end_date': dates.max(),
                        'duration': dates.max() - dates.min(),
                        'column': col
                    }
                except Exception:
                    continue
        
        return None
    
    def sample_data(self, n=1000, method='random'):
        """Sample data for testing"""
        if self.data is None:
            raise ValueError("No data loaded")
        
        if method == 'random':
            return self.data.sample(n=min(n, len(self.data)))
        elif method == 'head':
            return self.data.head(n)
        elif method == 'tail':
            return self.data.tail(n)
        else:
            raise ValueError(f"Unknown sampling method: {method}")


# Example usage and testing
if __name__ == "__main__":
    # Create data manager
    dm = DataManager()
    
    # Example usage
    print("DataManager class ready for use!")
    print("\nExample usage:")
    print("1. Load CSV: dm.load_csv('data.csv')")
    print("2. Validate data: dm.validate_data()")
    print("3. Clean data: dm.clean_data()")
    print("4. Get summary: dm.get_data_summary()")
    print("5. Export data: dm.export_data('output.csv')")
