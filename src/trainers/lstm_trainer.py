'''
================================================================================
LSTM TRAINER - DEEP LEARNING FOR FINANCIAL TIME SERIES
================================================================================

OVERVIEW:
This module implements Long Short-Term Memory (LSTM) neural networks for
financial time series forecasting. It handles sequential pattern recognition
in trading data and generates deep learning models for signal classification.

ROLE IN CODEBASE:
- Provides deep learning capabilities for the ML trading system
- Handles sequential/temporal patterns in financial data
- Implements TensorFlow/Keras-based LSTM architecture
- Generates neural network models for MQH header conversion
- Offers advanced pattern recognition beyond traditional ML algorithms

LSTM ARCHITECTURE:
- Dual-layer LSTM network with batch normalisation
- Dropout layers for regularisation and overfitting prevention
- Dense layers for final classification decisions
- Binary sigmoid output for buy/sell signal classification
- Adam optimiser with learning rate scheduling

SEQUENCE PROCESSING:
- Converts tabular data into sequential format for LSTM consumption
- Configurable sequence length (default: adaptive based on data size)
- Handles variable-length sequences gracefully
- Uses last timestep approach for feature extraction in MQH generation

DATA HANDLING FEATURES:
- Robust data cleaning for float32 overflow prevention
- NaN and infinite value handling with replacement strategies  
- Scaling using RobustScaler for outlier resilience
- Comprehensive data validation before training begins

DEEP LEARNING SPECIFICS:
- TensorFlow/Keras backend with GPU acceleration support
- Early stopping and learning rate reduction callbacks
- Validation split for training monitoring
- Model checkpointing for best weight preservation

INTEGRATION POINTS:
- Inherits from BaseTrainer for consistency with other ML algorithms
- Compatible with Data_Exporter2.0 24-feature format
- Generates models suitable for MQH header conversion
- Provides confidence scoring for trading signal evaluation
================================================================================
'''

from datetime import datetime
import pandas as pd
import numpy as np
import warnings
# Suppress specific TensorFlow and Keras warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

import sys
import os
import joblib

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# TensorFlow/Keras imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    TF_AVAILABLE = True
    
    # Set TensorFlow to use less memory
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. LSTM trainer will not work.")

try:
    from src.trainers.base_trainer import BaseTrainer
    from src.utils.logging_utils import get_logger
    from src.data_processing.feature_engineering import FeatureEngineer
except ImportError:
    # Fallback if imports fail
    class BaseTrainer:
        def __init__(self, algorithm="LSTM", test_size=0.2, random_state=42):
            self.algorithm = algorithm
            self.test_size = test_size
            self.random_state = random_state
            self.model = None
            self.scaler = None
            self.feature_columns = None
            self.performance_metrics = {}
            self.training_history = []
        
        def validate_input_data(self, df):
            return True
        
        def preprocess_data(self, df):
            return df
        
        def create_time_based_splits(self, df):
            split_point = int(len(df) * (1 - self.test_size))
            return df.iloc[:split_point], df.iloc[split_point:]
        
        def record_training_session(self, train_df, test_df):
            pass
    
    import logging
    def get_logger(name):
        return logging.getLogger(name)

    class FeatureEngineer:
        def validate_data_exporter_2_features(self, df):
            return True, []

        def engineer_all_features(self, df):
            return df

        def get_feature_columns(self):
            return []

logger = get_logger('lstm_trainer')

class LSTMTrainer(BaseTrainer):
    """LSTM trainer for financial time series forecasting"""
    
    def __init__(self, csv_file=None, **params):
        super().__init__(algorithm="LSTM")
        
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM trainer. Please install: pip install tensorflow")
        
        self.csv_file = csv_file
        self.df = None
        
        # LSTM-specific parameters
        self.lstm_params = {
            'sequence_length': params.get('sequence_length', 5),  # Reduced from 10 to 5
            'hidden_units': params.get('hidden_units', 50),
            'dropout': params.get('dropout', 0.2),
            'epochs': params.get('epochs', 50),
            'batch_size': params.get('batch_size', 32),
            'learning_rate': params.get('learning_rate', 0.001),
            'validation_split': params.get('validation_split', 0.2),
            'patience': params.get('patience', 10),
            'min_delta': params.get('min_delta', 0.001)
        }
        
        # LSTM-specific attributes
        self.sequence_length = self.lstm_params['sequence_length']
        self.keras_model = None
        self.training_history_keras = None
        self.feature_scaler = None
        self.target_scaler = None
        
        # For compatibility with rule-based system
        self.X_test = None
        self.y_test = None
        self.y_pred_proba = None
        
        logger.info(f"LSTM Trainer initialised with sequence length: {self.sequence_length}")
    
    def prepare_data(self, df, progress_callback=None):
        """Prepare data for LSTM training - required by base class"""
        if progress_callback:
            progress_callback(10, "Preparing LSTM data...")
        
        # Use the data passed in
        self.df = df
        return self.preprocess_data(df)
    
    def load_and_validate_data(self, progress_callback=None):
        """Load and validate the CSV data"""
        if progress_callback:
            progress_callback(5, "Loading data...")
        
        if self.csv_file:
            self.df = pd.read_csv(self.csv_file)
        
        if self.df is None or self.df.empty:
            raise ValueError("No data loaded. Please provide a CSV file or set df directly.")
        
        # Validate data format
        if not self.validate_input_data(self.df):
            raise ValueError("Data validation failed")
        
        if progress_callback:
            progress_callback(10, f"Loaded {len(self.df)} rows of data")
        
        return True
    
    def engineer_features(self, progress_callback=None):
        """Create features for LSTM model using Data_Exporter2.0 format"""
        if progress_callback:
            progress_callback(15, "Engineering features for LSTM...")
        
        # Use the FeatureEngineer class
        feature_engineer = FeatureEngineer()
        
        # Validate Data_Exporter2.0 features
        is_valid, missing = feature_engineer.validate_data_exporter_2_features(self.df)
        if not is_valid:
            if progress_callback:
                progress_callback(20, f"Missing features: {missing[:3]}...")
            logger.warning(f"Missing Data_Exporter2.0 features: {missing}")
        
        # Create base + derived features via the shared pipeline.
        # NOTE: the previous implementation hand-listed feature columns that
        # included 'Profit_Loss' and 'Entry_Price'. Profit_Loss is the source
        # of the Signal_Quality label, so training on it was direct target
        # leakage (the model could read the answer). It also called a method
        # (create_features_data_exporter_2_format) that does not exist on the
        # real FeatureEngineer. Both are fixed by using the same
        # engineer_all_features()/get_feature_columns() path as every other
        # trainer.
        if progress_callback:
            progress_callback(25, "Creating engineered features...")

        self.df = feature_engineer.engineer_all_features(self.df)
        self.feature_columns = feature_engineer.get_feature_columns()
        
        if progress_callback:
            progress_callback(30, f"Created {len(self.feature_columns)} features for LSTM")
        
        return True
    
    def create_sequences(self, X, y, sequence_length):
        """Create sequences for LSTM training"""
        logger.info(f"Creating sequences with length {sequence_length} from data shape {X.shape}")
        logger.info(f"Input data types - X: {type(X)}, y: {type(y)}")
        logger.info(f"Input data shapes - X: {X.shape}, y: {y.shape if hasattr(y, 'shape') else len(y)}")
        
        if len(X) < sequence_length:
            logger.error(f"Cannot create sequences: data length {len(X)} < sequence length {sequence_length}")
            return np.array([]), np.array([])
        
        # Ensure y is a numpy array with sequential indexing
        if not isinstance(y, np.ndarray):
            y = np.array(y)
            logger.info(f"Converted y to numpy array: shape {y.shape}")
        
        X_sequences = []
        y_sequences = []
        
        max_index = len(X) - sequence_length + 1
        logger.info(f"Will create {max_index} sequences")
        
        for i in range(max_index):
            try:
                X_sequences.append(X[i:(i + sequence_length)])
                target_index = i + sequence_length - 1
                y_sequences.append(y[target_index])  # Use the last value in sequence
            except (IndexError, KeyError) as e:
                logger.error(f"Error creating sequence {i}: {e}")
                logger.error(f"Trying to access y[{target_index}] from y with shape {y.shape}")
                raise
        
        if len(X_sequences) == 0:
            logger.error("No sequences could be created")
            return np.array([]), np.array([])
        
        try:
            X_sequences = np.array(X_sequences)
            y_sequences = np.array(y_sequences)
            logger.info(f"Created {len(X_sequences)} sequences of shape {X_sequences.shape}")
            return X_sequences, y_sequences
        except Exception as e:
            logger.error(f"Error converting sequences to numpy arrays: {e}")
            return np.array([]), np.array([])
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model architecture"""
        logger.info(f"Building LSTM model with input shape: {input_shape}")
        
        model = Sequential([
            # First LSTM layer with return sequences
            LSTM(
                self.lstm_params['hidden_units'], 
                return_sequences=True, 
                input_shape=input_shape,
                dropout=self.lstm_params['dropout'],
                recurrent_dropout=self.lstm_params['dropout']
            ),
            
            # Batch normalization for stability
            BatchNormalization(),
            
            # Second LSTM layer
            LSTM(
                self.lstm_params['hidden_units'] // 2, 
                return_sequences=False,
                dropout=self.lstm_params['dropout'],
                recurrent_dropout=self.lstm_params['dropout']
            ),
            
            # Batch normalization
            BatchNormalization(),
            
            # Dropout for regularization
            Dropout(self.lstm_params['dropout']),
            
            # Dense layers for classification
            Dense(32, activation='relu'),
            Dropout(self.lstm_params['dropout'] / 2),
            Dense(16, activation='relu'),
            
            # Output layer for binary classification
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        optimizer = Adam(learning_rate=self.lstm_params['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        logger.info("LSTM model compiled successfully")
        logger.info(f"Model summary: {model.count_params()} parameters")
        
        return model
    
    def clean_data_for_float32(self, data):
        """Clean data to prevent float32 overflow errors"""
        data = np.array(data, dtype=np.float64)  # Use float64 for cleaning
        
        # Check for infinite values
        inf_mask = np.isinf(data)
        if np.any(inf_mask):
            inf_count = np.sum(inf_mask)
            logger.warning(f"Found {inf_count} infinite values, replacing with finite values")
            
            # Replace positive infinity with large finite value
            data[data == np.inf] = np.finfo(np.float32).max / 100
            # Replace negative infinity with small finite value
            data[data == -np.inf] = np.finfo(np.float32).min / 100
        
        # Check for NaN values
        nan_mask = np.isnan(data)
        if np.any(nan_mask):
            nan_count = np.sum(nan_mask)
            logger.warning(f"Found {nan_count} NaN values, replacing with zeros")
            data[nan_mask] = 0.0
        
        # Clip values to safe float32 range
        float32_max = np.finfo(np.float32).max / 1000  # Safety margin
        float32_min = np.finfo(np.float32).min / 1000  # Safety margin
        
        # Check if any values exceed safe range
        extreme_mask = (np.abs(data) > float32_max)
        if np.any(extreme_mask):
            extreme_count = np.sum(extreme_mask)
            logger.warning(f"Found {extreme_count} extreme values, clipping to safe range")
            data = np.clip(data, float32_min, float32_max)
        
        # Final validation
        if np.any(~np.isfinite(data)):
            logger.error("Data still contains non-finite values after cleaning")
            # Last resort: replace any remaining non-finite values with zeros
            data[~np.isfinite(data)] = 0.0
        
        logger.info(f"Data cleaned: min={np.min(data):.6f}, max={np.max(data):.6f}, "
                   f"mean={np.mean(data):.6f}, std={np.std(data):.6f}")
        
        return data
    
    def validate_data_integrity(self, data, name):
        """Validate data integrity for LSTM training"""
        # Check for non-finite values
        if not np.all(np.isfinite(data)):
            non_finite_count = np.sum(~np.isfinite(data))
            raise ValueError(f"{name} contains {non_finite_count} non-finite values (inf/nan)")
        
        # Check data range
        data_min, data_max = np.min(data), np.max(data)
        float32_max = np.finfo(np.float32).max / 10  # Conservative limit
        
        if data_max > float32_max or data_min < -float32_max:
            raise ValueError(f"{name} values exceed safe float32 range: min={data_min}, max={data_max}")
        
        logger.info(f"{name} validation passed: shape={data.shape}, range=[{data_min:.6f}, {data_max:.6f}]")
    
    def train_model(self, X_train, y_train, X_test, y_test, progress_callback=None):
        """Train the LSTM model"""
        logger.info("Starting LSTM model training")
        
        if progress_callback:
            progress_callback(35, "Preparing LSTM training data...")
        
        # Use the provided training data (already split)
        # Convert to numpy arrays with data validation
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        
        # Clean data: handle infinite and extreme values
        X_train = self.clean_data_for_float32(X_train)
        X_test = self.clean_data_for_float32(X_test)
        y_train = self.clean_data_for_float32(y_train)
        y_test = self.clean_data_for_float32(y_test)
        
        # Convert to float32 after cleaning
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)
        
        logger.info(f"Converted to numpy arrays - X_train: {X_train.shape}, y_train: {y_train.shape}")
        
        # Validate data before scaling
        self.validate_data_integrity(X_train, "X_train")
        self.validate_data_integrity(X_test, "X_test")
        self.validate_data_integrity(y_train, "y_train")
        self.validate_data_integrity(y_test, "y_test")
        
        # Scale features
        self.feature_scaler = RobustScaler()
        try:
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_test_scaled = self.feature_scaler.transform(X_test)
        except Exception as e:
            logger.error(f"Error during scaling: {e}")
            logger.error(f"X_train stats: min={np.min(X_train)}, max={np.max(X_train)}, "
                        f"mean={np.mean(X_train)}, std={np.std(X_train)}")
            raise ValueError(f"Scaling failed: {e}")
        
        if progress_callback:
            progress_callback(40, "Creating LSTM sequences...")
        
        # Create sequences for LSTM
        logger.info(f"Training data shape: {X_train_scaled.shape}, Test data shape: {X_test_scaled.shape}")
        logger.info(f"Required sequence length: {self.sequence_length}")
        
        if len(X_train_scaled) < self.sequence_length:
            raise ValueError(f"Not enough training data for sequence length {self.sequence_length}. Need at least {self.sequence_length} samples, but got {len(X_train_scaled)}")
        
        # Adapt sequence length based on available data
        min_data_size = min(len(X_train_scaled), len(X_test_scaled))
        logger.info(f"Data sizes - Train: {len(X_train_scaled)}, Test: {len(X_test_scaled)}, Min: {min_data_size}")
        
        if min_data_size < self.sequence_length:
            # For very small datasets, we need at least 3 samples to create 1 sequence of length 2
            if min_data_size < 3:
                raise ValueError(f"Dataset too small for LSTM. Need at least 3 samples, got {min_data_size}. "
                               f"Train: {len(X_train_scaled)}, Test: {len(X_test_scaled)}")
            
            # Calculate the maximum possible sequence length
            max_possible_seq_length = min_data_size - 1
            effective_sequence_length = max(2, max_possible_seq_length)
            logger.warning(f"Adapting sequence length from {self.sequence_length} to {effective_sequence_length} due to small dataset (size: {min_data_size})")
        else:
            effective_sequence_length = self.sequence_length
            
        logger.info(f"Using effective sequence length: {effective_sequence_length}")
        
        X_train_seq, y_train_seq = self.create_sequences(X_train_scaled, y_train, effective_sequence_length)
        X_test_seq, y_test_seq = self.create_sequences(X_test_scaled, y_test, effective_sequence_length)
        
        logger.info(f"Created sequences - Train: {X_train_seq.shape if len(X_train_seq) > 0 else 'empty'}, Test: {X_test_seq.shape if len(X_test_seq) > 0 else 'empty'}")
        
        # Check if we have valid sequences
        if len(X_train_seq) == 0:
            raise ValueError("Failed to create training sequences. Check your data size and sequence length.")
        if len(X_test_seq) == 0:
            raise ValueError("Failed to create test sequences. Check your data size and sequence length.")
        
        # Validate sequence dimensions
        if X_train_seq.ndim != 3:
            raise ValueError(f"Training sequences have wrong dimensions: expected 3D, got {X_train_seq.ndim}D with shape {X_train_seq.shape}")
        if X_test_seq.ndim != 3:
            raise ValueError(f"Test sequences have wrong dimensions: expected 3D, got {X_test_seq.ndim}D with shape {X_test_seq.shape}")
        
        if progress_callback:
            progress_callback(50, "Building LSTM model...")
        
        # Build model
        try:
            logger.info(f"X_train_seq details: shape={X_train_seq.shape}, ndim={X_train_seq.ndim}, dtype={X_train_seq.dtype}")
            logger.info(f"X_test_seq details: shape={X_test_seq.shape}, ndim={X_test_seq.ndim}, dtype={X_test_seq.dtype}")
            
            if X_train_seq.shape[0] == 0:
                raise ValueError("Training sequences array is empty")
            if X_train_seq.shape[1] == 0:
                raise ValueError("Sequence length is 0")
            if X_train_seq.shape[2] == 0:
                raise ValueError("Feature count is 0")
            
            input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
            logger.info(f"Input shape for LSTM model: {input_shape}")
            
            self.keras_model = self.build_lstm_model(input_shape)
            logger.info("LSTM model built successfully")
        except Exception as e:
            logger.error(f"Error building LSTM model: {e}")
            logger.error(f"X_train_seq shape: {X_train_seq.shape}")
            logger.error(f"Sequence creation details - effective_sequence_length: {effective_sequence_length}")
            logger.error(f"Original data shapes - X_train: {X_train_scaled.shape}, X_test: {X_test_scaled.shape}")
            raise ValueError(f"Failed to build LSTM model. Error: {e}")
        
        if progress_callback:
            progress_callback(60, "Training LSTM model...")
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.lstm_params['patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.lstm_params['patience'] // 2,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        self.training_history_keras = self.keras_model.fit(
            X_train_seq, y_train_seq,
            epochs=self.lstm_params['epochs'],
            batch_size=self.lstm_params['batch_size'],
            validation_split=self.lstm_params['validation_split'],
            callbacks=callbacks,
            verbose=1
        )
        
        if progress_callback:
            progress_callback(80, "Evaluating LSTM model...")
        
        # Make predictions
        y_pred_proba_train = self.keras_model.predict(X_train_seq)
        y_pred_proba_test = self.keras_model.predict(X_test_seq)
        
        # Convert probabilities to binary predictions
        y_pred_train = (y_pred_proba_train > 0.5).astype(int).flatten()
        y_pred_test = (y_pred_proba_test > 0.5).astype(int).flatten()
        
        # Store for rule-based system compatibility
        self.X_test = X_test_seq
        self.y_test = y_test_seq
        self.y_pred_proba = y_pred_proba_test.flatten()
        
        # Calculate performance metrics
        train_accuracy = accuracy_score(y_train_seq, y_pred_train)
        test_accuracy = accuracy_score(y_test_seq, y_pred_test)
        
        # Handle edge cases for precision, recall, f1
        try:
            train_precision = precision_score(y_train_seq, y_pred_train, zero_division=0)
            test_precision = precision_score(y_test_seq, y_pred_test, zero_division=0)
            train_recall = recall_score(y_train_seq, y_pred_train, zero_division=0)
            test_recall = recall_score(y_test_seq, y_pred_test, zero_division=0)
            train_f1 = f1_score(y_train_seq, y_pred_train, zero_division=0)
            test_f1 = f1_score(y_test_seq, y_pred_test, zero_division=0)
            
            # ROC AUC
            train_roc_auc = roc_auc_score(y_train_seq, y_pred_proba_train.flatten())
            test_roc_auc = roc_auc_score(y_test_seq, y_pred_proba_test.flatten())
            
        except Exception as e:
            logger.warning(f"Error calculating some metrics: {e}")
            train_precision = test_precision = 0.0
            train_recall = test_recall = 0.0
            train_f1 = test_f1 = 0.0
            train_roc_auc = test_roc_auc = 0.5
        
        # Store performance metrics
        self.performance_metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_precision': train_precision,
            'test_precision': test_precision,
            'train_recall': train_recall,
            'test_recall': test_recall,
            'train_f1': train_f1,
            'test_f1': test_f1,
            'train_roc_auc': train_roc_auc,
            'test_roc_auc': test_roc_auc,
            'sequence_length': self.sequence_length,
            'total_sequences_train': len(X_train_seq),
            'total_sequences_test': len(X_test_seq),
            'epochs_trained': len(self.training_history_keras.history['loss'])
        }
        
        # Set model for compatibility
        self.model = self.keras_model  # For save/load compatibility
        self.scaler = self.feature_scaler  # For save/load compatibility
        
        if progress_callback:
            progress_callback(95, f"LSTM training complete! Test accuracy: {test_accuracy:.3f}")
        
        # Record training session (using dummy dataframes for compatibility)
        train_df_dummy = pd.DataFrame({'dummy': range(len(X_train))})
        test_df_dummy = pd.DataFrame({'dummy': range(len(X_test))})
        self.record_training_session(train_df_dummy, test_df_dummy)
        
        if progress_callback:
            progress_callback(100, "LSTM model training finished!")
        
        logger.info(f"LSTM training completed. Test accuracy: {test_accuracy:.3f}")
        return X_test_seq, y_test_seq, y_pred_proba_test.flatten()
    
    def analyse_confidence_thresholds(self, X_test, y_test, y_pred_proba):
        """Analyse confidence thresholds for LSTM model"""
        logger.info("Analysing LSTM confidence thresholds")
        
        if len(y_pred_proba) == 0:
            logger.warning("No predictions available for threshold analysis")
            return {}
        
        thresholds = np.arange(0.1, 1.0, 0.05)
        threshold_analysis = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            
            # Calculate metrics
            try:
                accuracy = accuracy_score(y_test, y_pred_thresh)
                precision = precision_score(y_test, y_pred_thresh, zero_division=0)
                recall = recall_score(y_test, y_pred_thresh, zero_division=0)
                f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
                
                # Count predictions
                positive_predictions = np.sum(y_pred_thresh)
                
                threshold_analysis.append({
                    'threshold': threshold,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'positive_predictions': positive_predictions
                })
                
            except Exception as e:
                logger.warning(f"Error calculating metrics for threshold {threshold}: {e}")
                continue
        
        return threshold_analysis
    
    def save_model(self, model_path=None):
        """Save trained LSTM model"""
        if self.keras_model is None:
            logger.error("No trained model to save")
            return False
        
        if model_path is None:
            os.makedirs('models', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = f'models/lstm_model_{timestamp}.pkl'
        
        try:
            # LSTM doesn't have feature importance, so set to None for MQH generation
            feature_importance = None
            
            # Extract optimal threshold (default for LSTM)
            optimal_threshold = 0.6
            if 'threshold_analysis' in self.performance_metrics:
                optimal_threshold = self.performance_metrics['threshold_analysis'].get('optimal_f1_threshold', 0.6)
            
            # Prepare model data
            model_data = {
                'keras_model': self.keras_model,
                'feature_scaler': self.feature_scaler,
                'feature_columns': self.feature_columns,
                'algorithm': self.algorithm,
                'performance_metrics': self.performance_metrics,
                'lstm_params': self.lstm_params,
                'training_history': self.training_history_keras.history if self.training_history_keras else {},
                'trained_at': datetime.now().isoformat(),
                'sequence_length': self.sequence_length,
                # Include test data for rule-based system compatibility
                'X_test': self.X_test,
                'y_test': self.y_test,
                'y_pred_proba': self.y_pred_proba,
                'test_size': self.test_size,
                
                # MQH generation specific data
                'model': self.keras_model,  # For MQH compatibility
                'feature_importance': feature_importance,  # None for LSTM
                'optimal_threshold': optimal_threshold,
                'accuracy': self.performance_metrics.get('test_accuracy', 0.0),
                'precision': self.performance_metrics.get('test_precision', 0.0),
                'recall': self.performance_metrics.get('test_recall', 0.0),
                'f1_score': self.performance_metrics.get('test_f1', 0.0)
            }
            
            # Save with joblib
            joblib.dump(model_data, model_path)
            
            logger.info(f"LSTM model saved successfully to: {model_path} with MQH generation data")
            print(f"LSTM model saved successfully to: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving LSTM model: {e}")
            return False
    
    @staticmethod
    def load_model(model_path):
        """Load trained LSTM model"""
        try:
            model_data = joblib.load(model_path)
            
            # Create trainer instance
            trainer = LSTMTrainer()
            trainer.keras_model = model_data['keras_model']
            trainer.feature_scaler = model_data['feature_scaler']
            trainer.feature_columns = model_data['feature_columns']
            trainer.performance_metrics = model_data['performance_metrics']
            trainer.lstm_params = model_data.get('lstm_params', {})
            trainer.sequence_length = model_data.get('sequence_length', 10)
            
            # Set for compatibility
            trainer.model = trainer.keras_model
            trainer.scaler = trainer.feature_scaler
            
            logger.info(f"LSTM model loaded successfully from: {model_path}")
            return trainer
            
        except Exception as e:
            logger.error(f"Error loading LSTM model: {e}")
            return None
    
    def predict(self, X):
        """Make predictions using the trained LSTM model"""
        if self.keras_model is None:
            raise ValueError("No trained model available")
        
        # Scale features
        if self.feature_scaler is None:
            raise ValueError("No feature scaler available")
        
        X_scaled = self.feature_scaler.transform(X)
        
        # Create sequences
        if len(X_scaled) < self.sequence_length:
            raise ValueError(f"Not enough data for sequence prediction. Need at least {self.sequence_length} samples.")
        
        X_sequences = []
        for i in range(len(X_scaled) - self.sequence_length + 1):
            X_sequences.append(X_scaled[i:(i + self.sequence_length)])
        
        X_sequences = np.array(X_sequences)
        
        # Make predictions
        predictions = self.keras_model.predict(X_sequences)
        return predictions.flatten()
    
    def get_model_info(self):
        """Get information about the trained LSTM model"""
        if self.keras_model is None:
            return "No trained LSTM model available"
        
        info = f"""
LSTM Model Information:
======================
Algorithm: {self.algorithm}
Sequence Length: {self.sequence_length}
Hidden Units: {self.lstm_params['hidden_units']}
Dropout: {self.lstm_params['dropout']}
Total Parameters: {self.keras_model.count_params():,}
Features: {len(self.feature_columns)}

Performance Metrics:
===================
Test Accuracy: {self.performance_metrics.get('test_accuracy', 0):.4f}
Test Precision: {self.performance_metrics.get('test_precision', 0):.4f}
Test Recall: {self.performance_metrics.get('test_recall', 0):.4f}
Test F1-Score: {self.performance_metrics.get('test_f1', 0):.4f}
Test ROC-AUC: {self.performance_metrics.get('test_roc_auc', 0):.4f}

Training Info:
==============
Epochs Trained: {self.performance_metrics.get('epochs_trained', 0)}
Training Sequences: {self.performance_metrics.get('total_sequences_train', 0):,}
Test Sequences: {self.performance_metrics.get('total_sequences_test', 0):,}
        """
        return info

# Example usage and testing
if __name__ == "__main__":
    print("LSTM Trainer for Financial Forecasting")
    print("=====================================")
    print(f"TensorFlow Available: {TF_AVAILABLE}")
    
    if TF_AVAILABLE:
        # Test with sample parameters
        lstm_params = {
            'sequence_length': 10,
            'hidden_units': 50,
            'dropout': 0.2,
            'epochs': 5,  # Small for testing
            'batch_size': 32,
            'learning_rate': 0.001
        }
        
        trainer = LSTMTrainer(**lstm_params)
        print("LSTM Trainer initialised successfully!")
        print(f"Sequence Length: {trainer.sequence_length}")
        print(f"Parameters: {trainer.lstm_params}")
    else:
        print("TensorFlow not available. Install with: pip install tensorflow")