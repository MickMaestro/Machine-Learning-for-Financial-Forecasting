"""
Fixed imports for src/trainers/base_trainer.py
"""
from datetime import datetime
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import joblib
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

try:
    from src.utils.logging_utils import get_logger
    logger = get_logger('base_trainer')
except ImportError:
    import logging
    logger = logging.getLogger('base_trainer')

try:
    from src.data_processing.feature_engineering import FeatureEngineer, create_signal_labels
    from src.data_processing.preprocessor import DataPreprocessor
    IMPORTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Failed to import required modules: {e}")
    IMPORTS_AVAILABLE = False
    
    # Create minimal fallback classes
    class FeatureEngineer:
        def __init__(self):
            self.feature_columns = []
        
        def engineer_all_features(self, df):
            return df
        
        def get_feature_columns(self):
            return self.feature_columns
    
    class DataPreprocessor:
        def validate_data(self, df):
            return True, []
        
        def clean_data(self, df):
            return df
        
        def prepare_timestamps(self, df):
            return df
    
    def create_signal_labels(df):
        return df

class BaseTrainer(ABC):
    """Abstract base class for all trainers"""
    
    def __init__(self, algorithm="Random Forest", test_size=0.2, random_state=42):
        self.algorithm = algorithm
        self.test_size = test_size
        self.random_state = random_state
        
        # Core components
        self.model = None
        self.scaler = None
        self.feature_engineer = FeatureEngineer()
        self.preprocessor = DataPreprocessor()
        
        # Training data
        self.feature_columns = None
        self.training_history = []
        
        # Test data for analysis
        self.X_test = None
        self.y_test = None
        self.performance_metrics = {}
        
    @abstractmethod
    def prepare_data(self, df, progress_callback=None):
        """Prepare data for training - must be implemented by subclasses"""
        pass
    
    def create_time_based_splits(self, df):
        """Create time-based train/test splits"""
        # Sort by timestamp if available
        if 'Timestamp' in df.columns:
            df = df.sort_values('Timestamp').reset_index(drop=True)
        
        # Calculate split point
        split_point = int(len(df) * (1 - self.test_size))
        
        # Create chronological splits
        train_df = df.iloc[:split_point].copy()
        test_df = df.iloc[split_point:].copy()
        
        logger.info(f"Time-based split: Train {len(train_df)}, Test {len(test_df)}")
        return train_df, test_df
    
    @abstractmethod
    def train_model(self, X_train, y_train, X_test, y_test, progress_callback=None):
        """Train the model - must be implemented by subclasses"""
        pass
    
    def validate_input_data(self, df):
        """Validate input data"""
        logger.info("Validating input data")
        
        is_valid, issues = self.preprocessor.validate_data(df)
        
        if not is_valid:
            error_msg = "Data validation failed: " + "; ".join(issues)
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("Data validation passed")
        return True
    
    def preprocess_data(self, df):
        """Standard data preprocessing"""
        logger.info("Starting data preprocessing")
        
        # Clean data
        df = self.preprocessor.clean_data(df)
        
        # Prepare timestamps
        df = self.preprocessor.prepare_timestamps(df)
        
        # Create signal labels
        df = create_signal_labels(df)
        
        # Engineer features
        df = self.feature_engineer.engineer_all_features(df)
        self.feature_columns = self.feature_engineer.get_feature_columns()
        
        logger.info("Data preprocessing completed")
        return df
    
    def train(self, df, progress_callback=None):
        """Main training pipeline"""
        logger.info(f"Starting training with {self.algorithm} using time-based split")
        
        if progress_callback:
            progress_callback(0, "Validating data...")
        
        # Validate input data
        self.validate_input_data(df)
        
        if progress_callback:
            progress_callback(10, "Preprocessing data...")
        
        # Prepare data (subclass implementation)
        df = self.prepare_data(df, progress_callback)
        
        if progress_callback:
            progress_callback(40, "Creating time-based train/test splits...")
        
        # Create time-based splits
        train_df, test_df = self.create_time_based_splits(df)
        
        # Prepare features and labels
        X_train = train_df[self.feature_columns].fillna(0)
        y_train = train_df['Signal_Quality']
        X_test = test_df[self.feature_columns].fillna(0)
        y_test = test_df['Signal_Quality']
        
        # Store test data for threshold analysis
        self.X_test = X_test.copy()
        self.y_test = y_test.copy()
        
        if progress_callback:
            progress_callback(50, "Training model...")
        
        # Train model (subclass implementation)
        self.train_model(X_train, y_train, X_test, y_test, progress_callback)
        
        # Record training history
        self.record_training_session(train_df, test_df)
        
        if progress_callback:
            progress_callback(100, "Training completed")
        
        logger.info("Training completed successfully")
        return self
    
    def record_training_session(self, train_df, test_df):
        """Record training session details"""
        session = {
            'timestamp': datetime.now(),
            'algorithm': self.algorithm,
            'training_method': 'Time-Based Split',
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'test_size': self.test_size,
            'performance': self.performance_metrics.copy(),
            'feature_count': len(self.feature_columns) if self.feature_columns else 0
        }
        
        self.training_history.append(session)
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        from sklearn.metrics import (accuracy_score, precision_score, 
                                   recall_score, f1_score, roc_auc_score)
        
        if self.model is None:
            logger.error("No model to evaluate")
            return {}
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            metrics['feature_importance'] = feature_importance
        
        self.performance_metrics.update(metrics)
        
        logger.info(f"Model evaluation completed - F1: {metrics['f1_score']:.3f}, AUC: {metrics['roc_auc']:.3f}")
        return metrics
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained yet")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        return predictions, probabilities
    
    def save_model(self, file_path):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Extract feature importance for MQH generation
        feature_importance = None
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = self.model.feature_importances_
        
        # Extract key metrics for MQH generation
        metrics = self.performance_metrics
        optimal_threshold = 0.6  # Default
        if 'threshold_analysis' in metrics:
            optimal_threshold = metrics['threshold_analysis'].get('optimal_f1_threshold', 0.6)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'algorithm': self.algorithm,
            'training_method': 'Time-Based Split',
            'performance_metrics': self.performance_metrics,
            'training_history': self.training_history,
            'trainer_type': self.__class__.__name__,
            
            # MQH generation specific data
            'feature_importance': feature_importance,
            'optimal_threshold': optimal_threshold,
            'accuracy': metrics.get('accuracy', 0.0),
            'precision': metrics.get('precision', 0.0),
            'recall': metrics.get('recall', 0.0),
            'f1_score': metrics.get('f1_score', 0.0),
            'X_test': self.X_test.values if self.X_test is not None else None,
            'y_test': self.y_test.values if self.y_test is not None else None
        }
        
        joblib.dump(model_data, file_path)
        logger.info(f"Model saved to {file_path} with MQH generation data")
    
    def load_model(self, file_path):
        """Load trained model"""
        model_data = joblib.load(file_path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.algorithm = model_data['algorithm']
        self.performance_metrics = model_data.get('performance_metrics', {})
        self.training_history = model_data.get('training_history', [])
        
        logger.info(f"Model loaded from {file_path}")
        return self
    
    def get_performance_summary(self):
        """Get formatted performance summary"""
        if not self.performance_metrics:
            return "Model not trained yet"
        
        summary = f"""
Training Performance Summary
===========================
Algorithm: {self.algorithm}
Training Method: Time-Based Split
Test Size: {self.test_size:.1%}

Performance Metrics:
- Accuracy: {self.performance_metrics.get('accuracy', 'N/A'):.3f}
- Precision: {self.performance_metrics.get('precision', 'N/A'):.3f}
- Recall: {self.performance_metrics.get('recall', 'N/A'):.3f}
- F1-Score: {self.performance_metrics.get('f1_score', 'N/A'):.3f}
- ROC-AUC: {self.performance_metrics.get('roc_auc', 'N/A'):.3f}
"""
        
        return summary
    
    def get_feature_importance(self, top_n=10):
        """Get top feature importance"""
        if 'feature_importance' not in self.performance_metrics:
            return None
        
        return self.performance_metrics['feature_importance'].head(top_n)
    
    def is_trained(self):
        """Check if model is trained"""
        return self.model is not None and self.scaler is not None
    
    def analyse_confidence_thresholds(self, X_test, y_test, y_pred_proba, progress_callback=None):
        """Analyse performance at different confidence thresholds"""
        if progress_callback:
            progress_callback(100, "Analysing confidence threshold performance...")
        
        # Define threshold range (confidence scores from 0.1 to 0.9)
        thresholds = np.arange(0.1, 0.95, 0.05)
        results = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            
            # Calculate confusion matrix components
            tp = np.sum((y_pred_thresh == 1) & (y_test == 1))
            fp = np.sum((y_pred_thresh == 1) & (y_test == 0))
            tn = np.sum((y_pred_thresh == 0) & (y_test == 0))
            fn = np.sum((y_pred_thresh == 0) & (y_test == 1))
            
            # Calculate performance metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            
            # Calculate signal retention rate (what percentage of signals would be executed)
            signals_retained = np.sum(y_pred_thresh == 1) / len(y_test) if len(y_test) > 0 else 0
            
            results.append({
                'confidence_score': threshold,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'accuracy': accuracy,
                'signal_retention': signals_retained,
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn
            })
        
        results_df = pd.DataFrame(results)
        
        # Find optimal thresholds for different metrics
        optimal_f1_idx = results_df['f1_score'].idxmax()
        optimal_precision_idx = results_df['precision'].idxmax() 
        optimal_recall_idx = results_df['recall'].idxmax()
        
        optimal_thresholds = {
            'f1_score': results_df.loc[optimal_f1_idx, 'confidence_score'],
            'precision': results_df.loc[optimal_precision_idx, 'confidence_score'],
            'recall': results_df.loc[optimal_recall_idx, 'confidence_score']
        }
        
        # Store threshold analysis results
        self.performance_metrics['threshold_analysis'] = {
            'results_df': results_df,
            'optimal_thresholds': optimal_thresholds,
            'optimal_f1_threshold': optimal_thresholds['f1_score']
        }
        
        return optimal_thresholds['f1_score'], results_df
