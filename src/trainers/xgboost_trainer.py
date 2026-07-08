"""
XGBoost Trainer - Data_Exporter2.0 Compatible
==============================================
XGBoost classifier updated to work with Data_Exporter2.0 pre-calculated features
"""
from datetime import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import sys
import os
import warnings
# Suppress specific XGBoost warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='xgboost')
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

# Import Data_Exporter2.0 compatible feature engineering
from src.data_processing.feature_engineering import FeatureEngineer, create_signal_labels

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

try:
    from src.trainers.base_trainer import BaseTrainer
except ImportError:
    # Fallback if import fails
    class BaseTrainer:
        def __init__(self, algorithm="XGBoost", test_size=0.2, random_state=42):
            self.algorithm = algorithm
            self.test_size = test_size
            self.random_state = random_state
            self.model = None
            self.scaler = None
            self.feature_columns = None
            self.performance_metrics = {}

class XGBoostTrainer(BaseTrainer):
    """XGBoost trainer for signal filtering"""
    
    def __init__(self, csv_file=None, **params):
        super().__init__(algorithm="XGBoost")
        self.csv_file = csv_file
        self.df = None
        
        # XGBoost parameters with defaults
        self.xgb_params = {
            'n_estimators': params.get('n_estimators', 200),
            'max_depth': params.get('max_depth', 6),
            'learning_rate': params.get('learning_rate', 0.1),
            'subsample': params.get('subsample', 0.8),
            'colsample_bytree': params.get('colsample_bytree', 0.8),
            'random_state': params.get('random_state', 42),
            'eval_metric': 'logloss',
            'verbosity': 0
        }
    
    def prepare_data(self, df, progress_callback=None):
        """Prepare data for XGBoost training"""
        if progress_callback:
            progress_callback(10, "Preparing XGBoost data...")
        
        # Use the data passed in
        self.df = df
        return self.preprocess_data(df)
    
    def load_and_prepare_data(self, progress_callback=None):
        """Load the signal data"""
        if progress_callback:
            progress_callback(5, "Loading signal data...")
        
        if self.df is None:
            if self.csv_file is None:
                raise ValueError("No data provided")
            try:
                self.df = pd.read_csv(self.csv_file)
            except FileNotFoundError:
                raise ValueError(f"Error: {self.csv_file} not found")
        
        if progress_callback:
            progress_callback(10, f"Loaded {len(self.df)} signals")
        
        # Basic data validation (Data_Exporter2.0 format)
        required_columns = ['Signal_Type', 'Entry_Price', 'Profit_Loss', 'Gann_Zone_Width_Pips', 'Trading_Session']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return True
    
    def create_labels(self, progress_callback=None):
        """Create binary labels for signal quality using Data_Exporter2.0 format"""
        if progress_callback:
            progress_callback(15, "Creating signal quality labels...")
        
        # Use the new Data_Exporter2.0 compatible signal labeling
        self.df = create_signal_labels(self.df)
        
        # Calculate label statistics
        good_signals = self.df['Signal_Quality'].sum()
        total_signals = len(self.df)
        
        if progress_callback:
            progress_callback(20, f"Good signals: {good_signals} ({good_signals/total_signals:.1%})")
        
        return True
    
    def engineer_features(self, progress_callback=None):
        """Create features for XGBoost model using Data_Exporter2.0 format"""
        if progress_callback:
            progress_callback(25, "Engineering features with Data_Exporter2.0...")
        
        # Use the new FeatureEngineer class
        feature_engineer = FeatureEngineer()
        
        # Validate that we have the required Data_Exporter2.0 features
        is_valid, missing = feature_engineer.validate_data_exporter_2_features(self.df)
        if not is_valid:
            raise ValueError(f"Data_Exporter2.0 format validation failed. Missing features: {missing}")
        
        # Engineer all features (base + derived + interaction)
        self.df = feature_engineer.engineer_all_features(self.df)
        
        # Get the feature columns from the feature engineer
        self.feature_columns = feature_engineer.get_feature_columns()
        
        if progress_callback:
            base_count = len(feature_engineer.get_data_exporter_2_features())
            derived_count = len(feature_engineer.get_derived_features())
            progress_callback(35, f"XGBoost Data_Exporter2.0: {base_count} base + {derived_count} derived = {len(self.feature_columns)} total features")
        
        return True
    
    def train_model(self, X_train, y_train, X_test, y_test, progress_callback=None):
        """Train XGBoost model"""
        if progress_callback:
            progress_callback(60, "Training XGBoost model...")
        
        # Scale features (XGBoost can handle raw features but scaling often helps)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost - Updated for newer versions
        self.model = xgb.XGBClassifier(**self.xgb_params)
        
        # Train with proper early stopping for newer XGBoost versions
        try:
            # Try new style first (XGBoost 1.6+)
            self.model.fit(
                X_train_scaled, 
                y_train,
                eval_set=[(X_test_scaled, y_test)],
                callbacks=[xgb.callback.EarlyStopping(rounds=50, save_best=True)]
            )
        except Exception:
            # Fallback to older style or simple training
            try:
                self.model.fit(
                    X_train_scaled, 
                    y_train,
                    eval_set=[(X_test_scaled, y_test)],
                    early_stopping_rounds=50,
                    verbose=False
                )
            except Exception:
                # Simple training without early stopping
                self.model.fit(X_train_scaled, y_train)
        
        if progress_callback:
            progress_callback(75, "Evaluating model performance...")
        
        # Evaluate model
        self.evaluate_model(X_test_scaled, y_test)
        
        if progress_callback:
            progress_callback(85, f"Accuracy: {self.performance_metrics['accuracy']:.3f}")
            progress_callback(90, f"F1 Score: {self.performance_metrics['f1_score']:.3f}")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate XGBoost model performance"""
        if self.model is None:
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
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        metrics['feature_importance'] = feature_importance
        
        self.performance_metrics.update(metrics)
        
        return metrics
    
    def save_model(self, model_path=None):
        """Save trained model"""
        if model_path is None:
            import os
            os.makedirs('models', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = f'models/xgboost_model_{timestamp}.pkl'
        # Optimal threshold from threshold analysis (default 0.6)
        optimal_threshold = 0.6
        if 'threshold_analysis' in self.performance_metrics:
            optimal_threshold = self.performance_metrics['threshold_analysis'].get('optimal_f1_threshold', 0.6)

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'algorithm': 'XGBoost',
            'performance_metrics': self.performance_metrics,
            'xgb_params': self.xgb_params,
            # Include test data for threshold analysis
            'X_test': getattr(self, 'X_test', None),
            'y_test': getattr(self, 'y_test', None),
            'y_pred_proba': getattr(self, 'y_pred_proba', None),
            'csv_file': getattr(self, 'csv_file', None),
            'test_size': getattr(self, 'test_size', 0.2),

            # MQH generation specific data
            'feature_importance': self.model.feature_importances_ if self.model is not None else None,
            'optimal_threshold': optimal_threshold,
            'accuracy': self.performance_metrics.get('accuracy', 0.0),
            'precision': self.performance_metrics.get('precision', 0.0),
            'recall': self.performance_metrics.get('recall', 0.0),
            'f1_score': self.performance_metrics.get('f1_score', 0.0)
        }
        
        joblib.dump(model_data, model_path)
        
        # Save feature columns for reference (use unique filename)
        feature_file = model_path.replace('.pkl', '_features.txt')
        with open(feature_file, 'w') as f:
            for feature in self.feature_columns:
                f.write(f"{feature}\n")
        
        return True
    
    def run_complete_training_with_gui(self, progress_callback=None):
        """Run complete training with GUI integration"""
        try:
            if progress_callback:
                progress_callback(0, f"Starting {self.algorithm} training...")
            
            # Step 1: Load and prepare data
            if not self.load_and_prepare_data(progress_callback):
                return False
            
            # Step 2: Create labels
            if not self.create_labels(progress_callback):
                return False
            
            # Step 3: Engineer features
            if not self.engineer_features(progress_callback):
                return False
            
            # Step 4: Train model using time-based split
            if progress_callback:
                progress_callback(40, "Preparing training data...")
            
            # Prepare features and labels
            X = self.df[self.feature_columns].fillna(0)
            y = self.df['Signal_Quality']
            
            # Time-based split (chronological order)
            split_point = int(len(X) * 0.8)
            X_train = X.iloc[:split_point]
            X_test = X.iloc[split_point:]
            y_train = y.iloc[:split_point]
            y_test = y.iloc[split_point:]
            test_df = self.df.iloc[split_point:]  # Keep original data for export
            
            if progress_callback:
                progress_callback(50, f"Train: {len(X_train)}, Test: {len(X_test)}")
            
            # Train model
            self.train_model(X_train, y_train, X_test, y_test, progress_callback)
            
            # Store test data for confidence analysis
            self.X_test = X_test
            self.y_test = y_test
            
            # Step 5: Export predictions
            if progress_callback:
                progress_callback(92, "Exporting predictions...")
            
            # Get predictions on test set
            y_pred_proba = self.model.predict_proba(self.scaler.transform(X_test))[:, 1]
            
            # Store prediction probabilities for confidence analysis
            self.y_pred_proba = y_pred_proba
            
            # Export predictions to CSV
            self.export_predictions(
                test_df=test_df,
                X_test_scaled=self.scaler.transform(X_test),
                y_test=y_test,
                y_pred_proba=y_pred_proba,
                threshold=0.6
            )
            
            # Step 5.5: Analyse confidence thresholds
            if progress_callback:
                progress_callback(95, "Analysing optimal confidence thresholds...")
            
            optimal_threshold, threshold_df = self.analyse_confidence_thresholds(
                X_test, y_test, y_pred_proba, progress_callback
            )
            
            # Step 6: Save model
            if progress_callback:
                progress_callback(100, f"{self.algorithm} training completed! Optimal confidence score: {optimal_threshold:.2f}")
            
            return True
            
        except Exception as e:
            if progress_callback:
                progress_callback(100, f"Training failed: {str(e)}")
            return False
    
    def analyse_confidence_thresholds(self, X_test, y_test, y_pred_proba, progress_callback=None):
        """Analyse performance at different confidence thresholds"""
        if progress_callback:
            progress_callback(90, "Analysing confidence threshold performance...")
        
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
        if not hasattr(self, 'performance_metrics'):
            self.performance_metrics = {}
        self.performance_metrics['threshold_analysis'] = {
            'results_df': results_df,
            'optimal_thresholds': optimal_thresholds,
            'optimal_f1_threshold': optimal_thresholds['f1_score']
        }
        
        if progress_callback:
            progress_callback(95, f"Optimal confidence thresholds - F1: {optimal_thresholds['f1_score']:.2f}, Precision: {optimal_thresholds['precision']:.2f}, Recall: {optimal_thresholds['recall']:.2f}")
        
        return optimal_thresholds['f1_score'], results_df
    
    def get_performance_summary(self):
        """Get formatted performance summary"""
        if not self.performance_metrics:
            return "Model not trained yet"
        
        summary = f"""
XGBoost Performance Summary
==========================
Algorithm: XGBoost Classifier
Features: {len(self.feature_columns) if self.feature_columns else 'N/A'}

Performance Metrics:
- Accuracy: {self.performance_metrics.get('accuracy', 0):.3f}
- Precision: {self.performance_metrics.get('precision', 0):.3f}
- Recall: {self.performance_metrics.get('recall', 0):.3f}
- F1-Score: {self.performance_metrics.get('f1_score', 0):.3f}
- ROC-AUC: {self.performance_metrics.get('roc_auc', 0):.3f}

Model Parameters:
- n_estimators: {self.xgb_params.get('n_estimators', 200)}
- max_depth: {self.xgb_params.get('max_depth', 6)}
- learning_rate: {self.xgb_params.get('learning_rate', 0.1)}
- subsample: {self.xgb_params.get('subsample', 0.8)}
- colsample_bytree: {self.xgb_params.get('colsample_bytree', 0.8)}

Model saved as: xgboost_model.pkl
"""
        return summary

    def export_predictions(self, test_df, X_test_scaled, y_test, y_pred_proba, 
                      output_file=None, threshold=0.6):
        """Export predictions to CSV file"""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"data/exports/{self.algorithm.lower().replace(' ', '_')}_predictions_{timestamp}.csv"
        
        # Ensure exports directory exists
        import os
        os.makedirs('data/exports', exist_ok=True)
        
        # Create predictions dataframe
        predictions_df = test_df.copy()
        
        # Add model predictions
        predictions_df['ML_Confidence'] = y_pred_proba
        predictions_df['ML_Prediction'] = (y_pred_proba >= threshold).astype(int)
        predictions_df['ML_Recommendation'] = predictions_df['ML_Prediction'].map({
            1: 'EXECUTE', 
            0: 'FILTER'
        })
        predictions_df['Confidence_Threshold'] = threshold
        predictions_df['Model_Algorithm'] = self.algorithm
        predictions_df['Prediction_Timestamp'] = datetime.now()
        
        # Calculate actual vs predicted
        if 'Signal_Quality' in predictions_df.columns:
            predictions_df['Actual_Quality'] = y_test.values
            predictions_df['Correct_Prediction'] = (predictions_df['ML_Prediction'] == predictions_df['Actual_Quality']).astype(int)
        
        # Export to CSV
        predictions_df.to_csv(output_file, index=False)
        
        print(f"Predictions exported to: {output_file}")
        print(f"Total predictions: {len(predictions_df)}")
        print(f"Recommended to execute: {predictions_df['ML_Prediction'].sum()}")
        print(f"Recommended to filter: {(predictions_df['ML_Prediction'] == 0).sum()}")
        
        return output_file
