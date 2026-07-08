"""
Ensemble Trainer - Data_Exporter2.0 Compatible
==================================================
Updated to work with Data_Exporter2.0 pre-calculated features
"""

from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import sys
import os
import warnings
# Import Data_Exporter2.0 compatible feature engineering
from src.data_processing.feature_engineering import FeatureEngineer, create_signal_labels

from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

try:
    from src.trainers.base_trainer import BaseTrainer
except ImportError:
    # Fallback if import fails
    class BaseTrainer:
        def __init__(self, algorithm="Ensemble", test_size=0.2, random_state=42):
            self.algorithm = algorithm
            self.test_size = test_size
            self.random_state = random_state
            self.model = None
            self.scaler = None
            self.feature_columns = None
            self.performance_metrics = {}

class EnsembleTrainer(BaseTrainer):
    """Simplified ensemble trainer combining Random Forest, XGBoost, and LightGBM"""
    
    def __init__(self, csv_file=None, **params):
        super().__init__(algorithm="Ensemble")
        self.csv_file = csv_file
        self.df = None
        
        # Simple ensemble parameters
        self.base_estimators = []
        self.voting_type = 'soft'
        
        # Ensemble parameters for GUI compatibility
        self.ensemble_params = {
            'base_algorithms': ['Random Forest (Variant 1)', 'Random Forest (Variant 2)', 'Random Forest (Variant 3)'],
            'voting': 'soft',
            'ensemble_method': 'voting'
        }
    
    def prepare_data(self, df, progress_callback=None):
        """Prepare data for training - required by base class"""
        if progress_callback:
            progress_callback(10, "Preparing ensemble data...")
        
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
                raise ValueError("Error: CSV file not found")
        
        if progress_callback:
            total_rows = len(self.df)
            progress_callback(10, f"Loaded {total_rows} signals")
        
        # Basic data validation
        required_columns = ['Signal_Type', 'Entry_Price', 'Profit_Loss', 'Gann_Zone_Width_Pips', 'Trading_Session']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return True
    
    def create_labels(self, progress_callback=None):
        """Create binary labels for signal quality"""
        if progress_callback:
            progress_callback(15, "Creating signal quality labels...")
        
        self.df = create_signal_labels(self.df)
        
        # Calculate label statistics safely
        good_signals = self.df['Signal_Quality'].sum()
        total_signals = len(self.df)
        
        if progress_callback and total_signals > 0:
            percentage = good_signals / total_signals
            progress_callback(20, f"Good signals: {good_signals} ({percentage:.1%})")
        
        return True
    
    def engineer_features(self, progress_callback=None):
        """Create features using Data_Exporter2.0 format"""
        if progress_callback:
            progress_callback(25, "Engineering features with Data_Exporter2.0...")
        
        feature_engineer = FeatureEngineer()
        is_valid, missing = feature_engineer.validate_data_exporter_2_features(self.df)
        if not is_valid:
            raise ValueError(f"Data_Exporter2.0 format validation failed. Missing features: {missing}")
        
        self.df = feature_engineer.engineer_all_features(self.df)
        self.feature_columns = feature_engineer.get_feature_columns()
        
        if progress_callback:
            progress_callback(35, f"Ensemble Data_Exporter2.0: {len(self.feature_columns)} total features")
        
        return True
    
    def create_base_estimators(self, progress_callback=None):
        """Create base estimators for ensemble"""
        if progress_callback:
            progress_callback(45, "Creating base estimators...")
        
        # Simple base estimators - using only Random Forest variants to avoid import issues
        estimators = [
            ('rf1', RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                random_state=42,
                class_weight='balanced'
            )),
            ('rf2', RandomForestClassifier(
                n_estimators=150,
                max_depth=10,
                min_samples_split=5,
                random_state=123,
                class_weight='balanced'
            )),
            ('rf3', RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=3,
                random_state=456,
                class_weight='balanced'
            ))
        ]
        
        self.base_estimators = estimators
        return True
    
    def train_model(self, X_train, y_train, X_test, y_test, progress_callback=None):
        """Train the ensemble model"""
        if progress_callback:
            progress_callback(50, "Training ensemble model...")
        
        # Create base estimators
        self.create_base_estimators(progress_callback)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if progress_callback:
            progress_callback(60, "Creating voting ensemble...")
        
        # Create voting classifier
        self.model = VotingClassifier(
            estimators=self.base_estimators,
            voting='soft'
        )
        
        if progress_callback:
            progress_callback(70, "Training base estimators...")
        
        # Fit ensemble model
        self.model.fit(X_train_scaled, y_train)
        
        if progress_callback:
            progress_callback(85, "Evaluating ensemble performance...")
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        self.performance_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'base_estimators': ['Random Forest Variants'],
            'ensemble_method': 'voting',
            'num_base_models': len(self.base_estimators)
        }
        
        # Calculate ensemble feature importance by averaging base estimators
        if hasattr(self.model, 'estimators_') and self.feature_columns:
            try:
                # Get feature importances from all base estimators
                importances_list = []
                for estimator in self.model.estimators_:
                    if hasattr(estimator, 'feature_importances_'):
                        importances_list.append(estimator.feature_importances_)
                
                if importances_list:
                    # Average the feature importances
                    avg_importance = np.mean(importances_list, axis=0)
                    
                    # Create feature importance DataFrame
                    feature_importance = pd.DataFrame({
                        'feature': self.feature_columns,
                        'importance': avg_importance
                    }).sort_values('importance', ascending=False)
                    
                    self.performance_metrics['feature_importance'] = feature_importance
            except Exception as e:
                # Do NOT substitute random importances here: they would flow
                # into the MQH header and silently drive live trade filtering.
                print(f"Warning: Could not calculate ensemble feature importance: {e}")
        
        if progress_callback:
            progress_callback(100, "Ensemble training completed")
        
        return True
    
    def save_model(self, filepath=None):
        """Save the ensemble model"""
        if self.model is None:
            raise ValueError("No trained model to save")
        
        if filepath is None:
            import os
            os.makedirs('models', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f'models/ensemble_model_{timestamp}.pkl'
        
        # Optimal threshold from threshold analysis (default 0.6)
        optimal_threshold = 0.6
        if 'threshold_analysis' in self.performance_metrics:
            optimal_threshold = self.performance_metrics['threshold_analysis'].get('optimal_f1_threshold', 0.6)

        # Averaged base-estimator importances aligned with feature_columns
        feature_importance = None
        if hasattr(self.model, 'estimators_'):
            importances_list = [est.feature_importances_ for est in self.model.estimators_
                                if hasattr(est, 'feature_importances_')]
            if importances_list:
                feature_importance = np.mean(importances_list, axis=0)

        # Create model info
        model_info = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'performance_metrics': self.performance_metrics,
            'algorithm': 'Ensemble',
            'trained_at': datetime.now().isoformat(),
            # Include test data for threshold analysis / MQH approximation
            'X_test': getattr(self, 'X_test', None),
            'y_test': getattr(self, 'y_test', None),
            'y_pred_proba': getattr(self, 'y_pred_proba', None),
            'test_size': getattr(self, 'test_size', 0.2),

            # MQH generation specific data
            'feature_importance': feature_importance,
            'optimal_threshold': optimal_threshold,
            'accuracy': self.performance_metrics.get('accuracy', 0.0),
            'precision': self.performance_metrics.get('precision', 0.0),
            'recall': self.performance_metrics.get('recall', 0.0),
            'f1_score': self.performance_metrics.get('f1_score', 0.0)
        }
        
        # Save model
        joblib.dump(model_info, filepath)
        
        # Save feature columns for reference
        feature_file = filepath.replace('.pkl', '_features.txt')
        with open(feature_file, 'w') as f:
            for feature in self.feature_columns:
                f.write(f"{feature}\n")
        
        return True
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)
        return probabilities
    
    def run_complete_training_with_gui(self, progress_callback=None):
        """Run complete training with GUI integration"""
        try:
            if progress_callback:
                progress_callback(0, "Starting Ensemble training...")
            
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
                train_size = len(X_train)
                test_size = len(X_test)
                progress_callback(42, f"Train: {train_size}, Test: {test_size}")
            
            # Train model
            success = self.train_model(X_train, y_train, X_test, y_test, progress_callback)
            
            if not success:
                return False
            
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
                progress_callback(95, "Analyzing optimal confidence thresholds...")
            
            optimal_threshold, threshold_df = self.analyse_confidence_thresholds(
                X_test, y_test, y_pred_proba, progress_callback
            )
            
            # Step 6: Save model
            if progress_callback:
                progress_callback(100, f"Ensemble training completed! Optimal confidence score: {optimal_threshold:.2f}")
            
            return True
            
        except Exception as e:
            if progress_callback:
                error_msg = str(e)
                progress_callback(100, f"Training failed: {error_msg}")
            return False
    
    def analyse_confidence_thresholds(self, X_test, y_test, y_pred_proba, progress_callback=None):
        """Analyse performance at different confidence thresholds"""
        if progress_callback:
            progress_callback(90, "Analysing confidence threshold performance...")
        
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
        
        # Find optimal thresholds
        optimal_f1_idx = results_df['f1_score'].idxmax()
        optimal_precision_idx = results_df['precision'].idxmax() 
        optimal_recall_idx = results_df['recall'].idxmax()
        
        optimal_thresholds = {
            'f1_score': results_df.loc[optimal_f1_idx, 'confidence_score'],
            'precision': results_df.loc[optimal_precision_idx, 'confidence_score'],
            'recall': results_df.loc[optimal_recall_idx, 'confidence_score']
        }
        
        if not hasattr(self, 'performance_metrics'):
            self.performance_metrics = {}
        self.performance_metrics['threshold_analysis'] = {
            'results_df': results_df,
            'optimal_thresholds': optimal_thresholds,
            'optimal_f1_threshold': optimal_thresholds['f1_score']
        }
        
        if progress_callback:
            progress_callback(95, f"Optimal confidence thresholds - F1: {optimal_thresholds['f1_score']:.2f}")
        
        return optimal_thresholds['f1_score'], results_df
    
    def export_predictions(self, test_df, X_test_scaled, y_test, y_pred_proba, 
                      output_file=None, threshold=0.6):
        """Export predictions to CSV file"""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"data/exports/ensemble_predictions_{timestamp}.csv"
        
        # Ensure exports directory exists
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
        predictions_df['Model_Algorithm'] = 'Ensemble'
        predictions_df['Prediction_Timestamp'] = datetime.now()
        
        # Calculate actual vs predicted
        if 'Signal_Quality' in predictions_df.columns:
            predictions_df['Actual_Quality'] = y_test.values
            predictions_df['Correct_Prediction'] = (predictions_df['ML_Prediction'] == predictions_df['Actual_Quality']).astype(int)
        
        # Export to CSV
        predictions_df.to_csv(output_file, index=False)
        
        total_preds = len(predictions_df)
        execute_count = predictions_df['ML_Prediction'].sum()
        filter_count = total_preds - execute_count
        
        print(f"Predictions exported to: {output_file}")
        print(f"Total predictions: {total_preds}")
        print(f"Recommended to execute: {execute_count}")
        print(f"Recommended to filter: {filter_count}")
        
        return output_file