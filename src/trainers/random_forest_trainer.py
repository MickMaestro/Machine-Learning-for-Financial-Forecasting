"""
Random Forest Trainer with Hyperparameter Optimization - Data_Exporter2.0 Compatible
=====================================================================================
Random Forest classifier with GridSearchCV hyperparameter tuning
Updated to work with Data_Exporter2.0 pre-calculated features
"""
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import joblib
import matplotlib.pyplot as plt
import warnings
# Suppress specific scikit-learn warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Import Data_Exporter2.0 compatible feature engineering
from src.data_processing.feature_engineering import FeatureEngineer, create_signal_labels

class RandomForestTrainer:
    def __init__(self, csv_file='signal_data.csv', test_size=0.2):
        """Initialise the Random Forest trainer"""
        self.csv_file = csv_file
        self.df = None
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.performance_metrics = {}
        self.algorithm = "Random Forest"  # Algorithm name
        self.test_size = test_size  # Store test size for saving
        
    def load_and_prepare_data(self, progress_callback=None):
        """Load the exported signal data and prepare for ML"""
        if progress_callback:
            progress_callback(5, "Loading signal data...")
        
        if self.df is None:
            if self.csv_file is None:
                raise ValueError("No data provided")
            try:
                self.df = pd.read_csv(self.csv_file)
            except FileNotFoundError:
                raise ValueError(f"Error: {self.csv_file} not found. Run the MQL5 exporter first.")
        
        if progress_callback:
            progress_callback(10, f"Loaded {len(self.df)} signals")
            
        # Display basic statistics
        buy_signals = len(self.df[self.df['Signal_Type'] == 1])
        sell_signals = len(self.df[self.df['Signal_Type'] == -1])
        
        if progress_callback:
            progress_callback(15, f"Buy signals: {buy_signals}, Sell signals: {sell_signals}")
        
        return True
    
    def create_labels(self, progress_callback=None):
        """Create binary labels for signal quality"""
        if progress_callback:
            progress_callback(20, "Creating signal quality labels...")
        
        # Use the new Data_Exporter2.0 compatible signal labeling
        self.df = create_signal_labels(self.df)
        
        # Display label statistics
        good_signals = self.df['Signal_Quality'].sum()
        total_signals = len(self.df)
        
        if progress_callback:
            good_percentage = (good_signals/total_signals) if total_signals > 0 else 0
            poor_percentage = ((total_signals - good_signals)/total_signals) if total_signals > 0 else 0
            progress_callback(25, f"Good signals: {good_signals} ({good_percentage:.1%})")
            progress_callback(30, f"Poor signals: {total_signals - good_signals} ({poor_percentage:.1%})")
        
        # Profit statistics by label
        good_signals_data = self.df[self.df['Signal_Quality'] == 1]['Profit_Loss']
        poor_signals_data = self.df[self.df['Signal_Quality'] == 0]['Profit_Loss']
        
        good_avg_profit = good_signals_data.mean() * 10000 if not good_signals_data.empty else 0
        poor_avg_profit = poor_signals_data.mean() * 10000 if not poor_signals_data.empty else 0
        
        if progress_callback:
            progress_callback(35, f"Good signals avg profit: {good_avg_profit:.1f} pips")
            progress_callback(40, f"Poor signals avg profit: {poor_avg_profit:.1f} pips")
        
        return True
    
    def engineer_features(self, progress_callback=None):
        """Create features for ML model using Data_Exporter2.0 format"""
        if progress_callback:
            progress_callback(45, "Engineering features with Data_Exporter2.0...")
        
        # Use the new FeatureEngineer class
        feature_engineer = FeatureEngineer()
        
        # Debug: Print DataFrame info
        if progress_callback:
            progress_callback(42, f"DataFrame shape: {self.df.shape}")
            progress_callback(43, f"DataFrame columns: {list(self.df.columns)}")
        
        # Validate that we have the required Data_Exporter2.0 features
        is_valid, missing = feature_engineer.validate_data_exporter_2_features(self.df)
        if not is_valid:
            if progress_callback:
                progress_callback(44, f"Missing features: {missing}")
            raise ValueError(f"Data_Exporter2.0 format validation failed. Missing features: {missing}")
        
        # Engineer all features (base + derived + interaction)
        self.df = feature_engineer.engineer_all_features(self.df)
        
        # Get the feature columns from the feature engineer
        self.feature_columns = feature_engineer.get_feature_columns()
        
        if progress_callback:
            base_count = len(feature_engineer.get_data_exporter_2_features())
            derived_count = len(feature_engineer.get_derived_features())
            progress_callback(50, f"Data_Exporter2.0: {base_count} base + {derived_count} derived = {len(self.feature_columns)} total features")
            
        # Debug: Check if we have feature columns
        if len(self.feature_columns) == 0:
            if progress_callback:
                progress_callback(50, "ERROR: No feature columns found! Check Data_Exporter2.0 format.")
            raise ValueError("No feature columns found. Check if CSV has Data_Exporter2.0 format.")
        
        return True
    
    def train_model(self, progress_callback=None):
        """Train the Random Forest model with hyperparameter optimization"""
        if progress_callback:
            progress_callback(55, "Training ML model...")
        
        # Prepare features and labels
        X = self.df[self.feature_columns].fillna(0)  # Fill any NaN values
        y = self.df['Signal_Quality']
        
        # Check class distribution - fail if only one class exists
        class_counts = y.value_counts()
        if progress_callback:
            progress_callback(57, f"Class distribution: {dict(class_counts)}")
        
        if len(class_counts) == 1:
            single_class = class_counts.index[0]
            if single_class == 0:
                error_msg = (
                    f"Dataset contains NO profitable signals (all {len(y)} signals are unprofitable). "
                    f"Cannot train classifier. Suggestions:\n"
                    f"• Check EA parameters (stop loss, entry conditions)\n" 
                    f"• Use different time period with better market conditions\n"
                    f"• Review signal generation logic in MQL5 code"
                )
            else:
                error_msg = (
                    f"Dataset contains ONLY profitable signals (all {len(y)} signals are profitable). "
                    f"Cannot train classifier - need both good and bad examples."
                )
            
            if progress_callback:
                progress_callback(58, f"ERROR: {error_msg}")
            raise ValueError(error_msg)
        
        # Chronological split. The previous shuffled train_test_split leaked
        # future information into training and was inconsistent with every
        # other trainer in this project (they all split by time).
        split_point = int(len(X) * (1 - self.test_size))
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            raise ValueError(
                "Chronological split produced a single-class train or test set. "
                "Use a longer data period so both profitable and unprofitable "
                "signals appear in each portion."
            )
        
        if progress_callback:
            progress_callback(60, f"Training set: {len(X_train)} samples")
            progress_callback(65, f"Test set: {len(X_test)} samples")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train initial model
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        rf_model.fit(X_train_scaled, y_train)
        
        # Initial evaluation
        y_pred = rf_model.predict(X_test_scaled)
        y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
        
        initial_accuracy = rf_model.score(X_test_scaled, y_test)
        initial_auc = roc_auc_score(y_test, y_pred_proba)
        
        if progress_callback:
            progress_callback(70, f"Initial accuracy: {initial_accuracy:.3f}")
            progress_callback(75, f"Initial ROC-AUC: {initial_auc:.3f}")
        
        # Hyperparameter tuning
        if progress_callback:
            progress_callback(80, "Optimising hyperparameters...")
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # TimeSeriesSplit keeps validation folds strictly after their training
        # folds; plain KFold shuffles future signals into training.
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
            param_grid,
            cv=TimeSeriesSplit(n_splits=5),
            scoring='f1',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Use best model
        self.model = grid_search.best_estimator_
        
        if progress_callback:
            progress_callback(85, f"Best parameters: {grid_search.best_params_}")
            progress_callback(90, f"Best CV F1 score: {grid_search.best_score_:.3f}")
        
        # Final evaluation
        y_pred_final = self.model.predict(X_test_scaled)
        y_pred_proba_final = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Store performance metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        self.performance_metrics = {
            'accuracy': accuracy_score(y_test, y_pred_final),
            'precision': precision_score(y_test, y_pred_final, zero_division=0),
            'recall': recall_score(y_test, y_pred_final, zero_division=0),
            'f1_score': f1_score(y_test, y_pred_final, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba_final),
            'best_params': grid_search.best_params_,
            'cv_f1_score': grid_search.best_score_
        }
        
        if progress_callback:
            progress_callback(95, f"Final accuracy: {self.performance_metrics['accuracy']:.3f}")
            progress_callback(98, f"Final F1 score: {self.performance_metrics['f1_score']:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.performance_metrics['feature_importance'] = feature_importance
        
        if progress_callback:
            progress_callback(100, "Model training completed!")
        
        # Store test data for threshold analysis
        self.X_test = X_test  # Store original test features
        self.y_test = y_test  # Store test labels
        self.y_pred_proba = y_pred_proba_final  # Store prediction probabilities
        
        print(f"DEBUG: Stored test data - X_test: {self.X_test.shape}, y_test: {len(self.y_test)}, probabilities: {len(self.y_pred_proba)}")
        
        return X_test_scaled, y_test, y_pred_proba_final
    
    def save_model(self, model_path=None):
        """Save trained model"""
        if model_path is None:
            import os
            os.makedirs('models', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = f'models/random_forest_model_{timestamp}.pkl'
        # Optimal threshold from threshold analysis (default 0.6)
        optimal_threshold = 0.6
        if 'threshold_analysis' in self.performance_metrics:
            optimal_threshold = self.performance_metrics['threshold_analysis'].get('optimal_f1_threshold', 0.6)

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'algorithm': self.algorithm,
            'performance_metrics': self.performance_metrics,
            'trained_at': datetime.now().isoformat(),
            # Include test data for threshold analysis
            'X_test': getattr(self, 'X_test', None),
            'y_test': getattr(self, 'y_test', None),
            'y_pred_proba': getattr(self, 'y_pred_proba', None),
            'test_size': self.test_size,

            # MQH generation specific data
            'feature_importance': self.model.feature_importances_ if self.model is not None else None,
            'optimal_threshold': optimal_threshold,
            'accuracy': self.performance_metrics.get('accuracy', 0.0),
            'precision': self.performance_metrics.get('precision', 0.0),
            'recall': self.performance_metrics.get('recall', 0.0),
            'f1_score': self.performance_metrics.get('f1_score', 0.0)
        }
        
        joblib.dump(model_data, model_path)
        
        print(f"Model saved successfully to: {model_path}")
        return True
    
    def generate_plots(self, save_plots=True):
        """Generate feature importance and threshold analysis plots"""
        if self.model is None:
            return False
        
        # Feature importance plot
        if 'feature_importance' in self.performance_metrics:
            plt.figure(figsize=(10, 8))
            feature_imp = self.performance_metrics['feature_importance'].head(15)
            plt.barh(feature_imp['feature'][::-1], feature_imp['importance'][::-1])
            plt.xlabel('importance')
            plt.title('Feature Importance for Signal Quality Prediction')
            plt.xlabel('Importance')
            plt.tight_layout()
            
            if save_plots:
                import os
                os.makedirs('plots', exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                plt.savefig(f'plots/random_forest_feature_importance_{timestamp}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        return True
    
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
        self.performance_metrics['threshold_analysis'] = {
            'results_df': results_df,
            'optimal_thresholds': optimal_thresholds,
            'optimal_f1_threshold': optimal_thresholds['f1_score']
        }
        
        if progress_callback:
            progress_callback(95, f"Optimal confidence thresholds - F1: {optimal_thresholds['f1_score']:.2f}, Precision: {optimal_thresholds['precision']:.2f}, Recall: {optimal_thresholds['recall']:.2f}")
        
        return optimal_thresholds['f1_score'], results_df
    
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
            
            # Step 4: Train model
            X_test, y_test, y_pred_proba = self.train_model(progress_callback)
            
            # Step 5: Analyse confidence thresholds
            optimal_threshold, threshold_df = self.analyse_confidence_thresholds(
                X_test, y_test, y_pred_proba, progress_callback
            )
            
            # Step 6: Save model
            # Step 7: Generate plots
            self.generate_plots()
            
            if progress_callback:
                progress_callback(100, f"Training complete! Optimal confidence score: {optimal_threshold:.2f}")
            
            return True
            
        except Exception as e:
            if progress_callback:
                progress_callback(100, f"Training failed: {str(e)}")
            return False
    
    def get_performance_summary(self):
        """Get formatted performance summary"""
        if not self.performance_metrics:
            return "Model not trained yet"
        
        summary = f"""
Random Forest Performance Summary:
==================================
Algorithm: Random Forest Classifier with Hyperparameter Optimization
Features: {len(self.feature_columns) if self.feature_columns else 'N/A'}

Classification Metrics:
- Accuracy: {self.performance_metrics.get('accuracy', 0):.3f}
- Precision: {self.performance_metrics.get('precision', 0):.3f}
- Recall: {self.performance_metrics.get('recall', 0):.3f}
- F1-Score: {self.performance_metrics.get('f1_score', 0):.3f}
- ROC-AUC: {self.performance_metrics.get('roc_auc', 0):.3f}

Cross-Validation:
- CV F1 Score: {self.performance_metrics.get('cv_f1_score', 0):.3f}

Best Parameters:
{self.performance_metrics.get('best_params', 'N/A')}

Model files saved:
- random_forest_model.pkl
- feature importance analysis
- hyperparameter optimization results
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
