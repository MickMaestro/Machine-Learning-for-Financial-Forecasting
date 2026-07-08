"""
Model Evaluator for Financial ML
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report)
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from src.utils.logging_utils import get_logger
from src.utils.visualization import (plot_performance_metrics, plot_feature_importance,
                                   plot_threshold_analysis, plot_model_comparison)

logger = get_logger('model_evaluator')

class ModelEvaluator:
    """Comprehensive model evaluation for financial forecasting"""
    
    def __init__(self):
        self.evaluation_results = {}
    
    def evaluate_classification_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate standard classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        return metrics
    
    def evaluate_financial_metrics(self, test_data, y_pred, y_pred_proba, threshold=0.6):
        """Calculate financial performance metrics"""
        if 'Profit_Loss' not in test_data.columns:
            logger.warning("No Profit_Loss column found for financial metrics")
            return {}
        
        # Apply threshold to get final predictions
        final_predictions = (y_pred_proba >= threshold).astype(int)
        
        # Filter signals based on predictions
        executed_signals = test_data[final_predictions == 1]
        filtered_signals = test_data[final_predictions == 0]
        
        if len(executed_signals) == 0:
            logger.warning("No signals executed with current threshold")
            return {'signals_executed': 0}
        
        # Calculate financial metrics
        profits_pips = executed_signals['Profit_Loss'] * 10000
        baseline_profits_pips = test_data['Profit_Loss'] * 10000
        
        financial_metrics = {
            'total_signals': len(test_data),
            'signals_executed': len(executed_signals),
            'signals_filtered': len(filtered_signals),
            'execution_rate': len(executed_signals) / len(test_data),
            
            # Profit metrics
            'avg_profit_per_trade': profits_pips.mean(),
            'total_profit': profits_pips.sum(),
            'baseline_avg_profit': baseline_profits_pips.mean(),
            'baseline_total_profit': baseline_profits_pips.sum(),
            
            # Win rates
            'win_rate': (profits_pips > 0).mean(),
            'baseline_win_rate': (baseline_profits_pips > 0).mean(),
            
            # Risk metrics
            'max_loss': profits_pips.min(),
            'max_gain': profits_pips.max(),
            'profit_factor': profits_pips[profits_pips > 0].sum() / abs(profits_pips[profits_pips < 0].sum()) if (profits_pips < 0).any() else float('inf'),
            
            # Improvement metrics
            'profit_improvement': profits_pips.mean() - baseline_profits_pips.mean(),
            'win_rate_improvement': (profits_pips > 0).mean() - (baseline_profits_pips > 0).mean()
        }
        
        return financial_metrics
    
    def evaluate_threshold_performance(self, model, X_test, y_test, test_data=None, 
                                     threshold_range=(0.3, 0.9, 0.05)):
        """Evaluate performance across different confidence thresholds"""
        thresholds = np.arange(*threshold_range)
        results = []
        
        # Get predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            
            # Classification metrics
            metrics = self.evaluate_classification_metrics(y_test, y_pred_thresh, y_pred_proba)
            
            # Add threshold and retention rate
            metrics['threshold'] = threshold
            metrics['signals_retained'] = np.sum(y_pred_thresh == 1) / len(y_pred_thresh)
            
            # Financial metrics if available
            if test_data is not None:
                financial_metrics = self.evaluate_financial_metrics(
                    test_data, y_pred_thresh, y_pred_proba, threshold
                )
                metrics.update(financial_metrics)
            
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def cross_validate_model(self, model, X, y, cv=5, scoring='f1'):
        """Perform time-series cross-validation"""
        tscv = TimeSeriesSplit(n_splits=cv)
        scores = cross_val_score(model, X, y, cv=tscv, scoring=scoring)
        
        cv_results = {
            'cv_scores': scores,
            'cv_mean': scores.mean(),
            'cv_std': scores.std(),
            'cv_min': scores.min(),
            'cv_max': scores.max()
        }
        
        logger.info(f"Cross-validation {scoring}: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
        return cv_results
    
    def evaluate_feature_importance(self, model, feature_columns):
        """Evaluate feature importance"""
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not support feature importance")
            return None
        
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def extract_top_features(self, model, feature_columns, n_features=10):
        """Extract top N most important features for MQH generation"""
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not support feature importance, using default features")
            # Return first n_features as default
            indices = list(range(min(n_features, len(feature_columns))))
            scores = [1.0/len(indices)] * len(indices)
            return indices, scores
        
        # Get feature importance and sort by importance
        importance_scores = model.feature_importances_
        sorted_indices = np.argsort(importance_scores)[::-1][:n_features]
        sorted_scores = importance_scores[sorted_indices]
        
        logger.info(f"Top {n_features} features extracted for MQH generation")
        for i, (idx, score) in enumerate(zip(sorted_indices, sorted_scores)):
            feature_name = feature_columns[idx] if idx < len(feature_columns) else f"Feature_{idx}"
            logger.info(f"  {i+1}. {feature_name}: {score:.4f}")
        
        return sorted_indices.tolist(), sorted_scores.tolist()
    
    def evaluate_model_stability(self, model, X_test, y_test, n_iterations=10):
        """Evaluate model prediction stability"""
        predictions = []
        probabilities = []
        
        for i in range(n_iterations):
            # Add small random noise to test stability
            X_noisy = X_test + np.random.normal(0, 0.001, X_test.shape)
            pred = model.predict(X_noisy)
            prob = model.predict_proba(X_noisy)[:, 1]
            
            predictions.append(pred)
            probabilities.append(prob)
        
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        stability_metrics = {
            'prediction_variance': np.var(predictions, axis=0).mean(),
            'probability_variance': np.var(probabilities, axis=0).mean(),
            'prediction_stability': 1 - np.var(predictions, axis=0).mean(),
            'probability_stability': 1 - np.var(probabilities, axis=0).mean()
        }
        
        return stability_metrics
    
    def comprehensive_evaluation(self, model, X_test, y_test, test_data=None, 
                                feature_columns=None, model_name="Model"):
        """Perform comprehensive model evaluation"""
        logger.info(f"Starting comprehensive evaluation for {model_name}")
        
        results = {'model_name': model_name}
        
        # Basic predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Classification metrics
        results['classification_metrics'] = self.evaluate_classification_metrics(
            y_test, y_pred, y_pred_proba
        )
        
        # Financial metrics
        if test_data is not None:
            results['financial_metrics'] = self.evaluate_financial_metrics(
                test_data, y_pred, y_pred_proba
            )
        
        # Cross-validation
        results['cross_validation'] = self.cross_validate_model(model, X_test, y_test)
        
        # Feature importance
        if feature_columns is not None:
            results['feature_importance'] = self.evaluate_feature_importance(model, feature_columns)
        
        # Threshold analysis
        results['threshold_analysis'] = self.evaluate_threshold_performance(
            model, X_test, y_test, test_data
        )
        
        # Model stability
        results['stability_metrics'] = self.evaluate_model_stability(model, X_test, y_test)
        
        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        
        # Classification report
        results['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
        
        self.evaluation_results[model_name] = results
        logger.info(f"Comprehensive evaluation completed for {model_name}")
        
        return results
    
    def compare_models(self, models_results):
        """Compare multiple models"""
        comparison = {}
        
        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        for metric in metrics_to_compare:
            comparison[metric] = {}
            for model_name, results in models_results.items():
                comparison[metric][model_name] = results['classification_metrics'].get(metric, 0)
        
        # Find best model for each metric
        best_models = {}
        for metric in metrics_to_compare:
            best_model = max(comparison[metric], key=comparison[metric].get)
            best_models[metric] = {
                'model': best_model,
                'score': comparison[metric][best_model]
            }
        
        return comparison, best_models
    
    def generate_evaluation_report(self, model_name=None):
        """Generate comprehensive evaluation report"""
        if model_name is None:
            # Report on all models
            models_to_report = list(self.evaluation_results.keys())
        else:
            models_to_report = [model_name] if model_name in self.evaluation_results else []
        
        if not models_to_report:
            return "No evaluation results available"
        
        report = "=== MODEL EVALUATION REPORT ===\n\n"
        
        for model in models_to_report:
            results = self.evaluation_results[model]
            
            report += f"Model: {model}\n"
            report += "=" * 50 + "\n"
            
            # Classification metrics
            cm = results['classification_metrics']
            report += f"Classification Performance:\n"
            report += f"  Accuracy: {cm['accuracy']:.3f}\n"
            report += f"  Precision: {cm['precision']:.3f}\n"
            report += f"  Recall: {cm['recall']:.3f}\n"
            report += f"  F1-Score: {cm['f1_score']:.3f}\n"
            report += f"  ROC-AUC: {cm.get('roc_auc', 'N/A')}\n\n"
            
            # Financial metrics
            if 'financial_metrics' in results:
                fm = results['financial_metrics']
                report += f"Financial Performance:\n"
                report += f"  Total Signals: {fm.get('total_signals', 'N/A')}\n"
                report += f"  Signals Executed: {fm.get('signals_executed', 'N/A')}\n"
                report += f"  Execution Rate: {fm.get('execution_rate', 0):.1%}\n"
                report += f"  Win Rate: {fm.get('win_rate', 0):.1%}\n"
                report += f"  Avg Profit per Trade: {fm.get('avg_profit_per_trade', 0):.1f} pips\n"
                report += f"  Profit Factor: {fm.get('profit_factor', 'N/A')}\n\n"
            
            # Cross-validation
            cv = results['cross_validation']
            report += f"Cross-Validation:\n"
            report += f"  Mean Score: {cv['cv_mean']:.3f}\n"
            report += f"  Std Dev: {cv['cv_std']:.3f}\n"
            report += f"  Score Range: {cv['cv_min']:.3f} - {cv['cv_max']:.3f}\n\n"
            
            # Stability
            stability = results['stability_metrics']
            report += f"Model Stability:\n"
            report += f"  Prediction Stability: {stability['prediction_stability']:.3f}\n"
            report += f"  Probability Stability: {stability['probability_stability']:.3f}\n\n"
            
            report += "\n"
        
        return report
    
    def save_evaluation_plots(self, model_name, output_dir="plots"):
        """Save evaluation plots for a model"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if model_name not in self.evaluation_results:
            logger.error(f"No evaluation results for {model_name}")
            return []
        
        results = self.evaluation_results[model_name]
        saved_plots = []
        
        # Performance metrics plot
        if 'classification_metrics' in results:
            fig = plot_performance_metrics(
                results['classification_metrics'],
                title=f"{model_name} - Performance Metrics"
            )
            plot_path = os.path.join(output_dir, f"{model_name}_performance.png")
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            saved_plots.append(plot_path)
        
        # Feature importance plot
        if 'feature_importance' in results and results['feature_importance'] is not None:
            fig = plot_feature_importance(
                results['feature_importance'],
                title=f"{model_name} - Feature Importance"
            )
            plot_path = os.path.join(output_dir, f"{model_name}_features.png")
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            saved_plots.append(plot_path)
        
        # Threshold analysis plot
        if 'threshold_analysis' in results:
            fig = plot_threshold_analysis(
                results['threshold_analysis'],
                title=f"{model_name} - Threshold Analysis"
            )
            plot_path = os.path.join(output_dir, f"{model_name}_thresholds.png")
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            saved_plots.append(plot_path)
        
        logger.info(f"Saved {len(saved_plots)} evaluation plots for {model_name}")
        return saved_plots
    
    def export_results(self, output_file="evaluation_results.json"):
        """Export evaluation results to JSON"""
        import json
        
        # Convert numpy arrays and pandas objects to serializable format
        serializable_results = {}
        
        for model_name, results in self.evaluation_results.items():
            serializable_results[model_name] = {}
            
            for key, value in results.items():
                if key == 'confusion_matrix':
                    serializable_results[model_name][key] = value.tolist()
                elif key == 'feature_importance' and value is not None:
                    serializable_results[model_name][key] = value.to_dict('records')
                elif key == 'threshold_analysis':
                    serializable_results[model_name][key] = value.to_dict('records')
                elif isinstance(value, (dict, list, str, int, float, bool)):
                    serializable_results[model_name][key] = value
                else:
                    serializable_results[model_name][key] = str(value)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Evaluation results exported to {output_file}")
        return output_file
    

class PerformanceTracker:
    """Track model performance over time"""
    
    def __init__(self):
        self.performance_history = []
    
    def record_performance(self, model_name, metrics, timestamp=None):
        """Record performance metrics"""
        from datetime import datetime
        
        if timestamp is None:
            timestamp = datetime.now()
        
        record = {
            'timestamp': timestamp,
            'model_name': model_name,
            'metrics': metrics.copy()
        }
        
        self.performance_history.append(record)
    
    def get_performance_trend(self, model_name, metric='f1_score'):
        """Get performance trend for a specific model and metric"""
        model_records = [r for r in self.performance_history if r['model_name'] == model_name]
        
        if not model_records:
            return None
        
        timestamps = [r['timestamp'] for r in model_records]
        values = [r['metrics'].get(metric, 0) for r in model_records]
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'value': values,
            'metric': metric,
            'model': model_name
        })
    
    def compare_model_trends(self, metric='f1_score'):
        """Compare performance trends across all models"""
        trends = {}
        
        model_names = list(set(r['model_name'] for r in self.performance_history))
        
        for model_name in model_names:
            trend = self.get_performance_trend(model_name, metric)
            if trend is not None:
                trends[model_name] = trend
        
        return trends
    
    def get_best_performing_model(self, metric='f1_score'):
        """Get the best performing model for a specific metric"""
        if not self.performance_history:
            return None
        
        best_score = -1
        best_model = None
        
        for record in self.performance_history:
            score = record['metrics'].get(metric, 0)
            if score > best_score:
                best_score = score
                best_model = record['model_name']
        
        return {
            'model': best_model,
            'score': best_score,
            'metric': metric
        }
