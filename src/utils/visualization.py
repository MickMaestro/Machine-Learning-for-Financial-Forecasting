"""
Visualization utilities for ML Financial Forecasting
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import os

def setup_plot_style():
    """Setup consistent plot styling"""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 10

def plot_performance_metrics(metrics_dict, title="Model Performance", save_path=None):
    """Plot performance metrics as bar chart"""
    setup_plot_style()
    
    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, alpha=0.8)
    plt.title(title)
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    #Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def plot_feature_importance(feature_importance_df, top_n=15, title="Feature Importance", save_path=None):
    """Plot feature importance"""
    setup_plot_style()
    
    top_features = feature_importance_df.head(top_n)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=top_features, x='importance', y='feature', orient='h')
    plt.title(title)
    plt.xlabel('Importance Score')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def plot_training_history(history_list, title="Training History", save_path=None):
    """Plot training history over time"""
    setup_plot_style()
    
    if not history_list:
        return None
    
    timestamps = [h['timestamp'] for h in history_list]
    accuracies = [h.get('performance', {}).get('accuracy', 0) for h in history_list]
    
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, accuracies, 'o-', linewidth=2, markersize=6)
    plt.title(title)
    plt.xlabel('Training Time')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def plot_prediction_confidence(predictions_df, title="Prediction Confidence Distribution", save_path=None):
    """Plot prediction confidence distribution"""
    setup_plot_style()
    
    plt.figure(figsize=(10, 6))
    plt.hist(predictions_df['confidence'], bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(predictions_df['confidence'].mean(), color='red', linestyle='--',
               label=f'Mean: {predictions_df["confidence"].mean():.3f}')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def plot_threshold_analysis(threshold_results_df, title="Threshold Analysis", save_path=None):
    """Plot threshold performance analysis"""
    setup_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    #Precision vs Threshold
    axes[0, 0].plot(threshold_results_df['threshold'], threshold_results_df['precision'], 'b-', marker='o')
    axes[0, 0].set_title('Precision vs Threshold')
    axes[0, 0].set_xlabel('Threshold')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].grid(True, alpha=0.3)
    
    #Recall vs Threshold
    axes[0, 1].plot(threshold_results_df['threshold'], threshold_results_df['recall'], 'r-', marker='o')
    axes[0, 1].set_title('Recall vs Threshold')
    axes[0, 1].set_xlabel('Threshold')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].grid(True, alpha=0.3)
    
    #F1 Score vs Threshold
    axes[1, 0].plot(threshold_results_df['threshold'], threshold_results_df['f1_score'], 'g-', marker='o')
    axes[1, 0].set_title('F1 Score vs Threshold')
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].grid(True, alpha=0.3)
    
    #Signal Retention vs Threshold
    axes[1, 1].plot(threshold_results_df['threshold'], threshold_results_df['signals_retained'], 'm-', marker='o')
    axes[1, 1].set_title('Signal Retention Rate vs Threshold')
    axes[1, 1].set_xlabel('Threshold')
    axes[1, 1].set_ylabel('Signals Retained')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_model_comparison(models_performance, title="Model Comparison", save_path=None):
    """Compare multiple models performance"""
    setup_plot_style()
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    model_names = list(models_performance.keys())
    
    x = np.arange(len(metrics))
    width = 0.8 / len(model_names)
    
    plt.figure(figsize=(12, 6))
    
    for i, model_name in enumerate(model_names):
        values = [models_performance[model_name].get(metric, 0) for metric in metrics]
        plt.bar(x + i * width, values, width, label=model_name, alpha=0.8)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title(title)
    plt.xticks(x + width * (len(model_names) - 1) / 2, metrics)
    plt.legend()
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def save_all_plots(plots_dict, output_dir="plots"):
    """Save all plots to directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = []
    for plot_name, fig in plots_dict.items():
        if fig is not None:
            filename = f"{plot_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = os.path.join(output_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
    
    return saved_files
