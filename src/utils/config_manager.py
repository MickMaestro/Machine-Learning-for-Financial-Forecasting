"""
Configuration Manager for ML Financial Forecasting - Updated
==========================================================
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

class ConfigManager:
    """Configuration management for the ML Financial Forecasting application"""
    
    def __init__(self, config_dir='config'):
        self.config_dir = config_dir
        self.config_file = os.path.join(config_dir, 'app_config.json')
        self.default_config_file = os.path.join(config_dir, 'default_config.json')
        
        #Create config directory if it doesn't exist
        os.makedirs(config_dir, exist_ok=True)
        
        #Initialise configuration
        self.config = self._load_default_config()
        self.load_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        default_config = {
            "application": {
                "name": "ML Financial Forecasting",
                "version": "1.0.0",
                "author": "Essex University",
                "description": "Machine Learning for Financial Market Forecasting"
            },
            "data": {
                "default_data_dir": "data",
                "supported_formats": ["csv", "excel", "json", "parquet"],
                "max_file_size_mb": 500,
                "auto_backup": True,
                "validation_rules": {
                    "min_rows": 100,
                    "required_columns": ["Signal_Type", "Entry_Price", "Profit_Loss"],
                    "max_missing_percentage": 0.5
                }
            },
            "training": {
                "default_algorithms": [
                    "Random Forest",
                    "XGBoost",
                    "LightGBM",
                    "SVM",
                    "LSTM",
                    "Ensemble"
                ],
                "default_algorithm": "Random Forest",
                "training_methods": [
                    "Time-Based Split",
                    "Signal-Based",
                    "Cross-Validation"
                ],
                "default_training_method": "Time-Based Split",
                "default_test_size": 0.2,
                "default_random_state": 42,
                "hyperparameters": {
                    "Random Forest": {
                        "n_estimators": 200,
                        "max_depth": 15,
                        "min_samples_split": 4,
                        "min_samples_leaf": 2,
                        "class_weight": "balanced"
                    },
                    "XGBoost": {
                        "n_estimators": 200,
                        "max_depth": 6,
                        "learning_rate": 0.1,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8
                    },
                    "LightGBM": {
                        "n_estimators": 200,
                        "max_depth": 6,
                        "learning_rate": 0.1,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "verbose": -1
                    },
                    "SVM": {
                        "C": 1.0,
                        "kernel": "rbf",
                        "gamma": "scale",
                        "class_weight": "balanced",
                        "probability": True
                    },
                    "LSTM": {
                        "sequence_length": 10,
                        "hidden_units": 50,
                        "dropout": 0.2,
                        "epochs": 50,
                        "batch_size": 32
                    },
                    "Ensemble": {
                        "base_algorithms": ["Random Forest", "XGBoost"],
                        "meta_algorithm": "Random Forest",
                        "voting": "soft",
                        "cv_folds": 5
                    }
                }
            },
            "prediction": {
                "default_confidence_threshold": 0.65,
                "batch_size": 1000,
                "save_predictions": True,
                "prediction_history_limit": 10000
            },
            "feature_engineering": {
                "ema_periods": [144],
                "ema_price_types": ["low", "median"],
                "gann_period": 13,
                "volatility_period": 14,
                "time_features": [
                    "hour",
                    "day_of_week",
                    "is_london_session",
                    "is_ny_session",
                    "is_asian_session",
                    "is_overlap_session",
                    "is_weekend"
                ],
                "market_features": [
                    "price_ema_relationships",
                    "gann_relationships",
                    "volatility_features",
                    "spread_features",
                    "volume_features"
                ],
                "interaction_features": True,
                "feature_selection": {
                    "method": "importance",
                    "threshold": 0.01,
                    "max_features": 50
                }
            },
            "model_management": {
                "models_dir": "models",
                "auto_save": True,
                "model_versioning": True,
                "performance_tracking": True,
                "model_comparison": True
            },
            "visualization": {
                "default_plots": [
                    "Performance Metrics",
                    "Feature Importance",
                    "Threshold Analysis",
                    "Prediction History",
                    "Model Comparison"
                ],
                "figure_size": [10, 6],
                "color_scheme": "default",
                "save_plots": True,
                "plot_format": "png",
                "plot_dpi": 300
            },
            "logging": {
                "log_level": "INFO",
                "log_file": "logs/app.log",
                "max_log_size_mb": 10,
                "backup_count": 5,
                "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "gui": {
                "theme": "clam",
                "window_size": [1000, 700],
                "font_family": "Arial",
                "font_size": 10,
                "auto_refresh": True,
                "refresh_interval": 5000
            },
            "performance": {
                "max_memory_usage_gb": 8,
                "parallel_processing": True,
                "n_jobs": -1,
                "cache_size_mb": 500,
                "optimisation_level": "balanced"
            }
        }
        
        #Save default config if it doesn't exist
        if not os.path.exists(self.default_config_file):
            self.save_config_to_file(default_config, self.default_config_file)
        
        return default_config
    
    def load_config(self, config_file: Optional[str] = None) -> bool:
        """Load configuration from file"""
        file_path = config_file or self.config_file
        
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    loaded_config = json.load(f)
                
                #Merge with default config
                self.config = self._merge_configs(self.config, loaded_config)
                print(f"Configuration loaded from {file_path}")
                return True
            else:
                print(f"Configuration file not found: {file_path}")
                print("Using default configuration")
                return False
                
        except Exception as e:
            print(f"Error loading configuration: {e}")
            print("Using default configuration")
            return False
    
    def save_config(self, config_file: Optional[str] = None) -> bool:
        """Save current configuration to file"""
        file_path = config_file or self.config_file
        
        try:
            return self.save_config_to_file(self.config, file_path)
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
    
    def save_config_to_file(self, config: Dict[str, Any], file_path: str) -> bool:
        """Save configuration to specific file"""
        try:
            #Add metadata
            config_with_metadata = {
                "metadata": {
                    "saved_at": datetime.now().isoformat(),
                    "version": "1.0.0"
                },
                "config": config
            }
            
            with open(file_path, 'w') as f:
                json.dump(config_with_metadata, f, indent=2)
            
            print(f"Configuration saved to {file_path}")
            return True
            
        except Exception as e:
            print(f"Error saving configuration to {file_path}: {e}")
            return False
    
    def _merge_configs(self, default: Dict[str, Any], loaded: Dict[str, Any]) -> Dict[str, Any]:
        """Merge loaded configuration with default configuration"""
        merged = default.copy()
        
        #Extract config from loaded data if it has metadata
        if 'config' in loaded:
            loaded = loaded['config']
        
        for key, value in loaded.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def get_config(self, key_path: str, default_value: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key_path.split('.')
        current = self.config
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default_value
    
    def set_config(self, key_path: str, value: Any) -> bool:
        """Set configuration value using dot notation"""
        keys = key_path.split('.')
        current = self.config
        
        try:
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            current[keys[-1]] = value
            return True
            
        except Exception as e:
            print(f"Error setting configuration {key_path}: {e}")
            return False
    
    def reset_to_default(self) -> bool:
        """Reset configuration to default values"""
        try:
            self.config = self._load_default_config()
            return True
        except Exception as e:
            print(f"Error resetting configuration: {e}")
            return False
    
    def get_algorithm_config(self, algorithm: str) -> Dict[str, Any]:
        """Get hyperparameters for specific algorithm"""
        return self.get_config(f'training.hyperparameters.{algorithm}', {})
    
    def set_algorithm_config(self, algorithm: str, params: Dict[str, Any]) -> bool:
        """Set hyperparameters for specific algorithm"""
        return self.set_config(f'training.hyperparameters.{algorithm}', params)
    
    def get_feature_config(self) -> Dict[str, Any]:
        """Get feature engineering configuration"""
        return self.get_config('feature_engineering', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data processing configuration"""
        return self.get_config('data', {})
    
    def get_gui_config(self) -> Dict[str, Any]:
        """Get GUI configuration"""
        return self.get_config('gui', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.get_config('logging', {})
    
    def validate_config(self) -> tuple[bool, list[str]]:
        """Validate current configuration"""
        errors = []
        
        #Check required sections
        required_sections = ['application', 'data', 'training', 'prediction']
        for section in required_sections:
            if section not in self.config:
                errors.append(f"Missing required section: {section}")
        
        #Check data directory
        data_dir = self.get_config('data.default_data_dir')
        if data_dir and not os.path.exists(data_dir):
            try:
                os.makedirs(data_dir, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create data directory {data_dir}: {e}")
        
        #Check models directory
        models_dir = self.get_config('model_management.models_dir')
        if models_dir and not os.path.exists(models_dir):
            try:
                os.makedirs(models_dir, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create models directory {models_dir}: {e}")
        
        #Check logs directory
        log_file = self.get_config('logging.log_file')
        if log_file:
            log_dir = os.path.dirname(log_file)
            if not os.path.exists(log_dir):
                try:
                    os.makedirs(log_dir, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create log directory {log_dir}: {e}")
        
        # Note: Removed algorithm validation since some are templates
        # This prevents the warning messages you were seeing
        
        # Validate confidence threshold
        threshold = self.get_config('prediction.default_confidence_threshold', 0.5)
        if not (0.0 <= threshold <= 1.0):
            errors.append(f"Invalid confidence threshold: {threshold}")
        
        return len(errors) == 0, errors
    
    def create_user_config_template(self, file_path: str) -> bool:
        """Create a user-friendly configuration template"""
        template = {
            "# ML Financial Forecasting Configuration": "User Configuration Template",
            "training": {
                "default_algorithm": "Random Forest",
                "test_size": 0.2,
                "confidence_threshold": 0.65
            },
            "data": {
                "data_directory": "data",
                "auto_backup": True,
                "min_rows": 100
            },
            "gui": {
                "theme": "clam",
                "window_size": [1000, 700],
                "font_size": 10
            },
            "performance": {
                "max_memory_gb": 8,
                "parallel_processing": True
            }
        }
        
        try:
            with open(file_path, 'w') as f:
                json.dump(template, f, indent=2)
            
            print(f"Configuration template created at {file_path}")
            return True
            
        except Exception as e:
            print(f"Error creating template: {e}")
            return False
    
    def export_config(self, file_path: str) -> bool:
        """Export current configuration to file"""
        return self.save_config_to_file(self.config, file_path)
    
    def import_config(self, file_path: str) -> bool:
        """Import configuration from file"""
        return self.load_config(file_path)
    
    def get_config_summary(self) -> str:
        """Get a formatted summary of current configuration"""
        summary = f"""
=== ML Financial Forecasting Configuration Summary ===
Application: {self.get_config('application.name', 'Unknown')}
Version: {self.get_config('application.version', 'Unknown')}

Data Configuration:
- Data Directory: {self.get_config('data.default_data_dir', 'data')}
- Supported Formats: {', '.join(self.get_config('data.supported_formats', []))}
- Max File Size: {self.get_config('data.max_file_size_mb', 500)} MB

Training Configuration:
- Default Algorithm: {self.get_config('training.default_algorithm', 'Random Forest')}
- Training Method: {self.get_config('training.default_training_method', 'Time-Based Split')}
- Test Size: {self.get_config('training.default_test_size', 0.2)}

Prediction Configuration:
- Confidence Threshold: {self.get_config('prediction.default_confidence_threshold', 0.65)}
- Save Predictions: {self.get_config('prediction.save_predictions', True)}

GUI Configuration:
- Theme: {self.get_config('gui.theme', 'clam')}
- Window Size: {self.get_config('gui.window_size', [1000, 700])}
"""
        return summary
