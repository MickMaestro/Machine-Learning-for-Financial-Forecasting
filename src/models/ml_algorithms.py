"""
ML Algorithm implementations for Financial Forecasting - Updated with LightGBM
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import numpy as np

# Try to import advanced ML libraries with fallbacks
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from src.utils.logging_utils import get_logger
    logger = get_logger('ml_algorithms')
except ImportError:
    import logging
    logger = logging.getLogger('ml_algorithms')

class AlgorithmFactory:
    """Factory for creating ML algorithms"""
    
    @staticmethod
    def create_algorithm(algorithm_name, random_state=42, **kwargs):
        """Create algorithm instance"""
        
        if algorithm_name == "Random Forest":
            return RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 200),
                max_depth=kwargs.get('max_depth', 15),
                min_samples_split=kwargs.get('min_samples_split', 4),
                min_samples_leaf=kwargs.get('min_samples_leaf', 2),
                random_state=random_state,
                class_weight=kwargs.get('class_weight', 'balanced'),
                n_jobs=-1
            )
        
        elif algorithm_name == "XGBoost":
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not available. Install with: pip install xgboost")
            
            return xgb.XGBClassifier(
                n_estimators=kwargs.get('n_estimators', 200),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                subsample=kwargs.get('subsample', 0.8),
                colsample_bytree=kwargs.get('colsample_bytree', 0.8),
                random_state=random_state,
                eval_metric='logloss',
                verbosity=0,
                use_label_encoder=False
            )
        
        elif algorithm_name == "LightGBM":
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM not available. Install with: pip install lightgbm")
            
            return LGBMClassifier(
                n_estimators=kwargs.get('n_estimators', 200),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                subsample=kwargs.get('subsample', 0.8),
                colsample_bytree=kwargs.get('colsample_bytree', 0.8),
                num_leaves=kwargs.get('num_leaves', 31),
                min_child_samples=kwargs.get('min_child_samples', 20),
                reg_alpha=kwargs.get('reg_alpha', 0.0),
                reg_lambda=kwargs.get('reg_lambda', 0.0),
                random_state=random_state,
                verbose=-1,
                objective='binary',
                metric='binary_logloss',
                boosting_type='gbdt'
            )
        
        elif algorithm_name == "SVM":
            return SVC(
                C=kwargs.get('C', 1.0),
                kernel=kwargs.get('kernel', 'rbf'),
                gamma=kwargs.get('gamma', 'scale'),
                random_state=random_state,
                probability=True,
                class_weight=kwargs.get('class_weight', 'balanced')
            )
        
        elif algorithm_name == "LSTM":
            # Import here to avoid circular imports
            from src.trainers.lstm_trainer import LSTMTrainer, TF_AVAILABLE
            if not TF_AVAILABLE:
                raise ImportError("TensorFlow not available. Install with: pip install tensorflow")
            # Return a placeholder that will be handled by the trainer
            return "LSTM_PLACEHOLDER"
        
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    @staticmethod
    def get_available_algorithms():
        """Get list of available algorithms"""
        algorithms = ["Random Forest", "SVM"]
        
        if XGBOOST_AVAILABLE:
            algorithms.append("XGBoost")
        
        if LIGHTGBM_AVAILABLE:
            algorithms.append("LightGBM")
        
        # Check LSTM availability
        try:
            from src.trainers.lstm_trainer import TF_AVAILABLE
            if TF_AVAILABLE:
                algorithms.append("LSTM")
        except ImportError:
            pass
        
        return algorithms
    
    @staticmethod
    def get_algorithm_status():
        """Get status of algorithm availability"""
        # Check LSTM availability
        try:
            from src.trainers.lstm_trainer import TF_AVAILABLE
            lstm_available = TF_AVAILABLE
        except ImportError:
            lstm_available = False
        
        status = {
            "Random Forest": True,
            "SVM": True,
            "LSTM": lstm_available,
            "XGBoost": XGBOOST_AVAILABLE,
            "LightGBM": LIGHTGBM_AVAILABLE,
            "Ensemble": True
        }
        return status
    
    @staticmethod
    def get_default_params(algorithm_name):
        """Get default parameters for algorithm"""
        defaults = {
            "Random Forest": {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 4,
                'min_samples_leaf': 2,
                'class_weight': 'balanced'
            },
            "XGBoost": {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            },
            "LightGBM": {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'num_leaves': 31,
                'min_child_samples': 20,
                'reg_alpha': 0.0,
                'reg_lambda': 0.0
            },
            "SVM": {
                'C': 1.0,
                'kernel': 'rbf',
                'gamma': 'scale',
                'class_weight': 'balanced'
            },
            "LSTM": {
                'sequence_length': 10,
                'hidden_units': 50,
                'dropout': 0.2,
                'epochs': 50,
                'batch_size': 32,
                'learning_rate': 0.001,
                'validation_split': 0.2,
                'patience': 10,
                'min_delta': 0.001
            }
        }
        
        return defaults.get(algorithm_name, {})

class EnsembleAlgorithms:
    """Ensemble algorithm implementations"""
    
    @staticmethod
    def create_voting_classifier(algorithms_list, random_state=42):
        """Create voting classifier from multiple algorithms"""
        from sklearn.ensemble import VotingClassifier
        
        estimators = []
        for algo_name in algorithms_list:
            try:
                model = AlgorithmFactory.create_algorithm(algo_name, random_state)
                estimators.append((algo_name.lower().replace(' ', '_'), model))
            except ImportError as e:
                logger.warning(f"Skipping {algo_name}: {e}")
                continue
        
        if not estimators:
            raise ValueError("No valid algorithms available for ensemble")
        
        return VotingClassifier(estimators=estimators, voting='soft')
    
    @staticmethod
    def create_stacking_classifier(base_algorithms, meta_algorithm="Logistic Regression", random_state=42):
        """Create stacking classifier"""
        from sklearn.ensemble import StackingClassifier
        
        base_estimators = []
        for algo_name in base_algorithms:
            try:
                model = AlgorithmFactory.create_algorithm(algo_name, random_state)
                base_estimators.append((algo_name.lower().replace(' ', '_'), model))
            except ImportError as e:
                logger.warning(f"Skipping {algo_name}: {e}")
                continue
        
        if not base_estimators:
            raise ValueError("No valid base algorithms available for stacking")
        
        meta_learner = AlgorithmFactory.create_algorithm(meta_algorithm, random_state)
        
        return StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=5
        )

class LSTMClassifier:
    """LSTM implementation for time series classification"""
    
    def __init__(self, sequence_length=10, hidden_units=50, dropout=0.2, random_state=42):
        self.sequence_length = sequence_length
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.random_state = random_state
        self.model = None
        self.scaler = None
        
    def _create_sequences(self, X, y=None):
        """Create sequences for LSTM input"""
        sequences = []
        targets = [] if y is not None else None
        
        for i in range(len(X) - self.sequence_length + 1):
            sequences.append(X[i:i + self.sequence_length])
            if y is not None:
                targets.append(y[i + self.sequence_length - 1])
        
        sequences = np.array(sequences)
        if targets is not None:
            targets = np.array(targets)
            return sequences, targets
        
        return sequences
    
    def fit(self, X, y):
        """Train LSTM model"""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from sklearn.preprocessing import StandardScaler
            
            # Set random seed
            tf.random.set_seed(self.random_state)
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Create sequences
            X_seq, y_seq = self._create_sequences(X_scaled, y)
            
            # Build model
            self.model = Sequential([
                LSTM(self.hidden_units, return_sequences=True, input_shape=(self.sequence_length, X.shape[1])),
                Dropout(self.dropout),
                LSTM(self.hidden_units, return_sequences=False),
                Dropout(self.dropout),
                Dense(32, activation='relu'),
                Dropout(self.dropout),
                Dense(1, activation='sigmoid')
            ])
            
            self.model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Train model
            self.model.fit(
                X_seq, y_seq,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            return self
            
        except ImportError:
            logger.warning("TensorFlow not available, LSTM training skipped")
            return self
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None or self.scaler is None:
            return np.zeros(len(X))
        
        X_scaled = self.scaler.transform(X)
        X_seq = self._create_sequences(X_scaled)
        
        if len(X_seq) == 0:
            return np.zeros(len(X))
        
        predictions = self.model.predict(X_seq, verbose=0)
        
        # Pad predictions to match input length
        full_predictions = np.zeros(len(X))
        full_predictions[-len(predictions):] = predictions.flatten()
        
        return (full_predictions > 0.5).astype(int)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.model is None or self.scaler is None:
            probs = np.full((len(X), 2), 0.5)
            return probs
        
        X_scaled = self.scaler.transform(X)
        X_seq = self._create_sequences(X_scaled)
        
        if len(X_seq) == 0:
            probs = np.full((len(X), 2), 0.5)
            return probs
        
        predictions = self.model.predict(X_seq, verbose=0)
        
        # Pad predictions to match input length
        full_predictions = np.full(len(X), 0.5)
        full_predictions[-len(predictions):] = predictions.flatten()
        
        # Convert to probability format
        probs = np.column_stack([1 - full_predictions, full_predictions])
        return probs

class AlgorithmTuner:
    """Hyperparameter tuning for algorithms"""
    
    @staticmethod
    def tune_random_forest(X, y, cv=5):
        """Tune Random Forest hyperparameters"""
        from sklearn.model_selection import GridSearchCV
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 4, 6],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
        grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring='f1', n_jobs=-1)
        grid_search.fit(X, y)
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    @staticmethod
    def tune_xgboost(X, y, cv=5):
        """Tune XGBoost hyperparameters"""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available")
        
        from sklearn.model_selection import GridSearchCV
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0)
        grid_search = GridSearchCV(xgb_model, param_grid, cv=cv, scoring='f1', n_jobs=-1)
        grid_search.fit(X, y)
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    @staticmethod
    def tune_lightgbm(X, y, cv=5):
        """Tune LightGBM hyperparameters"""
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available")
        
        from sklearn.model_selection import GridSearchCV
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [15, 31, 63],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        lgb_model = LGBMClassifier(random_state=42, verbose=-1, objective='binary')
        grid_search = GridSearchCV(lgb_model, param_grid, cv=cv, scoring='f1', n_jobs=-1)
        grid_search.fit(X, y)
        
        return grid_search.best_estimator_, grid_search.best_params_

# Utility functions
def check_algorithm_availability():
    """Check which algorithms are available"""
    status = AlgorithmFactory.get_algorithm_status()
    
    print("Algorithm Availability Status:")
    print("=" * 40)
    for algo, available in status.items():
        status_str = "Available" if available else "Not Available"
        print(f"{algo:<20}: {status_str}")
    
    if not XGBOOST_AVAILABLE:
        print("\nTo install XGBoost: pip install xgboost")
    
    if not LIGHTGBM_AVAILABLE:
        print("To install LightGBM: pip install lightgbm")
    
    return status

def get_recommended_algorithm(dataset_size, complexity="medium"):
    """Get recommended algorithm based on dataset characteristics"""
    if dataset_size < 1000:
        return "Random Forest"
    elif dataset_size < 10000:
        if LIGHTGBM_AVAILABLE and complexity == "high":
            return "LightGBM"
        elif XGBOOST_AVAILABLE:
            return "XGBoost"
        else:
            return "Random Forest"
    else:
        if LIGHTGBM_AVAILABLE:
            return "LightGBM"  # Best for large datasets
        elif XGBOOST_AVAILABLE:
            return "XGBoost"
        else:
            return "Random Forest"

if __name__ == "__main__":
    # Test algorithm availability
    check_algorithm_availability()
    
    # Test algorithm creation
    try:
        rf = AlgorithmFactory.create_algorithm("Random Forest")
        print("\nRandom Forest created successfully")
    except Exception as e:
        print(f"\nRandom Forest failed: {e}")
    
    if LIGHTGBM_AVAILABLE:
        try:
            lgb = AlgorithmFactory.create_algorithm("LightGBM")
            print("LightGBM created successfully")
        except Exception as e:
            print(f"LightGBM failed: {e}")
    
    print(f"\nRecommended algorithm for 5000 samples: {get_recommended_algorithm(5000)}")
    print(f"Recommended algorithm for 50000 samples: {get_recommended_algorithm(50000)}")