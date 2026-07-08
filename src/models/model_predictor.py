"""
Model Predictor for Real-Time Financial Forecasting - Data_Exporter2.0 Compatible
==============================================================================
Handles model loading, prediction, and real-time inference with Data_Exporter2.0 features
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import warnings
# Suppress specific ML library warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

class ModelPredictor:
    """Real-time model predictor for financial forecasting with Data_Exporter2.0 compatibility"""
    
    def __init__(self, model_path=None):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.algorithm = None
        self.performance_metrics = {}
        self.prediction_history = []
        # Test data for threshold analysis
        self.X_test = None
        self.y_test = None
        self.csv_file = None
        self.test_size = 0.2
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load a trained model from file"""
        try:
            model_data = joblib.load(model_path)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.algorithm = model_data.get('algorithm', 'Unknown')
            self.performance_metrics = model_data.get('performance_metrics', {})
            
            # Load test data for threshold analysis (if available)
            self.X_test = model_data.get('X_test', None)
            self.y_test = model_data.get('y_test', None)
            self.y_pred_proba = model_data.get('y_pred_proba', None)
            self.csv_file = model_data.get('csv_file', None)
            self.test_size = model_data.get('test_size', 0.2)
            
            print(f"Model loaded: {self.algorithm}")
            if self.X_test is not None:
                print(f"Test data available: {len(self.X_test)} samples")
            else:
                print("No test data available in saved model")
            
            print(f"Model loaded successfully: {self.algorithm}")
            print(f"Features expected: {len(self.feature_columns)}")
            if self.performance_metrics:
                print(f"Model accuracy: {self.performance_metrics.get('accuracy', 'N/A'):.3f}")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    @staticmethod
    def load_model_static(model_path):
        """Static method to load model and return ModelPredictor instance"""
        try:
            predictor = ModelPredictor()
            if predictor.load_model(model_path):
                return predictor
            else:
                return None
        except Exception as e:
            print(f"Error in load_model_static: {e}")
            return None
    
    def predict(self, market_data, confidence_threshold=0.65):
        """
        Make prediction with confidence threshold
        
        Args:
            market_data: Dictionary containing Data_Exporter2.0 features or raw market data
            confidence_threshold: Minimum confidence for EXECUTE recommendation
            
        Returns:
            Dictionary with prediction, confidence, and recommendation
        """
        if self.model is None:
            raise ValueError("No model loaded. Use load_model() first.")
        
        # Prepare features from market data
        features_df = self.prepare_features_data_exporter_2(market_data)
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(features_df)
        prediction = self.model.predict(features_df)[0]
        confidence = np.max(probabilities[0])
        
        # Determine recommendation
        recommendation = "EXECUTE" if confidence >= confidence_threshold else "FILTER"
        
        # Create result
        result = {
            'prediction': int(prediction),
            'confidence': float(confidence),
            'recommendation': recommendation,
            'threshold': confidence_threshold,
            'signal_type': market_data.get('signal_type', 'Unknown'),
            'timestamp': datetime.now(),
            'model_algorithm': self.algorithm,
            'features_used': len(self.feature_columns)
        }
        
        # Store in history
        self.prediction_history.append(result)
        
        return result
    
    def prepare_features_data_exporter_2(self, market_data):
        """
        Prepare features from Data_Exporter2.0 format or convert from raw data
        
        Expected market_data keys for Data_Exporter2.0:
        - All 24 base features from Data_Exporter2.0
        - Or raw data that can be converted to approximate features
        """
        
        # Check if we already have Data_Exporter2.0 features
        data_exporter_2_features = [
            'Gann_Zone_Width_Pips', 'Gann_Zone_Slope', 'Price_Distance_From_Zone_Center',
            'Price_Distance_From_Zone_Boundary', 'Active_Gann_Line_Slope', 'Bars_Since_Last_Gann_Color_Change',
            'EMA_Separation_Pips', 'Median_EMA_Slope', 'Low_EMA_Slope', 'Price_Distance_From_Relevant_EMA',
            'EMA_Convergence_Divergence', 'Trading_Session', 'Day_Of_Week', 'Hour_Of_Day',
            'Entry_Candle_Size_Pips', 'Entry_Candle_Body_Ratio', 'Previous_Candle_Size_Pips',
            'Bars_Since_Relevant_EMA_Cross', 'Initial_SL_Distance_Pips', 'Recent_Gann_Signal_Frequency'
        ]
        
        # Check how many Data_Exporter2.0 features we have
        available_features = [f for f in data_exporter_2_features if f in market_data]
        
        if len(available_features) >= 15:  # Most features available
            print("Using Data_Exporter2.0 features directly")
            features = self._use_data_exporter_2_features(market_data)
        else:
            print("Converting from raw data to approximate Data_Exporter2.0 features")
            features = self._convert_raw_to_data_exporter_2(market_data)
        
        # Create DataFrame with all expected features
        features_df = pd.DataFrame([features])
        
        # Ensure all model features are present
        for col in self.feature_columns:
            if col not in features_df.columns:
                # Provide default values for missing features
                features_df[col] = self._get_default_feature_value(col)
        
        # Select only the features the model expects
        features_df = features_df[self.feature_columns]
        
        # Scale features if scaler is available
        if self.scaler is not None:
            features_df = pd.DataFrame(
                self.scaler.transform(features_df),
                columns=self.feature_columns
            )
        
        return features_df
    
    def _use_data_exporter_2_features(self, market_data):
        """Use Data_Exporter2.0 features directly with derived features"""
        features = {}
        
        # Copy all Data_Exporter2.0 base features
        base_features = [
            'Gann_Zone_Width_Pips', 'Gann_Zone_Slope', 'Price_Distance_From_Zone_Center',
            'Price_Distance_From_Zone_Boundary', 'Active_Gann_Line_Slope', 'Bars_Since_Last_Gann_Color_Change',
            'EMA_Separation_Pips', 'Median_EMA_Slope', 'Low_EMA_Slope', 'Price_Distance_From_Relevant_EMA',
            'EMA_Convergence_Divergence', 'Trading_Session', 'Day_Of_Week', 'Hour_Of_Day',
            'Entry_Candle_Size_Pips', 'Entry_Candle_Body_Ratio', 'Previous_Candle_Size_Pips',
            'Bars_Since_Relevant_EMA_Cross', 'Initial_SL_Distance_Pips', 'Recent_Gann_Signal_Frequency'
        ]
        
        for feature in base_features:
            features[feature] = market_data.get(feature, 0)
        
        # Add derived features
        signal_type = market_data.get('signal_type', market_data.get('Signal_Type', 1))
        spread = market_data.get('spread', market_data.get('Spread', 0.0001))
        trading_session = features.get('Trading_Session', 2)
        
        # Session-based features
        features['Is_Asian_Session'] = 1 if trading_session == 1 else 0
        features['Is_London_Session'] = 1 if trading_session == 2 else 0
        features['Is_NY_Session'] = 1 if trading_session == 3 else 0
        features['Is_Overlap_Session'] = 1 if trading_session == 4 else 0
        
        # Signal type features
        features['Is_Buy_Signal'] = 1 if signal_type == 1 else 0
        features['Is_Sell_Signal'] = 1 if signal_type == -1 else 0
        
        # Spread features
        features['High_Spread'] = 1 if spread > 0.0002 else 0
        features['Low_Spread'] = 1 if spread < 0.00005 else 0
        
        # Alignment features
        price_dist_boundary = features.get('Price_Distance_From_Zone_Boundary', 0)
        price_dist_ema = features.get('Price_Distance_From_Relevant_EMA', 0)
        
        features['Gann_EMA_Alignment'] = 0
        if signal_type == 1 and price_dist_boundary > 0 and price_dist_ema > 0:
            features['Gann_EMA_Alignment'] = 1
        elif signal_type == -1 and price_dist_boundary < 0 and price_dist_ema < 0:
            features['Gann_EMA_Alignment'] = 1
        
        # Momentum alignment
        gann_slope = features.get('Active_Gann_Line_Slope', 0)
        ema_slope = features.get('Median_EMA_Slope', 0)
        
        features['Momentum_Alignment'] = 0
        if (signal_type == 1 and gann_slope > 0 and ema_slope > 0) or \
           (signal_type == -1 and gann_slope < 0 and ema_slope < 0):
            features['Momentum_Alignment'] = 1
        
        return features
    
    def _convert_raw_to_data_exporter_2(self, market_data):
        """Convert raw market data to approximate Data_Exporter2.0 features"""
        print("WARNING: Using approximated features. For best results, use Data_Exporter2.0 CSV files.")
        
        # Extract raw values with defaults
        entry_price = market_data.get('entry_price', market_data.get('Entry_Price', 0))
        signal_type = market_data.get('signal_type', market_data.get('Signal_Type', 1))
        hour = market_data.get('hour', market_data.get('Hour_Of_Day', 12))
        day_of_week = market_data.get('day_of_week', market_data.get('Day_Of_Week', 2))
        spread = market_data.get('spread', market_data.get('Spread', 0.0001))
        
        # Approximate Data_Exporter2.0 features from raw data (if available)
        ema_low = market_data.get('ema_low', entry_price * 0.999)
        ema_median = market_data.get('ema_median', entry_price * 1.001)
        gann_high = market_data.get('gann_high', entry_price * 1.01)
        gann_low = market_data.get('gann_low', entry_price * 0.99)
        
        features = {}
        
        # Approximate Gann features
        features['Gann_Zone_Width_Pips'] = abs(gann_high - gann_low) * 10000
        features['Gann_Zone_Slope'] = 0  # Cannot calculate without history
        features['Price_Distance_From_Zone_Center'] = (entry_price - (gann_high + gann_low)/2) * 10000
        features['Price_Distance_From_Zone_Boundary'] = min(abs(entry_price - gann_high), abs(entry_price - gann_low)) * 10000
        features['Active_Gann_Line_Slope'] = 0  # Cannot calculate without history
        features['Bars_Since_Last_Gann_Color_Change'] = 10  # Default estimate
        
        # Approximate EMA features
        features['EMA_Separation_Pips'] = abs(ema_median - ema_low) * 10000
        features['Median_EMA_Slope'] = 0  # Cannot calculate without history
        features['Low_EMA_Slope'] = 0  # Cannot calculate without history
        features['Price_Distance_From_Relevant_EMA'] = abs(entry_price - ema_low) * 10000
        features['EMA_Convergence_Divergence'] = 0  # Cannot calculate without history
        
        # Time features
        if 21 <= hour <= 23 or 0 <= hour <= 6:
            features['Trading_Session'] = 1  # Asian
        elif 7 <= hour <= 11:
            features['Trading_Session'] = 2  # London
        elif 12 <= hour <= 16:
            features['Trading_Session'] = 4  # Overlap
        else:
            features['Trading_Session'] = 3  # NY
            
        features['Day_Of_Week'] = day_of_week
        features['Hour_Of_Day'] = hour
        
        # Candle features (approximated)
        features['Entry_Candle_Size_Pips'] = 20  # Default estimate
        features['Entry_Candle_Body_Ratio'] = 0.7  # Default estimate
        features['Previous_Candle_Size_Pips'] = 20  # Default estimate
        features['Bars_Since_Relevant_EMA_Cross'] = 5  # Default estimate
        
        # Risk features
        features['Initial_SL_Distance_Pips'] = 50  # Default estimate
        features['Recent_Gann_Signal_Frequency'] = 3  # Default estimate
        
        # Derived features
        features['Is_Asian_Session'] = 1 if features['Trading_Session'] == 1 else 0
        features['Is_London_Session'] = 1 if features['Trading_Session'] == 2 else 0
        features['Is_NY_Session'] = 1 if features['Trading_Session'] == 3 else 0
        features['Is_Overlap_Session'] = 1 if features['Trading_Session'] == 4 else 0
        features['Is_Buy_Signal'] = 1 if signal_type == 1 else 0
        features['Is_Sell_Signal'] = 1 if signal_type == -1 else 0
        features['High_Spread'] = 1 if spread > 0.0002 else 0
        features['Low_Spread'] = 1 if spread < 0.00005 else 0
        features['Gann_EMA_Alignment'] = 1  # Default to aligned
        features['Momentum_Alignment'] = 1  # Default to aligned
        
        return features
    
    def _get_default_feature_value(self, feature_name):
        """Get default value for missing features"""
        defaults = {
            'Gann_Zone_Width_Pips': 50,
            'Gann_Zone_Slope': 0,
            'Price_Distance_From_Zone_Center': 0,
            'Price_Distance_From_Zone_Boundary': 25,
            'Active_Gann_Line_Slope': 0,
            'Bars_Since_Last_Gann_Color_Change': 10,
            'EMA_Separation_Pips': 20,
            'Median_EMA_Slope': 0,
            'Low_EMA_Slope': 0,
            'Price_Distance_From_Relevant_EMA': 30,
            'EMA_Convergence_Divergence': 0,
            'Trading_Session': 2,
            'Day_Of_Week': 2,
            'Hour_Of_Day': 12,
            'Entry_Candle_Size_Pips': 20,
            'Entry_Candle_Body_Ratio': 0.7,
            'Previous_Candle_Size_Pips': 20,
            'Bars_Since_Relevant_EMA_Cross': 5,
            'Initial_SL_Distance_Pips': 50,
            'Recent_Gann_Signal_Frequency': 3,
            'Is_Asian_Session': 0,
            'Is_London_Session': 1,
            'Is_NY_Session': 0,
            'Is_Overlap_Session': 0,
            'Is_Buy_Signal': 1,
            'Is_Sell_Signal': 0,
            'High_Spread': 0,
            'Low_Spread': 0,
            'Gann_EMA_Alignment': 1,
            'Momentum_Alignment': 1
        }
        return defaults.get(feature_name, 0)
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.model is None:
            return {"status": "No model loaded"}
        
        return {
            "algorithm": self.algorithm,
            "features_count": len(self.feature_columns),
            "feature_names": self.feature_columns,
            "performance_metrics": self.performance_metrics,
            "predictions_made": len(self.prediction_history)
        }
    
    def get_prediction_history(self, last_n=10):
        """Get recent prediction history"""
        return self.prediction_history[-last_n:] if self.prediction_history else []