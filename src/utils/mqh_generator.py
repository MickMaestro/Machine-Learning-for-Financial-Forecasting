'''
================================================================================
MQH HEADER GENERATOR - ML MODEL TO MQL5 CONVERTER
================================================================================

OVERVIEW:
This module converts trained Python ML models into MQL5 header files for
integration with MetaTrader 5 Expert Advisors. It bridges the gap between
Python-based machine learning and real-time trading execution.

HOW THE CONVERSION WORKS:
The trained models consume ~40 engineered features (20 base features plus
derived/quantile features), scaled with a StandardScaler. The MQL5 EA can only
compute the 20 base features (plus Spread and Signal_Type) in real time, and
it works with RAW (unscaled) values. The generator therefore:

1. Computes the model's predicted probabilities on the held-out test set,
   using the SAME preprocessing as training (scaler.transform).
2. Selects the top-N most important features, restricted to features the EA
   can actually calculate live (mapped BY NAME to the EA's 24-slot array).
3. Fits a linear regression from the RAW values of those features to the
   model's predicted probabilities. The linear fit's coefficients operate in
   raw-feature space, so the EA can apply them directly without a scaler.
4. Emits MQL5 code: confidence = clamp(bias + sum(coef*feature), 0, 1),
   compared against the model's optimal threshold.

The R^2 of the linear approximation is embedded in the generated header so
the fidelity of the simplification is visible. For an exact (non-approximated)
port of the models, see the ONNX_UPDATE plan in the project root.

CRITICAL CONSISTENCY RULES (must match ML_FF_AI.mq5 calculateMLFeatures):
- Feature indices refer to the EA's 24-element features[] array.
- Trading_Session encoding: 1=Asian, 2=London(8-12), 3=NY(16-21), 4=Overlap(13-15).
- Pip values: price difference / (Point()*10).
================================================================================
'''

import os
import numpy as np
from datetime import datetime
from typing import List, Optional, Tuple
from sklearn.linear_model import LinearRegression
import joblib


# The EA's features[] array layout - MUST match calculateMLFeatures() in
# ML_FF_AI.mq5. Index in this list == index in the EA's features[] array.
EA_FEATURE_NAMES = [
    "Gann_Zone_Width_Pips",              # 0
    "Gann_Zone_Slope",                   # 1
    "Price_Distance_From_Zone_Center",   # 2
    "Price_Distance_From_Zone_Boundary", # 3
    "Active_Gann_Line_Slope",            # 4
    "Bars_Since_Last_Gann_Color_Change", # 5
    "EMA_Separation_Pips",               # 6
    "Median_EMA_Slope",                  # 7
    "Low_EMA_Slope",                     # 8
    "Price_Distance_From_Relevant_EMA",  # 9
    "EMA_Convergence_Divergence",        # 10
    "Trading_Session",                   # 11
    "Day_Of_Week",                       # 12
    "Hour_Of_Day",                       # 13
    "Entry_Candle_Size_Pips",            # 14
    "Entry_Candle_Body_Ratio",           # 15
    "Previous_Candle_Size_Pips",         # 16
    "Bars_Since_Relevant_EMA_Cross",     # 17
    "Initial_SL_Distance_Pips",          # 18
    "Recent_Gann_Signal_Frequency",      # 19
    "Spread",                            # 20
    "Signal_Type",                       # 21
    "Entry_Price",                       # 22 (not usable as model input)
    "Profit_Loss",                       # 23 (unknown at trade time - never usable)
]

# Features the EA can genuinely calculate at signal time. Entry_Price is
# excluded because its absolute magnitude is non-stationary; Profit_Loss is
# excluded because it is the label.
EA_USABLE_FEATURES = {name: idx for idx, name in enumerate(EA_FEATURE_NAMES[:22])}


class MQHGenerator:
    '''Generate MQL5 header files from trained ML models'''

    def __init__(self, model_path: str, symbol: str, timeframe: str):
        '''
        Initialise MQH generator

        Args:
            model_path: Path to saved model file
            symbol: Trading symbol (e.g., GER40)
            timeframe: Timeframe (e.g., H1)
        '''
        self.model_path = model_path
        self.symbol = symbol
        self.timeframe = timeframe
        self.model_data = None
        self.algorithm = None
        self.used_fallback = False
        self.approximation_r2 = None

    def load_model(self) -> bool:
        '''Load the trained model'''
        try:
            self.model_data = joblib.load(self.model_path)
            self.algorithm = self.model_data.get('algorithm', 'Unknown')
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_trained_columns(self) -> List[str]:
        '''Names of the columns the model was trained on, in order.'''
        cols = self.model_data.get('feature_columns') or []
        return list(cols)

    def _get_raw_X_test(self) -> Optional[np.ndarray]:
        '''Return the test-set feature matrix as a RAW (unscaled) 2D array
        aligned with feature_columns, or None if unavailable.'''
        X_test = self.model_data.get('X_test')
        if X_test is None:
            return None

        if hasattr(X_test, 'values'):  # pandas DataFrame - stored unscaled
            return np.asarray(X_test.values, dtype=float)

        X_test = np.asarray(X_test)
        if X_test.ndim == 2:
            return X_test.astype(float)

        if X_test.ndim == 3:
            # LSTM sequences (samples, timesteps, features), stored SCALED.
            # Recover raw values of the last timestep via the feature scaler.
            last_step = X_test[:, -1, :]
            scaler = self.model_data.get('feature_scaler') or self.model_data.get('scaler')
            if scaler is not None:
                try:
                    return scaler.inverse_transform(last_step)
                except Exception:
                    pass
            return last_step.astype(float)

        return None

    def _get_model_probabilities(self, X_raw: np.ndarray) -> Optional[np.ndarray]:
        '''Predicted probability of the positive class on the test set.

        Prefers probabilities stored at training time (guaranteed to have
        used the correct preprocessing). Otherwise recomputes them, applying
        the saved scaler first - the models were trained on scaled data, so
        predicting on raw values would produce meaningless outputs.
        '''
        stored = self.model_data.get('y_pred_proba')
        if stored is not None:
            stored = np.asarray(stored, dtype=float).flatten()
            if len(stored) == len(X_raw):
                return stored

        model = self.model_data.get('model')
        if model is None:
            return None

        scaler = self.model_data.get('scaler') or self.model_data.get('feature_scaler')
        X_in = X_raw
        if scaler is not None:
            try:
                X_in = scaler.transform(X_raw)
            except Exception as e:
                print(f"Warning: could not apply scaler ({e}); using raw features")

        try:
            if hasattr(model, 'predict_proba'):
                # sklearn API: RandomForest, XGBClassifier, VotingClassifier, SVC
                return model.predict_proba(X_in)[:, 1]
            # LightGBM Booster and Keras models expose predict() returning
            # positive-class probability directly
            preds = model.predict(X_in)
            return np.asarray(preds, dtype=float).flatten()
        except Exception as e:
            print(f"Warning: could not compute model predictions: {e}")
            return None

    def _rank_ea_features(self, X_raw: Optional[np.ndarray],
                          probabilities: Optional[np.ndarray],
                          n_features: int) -> List[Tuple[str, int, int, float]]:
        '''Rank EA-computable features by importance.

        Returns list of (name, trained_column_index, ea_index, importance).
        '''
        trained_cols = self._get_trained_columns()

        # Candidates: trained columns the EA can compute live, by name.
        candidates = [(name, col_idx, EA_USABLE_FEATURES[name])
                      for col_idx, name in enumerate(trained_cols)
                      if name in EA_USABLE_FEATURES]
        if not candidates:
            return []

        importance = self.model_data.get('feature_importance')
        scores = {}

        if importance is not None and len(importance) == len(trained_cols):
            importance = np.asarray(importance, dtype=float)
            for name, col_idx, ea_idx in candidates:
                scores[name] = float(abs(importance[col_idx]))
        elif X_raw is not None and probabilities is not None:
            # No native importance (SVM, LSTM): use |correlation| between each
            # raw feature and the model's predicted probability.
            for name, col_idx, ea_idx in candidates:
                col = X_raw[:, col_idx]
                if np.std(col) > 0 and np.std(probabilities) > 0:
                    scores[name] = float(abs(np.corrcoef(col, probabilities)[0, 1]))
                else:
                    scores[name] = 0.0
        else:
            # Nothing to rank with - uniform scores, keep column order
            for name, col_idx, ea_idx in candidates:
                scores[name] = 1.0

        ranked = sorted(candidates, key=lambda c: scores[c[0]], reverse=True)[:n_features]
        return [(name, col_idx, ea_idx, scores[name]) for name, col_idx, ea_idx in ranked]

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract_model_components(self, n_features: int = 10):
        '''
        Extract everything needed for the header: selected EA feature indices,
        coefficients in raw-feature space, bias, importances and names.
        '''
        X_raw = self._get_raw_X_test()
        probabilities = self._get_model_probabilities(X_raw) if X_raw is not None else None

        if X_raw is None or probabilities is None:
            print("Warning: no usable test data/predictions in model file - "
                  "falling back to heuristic coefficients. Retrain and re-save "
                  "the model with the updated trainers for a faithful header.")
            self.used_fallback = True
            return self._heuristic_fallback(n_features)

        selected = self._rank_ea_features(X_raw, probabilities, n_features)
        if not selected:
            print("Warning: none of the trained features are EA-computable - "
                  "falling back to heuristic coefficients.")
            self.used_fallback = True
            return self._heuristic_fallback(n_features)

        col_indices = [c[1] for c in selected]
        X_sel = X_raw[:, col_indices]

        # Guard against NaN/inf in the raw test matrix
        X_sel = np.nan_to_num(X_sel, nan=0.0, posinf=0.0, neginf=0.0)

        linear_model = LinearRegression()
        linear_model.fit(X_sel, probabilities)
        self.approximation_r2 = float(linear_model.score(X_sel, probabilities))
        print(f"Linear approximation R^2 vs model probabilities: {self.approximation_r2:.3f}")

        ea_indices = [c[2] for c in selected]
        names = [c[0] for c in selected]
        importances = [c[3] for c in selected]

        return ea_indices, names, importances, linear_model.coef_, float(linear_model.intercept_)

    def _heuristic_fallback(self, n_features: int = 10):
        '''Heuristic coefficients for legacy model files without test data.
        The generated header is clearly marked as NOT model-derived.'''
        fallback = [
            ("Gann_Zone_Width_Pips", 0.010),
            ("Price_Distance_From_Zone_Center", -0.005),
            ("EMA_Separation_Pips", 0.006),
            ("Median_EMA_Slope", 0.005),
            ("EMA_Convergence_Divergence", 0.002),
            ("Trading_Session", 0.020),
            ("Entry_Candle_Size_Pips", 0.002),
            ("Initial_SL_Distance_Pips", -0.001),
            ("Spread", -0.010),
            ("Signal_Type", 0.010),
        ][:n_features]

        ea_indices = [EA_USABLE_FEATURES[name] for name, _ in fallback]
        names = [name for name, _ in fallback]
        coeffs = np.array([c for _, c in fallback])
        importances = [1.0 - i * 0.1 for i in range(len(fallback))]
        bias = 0.4
        return ea_indices, names, importances, coeffs, bias

    def calculate_optimal_threshold(self) -> float:
        '''Optimal confidence threshold from training, clamped to sane range'''
        threshold = self.model_data.get('optimal_threshold', 0.5)
        return min(max(threshold, 0.3), 0.8)

    # ------------------------------------------------------------------
    # Code generation
    # ------------------------------------------------------------------

    def generate_mqh_content(self) -> str:
        '''Generate the complete MQH file content'''
        ea_indices, names, importances, coefficients, bias = self.extract_model_components(10)
        threshold = self.calculate_optimal_threshold()

        accuracy = self.model_data.get('accuracy', 0.0)
        precision = self.model_data.get('precision', 0.0)
        recall = self.model_data.get('recall', 0.0)
        f1_score = self.model_data.get('f1_score', 0.0)

        guard_name = f"{self.algorithm.upper().replace(' ', '_')}_{self.symbol}_{self.timeframe}_MODEL_MQH"
        n = len(ea_indices)

        fallback_warning = ""
        if self.used_fallback:
            fallback_warning = (
                "//\n"
                "// *** WARNING: HEURISTIC FALLBACK COEFFICIENTS ***\n"
                "// The model file lacked test data, so these coefficients are NOT\n"
                "// derived from the trained model. Retrain with the updated Python\n"
                "// pipeline and regenerate this header before trusting it live.\n"
                "//\n"
            )

        r2_line = ""
        if self.approximation_r2 is not None:
            r2_line = f"// Linear approximation fidelity (R^2 vs model probabilities): {self.approximation_r2:.4f}\n"

        content = f"""//+------------------------------------------------------------------+
//| {self.algorithm}_{self.symbol}_{self.timeframe}_Model.mqh       |
//| ML Model Header File Generated by Python Training System         |
//| Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}       |
//+------------------------------------------------------------------+
{fallback_warning}{r2_line}#ifndef {guard_name}
#define {guard_name}

// Model Metadata
#define MODEL_ALGORITHM "{self.algorithm}"
#define MODEL_SYMBOL "{self.symbol}"
#define MODEL_TIMEFRAME "{self.timeframe}"
#define MODEL_THRESHOLD {threshold:.6f}
#define MODEL_ACCURACY {accuracy:.4f}
#define MODEL_PRECISION {precision:.4f}
#define MODEL_RECALL {recall:.4f}
#define MODEL_F1_SCORE {f1_score:.4f}
#define TRAINING_DATE "{datetime.now().strftime('%Y-%m-%d')}"

// Indices into the EA's 24-element features[] array (see calculateMLFeatures)
#define NUM_FEATURES {n}
static int FEATURE_INDICES[{n}] = {{{', '.join(map(str, ea_indices))}}};

// Feature names for debugging (optional)
static string FEATURE_NAMES[{n}] = {{
{self._format_feature_names(names)}
}};

// Coefficients (raw feature space - apply directly to EA-calculated values)
static double FEATURE_COEFFS[{n}] = {{
{self._format_coefficients(coefficients, importances, names)}
}};

// Model bias term
static double MODEL_BIAS = {bias:.8f};

//+------------------------------------------------------------------+
//| Main ML confidence evaluation function                          |
//+------------------------------------------------------------------+
bool evaluateMLConfidence(double &features[])
{{
    return getConfidenceScore(features) >= MODEL_THRESHOLD;
}}

//+------------------------------------------------------------------+
//| Get actual confidence score (for debugging/analysis)            |
//+------------------------------------------------------------------+
double getConfidenceScore(double &features[])
{{
    double score = MODEL_BIAS;

    for(int i = 0; i < NUM_FEATURES; i++)
    {{
        int featureIdx = FEATURE_INDICES[i];
        if(featureIdx >= 0 && featureIdx < 24)
        {{
            score += features[featureIdx] * FEATURE_COEFFS[i];
        }}
    }}

    // The linear model approximates a probability - clamp to [0, 1]
    return MathMax(0.0, MathMin(1.0, score));
}}

//+------------------------------------------------------------------+
//| Get model information string                                    |
//+------------------------------------------------------------------+
string getModelInfo()
{{
    return StringFormat("Algorithm: %s | Symbol: %s | Timeframe: %s | Accuracy: %.2f%% | Threshold: %.3f",
                       MODEL_ALGORITHM, MODEL_SYMBOL, MODEL_TIMEFRAME,
                       MODEL_ACCURACY * 100, MODEL_THRESHOLD);
}}

#endif // {guard_name}
//+------------------------------------------------------------------+
"""
        return content

    def _format_feature_names(self, names: List[str]) -> str:
        lines = []
        for i, name in enumerate(names):
            comma = "," if i < len(names) - 1 else ""
            lines.append(f'    "{name}"{comma}')
        return '\n'.join(lines)

    def _format_coefficients(self, coeffs, importances, names) -> str:
        lines = []
        for i, (coeff, name) in enumerate(zip(coeffs, names)):
            comma = "," if i < len(names) - 1 else ""
            lines.append(f"    {coeff:.8f}{comma}  // {name} (importance: {importances[i]:.3f})")
        return '\n'.join(lines)

    def save_mqh_file(self, output_dir: str = "header_files") -> str:
        '''
        Save the MQH file to disk

        Args:
            output_dir: Directory to save the file

        Returns:
            Path to saved file
        '''
        os.makedirs(output_dir, exist_ok=True)

        filename = f"{self.algorithm.replace(' ', '')}_{self.symbol}_{self.timeframe}_Model.mqh"
        filepath = os.path.join(output_dir, filename)

        content = self.generate_mqh_content()
        with open(filepath, 'w') as f:
            f.write(content)

        print(f"MQH file generated: {filepath}")
        return filepath


def generate_mqh_from_model(model_path: str, symbol: str, timeframe: str, output_dir: str = "header_files") -> str:
    '''
    Convenience function to generate MQH file from a model

    Args:
        model_path: Path to saved model
        symbol: Trading symbol
        timeframe: Timeframe
        output_dir: Output directory

    Returns:
        Path to generated MQH file
    '''
    generator = MQHGenerator(model_path, symbol, timeframe)

    if not generator.load_model():
        raise ValueError(f"Failed to load model from {model_path}")

    return generator.save_mqh_file(output_dir)
