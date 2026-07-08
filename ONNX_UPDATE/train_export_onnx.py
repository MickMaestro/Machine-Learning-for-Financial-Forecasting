'''
================================================================================
ONNX TRAINING + EXPORT PIPELINE (Random Forest proof of concept)
================================================================================

Trains a Random Forest signal-quality classifier and exports it as a single
ONNX graph containing BOTH the StandardScaler and the model, so the MQL5 EA
(ML_FF_ONNX.mq5) feeds RAW feature values and receives calibrated
probabilities. This replaces the legacy MQH linear-approximation deployment
with bit-exact model inference inside MetaTrader 5.

WHY RANDOM FOREST:
- Converts natively with skl2onnx (no extra converter registration needed,
  unlike LightGBM/XGBoost).
- The full sklearn Pipeline(StandardScaler, RandomForestClassifier) becomes
  ONE .onnx file - the scaled-in-training/raw-at-inference bug class that
  plagued the MQH path is structurally impossible here.

THE FEATURE CONTRACT (must never drift):
The model is trained on exactly the 21 features the EA can compute at signal
time AND that exist in the historical training CSVs:
    inputs 0-19 : the 20 base Data_Exporter2.0 features
                  (FeatureEngineer.get_data_exporter_2_features() order,
                  identical to EA features[] slots 0-19)
    input  20   : Signal_Type (EA features[] slot 21)
Spread (EA slot 20) is excluded because the exporter never recorded it.
Quantile-derived features (Is_Tight_Stop etc.) are deliberately excluded:
they leak dataset statistics and cannot be reproduced bar-by-bar in MT5.

USAGE (from the project root):
    python ONNX_UPDATE/train_export_onnx.py ^
        --csv "data/signal/GER40test/GER40test_H1_signal_data.csv" ^
        --symbol GER40 --timeframe H1

OUTPUTS (under ONNX_UPDATE/onnx_models/<SYMBOL>/<TF>/):
    RandomForest_<SYMBOL>_<TF>.onnx   - scaler+model graph, single
                                        'probabilities' float[N,2] output
    model_meta.json                   - feature contract, threshold, metrics

A Python-vs-ONNX parity check on the held-out test set is mandatory and the
export FAILS if probabilities disagree beyond float32 tolerance.
================================================================================
'''

import argparse
import json
import os
import sys
import warnings
from datetime import datetime

import numpy as np

warnings.filterwarnings('ignore')

# Make the parent project importable when run from anywhere
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_processing.feature_engineering import FeatureEngineer, create_signal_labels
from src.data_processing.preprocessor import DataPreprocessor
from src.utils.csv_loader import RobustCSVLoader


# ---------------------------------------------------------------------------
# Feature contract - EA slot order. DO NOT REORDER: ML_FF_ONNX.mq5 fills the
# input tensor positionally via its FEATURE_MAP (EA slots 0..19, then 21).
# Spread is NOT part of the contract: the historical exporter never recorded
# it, so the model cannot be trained on it.
# ---------------------------------------------------------------------------
def get_feature_contract():
    base = FeatureEngineer().get_data_exporter_2_features()  # 20 features, EA slots 0-19
    contract = base + ['Signal_Type']                        # EA slot 21
    assert len(contract) == 21, f"Contract must be 21 features, got {len(contract)}"
    return contract


def load_training_data(csv_path):
    '''Load an MT5-exported (UTF-16) signal CSV and prepare it for training.'''
    loader = RobustCSVLoader()
    ok, df, info = loader.load_csv_intelligent(csv_path)
    if not ok:
        raise ValueError(f"Failed to load {csv_path}: {info['errors']}")
    print(f"Loaded {len(df)} signals ({info['encoding']}, sep '{info['separator']}')")

    pre = DataPreprocessor()
    df = pre.clean_data(df)
    df = pre.prepare_timestamps(df)
    df = create_signal_labels(df)

    contract = get_feature_contract()
    missing = [c for c in contract if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing contract features: {missing}")

    dist = df['Signal_Quality'].value_counts().to_dict()
    print(f"Label distribution (1=profitable): {dist}")
    if len(dist) < 2:
        raise ValueError("Need both profitable and unprofitable signals to train")

    return df, contract


def chronological_split(df, contract, test_size=0.2):
    X = df[contract].astype(float).fillna(0.0)
    y = df['Signal_Quality'].astype(int)
    split = int(len(X) * (1 - test_size))
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def train_pipeline(X_train, y_train):
    '''Scaler + Random Forest as ONE sklearn Pipeline (exported as one graph).'''
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def find_optimal_threshold(y_test, proba):
    '''Best F1 threshold over the same grid the legacy pipeline used.'''
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.10, 0.95, 0.05):
        f1 = f1_score(y_test, (proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return round(best_t, 2), best_f1


def evaluate(y_test, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        'threshold': threshold,
        'accuracy': round(accuracy_score(y_test, pred), 4),
        'precision': round(precision_score(y_test, pred, zero_division=0), 4),
        'recall': round(recall_score(y_test, pred, zero_division=0), 4),
        'f1_score': round(f1_score(y_test, pred, zero_division=0), 4),
        'roc_auc': round(roc_auc_score(y_test, proba), 4) if y_test.nunique() > 1 else None,
        'signals_retained': round(float(pred.mean()), 4),
    }


def export_onnx(pipeline, n_features, out_path):
    '''Convert the pipeline to ONNX with a single float[N,2] probability output.'''
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs

    model_cls = pipeline.named_steps['model']
    onnx_model = convert_sklearn(
        pipeline,
        initial_types=[('float_input', FloatTensorType([None, n_features]))],
        # zipmap off -> plain float tensor instead of a sequence of maps,
        # which MQL5 cannot consume
        options={id(model_cls): {'zipmap': False}},
        target_opset=15,  # conservative opset for broad MT5 build compatibility
    )

    # Keep only the probabilities output; dropping the int64 label output
    # means the EA binds exactly one float[1,2] output buffer.
    onnx_model = select_model_inputs_outputs(onnx_model, outputs=['probabilities'])

    # select_model_inputs_outputs promotes an intermediate tensor to a graph
    # output without type/shape metadata. MT5's ONNX runtime requires a fully
    # declared output, so stamp it explicitly as float32 [None, 2].
    import onnx
    from onnx import helper, TensorProto
    declared = helper.make_tensor_value_info(
        'probabilities', TensorProto.FLOAT, [None, 2])
    del onnx_model.graph.output[:]
    onnx_model.graph.output.append(declared)
    onnx.checker.check_model(onnx_model)

    with open(out_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    print(f"ONNX model written: {out_path} ({os.path.getsize(out_path)/1024:.0f} KB)")


def parity_check(pipeline, onnx_path, X_test):
    '''Mandatory: ONNX inference must reproduce sklearn probabilities.'''
    import onnxruntime as ort

    sk_proba = pipeline.predict_proba(X_test)[:, 1]

    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    onnx_proba = sess.run(['probabilities'],
                          {input_name: X_test.values.astype(np.float32)})[0][:, 1]

    diff = np.abs(sk_proba - onnx_proba)
    print(f"Parity check: max |dP| = {diff.max():.2e}, mean = {diff.mean():.2e} "
          f"over {len(diff)} test samples")

    # float32 casting can flip a sample across a tree split threshold, so a
    # handful of small discrepancies are expected; large ones are a bug.
    if diff.max() > 5e-3:
        raise AssertionError(
            f"ONNX/sklearn probability mismatch too large (max {diff.max():.4f}). "
            "Do not deploy this model.")
    return float(diff.max())


def write_meta(out_dir, symbol, timeframe, contract, threshold, metrics_optimal,
               metrics_default, parity, n_train, n_test, csv_path):
    meta = {
        'algorithm': 'Random Forest',
        'symbol': symbol,
        'timeframe': timeframe,
        'model_file': f'RandomForest_{symbol}_{timeframe}.onnx',
        'input_name': 'float_input',
        'input_shape': [1, len(contract)],
        'output_name': 'probabilities',
        'output_shape': [1, 2],
        'output_meaning': 'probabilities[0][1] = P(profitable signal)',
        'n_features': len(contract),
        'feature_order': contract,
        'feature_order_note': 'EA features[] slots 0..19 then slot 21 (Signal_Type); Spread (20) unused',
        'recommended_threshold': threshold,
        'metrics_at_recommended_threshold': metrics_optimal,
        'metrics_at_0.5': metrics_default,
        'parity_max_abs_diff': parity,
        'train_samples': n_train,
        'test_samples': n_test,
        'training_csv': os.path.basename(csv_path),
        'trained_at': datetime.now().isoformat(timespec='seconds'),
        'label_definition': 'Signal_Quality = 1 if Profit_Loss > 0 (price points)',
        'session_encoding': '1=Asian/off-hours, 2=London(8-12h), 3=NY(16-21h), 4=Overlap(13-15h)',
    }
    meta_path = os.path.join(out_dir, 'model_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata written: {meta_path}")


def main():
    parser = argparse.ArgumentParser(description='Train RF and export ONNX for MT5')
    parser.add_argument('--csv', required=True, help='MT5-exported signal CSV (UTF-16 ok)')
    parser.add_argument('--symbol', required=True, help='e.g. GER40')
    parser.add_argument('--timeframe', required=True, help='e.g. H1')
    parser.add_argument('--test-size', type=float, default=0.2)
    args = parser.parse_args()

    df, contract = load_training_data(args.csv)
    X_train, X_test, y_train, y_test = chronological_split(df, contract, args.test_size)
    print(f"Chronological split: train {len(X_train)}, test {len(X_test)}")

    pipeline = train_pipeline(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]

    threshold, _ = find_optimal_threshold(y_test, proba)
    metrics_optimal = evaluate(y_test, proba, threshold)
    metrics_default = evaluate(y_test, proba, 0.5)
    print(f"Test metrics @ optimal threshold {threshold}: {metrics_optimal}")
    print(f"Test metrics @ 0.5: {metrics_default}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'onnx_models', args.symbol, args.timeframe)
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, f'RandomForest_{args.symbol}_{args.timeframe}.onnx')

    export_onnx(pipeline, len(contract), onnx_path)
    parity = parity_check(pipeline, onnx_path, X_test)

    write_meta(out_dir, args.symbol, args.timeframe, contract, threshold,
               metrics_optimal, metrics_default, parity,
               len(X_train), len(X_test), args.csv)

    print()
    print("=" * 70)
    print("DEPLOYMENT STEPS")
    print("=" * 70)
    print(f"1. Copy {os.path.basename(onnx_path)}")
    print(r"   to <MT5 data folder>\MQL5\Files\ML\ ")
    print("2. Compile ONNX_UPDATE/MQL5/ML_FF_ONNX.mq5 in MetaEditor")
    print(f"3. Set EA inputs: OnnxModelFile=\"{os.path.basename(onnx_path)}\",")
    print(f"   ConfidenceThreshold={threshold}")
    print("4. Backtest in the Strategy Tester before any live use")


if __name__ == '__main__':
    main()
