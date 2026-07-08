'''
Standalone sanity check: run an exported .onnx model over a signal CSV and
report the probability distribution and (if labels are derivable) the
classification metrics at the recommended threshold from model_meta.json.

Use this to re-validate a model file at any time - e.g. after moving it
between machines, or before dropping it into MQL5\\Files\\ML\\.

USAGE (from the project root):
    python ONNX_UPDATE/verify_onnx.py ^
        --model "ONNX_UPDATE/onnx_models/GER40/H1/RandomForest_GER40_H1.onnx" ^
        --csv   "data/signal/GER40test/GER40test_H1_signal_data.csv"
'''

import argparse
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_processing.feature_engineering import create_signal_labels
from src.data_processing.preprocessor import DataPreprocessor
from src.utils.csv_loader import RobustCSVLoader


def main():
    parser = argparse.ArgumentParser(description='Verify an exported ONNX model')
    parser.add_argument('--model', required=True, help='Path to .onnx file')
    parser.add_argument('--csv', required=True, help='Signal CSV to score')
    args = parser.parse_args()

    import onnxruntime as ort

    meta_path = os.path.join(os.path.dirname(args.model), 'model_meta.json')
    with open(meta_path) as f:
        meta = json.load(f)
    contract = meta['feature_order']
    threshold = meta['recommended_threshold']
    print(f"Model: {meta['algorithm']} {meta['symbol']} {meta['timeframe']} "
          f"| {meta['n_features']} features | threshold {threshold}")

    loader = RobustCSVLoader()
    ok, df, info = loader.load_csv_intelligent(args.csv)
    if not ok:
        raise ValueError(f"Failed to load {args.csv}: {info['errors']}")
    pre = DataPreprocessor()
    df = pre.clean_data(df)
    df = pre.prepare_timestamps(df)

    missing = [c for c in contract if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing contract features: {missing}")
    X = df[contract].astype(float).fillna(0.0).values.astype(np.float32)

    sess = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    proba = sess.run([meta['output_name']], {meta['input_name']: X})[0][:, 1]

    executed = (proba >= threshold)
    print(f"Scored {len(proba)} signals: "
          f"P(profitable) min={proba.min():.3f} median={np.median(proba):.3f} "
          f"max={proba.max():.3f}")
    print(f"Would EXECUTE {executed.sum()} / {len(proba)} signals "
          f"({executed.mean():.1%}) at threshold {threshold}")

    if 'Profit_Loss' in df.columns:
        from sklearn.metrics import f1_score, precision_score, recall_score
        y = create_signal_labels(df)['Signal_Quality'].astype(int).values
        pred = executed.astype(int)
        print(f"vs actual outcomes (WHOLE file incl. training rows - "
              f"optimistic, use only as a smoke test):")
        print(f"  precision={precision_score(y, pred, zero_division=0):.3f} "
              f"recall={recall_score(y, pred, zero_division=0):.3f} "
              f"f1={f1_score(y, pred, zero_division=0):.3f}")


if __name__ == '__main__':
    main()
