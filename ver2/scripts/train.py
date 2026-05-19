#!/usr/bin/env python3
"""
Train fall classifier (LSTM baseline or Transformer) on video-level split data.

Examples:
  python scripts/train.py --model lstm
  python scripts/train.py --model transformer --data-dir data/splits
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src.data_loader import load_dataset
from src.export_tflite import export_to_tflite, verify_tflite
from src.metrics import evaluate_at_threshold, find_best_threshold, metrics_summary_row
from src.models.lstm import create_lstm_classifier
from src.models.transformer import create_transformer_classifier


def train_model(
    model: tf.keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    model_tag: str,
    epochs: int,
    batch_size: int,
) -> tf.keras.Model:
    callbacks = [
        EarlyStopping(
            monitor="val_f1_macro",
            patience=config.EARLY_STOPPING_PATIENCE,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_f1_macro",
            factor=0.5,
            patience=5,
            mode="max",
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    y_train_2d = y_train.reshape(-1, 1)
    y_val_2d = y_val.reshape(-1, 1)
    history = model.fit(
        x_train,
        y_train_2d,
        validation_data=(x_val, y_val_2d),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    hist_path = config.ARTIFACTS_DIR / f"{model_tag}_history.json"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    hist_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("lstm", "transformer"), required=True)
    parser.add_argument("--data-dir", type=Path, default=config.DATA_SPLITS)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--no-export-tflite", action="store_true")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        print(f"ERROR: data dir not found: {data_dir}")
        print("Run: python tools/build_video_split.py --source data/raw")
        sys.exit(1)

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    x_train, y_train, _ = load_dataset(data_dir / "train")
    x_val, y_val, _ = load_dataset(data_dir / "val")
    x_test, y_test, test_paths = load_dataset(data_dir / "test")

    if len(x_train) == 0 or len(x_val) == 0:
        print("ERROR: train or val set is empty.")
        sys.exit(1)

    print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

    if args.model == "lstm":
        model = create_lstm_classifier()
        model_tag = "lstm"
        keras_path = config.ARTIFACTS_DIR / "lstm_best.keras"
    else:
        model = create_transformer_classifier()
        model_tag = "transformer"
        keras_path = config.ARTIFACTS_DIR / "transformer_best.keras"

    model = train_model(
        model, x_train, y_train, x_val, y_val, model_tag, args.epochs, args.batch_size
    )
    model.save(keras_path)
    print(f"Saved Keras model: {keras_path}")

    val_probs = model.predict(x_val, verbose=0).reshape(-1)
    best_val = find_best_threshold(y_val, val_probs, metric="f2_fall")
    print(f"\nBest threshold on VAL: {best_val.threshold:.2f}")
    print(f"  Recall(fall)={best_val.recall_fall:.4f} Precision(fall)={best_val.precision_fall:.4f}")
    print(f"  F1={best_val.f1_fall:.4f} F2={best_val.f2_fall:.4f}")
    print(best_val.report)

    threshold_path = config.ARTIFACTS_DIR / f"{model_tag}_threshold.json"
    threshold_path.write_text(
        json.dumps({"threshold": best_val.threshold, "metric": "f2_fall"}, indent=2),
        encoding="utf-8",
    )

    if len(x_test) > 0:
        test_probs = model.predict(x_test, verbose=0).reshape(-1)
        test_res = evaluate_at_threshold(y_test, test_probs, best_val.threshold)
        print(f"\nTEST @ threshold {best_val.threshold:.2f}:")
        print(test_res.report)
        summary = metrics_summary_row(model_tag, "test", test_res)
        import pandas as pd

        pd.DataFrame([summary]).to_csv(
            config.REPORT_DIR / f"{model_tag}_test_metrics.csv", index=False
        )

    if args.model == "transformer" and not args.no_export_tflite:
        tflite_path = export_to_tflite(model)
        verify_tflite(tflite_path)
        deploy_copy = ROOT / "deploy" / config.TFLITE_MODEL_NAME
        deploy_copy.write_bytes(tflite_path.read_bytes())
        print(f"Copied TFLite to deploy: {deploy_copy}")

        deploy_threshold = ROOT / "deploy" / "threshold.json"
        deploy_threshold.write_text(threshold_path.read_text(encoding="utf-8"), encoding="utf-8")

    print("Training complete.")


if __name__ == "__main__":
    main()
