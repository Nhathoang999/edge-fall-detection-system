#!/usr/bin/env python3
"""
Evaluate a saved Keras model on val/test splits; export error analysis CSV.

  python scripts/evaluate.py --model artifacts/transformer_best.keras --tag transformer
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src.data_loader import load_dataset
from src.metrics import (
    error_analysis_df,
    evaluate_at_threshold,
    find_best_threshold,
    metrics_summary_row,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tag", type=str, default="model")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_SPLITS)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Use fixed threshold; if omitted, tune on val (F2 fall)",
    )
    args = parser.parse_args()

    if not args.model.is_file():
        print(f"ERROR: model not found: {args.model}")
        sys.exit(1)

    model = tf.keras.models.load_model(args.model, compile=False)
    data_dir = args.data_dir.resolve()

    rows = []
    threshold = args.threshold

    for split in ("val", "test"):
        x, y, paths = load_dataset(data_dir / split)
        if len(x) == 0:
            continue
        probs = model.predict(x, verbose=0).reshape(-1)
        if split == "val" and threshold is None:
            best = find_best_threshold(y, probs, metric="f2_fall")
            threshold = best.threshold
            print(f"Threshold from val: {threshold:.2f}")
            rows.append(metrics_summary_row(args.tag, "val", best))
        else:
            assert threshold is not None
            res = evaluate_at_threshold(y, probs, threshold)
            print(f"\n=== {split.upper()} @ {threshold:.2f} ===")
            print(res.report)
            rows.append(metrics_summary_row(args.tag, split, res))
            err = error_analysis_df(y, probs, paths, threshold, split)
            err_path = config.REPORT_DIR / f"{args.tag}_{split}_errors.csv"
            err.to_csv(err_path, index=False)
            print(f"Wrote {err_path} ({len(err[err['Type'].isin(['FP', 'FN'])])} errors)")

    if rows:
        out = config.REPORT_DIR / f"{args.tag}_metrics_summary.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"Wrote {out}")

    if threshold is not None:
        (config.ARTIFACTS_DIR / f"{args.tag}_threshold.json").write_text(
            json.dumps({"threshold": threshold}, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
