#!/usr/bin/env python3
"""Verify TFLite model input/output shapes and run a dummy inference."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src.export_tflite import verify_tflite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=config.ARTIFACTS_DIR / config.TFLITE_MODEL_NAME,
    )
    args = parser.parse_args()
    if not args.model.is_file():
        print(f"ERROR: model not found: {args.model}")
        sys.exit(1)

    info = verify_tflite(args.model)
    print("TFLite OK:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    interp = tf.lite.Interpreter(model_path=str(args.model))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    dummy = np.zeros(tuple(inp["shape"]), dtype=np.float32)
    interp.set_tensor(inp["index"], dummy)
    interp.invoke()
    prob = float(interp.get_tensor(out["index"])[0][0])
    print(f"Dummy inference P(fall) = {prob:.4f}")


if __name__ == "__main__":
    main()
