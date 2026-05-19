"""Export Keras model to TFLite."""
from __future__ import annotations

from pathlib import Path

import tensorflow as tf

import config


def export_to_tflite(
    model: tf.keras.Model,
    export_dir: Path | None = None,
    tflite_path: Path | None = None,
) -> Path:
    export_dir = export_dir or (config.ARTIFACTS_DIR / config.SAVED_MODEL_DIR)
    tflite_path = tflite_path or (config.ARTIFACTS_DIR / config.TFLITE_MODEL_NAME)

    export_dir = Path(export_dir)
    tflite_path = Path(tflite_path)
    export_dir.mkdir(parents=True, exist_ok=True)
    tflite_path.parent.mkdir(parents=True, exist_ok=True)

    model.export(str(export_dir))
    converter = tf.lite.TFLiteConverter.from_saved_model(str(export_dir))
    tflite_bytes = converter.convert()
    tflite_path.write_bytes(tflite_bytes)
    size_kb = len(tflite_bytes) / 1024
    print(f"Saved TFLite: {tflite_path} ({size_kb:.2f} KB)")
    return tflite_path


def verify_tflite(tflite_path: Path) -> dict:
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    info = {
        "input_shape": inp["shape"],
        "input_dtype": str(inp["dtype"]),
        "output_shape": out["shape"],
        "output_dtype": str(out["dtype"]),
    }
    expected = [1, config.INPUT_TIMESTEPS, config.NUM_FEATURES]
    if list(inp["shape"]) != expected:
        raise ValueError(f"TFLite input {inp['shape']} != expected {expected}")
    return info
