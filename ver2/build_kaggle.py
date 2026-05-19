import json
import re
from pathlib import Path

def create_notebook():
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    def add_markdown(text):
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.split("\n")]
        })

    def add_code(code):
        # Remove relative imports and config imports
        cleaned_lines = []
        for line in code.split("\n"):
            if "import config" in line or "from src." in line or "sys.path.insert" in line:
                continue
            cleaned_lines.append(line + "\n")
            
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": cleaned_lines
        })

    # Read files
    root = Path(__file__).resolve().parent
    
    config_code = (root / "config.py").read_text(encoding="utf-8")
    data_loader_code = (root / "src/data_loader.py").read_text(encoding="utf-8")
    transformer_code = (root / "src/models/transformer.py").read_text(encoding="utf-8")
    lstm_code = (root / "src/models/lstm.py").read_text(encoding="utf-8")
    metrics_code = (root / "src/metrics.py").read_text(encoding="utf-8")
    export_tflite_code = (root / "src/export_tflite.py").read_text(encoding="utf-8")
    
    # Train code needs more modification to run straight through instead of argparse
    train_code = (root / "scripts/train.py").read_text(encoding="utf-8")
    
    # Replace argparse in train_code with hardcoded values for Kaggle
    train_main_block = """
# KAGGLE TRAINING EXECUTION
print("Loading datasets from Kaggle input path...")
# Change this path to match your Kaggle dataset path after uploading
KAGGLE_DATA_DIR = Path("/kaggle/input/fall-dataset6/splits")

# If dataset is uploaded directly:
if not KAGGLE_DATA_DIR.exists():
    print(f"Warning: {KAGGLE_DATA_DIR} not found. Please update KAGGLE_DATA_DIR to point to your dataset splits.")
else:
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    x_train, y_train, _ = load_dataset(KAGGLE_DATA_DIR / "train")
    x_val, y_val, _ = load_dataset(KAGGLE_DATA_DIR / "val")
    x_test, y_test, test_paths = load_dataset(KAGGLE_DATA_DIR / "test")

    print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

    # Choose model: 'transformer' or 'lstm'
    MODEL_TYPE = 'transformer'

    if MODEL_TYPE == "lstm":
        model = create_lstm_classifier()
        model_tag = "lstm"
        keras_path = config.ARTIFACTS_DIR / "lstm_best.keras"
    else:
        model = create_transformer_classifier()
        model_tag = "transformer"
        keras_path = config.ARTIFACTS_DIR / "transformer_best.keras"

    model = train_model(
        model, x_train, y_train, x_val, y_val, model_tag, config.EPOCHS, config.BATCH_SIZE
    )
    model.save(keras_path)
    print(f"Saved Keras model: {keras_path}")

    val_probs = model.predict(x_val, verbose=0).reshape(-1)
    best_val = find_best_threshold(y_val, val_probs, metric="f2_fall")
    print(f"\\nBest threshold on VAL: {best_val.threshold:.2f}")
    print(best_val.report)

    threshold_path = config.ARTIFACTS_DIR / f"{model_tag}_threshold.json"
    threshold_path.write_text(
        json.dumps({"threshold": best_val.threshold, "metric": "f2_fall"}, indent=2),
        encoding="utf-8",
    )

    if len(x_test) > 0:
        test_probs = model.predict(x_test, verbose=0).reshape(-1)
        test_res = evaluate_at_threshold(y_test, test_probs, best_val.threshold)
        print(f"\\nTEST @ threshold {best_val.threshold:.2f}:")
        print(test_res.report)

    if MODEL_TYPE == "transformer":
        tflite_path = export_to_tflite(model)
        verify_tflite(tflite_path)
        print(f"TFLite exported successfully to: {tflite_path}")
"""

    add_markdown("# Fall Detection - Kaggle Training Pipeline\n\nThis notebook contains the complete pipeline for training the Fall Detection model on Kaggle.")
    
    add_markdown("## 1. Imports and Installation")
    add_code("!pip install -q scikit-learn pandas tf-keras")
    add_code("import json\nimport sys\nimport os\nfrom pathlib import Path\nimport numpy as np\nimport tensorflow as tf\nfrom tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau\nfrom sklearn.metrics import classification_report, confusion_matrix\nimport pandas as pd")
    
    # Create a config class instead of config.py to emulate the module
    config_code = config_code.replace('ROOT = Path(__file__).resolve().parent', 'ROOT = Path("/kaggle/working")\nDATA_SPLITS = Path("/kaggle/input/fall-dataset6/splits")')
    add_markdown("## 2. Configuration")
    # Wrap config code into a class config
    config_class = "class config:\n" + "\n".join(["    " + line for line in config_code.split("\n") if "import " not in line])
    add_code(config_class)
    
    add_markdown("## 3. Data Loader")
    add_code(data_loader_code)
    
    add_markdown("## 4. Models (LSTM & Transformer)")
    add_code(lstm_code)
    add_code(transformer_code)
    
    add_markdown("## 5. Metrics & Export TFLite")
    add_code(metrics_code)
    add_code(export_tflite_code)
    
    add_markdown("## 6. Training Functions")
    # Extract train_model function from train.py
    train_model_func = []
    in_train_model = False
    for line in train_code.split("\n"):
        if line.startswith("def train_model("):
            in_train_model = True
        if in_train_model:
            train_model_func.append(line)
            if line == "    return model":
                in_train_model = False
                break
                
    add_code("\n".join(train_model_func))
    
    add_markdown("## 7. Execution (Train & Evaluate)")
    add_code(train_main_block)
    
    out_path = root / "kaggle_train_pipeline.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    
    print(f"Created notebook at {out_path}")

if __name__ == "__main__":
    create_notebook()
