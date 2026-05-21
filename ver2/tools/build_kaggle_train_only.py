import json
from pathlib import Path

def create_notebook():
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.10.12"}
        },
        "nbformat": 4, "nbformat_minor": 5
    }

    def add_markdown(text):
        notebook["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.split("\n")]})

    def add_code(code):
        cleaned_lines = []
        for line in code.split("\n"):
            if "import config" in line or "from src." in line or "sys.path.insert" in line:
                continue
            cleaned_lines.append(line + "\n")
        notebook["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cleaned_lines})

    root = Path(__file__).resolve().parents[1]
    
    config_code = (root / "config.py").read_text(encoding="utf-8")
    keypoints_code = (root / "src/keypoints.py").read_text(encoding="utf-8")
    skeleton_code = (root / "src/skeleton.py").read_text(encoding="utf-8")
    data_loader_code = (root / "src/data_loader.py").read_text(encoding="utf-8")
    transformer_code = (root / "src/models/transformer.py").read_text(encoding="utf-8")
    lstm_code = (root / "src/models/lstm.py").read_text(encoding="utf-8")
    metrics_code = (root / "src/metrics.py").read_text(encoding="utf-8")
    export_tflite_code = (root / "src/export_tflite.py").read_text(encoding="utf-8")
    train_code = (root / "scripts/train.py").read_text(encoding="utf-8")

    add_markdown("# BƯỚC 3: HUẤN LUYỆN MÔ HÌNH TỪ TẬP DATA ĐÃ CHIA\n\nNotebook này chỉ tập trung vào việc Train Model dựa trên Output của Notebook trước.")
    
    add_code("!pip install -q scikit-learn pandas tf-keras")
    
    add_code("import json\nimport sys\nimport os\nfrom pathlib import Path\nimport numpy as np\nimport pandas as pd\nimport tensorflow as tf\nfrom tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau\nfrom sklearn.metrics import accuracy_score, classification_report, confusion_matrix, fbeta_score, precision_recall_fscore_support")
    
    add_markdown("## 2. Cấu hình hệ thống (Config)")
    # Ở đây chúng ta ghi đè cấu hình để đọc data từ thư mục Input của Kaggle
    config_class = config_code.replace(
        'ROOT = Path(__file__).resolve().parent',
        'ROOT = Path("/kaggle/working")\n'
        '# ĐƯỜNG DẪN TỚI THƯ MỤC SPLITS CỦA NOTEBOOK TRƯỚC (THAY ĐỔI DÒNG DƯỚI NẾU CẦN)\n'
        'DATA_SPLITS = Path("/kaggle/input/kaggle-full-pipeline/data/splits")'
    )
    # Xóa định nghĩa cũ ở bên dưới để tránh bị ghi đè
    config_class = config_class.replace('DATA_RAW = ROOT / "data" / "raw"', '')
    config_class = config_class.replace('DATA_SPLITS = ROOT / "data" / "splits"', '')
    config_class = "class config:\n" + "\n".join(["    " + line for line in config_class.split("\n") if "import " not in line])
    add_code(config_class)
    
    add_markdown("## 3. Khai báo các Hàm (Keypoints, Skeleton, Dataloader, Model, Metrics)")
    add_code(keypoints_code)
    add_code(skeleton_code)
    add_code(data_loader_code)
    add_code(lstm_code)
    add_code(transformer_code)
    add_code(metrics_code)
    add_code(export_tflite_code)
    
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
    
    add_markdown("## 4. CHẠY HUẤN LUYỆN (EXECUTION)")
    
    exec_code = """
import os
config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
config.REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Tự động dò tìm đường dẫn thực tế của thư mục splits trong /kaggle/input
found_splits = None
input_base = Path("/kaggle/input")
if input_base.exists():
    for root, dirs, files in os.walk(input_base):
        # Nếu thư mục hiện tại tên là splits và bên trong có thư mục train
        if Path(root).name == "splits" and "train" in dirs:
            found_splits = Path(root)
            break

if found_splits:
    config.DATA_SPLITS = found_splits
    print(f"Đã TỰ ĐỘNG tìm thấy dữ liệu tại: {config.DATA_SPLITS}")
else:
    print(f"Đang đọc dữ liệu mặc định từ: {config.DATA_SPLITS}")

if not config.DATA_SPLITS.exists():
    print(f"\\nLỖI: Không tìm thấy thư mục {config.DATA_SPLITS}!")
    print("Vui lòng click 'Add Input' trên Kaggle, chọn Notebook cũ của bạn, sau đó copy đường dẫn của thư mục data/splits paste vào biến DATA_SPLITS ở phần Config phía trên.")
else:
    # Load dataset
    x_train, y_train, _ = load_dataset(config.DATA_SPLITS / "train")
    x_val, y_val, _ = load_dataset(config.DATA_SPLITS / "val")
    x_test, y_test, test_paths = load_dataset(config.DATA_SPLITS / "test")

    print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

    # Chọn 'lstm' hoặc 'transformer'
    MODEL_TYPE = 'transformer'

    if MODEL_TYPE == "lstm":
        model = create_lstm_classifier()
        model_tag = "lstm"
    else:
        model = create_transformer_classifier()
        model_tag = "transformer"
        
    keras_path = config.ARTIFACTS_DIR / f"{model_tag}_best.keras"
    model = train_model(model, x_train, y_train, x_val, y_val, model_tag, config.EPOCHS, config.BATCH_SIZE)
    model.save(keras_path)

    val_probs = model.predict(x_val, verbose=0).reshape(-1)
    best_val = find_best_threshold(y_val, val_probs, metric="f2_fall")
    print(f"\\nNgưỡng tốt nhất trên VAL: {best_val.threshold:.2f}")
    print(best_val.report)

    if len(x_test) > 0:
        test_probs = model.predict(x_test, verbose=0).reshape(-1)
        test_res = evaluate_at_threshold(y_test, test_probs, best_val.threshold)
        print(f"\\nKẾT QUẢ TEST @ threshold {best_val.threshold:.2f}:")
        print(test_res.report)

    if MODEL_TYPE == "transformer":
        tflite_path = export_to_tflite(model)
        verify_tflite(tflite_path)
        print(f"TFLite exported successfully to: {tflite_path}")
"""
    add_code(exec_code)
    
    out_path = root / "kaggle_train_only.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print(f"Created training notebook at {out_path}")

if __name__ == "__main__":
    create_notebook()
