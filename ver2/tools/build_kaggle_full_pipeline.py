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
        # Remove relative imports
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

    root = Path(__file__).resolve().parents[1]
    
    config_code = (root / "config.py").read_text(encoding="utf-8")
    data_loader_code = (root / "src/data_loader.py").read_text(encoding="utf-8")
    transformer_code = (root / "src/models/transformer.py").read_text(encoding="utf-8")
    lstm_code = (root / "src/models/lstm.py").read_text(encoding="utf-8")
    metrics_code = (root / "src/metrics.py").read_text(encoding="utf-8")
    export_tflite_code = (root / "src/export_tflite.py").read_text(encoding="utf-8")
    train_code = (root / "scripts/train.py").read_text(encoding="utf-8")
    extract_code = (root / "tools/extract_features.py").read_text(encoding="utf-8")
    split_code = (root / "tools/build_video_split.py").read_text(encoding="utf-8")
    keypoints_code = (root / "src/keypoints.py").read_text(encoding="utf-8")
    skeleton_code = (root / "src/skeleton.py").read_text(encoding="utf-8")

    # Add all required Kaggle markdown and code cells
    add_markdown("# FULL PIPELINE: TỪ VIDEO GỐC ĐẾN TFLITE TRÊN KAGGLE\n\nNotebook này bao gồm toàn bộ quá trình: Trích xuất khung xương (MediaPipe) -> Chia tập dữ liệu (Video-level) -> Huấn luyện mô hình (LSTM/Transformer) -> Đánh giá & Xuất TFLite.")
    
    add_markdown("## 1. Cài đặt thư viện")
    add_code("!pip install -q scikit-learn pandas tf-keras mediapipe opencv-python-headless kagglehub")
    
    add_code("import json\nimport sys\nimport os\nfrom pathlib import Path\nimport numpy as np\nimport pandas as pd\nimport tensorflow as tf\nfrom tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau\nfrom sklearn.metrics import accuracy_score, classification_report, confusion_matrix, fbeta_score, precision_recall_fscore_support\nimport cv2\nimport mediapipe as mp\nfrom sklearn.model_selection import train_test_split\nimport shutil\nfrom collections import defaultdict")
    
    add_markdown("## 2. Cấu hình hệ thống (Config)")
    config_class = config_code.replace('ROOT = Path(__file__).resolve().parent', 'ROOT = Path("/kaggle/working")\nDATA_RAW = ROOT / "data" / "raw"\nDATA_SPLITS = ROOT / "data" / "splits"')
    config_class = "class config:\n" + "\n".join(["    " + line for line in config_class.split("\n") if "import " not in line])
    add_code(config_class)
    
    add_markdown("## 3. Trích xuất đặc trưng (MediaPipe)")
    # Extract only process_video from extract_code
    process_func = []
    in_process = False
    for line in extract_code.split("\n"):
        if line.startswith("def process_video("):
            in_process = True
        if in_process:
            if line.startswith("def main("):
                break
            process_func.append(line)
    import_mp_block = """
import mediapipe as mp
try:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing
"""
    add_code(import_mp_block + "\n" + "\n".join(process_func))
    
    add_markdown("## 4. Chia tập dữ liệu (Video Split)")
    split_funcs = []
    for line in split_code.split("\n"):
        if line.startswith("def main()"):
            break
        if "argparse" not in line and "import " not in line and "from __future__" not in line and "ROOT" not in line and "sys.path" not in line:
            split_funcs.append(line)
    add_code("\n".join(split_funcs))
    
    add_markdown("## 5. Dataloader & Models & Metrics & Export TFLite")
    add_code(keypoints_code)
    add_code(skeleton_code)
    add_code(data_loader_code)
    add_code(lstm_code)
    add_code(transformer_code)
    add_code(metrics_code)
    add_code(export_tflite_code)
    
    add_markdown("## 6. Huấn luyện Mô hình (Training Logic)")
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
    
    add_markdown("## 7. CHẠY TOÀN BỘ PIPELINE (EXECUTION)\nThay đổi đường dẫn thư mục chứa video gốc của bạn vào 2 biến `RAW_VIDEO_FALL_DIR` và `RAW_VIDEO_NO_FALL_DIR` ở bên dưới.")
    
    exec_code = """
# TẢI DATASET TỪ KAGGLEHUB
import kagglehub
dataset_path = kagglehub.dataset_download("payutch/fall-video-dataset")
print("Path to dataset files:", dataset_path)

# Thư mục gốc của video
RAW_VIDEO_FALL_DIR = Path(dataset_path) / "fall"
RAW_VIDEO_NO_FALL_DIR = Path(dataset_path) / "no_fall"

config.DATA_RAW.mkdir(parents=True, exist_ok=True)
config.DATA_SPLITS.mkdir(parents=True, exist_ok=True)
config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
config.REPORT_DIR.mkdir(parents=True, exist_ok=True)

# 7.1 TRÍCH XUẤT ĐẶC TRƯNG TỪ VIDEO (Bỏ qua nếu đã có file .npy)
print("--- BƯỚC 1: TRÍCH XUẤT KHUNG XƯƠNG (MEDIAPIPE) ---")
if RAW_VIDEO_FALL_DIR.exists() and RAW_VIDEO_NO_FALL_DIR.exists():
    out_fall = config.DATA_RAW / "fall"
    out_nofall = config.DATA_RAW / "no_fall"
    out_fall.mkdir(parents=True, exist_ok=True)
    out_nofall.mkdir(parents=True, exist_ok=True)
    
    for vid in RAW_VIDEO_FALL_DIR.glob("*.mp4"):
        process_video(str(vid), "fall", out_fall, config.INPUT_TIMESTEPS, 15)
    for vid in RAW_VIDEO_NO_FALL_DIR.glob("*.mp4"):
        process_video(str(vid), "no_fall", out_nofall, config.INPUT_TIMESTEPS, 15)
else:
    print(f"CẢNH BÁO: Không tìm thấy thư mục video thô. Vui lòng kiểm tra lại đường dẫn!")

# 7.2 CHIA TẬP DỮ LIỆU
print("\\n--- BƯỚC 2: CHIA TẬP DỮ LIỆU (VIDEO-LEVEL) ---")
try:
    records = collect_npy_files(config.DATA_RAW, "flat")
    video_label = assign_video_labels(records)
    video_ids = sorted(video_label.keys())
    labels = [video_label[v] for v in video_ids]
    assignment = split_videos(video_ids, labels, config.TRAIN_RATIO, config.VAL_RATIO, config.TEST_RATIO, config.SPLIT_RANDOM_STATE)
    
    rows = [{"split": assignment[r["video_id"]], **r} for r in records]
    manifest = pd.DataFrame(rows)
    check_leakage(manifest)
    manifest.to_csv(config.MANIFEST_PATH, index=False)
    
    copy_split_files(manifest, config.DATA_SPLITS, copy_files=True)
    print("Chia dữ liệu thành công! Đã sao chép vào data/splits.")
except Exception as e:
    print(f"Lỗi khi chia dữ liệu: {e}")

# 7.3 TRAIN MODEL
print("\\n--- BƯỚC 3: HUẤN LUYỆN MÔ HÌNH ---")
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
    
    out_path = root / "kaggle_full_pipeline.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    
    print(f"Created notebook at {out_path}")

if __name__ == "__main__":
    create_notebook()
