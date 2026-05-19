"""Central configuration for ver2 fall-detection pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Data layout
DATA_RAW = ROOT / "data" / "raw"
DATA_SPLITS = ROOT / "data" / "splits"
MANIFEST_PATH = ROOT / "data" / "video_split_manifest.csv"
SPLIT_STATS_PATH = ROOT / "data" / "split_stats.json"

# Training artifacts
ARTIFACTS_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "report"

# Sequence / features (must match deploy/app.py)
INPUT_TIMESTEPS = 30
NUM_KEYPOINTS = 17
NUM_FEATURES = NUM_KEYPOINTS * 3  # x, y, visibility

KEYPOINT_NAMES = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle",
]
SORTED_KEYPOINT_NAMES = sorted(KEYPOINT_NAMES)
KEYPOINT_DICT = {name: i for i, name in enumerate(SORTED_KEYPOINT_NAMES)}

MIN_KEYPOINT_CONFIDENCE = 0.3

# Train/val/test ratios (video-level)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SPLIT_RANDOM_STATE = 42

# Transformer hyperparameters
NUM_ENCODER_BLOCKS = 3
D_MODEL = 64
NUM_HEADS = 4
FF_DIM = D_MODEL * 2
PROJECTION_DIM = D_MODEL
FINAL_DENSE_UNITS = 32
DROPOUT_RATE = 0.1
LEARNING_RATE = 5e-4

# LSTM baseline
LSTM_UNITS = 64
LSTM_DROPOUT = 0.2

# Training
BATCH_SIZE = 32
EPOCHS = 60
EARLY_STOPPING_PATIENCE = 15

# Inference / deployment
DEFAULT_THRESHOLD = 0.5
TFLITE_MODEL_NAME = "fall_detection_transformer.tflite"
SAVED_MODEL_DIR = "fall_model_exported_sm"
