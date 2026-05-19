# Fall Detection — ver2 (video-level split + TFLite)

Pipeline mới theo hướng GV: **chia dữ liệu theo video**, baseline LSTM, Transformer, export TFLite, đánh giá metric chuẩn.

## Cấu trúc

```
ver2/
├── config.py                 # Hằng số chung (30×51, hyperparams)
├── data/
│   ├── raw/                  # .npy gốc (fall/, no_fall/)
│   ├── splits/               # train/val/test (tạo tự động)
│   ├── video_split_manifest.csv
│   └── split_stats.json
├── tools/
│   ├── build_video_split.py  # Chia video-level, chống leakage
│   └── verify_tflite.py
├── scripts/
│   ├── train.py              # Train LSTM hoặc Transformer
│   ├── evaluate.py           # Đánh giá + CSV lỗi FP/FN
│   └── run_pipeline.py       # Chạy full pipeline
├── src/                      # Data loader, models, metrics, export
├── artifacts/                # Model .keras, .tflite, threshold
├── report/                   # CSV metric & error analysis
└── deploy/                   # Gradio app + TFLite
```

## Bước 1 — Chuẩn bị dữ liệu

### Cách A: Dataset đã có cấu trúc train/val/test (fall-dataset6)

```powershell
cd c:\KLTN\THESIS_CS\ver2
pip install -r requirements.txt

python tools/build_video_split.py --source "D:\path\to\fall-dataset6" --source-mode nested
```

### Cách B: Đã gom file vào `data/raw`

```powershell
python tools/build_video_split.py --source data/raw --source-mode flat
```

Kiểm tra `data/split_stats.json` và `data/video_split_manifest.csv`.

## Bước 2 — Train

```powershell
# Baseline LSTM
python scripts/train.py --model lstm

# Transformer + export TFLite
python scripts/train.py --model transformer
```

Kết quả:

- `artifacts/lstm_best.keras`, `artifacts/transformer_best.keras`
- `artifacts/fall_detection_transformer.tflite`
- `artifacts/*_threshold.json` (ngưỡng tune trên **val**, metric F2 fall)
- `deploy/fall_detection_transformer.tflite`, `deploy/threshold.json`
- `report/*_test_metrics.csv`

## Bước 3 — Đánh giá lại (tùy chọn)

```powershell
python scripts/evaluate.py --model artifacts/transformer_best.keras --tag transformer
```

## Bước 4 — Chạy demo

```powershell
cd deploy
pip install -r requirements.txt
python app.py
```

## Full pipeline một lệnh

```powershell
python scripts/run_pipeline.py --source "D:\path\to\fall-dataset6" --source-mode nested
```

## Train trên Kaggle (GPU)

1. Upload thư mục `data/splits` lên Kaggle Dataset (hoặc chạy `build_video_split` trên notebook).
2. Upload folder `ver2/src`, `config.py`, `scripts/train.py`.
3. Trong notebook:

```python
!pip install -q scikit-learn pandas
%cd /kaggle/working
# copy ver2 files vào working
!python scripts/train.py --model transformer --data-dir /kaggle/input/your-dataset/splits
```

## Metric báo cáo

| Metric | Ý nghĩa |
|--------|---------|
| Recall (fall) | Ưu tiên — không bỏ sót ngã |
| Precision (fall) | Giảm báo động giả |
| F1 / F2 (fall) | F2 nhấn recall hơn |
| Confusion matrix | Phân tích FP/FN |

**Lưu ý:** Sau split video-level, accuracy/F1 có thể **thấp hơn ver1** — đó là kết quả đáng tin, dùng cho luận văn.

## So với ver1

| ver1 | ver2 |
|------|------|
| Split seq/folder cũ, rủi ro leakage | Split **video-level** + manifest |
| Notebook Kaggle rời rạc | Script `train.py` + `evaluate.py` tái lập được |
| Threshold cố định 0.90 | `threshold.json` từ val |
| Báo cáo CSV rải rác | `report/` thống nhất |
