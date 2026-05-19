# Dữ liệu thô (.npy)

Đặt toàn bộ sequence skeleton vào đây **trước khi chia tập**:

```
data/raw/
  fall/       ← *.npy nhãn fall
  no_fall/    ← *.npy nhãn no_fall
```

**Quy tắc tên file** (bắt buộc):

```
{video_id}_fall_seq_000.npy
{video_id}_no_fall_seq_001.npy
```

Ví dụ: `B_D_0176_no_fall_seq_002.npy` → `video_id = B_D_0176`

## Nguồn dữ liệu

1. **Gộp từ dataset cũ** (đã có train/val/test):

   ```bash
   python tools/build_video_split.py --source "D:/path/to/fall-dataset6" --source-mode nested
   ```

2. **Hoặc** copy thẳng vào `fall/` và `no_fall/` rồi:

   ```bash
   python tools/build_video_split.py --source data/raw --source-mode flat
   ```
