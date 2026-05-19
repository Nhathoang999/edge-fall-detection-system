"""Load .npy skeleton sequences from split folders."""
from __future__ import annotations

import os
from glob import glob
from pathlib import Path

import numpy as np

import config
from src.skeleton import normalize_skeleton


def expected_shape() -> tuple[int, int]:
    return config.INPUT_TIMESTEPS, config.NUM_FEATURES


def load_dataset(
    data_path: str | Path,
    normalize: bool = True,
    min_confidence: float = config.MIN_KEYPOINT_CONFIDENCE,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    data_path = Path(data_path)
    exp_shape = expected_shape()
    x_list: list[np.ndarray] = []
    y_list: list[int] = []
    paths: list[str] = []

    print(f"Loading from {data_path}, expected shape {exp_shape}")
    for label_name, label_val in [("no_fall", 0), ("fall", 1)]:
        folder = data_path / label_name
        if not folder.is_dir():
            print(f"  Warning: missing folder {folder}")
            continue
        files = sorted(glob(str(folder / "*.npy")))
        loaded = 0
        for fp in files:
            try:
                arr = np.load(fp)
            except Exception as e:
                print(f"  Warning: cannot load {fp}: {e}")
                continue
            if arr.shape != exp_shape:
                print(f"  Warning: skip {fp} shape {arr.shape} != {exp_shape}")
                continue
            if normalize:
                arr = normalize_skeleton(arr, min_confidence=min_confidence)
                if np.isnan(arr).any():
                    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            x_list.append(arr.astype(np.float32))
            y_list.append(label_val)
            paths.append(fp)
            loaded += 1
        print(f"  {label_name}: {loaded}/{len(files)} sequences")

    if not x_list:
        return np.array([]), np.array([]), []
    return np.stack(x_list), np.array(y_list, dtype=np.float32), paths
