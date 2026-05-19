"""Keypoint indexing consistent with training and TFLite deployment."""
from __future__ import annotations

import config


def get_kpt_indices(keypoint_name: str) -> tuple[int, int, int]:
    if keypoint_name not in config.KEYPOINT_DICT:
        raise ValueError(
            f"Keypoint '{keypoint_name}' not in KEYPOINT_DICT. "
            f"Available: {list(config.KEYPOINT_DICT.keys())}"
        )
    kp_idx = config.KEYPOINT_DICT[keypoint_name]
    return kp_idx * 3, kp_idx * 3 + 1, kp_idx * 3 + 2
