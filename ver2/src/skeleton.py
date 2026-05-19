"""Skeleton normalization (hip-centered, shoulder-hip scale)."""
from __future__ import annotations

import numpy as np

import config
from src.keypoints import get_kpt_indices


def normalize_skeleton_frame(
    frame_features: np.ndarray,
    min_confidence: float = config.MIN_KEYPOINT_CONFIDENCE,
) -> np.ndarray:
    normalized = np.copy(frame_features)
    refs = {
        "ls": "Left Shoulder",
        "rs": "Right Shoulder",
        "lh": "Left Hip",
        "rh": "Right Hip",
    }
    try:
        ls_x_i, ls_y_i, ls_c_i = get_kpt_indices(refs["ls"])
        rs_x_i, rs_y_i, rs_c_i = get_kpt_indices(refs["rs"])
        lh_x_i, lh_y_i, lh_c_i = get_kpt_indices(refs["lh"])
        rh_x_i, rh_y_i, rh_c_i = get_kpt_indices(refs["rh"])
    except ValueError:
        return frame_features

    ls_x, ls_y, ls_c = frame_features[ls_x_i], frame_features[ls_y_i], frame_features[ls_c_i]
    rs_x, rs_y, rs_c = frame_features[rs_x_i], frame_features[rs_y_i], frame_features[rs_c_i]
    lh_x, lh_y, lh_c = frame_features[lh_x_i], frame_features[lh_y_i], frame_features[lh_c_i]
    rh_x, rh_y, rh_c = frame_features[rh_x_i], frame_features[rh_y_i], frame_features[rh_c_i]

    mid_shoulder_x, mid_shoulder_y = np.nan, np.nan
    if ls_c > min_confidence and rs_c > min_confidence:
        mid_shoulder_x, mid_shoulder_y = (ls_x + rs_x) / 2, (ls_y + rs_y) / 2
    elif ls_c > min_confidence:
        mid_shoulder_x, mid_shoulder_y = ls_x, ls_y
    elif rs_c > min_confidence:
        mid_shoulder_x, mid_shoulder_y = rs_x, rs_y

    mid_hip_x, mid_hip_y = np.nan, np.nan
    if lh_c > min_confidence and rh_c > min_confidence:
        mid_hip_x, mid_hip_y = (lh_x + rh_x) / 2, (lh_y + rh_y) / 2
    elif lh_c > min_confidence:
        mid_hip_x, mid_hip_y = lh_x, lh_y
    elif rh_c > min_confidence:
        mid_hip_x, mid_hip_y = rh_x, rh_y

    if np.isnan(mid_hip_x) or np.isnan(mid_hip_y):
        return frame_features

    reference_height = np.nan
    if not np.isnan(mid_shoulder_y) and not np.isnan(mid_hip_y):
        reference_height = abs(mid_shoulder_y - mid_hip_y)

    perform_scaling = not (np.isnan(reference_height) or reference_height < 1e-5)

    for kp_name in config.SORTED_KEYPOINT_NAMES:
        try:
            x_i, y_i, _ = get_kpt_indices(kp_name)
            normalized[x_i] -= mid_hip_x
            normalized[y_i] -= mid_hip_y
            if perform_scaling:
                normalized[x_i] /= reference_height
                normalized[y_i] /= reference_height
        except ValueError:
            pass
    return normalized


def normalize_skeleton(
    sequence: np.ndarray,
    min_confidence: float = config.MIN_KEYPOINT_CONFIDENCE,
) -> np.ndarray:
    out = np.copy(sequence)
    for t in range(sequence.shape[0]):
        out[t] = normalize_skeleton_frame(sequence[t], min_confidence=min_confidence)
    return out
