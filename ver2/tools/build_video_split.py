#!/usr/bin/env python3
"""
Build video-level train/val/test splits from .npy skeleton sequences.

Usage:
  python tools/build_video_split.py --source data/raw
  python tools/build_video_split.py --source "D:/datasets/fall-dataset6" --source-mode nested

Source modes:
  flat   - data/raw/fall/*.npy and data/raw/no_fall/*.npy
  nested - merges existing train/val/test/*/fall|no_fall into one pool, then re-splits by video_id
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config

def parse_npy_file(path: Path) -> tuple[str, str] | None:
    """
  Parse '{video_id}_fall_seq_###.npy' or '{video_id}_no_fall_seq_###.npy'.
  String-based parsing avoids regex ambiguity (e.g. IDs containing '_no').
    """
    name = path.name.lower()
    if not name.endswith(".npy"):
        return None
    if "_no_fall_seq_" in name:
        base, _ = name.rsplit("_no_fall_seq_", 1)
        return base, "no_fall"
    if "_fall_seq_" in name:
        base, _ = name.rsplit("_fall_seq_", 1)
        return base, "fall"
    return None


def collect_npy_files(source: Path, mode: str) -> list[dict]:
    records: list[dict] = []
    if mode == "flat":
        for label in ("fall", "no_fall"):
            folder = source / label
            if not folder.is_dir():
                continue
            for fp in sorted(folder.glob("*.npy")):
                parsed = parse_npy_file(fp)
                if not parsed:
                    print(f"  Skip (bad name): {fp.name}")
                    continue
                vid, file_label = parsed
                if file_label != label:
                    print(f"  Skip (label mismatch): {fp.name}")
                    continue
                records.append(
                    {
                        "source_path": str(fp.resolve()),
                        "filename": fp.name,
                        "video_id": vid,
                        "label": label,
                    }
                )
    elif mode == "nested":
        for split in ("train", "val", "test"):
            for label in ("fall", "no_fall"):
                folder = source / split / label
                if not folder.is_dir():
                    continue
                for fp in sorted(folder.glob("*.npy")):
                    parsed = parse_npy_file(fp)
                    if not parsed:
                        print(f"  Skip (bad name): {fp.name}")
                        continue
                    vid, file_label = parsed
                    if file_label != label:
                        continue
                    records.append(
                        {
                            "source_path": str(fp.resolve()),
                            "filename": fp.name,
                            "video_id": vid,
                            "label": label,
                        }
                    )
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return records


def assign_video_labels(records: list[dict]) -> dict[str, str]:
    """One label per video_id; warn if mixed fall/no_fall under same video."""
    by_video: dict[str, set[str]] = defaultdict(set)
    for r in records:
        by_video[r["video_id"]].add(r["label"])
    video_label: dict[str, str] = {}
    for vid, labels in by_video.items():
        if len(labels) > 1:
            print(f"  Warning: video_id '{vid}' has mixed labels {labels}; using majority fall if any fall")
            video_label[vid] = "fall" if "fall" in labels else "no_fall"
        else:
            video_label[vid] = next(iter(labels))
    return video_label


def _safe_train_test_split(*args, stratify=None, **kwargs):
    try:
        return train_test_split(*args, stratify=stratify, **kwargs)
    except ValueError:
        return train_test_split(*args, stratify=None, **kwargs)


def split_videos(
    video_ids: list[str],
    labels: list[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
) -> dict[str, str]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    n = len(video_ids)
    if n < 3:
        raise ValueError(
            f"Need at least 3 unique video_ids to split train/val/test, got {n}. "
            "Add more source videos before running this tool."
        )

    stratify = labels if len(set(labels)) > 1 else None
    train_ids, temp_ids, _, temp_y = _safe_train_test_split(
        video_ids,
        labels,
        test_size=(1 - train_ratio),
        stratify=stratify,
        random_state=random_state,
    )
    val_share = val_ratio / (val_ratio + test_ratio)
    if len(temp_ids) < 2:
        raise ValueError(
            f"Not enough videos in holdout pool ({len(temp_ids)}). "
            "Use a larger dataset or adjust split ratios."
        )
    stratify_temp = temp_y if len(set(temp_y)) > 1 else None
    val_ids, test_ids, _, _ = _safe_train_test_split(
        temp_ids,
        temp_y,
        test_size=(1 - val_share),
        stratify=stratify_temp,
        random_state=random_state,
    )
    assignment: dict[str, str] = {}
    for vid in train_ids:
        assignment[vid] = "train"
    for vid in val_ids:
        assignment[vid] = "val"
    for vid in test_ids:
        assignment[vid] = "test"
    return assignment


def copy_split_files(manifest: pd.DataFrame, output_root: Path, copy_files: bool) -> None:
    for split in ("train", "val", "test"):
        for label in ("fall", "no_fall"):
            (output_root / split / label).mkdir(parents=True, exist_ok=True)

    if not copy_files:
        return

    for _, row in manifest.iterrows():
        src = Path(row["source_path"])
        dst = output_root / row["split"] / row["label"] / row["filename"]
        if dst.exists():
            continue
        shutil.copy2(src, dst)


def check_leakage(manifest: pd.DataFrame) -> None:
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        va = set(manifest.loc[manifest["split"] == a, "video_id"])
        vb = set(manifest.loc[manifest["split"] == b, "video_id"])
        overlap = va & vb
        if overlap:
            raise RuntimeError(f"Leakage between {a} and {b}: {len(overlap)} videos, e.g. {list(overlap)[:5]}")


def build_stats(manifest: pd.DataFrame) -> dict:
    stats: dict = {"by_split": {}, "totals": {}}
    for split in ("train", "val", "test"):
        sub = manifest[manifest["split"] == split]
        stats["by_split"][split] = {
            "num_videos": int(sub["video_id"].nunique()),
            "num_sequences": len(sub),
            "num_fall_sequences": int((sub["label"] == "fall").sum()),
            "num_no_fall_sequences": int((sub["label"] == "no_fall").sum()),
        }
    stats["totals"] = {
        "num_videos": int(manifest["video_id"].nunique()),
        "num_sequences": len(manifest),
    }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Video-level split for fall detection .npy data")
    parser.add_argument(
        "--source",
        type=Path,
        default=config.DATA_RAW,
        help="Source directory (flat or nested layout)",
    )
    parser.add_argument(
        "--source-mode",
        choices=("flat", "nested"),
        default="flat",
        help="flat: raw/fall|no_fall; nested: raw/train|val|test/fall|no_fall merged then re-split",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.DATA_SPLITS,
        help="Output split directory",
    )
    parser.add_argument("--manifest", type=Path, default=config.MANIFEST_PATH)
    parser.add_argument("--stats", type=Path, default=config.SPLIT_STATS_PATH)
    parser.add_argument("--train-ratio", type=float, default=config.TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=config.VAL_RATIO)
    parser.add_argument("--test-ratio", type=float, default=config.TEST_RATIO)
    parser.add_argument("--seed", type=int, default=config.SPLIT_RANDOM_STATE)
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Only write manifest/stats without copying files",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.is_dir():
        print(f"ERROR: source not found: {source}")
        print("Place .npy files under data/raw/fall and data/raw/no_fall, then re-run.")
        sys.exit(1)

    print(f"Collecting .npy from {source} (mode={args.source_mode})")
    records = collect_npy_files(source, args.source_mode)
    if not records:
        print("ERROR: no valid .npy files found.")
        sys.exit(1)

    video_label = assign_video_labels(records)
    video_ids = sorted(video_label.keys())
    labels = [video_label[v] for v in video_ids]
    assignment = split_videos(
        video_ids,
        labels,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )

    rows = []
    for r in records:
        split = assignment[r["video_id"]]
        rows.append({**r, "split": split})
    manifest = pd.DataFrame(rows)
    check_leakage(manifest)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.manifest, index=False)
    print(f"Wrote manifest: {args.manifest} ({len(manifest)} sequences)")

    stats = build_stats(manifest)
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Wrote stats: {args.stats}")
    print(json.dumps(stats, indent=2))

    copy_split_files(manifest, args.output.resolve(), copy_files=not args.no_copy)
    if not args.no_copy:
        print(f"Copied files to: {args.output.resolve()}")
    print("Done. No video-level leakage detected.")


if __name__ == "__main__":
    main()
