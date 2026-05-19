#!/usr/bin/env python3
"""
Run full ver2 pipeline: split -> train LSTM -> train Transformer -> verify TFLite.

  python scripts/run_pipeline.py --source data/raw
  python scripts/run_pipeline.py --source "PATH/to/fall-dataset6" --source-mode nested --skip-train
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--source-mode", choices=("flat", "nested"), default="flat")
    parser.add_argument("--skip-split", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()

    py = sys.executable

    if not args.skip_split:
        run(
            [
                py,
                "tools/build_video_split.py",
                "--source",
                str(args.source.resolve()),
                "--source-mode",
                args.source_mode,
            ]
        )

    if not args.skip_train:
        run([py, "scripts/train.py", "--model", "lstm", "--epochs", str(args.epochs)])
        run([py, "scripts/train.py", "--model", "transformer", "--epochs", str(args.epochs)])
        run([py, "tools/verify_tflite.py"])

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
