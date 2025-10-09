"""Wrapper to launch YOLOX training using our custom exp.

This script sets environment variables to point the YOLOX exp to your data
and then invokes YOLOX's training entrypoint. Requires `yolox` to be
installed in the current Python environment.

Example (Windows, using your dataset without copying):
  python tools/train_yolox.py \
    --data-dir "D:\\RZQ\\Coding\\Datasets\\ALPR_First trial" \
    --train-ann annotations\\instances_Train.json \
    --val-ann annotations\\instances_Validation.json \
    --train-name images\\Train \
    --val-name images\\Validation \
    --batch 16 --epochs 50 --fp16
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train YOLOX-s for license plates")
    p.add_argument("--data-dir", required=True, help="Base data directory")
    p.add_argument("--train-ann", required=True, help="Train annotation JSON (relative to data-dir or absolute)")
    p.add_argument("--val-ann", required=True, help="Val annotation JSON (relative to data-dir or absolute)")
    p.add_argument("--train-name", default="train", help="Train images subfolder")
    p.add_argument("--val-name", default="val", help="Val images subfolder")
    p.add_argument("--batch", type=int, default=16, help="Batch size per GPU")
    p.add_argument("--epochs", type=int, default=50, help="Max epochs")
    p.add_argument("--fp16", action="store_true", help="Enable mixed precision")
    p.add_argument("--devices", type=int, default=1, help="Number of GPUs")
    p.add_argument("--ckpt", default="", help="Path to pretrained checkpoint (e.g., YOLOX_S.pth)")
    p.add_argument("--expn", default="plate_yolox_s", help="Experiment name for outputs/")
    p.add_argument("--seed", type=int, default=42, help="Training seed for reproducibility")
    p.add_argument("--cache", action="store_true", help="Cache images to RAM for faster IO (if enough memory)")
    p.add_argument("--no-aug-epochs", type=int, default=15, help="No-augmentation tail epochs (stability phase)")
    return p


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    env = os.environ.copy()
    env["YOLOX_DATA_DIR"] = str(args.data_dir)
    env["YOLOX_TRAIN_ANN"] = str(args.train_ann)
    env["YOLOX_VAL_ANN"] = str(args.val_ann)
    env["YOLOX_TRAIN_NAME"] = str(args.train_name)
    env["YOLOX_VAL_NAME"] = str(args.val_name)

    # Allow overriding epochs via env recognized by Exp
    env["YOLOX_MAX_EPOCH"] = str(args.epochs)
    env["YOLOX_NO_AUG"] = str(args.no_aug_epochs)
    if args.ckpt:
        env["YOLOX_PRETRAIN"] = str(args.ckpt)

    cmd = [
        sys.executable,
        "-m",
        "yolox.tools.train",
        "-f",
        "exps/yolox/exp_plate_yolox_s.py",
        "-d",
        str(args.devices),
        "-b",
        str(args.batch),
        "--expn",
        str(args.expn),
        "-o",  # occupy GPU memory to prevent fragmentation (YOLOX convention)
    ]
    if args.fp16:
        cmd.append("--fp16")
    if args.ckpt:
        cmd.extend(["-c", str(args.ckpt)])
    if args.cache:
        cmd.append("--cache")
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])

    print("Launching:", " ".join(cmd))
    try:
        rc = subprocess.call(cmd, env=env)
    except FileNotFoundError:
        print("error: YOLOX is not installed. Install with 'pip install yolox' (and PyTorch).", file=sys.stderr)
        sys.exit(2)
    sys.exit(rc)


if __name__ == "__main__":
    main()
