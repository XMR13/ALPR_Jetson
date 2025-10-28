#!/usr/bin/env python3
"""
Generate a YOLOv9-compatible dataset YAML from a YOLO directory layout.

Assumes the common structure:
  <root>/images/train
  <root>/images/val
  <root>/labels/train
  <root>/labels/val

Writes a YAML with keys: train, val, nc, names

Example:
  python tools/gen_yolov9_data_yaml.py \
    --root /mnt/data/plates_yolo \
    --names plate \
    --out configs/training/plates_yolov9.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List


def run(root: Path, names: List[str], out: Path) -> Path:
    img_train = root / "images" / "train"
    img_val = root / "images" / "val"
    if not img_train.exists() or not img_val.exists():
        raise FileNotFoundError("expected images/train and images/val under dataset root")
    out.parent.mkdir(parents=True, exist_ok=True)
    nc = len(names)
    yaml = (
        f"train: {img_train}\n"
        f"val: {img_val}\n"
        f"nc: {nc}\n"
        f"names: [{', '.join(names)}]\n"
    )
    out.write_text(yaml, encoding="utf-8")
    print(f"wrote {out}")
    return out


def build_parser():
    p = argparse.ArgumentParser(description="Generate YOLOv9 dataset YAML")
    p.add_argument("--root", required=True, help="YOLO dataset root (images/ and labels/ under it)")
    p.add_argument(
        "--names",
        required=True,
        help="Comma-separated class names in order (e.g., 'plate' or 'plate,vehicle')",
    )
    p.add_argument("--out", required=True, help="Output YAML path")
    return p


def main(argv: List[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    names = [s.strip() for s in str(args.names).split(",") if s.strip()]
    if not names:
        raise SystemExit("error: --names must provide at least one class name")
    run(Path(args.root), names, Path(args.out))


if __name__ == "__main__":
    main()

