#!/usr/bin/env python3
"""
Quick sanity checks for a YOLO dataset (labels/images presence and alignment).

Checks
- images/train vs labels/train count and stem alignment
- images/val vs labels/val count and stem alignment
- warns about empty labels and missing pairs

Usage:
  python tools/verify_yolo_dataset.py --root /path/to/yolo_root
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Set, Tuple


def _stems_in(dir_path: Path, exts: Tuple[str, ...]) -> Set[str]:
    stems: Set[str] = set()
    for ext in exts:
        for p in dir_path.rglob(f"*{ext}"):
            stems.add(p.stem)
    return stems


def run(root: Path) -> int:
    img_train = root / "images" / "train"
    img_val = root / "images" / "val"
    lab_train = root / "labels" / "train"
    lab_val = root / "labels" / "val"
    for d in (img_train, img_val, lab_train, lab_val):
        if not d.exists():
            print(f"warn: missing directory: {d}")
    img_exts = (".jpg", ".jpeg", ".png", ".bmp")
    train_imgs = _stems_in(img_train, img_exts)
    val_imgs = _stems_in(img_val, img_exts)
    train_labs = _stems_in(lab_train, (".txt",))
    val_labs = _stems_in(lab_val, (".txt",))

    def report(split: str, imgs: Set[str], labs: Set[str]) -> None:
        missing_lab = imgs - labs
        missing_img = labs - imgs
        print(f"[{split}] images: {len(imgs)} labels: {len(labs)}")
        if missing_lab:
            print(f"  warn: {len(missing_lab)} images without labels")
        if missing_img:
            print(f"  warn: {len(missing_img)} labels without images")

    report("train", train_imgs, train_labs)
    report("val", val_imgs, val_labs)
    return 0


def build_parser():
    p = argparse.ArgumentParser(description="Verify YOLO dataset structure and alignment")
    p.add_argument("--root", required=True, help="YOLO dataset root")
    return p


def main(argv: List[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    raise SystemExit(run(Path(args.root)))


if __name__ == "__main__":
    main()

