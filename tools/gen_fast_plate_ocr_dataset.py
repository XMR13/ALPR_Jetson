#!/usr/bin/env python3
"""Generate annotations.csv for fast-plate-ocr from cropped plate images.

This script parses plate text from filenames based on the convention used in
`data/images_test` and `data/images_test_det`:

Example:
    1_20251106123924_MG3 Tengah Gate 1_CCTV1_BB8043LH_123924_det0.jpg

The plate text is taken from the segment immediately following `CCTV<id>_`
and before the next underscore. If that segment is empty (e.g.
`..._CCTV1__010221...`) the sample is treated as "unregistered" and skipped.

The output CSV follows the format expected by fast-plate-ocr:

    image_path,plate_text

where `image_path` is a path to the crop image *relative to* the directory
containing the CSV file.

Usage (from repo root):
    python tools/gen_fast_plate_ocr_dataset.py \
        --crops-dir data/images_test_det \
        --out-csv data/fast_plate_ocr/train/annotations.csv

This will NOT move or copy images; it only writes a CSV that points at the
existing crop files via relative paths.
"""

from __future__ import annotations

import argparse
import csv
import re
import os
import random
import shutil
from pathlib import Path
from typing import List, Sequence, Tuple


_PLATE_FROM_NAME = re.compile(r"CCTV\d*_([^_]*?)_")


def parse_plate_from_name(stem: str) -> str | None:
    """Extract plate string from a filename stem.

    Returns:
        Uppercased plate text, or None if the plate segment is empty or
        not found.
    """
    m = _PLATE_FROM_NAME.search(stem)
    if not m:
        return None
    plate_raw = m.group(1)
    if not plate_raw:
        return None
    plate = plate_raw.strip().upper()
    return plate or None


def find_crops(crops_dir: Path) -> List[Path]:
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    return sorted(p for p in crops_dir.iterdir() if p.suffix.lower() in exts and p.is_file())


def split_samples(
    samples: Sequence[Tuple[Path, str]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[Path, str]], List[Tuple[Path, str]]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("--val-ratio must be in [0,1)")
    if val_ratio == 0.0:
        return list(samples), []

    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    val_count = int(round(len(shuffled) * val_ratio))
    val_samples = shuffled[:val_count]
    train_samples = shuffled[val_count:]
    return train_samples, val_samples


def write_csv(
    out_csv: Path,
    samples: Sequence[Tuple[Path, str]],
    *,
    copy_images: bool,
    images_subdir: str,
) -> int:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    images_dir = out_csv.parent / images_subdir if copy_images else None
    if images_dir:
        images_dir.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "plate_text"])
        for path, plate in samples:
            rel_source = path
            if images_dir is not None:
                dest_path = images_dir / path.name
                shutil.copy2(path, dest_path)
                rel_source = dest_path
            rel_path = os.path.relpath(rel_source, out_csv.parent).replace("\\", "/")
            writer.writerow([rel_path, plate])
    return len(samples)


def gather_samples(crops_dir: Path) -> Tuple[int, List[Tuple[Path, str]]]:
    crops = find_crops(crops_dir)
    if not crops:
        raise SystemExit(f"no crop images found in {crops_dir}")
    total = len(crops)
    labeled: List[Tuple[Path, str]] = []
    for img_path in crops:
        plate = parse_plate_from_name(img_path.stem)
        if not plate:
            continue
        labeled.append((img_path, plate))
    return total, labeled


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate fast-plate-ocr annotations.csv from cropped plate images.",
    )
    parser.add_argument(
        "--crops-dir",
        required=True,
        help="Directory containing cropped plate images (e.g. data/images_test_det).",
    )
    parser.add_argument(
        "--out-csv",
        required=True,
        help="Output CSV path (e.g. data/fast_plate_ocr/train/annotations.csv).",
    )
    parser.add_argument(
        "--val-csv",
        default="",
        help="Optional validation CSV path. If provided, --val-ratio controls split size.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (ignored unless --val-csv is set).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for splitting when --val-csv is provided.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy crop files next to the CSVs (under <csv_dir>/<images_subdir>).",
    )
    parser.add_argument(
        "--images-subdir",
        default="images",
        help="Subdirectory name (inside CSV directory) to store copied images when --copy-images is set.",
    )
    args = parser.parse_args()

    crops_dir = Path(args.crops_dir)
    out_csv = Path(args.out_csv)
    val_csv = Path(args.val_csv) if args.val_csv else None

    if not crops_dir.is_dir():
        raise SystemExit(f"--crops-dir is not a directory: {crops_dir}")

    total_files, labeled = gather_samples(crops_dir=crops_dir)
    if not labeled:
        raise SystemExit("no labeled crops found (all filenames missing plate segment after CCTV*)")

    if val_csv:
        train_samples, val_samples = split_samples(labeled, val_ratio=float(args.val_ratio), seed=int(args.seed))
    else:
        train_samples, val_samples = labeled, []

    written_train = write_csv(
        out_csv=out_csv,
        samples=train_samples,
        copy_images=bool(args.copy_images),
        images_subdir=str(args.images_subdir),
    )
    print(f"Scanned {total_files} crop files in {crops_dir}")
    print(f"Wrote {written_train} labeled rows to {out_csv}")
    if val_samples:
        written_val = write_csv(
            out_csv=val_csv,
            samples=val_samples,
            copy_images=bool(args.copy_images),
            images_subdir=str(args.images_subdir),
        )
        print(f"Wrote {written_val} labeled rows to {val_csv}")
    if written_train == 0:
        print("WARNING: no labeled plates were found; check filename pattern and CCTV segments.")


if __name__ == "__main__":
    main()
