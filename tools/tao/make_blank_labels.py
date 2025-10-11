"""Generate a blank labels CSV from a crops directory.

Creates a CSV with header `filename,text` where `filename` is the basename
of each image in the crops directory and `text` is empty. This is useful to
bootstrap TAO LPRNet datasets when ground-truth transcripts are not yet ready.

Usage:
  python tools/tao/make_blank_labels.py --crops data/processed/ocr_crops/train \
    --out data/processed/ocr_crops/train_labels.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make blank labels CSV from crops directory")
    p.add_argument("--crops", required=True, help="Directory containing crop images")
    p.add_argument("--out", required=True, help="Output CSV path")
    p.add_argument("--exts", nargs="*", default=[".jpg", ".jpeg", ".png"], help="Image extensions to include")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    crops = Path(args.crops)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    files = sorted([p.name for p in crops.iterdir() if p.suffix.lower() in {e.lower() for e in args.exts}])
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "text"])
        for name in files:
            writer.writerow([name, ""])  # empty transcript
    print(f"[make_blank_labels] wrote {len(files)} rows to {out}")


if __name__ == "__main__":
    main()

