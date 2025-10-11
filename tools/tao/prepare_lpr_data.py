"""Prepare TAO LPRNet crops + transcripts from plate crops.

Expected input: a directory of plate crop images and a CSV/JSON file with filenames and GT text.
Output: TAO LPRNet formatted dataset directory.

Usage example:
  python tools/tao/prepare_lpr_data.py \
    --crops data/processed/ocr_crops/train \
    --labels data/processed/ocr_crops/train_labels.csv \
    --out-dir data/tao/lpr/train
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare TAO LPRNet dataset")
    p.add_argument("--crops", required=True, help="Plate crops directory")
    p.add_argument("--labels", required=True, help="CSV/JSON labels (fname,text)")
    p.add_argument("--out-dir", required=True, help="Output directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # TODO: implement copy/symlink of crops and normalized transcripts to TAO format.
    (out / "README.txt").write_text(
        "Prepared by prepare_lpr_data.py — TODO: write actual conversion.\n"
    )
    print(f"[prepare_lpr_data] Wrote placeholder to {out}")


if __name__ == "__main__":
    main()

