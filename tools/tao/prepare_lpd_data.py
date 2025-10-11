"""Convert CVAT/COCO plate boxes to TAO LPDNet format.

This is a scaffold; fill in I/O specifics once data paths are confirmed.

Usage example:
  python tools/tao/prepare_lpd_data.py \
    --coco data/processed/cam01/annotations/instances_Train.json \
    --images-root data/processed/cam01/images/Train \
    --out-dir data/tao/lpd/train
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert COCO to TAO LPD format")
    p.add_argument("--coco", required=True, help="COCO annotations JSON")
    p.add_argument("--images-root", required=True, help="Images root directory")
    p.add_argument("--out-dir", required=True, help="Output directory for TAO format")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # TODO: implement conversion to KITTI-style or TAO-expected LPD format.
    # Leave a small marker file to make directory visible.
    (out / "README.txt").write_text(
        "Prepared by prepare_lpd_data.py — TODO: write actual conversion.\n"
    )
    print(f"[prepare_lpd_data] Wrote placeholder to {out}")


if __name__ == "__main__":
    main()

