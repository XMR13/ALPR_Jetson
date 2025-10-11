"""Unified TAO data prep wrapper for LPD and LPR.

Usage:
  python tools/tao/prepare_data.py lpd --coco <train_coco.json> --images-root <train_images_dir> --out-dir data/tao/lpd/train
  python tools/tao/prepare_data.py lpr --crops <crops_dir> --labels <labels.csv|json> --out-dir data/tao/lpr/train

This wrapper simply forwards args to the specific scripts so you don't have to remember which CLI flags go with which task.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="TAO data prep wrapper")
    sub = p.add_subparsers(dest="task", required=True)

    p_lpd = sub.add_parser("lpd", help="Prepare LPDNet (detector) dataset from COCO")
    p_lpd.add_argument("--coco", required=True)
    p_lpd.add_argument("--images-root", required=True)
    p_lpd.add_argument("--out-dir", required=True)
    p_lpd.add_argument("--category-name", default="license_plate")
    p_lpd.add_argument("--copy-mode", choices=["copy", "symlink", "none"], default="copy")

    p_lpr = sub.add_parser("lpr", help="Prepare LPRNet (recognizer) dataset from crops + labels")
    p_lpr.add_argument("--crops", required=True)
    p_lpr.add_argument("--labels", required=True)
    p_lpr.add_argument("--out-dir", required=True)
    p_lpr.add_argument("--copy-mode", choices=["copy", "symlink"], default="copy")

    args, unknown = p.parse_known_args()

    tools_dir = Path(__file__).parent
    if args.task == "lpd":
        cmd = [
            sys.executable,
            str(tools_dir / "prepare_lpd_data.py"),
            "--coco",
            args.coco,
            "--images-root",
            args.images_root,
            "--out-dir",
            args.out_dir,
            "--category-name",
            args.category_name,
            "--copy-mode",
            args.copy_mode,
        ] + unknown
    else:
        cmd = [
            sys.executable,
            str(tools_dir / "prepare_lpr_data.py"),
            "--crops",
            args.crops,
            "--labels",
            args.labels,
            "--out-dir",
            args.out_dir,
            "--copy-mode",
            args.copy_mode,
        ] + unknown

    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

