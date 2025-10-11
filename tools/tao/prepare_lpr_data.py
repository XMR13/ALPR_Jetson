"""Prepare TAO LPRNet crops + transcripts from plate crops.

Produces a directory with:
- images/ (copied files)
- labels.txt (TAB-separated: relative_image_path\tTRANSCRIPT)

Usage example:
  python tools/tao/prepare_lpr_data.py \
    --crops data/processed/ocr_crops/train \
    --labels data/processed/ocr_crops/train_labels.csv \
    --out-dir data/tao/lpr/train
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare TAO LPRNet dataset")
    p.add_argument("--crops", required=True, help="Plate crops directory")
    p.add_argument("--labels", required=True, help="CSV or JSON labels (filename,text)")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--copy-mode", choices=["copy", "symlink"], default="copy")
    return p.parse_args()


def load_labels(labels_path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if labels_path.suffix.lower() == ".csv":
        with open(labels_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Accept common column names
            candidates = [
                ("filename", "text"),
                ("file", "text"),
                ("image", "label"),
            ]
            for row in reader:
                key_col = next((k for k, _ in candidates if k in row), None)
                val_col = next((v for _, v in candidates if v in row), None)
                if key_col and val_col:
                    data[row[key_col]] = str(row[val_col]).strip()
    else:
        # JSON: either list of {filename,text} or dict {filename: text}
        obj = json.loads(Path(labels_path).read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            data = {str(k): str(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            for item in obj:
                fn = item.get("filename") or item.get("file") or item.get("image")
                tx = item.get("text") or item.get("label")
                if fn and tx:
                    data[str(fn)] = str(tx)
    return data


def main() -> None:
    args = parse_args()
    src_root = Path(args.crops)
    labels_path = Path(args.labels)
    out = Path(args.out_dir)
    out_images = out / "images"
    out.mkdir(parents=True, exist_ok=True)
    out_images.mkdir(parents=True, exist_ok=True)

    mapping = load_labels(labels_path)
    if not mapping:
        raise SystemExit("No labels parsed; ensure CSV has filename/text headers or JSON structure is supported.")

    lines = []
    for rel, text in mapping.items():
        src = src_root / rel
        if not src.exists():
            print(f"[warn] missing crop: {src}")
            continue
        dst = out_images / Path(rel).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            if args.copy_mode == "symlink":
                if dst.exists():
                    dst.unlink()
                dst.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dst)
        except Exception:
            shutil.copy2(src, dst)
        lines.append(f"images/{dst.name}\t{text}")

    (out / "labels.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[prepare_lpr_data] images: {len(lines)} → {out}")


if __name__ == "__main__":
    main()
