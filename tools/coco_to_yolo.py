#!/usr/bin/env python3
"""
Convert COCO detection annotations to YOLO (v5/v8/v9) TXT labels.

Outputs one TXT per image with lines: "<class_id> <cx> <cy> <w> <h>" in
normalized coordinates relative to the image width/height.

Usage examples
  python tools/coco_to_yolo.py \
    --coco data/processed/cam01/train/coco.json \
    --outdir data/processed/cam01_yolo/labels

Notes
- This script writes label files only. Point your YOLO images directory
  to the corresponding images root when training.
- If multiple categories exist, provide --class-map to select/remap ids.
  Otherwise, category ids are normalized to a 0..K-1 range by ascending
  sorted unique category ids found in the COCO file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _build_class_id_map(categories: List[Dict[str, Any]], class_map: Dict[str, int] | None) -> Dict[int, int]:
    # If class_map provided, accept mapping from category name → new id.
    if class_map:
        name_to_id = {str(c.get("name")): int(c.get("id")) for c in categories}
        out: Dict[int, int] = {}
        for name, new_id in class_map.items():
            if name not in name_to_id:
                raise ValueError(f"class '{name}' not found in categories")
            out[int(name_to_id[name])] = int(new_id)
        return out
    # Default: compact ids from sorted category ids.
    src_ids = sorted(set(int(c.get("id")) for c in categories))
    return {src_id: i for i, src_id in enumerate(src_ids)}


def _xywh_to_yolo(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    cx = (x + w / 2.0) / max(1.0, float(img_w))
    cy = (y + h / 2.0) / max(1.0, float(img_h))
    ww = w / max(1.0, float(img_w))
    hh = h / max(1.0, float(img_h))
    # Clip to [0,1] for robustness
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    ww = min(max(ww, 0.0), 1.0)
    hh = min(max(hh, 0.0), 1.0)
    return cx, cy, ww, hh


def run(coco_path: Path, outdir: Path, class_map: Dict[str, int] | None = None) -> int:
    with coco_path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    images = {int(i.get("id")): i for i in data.get("images", [])}
    categories: List[Dict[str, Any]] = data.get("categories", [])
    cat_to_new = _build_class_id_map(categories, class_map)

    # Prepare per-image label lines
    labels: Dict[int, List[str]] = {int(i): [] for i in images.keys()}
    for ann in data.get("annotations", []):
        img_id = int(ann.get("image_id"))
        if img_id not in images:
            continue
        x, y, w, h = [float(v) for v in ann.get("bbox", [0, 0, 0, 0])]
        if w <= 0 or h <= 0:
            continue
        cat_id = int(ann.get("category_id"))
        if cat_id not in cat_to_new:
            # Skip categories not in the mapping
            continue
        new_id = int(cat_to_new[cat_id])
        img_w = int(images[img_id].get("width", 0))
        img_h = int(images[img_id].get("height", 0))
        cx, cy, ww, hh = _xywh_to_yolo(x, y, w, h, img_w, img_h)
        labels.setdefault(img_id, []).append(f"{new_id} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")

    # Write out
    outdir.mkdir(parents=True, exist_ok=True)
    count = 0
    for img_id, lines in labels.items():
        # Use image file stem as label name by default
        file_name = str(images[img_id].get("file_name", f"img{img_id}.jpg"))
        stem = Path(file_name).stem
        label_path = outdir / f"{stem}.txt"
        with label_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        count += 1
    print(f"wrote YOLO labels for {count} images under {outdir}")
    return count


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="COCO → YOLO label converter")
    p.add_argument("--coco", required=True, help="Path to COCO annotations JSON")
    p.add_argument("--outdir", required=True, help="Output directory for YOLO labels/*.txt")
    p.add_argument(
        "--class-map",
        default="",
        help="Optional mapping 'name0=0,name1=1' to control class ids",
    )
    return p


def _parse_class_map(expr: str) -> Dict[str, int] | None:
    expr = (expr or "").strip()
    if not expr:
        return None
    out: Dict[str, int] = {}
    for part in expr.split(","):
        if not part:
            continue
        if "=" not in part:
            raise ValueError("class-map entries must be 'name=id'")
        name, val = part.split("=", 1)
        out[name.strip()] = int(val.strip())
    return out


def main(argv: List[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    coco_path = Path(args.coco)
    outdir = Path(args.outdir)
    class_map = _parse_class_map(args.class_map)
    if not coco_path.exists():
        raise SystemExit(f"error: COCO file not found: {coco_path}")
    count = run(coco_path, outdir, class_map)
    raise SystemExit(0 if count > 0 else 3)


if __name__ == "__main__":
    main()

