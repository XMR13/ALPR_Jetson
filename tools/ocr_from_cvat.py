#!/usr/bin/env python3
"""Build OCR crops and labels.csv from a CVAT export.

Supports two JSON formats:
  1) "CVAT for images 1.1" export (preferred for OCR), where each item contains
     annotations with a string attribute (e.g., name="text").
  2) COCO-style JSON where annotations may include attributes with plate text.

This script crops the labeled rectangles and writes a `labels.csv` file with
`filename,text` rows suitable for PaddleOCR fine-tuning.

Usage
  python tools/ocr_from_cvat.py \
    --json path/to/cvat.json \
    --images-dir path/to/images \
    --outdir data/ocr/train

Notes
- In CVAT, define a label "plate" with a string attribute, e.g., "text".
- The script looks for attributes named one of: ["text", "plate", "plate_text"].

"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image


AttrKeys = ("text", "plate", "plate_text")


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _ensure_out(outdir: Path) -> Tuple[Path, Path]:
    crops = outdir / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    return outdir, crops


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_attr(attrs: Any) -> Optional[str]:
    # attrs can be a list of {name,value} or a dict
    if isinstance(attrs, list):
        for a in attrs:
            n = str(a.get("name", "")).lower()
            if n in AttrKeys and isinstance(a.get("value"), str):
                return a["value"]
    elif isinstance(attrs, dict):
        for k in AttrKeys:
            if k in attrs and isinstance(attrs[k], str):
                return attrs[k]
    return None


def _crop(img_path: Path, box: Tuple[int, int, int, int], out_path: Path) -> bool:
    if not img_path.exists():
        return False
    try:
        with Image.open(img_path) as im:
            x1, y1, x2, y2 = box
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(im.width, x2)
            y2 = min(im.height, y2)
            if x2 <= x1 or y2 <= y1:
                return False
            crop = im.crop((x1, y1, x2, y2))
            crop.save(out_path, format="JPEG", quality=95)
            return True
    except Exception:
        return False


def _run_cvat_images(json_data: Dict[str, Any], images_dir: Path, outdir: Path) -> int:
    outdir, crops_dir = _ensure_out(outdir)
    items = json_data.get("items") or []
    rows: List[Tuple[str, str]] = []
    count = 0
    for it in items:
        name = it.get("name") or it.get("filename")
        if not name:
            continue
        img_path = images_dir / str(name)
        anns = it.get("annotations") or []
        for idx, ann in enumerate(anns):
            if str(ann.get("type", "")).lower() not in ("rectangle", "bbox", "box"):
                continue
            label = str(ann.get("label", "")).lower()
            if label and label not in ("plate", "license_plate", "licence_plate"):
                continue
            pts = ann.get("points") or []
            if len(pts) >= 4:
                x1, y1, x2, y2 = map(int, [pts[0], pts[1], pts[2], pts[3]])
            else:
                # Some exports use {x,y,width,height}
                x = _safe_int(ann.get("x"))
                y = _safe_int(ann.get("y"))
                w = _safe_int(ann.get("width"))
                h = _safe_int(ann.get("height"))
                x1, y1, x2, y2 = x, y, x + w, y + h
            text = _find_attr(ann.get("attributes", {}))
            if not text:
                continue
            out_name = f"{Path(name).stem}_ann{idx:04d}.jpg"
            ok = _crop(img_path, (x1, y1, x2, y2), crops_dir / out_name)
            if not ok:
                continue
            rows.append((out_name, text))
            count += 1
    # write labels.csv
    with (outdir / "labels.csv").open("w", encoding="utf-8") as f:
        f.write("filename,text\n")
        for fn, txt in rows:
            f.write(f"{fn},{txt}\n")
    print(f"wrote {count} crops to {crops_dir} and labels.csv with {len(rows)} rows")
    return count


def _run_coco(json_data: Dict[str, Any], images_dir: Path, outdir: Path) -> int:
    outdir, crops_dir = _ensure_out(outdir)
    imgs = {int(i.get("id")): i for i in json_data.get("images", [])}
    cats = {int(c.get("id")): c for c in json_data.get("categories", [])}
    rows: List[Tuple[str, str]] = []
    count = 0
    for ann in json_data.get("annotations", []):
        cid = int(ann.get("category_id", -1))
        cname = str(cats.get(cid, {}).get("name", "")).lower()
        if cname and cname not in ("plate", "license_plate", "licence_plate"):
            continue
        img = imgs.get(int(ann.get("image_id", -1)))
        if not img:
            continue
        file_name = img.get("file_name")
        if not file_name:
            continue
        x, y, w, h = [int(v) for v in ann.get("bbox", [0, 0, 0, 0])]
        text = None
        attrs = ann.get("attributes")
        if attrs:
            text = _find_attr(attrs)
        if not text:
            # CVAT sometimes stores attributes in 'text' directly
            t = ann.get("text")
            if isinstance(t, str):
                text = t
        if not text:
            continue
        img_path = images_dir / str(file_name)
        out_name = f"img{ann.get('image_id')}_ann{ann.get('id', count)}.jpg"
        ok = _crop(img_path, (x, y, x + w, y + h), crops_dir / out_name)
        if not ok:
            continue
        rows.append((out_name, text))
        count += 1

    with (outdir / "labels.csv").open("w", encoding="utf-8") as f:
        f.write("filename,text\n")
        for fn, txt in rows:
            f.write(f"{fn},{txt}\n")
    print(f"wrote {count} crops to {crops_dir} and labels.csv with {len(rows)} rows")
    return count


def run(json_path: Path, images_dir: Path, outdir: Path) -> int:
    data = _load_json(json_path)
    if "items" in data:  # CVAT for images 1.1
        return _run_cvat_images(data, images_dir, outdir)
    if set(["images", "annotations"]).issubset(data.keys()):  # COCO-like
        return _run_coco(data, images_dir, outdir)
    raise ValueError("Unsupported JSON format: expected CVAT for images or COCO export")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Crop OCR dataset from CVAT export")
    p.add_argument("--json", required=True, help="Path to CVAT JSON export")
    p.add_argument("--images-dir", required=True, help="Directory where source images live")
    p.add_argument("--outdir", required=True, help="Output directory for OCR dataset")
    return p


def main() -> None:
    args = build_parser().parse_args()
    json_path = Path(args.json)
    images_dir = Path(args.images_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    count = run(json_path, images_dir, outdir)
    raise SystemExit(0 if count > 0 else 3)


if __name__ == "__main__":
    main()

