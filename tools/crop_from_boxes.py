"""Crop plate ROIs from COCO boxes to build OCR datasets.

This reads a COCO JSON and crops bounding boxes from the corresponding
images directory into an output folder. Cropping uses Pillow when
available; if not installed, the script exits with a helpful message.

Usage
    python tools/crop_from_boxes.py \
        --coco path/to/coco.json --images path/to/images --outdir crops/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _require_pillow():
    try:
        from PIL import Image  # noqa: F401
    except Exception as e:  # pragma: no cover - environment dependent
        print(
            "Pillow (PIL) not installed. Install with 'pip install Pillow' to enable cropping.",
            file=sys.stderr,
        )
        raise SystemExit(3) from e


def _load_coco(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _index_images(coco: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {int(img["id"]): img for img in coco.get("images", [])}


def _xywh_to_xyxy(b: List[float]) -> Tuple[int, int, int, int]:
    x, y, w, h = b
    return int(x), int(y), int(x + w), int(y + h)


def run(coco_json: Path, images_dir: Path, outdir: Path) -> int:
    _require_pillow()
    from PIL import Image

    coco = _load_coco(coco_json)
    img_index = _index_images(coco)
    outdir.mkdir(parents=True, exist_ok=True)

    ann_count = 0
    for ann in coco.get("annotations", []):
        if ann.get("iscrowd"):
            continue
        img_id = int(ann["image_id"])  # type: ignore[index]
        bbox = ann.get("bbox")
        if not bbox:
            continue
        img_info = img_index.get(img_id)
        if not img_info:
            continue
        file_name = img_info.get("file_name")
        if not file_name:
            continue
        img_path = images_dir / file_name
        if not img_path.exists():
            continue

        try:
            with Image.open(img_path) as im:
                crop_box = _xywh_to_xyxy([float(v) for v in bbox])
                # Clamp crop to image bounds
                x1 = max(0, crop_box[0])
                y1 = max(0, crop_box[1])
                x2 = min(im.width, crop_box[2])
                y2 = min(im.height, crop_box[3])
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = im.crop((x1, y1, x2, y2))
                out_name = f"img{img_id}_ann{ann.get('id', ann_count)}.jpg"
                crop.save(outdir / out_name, format="JPEG", quality=95)
                ann_count += 1
        except Exception:
            # Skip problematic files/annotations and continue
            continue

    print(f"wrote {ann_count} crops to {outdir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Crop OCR regions from COCO boxes")
    p.add_argument("--coco", required=True, help="Path to coco.json")
    p.add_argument("--images", required=True, help="Directory of source images (COCO file_name roots)")
    p.add_argument("--outdir", required=True, help="Output directory for crops")
    return p


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    rc = run(Path(args.coco), Path(args.images), Path(args.outdir))
    sys.exit(rc)


if __name__ == "__main__":
    main()
