"""Convert COCO plate boxes to TAO LPDNet (KITTI-style) format.

Outputs a directory with:
- images/  (copied files, by default)
- labels/  (KITTI-style per-image txt labels)
- list.txt (relative paths to images/, one per line)

Usage example:
  python tools/tao/prepare_lpd_data.py \
    --coco data/processed/cam01/annotations/instances_Train.json \
    --images-root data/processed/cam01/images/Train \
    --out-dir data/tao/lpd/train
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert COCO to TAO LPD (KITTI-style)")
    p.add_argument("--coco", required=True, help="COCO annotations JSON")
    p.add_argument("--images-root", required=True, help="Images root directory")
    p.add_argument("--out-dir", required=True, help="Output directory for TAO format")
    p.add_argument("--category-name", default="license_plate", help="Category name for plates in COCO")
    p.add_argument("--copy-mode", choices=["copy", "symlink", "none"], default="copy", help="How to place images into out/images")
    return p.parse_args()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _kitti_line(cls: str, xmin: float, ymin: float, xmax: float, ymax: float) -> str:
    # KITTI label: <type> <trunc> <occ> <alpha> <xmin> <ymin> <xmax> <ymax> <3D dims/location/rot>
    # Unused fields set to 0.0
    return f"{cls} 0.00 0 0.00 {xmin:.2f} {ymin:.2f} {xmax:.2f} {ymax:.2f} 0 0 0 0 0 0 0"


def main() -> None:
    args = parse_args()
    images_root = Path(args.images_root)
    out = Path(args.out_dir)
    out_images = out / "images"
    out_labels = out / "labels"
    _ensure_dir(out_images)
    _ensure_dir(out_labels)

    with open(args.coco, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Build indices
    img_by_id: Dict[int, Dict] = {im["id"]: im for im in coco.get("images", [])}
    cat_id_by_name: Dict[str, int] = {c["name"]: c["id"] for c in coco.get("categories", [])}
    plate_cat = cat_id_by_name.get(args.category_name)
    if plate_cat is None:
        raise SystemExit(f"Category '{args.category_name}' not found in COCO categories")

    anns_by_img: Dict[int, List[Dict]] = {}
    for a in coco.get("annotations", []):
        if a.get("category_id") != plate_cat:
            continue
        anns_by_img.setdefault(a["image_id"], []).append(a)

    listed: List[str] = []
    for img_id, anns in anns_by_img.items():
        info = img_by_id.get(img_id)
        if not info:
            continue
        file_name = info["file_name"]
        src = images_root / file_name
        if not src.exists():
            print(f"[warn] missing image: {src}")
            continue
        dst = out_images / Path(file_name).name
        if args.copy_mode == "copy":
            _ensure_dir(dst.parent)
            shutil.copy2(src, dst)
        elif args.copy_mode == "symlink":
            _ensure_dir(dst.parent)
            try:
                if dst.exists():
                    dst.unlink()
                dst.symlink_to(src.resolve())
            except Exception:
                shutil.copy2(src, dst)
        else:  # none
            dst = src

        # Write KITTI label
        lines: List[str] = []
        for a in anns:
            x, y, w, h = a["bbox"]
            xmin, ymin = x, y
            xmax, ymax = x + w, y + h
            lines.append(_kitti_line("license_plate", xmin, ymin, xmax, ymax))
        (out_labels / (Path(file_name).stem + ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
        listed.append(str((Path("images") / Path(file_name).name).as_posix()))

    (out / "list.txt").write_text("\n".join(sorted(listed)) + "\n", encoding="utf-8")
    print(f"[prepare_lpd_data] images: {len(listed)} → {out}")


if __name__ == "__main__":
    main()
