#!/usr/bin/env python3
"""
COCO dataset quick stats to tune detector training.

Reports image count, annotation count, bbox height/width percentiles,
and suggests training input size considerations for small objects.

Usage:
  # COCO format
  python tools/dataset_stats.py --coco path/to/coco.json

  # YOLO format (expects labels under <root>/labels/{train,val})
  python tools/dataset_stats.py --yolo-root /path/to/dataset_root
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def pct(vals: List[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    i = int(max(0, min(len(s) - 1, round(q * (len(s) - 1)))))
    return float(s[i])


def _report(widths: List[float], heights: List[float], images_count: int) -> int:
    small_cnt = sum(1 for h in heights if h < 28)
    total = len(heights)
    h_p50 = pct(heights, 0.5)
    h_p90 = pct(heights, 0.9)
    h_p95 = pct(heights, 0.95)
    w_p50 = pct(widths, 0.5)
    w_p90 = pct(widths, 0.9)
    w_p95 = pct(widths, 0.95)

    print(f"images: {images_count}  annotations: {total}")
    print(f"bbox height px  p50:{h_p50:.1f}  p90:{h_p90:.1f}  p95:{h_p95:.1f}")
    print(f"bbox width  px  p50:{w_p50:.1f}  p90:{w_p90:.1f}  p95:{w_p95:.1f}")
    if total:
        print(f"small (<28px height): {small_cnt} ({100.0*small_cnt/total:.1f}%)")

    if h_p50 < 40:
        print("suggestion: consider training at 736-800 input or stronger mosaic to help small plates")
    else:
        print("suggestion: 640 input is likely sufficient; validate on val set")
    return 0


def run_coco(coco_path: Path) -> int:
    with coco_path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
    anns = [a for a in data.get("annotations", []) if isinstance(a.get("bbox"), list)]
    imgs = data.get("images", [])
    widths: List[float] = []
    heights: List[float] = []
    for a in anns:
        x, y, w, h = a["bbox"]
        widths.append(float(w))
        heights.append(float(h))
    return _report(widths, heights, images_count=len(imgs))


def run_yolo(root: Path) -> int:
    """Parse YOLO TXT labels to compute bbox size stats.

    Expects labels under <root>/labels/{train,val} and images under
    <root>/images/{train, val}. Uses image sizes if suffix-matching image
    is found; otherwise assumes 640 and reports relative sizes scaled.
    """
    import cv2  # lazy import

    labels_dirs = [root / "labels" / "train", root / "labels" / "val"]
    images_roots = [root / "images" / "train", root / "images" / "val"]
    widths: List[float] = []
    heights: List[float] = []
    images_seen = set()

    def _find_image(stem: str) -> Path | None:
        for r in images_roots:
            for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                p = r / f"{stem}{ext}"
                if p.exists():
                    return p
        return None

    for d in labels_dirs:
        if not d.exists():
            continue
        for txt in d.rglob("*.txt"):
            stem = txt.stem
            img_path = _find_image(stem)
            iw, ih = 640, 640
            if img_path is not None:
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        ih, iw = img.shape[:2]
                        images_seen.add(str(img_path))
                except Exception:
                    pass
            # Read labels
            try:
                for line in txt.read_text(encoding="utf-8").splitlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    _, cx, cy, ww, hh = parts
                    w = float(ww) * iw
                    h = float(hh) * ih
                    widths.append(w)
                    heights.append(h)
            except Exception:
                continue

    return _report(widths, heights, images_count=len(images_seen))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dataset stats for tuning (COCO or YOLO)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--coco", help="Path to coco.json")
    g.add_argument("--yolo-root", help="YOLO dataset root (images/ and labels/ under it)")
    return p


def main(argv: List[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if getattr(args, "coco", None):
        rc = run_coco(Path(args.coco))
    else:
        rc = run_yolo(Path(args.yolo_root))
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
