#!/usr/bin/env python3
"""Generate COCO-format detections using a TensorRT YOLOv9 engine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from inference.yolov9_trt import infer_image, load_engine  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run YOLOv9 TRT over a COCO image list.")
    p.add_argument("--engine", required=True, help="Path to TensorRT .engine")
    p.add_argument("--coco", required=True, help="Ground-truth COCO JSON (for image list)")
    p.add_argument(
        "--images-root",
        help="Root directory containing images referenced by the COCO file "
        "(defaults to the COCO JSON parent directory).",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Destination JSON for predictions (COCO detections format).",
    )
    p.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    p.add_argument("--category-id", type=int, default=1, help="COCO category id for license plates")
    p.add_argument("--limit", type=int, help="Optional limit on number of images to process")
    p.add_argument("--print-plugins", action="store_true", help="Print TensorRT plugin registry")
    return p


def coco_detections(
    engine_path: Path,
    coco_path: Path,
    images_root: Path,
    conf: float,
    iou: float,
    category_id: int,
    limit: int | None,
    print_plugins: bool,
) -> List[Dict[str, Any]]:
    engine = load_engine(str(engine_path), print_plugins=print_plugins)

    with coco_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    images: List[Dict[str, Any]] = coco.get("images", [])

    preds: List[Dict[str, Any]] = []
    missing = 0
    processed = 0

    for img_info in images:
        if limit is not None and processed >= limit:
            break
        processed += 1
        file_name = img_info.get("file_name")
        image_id = img_info.get("id")
        if file_name is None or image_id is None:
            print(f"[WARN] skipping entry without file_name/id: {img_info}", file=sys.stderr)
            continue

        img_path = images_root / file_name
        if not img_path.exists():
            print(f"[WARN] missing image: {img_path}", file=sys.stderr)
            missing += 1
            continue

        try:
            _, dets = infer_image(engine, str(img_path), conf=conf, iou=iou)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] inference failed for {img_path}: {exc}", file=sys.stderr)
            continue

        for (x1, y1, x2, y2), score, cls in dets:
            w = max(0.0, float(x2) - float(x1))
            h = max(0.0, float(y2) - float(y1))
            preds.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(category_id),
                    "bbox": [float(x1), float(y1), w, h],
                    "score": float(score),
                }
            )

    if missing:
        print(f"[INFO] skipped {missing} images (not found under {images_root})", file=sys.stderr)
    return preds


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    engine_path = Path(args.engine)
    coco_path = Path(args.coco)
    if args.images_root:
        images_root = Path(args.images_root)
    else:
        images_root = coco_path.parent

    if not engine_path.exists():
        parser.error(f"engine not found: {engine_path}")
    if not coco_path.exists():
        parser.error(f"COCO file not found: {coco_path}")
    if not images_root.exists():
        parser.error(f"images root not found: {images_root}")

    preds = coco_detections(
        engine_path=engine_path,
        coco_path=coco_path,
        images_root=images_root,
        conf=args.conf,
        iou=args.iou,
        category_id=args.category_id,
        limit=args.limit,
        print_plugins=args.print_plugins,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(preds, f)
    print(f"Wrote {len(preds)} detections to {output_path}")


if __name__ == "__main__":
    main()
