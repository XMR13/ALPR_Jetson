from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple


def add_subcommand(sub):
    p_det = sub.add_parser("det-infer", help="Run detector on images (no OCR)")
    p_det.add_argument("--det-engine", required=True, help="Path to detector TensorRT .engine")
    p_det.add_argument("--source", required=True, help="Image file or directory")
    p_det.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold")
    p_det.add_argument("--iou", type=float, default=0.45, help="Detection IoU threshold")
    p_det.add_argument("--labels", default="", help="Optional class labels file (one name per line)")
    p_det.add_argument("--annotate-dir", default="", help="Directory to save annotated detections")
    p_det.add_argument("--crop-dir", default="", help="Directory to export plate crops for each detection")
    p_det.add_argument("--print-plugins", action="store_true", help="Print TensorRT plugin registry before loading engine")
    p_det.set_defaults(func=cmd_det_infer)


def cmd_det_infer(args: argparse.Namespace) -> int:
    try:
        import glob
        import cv2  # type: ignore
        from inference.yolov9_trt import load_engine, infer_image  # type: ignore
    except Exception as exc:
        print(f"Detector runtime dependencies missing: {exc}", file=sys.stderr)
        return 2

    if not (0.0 <= args.iou <= 1.0):
        print("--iou must be in [0,1]", file=sys.stderr)
        return 2

    labels = None
    if args.labels:
        try:
            with open(args.labels, "r", encoding="utf-8") as f:
                labels = [line.strip() for line in f if line.strip()]
        except Exception as exc:
            print(f"failed to read labels file: {exc}", file=sys.stderr)
            return 2

    try:
        det_engine = load_engine(args.det_engine, print_plugins=args.print_plugins)
    except Exception as exc:
        print(f"failed to load detector engine: {exc}", file=sys.stderr)
        return 2

    paths: List[str] = []
    src = Path(args.source)
    if src.is_dir():
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            paths.extend(sorted(glob.glob(str(src / ext))))
    else:
        paths = [str(src)]
    if not paths:
        print("no input images found", file=sys.stderr)
        return 2

    annotate_dir = Path(args.annotate_dir) if args.annotate_dir else None
    if annotate_dir:
        annotate_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = Path(args.crop_dir) if getattr(args, "crop_dir", "") else None
    if crop_dir:
        crop_dir.mkdir(parents=True, exist_ok=True)

    for img_path in paths:
        try:
            img0, dets = infer_image(det_engine, img_path, conf=args.conf, iou=args.iou)
        except Exception as exc:
            print(f"failed detection on {img_path}: {exc}", file=sys.stderr)
            continue

        det_info = []
        for idx, (bbox, score, cls) in enumerate(dets):
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            label = str(cls)
            if labels and 0 <= cls < len(labels):
                label = labels[cls]
            det_info.append((idx, (x1, y1, x2, y2), float(score), label))

        print(img_path)
        if not det_info:
            print("  no detections above threshold")
        else:
            for idx, (x1, y1, x2, y2), score, label in det_info:
                print(f"  det#{idx}: conf={score:.2f} cls={label} bbox={(x1, y1, x2, y2)}")

        if annotate_dir and det_info:
            annotated = img0.copy()
            for idx, (x1, y1, x2, y2), score, label in det_info:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"{label}:{score:.2f}",
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
            out_path = annotate_dir / (Path(img_path).stem + "_det.jpg")
            cv2.imwrite(str(out_path), annotated)
            print(f"  annotated: {out_path}")

        if crop_dir and det_info:
            for idx, (x1, y1, x2, y2), score, label in det_info:
                crop = img0[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop_path = crop_dir / f"{Path(img_path).stem}_det{idx}.jpg"
                cv2.imwrite(str(crop_path), crop)
                print(f"  crop saved: {crop_path}")

    return 0

