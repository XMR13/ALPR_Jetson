from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from alpr_jetson.cli.common import add_ocr_backend_args, init_ocr_backend
from pipeline.alpr_runner import run_e2e_single  # type: ignore


def _load_defaults():
    try:
        import yaml  # type: ignore
        cfg_path = Path("configs/ocr/plate_defaults.yaml")
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            defaults = {
                "min_h": int(data.get("min_plate_h", 28)),
                "min_ar": float(data.get("min_ar", 1.5)),
                "max_ar": float(data.get("max_ar", 5.0)),
                "topk": int(data.get("topk", 1)),
                "postproc": str(data.get("postproc", "indonesia")),
                "allowed_prefix": list(data.get("allowed_prefix", ["A","B","D","F","E","Z","T"]))
            }
            # Optional: override allowed_prefix from a separate regional file if present
            pref_path = Path("configs/ocr/indonesia_prefixes.yaml")
            if pref_path.exists():
                try:
                    with open(pref_path, "r", encoding="utf-8") as pf:
                        pdata = yaml.safe_load(pf) or {}
                        if isinstance(pdata.get("prefixes"), list) and pdata.get("prefixes"):
                            defaults["allowed_prefix"] = [str(x) for x in pdata["prefixes"]]
                except Exception:
                    pass
            return defaults
    except Exception:
        pass
    return {"min_h": 28, "min_ar": 1.5, "max_ar": 5.0, "topk": 1, "postproc": "indonesia", "allowed_prefix": ["A","B","D","F","E","Z","T"]}


def add_subcommands(sub):
    d = _load_defaults()
    p_e2e = sub.add_parser("e2e", help="Run detector + OCR on images")
    p_e2e.add_argument("--det-engine", required=True, help="Path to detector TensorRT .engine")
    p_e2e.add_argument("--source", required=True, help="Image file or directory")
    p_e2e.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold")
    p_e2e.add_argument("--iou", type=float, default=0.45, help="Detection IoU threshold")
    p_e2e.add_argument("--min-plate-h", type=int, default=d["min_h"], help="Minimum plate bbox height (px) to OCR")
    p_e2e.add_argument("--min-ar", type=float, default=d["min_ar"], help="Minimum plate aspect ratio (w/h)")
    p_e2e.add_argument("--max-ar", type=float, default=d["max_ar"], help="Maximum plate aspect ratio (w/h)")
    p_e2e.add_argument("--annotate-dir", default="", help="Optional directory to save annotated outputs")
    p_e2e.add_argument("--topk", type=int, default=d["topk"], help="Max plates per image to OCR (1=highest confidence only)")
    p_e2e.add_argument("--text-only", action="store_true", help="Print only plate texts (one per line)")
    p_e2e.add_argument("--text-out", default="", help="Optional path to write plate texts (one per line)")
    p_e2e.add_argument(
        "--strict-filters",
        action="store_true",
        help=(
            "Enforce size/aspect filters like JSON path. By default e2e is permissive "
            "(legacy behavior): accepts all detections above --conf/--iou and lets OCR/postproc decide."
        ),
    )
    add_ocr_backend_args(p_e2e)
    p_e2e.add_argument("--postproc", choices=["none", "indonesia"], default=d["postproc"], help="Apply plate post-processing")
    p_e2e.add_argument("--allowed-prefix", nargs="*", default=d["allowed_prefix"], help="Allowed prefixes for postproc")
    p_e2e.set_defaults(func=cmd_e2e)

    p_e2e_json = sub.add_parser("e2e-json", help="Run detector + OCR on a single image and emit JSON to stdout")
    p_e2e_json.add_argument("--det-engine", required=True, help="Path to detector TensorRT .engine")
    p_e2e_json.add_argument("--source", required=True, help="Image file path")
    p_e2e_json.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold")
    p_e2e_json.add_argument("--iou", type=float, default=0.45, help="Detection IoU threshold")
    p_e2e_json.add_argument("--min-plate-h", type=int, default=d["min_h"], help="Minimum plate bbox height (px) to OCR")
    p_e2e_json.add_argument("--min-ar", type=float, default=d["min_ar"], help="Minimum plate aspect ratio (w/h)")
    p_e2e_json.add_argument("--max-ar", type=float, default=d["max_ar"], help="Maximum plate aspect ratio (w/h)")
    add_ocr_backend_args(p_e2e_json)
    p_e2e_json.add_argument("--postproc", choices=["none", "indonesia"], default=d["postproc"], help="Apply plate post-processing")
    p_e2e_json.add_argument("--allowed-prefix", nargs="*", default=d["allowed_prefix"], help="Allowed prefixes for postproc")
    p_e2e_json.add_argument("--debug-crops", action="store_true", help="Include crop acceptance debug info in JSON")
    p_e2e_json.add_argument("--topk", type
    =int, default=d["topk"], help="Max plates per image to OCR (1=highest confidence only)")
    p_e2e_json.add_argument("--accept-all", action="store_true", help="Bypass size/AR filters (debug)")
    p_e2e_json.set_defaults(func=cmd_e2e_json)

    p_e2e_stream = sub.add_parser(
        "e2e-json-stream",
        help="Run detector + OCR once, then read image paths from stdin and emit JSON lines",
    )
    p_e2e_stream.add_argument("--det-engine", required=True, help="Path to detector TensorRT .engine")
    p_e2e_stream.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold")
    p_e2e_stream.add_argument("--iou", type=float, default=0.45, help="Detection IoU threshold")
    p_e2e_stream.add_argument("--min-plate-h", type=int, default=d["min_h"], help="Minimum plate bbox height (px) to OCR")
    p_e2e_stream.add_argument("--min-ar", type=float, default=d["min_ar"], help="Minimum plate aspect ratio (w/h)")
    p_e2e_stream.add_argument("--max-ar", type=float, default=d["max_ar"], help="Maximum plate aspect ratio (w/h)")
    add_ocr_backend_args(p_e2e_stream)
    p_e2e_stream.add_argument("--postproc", choices=["none", "indonesia"], default=d["postproc"], help="Apply plate post-processing")
    p_e2e_stream.add_argument("--allowed-prefix", nargs="*", default=d["allowed_prefix"], help="Allowed prefixes for postproc")
    p_e2e_stream.add_argument("--stop-on-error", action="store_true", help="Stop processing on first error")
    p_e2e_stream.add_argument("--debug-crops", action="store_true", help="Include crop acceptance debug info in NDJSON")
    p_e2e_stream.add_argument("--topk", type=int, default=d["topk"], help="Max plates per image to OCR (1=highest confidence only)")
    p_e2e_stream.add_argument("--accept-all", action="store_true", help="Bypass size/AR filters (debug)")
    p_e2e_stream.set_defaults(func=cmd_e2e_json_stream)


def _load_detector(det_engine_path: str):
    try:
        from inference.yolov9_trt import load_engine  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"detector runtime missing: {exc}") from exc
    return load_engine(det_engine_path, print_plugins=False)


def cmd_e2e_json(args: argparse.Namespace) -> int:
    if args.iou < 0.0 or args.iou > 1.0:
        print(json.dumps({"error": "--iou must be in [0,1]"}))
        return 2
    try:
        from ocr_service.postprocess import postprocess_indonesia  # type: ignore
    except Exception as exc:  # pragma: no cover
        print(json.dumps({"error": f"runtime dependencies missing: {exc}"}))
        return 2

    try:
        backend, ocr_runner = init_ocr_backend(args)
        det_engine = _load_detector(args.det_engine)
        result = run_e2e_single(
            args.source,
            det_engine=det_engine,
            ocr_runner=ocr_runner,
            backend=backend,
            conf=args.conf,
            iou=args.iou,
            postproc=args.postproc,
            allowed_prefix=args.allowed_prefix or [],
            postprocess_fn=postprocess_indonesia,
            min_plate_h=getattr(args, "min_plate_h", 28),
            min_ar=getattr(args, "min_ar", 1.5),
            max_ar=getattr(args, "max_ar", 5.0),
            debug_crops=bool(getattr(args, "debug_crops", False)),
            accept_all=bool(getattr(args, "accept_all", False)),
            topk=int(getattr(args, "topk", 1)),
        )
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        return 2

    print(json.dumps(result, ensure_ascii=False))
    return 0


def cmd_e2e_json_stream(args: argparse.Namespace) -> int:
    if args.iou < 0.0 or args.iou > 1.0:
        print(json.dumps({"error": "--iou must be in [0,1]"}))
        return 2
    try:
        from ocr_service.postprocess import postprocess_indonesia  # type: ignore
    except Exception as exc:  # pragma: no cover
        print(json.dumps({"error": f"runtime dependencies missing: {exc}"}))
        return 2

    try:
        backend, ocr_runner = init_ocr_backend(args)
        det_engine = _load_detector(args.det_engine)
    except Exception as exc:
        print(json.dumps({"error": f"init failed: {exc}"}))
        return 2

    stop_on_error = bool(getattr(args, "stop_on_error", False))
    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue
        try:
            result = run_e2e_single(
                path,
                det_engine=det_engine,
                ocr_runner=ocr_runner,
                backend=backend,
                conf=args.conf,
                iou=args.iou,
                postproc=args.postproc,
                allowed_prefix=args.allowed_prefix or [],
                postprocess_fn=postprocess_indonesia,
                min_plate_h=getattr(args, "min_plate_h", 28),
                min_ar=getattr(args, "min_ar", 1.5),
                max_ar=getattr(args, "max_ar", 5.0),
                debug_crops=bool(getattr(args, "debug_crops", False)),
                accept_all=bool(getattr(args, "accept_all", False)),
                topk=int(getattr(args, "topk", 1)),
            )
            payload = {"input": path}
            payload.update(result)
            print(json.dumps(payload, ensure_ascii=False))
            sys.stdout.flush()
        except Exception as exc:
            print(json.dumps({"input": path, "error": str(exc)}))
            sys.stdout.flush()
            if stop_on_error:
                return 2
    return 0


def cmd_e2e(args: argparse.Namespace) -> int:
    """Same canonical pipeline as JSON, with optional annotation and text summary."""
    try:
        import os, glob
        import cv2  # type: ignore
        from ocr_service.postprocess import postprocess_indonesia  # type: ignore
    except Exception as exc:
        print(f"E2E runtime dependencies missing: {exc}", file=sys.stderr)
        return 2

    if args.iou < 0.0 or args.iou > 1.0:
        print("--iou must be in [0,1]", file=sys.stderr)
        return 2

    try:
        backend, ocr_runner = init_ocr_backend(args)
        det_engine = _load_detector(args.det_engine)
    except Exception as exc:
        print(f"failed to initialize runtime: {exc}", file=sys.stderr)
        return 2

    # Collect images
    paths: List[str] = []
    src = Path(args.source)
    if src.is_dir():
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            paths.extend(sorted(Path(src).glob(ext)))
        paths = [str(p) for p in paths]
    else:
        paths = [str(src)]
    if not paths:
        print("no input images found", file=sys.stderr)
        return 2

    annotate_dir = Path(args.annotate_dir) if args.annotate_dir else None
    if annotate_dir:
        annotate_dir.mkdir(parents=True, exist_ok=True)

    text_lines: List[str] = []

    for img_path in paths:
        try:
            result = run_e2e_single(
                img_path,
                det_engine=det_engine,
                ocr_runner=ocr_runner,
                backend=backend,
                conf=args.conf,
                iou=args.iou,
                postproc=args.postproc,
                allowed_prefix=args.allowed_prefix or [],
                postprocess_fn=postprocess_indonesia,
                min_plate_h=getattr(args, "min_plate_h", 28),
                min_ar=getattr(args, "min_ar", 1.5),
                max_ar=getattr(args, "max_ar", 5.0),
                accept_all=not bool(getattr(args, "strict_filters", False)),
                topk=int(getattr(args, "topk", 1)),
            )
        except Exception as exc:
            print(f"failed e2e on {img_path}: {exc}", file=sys.stderr)
            continue

        plates = result.get("plates", [])
        final_texts = [p.get("text", "") for p in plates]

        if getattr(args, "text_only", False):
            for t in final_texts:
                if t:
                    print(t)
                    text_lines.append(t)
        else:
            print(f"{img_path}")
            for idx, p in enumerate(plates):
                bbox = tuple(int(v) for v in p.get("bbox", [0, 0, 0, 0]))
                det_score = float(p.get("det_conf", 0.0))
                label = p.get("text", "")
                print(f"  det#{idx}: conf={det_score:.2f} plate='{label}' bbox={bbox}")
            if not plates:
                print("  no detections above threshold")
            for t in final_texts:
                if t:
                    text_lines.append(t)

        if annotate_dir:
            try:
                img0 = cv2.imread(str(img_path))
                if img0 is None:
                    raise RuntimeError("failed to read image for annotation")
                annotated = img0.copy()
                for idx, p in enumerate(plates):
                    x1, y1, x2, y2 = [int(v) for v in p.get("bbox", [0, 0, 0, 0])]
                    det_score = float(p.get("det_conf", 0.0))
                    label = p.get("text", "") or "<unk>"
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        annotated,
                        f"{label}:{det_score:.2f}",
                        (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                out_path = annotate_dir / (Path(img_path).stem + "_alpr.jpg")
                cv2.imwrite(str(out_path), annotated)
                print(f"  annotated: {out_path}")
            except Exception as exc:
                print(f"warn: annotation skipped for {img_path}: {exc}", file=sys.stderr)

    if getattr(args, "text_out", ""):
        try:
            outp = Path(args.text_out)
            outp.parent.mkdir(parents=True, exist_ok=True)
            with open(outp, "w", encoding="utf-8") as f:
                for line in text_lines:
                    f.write(f"{line}\n")
            if not getattr(args, "text_only", False):
                print(f"wrote plate texts: {outp}")
        except Exception as exc:
            print(f"failed to write --text-out file: {exc}", file=sys.stderr)
            return 2
    return 0
