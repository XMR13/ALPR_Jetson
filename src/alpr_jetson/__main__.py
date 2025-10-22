import argparse
import subprocess
import sys
from pathlib import Path


def _add_ocr_backend_args(p: argparse.ArgumentParser) -> None:
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--engine", help="Path to OCR TensorRT .engine (CTC)")
    group.add_argument("--onnx", help="Path to OCR ONNX model (slot-based)")
    p.add_argument("--charset", help="Path to charset.txt (TensorRT only)")
    p.add_argument("--plate-config", help="YAML config for ONNX slot model")
    p.add_argument("--onnx-provider", choices=["cuda", "cpu"], default="cuda", help="ONNXRuntime provider")
    p.add_argument("--onnx-gpu-mem-limit-mb", type=int, default=768, help="Cap CUDA EP allocator (MB)")
    p.add_argument("--input-width", type=int, default=160, help="Model input width (TensorRT)")
    p.add_argument("--input-height", type=int, default=32, help="Model input height (TensorRT)")
    p.add_argument("--channels", type=int, default=1, help="Input channels (TensorRT)")
    p.add_argument("--no-clahe", action="store_true", help="Disable CLAHE (TensorRT)")
    p.add_argument(
        "--input-layout",
        choices=["NCHW", "NHWC"],
        default="NCHW",
        help="TensorRT engine input layout",
    )
    p.add_argument(
        "--logits-layout",
        choices=["NTC", "NCT"],
        default="NTC",
        help="TensorRT logits layout",
    )
    p.add_argument("--blank-index", type=int, default=0, help="CTC blank index (TensorRT)")


def _init_ocr_backend(args: argparse.Namespace):
    try:
        import yaml  # type: ignore
        from ocr_service.trt_infer import OCRService  # type: ignore
        from ocr_service.preprocess import PreprocConfig  # type: ignore
        from ocr_service.onnx_infer import OnnxPlateOCR, PlateConfig  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"OCR dependencies missing: {exc}") from exc

    if getattr(args, "onnx", None):
        cfg_path = getattr(args, "plate_config", "")
        if not cfg_path:
            raise ValueError("--plate-config is required when using --onnx")
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        except Exception as exc:
            raise ValueError(f"failed to read plate config: {exc}") from exc

        required = ["max_plate_slots", "alphabet", "pad_char", "img_height", "img_width"]
        missing = [k for k in required if k not in cfg]
        if missing:
            raise ValueError(f"missing keys in plate config: {', '.join(missing)}")
        plate_cfg = PlateConfig(
            max_plate_slots=int(cfg["max_plate_slots"]),
            alphabet=str(cfg["alphabet"]),
            pad_char=str(cfg["pad_char"]),
            img_height=int(cfg["img_height"]),
            img_width=int(cfg["img_width"]),
            keep_aspect_ratio=bool(cfg.get("keep_aspect_ratio", False)),
            interpolation=str(cfg.get("interpolation", "linear")),
            image_color_mode=str(cfg.get("image_color_mode", "rgb")),
            padding_color=cfg.get("padding_color", (144, 144, 144)),
        )
        runner = OnnxPlateOCR(
            args.onnx,
            plate_cfg,
            prefer_trt=False,
            provider=getattr(args, "onnx_provider", "cuda"),
            gpu_mem_limit_mb=getattr(args, "onnx_gpu_mem_limit_mb", 768),
        )
        return "onnx", runner

    engine_path = getattr(args, "engine", "")
    charset = getattr(args, "charset", "")
    if not engine_path:
        raise ValueError("--engine is required when --onnx is not provided")
    if not charset:
        raise ValueError("--charset is required for TensorRT OCR")

    svc = OCRService(
        engine_path=engine_path,
        charset_path=charset,
        preproc=PreprocConfig(
            input_width=getattr(args, "input_width", 160),
            input_height=getattr(args, "input_height", 32),
            channels=getattr(args, "channels", 1),
            mean=0.5,
            std=0.5,
            use_clahe=not getattr(args, "no_clahe", False),
        ),
        logits_layout=getattr(args, "logits_layout", "NTC"),
        input_layout=getattr(args, "input_layout", "NCHW"),
        blank_index=getattr(args, "blank_index", 0),
    )
    return "trt", svc


def cmd_ocr_infer(args: argparse.Namespace) -> int:
    """Infer OCR text for an image or all images in a directory.

    Requires TensorRT engine and charset file to be present on the host.
    Prints results to stdout and optionally writes a CSV.
    """
    try:
        import os, glob, csv
        import cv2  # type: ignore
        from ocr_service.postprocess import postprocess_indonesia  # type: ignore
    except Exception as e:
        print(f"OCR runtime dependencies missing: {e}", file=sys.stderr)
        return 2

    try:
        backend, runner = _init_ocr_backend(args)
    except Exception as exc:
        print(f"failed to initialize OCR backend: {exc}", file=sys.stderr)
        return 2

    paths = []
    if Path(args.source).is_dir():
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            paths.extend(glob.glob(str(Path(args.source) / ext)))
        paths.sort()
    else:
        paths = [args.source]

    results = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"warn: failed to read {p}", file=sys.stderr)
            continue
        texts = runner.infer_batch([img])  # type: ignore[arg-type]
        text = texts[0] if texts else ""
        final = text
        if args.postproc == "indonesia":
            final, _ = postprocess_indonesia(text, allowed_prefix=args.allowed_prefix or None)
        print(f"{p}: {final}")
        results.append((p, final))

    if args.output:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "text"])
            for p, t in results:
                w.writerow([p, t])
        print(f"wrote: {args.output}")
    return 0


def cmd_rtsp_smoke(args: argparse.Namespace) -> int:
    script = Path("tools/rtsp_smoke.sh")
    if not script.exists():
        print("tools/rtsp_smoke.sh not found", file=sys.stderr)
        return 2
    uri = args.uri
    latency = str(args.latency)
    return subprocess.call(["bash", str(script), uri, latency])


def cmd_deepstream_smoke(args: argparse.Namespace) -> int:
    script = Path("tools/deepstream_smoke.sh")
    if not script.exists():
        print("tools/deepstream_smoke.sh not found", file=sys.stderr)
        return 2
    cfg = args.config
    return subprocess.call(["bash", str(script), cfg])


def cmd_e2e(args: argparse.Namespace) -> int:
    try:
        import os, glob
        from typing import List
        import cv2  # type: ignore
        from ocr_service.postprocess import postprocess_indonesia  # type: ignore
        from inference.yolov9_trt import load_engine, infer_image  # type: ignore
    except Exception as e:
        print(f"E2E runtime dependencies missing: {e}", file=sys.stderr)
        return 2

    if args.iou < 0.0 or args.iou > 1.0:
        print("--iou must be in [0,1]", file=sys.stderr)
        return 2

    try:
        backend, ocr_runner = _init_ocr_backend(args)
    except Exception as exc:
        print(f"failed to initialize OCR backend: {exc}", file=sys.stderr)
        return 2

    det_engine = load_engine(args.det_engine, print_plugins=False)

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

    for img_path in paths:
        try:
            img0, dets = infer_image(det_engine, img_path, conf=args.conf, iou=args.iou)
        except Exception as exc:
            print(f"failed detection on {img_path}: {exc}", file=sys.stderr)
            continue

        crops = []
        boxes = []
        scores = []
        for box, score, cls in dets:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            x1 = max(0, min(x1, img0.shape[1] - 1))
            x2 = max(0, min(x2, img0.shape[1] - 1))
            y1 = max(0, min(y1, img0.shape[0] - 1))
            y2 = max(0, min(y2, img0.shape[0] - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img0[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crops.append(crop)
            boxes.append((x1, y1, x2, y2))
            scores.append(float(score))

        texts = ocr_runner.infer_batch(crops) if crops else []  # type: ignore[arg-type]

        print(f"{img_path}")
        for idx, (bbox, det_score) in enumerate(zip(boxes, scores)):
            text = texts[idx] if idx < len(texts) else ""
            final = text
            if args.postproc == "indonesia" and text:
                final, _ = postprocess_indonesia(text, allowed_prefix=args.allowed_prefix or None)
            print(f"  det#{idx}: conf={det_score:.2f} plate='{final}' bbox={bbox}")
        if not boxes:
            print("  no detections above threshold")

        if annotate_dir:
            annotated = img0.copy()
            for idx, ((x1, y1, x2, y2), det_score) in enumerate(zip(boxes, scores)):
                label = texts[idx] if idx < len(texts) else ""
                if args.postproc == "indonesia" and label:
                    label, _ = postprocess_indonesia(label, allowed_prefix=args.allowed_prefix or None)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"{label or '<unk>'}:{det_score:.2f}",
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

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="alpr-jetson", description="ALPR Jetson CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_rtsp = sub.add_parser("rtsp-smoke", help="Run RTSP GStreamer smoke test")
    p_rtsp.add_argument("uri", help="RTSP URI")
    p_rtsp.add_argument("--latency", type=int, default=200, help="rtspsrc latency (ms)")
    p_rtsp.set_defaults(func=cmd_rtsp_smoke)

    p_ds = sub.add_parser("ds-smoke", help="Run DeepStream app smoke test")
    p_ds.add_argument("--config", default="configs/deepstream/app_config.txt", help="deepstream-app config path")
    p_ds.set_defaults(func=cmd_deepstream_smoke)

    p_ocr = sub.add_parser("ocr-infer", help="Run OCR on an image or directory")
    _add_ocr_backend_args(p_ocr)
    p_ocr.add_argument("--source", required=True, help="Image file or directory of images")
    p_ocr.add_argument("--output", default="", help="Optional CSV output path")
    p_ocr.add_argument("--postproc", choices=["none", "indonesia"], default="none", help="Apply plate post-processing")
    p_ocr.add_argument("--allowed-prefix", nargs="*", default=["A","B","D","F","E","Z","T"], help="Allowed region prefixes for postproc")
    p_ocr.set_defaults(func=cmd_ocr_infer)

    p_e2e = sub.add_parser("e2e", help="Run detector + OCR on images")
    p_e2e.add_argument("--det-engine", required=True, help="Path to detector TensorRT .engine")
    p_e2e.add_argument("--source", required=True, help="Image file or directory")
    p_e2e.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold")
    p_e2e.add_argument("--iou", type=float, default=0.45, help="Detection IoU threshold")
    p_e2e.add_argument("--annotate-dir", default="", help="Optional directory to save annotated outputs")
    _add_ocr_backend_args(p_e2e)
    p_e2e.add_argument("--postproc", choices=["none", "indonesia"], default="indonesia", help="Apply plate post-processing")
    p_e2e.add_argument("--allowed-prefix", nargs="*", default=["A","B","D","F","E","Z","T"], help="Allowed prefixes for postproc")
    p_e2e.set_defaults(func=cmd_e2e)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    rc = args.func(args)
    sys.exit(rc)


if __name__ == "__main__":
    main()
