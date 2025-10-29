import argparse
import subprocess
import sys
from pathlib import Path

#Menambahkan argumen untuk bakcend dari ocr
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
    """
    Menginisiasi backend OCR berdasarkan argumen yang telah diberikan
    """
    try:
        import yaml  # type: ignore
        from ocr_service.trt_infer import OCRService  # type: ignore
        from ocr_service.preprocess import PreprocConfig  # type: ignore
        from ocr_service.onnx_infer import OnnxPlateOCR, PlateConfig  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"OCR dependencies missing: {exc}") from exc

    #jika ingin melakukan inferenc dengan onnx maka perlu menambahkan beberapa konfigurasi tambahan
    if getattr(args, "onnx", None):
        cfg_path = getattr(args, "plate_config", "") #plate config argumen jika menggunakan onnx
        if not cfg_path:
            raise ValueError("--plate-config is required when using --onnx")
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        except Exception as exc:
            raise ValueError(f"failed to read plate config: {exc}") from exc

        required = ["max_plate_slots", "alphabet", "pad_char", "img_height", "img_width"] #argumen wajib yang perlu dimasukkan
        missing = [k for k in required if k not in cfg]
        if missing:
            raise ValueError(f"missing keys in plate config: {', '.join(missing)}")
        plate_cfg = PlateConfig(
            max_plate_slots=int(cfg["max_plate_slots"]),
            alphabet=str(cfg["alphabet"]),
            pad_char=str(cfg["pad_char"]),
            img_height=int(cfg["img_height"]),
            img_width=int(cfg["img_width"]),
            keep_aspect_ratio=bool(cfg.get("keep_aspect_ratio", True)),
            interpolation=str(cfg.get("interpolation", "area")),
            image_color_mode=str(cfg.get("image_color_mode", "grayscale")),
            padding_color=cfg.get("padding_color", (144, 144, 144)),
            use_clahe=bool(cfg.get("use_clahe", False)),
            clahe_clip=float(cfg.get("clahe_clip", 2.0)),
            clahe_tile=int(cfg.get("clahe_tile", 8)),
            clahe_brightness_gate=float(cfg.get("clahe_brightness_gate", 0.0)),
            auto_deskew=bool(cfg.get("auto_deskew", False)),
            deskew_threshold_deg=float(cfg.get("deskew_threshold_deg", 12.0)),
        )
        runner = OnnxPlateOCR(
            args.onnx,
            plate_cfg,
            prefer_trt=False,
            provider=getattr(args, "onnx_provider", "cuda"),
            gpu_mem_limit_mb=getattr(args, "onnx_gpu_mem_limit_mb", 768),
        )
        return "onnx", runner
    
    #apabila ingin menggunakan infernce dengan tensor Rt maka maksukkan argumen --engine
    engine_path = getattr(args, "engine", "")
    charset = getattr(args, "charset", "")  #tambahan arguemn yang diberikan jika ingin menggunakan tensorRT sebagai inferrence
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

#inference hanya untuk OCR
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

# inference hanya untuk deteksi plat saja
def cmd_det_infer(args: argparse.Namespace) -> int:
    """Run detector only (no OCR) on an image or directory."""
    try:
        import glob
        from typing import List
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

    text_lines = []  # collect final plate texts for --text-out / --text-only

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

        # Prepare final texts (optionally postprocessed)
        final_texts = []
        for idx in range(len(boxes)):
            t = texts[idx] if idx < len(texts) else ""
            if args.postproc == "indonesia" and t:
                t, _ = postprocess_indonesia(t, allowed_prefix=args.allowed_prefix or None)
            final_texts.append(t)

        # Emit output depending on flags
        if args.text_only:
            for t in final_texts:
                if t:
                    print(t)
                    text_lines.append(t)
            # If no detections, print nothing in text-only mode
        else:
            print(f"{img_path}")
            for idx, (bbox, det_score) in enumerate(zip(boxes, scores)):
                label = final_texts[idx] if idx < len(final_texts) else ""
                print(f"  det#{idx}: conf={det_score:.2f} plate='{label}' bbox={bbox}")
            if not boxes:
                print("  no detections above threshold")
            # Also collect plain texts for optional --text-out
            for t in final_texts:
                if t:
                    text_lines.append(t)

        if annotate_dir:
            annotated = img0.copy()
            for idx, ((x1, y1, x2, y2), det_score) in enumerate(zip(boxes, scores)):
                label = final_texts[idx] if idx < len(final_texts) else ""
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

    # If requested, write plain plate texts to a file (one per line)
    if getattr(args, "text_out", ""):
        try:
            outp = Path(args.text_out)
            outp.parent.mkdir(parents=True, exist_ok=True)
            with open(outp, "w", encoding="utf-8") as f:
                for line in text_lines:
                    f.write(f"{line}\n")
            if not args.text_only:
                print(f"wrote plate texts: {outp}")
        except Exception as exc:
            print(f"failed to write --text-out file: {exc}", file=sys.stderr)
            return 2

    return 0


def _plate_conf(det_conf: float, char_confs: list[float]) -> float:
    if not char_confs:
        return float(det_conf)
    avg_char = sum(char_confs) / max(1, len(char_confs))
    return float(det_conf) * float(avg_char)


def _run_e2e_single(
    image_path: str,
    *,
    det_engine,
    ocr_runner,
    backend: str,
    conf: float,
    iou: float,
    postproc: str,
    allowed_prefix: list[str],
    postprocess_fn=None,
):
    import time
    from typing import List, Tuple

    import cv2  # type: ignore
    from inference.yolov9_trt import infer_image  # type: ignore

    img_path = str(Path(image_path))
    t0 = time.time()
    img0, dets = infer_image(det_engine, img_path, conf=conf, iou=iou)
    det_ms = (time.time() - t0) * 1000.0

    h, w = img0.shape[:2]
    MIN_H = 28
    AR_MIN, AR_MAX = 1.5, 5.0
    crops: List[Tuple[Tuple[int, int, int, int], float]] = []
    for bbox, score, _cls in dets:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        hbox = max(1, y2 - y1)
        wbox = max(1, x2 - x1)
        ar = float(wbox) / float(hbox)
        if (hbox < MIN_H) or (ar < AR_MIN) or (ar > AR_MAX):
            continue
        crops.append(((x1, y1, x2, y2), float(score)))

    texts: List[str] = []
    char_confs: List[List[float]] = []
    ocr_ms = 0.0
    if crops:
        crop_imgs = [img0[y1:y2, x1:x2] for (x1, y1, x2, y2), _ in crops]
        t1 = time.time()
        if backend == "onnx":
            res = ocr_runner.infer_batch(crop_imgs, return_confidence=True)  # type: ignore[attr-defined]
            if isinstance(res, tuple) and len(res) == 2:
                texts, char_confs = res  # type: ignore[misc]
            else:
                texts = list(res)  # type: ignore[arg-type]
        else:
            texts = ocr_runner.infer_batch(crop_imgs)  # type: ignore[attr-defined]
        ocr_ms = (time.time() - t1) * 1000.0

    plates = []
    for i, ((x1, y1, x2, y2), det_conf) in enumerate(crops):
        raw = texts[i] if i < len(texts) else ""
        confs = char_confs[i] if i < len(char_confs) else []
        norm_text = raw
        is_valid = True
        if postproc == "indonesia" and postprocess_fn is not None:
            norm_text, is_valid = postprocess_fn(raw, allowed_prefix=allowed_prefix or None)
        plates.append(
            {
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "det_conf": float(det_conf),
                "ocr_raw": raw,
                "text": norm_text,
                "valid": bool(is_valid),
                "plate_conf": _plate_conf(det_conf, confs),
                "char_confs": [float(c) for c in confs],
            }
        )

    status = "ok" if plates else "no_plate"
    return {
        "status": status,
        "plates": plates,
        "latency_ms": {"det": det_ms, "ocr": ocr_ms, "total": det_ms + ocr_ms},
    }


def cmd_e2e_json(args: argparse.Namespace) -> int:
    """Run detector + OCR on a single image and print JSON to stdout."""

    import json

    if args.iou < 0.0 or args.iou > 1.0:
        print(json.dumps({"error": "--iou must be in [0,1]"}))
        return 2

    try:
        from ocr_service.postprocess import postprocess_indonesia  # type: ignore
        from inference.yolov9_trt import load_engine  # type: ignore
    except Exception as exc:
        print(json.dumps({"error": f"runtime dependencies missing: {exc}"}))
        return 2

    try:
        backend, ocr_runner = _init_ocr_backend(args)
    except Exception as exc:
        print(json.dumps({"error": f"failed to initialize OCR backend: {exc}"}))
        return 2

    try:
        det_engine = load_engine(args.det_engine, print_plugins=False)
    except Exception as exc:
        print(json.dumps({"error": f"failed to load detector: {exc}"}))
        return 2

    try:
        result = _run_e2e_single(
            args.source,
            det_engine=det_engine,
            ocr_runner=ocr_runner,
            backend=backend,
            conf=args.conf,
            iou=args.iou,
            postproc=args.postproc,
            allowed_prefix=args.allowed_prefix or [],
            postprocess_fn=postprocess_indonesia,
        )
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        return 2

    print(json.dumps(result, ensure_ascii=False))
    return 0


def cmd_e2e_json_stream(args: argparse.Namespace) -> int:
    """Run detector + OCR once, then read image paths from stdin and emit NDJSON."""

    import json
    import sys

    if args.iou < 0.0 or args.iou > 1.0:
        print(json.dumps({"error": "--iou must be in [0,1]"}))
        return 2

    try:
        from ocr_service.postprocess import postprocess_indonesia  # type: ignore
        from inference.yolov9_trt import load_engine  # type: ignore
    except Exception as exc:
        print(json.dumps({"error": f"runtime dependencies missing: {exc}"}))
        return 2

    try:
        backend, ocr_runner = _init_ocr_backend(args)
    except Exception as exc:
        print(json.dumps({"error": f"failed to initialize OCR backend: {exc}"}))
        return 2

    try:
        det_engine = load_engine(args.det_engine, print_plugins=False)
    except Exception as exc:
        print(json.dumps({"error": f"failed to load detector: {exc}"}))
        return 2

    stop_on_error = bool(getattr(args, "stop_on_error", False))
    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue
        try:
            result = _run_e2e_single(
                path,
                det_engine=det_engine,
                ocr_runner=ocr_runner,
                backend=backend,
                conf=args.conf,
                iou=args.iou,
                postproc=args.postproc,
                allowed_prefix=args.allowed_prefix or [],
                postprocess_fn=postprocess_indonesia,
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

    p_e2e = sub.add_parser("e2e", help="Run detector + OCR on images")
    p_e2e.add_argument("--det-engine", required=True, help="Path to detector TensorRT .engine")
    p_e2e.add_argument("--source", required=True, help="Image file or directory")
    p_e2e.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold")
    p_e2e.add_argument("--iou", type=float, default=0.45, help="Detection IoU threshold")
    p_e2e.add_argument("--annotate-dir", default="", help="Optional directory to save annotated outputs")
    p_e2e.add_argument("--text-only", action="store_true", help="Print only plate texts (one per line)")
    p_e2e.add_argument("--text-out", default="", help="Optional path to write plate texts (one per line)")
    _add_ocr_backend_args(p_e2e)
    p_e2e.add_argument("--postproc", choices=["none", "indonesia"], default="indonesia", help="Apply plate post-processing")
    p_e2e.add_argument("--allowed-prefix", nargs="*", default=["A","B","D","F","E","Z","T"], help="Allowed prefixes for postproc")
    p_e2e.set_defaults(func=cmd_e2e)

    p_e2e_json = sub.add_parser("e2e-json", help="Run detector + OCR on a single image and emit JSON to stdout")
    p_e2e_json.add_argument("--det-engine", required=True, help="Path to detector TensorRT .engine")
    p_e2e_json.add_argument("--source", required=True, help="Image file path")
    p_e2e_json.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold")
    p_e2e_json.add_argument("--iou", type=float, default=0.45, help="Detection IoU threshold")
    _add_ocr_backend_args(p_e2e_json)
    p_e2e_json.add_argument("--postproc", choices=["none", "indonesia"], default="indonesia", help="Apply plate post-processing")
    p_e2e_json.add_argument("--allowed-prefix", nargs="*", default=["A","B","D","F","E","Z","T"], help="Allowed prefixes for postproc")
    p_e2e_json.set_defaults(func=cmd_e2e_json)

    p_e2e_stream = sub.add_parser(
        "e2e-json-stream",
        help="Run detector + OCR once, then read image paths from stdin and emit JSON lines",
    )
    p_e2e_stream.add_argument("--det-engine", required=True, help="Path to detector TensorRT .engine")
    p_e2e_stream.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold")
    p_e2e_stream.add_argument("--iou", type=float, default=0.45, help="Detection IoU threshold")
    _add_ocr_backend_args(p_e2e_stream)
    p_e2e_stream.add_argument("--postproc", choices=["none", "indonesia"], default="indonesia", help="Apply plate post-processing")
    p_e2e_stream.add_argument("--allowed-prefix", nargs="*", default=["A","B","D","F","E","Z","T"], help="Allowed prefixes for postproc")
    p_e2e_stream.add_argument("--stop-on-error", action="store_true", help="Stop processing on first error")
    p_e2e_stream.set_defaults(func=cmd_e2e_json_stream)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    rc = args.func(args)
    sys.exit(rc)


if __name__ == "__main__":
    main()
