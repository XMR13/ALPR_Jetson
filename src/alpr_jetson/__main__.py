import argparse
import subprocess
import sys
from pathlib import Path


def cmd_ocr_infer(args: argparse.Namespace) -> int:
    """Infer OCR text for an image or all images in a directory.

    Requires TensorRT engine and charset file to be present on the host.
    Prints results to stdout and optionally writes a CSV.
    """
    try:
        import os, glob, csv
        import cv2  # type: ignore
        from ocr_service.trt_infer import OCRService  # type: ignore
    except Exception as e:
        print(f"OCR runtime dependencies missing: {e}", file=sys.stderr)
        return 2

    svc = OCRService(engine_path=args.engine, charset_path=args.charset)

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
        texts = svc.infer_batch([img])
        text = texts[0] if texts else ""
        print(f"{p}: {text}")
        results.append((p, text))

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
    p_ocr.add_argument("--engine", required=True, help="Path to OCR TensorRT .engine")
    p_ocr.add_argument("--charset", required=True, help="Path to charset.txt (one char per line)")
    p_ocr.add_argument("--source", required=True, help="Image file or directory of images")
    p_ocr.add_argument("--output", default="", help="Optional CSV output path")
    p_ocr.set_defaults(func=cmd_ocr_infer)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    rc = args.func(args)
    sys.exit(rc)


if __name__ == "__main__":
    main()
