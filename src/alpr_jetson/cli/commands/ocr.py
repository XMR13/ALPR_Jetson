from __future__ import annotations

import argparse
import sys
from pathlib import Path

from alpr_jetson.cli.common import add_ocr_backend_args, init_ocr_backend


def add_subcommand(sub):
    p_ocr = sub.add_parser("ocr-infer", help="Run OCR on an image or directory")
    add_ocr_backend_args(p_ocr)
    p_ocr.add_argument("--source", required=True, help="Image file or directory of images")
    p_ocr.add_argument("--output", default="", help="Optional CSV output path")
    p_ocr.add_argument("--postproc", choices=["none", "indonesia"], default="none", help="Apply plate post-processing")
    p_ocr.add_argument("--allowed-prefix", nargs="*", default=["A","B","D","F","E","Z","T"], help="Allowed region prefixes for postproc")
    p_ocr.set_defaults(func=cmd_ocr_infer)


def cmd_ocr_infer(args: argparse.Namespace) -> int:
    """Infer OCR text for an image or all images in a directory."""
    try:
        import glob, csv
        import cv2  # type: ignore
        from ocr_service.postprocess import postprocess_indonesia  # type: ignore
    except Exception as e:
        print(f"OCR runtime dependencies missing: {e}", file=sys.stderr)
        return 2

    try:
        backend, runner = init_ocr_backend(args)
    except Exception as exc:
        print(f"failed to initialize OCR backend: {exc}", file=sys.stderr)
        return 2

    paths = []
    src = Path(args.source)
    if src.is_dir():
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            paths.extend([str(p) for p in sorted(src.glob(ext))])
    else:
        paths = [str(src)]

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

