#!/usr/bin/env python3
"""Proxy text-only runner that shells out to the known-good CLI (e2e).

Why: avoids CUDA context issues by delegating to the existing pipeline.

Usage:
  python tools/alpr_text_cli.py /abs/path/image.jpg

Defaults (override via env):
  DET_ENGINE   = models/detector/yolov9-s_plate_fp16.engine
  OCR_ONNX     = models/ocr/cct_s_v1_global.onnx
  PLATE_CONFIG = models/ocr/cct_s_v1_global_plate_config.yaml
  CONF         = 0.5
  IOU          = 0.45
  TOPK         = 1
"""

from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python tools/alpr_text_cli.py /path/to/image.jpg", file=sys.stderr)
        return 2
    img = Path(sys.argv[1]).resolve()
    if not img.is_file():
        print(f"image not found: {img}", file=sys.stderr)
        return 2

    repo = Path(__file__).resolve().parents[1]
    det = os.getenv("DET_ENGINE", str(repo / "models/detector/yolov9-s_plate_fp16.engine"))
    onnx = os.getenv("OCR_ONNX", str(repo / "models/ocr/cct_s_v1_global.onnx"))
    plate = os.getenv("PLATE_CONFIG", str(repo / "models/ocr/cct_s_v1_global_plate_config.yaml"))
    conf = os.getenv("CONF", "0.5")
    iou = os.getenv("IOU", "0.45")
    topk = os.getenv("TOPK", "1")

    for p, name in ((det, "DET_ENGINE"), (onnx, "OCR_ONNX"), (plate, "PLATE_CONFIG")):
        if not Path(p).is_file():
            print(f"missing {name} file: {p}", file=sys.stderr)
            return 2

    cmd = [
        sys.executable, "-m", "alpr_jetson", "e2e",
        "--det-engine", det,
        "--onnx", onnx,
        "--plate-config", plate,
        "--source", str(img),
        "--conf", str(conf),
        "--iou", str(iou),
        "--topk", str(topk),
        "--text-only",
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # If the CLI returns non-zero, forward its rc and stderr
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return proc.returncode

    # The e2e text-only prints one line per detected plate; output the first non-empty
    for line in proc.stdout.splitlines():
        s = line.strip()
        if s:
            print(s)
            return 0

    # No text printed by e2e (no plate)
    return 3


if __name__ == "__main__":
    sys.exit(main())

