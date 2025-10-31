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

    # Use JSON path and parse robustly to avoid mixed logs on stdout
    cmd = [
        sys.executable, "-m", "alpr_jetson", "e2e-json",
        "--det-engine", det,
        "--onnx", onnx,
        "--plate-config", plate,
        "--source", str(img),
        "--conf", str(conf),
        "--iou", str(iou),
        "--topk", str(topk),
    ]

    env = os.environ.copy()
    # Ensure package imports resolve even without editable install
    repo_src = str(Path(__file__).resolve().parents[1] / "src")
    if os.path.isdir(repo_src):
        env.setdefault("PYTHONPATH", repo_src)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env)
    if proc.returncode != 0:
        return proc.returncode

    try:
        import json
        s = (proc.stdout or "").splitlines()
        lines = [ln.strip() for ln in s if ln.strip()]
        if not lines:
            return 2
        d = json.loads(lines[-1])
    except Exception:
        return 2

    if d.get("status") != "ok" or not d.get("plates"):
        return 3
    best = d["plates"][0]
    text = (best.get("text") or "").strip()
    valid = bool(best.get("valid", False))
    if not text or not valid:
        return 3
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
