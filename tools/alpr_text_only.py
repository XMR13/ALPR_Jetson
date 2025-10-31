#!/usr/bin/env python3
"""Minimal text-only ALPR runner with predetermined models.

Usage:
  python tools/alpr_text_only.py /absolute/or/relative/path/to/image.jpg

Behavior:
  - Loads detector (TensorRT) and ONNX OCR once per call
  - Runs full E2E (det + OCR) permissively (no pre-OCR crop gating)
  - Prints only the best plate text to stdout
  - Exit codes: 0=text printed, 3=no plate/invalid text, 2=error

Environment overrides (optional):
  DET_ENGINE   = models/detector/yolov9-s_plate_fp16.engine
  OCR_ONNX     = models/ocr/cct_s_v1_global.onnx
  PLATE_CONFIG = models/ocr/cct_s_v1_global_plate_config.yaml
  CONF         = 0.5
  IOU          = 0.45
  TOPK         = 1
  ONNX_GPU_MEM_MB = 512
  RAW_FALLBACK = 1  (use ocr_raw when normalized text is empty)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


def _load_allowed_prefix() -> List[str]:
    # Defaults; optionally read configs/ocr/indonesia_prefixes.yaml if present
    defaults = ["A", "B", "D", "F", "E", "Z", "T"]
    try:
        import yaml  # type: ignore
        p = Path("configs/ocr/indonesia_prefixes.yaml")
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if isinstance(data, dict) and isinstance(data.get("prefixes"), list):
                vals = [str(x) for x in data["prefixes"] if x]
                return vals or defaults
    except Exception:
        pass
    return defaults


def main() -> int:
    if len(sys.argv) < 2:
        _err("Usage: python tools/alpr_text_only.py /path/to/image.jpg")
        return 2
    src = sys.argv[1]
    if not Path(src).is_file():
        _err(f"image not found: {src}")
        return 2

    # Read env/config
    det_engine_path = os.getenv("DET_ENGINE", "models/detector/yolov9-s_plate_fp16.engine")
    ocr_onnx_path = os.getenv("OCR_ONNX", "models/ocr/cct_s_v1_global.onnx")
    plate_cfg_path = os.getenv("PLATE_CONFIG", "models/ocr/cct_s_v1_global_plate_config.yaml")
    conf = float(os.getenv("CONF", "0.5"))
    iou = float(os.getenv("IOU", "0.45"))
    topk = int(os.getenv("TOPK", "1"))
    onnx_gpu_mem_mb = int(os.getenv("ONNX_GPU_MEM_MB", "512"))
    onnx_provider = os.getenv("ONNX_PROVIDER", "cuda").lower()
    raw_fallback = os.getenv("RAW_FALLBACK", "1") in ("1", "true", "True")

    # Validate files
    for pth, name in (
        (det_engine_path, "DET_ENGINE"),
        (ocr_onnx_path, "OCR_ONNX"),
        (plate_cfg_path, "PLATE_CONFIG"),
    ):
        if not Path(pth).is_file():
            _err(f"missing {name} file: {pth}")
            return 2

    try:
        # Lazy imports so the script stays importable without deps
        import yaml  # type: ignore
        from inference.yolov9_trt import load_engine  # type: ignore
        from pipeline.alpr_runner import run_e2e_single  # type: ignore
        from ocr_service.onnx_infer import OnnxPlateOCR, PlateConfig  # type: ignore
        from ocr_service.postprocess import postprocess_indonesia  # type: ignore
    except Exception as exc:
        _err(f"runtime dependencies missing: {exc}")
        return 2

    # Load plate config YAML for ONNX runner
    try:
        with open(plate_cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        required = ["max_plate_slots", "alphabet", "pad_char", "img_height", "img_width"]
        missing = [k for k in required if k not in cfg]
        if missing:
            _err(f"missing keys in plate config: {', '.join(missing)}")
            return 2
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
            use_clahe=bool(cfg.get("use_clahe", False)),
            clahe_clip=float(cfg.get("clahe_clip", 2.0)),
            clahe_tile=int(cfg.get("clahe_tile", 8)),
            clahe_brightness_gate=float(cfg.get("clahe_brightness_gate", 0.0)),
            auto_deskew=bool(cfg.get("auto_deskew", False)),
            deskew_threshold_deg=float(cfg.get("deskew_threshold_deg", 12.0)),
        )
    except Exception as exc:
        _err(f"failed to read plate config: {exc}")
        return 2

    # Initialize ONNX first to reduce CUDA primary-context conflicts, then TRT
    try:
        ocr_runner = OnnxPlateOCR(
            ocr_onnx_path,
            plate_cfg,
            prefer_trt=False,
            provider=onnx_provider,
            gpu_mem_limit_mb=onnx_gpu_mem_mb,
        )
    except Exception as exc:
        # Fallback to CPU if CUDA init fails and provider was not forced to CPU
        if onnx_provider != "cpu":
            try:
                ocr_runner = OnnxPlateOCR(
                    ocr_onnx_path,
                    plate_cfg,
                    prefer_trt=False,
                    provider="cpu",
                    gpu_mem_limit_mb=None,
                )
            except Exception as exc2:
                _err(f"failed to init ONNX OCR: {exc2}")
                return 2
        else:
            _err(f"failed to init ONNX OCR: {exc}")
            return 2

    try:
        det_engine = load_engine(det_engine_path, print_plugins=False)
    except Exception as exc:
        _err(f"failed to load detector engine: {exc}")
        return 2

    try:
        result: Dict[str, Any] = run_e2e_single(
            src,
            det_engine=det_engine,
            ocr_runner=ocr_runner,
            backend="onnx",
            conf=conf,
            iou=iou,
            postproc="indonesia",
            allowed_prefix=_load_allowed_prefix(),
            postprocess_fn=postprocess_indonesia,
            # Permissive like legacy e2e
            accept_all=True,
            topk=topk,
        )
    except Exception as exc:
        _err(f"pipeline error: {exc}")
        return 2

    plates: List[Dict[str, Any]] = list(result.get("plates") or [])
    if not plates:
        return 3
    best = plates[0]
    text = (best.get("text") or "").strip()
    if not text and raw_fallback:
        text = (best.get("ocr_raw") or "").strip()
    if not text:
        return 3
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
