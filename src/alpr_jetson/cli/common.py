from __future__ import annotations

"""Shared CLI helpers (Python 3.8-safe).

- add_ocr_backend_args: injects backend-specific options
- init_ocr_backend: constructs OCR runner from args
"""

import argparse
import os
from pathlib import Path
from typing import Tuple

DEFAULT_ONNX_MODEL = Path("models/ocr/cct_s_v1_global.onnx")
DEFAULT_PLATE_CONFIG = Path("models/ocr/cct_s_v1_global_plate_config.yaml")
DEFAULT_TRT_ENGINE = Path("models/ocr/ppo_crnn_fp16.engine")
DEFAULT_CHARSET = Path("models/ocr/charset.txt")


def _existing_path(value: str) -> str:
    if not value:
        return ""
    path = Path(value)
    if path.is_file():
        return str(path)
    return ""


def _default_onnx_pair() -> Tuple[str, str]:
    model = _existing_path(str(DEFAULT_ONNX_MODEL))
    plate = _existing_path(str(DEFAULT_PLATE_CONFIG))
    if model and plate:
        return model, plate
    return "", ""


def add_ocr_backend_args(p: argparse.ArgumentParser) -> None:
    group = p.add_mutually_exclusive_group(required=False)
    group.add_argument("--engine", help="Path to OCR TensorRT .engine (CTC)")
    group.add_argument("--onnx", help="Path to OCR ONNX model (slot-based). Defaults to models/ocr/cct_s_v1_global.onnx if present")
    p.add_argument("--charset", help="Path to charset.txt (TensorRT only)")
    p.add_argument("--plate-config", help="YAML config for ONNX slot model")
    p.add_argument("--onnx-provider", choices=["cuda", "cpu"], default="cuda", help="ONNXRuntime provider")
    p.add_argument("--onnx-gpu-mem-limit-mb", type=int, default=768, help="Cap CUDA EP allocator (MB)")
    p.add_argument("--input-width", type=int, default=160, help="Model input width (TensorRT)")
    p.add_argument("--input-height", type=int, default=32, help="Model input height (TensorRT)")
    p.add_argument("--channels", type=int, default=1, help="Input channels (TensorRT)")
    p.add_argument("--no-clahe", action="store_true", help="Disable CLAHE (TensorRT)")
    p.add_argument("--input-layout", choices=["NCHW", "NHWC"], default="NCHW", help="TensorRT engine input layout")
    p.add_argument("--logits-layout", choices=["NTC", "NCT"], default="NTC", help="TensorRT logits layout")
    p.add_argument("--blank-index", type=int, default=0, help="CTC blank index (TensorRT)")


def init_ocr_backend(args: argparse.Namespace) -> Tuple[str, object]:
    """Return (backend, runner) from parsed args.

    Raises RuntimeError/ValueError if dependencies or configs are missing.
    """
    try:
        import yaml  # type: ignore
        from ocr_service.trt_infer import OCRService  # type: ignore
        from ocr_service.preprocess import PreprocConfig  # type: ignore
        from ocr_service.onnx_infer import OnnxPlateOCR, PlateConfig  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"OCR dependencies missing: {exc}") from exc

    onnx_model = getattr(args, "onnx", "") or os.getenv("ALPR_OCR_ONNX", "")
    plate_cfg_arg = getattr(args, "plate_config", "") or os.getenv("ALPR_PLATE_CONFIG", "")
    if not onnx_model:
        default_model, default_cfg = _default_onnx_pair()
        if default_model and default_cfg:
            onnx_model = default_model
            if not plate_cfg_arg:
                plate_cfg_arg = default_cfg

    if onnx_model:
        cfg_path = plate_cfg_arg
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
            onnx_model,
            plate_cfg,
            prefer_trt=False,
            provider=getattr(args, "onnx_provider", "cuda"),
            gpu_mem_limit_mb=getattr(args, "onnx_gpu_mem_limit_mb", 768),
        )
        return "onnx", runner

    # TensorRT path
    engine_path = getattr(args, "engine", "") or os.getenv("ALPR_OCR_ENGINE", "")
    charset = getattr(args, "charset", "") or os.getenv("ALPR_OCR_CHARSET", "")
    if not engine_path:
        engine_path = _existing_path(str(DEFAULT_TRT_ENGINE))
    if not charset:
        charset = _existing_path(str(DEFAULT_CHARSET))
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
