"""ALPR API server with persistence, queue bridge, and metrics.

Implements the HTTP surface described in plan.md §5:
- `/v1/alpr` synchronous detector + OCR pipeline (multipart upload)
- `/v1/crops` asynchronous crop ingestion (HTTP baseline for DS → OCR bridge)
- SQLite persistence for per-plate events and plate snapshots
- Prometheus metrics exposing queue depth, event counts, and latency stats

The module remains importable without FastAPI installed; `create_app` returns
`None` in that scenario so tests can import the package safely on minimal envs.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

try:  # Web framework
    from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket
    from fastapi.responses import JSONResponse, PlainTextResponse
    from pydantic import BaseModel, Field, validator
except Exception:  # FastAPI not installed yet
    FastAPI = None  # type: ignore
    JSONResponse = None  # type: ignore
    PlainTextResponse = None  # type: ignore
    HTTPException = Exception  # type: ignore
    WebSocket = object  # type: ignore
    UploadFile = object  # type: ignore
    File = object  # type: ignore
    Form = object  # type: ignore
    Request = object  # type: ignore
    BackgroundTasks = object  # type: ignore
    BaseModel = object  # type: ignore

    def validator(*args, **kwargs):  # type: ignore
        def _wrap(fn):
            return fn

        return _wrap

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

from .db import EventRecord, EventStore
from pipeline.track_aggregator import TrackAggregator

try:
    from inference.yolov9_trt import (
        _prepare_image,
        decode_trt_detections,
        load_engine,
    )
except Exception:  # pragma: no cover
    _prepare_image = None  # type: ignore
    decode_trt_detections = None  # type: ignore
    load_engine = None  # type: ignore

try:
    from ocr_service.postprocess import (
        postprocess_indonesia,  # type: ignore
        load_postprocess_config,  # type: ignore
        PostprocessTuning,  # type: ignore
    )
except Exception:  # pragma: no cover
    postprocess_indonesia = None  # type: ignore
    load_postprocess_config = None  # type: ignore
    PostprocessTuning = object  # type: ignore


LOGGER = logging.getLogger(__name__)


def _env_list(name: str, default: Optional[Sequence[str]] = None) -> List[str]:
    val = os.getenv(name)
    if not val:
        return list(default or [])
    return [item.strip() for item in val.split(",") if item.strip()]


def _safe_fragment(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]", "-", value) or "anon"


def _parse_ts_iso(ts: Optional[str]) -> datetime:
    if ts:
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            pass
    return datetime.now(timezone.utc)


def _generate_snapshot_path(base: Path, camera_id: str, dt: datetime, request_id: str, sequence_id: int, track_id: Optional[int]) -> Path:
    folder = base / _safe_fragment(camera_id) / dt.strftime("%Y") / dt.strftime("%m") / dt.strftime("%d")
    name = f"{_safe_fragment(request_id)}_seq{sequence_id}"
    if track_id is not None:
        name += f"_track{track_id}"
    return folder / f"{name}.jpg"


def _write_snapshot(path: Path, image: Any, quality: int) -> bool:
    if cv2 is None:
        raise RuntimeError("opencv-python is required to write snapshots")
    path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]))


def _compute_plate_conf(det_conf: float, char_confs: Sequence[float]) -> float:
    if not char_confs:
        return float(det_conf)
    avg_char = sum(char_confs) / max(1, len(char_confs))
    return float(det_conf) * float(avg_char)


def _decode_base64_image(data: str) -> Tuple[bytes, Optional[Any]]:
    raw = base64.b64decode(data, validate=True)
    if np is None or cv2 is None:
        return raw, None
    buf = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return raw, img


def _infer_ocr_single(
    runtime: "_Runtime",
    crop: Any,
    polygon_xy: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[str, List[float]]:
    runner = runtime.ocr_runner
    if runner is None:
        raise RuntimeError("OCR runner not initialised")
    if runtime.ocr_mode == "onnx":
        res = runner.infer_batch([crop], return_confidence=True, polygons=[polygon_xy])  # type: ignore[attr-defined]
        if isinstance(res, tuple) and len(res) == 2:
            texts, confs = res  # type: ignore[misc]
            text = texts[0] if texts else ""
            return text, list(confs[0]) if confs else []
        texts = list(res) if isinstance(res, Iterable) else []  # type: ignore[arg-type]
        return (texts[0] if texts else "", [])
    texts = runner.infer_batch([crop], polygons=[polygon_xy])  # type: ignore[attr-defined]
    return (texts[0] if texts else "", [])


@dataclass
class AppConfig:
    det_engine: str = ""
    ocr_engine: str = ""
    charset_path: str = ""
    ocr_input_width: int = 160
    ocr_input_height: int = 32
    ocr_channels: int = 1
    ocr_no_clahe: bool = False
    ocr_logits_layout: str = "NTC"
    ocr_input_layout: str = "NCHW"
    ocr_blank_index: int = 0
    ocr_onnx: str = ""
    plate_config: str = ""
    onnx_provider: str = "cuda"
    onnx_gpu_mem_limit_mb: int = 512
    auth_token: str = ""
    default_camera_id: str = "cam01"
    min_conf: float = 0.5
    allowed_prefixes: List[str] = field(default_factory=lambda: ["A", "B", "D", "F", "E", "Z", "T"])
    events_db: str = "export/events.sqlite"
    snapshots_dir: str = "export/snapshots"
    snapshot_quality: int = 90
    crop_queue_size: int = 64
    max_upload_bytes: int = 2_000_000
    vote_window: int = 8
    vote_min_consensus: int = 3
    # Postprocess tuning
    postproc_config: str = ""
    postproc_strict: bool = False
    # OCR preproc adaptive toggles
    ocr_auto_preproc: bool = True
    ocr_auto_color_cast: bool = True
    ocr_gamma_correction: bool = True
    ocr_gamma_dark_gate: float = 90.0
    ocr_gamma_value: float = 1.15
    ocr_auto_polarity: bool = True
    ocr_polarity_dark_mean: float = 110.0
    ocr_polarity_light_mean: float = 175.0
    ocr_invert_grayscale: bool = False
    # TRT OCR preprocess toggles (env-configurable)
    ocr_clahe_brightness_gate: float = 0.0
    ocr_suppress_highlights: bool = False
    ocr_highlight_threshold: int = 245
    ocr_highlight_inpaint_radius: int = 0
    ocr_remove_small_bright_specks: bool = False
    ocr_speck_area_px: int = 8


def _config_from_env() -> AppConfig:
    cfg = AppConfig()
    cfg.det_engine = os.getenv("ALPR_DET_ENGINE", cfg.det_engine)
    cfg.ocr_engine = os.getenv("ALPR_OCR_ENGINE", cfg.ocr_engine)
    cfg.charset_path = os.getenv("ALPR_OCR_CHARSET", cfg.charset_path)
    cfg.ocr_input_width = int(os.getenv("ALPR_OCR_INPUT_WIDTH", cfg.ocr_input_width))
    cfg.ocr_input_height = int(os.getenv("ALPR_OCR_INPUT_HEIGHT", cfg.ocr_input_height))
    cfg.ocr_channels = int(os.getenv("ALPR_OCR_CHANNELS", cfg.ocr_channels))
    cfg.ocr_no_clahe = os.getenv("ALPR_OCR_NO_CLAHE", "0") in {"1", "true", "True"}
    cfg.ocr_logits_layout = os.getenv("ALPR_OCR_LOGITS_LAYOUT", cfg.ocr_logits_layout)
    cfg.ocr_input_layout = os.getenv("ALPR_OCR_INPUT_LAYOUT", cfg.ocr_input_layout)
    cfg.ocr_blank_index = int(os.getenv("ALPR_OCR_BLANK_INDEX", cfg.ocr_blank_index))
    cfg.ocr_onnx = os.getenv("ALPR_OCR_ONNX", cfg.ocr_onnx)
    cfg.plate_config = os.getenv("ALPR_PLATE_CONFIG", cfg.plate_config)
    cfg.onnx_provider = os.getenv("ALPR_ONNX_PROVIDER", cfg.onnx_provider)
    cfg.onnx_gpu_mem_limit_mb = int(os.getenv("ALPR_ONNX_GPU_MEM_MB", cfg.onnx_gpu_mem_limit_mb))
    cfg.auth_token = os.getenv("ALPR_API_TOKEN", cfg.auth_token)
    cfg.default_camera_id = os.getenv("ALPR_DEFAULT_CAMERA_ID", cfg.default_camera_id)
    cfg.min_conf = float(os.getenv("ALPR_MIN_CONF", cfg.min_conf))
    cfg.allowed_prefixes = _env_list("ALPR_ALLOWED_PREFIXES", cfg.allowed_prefixes)
    cfg.events_db = os.getenv("ALPR_EVENTS_DB", cfg.events_db)
    cfg.snapshots_dir = os.getenv("ALPR_SNAPSHOTS_DIR", cfg.snapshots_dir)
    cfg.snapshot_quality = int(os.getenv("ALPR_SNAPSHOT_QUALITY", cfg.snapshot_quality))
    cfg.crop_queue_size = max(0, int(os.getenv("ALPR_CROP_QUEUE", cfg.crop_queue_size)))
    cfg.max_upload_bytes = int(os.getenv("ALPR_MAX_UPLOAD_BYTES", cfg.max_upload_bytes))
    cfg.vote_window = max(1, int(os.getenv("ALPR_VOTE_WINDOW", cfg.vote_window)))
    cfg.vote_min_consensus = max(1, int(os.getenv("ALPR_VOTE_MIN", cfg.vote_min_consensus)))
    cfg.postproc_config = os.getenv("ALPR_POSTPROC_CONFIG", cfg.postproc_config)
    cfg.postproc_strict = os.getenv("ALPR_POSTPROC_STRICT", "0").lower() in {"1", "true", "yes"}
    # Preprocess env toggles (TRT OCR)
    cfg.ocr_clahe_brightness_gate = float(os.getenv("ALPR_OCR_CLAHE_GATE", cfg.ocr_clahe_brightness_gate))
    cfg.ocr_suppress_highlights = os.getenv("ALPR_OCR_SUPPRESS_HL", "0").lower() in {"1", "true", "yes"}
    cfg.ocr_highlight_threshold = int(os.getenv("ALPR_OCR_HL_THRESHOLD", cfg.ocr_highlight_threshold))
    cfg.ocr_highlight_inpaint_radius = int(os.getenv("ALPR_OCR_HL_INPAINT", cfg.ocr_highlight_inpaint_radius))
    cfg.ocr_remove_small_bright_specks = os.getenv("ALPR_OCR_REMOVE_SPECKS", "0").lower() in {"1", "true", "yes"}
    cfg.ocr_speck_area_px = int(os.getenv("ALPR_OCR_SPECK_AREA", cfg.ocr_speck_area_px))
    cfg.ocr_auto_preproc = os.getenv("ALPR_OCR_AUTO_PREPROC", "1").lower() in {"1", "true", "yes"}
    cfg.ocr_auto_color_cast = os.getenv("ALPR_OCR_AUTO_COLOR", "1").lower() in {"1", "true", "yes"}
    cfg.ocr_gamma_correction = os.getenv("ALPR_OCR_GAMMA", "1").lower() in {"1", "true", "yes"}
    cfg.ocr_gamma_dark_gate = float(os.getenv("ALPR_OCR_GAMMA_GATE", cfg.ocr_gamma_dark_gate))
    cfg.ocr_gamma_value = float(os.getenv("ALPR_OCR_GAMMA_VALUE", cfg.ocr_gamma_value))
    cfg.ocr_auto_polarity = os.getenv("ALPR_OCR_AUTO_POLARITY", "1").lower() in {"1", "true", "yes"}
    cfg.ocr_polarity_dark_mean = float(os.getenv("ALPR_OCR_POLARITY_DARK", cfg.ocr_polarity_dark_mean))
    cfg.ocr_polarity_light_mean = float(os.getenv("ALPR_OCR_POLARITY_LIGHT", cfg.ocr_polarity_light_mean))
    cfg.ocr_invert_grayscale = os.getenv("ALPR_OCR_INVERT", "0").lower() in {"1", "true", "yes"}
    return cfg


@dataclass
class _Runtime:
    det_model: Any = None
    ocr_runner: Any = None
    ocr_mode: str = ""


@dataclass
class RuntimeErrorState:
    message: str
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CropTask:
    camera_id: str
    request_id: str
    crop: Any
    det_conf: float
    bbox: Tuple[int, int, int, int]
    ts: str
    frame_id: Optional[int] = None
    track_id: Optional[int] = None
    sequence_id: int = 0
    polygon_xy: Optional[List[Tuple[float, float]]] = None


def _initialize_runtime(cfg: AppConfig) -> Tuple[Optional[_Runtime], Optional[RuntimeErrorState]]:
    if cv2 is None or np is None:
        return None, RuntimeErrorState("opencv-python and numpy are required for OCR")
    if load_engine is None or decode_trt_detections is None or _prepare_image is None:
        return None, RuntimeErrorState("TensorRT detection utilities unavailable")
    if not cfg.det_engine:
        return None, RuntimeErrorState("ALPR_DET_ENGINE not configured")

    runtime = _Runtime()
    try:
        runtime.det_model = load_engine(cfg.det_engine, print_plugins=False)  # type: ignore[arg-type]
    except Exception as exc:  # pragma: no cover
        return None, RuntimeErrorState(f"failed to load detector engine: {exc}")

    if cfg.ocr_engine and cfg.charset_path:
        try:
            from ocr_service.preprocess import PreprocConfig  # type: ignore
            from ocr_service.trt_infer import OCRService  # type: ignore

            preproc = PreprocConfig(
                input_width=cfg.ocr_input_width,
                input_height=cfg.ocr_input_height,
                channels=cfg.ocr_channels,
                mean=0.5,
                std=0.5,
                use_clahe=not cfg.ocr_no_clahe,
                clahe_brightness_gate=cfg.ocr_clahe_brightness_gate,
                suppress_highlights=cfg.ocr_suppress_highlights,
                highlight_threshold=cfg.ocr_highlight_threshold,
                highlight_inpaint_radius=cfg.ocr_highlight_inpaint_radius,
                remove_small_bright_specks=cfg.ocr_remove_small_bright_specks,
                speck_area_px=cfg.ocr_speck_area_px,
                auto_preproc=cfg.ocr_auto_preproc,
                auto_color_cast=cfg.ocr_auto_color_cast,
                gamma_correction=cfg.ocr_gamma_correction,
                gamma_dark_gate=cfg.ocr_gamma_dark_gate,
                gamma_value=cfg.ocr_gamma_value,
                auto_polarity=cfg.ocr_auto_polarity,
                polarity_dark_mean=cfg.ocr_polarity_dark_mean,
                polarity_light_mean=cfg.ocr_polarity_light_mean,
                invert_grayscale=cfg.ocr_invert_grayscale,
            )
            runtime.ocr_runner = OCRService(
                engine_path=cfg.ocr_engine,
                charset_path=cfg.charset_path,
                preproc=preproc,
                logits_layout=cfg.ocr_logits_layout,
                input_layout=cfg.ocr_input_layout,
                blank_index=cfg.ocr_blank_index,
            )
            runtime.ocr_mode = "trt"
        except Exception as exc:  # pragma: no cover
            return None, RuntimeErrorState(f"failed to load OCR TensorRT engine: {exc}")
    elif cfg.ocr_onnx and cfg.plate_config:
        if yaml is None:
            return None, RuntimeErrorState("pyyaml is required for ONNX OCR plate config")
        try:
            from ocr_service.onnx_infer import OnnxPlateOCR, PlateConfig  # type: ignore
        except Exception as exc:  # pragma: no cover
            return None, RuntimeErrorState(f"ONNX OCR runtime unavailable: {exc}")
        try:
            with open(cfg.plate_config, "r", encoding="utf-8") as f:
                plate_yaml = yaml.safe_load(f)
        except Exception as exc:
            return None, RuntimeErrorState(f"failed to read plate config: {exc}")

        required = ["max_plate_slots", "alphabet", "pad_char", "img_height", "img_width"]
        missing = [k for k in required if k not in plate_yaml]
        if missing:
            return None, RuntimeErrorState(f"missing keys in plate config: {', '.join(missing)}")
        try:
            plate_cfg = PlateConfig(
                max_plate_slots=int(plate_yaml["max_plate_slots"]),
                alphabet=str(plate_yaml["alphabet"]),
                pad_char=str(plate_yaml["pad_char"]),
                img_height=int(plate_yaml["img_height"]),
                img_width=int(plate_yaml["img_width"]),
                keep_aspect_ratio=bool(plate_yaml.get("keep_aspect_ratio", True)),
                interpolation=str(plate_yaml.get("interpolation", "area")),
                image_color_mode=str(plate_yaml.get("image_color_mode", "grayscale")),
                padding_color=plate_yaml.get("padding_color", (144, 144, 144)),
                use_clahe=bool(plate_yaml.get("use_clahe", False)),
                clahe_clip=float(plate_yaml.get("clahe_clip", 2.0)),
                clahe_tile=int(plate_yaml.get("clahe_tile", 8)),
                clahe_brightness_gate=float(plate_yaml.get("clahe_brightness_gate", 0.0)),
                auto_deskew=bool(plate_yaml.get("auto_deskew", False)),
                deskew_threshold_deg=float(plate_yaml.get("deskew_threshold_deg", 12.0)),
            )
            runtime.ocr_runner = OnnxPlateOCR(
                cfg.ocr_onnx,
                plate_cfg,
                prefer_trt=False,
                provider=cfg.onnx_provider,
                gpu_mem_limit_mb=cfg.onnx_gpu_mem_limit_mb,
            )
            runtime.ocr_mode = "onnx"
        except Exception as exc:  # pragma: no cover
            return None, RuntimeErrorState(f"failed to initialize ONNX OCR: {exc}")
    else:
        return None, RuntimeErrorState(
            "OCR backend not configured (set ALPR_OCR_ENGINE+ALPR_OCR_CHARSET or ALPR_OCR_ONNX+ALPR_PLATE_CONFIG)"
        )

    return runtime, None


@dataclass
class _State:
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_frame_ts: Optional[str] = None
    fps: float = 0.0
    queue_len: int = 0
    queue_processed: int = 0
    queue_dropped: int = 0
    queue_errors: int = 0
    gpu_util: float = 0.0
    events: List[Dict[str, Any]] = field(default_factory=list)
    events_total: int = 0
    events_failures: int = 0
    webhooks: List[Dict[str, Any]] = field(default_factory=list)
    total_requests: int = 0
    last_latency_ms: float = 0.0
    last_status_ok: int = 0
    runtime: Optional[_Runtime] = None
    runtime_error: Optional[RuntimeErrorState] = None
    config: AppConfig = field(default_factory=AppConfig)
    event_store: Optional[EventStore] = None
    snapshots_dir: Optional[Path] = None
    crop_queue: Optional[asyncio.Queue[CropTask]] = None
    crop_worker: Optional[asyncio.Task[Any]] = None
    loop: Optional[asyncio.AbstractEventLoop] = None
    aggregator: TrackAggregator = field(default_factory=TrackAggregator)
    postproc_tuning: Optional[PostprocessTuning] = None


def create_app(cfg: Optional[AppConfig] = None) -> "Optional[FastAPI]":
    if FastAPI is None:
        return None

    config = cfg or _config_from_env()
    app = FastAPI(title="ALPR API", version="0.1.0")

    runtime, err = _initialize_runtime(config)

    snapshots_dir: Optional[Path] = None
    if config.snapshots_dir:
        snapshots_dir = Path(config.snapshots_dir)
        snapshots_dir.mkdir(parents=True, exist_ok=True)

    event_store: Optional[EventStore] = None
    if config.events_db:
        try:
            event_store = EventStore(Path(config.events_db))
        except Exception as exc:  # pragma: no cover
            LOGGER.error("failed to init EventStore: %s", exc)
            err = err or RuntimeErrorState(f"event store unavailable: {exc}")

    state = _State(
        runtime=runtime,
        runtime_error=err,
        config=config,
        event_store=event_store,
        snapshots_dir=snapshots_dir,
        aggregator=TrackAggregator(window=config.vote_window, min_consensus=config.vote_min_consensus),
    )

    # Load optional OCR postprocess tuning YAML
    if postprocess_indonesia is not None and load_postprocess_config is not None and config.postproc_config:
        try:
            state.postproc_tuning = load_postprocess_config(config.postproc_config)  # type: ignore[misc]
            LOGGER.info("Loaded postprocess config: %s", config.postproc_config)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to load postprocess config %s: %s", config.postproc_config, exc)

    async def _persist_event_records(records: List[EventRecord]) -> None:
        if not records or state.event_store is None:
            return
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, state.event_store.insert_many, records)
            state.events_total += len(records)
        except Exception as exc:  # pragma: no cover
            state.events_failures += len(records)
            LOGGER.error("failed to persist events: %s", exc)

    async def _save_snapshot(
        crop: Any,
        camera_id: str,
        request_id: str,
        ts: str,
        sequence_id: int,
        track_id: Optional[int],
    ) -> Optional[str]:
        if crop is None or state.snapshots_dir is None or cv2 is None:
            return None
        dt = _parse_ts_iso(ts)
        path = _generate_snapshot_path(state.snapshots_dir, camera_id, dt, request_id, sequence_id, track_id)
        loop = asyncio.get_running_loop()

        def _write() -> bool:
            return _write_snapshot(path, crop, state.config.snapshot_quality)

        success = await loop.run_in_executor(None, _write)
        return str(path) if success else None

    async def _record_events(
        *,
        request_id: str,
        camera_id: str,
        ts: str,
        status: str,
        detections: Sequence[Dict[str, Any]],
        crops: Sequence[Any],
    ) -> None:
        records: List[EventRecord] = []
        for idx, det in enumerate(detections):
            bbox = det.get("bbox", [0, 0, 0, 0])
            snapshot_path = await _save_snapshot(
                crop=crops[idx] if idx < len(crops) else None,
                camera_id=camera_id,
                request_id=request_id,
                ts=ts,
                sequence_id=idx,
                track_id=det.get("track_id"),
            )
            records.append(
                EventRecord(
                    request_id=request_id,
                    camera_id=camera_id,
                    ts=ts,
                    status=status,
                    plate=str(det.get("text", "")),
                    plate_conf=float(det.get("plate_conf", 0.0)),
                    det_conf=float(det.get("det_conf", 0.0)),
                    valid=bool(det.get("valid", False)),
                    bbox=bbox,
                    track_id=det.get("track_id"),
                    frame_id=det.get("frame_id"),
                    snapshot_path=snapshot_path,
                    char_confs=det.get("char_confs", []),
                    raw_event={**det, "camera_id": camera_id, "request_id": request_id, "status": status},
                )
            )

            # Keep a rolling in-memory list for quick GET /v1/events
            event_repr = {
                "request_id": request_id,
                "camera_id": camera_id,
                "ts": ts,
                **det,
                "snapshot_path": snapshot_path,
            }
            state.events.append(event_repr)
        # Limit in-memory cache to 500 entries
        if len(state.events) > 500:
            state.events = state.events[-500:]

        await _persist_event_records(records)

    async def _process_crop_task(task: CropTask) -> None:
        if state.runtime is None or state.runtime_error is not None:
            raise RuntimeError(state.runtime_error.message if state.runtime_error else "runtime unavailable")
        if postprocess_indonesia is None:
            raise RuntimeError("postprocess module unavailable")
        loop = asyncio.get_running_loop()
        text, char_confs = await loop.run_in_executor(
            None, _infer_ocr_single, state.runtime, task.crop, task.polygon_xy
        )
        norm_text, is_valid = postprocess_indonesia(
            text,
            allowed_prefix=state.config.allowed_prefixes or None,
            tuning=state.postproc_tuning,  # type: ignore[arg-type]
            char_confs=char_confs,
            strict=bool(state.config.postproc_strict),
        )
        plate_conf = _compute_plate_conf(task.det_conf, char_confs)
        detection_payload = {
            "bbox": list(task.bbox),
            "det_conf": float(task.det_conf),
            "ocr_raw": text,
            "text": norm_text,
            "valid": bool(is_valid),
            "plate_conf": plate_conf,
            "char_confs": list(char_confs),
            "track_id": task.track_id,
            "frame_id": task.frame_id,
        }
        await _record_events(
            request_id=task.request_id,
            camera_id=task.camera_id,
            ts=task.ts,
            status="queued",
            detections=[detection_payload],
            crops=[task.crop],
        )

        # Feed aggregator to emit stabilized events when consensus reached
        try:
            w = max(0, task.bbox[2] - task.bbox[0])
            h = max(0, task.bbox[3] - task.bbox[1])
            event = state.aggregator.update(
                track_id=int(task.track_id or 0),
                text=norm_text,
                conf=plate_conf,
                bbox=(task.bbox[0], task.bbox[1], w, h),
                frame_id=int(task.frame_id or -1),
                camera_id=task.camera_id,
                ts_iso=task.ts,
                char_confs=list(char_confs),
            )
        except Exception as exc:  # pragma: no cover
            LOGGER.debug("aggregator update failed: %s", exc)
            event = None

        if event:
            await _record_events(
                request_id=task.request_id,
                camera_id=task.camera_id,
                ts=event.get("ts", task.ts),
                status="ok",
                detections=[
                    {
                        "bbox": event.get("bbox", [0, 0, 0, 0]),
                        "det_conf": task.det_conf,
                        "ocr_raw": text,
                        "text": event.get("plate", norm_text),
                        "valid": True,
                        "plate_conf": event.get("plate_conf", plate_conf),
                        "char_confs": list(char_confs),
                        "track_id": event.get("track_id"),
                        "frame_id": event.get("frame_id"),
                    }
                ],
                crops=[task.crop],
            )

        state.last_frame_ts = task.ts

    async def _crop_worker() -> None:
        assert state.crop_queue is not None
        while True:
            task = await state.crop_queue.get()
            try:
                await _process_crop_task(task)
                state.queue_processed += 1
            except Exception as exc:  # pragma: no cover
                state.queue_errors += 1
                LOGGER.error("crop worker error: %s", exc)
            finally:
                state.queue_len = state.crop_queue.qsize()
                state.crop_queue.task_done()

    @app.on_event("startup")
    async def _startup() -> None:  # pragma: no cover - exercised in integration tests
        loop = asyncio.get_running_loop()
        state.loop = loop
        if state.config.crop_queue_size > 0:
            state.crop_queue = asyncio.Queue(maxsize=state.config.crop_queue_size)
            state.crop_worker = loop.create_task(_crop_worker())

    @app.on_event("shutdown")
    async def _shutdown() -> None:  # pragma: no cover
        if state.crop_worker is not None:
            state.crop_worker.cancel()
            try:
                await state.crop_worker
            except Exception:
                pass
        if state.event_store is not None:
            state.event_store.close()

    @app.get("/healthz")
    def healthz():  # type: ignore[no-redef]
        now = datetime.now(timezone.utc)
        uptime = (now - state.started_at).total_seconds()
        return JSONResponse(
            {
                "status": "ok",
                "uptime_s": uptime,
                "gpu": {"util": state.gpu_util},
                "last_frame_ts": state.last_frame_ts,
                "runtime_ready": state.runtime is not None and state.runtime_error is None,
                "runtime_error": state.runtime_error.message if state.runtime_error else None,
                "queue_len": state.queue_len,
                "queue_dropped": state.queue_dropped,
                "queue_errors": state.queue_errors,
                "events_total": state.events_total,
                "events_failures": state.events_failures,
            }
        )

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics():  # type: ignore[no-redef]
        lines = [
            "# HELP alpr_fps Current estimated frames per second",
            "# TYPE alpr_fps gauge",
            f"alpr_fps {state.fps}",
            "# HELP alpr_queue_len Current crop queue depth",
            "# TYPE alpr_queue_len gauge",
            f"alpr_queue_len {state.queue_len}",
            "# HELP alpr_queue_processed_total Crops processed by queue worker",
            "# TYPE alpr_queue_processed_total counter",
            f"alpr_queue_processed_total {state.queue_processed}",
            "# HELP alpr_queue_dropped_total Crops dropped due to full queue",
            "# TYPE alpr_queue_dropped_total counter",
            f"alpr_queue_dropped_total {state.queue_dropped}",
            "# HELP alpr_queue_errors_total Errors during queue processing",
            "# TYPE alpr_queue_errors_total counter",
            f"alpr_queue_errors_total {state.queue_errors}",
            "# HELP alpr_gpu_util GPU utilization percent",
            "# TYPE alpr_gpu_util gauge",
            f"alpr_gpu_util {state.gpu_util}",
            "# HELP alpr_requests_total Total synchronous /v1/alpr requests",
            "# TYPE alpr_requests_total counter",
            f"alpr_requests_total {state.total_requests}",
            "# HELP alpr_last_latency_ms Last synchronous /v1/alpr total latency in ms",
            "# TYPE alpr_last_latency_ms gauge",
            f"alpr_last_latency_ms {state.last_latency_ms}",
            "# HELP alpr_last_status_ok 1 if last /v1/alpr request succeeded",
            "# TYPE alpr_last_status_ok gauge",
            f"alpr_last_status_ok {state.last_status_ok}",
            "# HELP alpr_events_total Events persisted to SQLite",
            "# TYPE alpr_events_total counter",
            f"alpr_events_total {state.events_total}",
            "# HELP alpr_events_failures_total Event persistence failures",
            "# TYPE alpr_events_failures_total counter",
            f"alpr_events_failures_total {state.events_failures}",
        ]
        return "\n".join(lines) + "\n"

    if isinstance(BaseModel, type):

        class PreprocUpdate(BaseModel):  # type: ignore[misc]
            clahe_brightness_gate: Optional[float] = None
            suppress_highlights: Optional[bool] = None
            highlight_threshold: Optional[int] = None
            highlight_inpaint_radius: Optional[int] = None
            remove_small_bright_specks: Optional[bool] = None
            speck_area_px: Optional[int] = None
            auto_preproc: Optional[bool] = None
            glare_frac_gate: Optional[float] = None
            auto_highlight_quantile: Optional[float] = None
            speck_area_frac_gate: Optional[float] = None

    @app.post("/v1/config/preproc")
    def update_preproc(payload: Dict[str, Any]):  # type: ignore[no-redef]
        # Update runtime config in-memory without restart
        c = state.config
        data = payload or {}
        def _bool(v: Any) -> Optional[bool]:
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                return v.lower() in {"1", "true", "yes"}
            if v is None:
                return None
            return bool(v)

        def _set(name: str, value: Any):
            if hasattr(c, name) and value is not None:
                setattr(c, name, value)

        _set("ocr_clahe_brightness_gate", float(data.get("clahe_brightness_gate")) if data.get("clahe_brightness_gate") is not None else None)
        _set("ocr_suppress_highlights", _bool(data.get("suppress_highlights")))
        _set("ocr_highlight_threshold", int(data.get("highlight_threshold")) if data.get("highlight_threshold") is not None else None)
        _set("ocr_highlight_inpaint_radius", int(data.get("highlight_inpaint_radius")) if data.get("highlight_inpaint_radius") is not None else None)
        _set("ocr_remove_small_bright_specks", _bool(data.get("remove_small_bright_specks")))
        _set("ocr_speck_area_px", int(data.get("speck_area_px")) if data.get("speck_area_px") is not None else None)
        _set("ocr_auto_preproc", _bool(data.get("auto_preproc")))
        _set("ocr_auto_color_cast", _bool(data.get("auto_color_cast")))
        _set("ocr_gamma_correction", _bool(data.get("gamma_correction")))
        _set("ocr_gamma_dark_gate", float(data.get("gamma_dark_gate")) if data.get("gamma_dark_gate") is not None else None)
        _set("ocr_gamma_value", float(data.get("gamma_value")) if data.get("gamma_value") is not None else None)
        _set("ocr_auto_polarity", _bool(data.get("auto_polarity")))
        _set("ocr_polarity_dark_mean", float(data.get("polarity_dark_mean")) if data.get("polarity_dark_mean") is not None else None)
        _set("ocr_polarity_light_mean", float(data.get("polarity_light_mean")) if data.get("polarity_light_mean") is not None else None)
        _set("ocr_invert_grayscale", _bool(data.get("invert_grayscale")))

        # Rebuild TRT preprocess config if present
        if state.runtime and state.runtime.ocr_mode == "trt" and state.runtime.ocr_runner is not None:
            try:
                from ocr_service.preprocess import PreprocConfig  # type: ignore
                pp = PreprocConfig(
                    input_width=c.ocr_input_width,
                    input_height=c.ocr_input_height,
                    channels=c.ocr_channels,
                    mean=0.5,
                    std=0.5,
                    use_clahe=not c.ocr_no_clahe,
                    clahe_brightness_gate=c.ocr_clahe_brightness_gate,
                    suppress_highlights=c.ocr_suppress_highlights,
                    highlight_threshold=c.ocr_highlight_threshold,
                    highlight_inpaint_radius=c.ocr_highlight_inpaint_radius,
                    remove_small_bright_specks=c.ocr_remove_small_bright_specks,
                    speck_area_px=c.ocr_speck_area_px,
                    auto_preproc=c.ocr_auto_preproc,
                    auto_color_cast=c.ocr_auto_color_cast,
                    gamma_correction=c.ocr_gamma_correction,
                    gamma_dark_gate=c.ocr_gamma_dark_gate,
                    gamma_value=c.ocr_gamma_value,
                    auto_polarity=c.ocr_auto_polarity,
                    polarity_dark_mean=c.ocr_polarity_dark_mean,
                    polarity_light_mean=c.ocr_polarity_light_mean,
                    invert_grayscale=c.ocr_invert_grayscale,
                )
                state.runtime.ocr_runner.preproc = pp  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover
                raise HTTPException(status_code=500, detail=f"failed to update preprocess: {exc}")
        return JSONResponse({"ok": True})

    @app.get("/v1/stream/info")
    def stream_info():  # type: ignore[no-redef]
        return JSONResponse(
            {
                "camera_id": state.config.default_camera_id,
                "fps": state.fps,
                "buffer_backlog": state.queue_len,
                "last_frame_ts": state.last_frame_ts,
                "runtime_ready": state.runtime is not None and state.runtime_error is None,
            }
        )

    @app.post("/v1/hooks")
    def register_hook(payload: Dict[str, Any]):  # type: ignore[no-redef]
        url = payload.get("url")
        if not url:
            raise HTTPException(status_code=400, detail="missing url")
        hook = {
            "url": url,
            "secret": payload.get("secret", ""),
            "retries": int(payload.get("retries", 3)),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        state.webhooks.append(hook)
        return JSONResponse({"ok": True, "count": len(state.webhooks)})

    @app.get("/v1/events")
    def list_events(since: Optional[str] = None, limit: int = 100):  # type: ignore[no-redef]
        events = state.events
        if since:
            try:
                since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
            except Exception:
                raise HTTPException(status_code=400, detail="invalid since ISO timestamp")
            events = [e for e in events if datetime.fromisoformat(e["ts"].replace("Z", "+00:00")) >= since_dt]
        lim = max(0, min(1000, int(limit)))
        return JSONResponse({"events": events[:lim]})

    @app.websocket("/v1/ws")
    async def ws_endpoint(ws: WebSocket):  # type: ignore[no-redef]
        await ws.accept()
        try:
            await ws.send_text("ready")
            while True:
                _ = await ws.receive_text()
                await ws.send_text("pong")
        except Exception:  # pragma: no cover
            pass

    if isinstance(BaseModel, type):

        class CropIn(BaseModel):  # type: ignore
            request_id: str = Field(..., description="Unique identifier per detection batch")
            crop_b64: str = Field(..., description="Base64 encoded JPEG/PNG crop")
            camera_id: str = Field(default_factory=lambda: state.config.default_camera_id)
            frame_id: Optional[int] = None
            track_id: Optional[int] = None
            det_conf: float = Field(0.0, ge=0.0, le=1.0)
            bbox: Tuple[int, int, int, int] = Field(..., description="[x1,y1,x2,y2] in source frame")
            ts: Optional[str] = None
            sequence_id: int = Field(0, ge=0)
            polygon: Optional[List[float]] = Field(
                None,
                description="Optional quadrilateral as 8 floats [x0,y0,x1,y1,x2,y2,x3,y3] in source frame coords",
            )

            @validator("bbox")
            def _bbox_len(cls, value: Tuple[int, int, int, int]):  # type: ignore
                if len(value) != 4:
                    raise ValueError("bbox must have four elements")
                return value
            @validator("polygon")
            def _poly_len(cls, value: Optional[List[float]]):  # type: ignore
                if value is None:
                    return value
                if len(value) != 8:
                    raise ValueError("polygon must have 8 elements [x0,y0,..,x3,y3]")
                return value

        @app.post("/v1/crops")
        async def enqueue_crop(payload: CropIn):  # type: ignore[no-redef]
            if state.runtime_error is not None:
                raise HTTPException(status_code=503, detail=state.runtime_error.message)
            if state.crop_queue is None:
                raise HTTPException(status_code=503, detail="crop queue not available")
            try:
                raw_bytes, crop = _decode_base64_image(payload.crop_b64)
            except Exception:
                raise HTTPException(status_code=400, detail="invalid base64 crop")
            if len(raw_bytes) > state.config.max_upload_bytes:
                raise HTTPException(status_code=413, detail="crop exceeds size limit")
            if crop is None:
                raise HTTPException(status_code=500, detail="numpy/opencv not available")
            # convert polygon flat list to list of (x,y) tuples
            poly_xy: Optional[List[Tuple[float, float]]] = None
            if payload.polygon is not None:
                coords = [float(v) for v in payload.polygon]
                poly_xy = [(coords[i], coords[i + 1]) for i in range(0, 8, 2)]

            task = CropTask(
                camera_id=payload.camera_id,
                request_id=payload.request_id,
                crop=crop,
                det_conf=float(payload.det_conf or 0.0),
                bbox=tuple(int(v) for v in payload.bbox),
                ts=_parse_ts_iso(payload.ts).isoformat(),
                frame_id=payload.frame_id,
                track_id=payload.track_id,
                sequence_id=payload.sequence_id,
                polygon_xy=poly_xy,
            )
            try:
                state.crop_queue.put_nowait(task)
            except asyncio.QueueFull:
                state.queue_dropped += 1
                raise HTTPException(status_code=429, detail="crop queue full")
            state.queue_len = state.crop_queue.qsize()
            return JSONResponse({"queued": True, "queue_len": state.queue_len})

    @app.post("/v1/alpr")
    async def alpr_detect(
        request: Request,  # type: ignore[type-arg]
        background_tasks: BackgroundTasks,  # type: ignore[assignment]
        image: UploadFile = File(...),  # type: ignore[assignment]
        camera_id: str = Form(""),
        request_id: str = Form(""),
        min_conf: float = Form(config.min_conf),
    ):  # type: ignore[no-redef]
        if state.runtime_error is not None:
            raise HTTPException(status_code=503, detail=state.runtime_error.message)
        if state.runtime is None:
            raise HTTPException(status_code=503, detail="runtime unavailable")
        token_expected = state.config.auth_token
        if token_expected:
            supplied = request.headers.get("X-ALPR-Token")
            if supplied != token_expected:
                raise HTTPException(status_code=401, detail="invalid token")
        if np is None or cv2 is None or postprocess_indonesia is None:
            raise HTTPException(status_code=500, detail="numpy/opencv/postprocess not available")

        data = await image.read()
        if not data:
            raise HTTPException(status_code=400, detail="empty image payload")
        if len(data) > state.config.max_upload_bytes:
            raise HTTPException(status_code=413, detail="image exceeds size limit")
        buf = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="unable to decode image")

        cam_id = camera_id or state.config.default_camera_id
        req_id = request_id or f"req-{int(time.time() * 1000)}"

        start_det = time.time()
        try:
            idx, inp, input_hw, ratio_pad = _prepare_image(state.runtime.det_model, frame)  # type: ignore[attr-defined]
            outputs = state.runtime.det_model.infer({idx: inp})  # type: ignore[arg-type]
            detections = decode_trt_detections(
                outputs,
                img0_shape=frame.shape,
                input_hw=input_hw,
                ratio_pad=ratio_pad,
                conf_thres=min_conf,
                iou_thres=0.45,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"detection failed: {exc}") from exc
        det_ms = (time.time() - start_det) * 1000.0

        crops: List[Any] = []
        det_meta: List[Tuple[Tuple[int, int, int, int], float, int]] = []
        h, w = frame.shape[:2]
        # Heuristics from plan.md: ignore tiny or implausible aspect crops
        MIN_H = 28
        AR_MIN, AR_MAX = 1.5, 5.0
        for bbox, score, cls in detections:
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1 or score < min_conf:
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            # gate by height and aspect ratio to reduce garbage OCRs
            hbox = max(1, y2 - y1)
            wbox = max(1, x2 - x1)
            ar = float(wbox) / float(hbox)
            if (hbox < MIN_H) or (ar < AR_MIN) or (ar > AR_MAX):
                continue
            crops.append(crop)
            det_meta.append(((x1, y1, x2, y2), float(score), int(cls)))

        texts: List[str] = []
        char_confs: List[List[float]] = []
        ocr_ms = 0.0
        if crops and state.runtime.ocr_runner is not None:
            start_ocr = time.time()
            runner = state.runtime.ocr_runner
            try:
                if state.runtime.ocr_mode == "onnx":
                    res = runner.infer_batch(crops, return_confidence=True)  # type: ignore[attr-defined]
                    if isinstance(res, tuple) and len(res) == 2:
                        texts, char_confs = res  # type: ignore[misc]
                    else:
                        texts = list(res)  # type: ignore[arg-type]
                else:
                    texts = runner.infer_batch(crops)  # type: ignore[attr-defined]
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"OCR failed: {exc}") from exc
            ocr_ms = (time.time() - start_ocr) * 1000.0

        allowed = state.config.allowed_prefixes or None
        plates: List[Dict[str, Any]] = []
        for idx, ((x1, y1, x2, y2), det_conf, cls) in enumerate(det_meta):
            raw_text = texts[idx] if idx < len(texts) else ""
            char_conf = char_confs[idx] if idx < len(char_confs) else []
            norm_text, is_valid = postprocess_indonesia(
                raw_text,
                allowed_prefix=allowed,
                tuning=state.postproc_tuning,  # type: ignore[arg-type]
                char_confs=char_conf,
                strict=bool(state.config.postproc_strict),
            )
            plates.append(
                {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "det_conf": float(det_conf),
                    "ocr_raw": raw_text,
                    "text": norm_text,
                    "valid": bool(is_valid),
                    "plate_conf": _compute_plate_conf(det_conf, char_conf),
                    "char_confs": [float(c) for c in char_conf],
                    "class_id": int(cls),
                    "track_id": None,
                    "frame_id": None,
                }
            )

        total_ms = det_ms + ocr_ms
        state.total_requests += 1
        state.last_latency_ms = total_ms
        status_label = "ok" if plates else "no_plate"
        state.last_status_ok = 1 if status_label == "ok" else 0
        state.last_frame_ts = datetime.now(timezone.utc).isoformat()

        background_tasks.add_task(
            _record_events,
            request_id=req_id,
            camera_id=cam_id,
            ts=state.last_frame_ts,
            status=status_label,
            detections=plates,
            crops=crops,
        )

        return JSONResponse(
            {
                "request_id": req_id,
                "camera_id": cam_id,
                "ts": state.last_frame_ts,
                "status": status_label,
                "plates": plates,
                "latency_ms": {"det": det_ms, "ocr": ocr_ms, "total": total_ms},
            }
        )

    return app


def main() -> None:  # pragma: no cover
    print("alpr-api stub (serve via uvicorn)")


if __name__ == "__main__":  # pragma: no cover
    main()
