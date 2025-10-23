"""ALPR API server skeleton with safe stub endpoints.

Per plan.md §5 (Week 2 Day 12–13), this exposes health, metrics, stream info,
event listing, webhook registration, and a WS placeholder. The module avoids a
hard dependency on FastAPI so imports succeed in minimal environments; when
FastAPI is available, ``create_app`` returns a fully wired app.

Additionally, this module now provides a minimal synchronous `/v1/alpr`
endpoint to accept a captured image (multipart upload), run detector + OCR,
and synchronously return plate text(s). This enables integration testing with
existing capture systems while the real-time DeepStream pipeline is completed.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, Request
    from fastapi.responses import JSONResponse, PlainTextResponse
except Exception:  # FastAPI not installed yet in this scaffold
    FastAPI = None  # type: ignore
    PlainTextResponse = None  # type: ignore
    JSONResponse = None  # type: ignore
    HTTPException = Exception  # type: ignore
    WebSocket = object  # type: ignore
    UploadFile = object  # type: ignore
    File = object  # type: ignore
    Form = object  # type: ignore
    Request = object  # type: ignore

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

try:
    from inference.yolov9_trt import (
        load_engine,
        decode_trt_detections,
        _prepare_image,  # type: ignore
    )
except Exception:  # pragma: no cover
    load_engine = None  # type: ignore
    decode_trt_detections = None  # type: ignore
    _prepare_image = None  # type: ignore

try:
    from ocr_service.postprocess import postprocess_indonesia  # type: ignore
except Exception:  # pragma: no cover
    postprocess_indonesia = None  # type: ignore


def _env_list(name: str, default: Sequence[str] | None = None) -> List[str]:
    val = os.getenv(name)
    if not val:
        return list(default or [])
    return [item.strip() for item in val.split(",") if item.strip()]


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


def _initialize_runtime(cfg: AppConfig) -> Tuple[Optional[_Runtime], Optional[RuntimeErrorState]]:
    if cv2 is None or np is None:
        return None, RuntimeErrorState("opencv-python and numpy are required for /v1/alpr")
    if load_engine is None or decode_trt_detections is None or _prepare_image is None:
        return None, RuntimeErrorState("TensorRT utilities unavailable (inference.yolov9_trt not importable)")
    if not cfg.det_engine:
        return None, RuntimeErrorState("ALPR_DET_ENGINE not configured")

    runtime = _Runtime()
    try:
        runtime.det_model = load_engine(cfg.det_engine, print_plugins=False)  # type: ignore[arg-type]
    except Exception as exc:  # pragma: no cover
        return None, RuntimeErrorState(f"failed to load detector engine: {exc}")

    if cfg.ocr_engine and cfg.charset_path:
        try:
            from ocr_service.trt_infer import OCRService  # type: ignore
            from ocr_service.preprocess import PreprocConfig  # type: ignore

            preproc = PreprocConfig(
                input_width=cfg.ocr_input_width,
                input_height=cfg.ocr_input_height,
                channels=cfg.ocr_channels,
                mean=0.5,
                std=0.5,
                use_clahe=not cfg.ocr_no_clahe,
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
    gpu_util: float = 0.0
    events: List[Dict[str, Any]] = field(default_factory=list)
    webhooks: List[Dict[str, Any]] = field(default_factory=list)
    total_requests: int = 0
    last_latency_ms: float = 0.0
    last_status_ok: int = 0
    runtime: Optional[_Runtime] = None
    runtime_error: Optional[RuntimeErrorState] = None
    config: AppConfig = field(default_factory=AppConfig)


def create_app(cfg: Optional[AppConfig] = None) -> "FastAPI | None":
    if FastAPI is None:
        return None
    config = cfg or _config_from_env()
    app = FastAPI(title="ALPR API", version="0.0.0")
    runtime, err = _initialize_runtime(config)
    state = _State(runtime=runtime, runtime_error=err, config=config)

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
            "# HELP alpr_gpu_util GPU utilization percent",
            "# TYPE alpr_gpu_util gauge",
            f"alpr_gpu_util {state.gpu_util}",
            "# HELP alpr_requests_total Total synchronous /v1/alpr requests",
            "# TYPE alpr_requests_total counter",
            f"alpr_requests_total {state.total_requests}",
            "# HELP alpr_last_latency_ms Last synchronous /v1/alpr total latency in ms",
            "# TYPE alpr_last_latency_ms gauge",
            f"alpr_last_latency_ms {state.last_latency_ms}",
            "# HELP alpr_last_status_ok 1 if last /v1/alpr request returned status=ok else 0",
            "# TYPE alpr_last_status_ok gauge",
            f"alpr_last_status_ok {state.last_status_ok}",
        ]
        return "\n".join(lines) + "\n"

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
        return JSONResponse({"events": events[: max(0, min(1000, int(limit)))]})

    @app.websocket("/v1/ws")
    async def ws_endpoint(ws: WebSocket):  # type: ignore[no-redef]
        await ws.accept()
        try:
            await ws.send_text("ready")
            while True:
                _ = await ws.receive_text()
                await ws.send_text("pong")
        except Exception:
            pass

    @app.post("/v1/alpr")
    async def alpr_detect(
        request: Request,  # type: ignore[type-arg]
        image: UploadFile = File(...),  # type: ignore[assignment]
        camera_id: str = Form(""),
        request_id: str = Form(""),
        min_conf: float = Form(config.min_conf),
    ):  # type: ignore[no-redef]
        if state.runtime_error is not None:
            raise HTTPException(status_code=503, detail=state.runtime_error.message)
        if state.runtime is None:
            raise HTTPException(status_code=503, detail="runtime not initialized")

        token_expected = state.config.auth_token
        if token_expected:
            supplied = request.headers.get("X-ALPR-Token")
            if supplied != token_expected:
                raise HTTPException(status_code=401, detail="invalid token")

        if np is None or cv2 is None:
            raise HTTPException(status_code=500, detail="numpy/opencv not available")
        if postprocess_indonesia is None:
            raise HTTPException(status_code=500, detail="postprocess module unavailable")

        data = await image.read()
        if not data:
            raise HTTPException(status_code=400, detail="empty image payload")
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
        for idx, (bbox, det_conf, cls) in enumerate(det_meta):
            raw_text = texts[idx] if idx < len(texts) else ""
            char_conf = char_confs[idx] if idx < len(char_confs) else []
            norm_text, is_valid = postprocess_indonesia(raw_text, allowed_prefix=allowed)
            if char_conf:
                avg_char = float(sum(char_conf) / max(1, len(char_conf)))
                plate_conf = float(det_conf * avg_char)
            else:
                plate_conf = float(det_conf)
            plates.append(
                {
                    "bbox": [int(v) for v in bbox],
                    "det_conf": float(det_conf),
                    "ocr_raw": raw_text,
                    "text": norm_text,
                    "valid": bool(is_valid),
                    "plate_conf": plate_conf,
                    "char_confs": [float(c) for c in char_conf],
                    "class_id": int(cls),
                }
            )

        total_ms = det_ms + ocr_ms
        state.total_requests += 1
        state.last_latency_ms = total_ms
        status_label = "ok" if plates else "no_plate"
        state.last_status_ok = 1 if status_label == "ok" else 0
        state.last_frame_ts = datetime.now(timezone.utc).isoformat()

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


def main() -> None:
    print("alpr-api stub (to be implemented)")


if __name__ == "__main__":
    main()
