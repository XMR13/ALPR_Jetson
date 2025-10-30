"""FastAPI OCR microservice skeleton + optional ZeroMQ IPC receiver.

This app wraps the OCRService and exposes a simple HTTP endpoint. It can
optionally run a ZeroMQ PULL receiver (DS→OCR IPC) behind a config toggle.
All optional imports are guarded for Jetson-friendly minimal environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except Exception:  # pragma: no cover
    FastAPI = None  # type: ignore
    BaseModel = object  # type: ignore
    HTTPException = Exception  # type: ignore

import base64
import cv2
import numpy as np
import os
import json
import threading
from pathlib import Path

from .trt_infer import OCRService
from .preprocess import PreprocConfig


@dataclass
class AppConfig:
    engine_path: Optional[str] = None
    charset_path: Optional[str] = None
    input_width: int = 160
    input_height: int = 32
    mean: float = 0.5
    std: float = 0.5
    channels: int = 1
    logits_layout: str = "NTC"


class OCRRequest(BaseModel):  # type: ignore[misc]
    image_b64: str


@dataclass
class IPCConfig:
    enabled: bool = False
    endpoint: str = "ipc:///tmp/alpr.ds2ocr.sock"
    rcv_hwm: int = 256
    recv_timeout_ms: int = 1000
    cleanup_socket: bool = True


def _load_ipc_config() -> IPCConfig:
    """Load IPC config from configs/ocr/ipc.yaml if present, else env, else defaults.

    Env overrides:
      - ALPR_OCR_IPC_ENABLED=1
      - ALPR_OCR_IPC_ENDPOINT=ipc:///tmp/alpr.ds2ocr.sock
      - ALPR_OCR_IPC_RCVHWM=256
      - ALPR_OCR_IPC_RECV_TIMEOUT_MS=1000
    """
    cfg = IPCConfig()
    try:
        import yaml  # type: ignore
        p = Path("configs/ocr/ipc.yaml")
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if isinstance(data, dict) and isinstance(data.get("ipc"), dict):
                d = data["ipc"]
                cfg.enabled = bool(d.get("enabled", cfg.enabled))
                cfg.endpoint = str(d.get("endpoint", cfg.endpoint))
                cfg.rcv_hwm = int(d.get("rcv_hwm", cfg.rcv_hwm))
                cfg.recv_timeout_ms = int(d.get("recv_timeout_ms", cfg.recv_timeout_ms))
                cfg.cleanup_socket = bool(d.get("cleanup_socket", cfg.cleanup_socket))
    except Exception:
        pass
    # Env overrides
    if os.getenv("ALPR_OCR_IPC_ENABLED") is not None:
        cfg.enabled = os.getenv("ALPR_OCR_IPC_ENABLED", "0") in ("1", "true", "True")
    if os.getenv("ALPR_OCR_IPC_ENDPOINT"):
        cfg.endpoint = str(os.getenv("ALPR_OCR_IPC_ENDPOINT"))
    if os.getenv("ALPR_OCR_IPC_RCVHWM"):
        try:
            cfg.rcv_hwm = int(os.getenv("ALPR_OCR_IPC_RCVHWM", "256"))
        except Exception:
            pass
    if os.getenv("ALPR_OCR_IPC_RECV_TIMEOUT_MS"):
        try:
            cfg.recv_timeout_ms = int(os.getenv("ALPR_OCR_IPC_RECV_TIMEOUT_MS", "1000"))
        except Exception:
            pass
    return cfg


def _cleanup_ipc_path(endpoint: str) -> None:
    # Remove stale ipc:// socket file to avoid bind errors
    if endpoint.startswith("ipc://"):
        # Normalize both ipc:///tmp/foo.sock and ipc://tmp/foo.sock
        path = endpoint[len("ipc://"):]
        if path.startswith("/"):
            sock_path = path
        else:
            sock_path = "/" + path
        try:
            if os.path.exists(sock_path):
                os.remove(sock_path)
        except Exception:
            pass


def start_ipc_receiver(service: OCRService, ipc_cfg: Optional[IPCConfig] = None) -> Optional[threading.Thread]:
    """Start a background ZeroMQ PULL receiver that decodes JPEG crops and runs OCR.

    Returns the daemon thread if started, else None. Import-time safe if pyzmq
    is unavailable; will no-op and return None.
    """
    cfg = ipc_cfg or _load_ipc_config()
    if not cfg.enabled:
        return None
    try:
        import zmq  # type: ignore
    except Exception:
        print("[OCR IPC] pyzmq not available; IPC disabled")
        return None

    stats: Dict[str, Any] = {
        "rx_total": 0,
        "malformed_total": 0,
        "decode_fail_total": 0,
        "last_text": "",
    }

    def _serve() -> None:
        if cfg.cleanup_socket:
            _cleanup_ipc_path(cfg.endpoint)
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.PULL)
        try:
            sock.setsockopt(zmq.RCVHWM, int(cfg.rcv_hwm))
            sock.setsockopt(zmq.RCVTIMEO, int(cfg.recv_timeout_ms))
        except Exception:
            pass
        sock.bind(cfg.endpoint)
        while True:
            try:
                hdr_b = sock.recv(flags=0)
                payload = sock.recv(flags=0)
            except zmq.Again:
                continue
            except Exception:
                continue
            try:
                hdr = json.loads(hdr_b.decode("utf-8"))
            except Exception:
                stats["malformed_total"] = int(stats.get("malformed_total", 0)) + 1
                continue
            enc = str(hdr.get("encoding", "jpeg")).lower()
            if enc == "jpeg":
                arr = np.frombuffer(payload, np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            else:
                # Unsupported/raw path for now
                bgr = None
            if bgr is None:
                stats["decode_fail_total"] = int(stats.get("decode_fail_total", 0)) + 1
                continue
            texts = service.infer_batch([bgr])
            text = texts[0] if texts else ""
            stats["rx_total"] = int(stats.get("rx_total", 0)) + 1
            stats["last_text"] = text
            # Optional: print for smoke visibility
            if os.getenv("ALPR_OCR_IPC_LOG", "0") in ("1", "true", "True"):
                cam = hdr.get("camera_id", "?")
                tid = hdr.get("track_id", "?")
                print(f"[OCR IPC] cam={cam} track={tid} text={text}")

    th = threading.Thread(target=_serve, name="ocr-ipc-pull", daemon=True)
    th.start()
    return th


def create_app(cfg: AppConfig = AppConfig()):  # -> "FastAPI | None":
    if FastAPI is None:
        return None

    app = FastAPI(title="ALPR OCR Service", version="0.0.0")
    svc = OCRService(
        engine_path=cfg.engine_path,
        charset_path=cfg.charset_path,
        preproc=PreprocConfig(
            input_width=cfg.input_width,
            input_height=cfg.input_height,
            mean=cfg.mean,
            std=cfg.std,
            channels=cfg.channels,
        ),
        logits_layout=cfg.logits_layout,
    )

    # Optional: start IPC receiver when enabled via config/env
    try:
        ipc_cfg = _load_ipc_config()
        if ipc_cfg.enabled:
            start_ipc_receiver(svc, ipc_cfg)
    except Exception:
        pass

    @app.post("/v1/ocr")
    def ocr(req: OCRRequest):  # type: ignore[no-redef]
        try:
            data = base64.b64decode(req.image_b64)
        except Exception:
            raise HTTPException(status_code=400, detail="invalid base64 image")
        img_array = np.frombuffer(data, dtype=np.uint8)
        bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(status_code=400, detail="unable to decode image")

        texts = svc.infer_batch([bgr])
        return {"text": texts[0] if texts else ""}

    return app


def main() -> None:
    print("alpr-ocr app stub (serve via uvicorn). To enable IPC receiver, set ALPR_OCR_IPC_ENABLED=1 or configs/ocr/ipc.yaml: ipc.enabled: true")


if __name__ == "__main__":
    main()
