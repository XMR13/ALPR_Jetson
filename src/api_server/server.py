"""ALPR API server skeleton with safe stub endpoints.

Per plan.md §5 (Week 2 Day 12–13), this exposes health, metrics, stream info,
event listing, webhook registration, and a WS placeholder. The module avoids a
hard dependency on FastAPI so imports succeed in minimal environments; when
FastAPI is available, ``create_app`` returns a fully wired app.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException
    from fastapi import WebSocket
    from fastapi.responses import PlainTextResponse, JSONResponse
except Exception:  # FastAPI not installed yet in this scaffold
    FastAPI = None  # type: ignore
    PlainTextResponse = None  # type: ignore
    JSONResponse = None  # type: ignore
    HTTPException = Exception  # type: ignore
    WebSocket = object  # type: ignore


@dataclass
class _State:
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_frame_ts: Optional[str] = None
    fps: float = 0.0
    queue_len: int = 0
    gpu_util: float = 0.0
    events: List[Dict[str, Any]] = field(default_factory=list)
    webhooks: List[Dict[str, Any]] = field(default_factory=list)


def create_app() -> "FastAPI | None":
    if FastAPI is None:
        return None
    app = FastAPI(title="ALPR API", version="0.0.0")
    state = _State()

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
            }
        )

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics():  # type: ignore[no-redef]
        # Minimal Prometheus-style text output per plan.md §9
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
        ]
        return "\n".join(lines) + "\n"

    @app.get("/v1/stream/info")
    def stream_info():  # type: ignore[no-redef]
        return JSONResponse(
            {
                "camera_id": "cam01",
                "fps": state.fps,
                "buffer_backlog": state.queue_len,
                "last_frame_ts": state.last_frame_ts,
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
        # Minimal placeholder; echoes back a ping.
        await ws.accept()
        try:
            await ws.send_text("ready")
            while True:
                _ = await ws.receive_text()
                await ws.send_text("pong")
        except Exception:
            # client disconnected
            pass

    # Additional endpoints and persistent storage will be added later.
    return app


def main() -> None:
    # Placeholder entry; actual serve via Uvicorn to be added later.
    print("alpr-api stub (to be implemented)")


if __name__ == "__main__":
    main()
