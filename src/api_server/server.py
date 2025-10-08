"""FastAPI server skeleton with safe stub endpoints.

Per plan.md §5 (Week 2 Day 12–13), this will serve events and health
endpoints. This stub avoids hard dependency on FastAPI so the module remains
importable in minimal environments. When FastAPI is installed, it exposes
`/healthz` and `/metrics` basic endpoints consistent with plan.md §9.
"""

from __future__ import annotations

try:
    from fastapi import FastAPI
    from fastapi.responses import PlainTextResponse, JSONResponse
except Exception:  # FastAPI not installed yet in this scaffold
    FastAPI = None  # type: ignore
    PlainTextResponse = None  # type: ignore
    JSONResponse = None  # type: ignore


def create_app() -> "FastAPI | None":
    if FastAPI is None:
        return None
    app = FastAPI(title="ALPR API", version="0.0.0")

    @app.get("/healthz")
    def healthz():  # type: ignore[no-redef]
        return JSONResponse({
            "status": "ok",
            "uptime_s": None,
            "gpu": None,
            "last_frame_ts": None,
        })

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics():  # type: ignore[no-redef]
        # Minimal Prometheus-style text output per plan.md §9
        lines = [
            "# HELP alpr_fps Current estimated frames per second",
            "# TYPE alpr_fps gauge",
            "alpr_fps 0",
            "# HELP alpr_queue_len Current crop queue depth",
            "# TYPE alpr_queue_len gauge",
            "alpr_queue_len 0",
        ]
        return "\n".join(lines) + "\n"

    # Additional endpoints from plan.md §9 will be added as the API matures.
    return app


def main() -> None:
    # Placeholder entry; actual serve via Uvicorn to be added later.
    print("alpr-api stub (to be implemented)")


if __name__ == "__main__":
    main()
