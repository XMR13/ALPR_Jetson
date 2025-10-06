"""FastAPI server skeleton (no functional endpoints yet).

Per plan.md §5 (Week 2 Day 12–13), this will serve events and health endpoints.
"""

from __future__ import annotations

try:
    from fastapi import FastAPI
except Exception:  # FastAPI not installed yet in this scaffold
    FastAPI = None  # type: ignore


def create_app() -> "FastAPI | None":
    if FastAPI is None:
        return None
    app = FastAPI(title="ALPR API", version="0.0.0")
    # TODO: add routes per plan
    return app


def main() -> None:
    # Placeholder entry; actual serve via Uvicorn to be added later.
    print("alpr-api stub (to be implemented)")


if __name__ == "__main__":
    main()

