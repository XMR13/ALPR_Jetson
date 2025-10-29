# Project Context (Quick Start)

This file is a fast, session-start snapshot. It does not replace plan.md.
Source of truth for tasks and timeline remains plan.md (Section 5) and progress/* session logs.

Last updated: 2025-10-28 (session 2)

Current Week/Day
- Week 2 — Day 12–13 (per plan.md §5)

Open Items (mirrors plan.md §5)
- Upgrade DS→OCR transport to ZeroMQ/IPC (Week 3 target)
- Smoke-test rectification path end-to-end and capture metrics snapshots
- Live RTSP/E2E soak with metrics populated from queue bridge

Next 3 Actions
1) Run a local smoke test for `tools/alpr_e2e_json.sh` and `e2e-json-stream` (NDJSON) to confirm timing/dup filters before handing to PHP.
2) Walk through `docs/INTEGRATION_PHP.md` with the PHP team; agree on request_id hashing and retry thresholds.
3) Draft DS→OCR ZeroMQ/IPC migration plan (message schema + metrics).

Decisions/Risks
- HTTP bridge remains the short-term transport; ZeroMQ/IPC design pending (Week 3)
- OCR rectification now active; verify homography accuracy on real polygons before locking parameters
- CLI wrappers now exist for PHP handoff; monitor for filesystem contention or hash collisions once deployed.
- Base images switched to `nvcr.io/nvidia/l4t-ml:r35.5.0-py3` (Python 3.8). Confirm availability on target NX and document pull instructions.
- Local smoke tests require FastAPI/pytest installed; current environment lacks network access, so execute `python3 -m pip install -r requirements-dev.txt` (or `uv pip sync`) on a machine with access before re-running tests.

Key Links
- API server: src/api_server/server.py
- DeepStream app: src/deepstream_app/main.cpp, src/deepstream_app/crop_probe.cpp
- OCR runtime: src/ocr_service/
- CLI entry: src/alpr_jetson/__main__.py
- Plan: plan.md (Section 5)
- Latest session: progress/LATEST.md
- Jetson deps: requirements-jetson.txt, constraints-jetson.txt

Note
- If CONTEXT.md conflicts with plan.md §5 or progress logs, plan.md §5 wins.
