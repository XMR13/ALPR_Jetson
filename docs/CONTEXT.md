# Project Context (Quick Start)

This file is a fast, session-start snapshot. It does not replace plan.md.
Source of truth for tasks and timeline remains plan.md (Section 5) and progress/* session logs.

Last updated: 2025-10-29 (session 1)

Current Week/Day
- Week 2 — Day 12–13 (per plan.md §5)

Open Items (mirrors plan.md §5)
- Upgrade DS→OCR transport to ZeroMQ/IPC (Week 3 target)
- Smoke-test rectification path end-to-end and capture metrics snapshots
- Live RTSP/E2E soak with metrics populated from queue bridge

Next 3 Actions
1) Run local smoke for `tools/stream_cli_smoke.sh` (uses `e2e-json-stream`) and capture NDJSON + timing.
2) Implement OCR-side PULL (ZeroMQ IPC) behind a config toggle; keep HTTP baseline as default.
3) Add DS sender hook guarded in `crop_probe.cpp`; then plan a short RTSP soak to validate queue and latency.

Decisions/Risks
- HTTP bridge remains default; IPC design drafted (see `docs/IPC_BRIDGE.md`).
- Rectification enabled in preprocessing; verify on angled plates during the soak.
- Stream CLI smoke readiness depends on local deps/models present; install on a machine with network access if needed.

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
