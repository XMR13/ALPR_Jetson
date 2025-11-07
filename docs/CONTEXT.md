# Project Context (Quick Start)

This file is a fast, session-start snapshot. It does not replace plan.md.
Source of truth for tasks and timeline remains plan.md (Section 5) and progress/* session logs.

Last updated: 2025-11-05 (Session 6 — see progress/2025-11-05_session-1.md once logged)

Current Week/Day
- Week 2 — Day 12–13 (per plan.md §5)

Open Items (mirrors plan.md §5)
- DeepStream pipeline wiring + fresh CMake linking lives in `alpr-deepstream`; build it inside the Jetson DeepStream container and confirm probe counters/IPC metrics during the next device session.
- Execute NDJSON stream + `e2e --stats` smokes on Jetson following `docs/SMOKE_GUIDE.md`; archive outputs to progress logs.
- Run the RTSP soak per `docs/SOAK_RUNBOOK.md` and capture metrics/drops for Week 3 acceptance.

Next 3 Actions
1) Build `alpr-deepstream` using the updated CMake inside the DeepStream container, then run the RTSP probe with IPC logging enabled to validate counters + ZeroMQ wiring.
2) Run NDJSON smokes (`e2e-json-stream` + `e2e --stats`) on Jetson, storing summaries under `export/smoke/` and noting results in `progress/`.
3) Schedule and execute the 1–2 h RTSP soak per `docs/SOAK_RUNBOOK.md`, capturing metrics for Week 3 acceptance.

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
