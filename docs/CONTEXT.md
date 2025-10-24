# Project Context (Quick Start)

This file is a fast, session-start snapshot. It does not replace plan.md.
Source of truth for tasks and timeline remains plan.md (Section 5) and progress/* session logs.

Last updated: 2025-10-24

Current Week/Day
- Week 2 — Day 12–13 (per plan.md §5)

Open Items (mirrors plan.md §5)
- Upgrade DS→OCR transport to ZeroMQ/IPC (Week 3 target)
- OCR CER/SER tooling and low-confidence snapshot capture
- Rectification wiring or deskew path usage where available
- Update Jetson Compose/Docker images to Python 3.8 base (Week 4)

Next 3 Actions
1) Add CER/SER evaluator and low-confidence logging in tools/eval_e2e.py
2) Wire rectification/deskew path where polygon data exists; document usage
3) Update deploy/compose.jetson.yml to Jetson-friendly Python 3.8 base images

Decisions/Risks
- Start with HTTP bridge to ship quickly on Jetson; move to IPC later for lower latency
- Keep OCR engines at FP16; INT8 reserved for detector in later weeks
- Ensure memory caps on ONNXRuntime (if used) to avoid OOM on NX

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
