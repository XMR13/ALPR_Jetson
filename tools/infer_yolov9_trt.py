#!/usr/bin/env python3
"""Convenience wrapper to run YOLOv9 TRT inference via the reusable module."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from inference.yolov9_trt import main  # noqa: E402


if __name__ == "__main__":
    main()
