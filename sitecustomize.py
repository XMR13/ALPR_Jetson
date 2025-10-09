"""
Ensure local `src/` is on sys.path when running Python commands from the repo
root. This makes `python -m alpr_jetson -h` work in subprocesses launched by
tests without requiring an editable install.

This file is auto-imported by Python if found on sys.path (the current working
directory is on sys.path by default for scripts), so it is a safe, dev-only
convenience that does not affect production installs.
"""
from __future__ import annotations

import os
import sys


def _add_src_to_sys_path() -> None:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(repo_root, "src")
    if os.path.isdir(src_path) and src_path not in sys.path:
        # Prepend so it takes precedence during local dev/testing
        sys.path.insert(0, src_path)


_add_src_to_sys_path()

