from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def add_subcommands(sub):
    p_rtsp = sub.add_parser("rtsp-smoke", help="Run RTSP GStreamer smoke test")
    p_rtsp.add_argument("uri", help="RTSP URI")
    p_rtsp.add_argument("--latency", type=int, default=200, help="rtspsrc latency (ms)")
    p_rtsp.set_defaults(func=cmd_rtsp_smoke)

    p_ds = sub.add_parser("ds-smoke", help="Run DeepStream app smoke test")
    p_ds.add_argument("--config", default="configs/deepstream/app_config.txt", help="deepstream-app config path")
    p_ds.set_defaults(func=cmd_deepstream_smoke)


def cmd_rtsp_smoke(args: argparse.Namespace) -> int:
    script = Path("tools/rtsp_smoke.sh")
    if not script.exists():
        print("tools/rtsp_smoke.sh not found", file=sys.stderr)
        return 2
    uri = args.uri
    latency = str(args.latency)
    return subprocess.call(["bash", str(script), uri, latency])


def cmd_deepstream_smoke(args: argparse.Namespace) -> int:
    script = Path("tools/deepstream_smoke.sh")
    if not script.exists():
        print("tools/deepstream_smoke.sh not found", file=sys.stderr)
        return 2
    cfg = args.config
    return subprocess.call(["bash", str(script), cfg])

