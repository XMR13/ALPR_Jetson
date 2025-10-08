#!/usr/bin/env python3
"""
Capture frames from an RTSP source at a fixed interval and save to disk.

Jetson notes:
- Prefer GStreamer backend via OpenCV if available for lower CPU.
- Output is intended for data bootstrapping under data/raw/<camera_id>/.

Usage:
  python tools/capture_frames.py \
      --uri rtsp://USER:PASS@IP:554/Streaming/Channels/101 \
      --out data/raw/cam01 \
      --interval 2.0
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2  # type: ignore


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Capture frames from RTSP at intervals")
    p.add_argument("--uri", required=True, help="RTSP URI")
    p.add_argument("--out", required=True, help="Output directory for JPG frames")
    p.add_argument("--interval", type=float, default=2.0, help="Seconds between captures")
    p.add_argument("--width", type=int, default=1920, help="Desired width (hint)")
    p.add_argument("--height", type=int, default=1080, help="Desired height (hint)")
    p.add_argument("--limit", type=int, default=0, help="Stop after N frames (0 = infinite)")
    return p


def open_capture(uri: str, width: int, height: int) -> cv2.VideoCapture:
    # Try GStreamer pipeline first; fall back to direct URI
    gst = (
        f"rtspsrc location=\"{uri}\" latency=200 ! "
        "rtph264depay ! h264parse ! avdec_h264 ! videoscale ! videoconvert ! "
        f"videoscale method=0 ! video/x-raw,width={width},height={height} ! appsink drop=true"
    )
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        cap = cv2.VideoCapture(uri)
    return cap


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = open_capture(args.uri, args.width, args.height)
    if not cap.isOpened():
        print("Failed to open RTSP stream")
        return 2

    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Read failed; retrying after 1s...")
                time.sleep(1.0)
                continue

            ts = time.strftime("%Y%m%d_%H%M%S")
            fname = out_dir / f"frame_{ts}_{count:06d}.jpg"
            cv2.imwrite(str(fname), frame)
            print(f"Saved {fname}")
            count += 1

            if args.limit > 0 and count >= args.limit:
                break

            time.sleep(max(0.0, args.interval))
    finally:
        cap.release()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

