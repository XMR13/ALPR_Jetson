#!/usr/bin/env bash
set -euo pipefail

# Simple RTSP smoke test using GStreamer
# Usage: ./tools/rtsp_smoke.sh rtsp://user:pass@host:port/path [latency_ms]

URI="${1:-}"
LATENCY="${2:-200}"

if [[ -z "${URI}" ]]; then
  echo "Usage: $0 rtsp://user:pass@host:port/path [latency_ms]" >&2
  exit 1
fi

echo "[rtsp_smoke] Testing RTSP URI: ${URI} (latency=${LATENCY}ms)" >&2
gst-launch-1.0 -q rtspsrc location="${URI}" latency=${LATENCY} ! decodebin ! fakesink sync=false
echo "[rtsp_smoke] OK"

