#!/usr/bin/env bash
set -euo pipefail

# Simple NDJSON stream smoke test for e2e-json-stream.
#
# Usage examples:
#   tools/stream_cli_smoke.sh \
#     --det-engine models/detector/yolov9-s_plate_fp16.engine \
#     --onnx models/ocr/cct_s.onnx --plate-config models/ocr/cct_s_v1_global_plate_config.yaml \
#     images_dir
#
#   find /path/to/frames -maxdepth 1 -type f -iname '*.jpg' | \
#     tools/stream_cli_smoke.sh --det-engine ... --engine models/ocr/cct_s.engine --charset models/ocr/cct_charset.txt

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 [--det-engine <path>] [--onnx <path> --plate-config <path> | --engine <path> --charset <path>] <image_dir_or_stdin>" >&2
  exit 2
fi

ARGS=()
IMG_MODE="dir"
IMG_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --det-engine|--onnx|--plate-config|--engine|--charset|--conf|--iou|--onnx-provider|--onnx-gpu-mem-limit-mb)
      ARGS+=("$1" "$2"); shift 2 ;;
    -) IMG_MODE="stdin"; shift ;;
    *) IMG_PATH="$1"; shift ;;
  esac
done

if [[ "$IMG_MODE" == "stdin" ]]; then
  # Read image paths from stdin
  python -m alpr_jetson e2e-json-stream "${ARGS[@]}"
  exit $?
fi

if [[ -z "$IMG_PATH" ]]; then
  echo "Error: provide an images directory or '-' to read from stdin" >&2
  exit 2
fi

if [[ ! -d "$IMG_PATH" ]]; then
  echo "Error: '$IMG_PATH' is not a directory" >&2
  exit 2
fi

# Pipe images to the CLI
find "$IMG_PATH" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' \) \
  | sort \
  | python -m alpr_jetson e2e-json-stream "${ARGS[@]}"

