#!/usr/bin/env bash
# TensorRT engine builder using trtexec.
# Usage:
#   bash tools/trtexec_build.sh <model.onnx> <output.engine> [extra trtexec args]
# Examples:
#   bash tools/trtexec_build.sh models/detector/yolov9-s.onnx models/detector/yolov9-s_fp16.engine --fp16
#   bash tools/trtexec_build.sh models/detector/yolov9-s.onnx models/detector/yolov9-s_int8.engine --int8 --calib=<cache>

set -euo pipefail

if ! command -v trtexec >/dev/null 2>&1; then
  echo "ERROR: trtexec not found in PATH. Install TensorRT or enter the NVIDIA container." >&2
  exit 127
fi

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <model.onnx> <output.engine> [extra trtexec args]" >&2
  exit 2
fi

ONNX_PATH="$1"; shift
ENGINE_PATH="$1"; shift || true

if [[ ! -f "$ONNX_PATH" ]]; then
  echo "ERROR: ONNX not found: $ONNX_PATH" >&2
  exit 3
fi

ENGINE_DIR="$(dirname "$ENGINE_PATH")"
mkdir -p "$ENGINE_DIR"

# If neither --fp16 nor --int8 specified, default to --fp16 as per plan.md targets.
EXTRA=("$@")
if [[ " ${EXTRA[*]} " != *" --fp16 "* ]] && [[ " ${EXTRA[*]} " != *" --int8 "* ]]; then
  EXTRA+=("--fp16")
fi

# Reasonable defaults for Jetson; adjust as needed.
ARGS=(
  "--onnx=$ONNX_PATH"
  "--saveEngine=$ENGINE_PATH"
  "--workspace=2048"
  "--verbose"
)

echo "[trtexec_build] Building engine:" >&2
echo "  onnx   : $ONNX_PATH" >&2
echo "  engine : $ENGINE_PATH" >&2
echo "  extras : ${EXTRA[*]:-<none>}" >&2

set -x
trtexec "${ARGS[@]}" "${EXTRA[@]}"
set +x

echo "[trtexec_build] Done: $ENGINE_PATH" >&2
