#!/usr/bin/env bash
set -euo pipefail

# One-shot ALPR JSON wrapper for PHP integration
# Usage: tools/alpr_e2e_json.sh /absolute/or/relative/path/to/image.jpg
#
# Defaults target Jetson runtime (ONNX OCR). You can override via env vars:
#   OCR_BACKEND=trt|onnx
#   DET_ENGINE=models/detector/yolov9-s_plate_fp16.engine
#   OCR_ENGINE=models/ocr/ppo_crnn_fp16.engine
#   CHARSET=models/ocr/charset.txt
#   OCR_ONNX=models/ocr/cct_s_v1_global.onnx
#   PLATE_CONFIG=models/ocr/cct_s_v1_global_plate_config.yaml
#   CONF=0.65
#
# Emits JSON to stdout: {status, plates[], latency_ms{det,ocr,total}}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Ensure Python can import the package if not installed
export PYTHONPATH="${PYTHONPATH:-${REPO_ROOT}/src}"

if [[ $# -lt 1 ]];
then
  echo "Usage: $0 /path/to/image" >&2
  exit 2
fi

IMAGE="$1"
if [[ ! -f "$IMAGE" ]]; then
  echo "error: image not found: $IMAGE" >&2
  exit 2
fi

# Defaults (can be overridden by env)
OCR_BACKEND="${OCR_BACKEND:-onnx}"
DET_ENGINE="${DET_ENGINE:-${REPO_ROOT}/models/detector/yolov9-s_plate_fp16.engine}"
CONF="${CONF:-0.65}"

if [[ ! -f "$DET_ENGINE" ]]; then
  echo "error: detector engine not found: $DET_ENGINE" >&2
  exit 2
fi

ARGS=( -m alpr_jetson e2e-json --det-engine "$DET_ENGINE" --source "$IMAGE" --conf "$CONF" )

case "$OCR_BACKEND" in
  trt)
    OCR_ENGINE="${OCR_ENGINE:-${REPO_ROOT}/models/ocr/ppo_crnn_fp16.engine}"
    CHARSET="${CHARSET:-${REPO_ROOT}/models/ocr/charset.txt}"
    if [[ ! -f "$OCR_ENGINE" ]]; then
      echo "error: OCR TensorRT engine not found: $OCR_ENGINE" >&2
      exit 2
    fi
    if [[ ! -f "$CHARSET" ]]; then
      echo "error: charset.txt not found: $CHARSET" >&2
      exit 2
    fi
    ARGS+=( --engine "$OCR_ENGINE" --charset "$CHARSET" )
    ;;
  onnx)
    OCR_ONNX="${OCR_ONNX:-${REPO_ROOT}/models/ocr/cct_s_v1_global.onnx}"
    PLATE_CONFIG="${PLATE_CONFIG:-${REPO_ROOT}/models/ocr/cct_s_v1_global_plate_config.yaml}"
    if [[ ! -f "$OCR_ONNX" ]]; then
      echo "error: OCR ONNX model not found: $OCR_ONNX" >&2
      exit 2
    fi
    if [[ ! -f "$PLATE_CONFIG" ]]; then
      echo "error: plate_config.yaml not found: $PLATE_CONFIG" >&2
      exit 2
    fi
    ARGS+=( --onnx "$OCR_ONNX" --plate-config "$PLATE_CONFIG" --onnx-provider cuda --onnx-gpu-mem-limit-mb 512 )
    ;;
  *)
    echo "error: unknown OCR_BACKEND '$OCR_BACKEND' (use 'trt' or 'onnx')" >&2
    exit 2
    ;;
esac

exec python "${ARGS[@]}"

