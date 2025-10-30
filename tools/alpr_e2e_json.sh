#!/usr/bin/env bash
set -euo pipefail

# One-shot ALPR wrapper for PHP/Webmin integration
# Usage: tools/alpr_e2e_json.sh /absolute/or/relative/path/to/image.jpg
#
# Defaults target Jetson runtime with ONNX OCR. Override via env vars:
#   OCR_BACKEND=onnx|trt
#   DET_ENGINE=models/detector/yolov9-s_plate_fp16.engine
#   OCR_ENGINE=models/ocr/ppo_crnn_fp16.engine
#   CHARSET=models/ocr/charset.txt
#   OCR_ONNX=models/ocr/cct_s_v1_global.onnx
#   PLATE_CONFIG=models/ocr/cct_s_v1_global_plate_config.yaml
#   CONF=0.65
#
# Modes:
# - Default: print JSON to stdout: {status, plates[], latency_ms{det,ocr,total}}
# - TEXT_ONLY=1: print only plate text (best) to stdout; exit 3 when no plate/invalid text
# - ANNOTATE_DIR=/path/out: also save annotated image(s) using the same model args
#
# Exit codes:
#   0  success (JSON printed or text printed)
#   2  usage/model/path/runtime error
#   3  TEXT_ONLY mode: no plate detected or text invalid/empty

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

# Run JSON pipeline and capture output (do not exit on non-zero here)
set +e
JSON_OUT=$(python "${ARGS[@]}" 2>/dev/null)
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  # Propagate usage/model/runtime errors; echo captured text (may contain JSON error)
  [[ -n "$JSON_OUT" ]] && echo "$JSON_OUT"
  exit $RC
fi

# If ANNOTATE_DIR requested, run annotate path (best effort; does not affect rc)
if [[ -n "${ANNOTATE_DIR:-}" ]]; then
  ANN_ARGS=( -m alpr_jetson e2e --det-engine "$DET_ENGINE" --source "$IMAGE" --conf "$CONF" --annotate-dir "$ANNOTATE_DIR" )
  case "$OCR_BACKEND" in
    trt)
      ANN_ARGS+=( --engine "$OCR_ENGINE" --charset "$CHARSET" )
      ;;
    onnx)
      ANN_ARGS+=( --onnx "$OCR_ONNX" --plate-config "$PLATE_CONFIG" --onnx-provider cuda --onnx-gpu-mem-limit-mb 512 )
      ;;
  esac
  set +e
  python "${ANN_ARGS[@]}" >/dev/null 2>&1
  set -e
fi

if [[ "${TEXT_ONLY:-}" == "1" ]]; then
  # Print only best plate text; exit 3 on no/invalid result
  TEXT=$(python - <<'PY'
import sys, json
data = sys.stdin.read()
try:
    d = json.loads(data)
except Exception:
    sys.exit(2)
if d.get('status') != 'ok':
    sys.exit(3)
plates = d.get('plates') or []
if not plates:
    sys.exit(3)
best = plates[0]
text = (best.get('text') or '').strip()
valid = bool(best.get('valid', False))
if not text or not valid:
    sys.exit(3)
print(text)
sys.exit(0)
PY
<<< "$JSON_OUT" )
  RC=$?
  if [[ $RC -ne 0 ]]; then
    exit $RC
  fi
  printf '%s\n' "$TEXT"
  exit 0
fi

# Default: print JSON
printf '%s\n' "$JSON_OUT"
exit 0
