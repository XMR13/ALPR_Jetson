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
# - TEXT_ONLY=1: print only plate text to stdout (see TEXT_MODE/TEXT_ALLOW_INVALID/TEXT_NO_PLATE)
# - ANNOTATE_DIR=/path/out: also save annotated image(s) using the same model args
#
# Post-processing controls (optional):
# - POSTPROC=indonesia|none         -> override CLI default postproc
# - ALLOWED_PREFIX="B D F ..."       -> space- or comma-separated allowed prefixes when POSTPROC=indonesia
#
# Optional outputs (TEXT_ONLY=1):
# - TEXT_OUT_FILE=/path/out.txt  -> if set and RC==0, writes the plate text to this file
# - TEXT_RC_FILE=/path/rc.txt    -> if set, writes the exit code (0/2/3) to this file
# - TEXT_MODE=best|raw           -> best prints normalized text; raw prints OCR raw text (default: best)
# - TEXT_ALLOW_INVALID=1         -> when set, prints text even if postproc marks it invalid
# - TEXT_NO_PLATE="NO_PLATE"      -> when set and no/invalid plate (rc=3), prints this placeholder before exiting 3
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
# Match CLI e2e default
CONF="${CONF:-0.5}"

if [[ ! -f "$DET_ENGINE" ]]; then
  echo "error: detector engine not found: $DET_ENGINE" >&2
  exit 2
fi

ARGS=( -m alpr_jetson e2e-json --det-engine "$DET_ENGINE" --source "$IMAGE" --conf "$CONF" )
# Make JSON path permissive by default to mirror e2e behavior
ACCEPT_ALL="${ACCEPT_ALL:-1}"
if [[ "$ACCEPT_ALL" == "1" ]]; then
  ARGS+=( --accept-all )
fi

# Optional post-processing overrides
if [[ -n "${POSTPROC:-}" ]]; then
  ARGS+=( --postproc "$POSTPROC" )
fi
if [[ -n "${ALLOWED_PREFIX:-}" ]]; then
  # split by comma or space into an array
  IFS=',' read -r -a _pref_csv <<< "${ALLOWED_PREFIX}"
  if (( ${#_pref_csv[@]} > 1 )); then
    # comma separated provided; trim spaces
    _ap=()
    for item in "${_pref_csv[@]}"; do
      _ap+=("${item//[[:space:]]/}")
    done
  else
    # space-separated words
    read -r -a _ap <<< "${ALLOWED_PREFIX}"
  fi
  if (( ${#_ap[@]} > 0 )); then
    ARGS+=( --allowed-prefix )
    for p in "${_ap[@]}"; do ARGS+=( "$p" ); done
  fi
fi

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
  # mirror postproc flags for annotate path
  if [[ -n "${POSTPROC:-}" ]]; then
    ANN_ARGS+=( --postproc "$POSTPROC" )
  fi
  if [[ -n "${ALLOWED_PREFIX:-}" ]]; then
    ANN_ARGS+=( --allowed-prefix )
    for p in "${_ap[@]:-}"; do ANN_ARGS+=( "$p" ); done
  fi
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
  # Print only best plate text; supports raw/allow_invalid/no_plate placeholder
  TEXT=$(python - <<'PY'
import os, sys, json
mode = os.getenv('TEXT_MODE', 'best').lower()  # 'best' or 'raw'
allow_invalid = os.getenv('TEXT_ALLOW_INVALID', '0') in ('1','true','True')
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
text_best = (best.get('text') or '').strip()
text_raw = (best.get('ocr_raw') or '').strip()
valid = bool(best.get('valid', False))
if mode == 'raw':
    if text_raw:
        print(text_raw)
        sys.exit(0)
    sys.exit(3)
else:  # best (normalized)
    if text_best and (valid or allow_invalid):
        print(text_best if text_best else text_raw)
        sys.exit(0)
    # fall back to raw if allowed
    if allow_invalid and text_raw:
        print(text_raw)
        sys.exit(0)
    sys.exit(3)
PY
<<< "$JSON_OUT" )
  RC=$?
  # Optionally write exit code to file
  if [[ -n "${TEXT_RC_FILE:-}" ]]; then
    printf '%s\n' "$RC" > "$TEXT_RC_FILE" || true
  fi
  if [[ $RC -ne 0 ]]; then
    # Optional placeholder output when no/invalid plate
    if [[ -n "${TEXT_NO_PLATE:-}" ]]; then
      printf '%s\n' "$TEXT_NO_PLATE"
    fi
    exit $RC
  fi
  # Optionally write text to file
  if [[ -n "${TEXT_OUT_FILE:-}" ]]; then
    printf '%s\n' "$TEXT" > "$TEXT_OUT_FILE" || true
  fi
  printf '%s\n' "$TEXT"
  exit 0
fi

# Default: print JSON
printf '%s\n' "$JSON_OUT"
exit 0
