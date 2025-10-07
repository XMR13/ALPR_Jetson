#!/usr/bin/env bash
set -euo pipefail

# DeepStream smoke using provided app config
# Usage: ./tools/deepstream_smoke.sh configs/deepstream/app_config.txt

CFG="${1:-configs/deepstream/app_config.txt}"

if [[ ! -f "${CFG}" ]]; then
  echo "Config not found: ${CFG}" >&2
  exit 1
fi

echo "[deepstream_smoke] Running deepstream-app -c ${CFG}" >&2
deepstream-app -c "${CFG}"

