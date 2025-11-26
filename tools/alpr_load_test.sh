#!/usr/bin/env bash
# Lightweight curl-based soak/load tester for /v1/alpr.
# Usage:
#   ./tools/alpr_load_test.sh -u http://jetson:8000/v1/alpr -i sample.jpg -n 25 [-t token] [-c cam01]
# -u: Full /v1/alpr URL
# -i: Image file to send (required)
# -n: Number of sequential requests (default: 20)
# -t: Optional X-ALPR-Token
# -c: camera_id form value (default: cam01)
# Outputs p50/p95/avg/min/max latency based on curl time_total.

set -euo pipefail

URL=""
IMAGE=""
COUNT=20
TOKEN=""
CAMERA_ID="cam01"

usage() {
  grep '^#' "$0" | sed 's/^# //'
  exit 1
}

while getopts "u:i:n:t:c:h" opt; do
  case $opt in
    u) URL="$OPTARG" ;;
    i) IMAGE="$OPTARG" ;;
    n) COUNT="$OPTARG" ;;
    t) TOKEN="$OPTARG" ;;
    c) CAMERA_ID="$OPTARG" ;;
    h) usage ;;
    *) usage ;;
  esac
done

if [[ -z "$URL" || -z "$IMAGE" ]]; then
  usage
fi

if [[ ! -f "$IMAGE" ]]; then
  echo "Image not found: $IMAGE" >&2
  exit 2
fi

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT
TIMES="$TMPDIR/times.txt"
FAIL=0

echo "Running $COUNT requests against $URL" >&2
for i in $(seq 1 "$COUNT"); do
  REQ_ID="load-$i-$(date +%s%3N)"
  TOKEN_HEADER=()
  [[ -n "$TOKEN" ]] && TOKEN_HEADER=("-H" "X-ALPR-Token: $TOKEN")
  time_total=$(curl -s -o /dev/null -w "%{time_total}" "${TOKEN_HEADER[@]}" \
    -F "image=@$IMAGE" \
    -F "camera_id=$CAMERA_ID" \
    -F "request_id=$REQ_ID" \
    "$URL" || true)
  if [[ -z "$time_total" || "$time_total" == "000" ]]; then
    ((FAIL++))
    continue
  fi
  echo "$time_total" >> "$TIMES"
  echo "[$i/$COUNT] ${time_total}s" >&2
  sleep 0.05
done

if [[ ! -s "$TIMES" ]]; then
  echo "No successful samples collected" >&2
  exit 3
fi

TIMES_FILE="$TIMES" FAILURES=$FAIL python - <<'PY'
import os
import statistics
from pathlib import Path
p = Path(os.environ["TIMES_FILE"])
vals = [float(x.strip()) * 1000 for x in p.read_text().splitlines() if x.strip()]
vals.sort()

def pct(vs, q):
    if not vs:
        return 0.0
    idx = round((q/100) * (len(vs)-1))
    return vs[int(idx)]

fails = int(os.getenv("FAILURES", "0"))
print(f"samples={len(vals)} fail={fails}")
print(f"avg_ms={statistics.mean(vals):.2f} min_ms={min(vals):.2f} max_ms={max(vals):.2f}")
print(f"p50_ms={pct(vals,50):.2f} p95_ms={pct(vals,95):.2f}")
PY
