#!/usr/bin/env bash
# Snapshot kmalloc-128 / slab / GPU state to a log file (default /var/log/gpu_slab_watch.log).
# Designed to be:
#   - safe under watch/cron,
#   - low-overhead (no interactive slabtop),
#   - compatible with tools/plot_kmalloc.py (logs a kmalloc-128 line ending with "<N>K").
#
# Usage:
#   sudo tools/gpu_slab_watch.sh /var/log/alpr_kmalloc.log
#   watch -n 300 "sudo tools/gpu_slab_watch.sh /var/log/alpr_kmalloc.log"

set -euo pipefail

LOG=${1:-/var/log/gpu_slab_watch.log}
mkdir -p "$(dirname "$LOG")"

{
  echo "==== $(date -Iseconds) ===="

  echo "# kmalloc-128 (/proc/slabinfo)"
  if [[ -r /proc/slabinfo ]]; then
    slabinfo_cmd="cat /proc/slabinfo"
  elif command -v sudo >/dev/null 2>&1; then
    slabinfo_cmd="sudo cat /proc/slabinfo"
  else
    slabinfo_cmd=""
  fi

  if [[ -n "${slabinfo_cmd}" ]]; then
    # Emit a synthetic kmalloc-128 line that ends with "<N>K" so plot_kmalloc.py can parse it.
    ${slabinfo_cmd} 2>/dev/null | awk '
      $1 == "kmalloc-128" {
        name=$1; active=$2; objs=$3; size=$4;
        kb=int((objs * size) / 1024);
        printf "%s %s %s %s %dK\n", name, active, objs, size, kb;
      }
    ' || echo "kmalloc-128 not found in /proc/slabinfo"
  else
    echo "/proc/slabinfo not readable (try running with sudo)"
  fi

  echo "# meminfo"
  grep -E 'Slab:|SUnreclaim:' /proc/meminfo || true

  echo "# tegrastats"
  if command -v tegrastats >/dev/null 2>&1; then
    timeout 3 tegrastats 2>/dev/null | head -n 1 || echo "tegrastats failed"
  else
    echo "tegrastats not found"
  fi

  echo
} >> "$LOG"
