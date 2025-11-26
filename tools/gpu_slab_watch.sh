#!/usr/bin/env bash
# Snapshot slab/mem/GPU state to a log file (default /var/log/gpu_slab_watch.log).
# Safe to run under watch/cron; requires sudo for slabtop/tegrastats.

LOG=${1:-/var/log/gpu_slab_watch.log}
mkdir -p "$(dirname "$LOG")"

{
  echo "==== $(date -Iseconds) ===="
  echo "# slabtop"
  slabtop -sc | head -n 15 || echo "slabtop failed"
  echo "# meminfo"
  grep -E 'Slab|SUnreclaim' /proc/meminfo || true
  echo "# tegrastats"
  timeout 3 tegrastats 2>/dev/null | head -n 1 || echo "tegrastats failed"
  echo
} >> "$LOG"
