#!/usr/bin/env bash
set -euo pipefail

# Python 3.8 typing guard: fail on PEP 585 generics and PEP 604 unions in runtime code.
# Usage: tools/py38_guard.sh [paths...]
# Default paths: src

paths=("${@:-src}")

bad=0

echo "[py38-guard] scanning: ${paths[*]}"

# PEP 585 built-in generics in annotations (exclude commented lines)
if rg -n "(:|->)[^\n]*\\b(list|dict|tuple|set)\\s*\\[" -S -- ${paths[@]} | rg -v "#"; then
  echo "[py38-guard] ERROR: found PEP 585 built-in generics. Use typing.List/Dict/Tuple/Set instead." >&2
  bad=1
fi

# PEP 604 union syntax in annotations (exclude commented lines)
if rg -n "(:|->)[^\n]*\\|" -S -- ${paths[@]} | rg -v "#"; then
  echo "[py38-guard] ERROR: found PEP 604 union syntax. Use typing.Union/Optional instead." >&2
  bad=1
fi

if [[ $bad -ne 0 ]]; then
  echo "[py38-guard] FAIL" >&2
  exit 2
fi

echo "[py38-guard] OK"
