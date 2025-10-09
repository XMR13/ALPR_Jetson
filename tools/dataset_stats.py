#!/usr/bin/env python3
"""
COCO dataset quick stats to tune detector training.

Reports image count, annotation count, bbox height/width percentiles,
and suggests training input size considerations for small objects.

Usage:
  python tools/dataset_stats.py --coco path/to/coco.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def pct(vals: List[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    i = int(max(0, min(len(s) - 1, round(q * (len(s) - 1)))))
    return float(s[i])


def run(coco_path: Path) -> int:
    with coco_path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    anns = [a for a in data.get("annotations", []) if isinstance(a.get("bbox"), list)]
    imgs = data.get("images", [])

    widths: List[float] = []
    heights: List[float] = []
    small_cnt = 0
    for a in anns:
        x, y, w, h = a["bbox"]
        widths.append(float(w))
        heights.append(float(h))
        if h < 28:  # heuristic from plan.md
            small_cnt += 1

    total = len(anns)
    h_p50 = pct(heights, 0.5)
    h_p90 = pct(heights, 0.9)
    h_p95 = pct(heights, 0.95)
    w_p50 = pct(widths, 0.5)
    w_p90 = pct(widths, 0.9)
    w_p95 = pct(widths, 0.95)

    print(f"images: {len(imgs)}  annotations: {total}")
    print(f"bbox height px  p50:{h_p50:.1f}  p90:{h_p90:.1f}  p95:{h_p95:.1f}")
    print(f"bbox width  px  p50:{w_p50:.1f}  p90:{w_p90:.1f}  p95:{w_p95:.1f}")
    if total:
        print(f"small (<28px height): {small_cnt} ({100.0*small_cnt/total:.1f}%)")

    # Suggestion for input size
    if h_p50 < 40:
        print("suggestion: consider training at 736-800 input or stronger mosaic to help small plates")
    else:
        print("suggestion: 640 input is likely sufficient; validate on val set")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="COCO dataset stats for tuning")
    p.add_argument("--coco", required=True, help="Path to coco.json")
    return p


def main(argv: List[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rc = run(Path(args.coco))
    raise SystemExit(rc)


if __name__ == "__main__":
    main()

