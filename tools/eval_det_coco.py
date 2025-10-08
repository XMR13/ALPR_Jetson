"""Evaluate detector predictions with COCO metrics.

Requires pycocotools. If not installed, the script will exit with guidance.

Inputs
- --gt: Path to ground-truth COCO JSON (e.g., data/processed/.../coco.json)
- --pred: Path to predictions JSON with [{'image_id', 'category_id', 'bbox', 'score'}]

Outputs
- Prints AP metrics (AP, AP50, AP75, APS, APM, APL)
- Writes metrics JSON next to --pred as metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="COCO eval for detector predictions")
    p.add_argument("--gt", required=True, help="Ground-truth COCO JSON path")
    p.add_argument("--pred", required=True, help="Predictions JSON path")
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except Exception as e:
        print(
            "pycocotools not installed. Install with 'pip install pycocotools' to run COCO eval.",
            file=sys.stderr,
        )
        raise SystemExit(3) from e

    gt_path = Path(args.gt)
    pred_path = Path(args.pred)
    if not gt_path.exists():
        print(f"error: gt not found: {gt_path}", file=sys.stderr)
        sys.exit(2)
    if not pred_path.exists():
        print(f"error: pred not found: {pred_path}", file=sys.stderr)
        sys.exit(2)

    coco_gt = COCO(str(gt_path))
    with pred_path.open("r", encoding="utf-8") as f:
        preds = json.load(f)
    coco_dt = coco_gt.loadRes(preds)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Gather summary stats
    stats = coco_eval.stats  # type: ignore[attr-defined]
    metrics: Dict[str, Any] = {
        "AP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "APS": float(stats[3]),
        "APM": float(stats[4]),
        "APL": float(stats[5]),
    }
    out_json = pred_path.with_name("metrics.json")
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"wrote metrics to {out_json}")


if __name__ == "__main__":
    main()

