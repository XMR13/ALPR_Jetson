"""End-to-end evaluation utility for detector + OCR predictions.

Usage examples:

```bash
python tools/eval_e2e.py \
    --events export/events.sqlite \
    --ground-truth data/labels/e2e_gt.csv \
    --gt-id-column request_id \
    --gt-plate-column plate_gt \
    --low-conf-threshold 0.85 \
    --low-conf-dir export/low_confidence
```

The script joins predictions stored in the SQLite `events` table with a
ground-truth CSV, then reports exact-match accuracy, Character Error Rate (CER),
and Sequence Error Rate (SER). Rows with confidence below the given threshold
are copied (if snapshots exist) into the chosen low-confidence directory for
manual inspection.

All heavy computations (database read, Levenshtein distance) remain pure Python
to avoid extra Jetson dependencies.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Prediction:
    request_id: str
    camera_id: str
    plate: str
    plate_conf: float
    char_confs: List[float]
    snapshot_path: Optional[str]
    ts: str


@dataclass
class EvaluationResult:
    total: int
    matched: int
    exact_match_pct: float
    average_cer: float
    average_ser: float
    total_gt_chars: int
    per_camera: Dict[str, Dict[str, float]] = field(default_factory=dict)
    missing_gt: List[str] = field(default_factory=list)
    low_conf_predictions: List[Prediction] = field(default_factory=list)


def levenshtein_distance(source: str, target: str) -> int:
    """Compute Levenshtein distance using a simple dynamic-programming table."""

    if source == target:
        return 0
    if not source:
        return len(target)
    if not target:
        return len(source)

    prev = list(range(len(target) + 1))
    for i, sc in enumerate(source, start=1):
        curr = [i]
        for j, tc in enumerate(target, start=1):
            cost = 0 if sc == tc else 1
            curr.append(
                min(
                    curr[j - 1] + 1,  # insertion
                    prev[j] + 1,      # deletion
                    prev[j - 1] + cost,  # substitution
                )
            )
        prev = curr
    return prev[-1]


def char_error_rate(pred: str, truth: str) -> float:
    if not truth:
        return 0.0 if not pred else float(len(pred))
    return float(levenshtein_distance(pred, truth)) / float(len(truth))


def sequence_error_rate(pred: str, truth: str) -> float:
    return 0.0 if pred == truth else 1.0


def load_ground_truth_csv(
    csv_path: Path,
    *,
    id_column: str,
    plate_column: str,
) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get(id_column, "").strip()
            plate = row.get(plate_column, "").strip().upper()
            if not rid:
                continue
            mapping[rid] = plate
    return mapping


def load_predictions_from_sqlite(db_path: Path) -> List[Prediction]:
    preds: List[Prediction] = []
    conn = sqlite3.connect(str(db_path))
    query = """
    SELECT request_id, camera_id, plate, plate_conf, char_confs, snapshot_path, ts
    FROM events
    ORDER BY created_at DESC
    """
    try:
        for row in conn.execute(query):
            req_id, camera_id, plate, plate_conf, char_confs_json, snapshot_path, ts = row
            try:
                confs = json.loads(char_confs_json) if char_confs_json else []
            except Exception:
                confs = []
            preds.append(
                Prediction(
                    request_id=str(req_id),
                    camera_id=str(camera_id or "unknown"),
                    plate=str(plate or "").upper(),
                    plate_conf=float(plate_conf or 0.0),
                    char_confs=[float(c) for c in confs if isinstance(c, (int, float))],
                    snapshot_path=str(snapshot_path) if snapshot_path else None,
                    ts=str(ts or ""),
                )
            )
    finally:
        conn.close()
    return preds


def evaluate_predictions(
    predictions: Iterable[Prediction],
    ground_truth: Dict[str, str],
    *,
    low_conf_threshold: float,
) -> EvaluationResult:
    total = 0
    matched = 0
    sum_cer = 0.0
    sum_ser = 0.0
    total_chars = 0
    per_camera: Dict[str, Dict[str, float]] = {}
    missing_gt: List[str] = []
    low_conf: List[Prediction] = []

    seen: set[str] = set()

    for pred in predictions:
        if pred.request_id in seen:
            continue  # latest entry already processed (ordered DESC)
        seen.add(pred.request_id)

        truth = ground_truth.get(pred.request_id)
        if truth is None:
            missing_gt.append(pred.request_id)
            continue

        total += 1
        cer = char_error_rate(pred.plate, truth)
        ser = sequence_error_rate(pred.plate, truth)
        sum_cer += cer
        sum_ser += ser
        total_chars += max(1, len(truth))
        if ser == 0.0:
            matched += 1

        cam_stats = per_camera.setdefault(
            pred.camera_id,
            {"total": 0, "matched": 0, "sum_cer": 0.0, "sum_ser": 0.0},
        )
        cam_stats["total"] += 1
        cam_stats["sum_cer"] += cer
        cam_stats["sum_ser"] += ser
        if ser == 0.0:
            cam_stats["matched"] += 1

        if pred.plate_conf < low_conf_threshold:
            low_conf.append(pred)

    average_cer = sum_cer / max(1, total)
    average_ser = sum_ser / max(1, total)
    exact_match_pct = (matched / total * 100.0) if total else 0.0

    # Normalize per-camera stats into percentage/averages
    normalized_per_camera: Dict[str, Dict[str, float]] = {}
    for cam, stats in per_camera.items():
        t = stats["total"] or 1
        normalized_per_camera[cam] = {
            "total": float(stats["total"]),
            "exact_match_pct": stats["matched"] / t * 100.0,
            "average_cer": stats["sum_cer"] / t,
            "average_ser": stats["sum_ser"] / t,
        }

    return EvaluationResult(
        total=total,
        matched=matched,
        exact_match_pct=exact_match_pct,
        average_cer=average_cer,
        average_ser=average_ser,
        total_gt_chars=total_chars,
        per_camera=normalized_per_camera,
        missing_gt=missing_gt,
        low_conf_predictions=low_conf,
    )


def copy_low_confidence_snapshots(preds: Sequence[Prediction], output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for pred in preds:
        if not pred.snapshot_path:
            continue
        src = Path(pred.snapshot_path)
        if not src.exists():
            continue
        dest = output_dir / src.name
        try:
            shutil.copy2(src, dest)
            copied += 1
        except Exception:
            continue
    return copied


def _format_result(res: EvaluationResult) -> str:
    lines = [
        f"Total samples: {res.total}",
        f"Exact matches: {res.matched} ({res.exact_match_pct:.2f}%)",
        f"Average CER: {res.average_cer:.4f}",
        f"Average SER: {res.average_ser:.4f}",
        f"Missing ground-truth rows: {len(res.missing_gt)}",
        f"Low-confidence predictions: {len(res.low_conf_predictions)}",
    ]
    if res.per_camera:
        lines.append("Per camera:")
        for cam, stats in sorted(res.per_camera.items()):
            lines.append(
                f"  {cam}: total={int(stats['total'])} exact={stats['exact_match_pct']:.2f}% cer={stats['average_cer']:.4f} ser={stats['average_ser']:.4f}"
            )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ALPR predictions against ground truth")
    p.add_argument("--events", type=Path, required=True, help="Path to events SQLite database")
    p.add_argument("--ground-truth", type=Path, required=True, help="CSV with ground truth labels")
    p.add_argument("--gt-id-column", default="request_id", help="Column name for request IDs in ground truth CSV")
    p.add_argument("--gt-plate-column", default="plate", help="Column name for plate text in ground truth CSV")
    p.add_argument("--low-conf-threshold", type=float, default=0.85, help="Confidence threshold for flagging snapshots")
    p.add_argument("--low-conf-dir", type=Path, default=None, help="Directory to copy low-confidence snapshots")
    p.add_argument("--output-json", type=Path, default=None, help="Optional path to write metrics JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.events.exists():
        raise SystemExit(f"events DB not found: {args.events}")
    if not args.ground_truth.exists():
        raise SystemExit(f"ground truth CSV not found: {args.ground_truth}")

    gt = load_ground_truth_csv(
        args.ground_truth,
        id_column=args.gt_id_column,
        plate_column=args.gt_plate_column,
    )
    predictions = load_predictions_from_sqlite(args.events)
    result = evaluate_predictions(predictions, gt, low_conf_threshold=args.low_conf_threshold)

    print(_format_result(result))

    copied = 0
    if args.low_conf_dir and result.low_conf_predictions:
        copied = copy_low_confidence_snapshots(result.low_conf_predictions, args.low_conf_dir)
        print(f"Copied {copied} low-confidence snapshots to {args.low_conf_dir}")

    if args.output_json:
        payload: Dict[str, Any] = {
            "total": result.total,
            "matched": result.matched,
            "exact_match_pct": result.exact_match_pct,
            "average_cer": result.average_cer,
            "average_ser": result.average_ser,
            "missing_gt": result.missing_gt,
            "per_camera": result.per_camera,
            "low_conf_predictions": [pred.request_id for pred in result.low_conf_predictions],
            "low_conf_copied": copied,
            "threshold": args.low_conf_threshold,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
