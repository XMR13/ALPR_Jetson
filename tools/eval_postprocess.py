#!/usr/bin/env python3
"""Evaluate OCR post-processing heuristics against labeled samples.

Input formats:
- JSONL: each line `{"raw": "B9048VI", "expected": "B 9048 VIN"}`
- CSV: columns `raw,expected` (optionally `allowed_prefixes` comma-separated)

Usage:
    PYTHONPATH=src python3 tools/eval_postprocess.py \
        --input data/postprocess_eval.jsonl \
        --config configs/ocr/postproc_indonesia.yaml \
        --output export/metrics/postprocess_eval.json
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ocr_service.postprocess import (
    DEFAULT_REGEX,
    PostprocessTuning,
    load_postprocess_config,
    postprocess_indonesia,
)


def _parse_allowed(value: Optional[str], fallback: Optional[Sequence[str]]) -> Optional[List[str]]:
    if value is None or value == "":
        return list(fallback) if fallback else None
    if isinstance(value, (list, tuple)):
        return [str(v).strip().upper() for v in value if str(v).strip()]
    return [part.strip().upper() for part in str(value).split(",") if part.strip()]


def _read_jsonl(path: Path) -> List[Dict[str, str]]:
    samples: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def _read_csv(path: Path, delimiter: str = ",") -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        return [row for row in reader]


def _evaluate(
    samples: Iterable[Dict[str, str]],
    tuning: Optional[PostprocessTuning],
    regex: str,
    default_allowed: Optional[Sequence[str]],
) -> Dict[str, float]:
    total = 0
    exact = 0
    valid = 0
    changed = 0
    failures: List[Dict[str, str]] = []

    for sample in samples:
        raw = sample.get("raw") or sample.get("text")
        expected = sample.get("expected") or sample.get("gt")
        if not raw or not expected:
            continue
        allowed = _parse_allowed(sample.get("allowed_prefixes"), default_allowed)
        text, is_valid = postprocess_indonesia(raw, allowed_prefix=allowed, regex=regex, tuning=tuning)
        total += 1
        if is_valid:
            valid += 1
        if text == expected:
            exact += 1
        else:
            failures.append({"raw": raw, "pred": text, "expected": expected})
        if text != raw:
            changed += 1

    return {
        "total": total,
        "exact": exact,
        "exact_rate": (exact / total) if total else 0.0,
        "valid_rate": (valid / total) if total else 0.0,
        "changed": changed,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OCR post-processing heuristics")
    parser.add_argument("--input", required=True, help="Path to JSONL or CSV samples")
    parser.add_argument("--format", choices=["jsonl", "csv"], default="jsonl")
    parser.add_argument("--delimiter", default=",", help="CSV delimiter (when format=csv)")
    parser.add_argument("--config", help="Path to postprocess YAML config")
    parser.add_argument(
        "--allowed-prefixes",
        default="A,B,D,F,E,Z,T",
        help="Comma-separated default prefix whitelist",
    )
    parser.add_argument("--regex", default=DEFAULT_REGEX, help="Plate regex to enforce")
    parser.add_argument("--output", help="Optional path to write metrics JSON")
    args = parser.parse_args()

    path = Path(args.input)
    if args.format == "jsonl":
        samples = _read_jsonl(path)
    else:
        samples = _read_csv(path, delimiter=args.delimiter)

    tuning = load_postprocess_config(args.config) if args.config else None
    allowed = _parse_allowed(args.allowed_prefixes, None)
    metrics = _evaluate(samples, tuning, args.regex, allowed)

    print(json.dumps({k: (v if k != "failures" else len(v)) for k, v in metrics.items()}, indent=2))
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_payload = metrics.copy()
        out_payload["tuning"] = asdict(tuning) if tuning else None
        out_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")

    if metrics["failures"]:
        print("\nExamples (first 5 failures):")
        for item in metrics["failures"][:5]:
            print(f"raw={item['raw']} pred={item['pred']} expected={item['expected']}")

if __name__ == "__main__":
    main()
