#!/usr/bin/env python3
"""Evaluate OCR engine on a labeled crop dataset.

Computes exact-match and character error rate (CER) against ground truth.
Requires a TensorRT engine compatible with :mod:`ocr_service.trt_infer`.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import cv2  # type: ignore

from ocr_service.preprocess import PreprocConfig
from ocr_service.trt_infer import OCRService


def read_labels(labels_csv: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with labels_csv.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0].lower() == "filename" and len(row) > 1:
                # header row
                continue
            if len(row) < 2:
                continue
            mapping[row[0]] = row[1]
    return mapping


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    curr = [0] * (len(b) + 1)
    for i, ca in enumerate(a, 1):
        curr[0] = i
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev, curr = curr, prev
    return prev[-1]


def evaluate(
    crops_dir: Path,
    labels: Dict[str, str],
    engine: Path,
    charset: Path,
    channels: int,
    logits_layout: str,
) -> Tuple[int, int, float]:
    svc = OCRService(
        engine_path=str(engine),
        charset_path=str(charset),
        preproc=PreprocConfig(channels=channels),
        logits_layout=logits_layout,
    )
    if getattr(svc, "_runner", None) is None:
        raise RuntimeError("TensorRT engine not loaded; ensure TRT runtime is available and engine path is correct")

    samples: List[Tuple[str, str, str]] = []
    for name, gt in labels.items():
        img_path = crops_dir / name
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        pred = svc.infer_batch([img])[0]
        samples.append((name, gt, pred))

    total = len(samples)
    if total == 0:
        raise RuntimeError("no crops found for evaluation")

    exact = sum(1 for _, gt, pred in samples if gt == pred)
    lev_sum = 0
    len_sum = 0
    for _, gt, pred in samples:
        lev_sum += levenshtein(gt, pred)
        len_sum += max(len(gt), 1)
    cer = lev_sum / len_sum if len_sum else math.inf
    return total, exact, cer


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OCR TensorRT engine")
    parser.add_argument("--engine", required=True, help="Path to TensorRT engine")
    parser.add_argument("--charset", required=True, help="Path to charset.txt")
    parser.add_argument("--crops", required=True, help="Directory with crop images")
    parser.add_argument("--labels", required=True, help="CSV with filename,text")
    parser.add_argument("--channels", type=int, default=1, help="Input channels (1 or 3)")
    parser.add_argument(
        "--logits-layout",
        choices=["NTC", "NCT"],
        default="NTC",
        help="Layout of engine output logits",
    )
    args = parser.parse_args()

    crops_dir = Path(args.crops)
    labels = read_labels(Path(args.labels))
    total, exact, cer = evaluate(
        crops_dir=crops_dir,
        labels=labels,
        engine=Path(args.engine),
        charset=Path(args.charset),
        channels=int(args.channels),
        logits_layout=args.logits_layout,
    )
    em = exact / total * 100.0
    cer_pct = cer * 100.0
    print(f"Samples   : {total}")
    print(f"Exact     : {exact} ({em:.2f}%)")
    print(f"CER       : {cer_pct:.2f}%")


if __name__ == "__main__":
    main()
