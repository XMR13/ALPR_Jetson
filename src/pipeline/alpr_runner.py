from __future__ import annotations

"""Canonical E2E runner used by CLI paths (JSON and annotate).

Keeps core logic in a reusable module so the CLI entry remains thin.
3.8-safe typing is used throughout (no PEP 585/604).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
from pathlib import Path
import math


def plate_conf(det_conf: float, char_confs: List[float]) -> float:
    if not char_confs:
        return float(det_conf)
    avg_char = sum(char_confs) / max(1, len(char_confs))
    return float(det_conf) * float(avg_char)


def run_e2e_single(
    image_path: str,
    *,
    det_engine,
    ocr_runner,
    backend: str,
    conf: float,
    iou: float,
    postproc: str,
    allowed_prefix: List[str],
    postprocess_fn=None,
    min_plate_h: int = 28,
    min_ar: float = 1.5,
    max_ar: float = 5.0,
    debug_crops: bool = False,
    accept_all: bool = False,
    topk: int = 1,
) -> Dict[str, Any]:
    import time
    from typing import Tuple

    import cv2  # type: ignore
    from inference.yolov9_trt import infer_image  # type: ignore

    img_path = str(Path(image_path))
    t0 = time.time()
    img0, dets = infer_image(det_engine, img_path, conf=conf, iou=iou)
    det_ms = (time.time() - t0) * 1000.0

    h, w = img0.shape[:2]
    MIN_H = int(min_plate_h)
    AR_MIN, AR_MAX = float(min_ar), float(max_ar)
    crops: List[Tuple[Tuple[int, int, int, int], float]] = []
    rejected: List[Tuple[Tuple[int, int, int, int], str]] = []
    for bbox, score, _cls in dets:
        x1, y1, x2, y2 = [
            int(math.floor(bbox[0])),
            int(math.floor(bbox[1])),
            int(math.ceil(bbox[2])),
            int(math.ceil(bbox[3])),
        ]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            if debug_crops:
                rejected.append(((x1, y1, x2, y2), "invalid_xy"))
            continue
        hbox = max(1, y2 - y1)
        wbox = max(1, x2 - x1)
        ar = float(wbox) / float(hbox)
        if (not accept_all) and hbox < MIN_H:
            if debug_crops:
                rejected.append(((x1, y1, x2, y2), "too_small"))
            continue
        if (not accept_all) and ar < AR_MIN:
            if debug_crops:
                rejected.append(((x1, y1, x2, y2), "ar_low"))
            continue
        if (not accept_all) and ar > AR_MAX:
            if debug_crops:
                rejected.append(((x1, y1, x2, y2), "ar_high"))
            continue
        crops.append(((x1, y1, x2, y2), float(score)))

    # Keep only highest-confidence detections
    if crops and topk > 0:
        crops = sorted(crops, key=lambda x: x[1], reverse=True)[: int(topk)]

    texts: List[str] = []
    char_confs: List[List[float]] = []
    ocr_ms = 0.0
    if crops:
        crop_imgs = [img0[y1:y2, x1:x2] for (x1, y1, x2, y2), _ in crops]
        t1 = time.time()
        if backend == "onnx":
            res = ocr_runner.infer_batch(crop_imgs, return_confidence=True)  # type: ignore[attr-defined]
            if isinstance(res, tuple) and len(res) == 2:
                texts, char_confs = res  # type: ignore[misc]
            else:
                texts = list(res)  # type: ignore[arg-type]
        else:
            texts = ocr_runner.infer_batch(crop_imgs)  # type: ignore[attr-defined]
        ocr_ms = (time.time() - t1) * 1000.0

    plates = []
    for i, ((x1, y1, x2, y2), det_conf) in enumerate(crops):
        raw = texts[i] if i < len(texts) else ""
        confs = char_confs[i] if i < len(char_confs) else []
        norm_text = raw
        is_valid = True
        if postproc == "indonesia" and postprocess_fn is not None:
            norm_text, is_valid = postprocess_fn(raw, allowed_prefix=allowed_prefix or None)
        plates.append(
            {
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "det_conf": float(det_conf),
                "ocr_raw": raw,
                "text": norm_text,
                "valid": bool(is_valid),
                "plate_conf": plate_conf(det_conf, confs),
                "char_confs": [float(c) for c in confs],
            }
        )

    status = "ok" if plates else "no_plate"
    out: Dict[str, Any] = {
        "status": status,
        "plates": plates,
        "latency_ms": {"det": det_ms, "ocr": ocr_ms, "total": det_ms + ocr_ms},
    }
    if debug_crops:
        out["debug"] = {
            "det_count": int(len(dets)),
            "accepted": [
                {"bbox": [x1, y1, x2, y2]} for (x1, y1, x2, y2), _ in crops
            ],
            "rejected": [
                {"bbox": [x1, y1, x2, y2], "reason": reason} for ((x1, y1, x2, y2), reason) in rejected
            ],
            "params": {"min_h": MIN_H, "min_ar": AR_MIN, "max_ar": AR_MAX},
        }
    return out

