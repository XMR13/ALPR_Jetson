"""Placeholder TensorRT OCR inference service.

Implements batch inference later per plan.md §5 (Week 2 Day 8–9).
"""

from __future__ import annotations

from typing import List


class OCRService:
    def __init__(self) -> None:
        pass

    def infer_batch(self, images: List[bytes]) -> List[str]:
        # TODO: implement TRT-backed OCR
        return ["<stub>"] * len(images)

