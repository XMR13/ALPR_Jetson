"""OCR preprocessing utilities for Jetson NX.

Implements light-weight image preprocessing tailored for Indonesian plates:
- optional rectification (homography when polygon is available)
- grayscale + CLAHE to mitigate headlight glare
- normalization to the configured OCR input size

Notes (Jetson Xavier NX, JetPack 5.1.5):
- All operations rely on OpenCV CPU ops; they are fast enough for plate crops.
- Keep memory copies minimal and operate in-place where reasonable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


Number = Union[float, int]


@dataclass(frozen=True)
class PreprocConfig:
    input_width: int = 160
    input_height: int = 32
    mean: Union[Number, Sequence[Number]] = 0.5
    std: Union[Number, Sequence[Number]] = 0.5
    clahe_clip: float = 2.0
    clahe_tile: int = 8
    use_clahe: bool = True
    channels: int = 1  # 1 (default) for LPRNet-style, 3 for PaddleOCR models


def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR image to single-channel grayscale without copying more than needed."""
    if img_bgr.ndim == 2:
        return img_bgr
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def clahe(gray: np.ndarray, clip: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """Apply CLAHE to improve local contrast.

    Parameters
    - gray: HxW uint8 image
    - clip: CLAHE clip limit (typ. 2.0)
    - tile_grid: tile grid size (NxN)
    """
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tile_grid), int(tile_grid)))
    return clahe.apply(gray)


def rectify_polygon(
    img_bgr: np.ndarray,
    polygon_xy: Sequence[Tuple[float, float]],
    out_hw: Tuple[int, int],
) -> np.ndarray:
    """Rectify a quadrilateral region to a canonical HxW using homography.

    - polygon_xy: 4 points (x,y) in clockwise order.
    - out_hw: (H, W)
    """
    if len(polygon_xy) != 4:
        raise ValueError("polygon_xy must have 4 points")
    H, W = int(out_hw[0]), int(out_hw[1])
    src = np.array(polygon_xy, dtype=np.float32)
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img_bgr, M, (W, H), flags=cv2.INTER_LINEAR)
    return warped


def _to_array(value: Union[Number, Sequence[Number]], channels: int) -> np.ndarray:
    arr: np.ndarray
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=np.float32)
        if arr.size != channels:
            raise ValueError(f"expected {channels} values, got {arr.size}")
    else:
        arr = np.full((channels,), float(value), dtype=np.float32)
    return arr


def resize_normalize_gray(gray: np.ndarray, cfg: PreprocConfig) -> np.ndarray:
    """Resize grayscale image and normalize to NCHW float32."""
    H = int(cfg.input_height)
    W = int(cfg.input_width)
    resized = cv2.resize(gray, (W, H), interpolation=cv2.INTER_LINEAR)
    norm = resized.astype(np.float32) / 255.0

    channels = int(cfg.channels)
    if channels <= 0:
        raise ValueError("channels must be >= 1")

    mean = _to_array(cfg.mean, channels)
    std = _to_array(cfg.std, channels)
    if np.any(std <= 0):
        raise ValueError("std must be > 0 for all channels")

    if channels == 1:
        norm = (norm - mean[0]) / std[0]
        norm = norm[None, None, ...]
    else:
        stacked = np.repeat(norm[None, ...], channels, axis=0)
        norm = (stacked - mean[:, None, None]) / std[:, None, None]
        norm = norm[None, ...]
    return norm.astype(np.float32)


def prepare_ocr_input(
    img_bgr: np.ndarray,
    cfg: PreprocConfig = PreprocConfig(),
    polygon_xy: Optional[Sequence[Tuple[float, float]]] = None,
) -> np.ndarray:
    """End-to-end preprocessing for a single plate crop.

    - Optionally rectifies with `polygon_xy` if provided (4 points).
    - Converts to grayscale and applies CLAHE.
    - Resizes and normalizes to [1,C,H,W] float32 (NCHW) for TRT.
    """
    if polygon_xy is not None:
        img_bgr = rectify_polygon(img_bgr, polygon_xy, (cfg.input_height, cfg.input_width))
    gray = to_gray(img_bgr)
    if cfg.use_clahe:
        gray = clahe(gray, clip=cfg.clahe_clip, tile_grid=cfg.clahe_tile)
    x = resize_normalize_gray(gray, cfg)
    return x


def prepare_ocr_batch(
    imgs_bgr: Iterable[np.ndarray],
    cfg: PreprocConfig = PreprocConfig(),
    polygons: Optional[Iterable[Optional[Sequence[Tuple[float, float]]]]] = None,
) -> np.ndarray:
    """Vectorized batch builder from a sequence of BGR crops.

    Returns an array of shape [N, C, H, W] float32.
    """
    if polygons is None:
        arrs = [prepare_ocr_input(im, cfg) for im in imgs_bgr]
    else:
        # Pair each image with its polygon (or None)
        arrs = [
            prepare_ocr_input(im, cfg, poly)
            for im, poly in zip(imgs_bgr, polygons)
        ]
    if not arrs:
        return np.empty((0, cfg.channels, cfg.input_height, cfg.input_width), dtype=np.float32)
    return np.concatenate(arrs, axis=0)
