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
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


Number = Union[float, int]


_GAMMA_TABLE_CACHE: Dict[float, np.ndarray] = {}


@dataclass(frozen=True)
class PreprocConfig:
    input_width: int = 160
    input_height: int = 32
    mean: Union[Number, Sequence[Number]] = 0.5
    std: Union[Number, Sequence[Number]] = 0.5
    clahe_clip: float = 2.0
    clahe_tile: int = 8
    use_clahe: bool = True
    # Apply CLAHE only if mean brightness < gate (0 disables gating)
    clahe_brightness_gate: float = 0.0  # 0..255
    channels: int = 1  # 1 (default) for LPRNet-style, 3 for PaddleOCR models
    # Night-time/headlight handling (disabled by default)
    suppress_highlights: bool = False
    highlight_threshold: int = 245  # 0..255; pixels >= this are considered glare
    highlight_inpaint_radius: int = 0  # >0 to inpaint glare regions instead of clipping
    remove_small_bright_specks: bool = False
    speck_area_px: int = 8  # approximate smallest bright speck area to suppress
    # Automatic per-crop adaptation (day/night/dirty)
    auto_preproc: bool = True
    glare_frac_gate: float = 0.01         # fraction of pixels considered glare to trigger suppression
    auto_highlight_quantile: float = 0.998  # quantile for auto threshold if highlight_threshold==0
    speck_area_frac_gate: float = 0.002   # max fraction of area for specks cleanup
    # Color & polarity helpers
    auto_color_cast: bool = True
    color_cast_clip: float = 0.35
    gamma_correction: bool = True
    gamma_dark_gate: float = 90.0
    gamma_value: float = 1.15
    auto_polarity: bool = True
    polarity_dark_mean: float = 110.0
    polarity_light_mean: float = 175.0
    invert_grayscale: bool = False


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


def _gamma_table(gamma: float) -> np.ndarray:
    gamma = max(0.1, min(5.0, float(gamma)))
    table = _GAMMA_TABLE_CACHE.get(gamma)
    if table is None:
        inv = 1.0 / gamma
        arr = np.arange(256, dtype=np.float32) / 255.0
        table = np.clip((arr ** inv) * 255.0, 0, 255).astype(np.uint8)
        _GAMMA_TABLE_CACHE[gamma] = table
    return table


def _apply_gray_world(img_bgr: np.ndarray, clip: float = 0.35) -> np.ndarray:
    img = img_bgr.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0) + 1e-3
    mean_gray = float(means.mean())
    scales = mean_gray / means
    clip_val = max(0.0, min(1.0, clip))
    scales = np.clip(scales, 1.0 - clip_val, 1.0 + clip_val)
    balanced = img * scales
    return np.clip(balanced, 0, 255).astype(np.uint8)


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

    bgr = img_bgr
    if bool(cfg.auto_color_cast):
        try:
            bgr = _apply_gray_world(bgr, clip=cfg.color_cast_clip)
        except Exception:
            pass
    gray = to_gray(bgr)

    # Optional highlight suppression and speck cleanup for night-time glare/marks
    try:
        g = gray
        # Choose threshold automatically if requested
        th = int(cfg.highlight_threshold)
        if th <= 0:
            q = float(cfg.auto_highlight_quantile)
            q = min(max(q, 0.95), 0.999)
            th = int(np.quantile(g, q))
        th = max(0, min(255, th))

        # Measure glare fraction
        bright_mask = (g >= th)
        glare_frac = float(bright_mask.mean())
        apply_highlights = bool(cfg.suppress_highlights)
        apply_specks = bool(cfg.remove_small_bright_specks)
        opened = None
        # Auto triggers if enabled
        if bool(cfg.auto_preproc):
            if glare_frac >= float(cfg.glare_frac_gate):
                apply_highlights = True
            # Estimate small specks fraction via opening
            ksz = max(1, int(round((int(cfg.speck_area_px) ** 0.5))))
            ksz = min(max(ksz, 1), 5)
            k = np.ones((ksz, ksz), np.uint8)
            opened = cv2.morphologyEx((bright_mask.astype(np.uint8) * 255), cv2.MORPH_OPEN, k)
            speck_frac = float((opened > 0).mean())
            if 0 < speck_frac <= float(cfg.speck_area_frac_gate):
                apply_specks = True

        if apply_specks:
            if opened is None:
                ksz = max(1, int(round((int(cfg.speck_area_px) ** 0.5))))
                ksz = min(max(ksz, 1), 5)
                k = np.ones((ksz, ksz), np.uint8)
                opened = cv2.morphologyEx((bright_mask.astype(np.uint8) * 255), cv2.MORPH_OPEN, k)
            median_val = int(np.median(g))
            g = g.copy()
            g[opened > 0] = median_val
        if apply_highlights:
            mask2 = (g >= th).astype(np.uint8) * 255
            if int(cfg.highlight_inpaint_radius or 0) > 0:
                r = int(cfg.highlight_inpaint_radius)
                g = cv2.inpaint(g, mask2, r, cv2.INPAINT_TELEA)
            else:
                g = np.minimum(g, th).astype(g.dtype)
        gray = g
    except Exception:
        # keep preproc robust; ignore glare suppression errors
        pass

    if cfg.use_clahe:
        try:
            mean = float(gray.mean())
            gate = float(cfg.clahe_brightness_gate or 0.0)
            if gate <= 0.0 and bool(cfg.auto_preproc):
                gate = 170.0
            if gate <= 0.0 or mean < gate:
                gray = clahe(gray, clip=cfg.clahe_clip, tile_grid=cfg.clahe_tile)
        except Exception:
            gray = clahe(gray, clip=cfg.clahe_clip, tile_grid=cfg.clahe_tile)

    if bool(cfg.gamma_correction):
        try:
            mean_val = float(gray.mean())
            gate = float(cfg.gamma_dark_gate)
            if gate <= 0.0:
                gate = 90.0
            if mean_val < gate:
                table = _gamma_table(float(cfg.gamma_value))
                gray = cv2.LUT(gray, table)
        except Exception:
            pass

    try:
        auto_invert = False
        if bool(cfg.auto_polarity):
            mean_val = float(gray.mean())
            if mean_val <= float(cfg.polarity_dark_mean):
                auto_invert = True
            elif mean_val >= float(cfg.polarity_light_mean):
                auto_invert = False
        if bool(cfg.invert_grayscale) or auto_invert:
            gray = 255 - gray
    except Exception:
        if bool(cfg.invert_grayscale):
            gray = 255 - gray

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
