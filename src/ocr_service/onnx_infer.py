"""ONNX OCR inference (slot-based alphabet) with CUDA EP default.

This module implements an OCR runner for models that emit per-slot logits of
shape [N, S, V] or flattened [N*S, V], where:
- S = max plate slots
- V = vocabulary size (len(alphabet))

It mirrors the behavior in models like CCT_S: RGB or grayscale NHWC uint8
input, optional keep-aspect letterbox, and a pad character that is removed
after argmax decoding.

Jetson notes:
- Default provider preference is CUDAExecutionProvider, then CPUExecutionProvider.
- TensorRTExecutionProvider is intentionally not used by default to avoid long
  build times; it can be enabled by passing `prefer_trt=True`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cv2  # type: ignore
import numpy as np

try:
    import onnxruntime as ort  # type: ignore
except Exception as e:  # pragma: no cover
    ort = None  # type: ignore

#interpolasi yang ada pada opencv 
_INTERP = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "area": cv2.INTER_AREA,
    "cubic": cv2.INTER_CUBIC,
    "lanczos4": cv2.INTER_LANCZOS4,
}

#fungsi place holder untuk melakukan konversi ke color tipe rgb (opencv default colour system is BGR)
def _to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 2:
        return cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


#memastikan color yang digunakan oleh program sesuain dengan standar yang tleah diberikan
def _ensure_color_mode(img: np.ndarray, mode: str) -> np.ndarray:
    mode = mode.lower()
    if mode not in {"rgb", "grayscale"}:
        raise ValueError(f"image_color_mode must be 'rgb' or 'grayscale', got {mode}")
    if mode == "grayscale":
        if img.ndim == 2:
            return img
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        raise ValueError("invalid input for grayscale mode")
    # rgb
    if img.ndim == 3 and img.shape[2] == 3:
        return _to_rgb(img)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    raise ValueError("invalid input for rgb mode")


#m
def _resize_letterbox(
    img: np.ndarray,
    target_h: int,
    target_w: int,
    image_color_mode: str,
    keep_aspect_ratio: bool,
    interpolation: str,
    padding_color: Union[Sequence[int], int] = (144, 144, 144),
) -> np.ndarray:
    inter = _INTERP.get(interpolation.lower(), cv2.INTER_LINEAR)
    if not keep_aspect_ratio:
        return cv2.resize(img, (int(target_w), int(target_h)), interpolation=inter)

    oh, ow = img.shape[:2]
    r = min(float(target_h) / float(oh), float(target_w) / float(ow))
    new_w, new_h = int(round(ow * r)), int(round(oh * r))
    if (new_w, new_h) != (ow, oh):
        img = cv2.resize(img, (new_w, new_h), interpolation=inter)
    dw = (target_w - new_w) / 2.0
    dh = (target_h - new_h) / 2.0
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    #meenentukan bahwa image yang digunakan berupa grayscale
    if image_color_mode.lower() == "grayscale":
        if isinstance(padding_color, (list, tuple)):
            color_gray = int(padding_color[0])
        else:
            color_gray = int(padding_color)
        return cv2.copyMakeBorder(
            img, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=color_gray
        )
    # rgb
    if isinstance(padding_color, (list, tuple)):
        if len(padding_color) != 3:
            raise ValueError("padding_color must have length 3 for RGB")
        color_rgb = tuple(int(c) for c in padding_color)
    else:
        v = int(padding_color)
        color_rgb = (v, v, v)
    return cv2.copyMakeBorder(
        img, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=color_rgb
    )


def _decode_logits(
    logits: np.ndarray,
    max_slots: int,
    alphabet: str,
    pad_char: str,
    return_confidence: bool = False,
) -> Union[List[str], Tuple[List[str], List[List[float]]]]:
    arr = np.asarray(logits)
    if arr.ndim == 3:
        n, s, v = arr.shape
    elif arr.ndim == 2:
        if arr.shape[0] % max_slots != 0:
            raise ValueError(f"cannot reshape logits of shape {arr.shape} with max_slots={max_slots}")
        n = arr.shape[0] // max_slots
        s = max_slots
        v = arr.shape[1]
        arr = arr.reshape(n, s, v)
    else:
        raise ValueError(f"unexpected logits shape {arr.shape}")

    idx = arr.argmax(axis=-1)  # [N,S]
    probs = arr.max(axis=-1)   # [N,S]

    out_texts: List[str] = []
    out_confs: List[List[float]] = []
    for i in range(idx.shape[0]):
        chars: List[str] = []
        confs: List[float] = []
        for j in range(idx.shape[1]):
            k = int(idx[i, j])
            ch = alphabet[k]
            if ch == pad_char:
                continue
            chars.append(ch)
            confs.append(float(probs[i, j]))
        out_texts.append("".join(chars))
        out_confs.append(confs)
    return (out_texts, out_confs) if return_confidence else out_texts


@dataclass(frozen=True)
class PlateConfig:
    """
    Berisi konfigurasi plat yang disamakan dengan  tipe yang diperlukan

    """
    max_plate_slots: int
    alphabet: str
    pad_char: str
    img_height: int
    img_width: int
    keep_aspect_ratio: bool = False
    interpolation: str = "linear"
    image_color_mode: str = "rgb"  # or "grayscale"
    padding_color: Union[Sequence[int], int] = (144, 144, 144)
    # Optional preprocessing (grayscale mode recommended for these)
    use_clahe: bool = False
    clahe_clip: float = 2.0
    clahe_tile: int = 8
    # Apply CLAHE only if mean brightness < gate (0 disables gating)
    clahe_brightness_gate: float = 0.0  # 0..255
    auto_deskew: bool = False
    deskew_threshold_deg: float = 12.0


class OnnxPlateOCR:
    def __init__(
        self,
        onnx_path: str,
        plate_cfg: PlateConfig,
        prefer_trt: bool = False,
        provider: str = "cuda",
        gpu_mem_limit_mb: Optional[int] = None,
    ) -> None:
        if ort is None:
            raise RuntimeError("onnxruntime is not installed; install it to use ONNX OCR")
        available = set(ort.get_available_providers())
        # Session options tuned for low memory
        so = ort.SessionOptions()
        so.enable_mem_pattern = False
        so.enable_cpu_mem_arena = False
        so.intra_op_num_threads = 1

        selected_providers: List[str] = []
        provider_options: List[Dict[str, Any]] = []

        if prefer_trt and "TensorrtExecutionProvider" in available:
            selected_providers.append("TensorrtExecutionProvider")
            provider_options.append({})

        prov = provider.lower()
        if prov == "cuda" and "CUDAExecutionProvider" in available:
            selected_providers.append("CUDAExecutionProvider")
            opts: Dict[str, Any] = {
                # Limit GPU allocator to reduce OOM risk on Jetson; None means default
                # Value is in bytes
                "gpu_mem_limit": int(gpu_mem_limit_mb * 1024 * 1024) if gpu_mem_limit_mb else 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "do_copy_in_default_stream": 1,
                "cudnn_conv_algo_search": "HEURISTIC",
            }
            provider_options.append(opts)

        # Always include CPU fallback
        selected_providers.append("CPUExecutionProvider")
        provider_options.append({})

        #memulai session inference
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=so,
            providers=[p for p in selected_providers if p in available] or ["CPUExecutionProvider"],
            provider_options=provider_options[: len([p for p in selected_providers if p in available])],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.cfg = plate_cfg

    def _preprocess_one(self, img: np.ndarray) -> np.ndarray:
        mode = self.cfg.image_color_mode
        img1 = _ensure_color_mode(img, mode)

        # Optional deskew + CLAHE (grayscale only)
        if mode.lower() == "grayscale":
            gray = img1 if img1.ndim == 2 else cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

            # Auto deskew by minAreaRect orientation if requested
            if bool(self.cfg.auto_deskew):
                try:
                    g = cv2.GaussianBlur(gray, (3, 3), 0)
                    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    contours, _ = cv2.findContours(255 - bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        cnt = max(contours, key=cv2.contourArea)
                        rect = cv2.minAreaRect(cnt)
                        angle = rect[-1]
                        # cv2 returns angle in [-90,0); convert to small tilt around 0
                        if angle < -45:
                            angle = angle + 90
                        if abs(angle) >= float(self.cfg.deskew_threshold_deg) and abs(angle) <= 30.0:
                            h, w = gray.shape[:2]
                            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
                            gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                except Exception:
                    # Fail-safe: ignore deskew errors
                    pass

            # CLAHE with optional brightness gate
            if bool(self.cfg.use_clahe):
                try:
                    mean = float(gray.mean())
                    gate = float(self.cfg.clahe_brightness_gate or 0.0)
                    if gate <= 0.0 or mean < gate:
                        clahe = cv2.createCLAHE(clipLimit=float(self.cfg.clahe_clip), tileGridSize=(int(self.cfg.clahe_tile), int(self.cfg.clahe_tile)))
                        gray = clahe.apply(gray.astype(np.uint8))
                except Exception:
                    pass

            img1 = gray

        # Resize/letterbox to model input
        img2 = _resize_letterbox(
            img1,
            target_h=int(self.cfg.img_height),
            target_w=int(self.cfg.img_width),
            image_color_mode=self.cfg.image_color_mode,
            keep_aspect_ratio=bool(self.cfg.keep_aspect_ratio),
            interpolation=self.cfg.interpolation,
            padding_color=self.cfg.padding_color,
        )
        if self.cfg.image_color_mode.lower() == "grayscale" and img2.ndim == 2:
            img2 = img2[:, :, None]
        return img2.astype(np.uint8)

    def infer_batch(
        self,
        images_bgr: Iterable[np.ndarray],
        return_confidence: bool = False,
        *,
        polygons: Optional[Iterable[Optional[Sequence[Tuple[float, float]]]]] = None,
    ):
        imgs = list(images_bgr)
        if not imgs:
            return []
        polys: List[Optional[Sequence[Tuple[float, float]]]] = []
        if polygons is not None:
            polys = list(polygons)
            # pad/truncate to match imgs length
            if len(polys) < len(imgs):
                polys += [None] * (len(imgs) - len(polys))
            elif len(polys) > len(imgs):
                polys = polys[: len(imgs)]

        # If polygon provided, rectify to target dims before normal preprocessing
        rectified: List[np.ndarray] = []
        if polys:
            target_h = int(self.cfg.img_height)
            target_w = int(self.cfg.img_width)
            for im, poly in zip(imgs, polys):
                if poly and len(poly) == 4:
                    try:
                        src = np.array(poly, dtype=np.float32)
                        dst = np.array([[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]], dtype=np.float32)
                        M = cv2.getPerspectiveTransform(src, dst)
                        im = cv2.warpPerspective(im, M, (target_w, target_h), flags=cv2.INTER_LINEAR)
                    except Exception:
                        pass
                rectified.append(im)
            imgs = rectified

        batch = np.stack([self._preprocess_one(x) for x in imgs], axis=0)  # NHWC uint8
        out = self.session.run([self.output_name], {self.input_name: batch})[0]
        return _decode_logits(
            out,
            max_slots=int(self.cfg.max_plate_slots),
            alphabet=str(self.cfg.alphabet),
            pad_char=str(self.cfg.pad_char),
            return_confidence=return_confidence,
        )
