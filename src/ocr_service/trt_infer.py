"""TensorRT OCR inference service with graceful fallback.

Implements batch inference pipeline consistent with plan.md §5 (Week 2 Day 8–9):
- preprocess (grayscale → CLAHE → normalize)
- TRT execution (when available)
- greedy CTC decode to text

If TensorRT is unavailable in the environment, the service remains importable
and returns placeholder outputs so the rest of the app can be developed.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence
from pathlib import Path

import numpy as np

try:  # Optional: these are only available on Jetson/host with TensorRT
    import tensorrt as trt  # type: ignore
    import pycuda.driver as cuda  # type: ignore
    import pycuda.autoinit  # noqa: F401  # type: ignore
    _TRT_AVAILABLE = True
except Exception:  # pragma: no cover - on CI or minimal env
    trt = None  # type: ignore
    cuda = None  # type: ignore
    _TRT_AVAILABLE = False

from .preprocess import PreprocConfig, prepare_ocr_batch


class _TRTModule:
    """Minimal TRT runner for a single input (NCHW) and single logits output."""

    def __init__(self, engine_path: str):
        if not _TRT_AVAILABLE:
            raise RuntimeError("TensorRT is not available in this environment")
        logger = trt.Logger(trt.Logger.INFO)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            if self.engine is None:
                raise RuntimeError("Failed to deserialize OCR engine")
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.bindings = [0] * self.engine.num_bindings
        self.host = {}
        self.device = {}
        # assume single tensor input (NCHW) and one output logits [N,T,C]
        self.inp_idx = next(i for i in range(self.engine.num_bindings) if self.engine.binding_is_input(i))
        self.out_idx = next(i for i in range(self.engine.num_bindings) if not self.engine.binding_is_input(i))

    def _alloc(self, idx: int, shape: Sequence[int], dtype: np.dtype) -> None:
        size = int(np.prod(shape))
        host = cuda.pagelocked_empty(size, dtype)
        dev = cuda.mem_alloc(host.nbytes)
        self.host[idx] = (host, dtype, tuple(shape))
        self.device[idx] = dev
        self.bindings[idx] = int(dev)

    def infer(self, x: np.ndarray) -> np.ndarray:
        # Set shape if dynamic
        if -1 in tuple(self.engine.get_binding_shape(self.inp_idx)):
            self.context.set_binding_shape(self.inp_idx, x.shape)
        # allocate/reuse
        in_shape = tuple(self.context.get_binding_shape(self.inp_idx))
        in_dtype = np.float32
        if (self.inp_idx not in self.host) or (self.host[self.inp_idx][2] != in_shape):
            self._alloc(self.inp_idx, in_shape, in_dtype)
        np.copyto(self.host[self.inp_idx][0].reshape(in_shape), x.astype(in_dtype))

        # prepare output
        out_shape = tuple(self.context.get_binding_shape(self.out_idx))
        out_dtype = np.float32
        if (self.out_idx not in self.host) or (self.host[self.out_idx][2] != out_shape):
            self._alloc(self.out_idx, out_shape, out_dtype)

        # H2D
        cuda.memcpy_htod_async(self.device[self.inp_idx], self.host[self.inp_idx][0], self.stream)
        # Exec
        self.context.execute_async_v2(self.bindings, self.stream.handle)
        # D2H
        cuda.memcpy_dtoh_async(self.host[self.out_idx][0], self.device[self.out_idx], self.stream)
        self.stream.synchronize()
        return self.host[self.out_idx][0].reshape(out_shape)


def _load_charset(path: Optional[str]) -> List[str]:
    if not path:
        # Default Latin uppercase + digits; index 0 reserved for CTC blank
        return ["<blank>"] + list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    lines = [ln.strip("\n\r") for ln in Path(path).read_text(encoding="utf-8").splitlines()]
    # Expect file without blank token; prepend one
    return ["<blank>"] + [ch for ch in lines if ch]


def _ctc_greedy_decode(logits: np.ndarray, charset: List[str], blank_idx: int = 0) -> List[str]:
    """Greedy CTC decode.

    logits: [N, T, C] float32
    Returns: list of strings length N.
    """
    if logits.ndim != 3:
        raise ValueError(f"Expected [N,T,C] logits, got {logits.shape}")
    N, T, C = logits.shape
    idx = logits.argmax(axis=2)  # [N,T]
    out: List[str] = []
    for n in range(N):
        prev = -1
        chars: List[str] = []
        for t in range(T):
            k = int(idx[n, t])
            if k == blank_idx:
                prev = -1
                continue
            if k == prev:
                continue
            prev = k
            if 0 <= k < len(charset):
                chars.append(charset[k])
        out.append("".join(chars))
    return out


class OCRService:
    def __init__(
        self,
        engine_path: Optional[str] = None,
        charset_path: Optional[str] = None,
        preproc: PreprocConfig = PreprocConfig(),
        logits_layout: str = "NTC",
    ) -> None:
        if logits_layout not in {"NTC", "NCT"}:
            raise ValueError("logits_layout must be 'NTC' or 'NCT'")
        self.preproc = preproc
        self.charset = _load_charset(charset_path)
        self.logits_layout = logits_layout
        self._runner: Optional[_TRTModule] = None
        if engine_path and _TRT_AVAILABLE:
            try:
                self._runner = _TRTModule(engine_path)
            except Exception as e:  # keep importable
                print(f"[OCRService] Failed to load engine: {e}")
                self._runner = None

    def infer_batch(self, images_bgr: Iterable[np.ndarray]) -> List[str]:
        """Infer a batch of BGR plate crops to strings.

        Falls back to placeholder strings if TRT runtime is not available.
        """
        imgs = list(images_bgr)
        if not imgs:
            return []
        if self._runner is None:
            return ["<ocr-unavailable>"] * len(imgs)

        x = prepare_ocr_batch(imgs, self.preproc)  # [N,1,H,W]
        logits = self._runner.infer(x)
        if logits.ndim != 3:
            raise ValueError(f"Expected 3D logits, got shape {logits.shape}")
        if self.logits_layout == "NCT":
            logits = np.transpose(logits, (0, 2, 1))
        texts = _ctc_greedy_decode(logits, self.charset)
        return texts
