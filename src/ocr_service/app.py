"""FastAPI OCR microservice skeleton.

This app wraps the OCRService and exposes a simple HTTP endpoint.
It is importable even when FastAPI is not installed (for minimal envs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except Exception:  # pragma: no cover
    FastAPI = None  # type: ignore
    BaseModel = object  # type: ignore
    HTTPException = Exception  # type: ignore

import base64
import cv2
import numpy as np

from .trt_infer import OCRService
from .preprocess import PreprocConfig


@dataclass
class AppConfig:
    engine_path: Optional[str] = None
    charset_path: Optional[str] = None
    input_width: int = 160
    input_height: int = 32
    mean: float = 0.5
    std: float = 0.5


class OCRRequest(BaseModel):  # type: ignore[misc]
    image_b64: str


def create_app(cfg: AppConfig = AppConfig()):  # -> "FastAPI | None":
    if FastAPI is None:
        return None

    app = FastAPI(title="ALPR OCR Service", version="0.0.0")
    svc = OCRService(
        engine_path=cfg.engine_path,
        charset_path=cfg.charset_path,
        preproc=PreprocConfig(
            input_width=cfg.input_width,
            input_height=cfg.input_height,
            mean=cfg.mean,
            std=cfg.std,
        ),
    )

    @app.post("/v1/ocr")
    def ocr(req: OCRRequest):  # type: ignore[no-redef]
        try:
            data = base64.b64decode(req.image_b64)
        except Exception:
            raise HTTPException(status_code=400, detail="invalid base64 image")
        img_array = np.frombuffer(data, dtype=np.uint8)
        bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(status_code=400, detail="unable to decode image")

        texts = svc.infer_batch([bgr])
        return {"text": texts[0] if texts else ""}

    return app


def main() -> None:
    print("alpr-ocr app stub (serve via uvicorn)")


if __name__ == "__main__":
    main()

