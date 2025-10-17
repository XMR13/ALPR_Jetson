# OCR Notes — CRNN/CTC Export and TensorRT (Jetson Xavier NX)

This document captures the practical steps to export a CRNN/Paddle‑style OCR recognizer to ONNX, build a TensorRT engine on Jetson NX (JetPack 5.1.5, TRT 8.5.2), and integrate it with this repository.

## 1) Model I/O Contract

- Input: `NCHW` float32 tensor, shape `[N, 1, 32, 160]` (grayscale), normalized `(x - 0.5) / 0.5`.
- Output: logits tensor `[N, T, C]` for CTC decoding.
  - `C` = number of classes including CTC blank. We reserve index `0` for the blank symbol.
  - Charset covers digits `0–9` and uppercase `A–Z` (no lowercase). Store as `models/ocr/charset.txt` (one char per line; blank is prepended by runtime).

Preprocessing is implemented in `src/ocr_service/preprocess.py` and post‑processing in `src/ocr_service/postprocess.py`.

## 2) Export to ONNX (training environment)

Example PyTorch export sketch (adapt to your model):

```python
import torch

model = ...  # your CRNN/Paddle-style recognizer
model.eval()

dummy = torch.randn(1, 1, 32, 160)  # NCHW
torch.onnx.export(
    model,
    dummy,
    "models/ocr/ppo_crnn.onnx",
    input_names=["input"],
    output_names=["logits"],
    opset_version=13,
    dynamic_axes={"input": {0: "N"}, "logits": {0: "N"}},
)
```

Checklist:
- Output must be raw logits (no softmax) with shape `[N,T,C]`.
- Input layout is `NCHW`; keep grayscale single channel.
- Validate with `onnxruntime` on your dev box (compare greedy CTC decode against PyTorch inference).

## 3) Build TensorRT Engine (Jetson NX)

Build FP16 engine with `trtexec` (TensorRT 8.5.2):

```bash
trtexec \
  --onnx=models/ocr/ppo_crnn.onnx \
  --saveEngine=models/ocr/ppo_crnn_fp16.engine \
  --explicitBatch \
  --minShapes=input:1x1x32x160 \
  --optShapes=input:4x1x32x160 \
  --maxShapes=input:8x1x32x160 \
  --fp16 \
  --workspace=2048
```

Notes:
- Replace `input` with your actual ONNX input name (`trtexec --onnx=... --verbose` shows names).
- Keep input NCHW; if your export is NHWC, either re‑export or transpose in preprocessing.
- Save artifacts under `models/ocr/`.

## 4) Charset File

Create `models/ocr/charset.txt` with one character per line (no blank entry). Example:

```
0
1
2
...
9
A
B
...
Z
```

Our runtime (`OCRService`) prepends a blank token at index 0 automatically.

## 5) Sanity Verification (on Jetson)

Quick check for shapes and decoding correctness:

```python
from ocr_service.trt_infer import OCRService
from ocr_service.preprocess import PreprocConfig
import cv2

svc = OCRService(
    engine_path="models/ocr/ppo_crnn_fp16.engine",
    charset_path="models/ocr/charset.txt",
    preproc=PreprocConfig(input_width=160, input_height=32)
)
img = cv2.imread("data/labeled/ocr_crops/example.jpg")
print(svc.infer_batch([img]))
```

Record in `progress/`:
- Output dims `T` and `C` (from engine output), and a quick CER/SER on 50–100 validation crops.

## 6) Integration Choices

- Baseline: FastAPI endpoint `/v1/ocr` in `src/ocr_service/app.py` (OK for low volume).
- Preferred: ZeroMQ/IPC to send raw crop bytes from DeepStream to OCR for minimal overhead.
- Post‑processing and temporal voting per track are implemented in `src/ocr_service/postprocess.py`.

## 7) Troubleshooting

- If `trtexec` builds but runtime fails: check ONNX opset, ensure no unsupported ops (e.g., exotic activations). Use fully‑conv sequence heads to reduce LSTM plugin dependencies if needed.
- Verify input normalization matches training `(x-0.5)/0.5` and that inference uses grayscale.
- Mismatched `C` (charset) will cause wrong decoding; ensure `charset.txt` matches training.

