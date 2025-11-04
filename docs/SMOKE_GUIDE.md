# Workstation Smoke Guide (Non‑Jetson Only)

Audience: developer/operator. Goal: validate non‑Jetson modules on a workstation. Do NOT run TensorRT or DeepStream here. Use OCR‑only ONNX CPU smokes and library‑only tests.

## Prerequisites
- Python env prepared (`uv sync` or `pip install -e .`).
- Detector + OCR artifacts available at the paths referenced by commands below (adjust as needed).
- Optional: sample images under `data/raw/...` for quick sanity checks.

## OCR‑Only ONNX Smoke (workstation‑safe)

Run ONNX OCR on CPU over a folder of crops (no TRT, no DeepStream):

```bash
uv run python -m alpr_jetson ocr-infer \
  --onnx models/ocr/cct_s_v1_global.onnx \
  --plate-config models/ocr/cct_s_v1_global_plate_config.yaml \
  --onnx-provider cpu \
  --source data/labeled/ocr_crops/
```

Tips:
- Use `--text-only` to output only recognized strings.
- Gate preprocessing with the YAML (CLAHE/deskew) to simulate night shots.

## Latency Notes

On workstations, avoid `e2e-json`/`e2e-json-stream` and `e2e --stats` because they require a detector; the repo’s detector path targets TensorRT on Jetson. If you need timing, wrap `ocr-infer` with `time` or a minimal Python timer.

## Capturing Outputs for Progress Logs

Record OCR‑only throughput/latency and include brief notes in `progress/`.

## Troubleshooting

- Missing models: ensure the `.engine`/`.onnx` paths exist or update the command args.
- Slow startup: first call builds the CUDA kernels; subsequent runs should hit warmed paths.
- GPU OOM on constrained devices: add `--onnx-gpu-mem-limit-mb 512` or switch to CPU via `--onnx-provider cpu`.
- Invalid JSON: check that helper scripts don’t inject logging into stdout; they should keep logs on stderr.

## Jetson Notes

End‑to‑end and NDJSON smokes are Jetson‑only (TRT + detector). Use `docs/SOAK_RUNBOOK.md` once on device.
