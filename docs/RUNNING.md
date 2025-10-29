# Running ALPR Jetson (Refactored CLI)

This guide shows how to run the refactored CLI locally (uv) and on Jetson (pip), and how to interpret outputs. JSON/NDJSON is the contract; annotations are optional visualization derived from the same pipeline.

## Prereqs
- Python 3.8 on Jetson (>=3.8,<3.9). Local dev can be newer but keep code 3.8‑safe.
- Models on disk:
  - Detector (TensorRT): `models/detector/*.engine`
  - OCR backend: ONNX (`.onnx` + plate YAML) or TensorRT (`.engine` + `charset.txt`)
- Recommended: NVIDIA Jetson Xavier NX, JetPack 5.1.5, CUDA 11.4, TensorRT 8.5.2

## Install
### Local (uv)
- `uv sync`
- Run commands with `uv run ...` (optional) or your Python directly

### Jetson (pip)
- `python3 -m venv .venv && source .venv/bin/activate`
- `pip install -e . -c constraints-jetson.txt -r requirements-jetson.txt`

## Config defaults
- File: `configs/ocr/plate_defaults.yaml`
  - `min_plate_h`, `min_ar`, `max_ar`, `topk`, `postproc`, `allowed_prefix`
- Optional extended Indonesian prefixes: `configs/ocr/indonesia_prefixes.yaml`
  - If present, overrides `allowed_prefix`
- CLI flags override config values.

## Detector only (sanity)
- Single image:
  `python -m alpr_jetson det-infer --det-engine models/detector/yolov9s_plate_fp16.engine --source path/to/img.jpg --conf 0.5 --iou 0.45 --annotate-dir export/det_ann`
- Directory:
  `python -m alpr_jetson det-infer --det-engine models/detector/yolov9s_plate_fp16.engine --source data/samples --conf 0.5 --iou 0.45 --annotate-dir export/det_ann`

## End‑to‑End (detector + OCR)
- Single image (annotated):
  `python -m alpr_jetson e2e --det-engine models/detector/yolov9s_plate_fp16.engine --onnx models/ocr/cct_s.onnx --plate-config models/ocr/cct_s_v1_global_plate_config.yaml --source path/to/img.jpg --annotate-dir export/e2e_ann`
- Top‑1 detection only (default): use `--topk` to change (e.g., `--topk 3`).
- Size/aspect filters come from config; override with `--min-plate-h`, `--min-ar`, `--max-ar`.

## NDJSON stream (one line per image)
- Directory → NDJSON file:
  `find data/samples -maxdepth 1 -type f -iname '*.jpg' | python -m alpr_jetson e2e-json-stream --det-engine models/detector/yolov9s_plate_fp16.engine --onnx models/ocr/cct_s.onnx --plate-config models/ocr/cct_s_v1_global_plate_config.yaml > export/out.ndjson`
- Inspect: `jq -r '.input + " => " + ((.plates[0].text) // "NO_PLATE")' export/out.ndjson`
- Convert to JSON array: `jq -s '.' export/out.ndjson > export/out.json`

## Debugging
- Crop acceptance debug: add `--debug-crops` to JSON/stream commands
- Bypass acceptance filters: add `--accept-all` (for diagnosis only)
- Enforce Python 3.8 types: `uv run guard` (or `bash tools/py38_guard.sh src`)

## DeepStream stubs
- RTSP smoke: `python -m alpr_jetson rtsp-smoke <rtsp_uri> --latency 200`
- deepstream-app smoke: `python -m alpr_jetson ds-smoke --config configs/deepstream/app_config.txt`

## Notes
- JSON/NDJSON is authoritative. Annotations are derived from the same pipeline.
- Target Jetson ideally uses system OpenCV (avoid opencv-python wheels when possible).
- For performance testing, record NDJSON with latencies and review aggregate stats.
