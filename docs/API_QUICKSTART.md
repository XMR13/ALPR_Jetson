# ALPR API — Quickstart (Local + Jetson)

This is a single, step-by-step guide to run the ALPR API, tune it for night/glare, and integrate it with your existing program. It consolidates details from docs and code.

## 1) Prerequisites
- Hardware target: Jetson Xavier NX 16 GB, JetPack 5.1.5, CUDA 11.4, TensorRT 8.5.2
- Python: 3.8.x on Jetson; local dev can use uv but keep code 3.8-safe
- Models on disk:
  - Detector (TensorRT): `models/detector/*.engine`
  - OCR (default ONNX slot-based): `models/ocr/<ocr>.onnx` + plate YAML (e.g., `models/ocr/cct_s_v1_global_plate_config.yaml`)
    - TensorRT CTC remains available if you build an engine + `models/ocr/charset.txt`

References: server factory at `src/api_server/server.py`, API contract in `docs/API.md`.

## 2) Local (uv) — Start the API
1. Sync deps (optional if using uv):
   - `uv sync`
2. Export environment (ONNX OCR is the default):
  - Detector (required):
    - `export ALPR_DET_ENGINE=models/detector/yolov9-s_plate_fp16.engine`
  - ONNX OCR (recommended):
     - `export ALPR_OCR_ONNX=models/ocr/cct_s_v1_global.onnx`
     - `export ALPR_PLATE_CONFIG=models/ocr/cct_s_v1_global_plate_config.yaml`
     - Optional: `export ALPR_ONNX_PROVIDER=cuda` (or `cpu` on workstations)
  - TensorRT OCR (only if you have the `.engine` + charset):
     - `export ALPR_OCR_ENGINE=models/ocr/ppo_crnn_fp16.engine`
     - `export ALPR_OCR_CHARSET=models/ocr/charset.txt`
   - Optional tuning:
     - `export ALPR_POSTPROC_CONFIG=configs/ocr/postproc_indonesia.yaml`
     - `export ALPR_ALLOWED_PREFIXES=A,B,D,F,E,Z,T`
     - `export ALPR_POSTPROC_STRICT=1`  # enable stricter U/O-at-last truncation (requires char confidences)
3. Run the API:
   - `uv run uvicorn src.api_server.server:create_app --factory --host 0.0.0.0 --port 8080 --workers 1`
4. Smoke test:
   - Health: `curl http://127.0.0.1:8080/healthz` → look for `"runtime_ready": true`
   - ALPR: `curl -F "image=@/path/to/image.jpg" http://127.0.0.1:8080/v1/alpr`

Notes:
- Keep `--workers 1` so models load once and share GPU context.
- ONNX path returns per-character confidences; TRT path currently returns empty `char_confs`.

## 3) Jetson (pip) — Start the API
1. Install runtime:
   - `python3 -m venv .venv && source .venv/bin/activate`
   - `pip install -e . -c constraints-jetson.txt -r requirements-jetson.txt`
2. Export environment (same variables as local; ONNX pair is default, use absolute paths).
3. Run manually:
   - `python3 -m uvicorn src.api_server.server:create_app --factory --host 0.0.0.0 --port 8080 --workers 1`
4. Or install systemd service (recommended):
   - `sudo cp deploy/systemd/alpr-api.service /etc/systemd/system/alpr-api.service`
   - `sudo mkdir -p /etc/alpr && sudo cp deploy/systemd/alpr-api.env.example /etc/alpr/alpr-api.env`
   - Edit `/etc/alpr/alpr-api.env` with absolute model paths and optional YAML.
   - `sudo systemctl daemon-reload && sudo systemctl enable alpr-api && sudo systemctl start alpr-api`
   - Check: `systemctl status alpr-api` and `journalctl -u alpr-api -f`

References: `deploy/systemd/alpr-api.service`, `deploy/systemd/alpr-api.env.example`, docs/SYSTEMD_API.md.

## 4) API Endpoints You Will Use
- Health: `GET /healthz` — uptime, runtime readiness, queue stats
- Synchronous ALPR: `POST /v1/alpr`
  - multipart form: field `image` (jpeg/png)
  - optional: `camera_id`, `request_id`, `min_conf`
  - optional header: `X-ALPR-Token` if `ALPR_API_TOKEN` is set
- Metrics: `GET /metrics` — Prometheus text; includes requests and last latency
- Live preproc update: `POST /v1/config/preproc` — update OCR preprocessing at runtime (see §5)

Code: `src/api_server/server.py:660` (health), `src/api_server/server.py:918` (alpr), `src/api_server/server.py:733` (preproc update).

## 5) Night/Glare/Speck — Tune Without Restart
Use the live config endpoint to adapt to conditions per crop. Examples:

- Night/dark boost (gamma):
  ```bash
  curl -X POST http://127.0.0.1:8080/v1/config/preproc \
    -H 'Content-Type: application/json' \
    -d '{"auto_preproc":true, "gamma_correction":true, "gamma_dark_gate":90, "gamma_value":1.2}'
  ```

- Glare suppression + small bright speck cleanup:
  ```bash
  curl -X POST http://127.0.0.1:8080/v1/config/preproc \
    -H 'Content-Type: application/json' \
    -d '{"suppress_highlights":true, "highlight_threshold":245, "highlight_inpaint_radius":1, "remove_small_bright_specks":true, "speck_area_px":8}'
  ```

- Auto polarity (light-on-dark plates, colored variants):
  ```bash
  curl -X POST http://127.0.0.1:8080/v1/config/preproc \
    -H 'Content-Type: application/json' \
    -d '{"auto_polarity":true, "polarity_dark_mean":110, "polarity_light_mean":175}'
  ```

These map to the underlying TRT preprocessor config. Defaults are conservative; clean daylight crops are left untouched.

## 6) Indonesian Plate Postprocess (U/O ambiguity)
- Default heuristics live in `configs/ocr/postproc_indonesia.yaml` and are applied by the API.
- Optionally enable strict truncation of ambiguous trailing characters (helps U↔O-at-last) if your OCR path provides per-character confidences:
  - `export ALPR_POSTPROC_CONFIG=configs/ocr/postproc_indonesia.yaml`
  - `export ALPR_POSTPROC_STRICT=1`
- ONNX path supplies `char_confs`; TRT path returns empty `char_confs` for now.

Code: `src/ocr_service/postprocess.py`, API wiring at `src/api_server/server.py:1018` and queue path around `src/api_server/server.py:557`.

## 7) Integrate With Your Existing Program
Keep your upstream flow unchanged; call the warm API instead of spawning Python for every request.

- PHP bridge (example in repo): set `ALPR_API_URL=http://<jetson-ip>:8080` and use cURL to POST the captured image to `/v1/alpr`. Keep your current `exec` path as fallback.
  - See: `tools/php/tes.php`, `tools/php/alpr_cli_template.php`, and `docs/INTEGRATION_FLOW.md`.
- Generic HTTP (any language): send multipart with `image=@...` and parse `plates[0].text` in the JSON if present; handle `status == "no_plate"` quickly.

Example cURL (with optional token):
```bash
curl -X POST "http://<host>:8080/v1/alpr" \
  -H "X-ALPR-Token: $ALPR_API_TOKEN" \
  -F "image=@/path/to/capture.jpg" \
  -F "camera_id=gate01" \
  -F "request_id=test-123"
```

## 8) Validate and Smoke
- Quick health: `curl /healthz`
- Test one image: `curl -F "image=@..." /v1/alpr`
- Metrics: `curl /metrics`
- Optional workstation smoke (ONNX OCR only): see `docs/SMOKE_GUIDE.md`
- End-to-end smokes on Jetson: see `docs/RUNNING.md` and `docs/SOAK_RUNBOOK.md`

## 9) Troubleshooting
- 503 runtime unavailable: model paths missing/invalid, or FastAPI deps not installed. Verify env and that `requirements-jetson.txt` (or uv sync) completed.
- 400 empty/invalid image: ensure you POST multipart form-data with `image=@...` and a valid JPEG/PNG.
- Slow first request: CUDA/TRT warmup; subsequent calls should be fast.
- Suffix mistakes (U/O, duplicates): ensure `ALPR_POSTPROC_CONFIG` is set; enable `ALPR_POSTPROC_STRICT=1` if you have char confidences (ONNX path).
- Night/glare/colored plates: tune with `/v1/config/preproc` calls (see §5) instead of restarting.

## 10) Environment Variables (common)
- Detector: `ALPR_DET_ENGINE`
- OCR (default ONNX):
  - `ALPR_OCR_ONNX`, `ALPR_PLATE_CONFIG`, `ALPR_ONNX_PROVIDER=cuda|cpu`, `ALPR_ONNX_GPU_MEM_MB`
  - TensorRT fallback: `ALPR_OCR_ENGINE`, `ALPR_OCR_CHARSET`, optional `ALPR_OCR_INPUT_WIDTH/HEIGHT/CHANNELS`
- Postprocess: `ALPR_POSTPROC_CONFIG`, `ALPR_POSTPROC_STRICT`
- Access/limits: `ALPR_API_TOKEN`, `ALPR_MAX_UPLOAD_BYTES` (default 2_000_000), `ALPR_MIN_CONF`
- Persistence: `ALPR_SNAPSHOTS_DIR`, `ALPR_EVENTS_DB`
- Prefix policy: `ALPR_ALLOWED_PREFIXES` (comma-separated)
- Preproc toggles (TRT, also live via `/v1/config/preproc`): `ALPR_OCR_AUTO_PREPROC`, `ALPR_OCR_AUTO_COLOR`, `ALPR_OCR_GAMMA`, `ALPR_OCR_GAMMA_GATE`, `ALPR_OCR_GAMMA_VALUE`, `ALPR_OCR_AUTO_POLARITY`, `ALPR_OCR_POLARITY_DARK`, `ALPR_OCR_POLARITY_LIGHT`, `ALPR_OCR_INVERT`, `ALPR_OCR_SUPPRESS_HL`, `ALPR_OCR_HL_THRESHOLD`, `ALPR_OCR_HL_INPAINT`, `ALPR_OCR_REMOVE_SPECKS`, `ALPR_OCR_SPECK_AREA`

---

That’s it. Use this Quickstart for operators and integration devs; deeper details live in `docs/API.md`, `docs/RUNNING.md`, `docs/OCR_MODEL.md`, and `docs/OCR_POSTPROCESS.md`.
