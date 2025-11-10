# Current Integration Flow (RFID → PHP Bridge → ALPR)

This document describes how the system operates today and how to improve
performance without changing the upstream C++ gate program.

## As‑Is Flow

- Driver taps RFID at the gate.
- Gate C++ program captures a snapshot from CCTV.
- C++ posts the image to a PHP endpoint (bridge).
- PHP executes a local Python script (`tools/alpr_text_only.py`) which loads
  detection/OCR models, runs inference, and returns text/JSON to the gate.

Pros: simple, isolated. Cons: slow starts — Python and models load per request.

## Primary Performance Issue

Cold start on every request dominates latency:
- Python process spawn + model deserialization (detector + OCR) every time.
- No reuse across requests; GPU contexts re‑init each call.

## Recommended Mitigation (No C++ Changes)

Run a long‑lived ALPR API service and have PHP call it instead of `exec`.

### 1) Start the ALPR API (keeps models warm)

Environment (example):

```
export ALPR_DET_ENGINE=models/detector/yolov9-s_plate_fp16.engine
export ALPR_OCR_ENGINE=models/ocr/ppo_crnn_fp16.engine
export ALPR_OCR_CHARSET=models/ocr/charset.txt
export ALPR_SNAPSHOTS_DIR=export/snapshots
export ALPR_EVENTS_DB=export/events.sqlite
export ALPR_ALLOWED_PREFIXES=A,B,D,F,E,Z,T
```

Run with uvicorn:

```
uv run uvicorn src.api_server.server:create_app --factory --host 0.0.0.0 --port 8080 --workers 1
```

Notes:
- Models are loaded once at startup and kept in memory.
- Endpoint: `POST /v1/alpr` (multipart form: field `image`), returns JSON text and confidences.

### 2) Point PHP to the API (zero changes to C++)

Option A — External PHP (production):
- Use cURL to POST the image to `http://<jetson-ip>:8080/v1/alpr`.
- Set timeouts (connect ≤ 500 ms, total ≤ 1500 ms).
- On failure, fall back to the local exec path if needed.

Option B — Repo test PHP (for reference):
- `tools/php/tes.php` now supports `ALPR_API_URL`.
- Set `ALPR_API_URL=http://127.0.0.1:8080` and the script will call the API first and only `exec` Python as a fallback.

### 3) Optional Latency Tweaks

- Burst of 3–5 frames around RFID tap; pick the sharpest plate (largest bbox,
  highest focus). Implement selection in PHP before sending a single best frame.
- Enforce per‑request timeout and return a clear message for `no_plate` to allow
  a quick retry with the next frame.

## API Contract (Synchronous)

`POST /v1/alpr` (multipart)
- Form field: `image` — JPEG/PNG
- Optional: `camera_id` (string)

Example JSON response:

```
{
  "schema_version": "1.0",
  "camera_id": "rfid-gate",
  "ts": "2025-11-07T09:43:12.345Z",
  "plate": "B 9418 QW",
  "plate_conf": 0.97,
  "char_confs": [0.99, 0.98, 0.97, 0.96],
  "bbox": [x, y, w, h],
  "track_id": null,
  "frame_id": null,
  "snapshots": {
    "plate_jpeg_b64": "..."
  },
  "processing": { "det_ms": 8.5, "ocr_ms": 4.1, "total_ms": 18.3 }
}
```

## Deployment Notes

- For persistent service, add a systemd unit (see `plan.md §11`) pointing to
  the uvicorn command above. Ensure the `alpr` user has read access to models.
- On Jetson, install runtime deps via `requirements-jetson.txt -c constraints-jetson.txt`.

## Rollback / Fallback

- If the API is unreachable, PHP can fall back to the local `exec` path to
  preserve behavior (at a higher latency). The repo’s test PHP demonstrates
  this pattern.

## Next Steps (Optional)

- Add best‑of‑N burst selection at PHP layer for higher OCR stability.
- Add confidence thresholds and clear return codes (`no_plate`, `low_conf`).
- Log lightweight metrics in API (`/metrics`) and feed to your monitoring.

