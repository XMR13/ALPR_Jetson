# ALPR API — HTTP Contract

## Overview

The ALPR Jetson service exposes two sets of interfaces:

1. **Synchronous testing endpoint** — `/v1/alpr`  
   Accepts a captured image (multipart upload), runs detector + OCR, and
   returns plate text(s) immediately.  Useful for validating integrations with
   existing capture systems that already store snapshots.

2. **Pipeline endpoints** — `/healthz`, `/metrics`, `/v1/events`, `/v1/ws`, etc.  
   These will serve the real-time DeepStream → OCR pipeline once wired.

This document focuses on the synchronous endpoint used during integration tests.

---

## `POST /v1/alpr` — Synchronous Plate Recognition

**Authorization**

- Optional shared token set via `ALPR_API_TOKEN`.  
  Clients must send header `X-ALPR-Token: <token>`.

**Request**

- `Content-Type: multipart/form-data`

| Field        | Type        | Required | Description                                           |
| ------------ | ----------- | -------- | ----------------------------------------------------- |
| `image`      | file (jpg/png) | ✅      | CCTV snapshot or crop                                 |
| `camera_id`  | form field  | ❌       | Overrides default camera id (defaults to `cam01`)     |
| `request_id` | form field  | ❌       | Echoed back in response (auto-generated if omitted)   |
| `min_conf`   | form field (float) | ❌ | Detector confidence threshold (defaults to env `ALPR_MIN_CONF`, 0.5). |

Example:

```bash
curl -X POST "http://jetson.local:8000/v1/alpr" \
  -H "X-ALPR-Token: SECRET123" \
  -F "image=@/path/to/capture.jpg" \
  -F "camera_id=gate01" \
  -F "request_id=test-123" \
  -F "min_conf=0.45"
```

**Successful Response (HTTP 200)**

```json
{
  "request_id": "test-123",
  "camera_id": "gate01",
  "ts": "2025-10-23T09:12:34.567890Z",
  "status": "ok",
  "plates": [
    {
      "bbox": [512, 384, 676, 432],
      "det_conf": 0.92,
      "ocr_raw": "B9418QW",
      "text": "B 9418 QW",
      "valid": true,
      "plate_conf": 0.88,
      "char_confs": [0.99, 0.98, 0.93, 0.90, 0.88, 0.91, 0.92],
      "class_id": 0
    }
  ],
  "latency_ms": {
    "det": 8.37,
    "ocr": 4.12,
    "total": 12.49
  }
}
```

- `status` values: `"ok"` (plates detected), `"no_plate"` (no boxes above `min_conf`).
- `plate_conf` combines detector confidence and average per-character logits (if available).
- `char_confs` present when using ONNX OCR with confidence export.  (List is empty for TensorRT OCR.)

**Error Responses**

| HTTP Code | Condition                               | Message example                               |
|-----------|-----------------------------------------|-----------------------------------------------|
| 400       | Missing/invalid image payload           | `"empty image payload"`                       |
| 401       | Token required but missing/mismatch     | `"invalid token"`                             |
| 500       | Runtime exception (detector/OCR failure)| `"detection failed: ..."`                     |
| 503       | Runtime not initialized / config missing| `"runtime not initialized"`                   |

When the service starts, it attempts to preload detector and OCR runtimes using
config/environment variables:

- `ALPR_DET_ENGINE` — Detector TensorRT `.engine` (required).
- OCR (choose one):
  - TensorRT: `ALPR_OCR_ENGINE` + `ALPR_OCR_CHARSET` (plus optional dims/layout overrides).
  - ONNX slot-based: `ALPR_OCR_ONNX` + `ALPR_PLATE_CONFIG` (YAML, see README).
- Optional tuning:
  - `ALPR_ONNX_PROVIDER` (`cuda`/`cpu`), `ALPR_ONNX_GPU_MEM_MB`
  - `ALPR_MIN_CONF`, `ALPR_ALLOWED_PREFIXES` (comma-separated)
  - `ALPR_DEFAULT_CAMERA_ID`, `ALPR_API_TOKEN`

`/healthz` reports a `runtime_ready` flag and any load error message.

---

## Prometheus Metrics

`GET /metrics` exposes the following gauges/counters in Prometheus text format:

- `alpr_fps`, `alpr_queue_len`, `alpr_gpu_util` — placeholders for the real-time pipeline.
- `alpr_requests_total` — total `/v1/alpr` invocations.
- `alpr_last_latency_ms` — total latency of the most recent `/v1/alpr` call.
- `alpr_last_status_ok` — `1` if last response returned `status == "ok"`, else `0`.

Additional histograms (latency buckets, error counters) will be added alongside the DeepStream bridge.

---

## WebSocket & Future Endpoints

- `GET /v1/events`, `POST /v1/hooks`, `WS /v1/ws` are currently stubs.
- Once the DS → OCR pipeline is linked, these will emit/live-stream structured events per `plan.md §9`, with persistence to SQLite and snapshot storage.

Stay tuned for updates as milestones from plan.md are implemented.
