# Integration Testing Guide — PHP Capture System ↔ ALPR Jetson

This guide explains how to run the synchronous `/v1/alpr` endpoint so the existing
PHP-based capture system can send snapshots, receive plate text, and validate the
pipeline before the full real-time DeepStream flow is complete.

---

## 1. Prerequisites on Jetson

- JetPack 5.1.5 (CUDA 11.4, TensorRT 8.5.2), Python 3.8.x.
- TensorRT detector engine: `models/detector/yolov9-s_plate_fp16.engine`.
- OCR backend (choose one):
  - TensorRT CTC: `models/ocr/ppo_crnn_fp16.engine` + `models/ocr/charset.txt`
  - ONNX slot model: `models/ocr/cct_s_v1_global.onnx` +
    `models/ocr/cct_s_v1_global_plate_config.yaml`
- Python deps (inside virtualenv or system):
  - `pip install -e .`
  - `pip install fastapi uvicorn pyyaml` (for the HTTP service)

---

## 2. Environment Variables

Set on Jetson before launching the server (adapt paths as needed):

```bash
export ALPR_DET_ENGINE=models/detector/yolov9-s_plate_fp16.engine

# Option A – TensorRT OCR
export ALPR_OCR_ENGINE=models/ocr/ppo_crnn_fp16.engine
export ALPR_OCR_CHARSET=models/ocr/charset.txt

# Option B – ONNX OCR (slot-based)
# export ALPR_OCR_ONNX=models/ocr/cct_s_v1_global.onnx
# export ALPR_PLATE_CONFIG=models/ocr/cct_s_v1_global_plate_config.yaml
# export ALPR_ONNX_PROVIDER=cuda           # or cpu
# export ALPR_ONNX_GPU_MEM_MB=512          # limit CUDA EP allocator

# Optional settings
export ALPR_API_TOKEN=SECRET123            # shared token required from PHP
export ALPR_MIN_CONF=0.5                   # detector confidence threshold
export ALPR_ALLOWED_PREFIXES=A,B,D,F,E,Z,T # comma separated regional prefixes
export ALPR_DEFAULT_CAMERA_ID=cam01
```

---

## 3. Start the ALPR HTTP Service

```bash
uvicorn api_server.server:create_app --factory --host 0.0.0.0 --port 8000
```

Health & metrics checks:

```bash
curl http://<jetson_ip>:8000/healthz
curl http://<jetson_ip>:8000/metrics
```

`/healthz` returns `runtime_ready=true` when engines load successfully.

---

## 4. PHP Integration (Synchronous)

**HTTP contract** — documented in `docs/API.md`. Summary:
- Endpoint: `POST /v1/alpr`
- Auth header: `X-ALPR-Token: SECRET123` (if token set)
- Multipart fields: `image` (required), `camera_id`, `request_id`, `min_conf`
- JSON response contains `status`, `plates` array (text, confs, bbox), and latency.

**cURL example (manual test):**

```bash
curl -X POST "http://<jetson_ip>:8000/v1/alpr" \
  -H "X-ALPR-Token: SECRET123" \
  -F "image=@/path/to/capture.jpg" \
  -F "camera_id=gate01" \
  -F "request_id=test-123" \
  -F "min_conf=0.45"
```

**PHP snippet (using cURL extension):**

```php
$ch = curl_init("http://jetson.local:8000/v1/alpr");
curl_setopt_array($ch, [
    CURLOPT_HTTPHEADER => ["X-ALPR-Token: SECRET123"],
    CURLOPT_POST => true,
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_POSTFIELDS => [
        "image"      => new CURLFile("/path/to/capture.jpg"),
        "camera_id"  => "gate01",
        "request_id" => "event-123",
        "min_conf"   => "0.45",
    ],
]);
$response = curl_exec($ch);
$code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
curl_close($ch);

if ($code === 200) {
    $payload = json_decode($response, true);
    // $payload["plates"][0]["text"] contains the normalized plate string.
} else {
    error_log("ALPR request failed: HTTP $code – $response");
}
```

`status` will be `"ok"` if plates found, `"no_plate"` otherwise. Each plate entry also
contains `det_conf`, `plate_conf`, `ocr_raw`, `char_confs`, and bounding box coordinates.

For a PHP-only testing path (Postman → PHP upload) that doesn’t require running this HTTP service,
see `INTEGRATION_PHP_TESTING.md` and use the streaming CLI helper from your PHP endpoint.

---

## 5. Troubleshooting

| Symptom                                 | Check/Resolution                                                                 |
|-----------------------------------------|-----------------------------------------------------------------------------------|
| `/healthz` shows `runtime_ready=false`  | Verify env vars, model paths, TRT/PyCUDA imports. Restart service after fixing.   |
| HTTP 503 on `/v1/alpr`                  | Runtime failed to load; inspect `runtime_error` in `/healthz`.                    |
| HTTP 401                                | Token mismatch; ensure PHP header matches `ALPR_API_TOKEN`.                       |
| `"detection failed: ..."`               | Confirm detector engine matches TRT version; try `ALPR_DET_ENGINE` path.          |
| `"OCR failed: ..."`                     | Ensure OCR backend configured; ONNX requires `pyyaml` + valid plate YAML.         |
| Memory pressure (ONNX OCR)              | Reduce `ALPR_ONNX_GPU_MEM_MB` or switch provider to `cpu`.                        |

---

## 6. Next Steps toward Real-Time Pipeline

This synchronous flow is temporary for integration tests. The remaining plan items:
- Build the DeepStream → OCR bridge with back-pressure.
- Persist events + snapshots via SQLite and emit webhooks/metrics.
- Add OCR CER/SER evaluation and logging enhancements.
- Wire structured logging, tegrastats sampling, and final docs per plan.md.

These enhancements will allow the PHP system to receive live events once the
real-time pipeline is online.
