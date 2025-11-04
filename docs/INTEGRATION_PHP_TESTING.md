# PHP Integration (Testing Quickstart)

Purpose: a minimal, reliable way to test ALPR from PHP via Postman or a simple upload endpoint on Jetson. Use this when you just want to verify plates end‑to‑end without standing up the full FastAPI service.

## When to use this
- You want a quick test path from PHP → ALPR on the Jetson.
- You already save images under `/var/www/...` and call a PHP endpoint.
- You prefer a warmed process (faster) instead of per‑request model reloads.

If you need the full HTTP API instead, see `docs/INTEGRATION_TESTING.md`.

---

## 1) Prereqs on Jetson

- Models present under your repo (adjust paths if different):
  - `models/detector/yolov9-s_plate_fp16.engine`
  - `models/ocr/cct_s_v1_global.onnx`
  - `models/ocr/cct_s_v1_global_plate_config.yaml`
- Web user has GPU access (for TensorRT detector):
  ```bash
  sudo usermod -aG video,render www-data
  sudo systemctl restart php-fpm  # adjust service name if needed
  id -a www-data                  # verify groups include video, render
  ```
- Virtualenv python exists: `/home/iks-ai2/Development/ALPR_Jetson/venv/bin/python`

---

## 2) Recommended: Streaming CLI Helper (faster)

Use the provided helper to keep a warm `e2e-json-stream` process and call it from PHP.

- Include the helper once (absolute path):
  ```php
  require '/home/iks-ai2/Development/ALPR_Jetson/tools/php/alpr_cli_template.php';
  ```
- Call `run_alpr($imagePath, $textOnly=true, $envOverrides)` from your endpoint.
- Minimal endpoint example:
  ```php
  $env = [
    'PYTHON_BIN'   => '/home/iks-ai2/Development/ALPR_Jetson/venv/bin/python',
    'DET_ENGINE'   => '/home/iks-ai2/Development/ALPR_Jetson/models/detector/yolov9-s_plate_fp16.engine',
    'OCR_ONNX'     => '/home/iks-ai2/Development/ALPR_Jetson/models/ocr/cct_s_v1_global.onnx',
    'PLATE_CONFIG' => '/home/iks-ai2/Development/ALPR_Jetson/models/ocr/cct_s_v1_global_plate_config.yaml',
    'USE_STREAM'   => '1',
    'TEXT_ONLY'    => '1',
    'CONF'         => '0.5',
    'IOU'          => '0.45',
    'TOPK'         => '1',
  ];
  $res = run_alpr($absoluteImagePath, true, $env);
  if ($res['ok'] && $res['exit_code'] === 0) {
    echo json_encode(['status'=>'success','plate'=>trim($res['stdout'])]);
  } elseif ($res['exit_code'] === 3) {
    echo json_encode(['status'=>'no_plate','plate'=>'','code'=>3]);
  } else {
    echo json_encode(['status'=>'error','plate'=>'','code'=>$res['exit_code'],'message'=>(string)($res['stderr']??$res['stdout'])]);
  }
  ```

Notes:
- Set `'USE_STREAM'=>'0'` to force one‑shot wrapper calls (`tools/alpr_e2e_json.sh`).
- The helper keeps `PYTHONPATH` and working directory sane; pass absolute model paths.

---

## 3) Alternative: Direct one‑shot script (simpler, slower)

Use the updated `tes.php` in the repo root. It:
- Switches to the repo root and exports absolute model paths.
- Runs `tools/alpr_text_only.py` per request and returns JSON with proper exit codes.

This is good for quick functional checks, but slower due to model reloads.

---

## 4) Swap `tes.php` to use streaming helper (fast path)

Replace the exec block in your current `tes.php` with a call to `run_alpr()` as shown below. This keeps models warm and auto‑recovers from stream failures with a one‑shot fallback.

```php
require '/home/iks-ai2/Development/ALPR_Jetson/tools/php/alpr_cli_template.php';
$env = [
  'PYTHON_BIN'   => '/home/iks-ai2/Development/ALPR_Jetson/venv/bin/python',
  'DET_ENGINE'   => '/home/iks-ai2/Development/ALPR_Jetson/models/detector/yolov9-s_plate_fp16.engine',
  'OCR_ONNX'     => '/home/iks-ai2/Development/ALPR_Jetson/models/ocr/cct_s_v1_global.onnx',
  'PLATE_CONFIG' => '/home/iks-ai2/Development/ALPR_Jetson/models/ocr/cct_s_v1_global_plate_config.yaml',
  'USE_STREAM'   => '1',
  'TEXT_ONLY'    => '1',
  'CONF'         => '0.5',
  'IOU'          => '0.45',
  'TOPK'         => '1',
];
$res = run_alpr($absoluteImagePath, true, $env);
if ($res['ok'] && $res['exit_code'] === 0) {
  echo json_encode(['status'=>'success','plate'=>trim($res['stdout'])]);
} elseif ($res['exit_code'] === 3) {
  echo json_encode(['status'=>'no_plate','plate'=>'','code'=>3]);
} else {
  echo json_encode(['status'=>'error','plate'=>'','code'=>$res['exit_code'],'message'=>(string)($res['stderr']??$res['stdout'])]);
}
```

---

## 5) Postman Setup

- Method: `POST`
- URL: your PHP endpoint (e.g., `http://<jetson>/lpr/tes.php` or your controller)
- Body: `form-data` → key `image` → type `File` → pick a JPEG/PNG.
- Expected responses:
  - `{"status":"success","plate":"B 1234 XYZ"}` (exit 0)
  - `{"status":"no_plate","plate":"","code":3}` (exit 3)
  - `{"status":"error","code":2,"message":"..."}` on errors (missing model, CUDA permission, etc.)

---

## 6) Troubleshooting

- Empty plate with `status: success`:
  - Use the updated example or the streaming helper; ensure you handle the exit code (0/3).
- `cuInit failed` or `NvRmMemInit ... Permission denied`:
  - Add your PHP user to `video,render` groups and restart PHP-FPM.
  - Or run a separate GPU worker user and call it (recommended for production).
- `detector engine not found` or `plate_config not found`:
  - Use absolute model paths; PHP’s working dir is often `/var/www/...`.
- Slow responses:
  - Prefer the streaming helper; it avoids re‑loading TRT/ONNX per request.

---

## 7) Which doc to use?

- For quick testing from PHP: this doc (INTEGRATION_PHP_TESTING.md) + `alpr_cli_template.php` or updated `tes.php`.
- For the full HTTP service (FastAPI) path and more rigorous scenarios: `INTEGRATION_TESTING.md`.
- For PHP design details and options: `INTEGRATION_PHP.md`.
