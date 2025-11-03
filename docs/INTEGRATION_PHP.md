# PHP ↔️ ALPR Integration Guide

This note covers how to connect a PHP application to the Jetson-based ALPR pipeline, including duplicate suppression, retry logic, and OCR validation. It mirrors the tasks in plan.md §5 (Week 2, Day 12–13).

## 1. Preferred Architecture (HTTP)
- Run the API service once (`uvicorn --factory src.api_server.server:create_app`).
- PHP posts every frame to `POST /v1/alpr` using multipart form (`image` field + optional `camera_id`, `request_id`).
- API response contains:
  - `status`: `ok` or `no_plate`
  - `plates`: list of detections `{text, ocr_raw, det_conf, plate_conf, valid, bbox, char_confs}`
  - `latency_ms`: `{det, ocr, total}`
- `/healthz` provides readiness; `/metrics` exports Prometheus counters.

## 2. Temporary CLI Options (no HTTP)
- `python -m alpr_jetson e2e-json …`: single image → JSON (already used by `tools/alpr_e2e_json.sh`).
- `python -m alpr_jetson e2e-json-stream …`: loads models once, then reads image paths from `stdin` and emits NDJSON (`{"input": ",…"}`).
  ```bash
  tail -F /path/to/inbox.txt | python -m alpr_jetson e2e-json-stream \
    --det-engine models/detector/yolov9-s_plate_fp16.engine \
    --onnx models/ocr/cct_s_v1_global.onnx \
    --plate-config models/ocr/cct_s_v1_global_plate_config.yaml
  ```
- `tools/alpr_e2e_json.sh`: wrapper that only requires an image path; ideal when PHP spawns a subprocess or when you explicitly disable streaming (see `USE_STREAM=0` below).
  - Text-only: `TEXT_ONLY=1 tools/alpr_e2e_json.sh /path/to/frame.jpg` → prints only best plate text to stdout; defaults to exit 3 when no/invalid text (see options below for fallbacks).
  - Annotate: `ANNOTATE_DIR=/var/www/ann tools/alpr_e2e_json.sh /path/to/frame.jpg` → also saves annotated visualization.

  Wrapper controls (env):
  - `POSTPROC=indonesia|none` — override CLI default post-processing.
  - `ALLOWED_PREFIX="B D F ..."` — space/comma-separated allowed prefixes when using `indonesia` post-proc.
  - `TEXT_MODE=best|raw` — print normalized text (`best`, default) or raw OCR (`raw`).
  - `TEXT_ALLOW_INVALID=1` — print text even when `valid=false` (useful for triage or permissive flows).
  - `TEXT_NO_PLATE=NO_PLATE` — when rc=3, still print this placeholder to stdout (exit code remains 3).

## 3. Handling Continuous Image Streams

### 3.1 Atomic File Handoff
- Producer writes `frame.jpg.tmp` then renames to `frame.jpg` when closed.
- PHP waits for a stable file size before processing.
- After processing, archive or delete the file; keep a `failed/` folder for triage.

### 3.2 Duplicate Suppression
- Compute a SHA-1 hash of the image bytes and send it as part of `request_id` (e.g., `ts_hash`).
- Maintain an LRU cache (last 500 hashes for ~60 s). If hash repeats, skip or return the cached result.
- Optional: add perceptual hash (pHash/dHash) if near-identical frames are common.

### 3.3 Pacing & Backpressure
- Limit concurrent API calls (1–2 outstanding on Jetson NX).
- If input queue grows, drop frames strategically (e.g., keep one every N).
- Use the API’s `latency_ms.total` to detect overload and throttle upstream.

## 4. Failure & Quality Handling

### 4.1 Detection Failures
- If response `status == "no_plate"`, retry with the next 1–2 sequential frames (not the same image).
- Do not loop on the same file; log and move on after N retries.
- Keep metrics: count consecutive `no_plate` events to alert when the camera view is obstructed.

### 4.2 OCR Confidence / Wrong Text
- Use `plate_conf` (combined detector + char confidences) and `valid` flag to gate acceptance.
- Default heuristic: accept when `plate_conf ≥ 0.55` and `valid == true`.
- For streams, keep a sliding window of the last N results per source and choose the max-confidence plate (temporal vote).
- Surface disagreements for manual review (store raw crop + JSON in `failed/` for auditing).

### 4.3 Transport / Runtime Errors
- Timeouts & 5xx: exponential backoff (e.g., 0.5 s, 1 s, 2 s) with a cap of 3 attempts.
- 4xx (bad request, oversized image): fix the payload; do not retry automatically.
- If API is unreachable, enter a “circuit breaker” state: pause new work, probe `/healthz` every few seconds.

## 5. Suggested PHP Loop (Pseudo-code)
```php
// watch directory or queue for new images
foreach ($images as $imgPath) {
    if (!file_exists($imgPath)) continue;

    $hash = sha1_file($imgPath);
    if ($cache->seenRecently($hash)) {
        $cache->touch($hash);
        continue; // duplicate
    }

    $response = post_to_alpr($imgPath, $hash); // HTTP or shell wrapper
    if (!$response) {
        log_error($imgPath, 'request failed');
        continue;
    }

    if ($response['status'] === 'ok') {
        $best = pick_best_plate($response['plates']);
        if ($best && $best['plate_conf'] >= 0.55 && $best['valid']) {
            persist_result($imgPath, $best);
        } else {
            queue_for_review($imgPath, $response);
        }
    } else {
        enqueue_retry($imgPath, next_frame($imgPath));
    }

    archive($imgPath);
}
```

## 6. Tooling Checklist
- `python -m alpr_jetson e2e-json …` – one-shot JSON.
- `python -m alpr_jetson e2e-json-stream …` – NDJSON stream (stdin).
- `tools/alpr_e2e_json.sh` – simple wrapper for PHP (use `escapeshellarg($img)`):
  - JSON mode: `$cmd = 'tools/alpr_e2e_json.sh ' . escapeshellarg($img);`
  - TEXT_ONLY mode: `$cmd = 'TEXT_ONLY=1 tools/alpr_e2e_json.sh ' . escapeshellarg($img);`
- `/v1/alpr` HTTP endpoint – production target.

### Exit Codes (wrapper)
- 0: success (JSON printed or text printed)
- 2: usage/model/path/runtime error
- 3: TEXT_ONLY mode only — no plate detected or text invalid/empty

### PHP Examples
JSON mode (default):
```php
$cmd = 'tools/alpr_e2e_json.sh ' . escapeshellarg($img);
$json = shell_exec($cmd);
$data = json_decode($json, true);
```

TEXT_ONLY with ONNX and annotation:
```php
$cmd = 'TEXT_ONLY=1 TEXT_ALLOW_INVALID=1 OCR_BACKEND=onnx ANNOTATE_DIR=/var/www/plates tools/alpr_e2e_json.sh ' . escapeshellarg($img);
$out = [];
$rc = 1;
exec($cmd, $out, $rc);
if ($rc === 0) {
    $plate = trim(implode("\n", $out));
} else if ($rc === 3) {
    // No plate or OCR invalid — optionally set TEXT_NO_PLATE=NO_PLATE to emit a placeholder
} else {
    // Usage/model/path error
}
```

### PHP Helper Template
- File: `tools/php/alpr_cli_template.php`
- Fungsi utama: `run_alpr($imagePath, $textOnly = false, $envOverrides = [])`
  - Default mode keeps a warm `e2e-json-stream` process alive so subsequent calls avoid model reload latency (~150 ms steady once hot).
  - Set `USE_STREAM=0` (or export `ALPR_PHP_USE_STREAM=0`) when you need wrapper-specific features such as `ANNOTATE_DIR` or when debugging with single-shot runs.
  - Mengembalikan array dengan kunci `ok`, `exit_code`, `stdout`, `stderr`, dan `data` (jika sukses).
  - Sudah men-setup `PYTHONPATH`, `DET_ENGINE`, `OCR_ONNX`, dan `PLATE_CONFIG` sesuai struktur repo; override via `$envOverrides` jika path berbeda.
- Text-only helpers mirror the shell wrapper semantics (`TEXT_MODE`, `TEXT_ALLOW_INVALID`, `TEXT_NO_PLATE`, `TEXT_OUT_FILE`, `TEXT_RC_FILE`) but are now computed inside PHP so JSON output tetap tersedia untuk logging.
- Termasuk contoh handler upload (`$_FILES['image']`) yang merespon JSON langsung ke client.
- Untuk uji lokal CLI: `php tools/php/alpr_cli_template.php` (tambah logika Anda sendiri) atau panggil fungsi dari aplikasi Anda.

## 7. Next Steps
- Integrate redis/SQLite to store processed hashes if the PHP process restarts frequently.
- Transition from filesystem handoff to HTTP/WebSocket once the upstream program can POST directly.
- When DeepStream emits crops via IPC (Week 3 milestone), push events into the API without PHP polling.

For questions or updates, sync with plan.md §5 (Week 2 Day 12–13) and progress logs.
