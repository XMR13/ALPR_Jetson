# RTSP Soak Test Checklist

Purpose: validate long-running stability (1–2 hours) of the DeepStream → OCR pipeline on Jetson Xavier NX once IPC wiring is active.

## Prerequisites
- Jetson NX flashed with JetPack 5.1.5, CUDA 11.4, TensorRT 8.5.2.
- Models deployed under `/opt/alpr/models` (adjust paths as required).
- DeepStream app built with `maybe_send_crop_over_ipc()` integration.
- OCR service running with `ALPR_OCR_IPC_ENABLED=1`.
- Persistent storage for event logs/snapshots mounted (e.g., `/var/alpr/export`).

## Pre-Flight
1. Enable performance mode:
   ```bash
   sudo nvpmodel -m 0
   sudo jetson_clocks
   ```
2. Export IPC environment variables:
   ```bash
   export ALPR_DS_IPC_ENABLED=1
   export ALPR_DS_IPC_ENDPOINT=ipc:///tmp/alpr.ds2ocr.sock
   export ALPR_DS_IPC_MIN_PLATE_H=28
   export ALPR_DS_IPC_PRIORITY_ONLY=0
   export ALPR_DS_IPC_LOG=0
   export ALPR_DS_IPC_LOG_SKIPS=0
   ```
3. Start OCR service:
   ```bash
   ALPR_OCR_IPC_ENABLED=1 \
   ALPR_OCR_IPC_ENDPOINT=ipc:///tmp/alpr.ds2ocr.sock \
   uvicorn ocr_service.app:create_app --factory --host 0.0.0.0 --port 8081
   ```
4. Verify metrics endpoint (if exposed) or tail logs for `"OCR IPC"` entries.

## DeepStream Launch
```bash
./alpr-deepstream --config configs/deepstream/app_config.txt \
  --metadata-log /var/alpr/logs/ds_probe_metrics.log
```

Ensure the pad probe prints a startup banner (optional) with gating parameters:
```
[ds-ipc] gating: min_plate_h=28 priority_only=0 enabled=1
```

## Monitoring During Soak
- Every 5 minutes, sample `probe_counters()` and `ipc_stats()` (log or expose via `/metrics`):
  - `attempted`, `skipped_*`, `ipc_sent`, `ipc_send_fail`
  - `sent`, `send_fail`, `hwm_drop`, `encode_fail`
- Track GPU/CPU usage using `tegrastats` (log to `/var/alpr/logs/tegrastats.log`).
- Capture 10 sample snapshots/hour for visual inspection (`export/snapshots/`).
- Note any RTSP reconnects or probe skip reasons in `progress/` session log.

## Post-Run
1. Stop DeepStream and OCR services gracefully.
2. Gather artifacts:
   - `/var/alpr/logs/ds_probe_metrics.log`
   - `/var/alpr/logs/tegrastats.log`
   - `export/events.sqlite` (or NDJSON export)
   - Snapshot sample
3. Run CLI evaluation on captured frames (if available) via `docs/SMOKE_GUIDE.md`.
4. Summarize key metrics in `progress/<date>_session-*.md`:
   - Total crops attempted vs sent vs dropped.
   - Average OCR latency (from OCR service logs).
   - Number of RTSP reconnects (if any).
5. Update `plan.md` status (Week 3 — RTSP soak milestone).

## Contingencies
- **HWM drops increasing**: raise `ALPR_DS_IPC_SNDHWM` or adjust `ALPR_DS_IPC_PRIORITY_ONLY=1` for high-traffic scenarios.
- **OCR backlog**: raise OCR worker threads/batch size; check GPU utilization.
- **RTSP reconnect storm**: verify network stability; consider increasing DeepStream reconnection backoff.
- **Socket errors**: ensure `/tmp/alpr.ds2ocr.sock` is cleaned on restart; confirm permissions (`chmod 660`).

