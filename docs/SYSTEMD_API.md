# Systemd Setup: ALPR API (Jetson)

This sets up the warm ALPR API as a managed service so models load once and
stay hot across requests (and reboots).

## Files in this repo

- Unit: `deploy/systemd/alpr-api.service`
- Env example: `deploy/systemd/alpr-api.env.example`

---

## Quick Copy‑Paste Config (Jetson example)

If your Jetson user is `iks-ai2` and the repo is at `/home/iks-ai2/Development/ALPR_Jetson`, you can use these as direct templates.

### `/etc/systemd/system/alpr-api.service`

```ini
[Unit]
Description=ALPR API Server (Jetson warm service)
After=network-online.target
Wants=network-online.target

[Service]
User=iks-ai2
WorkingDirectory=/home/iks-ai2/Development/ALPR_Jetson

# Load Python path + ALPR_* config from env file
EnvironmentFile=/etc/alpr/alpr-api.env

# Use the venv Python directly (absolute path)
ExecStart=/home/iks-ai2/Development/ALPR_Jetson/venv/bin/python -m uvicorn api_server.server:create_app --factory --app-dir src --host 0.0.0.0 --port 8080 --workers 1

Restart=always
RestartSec=2

# Hardening (safe defaults; adjust later if needed)
NoNewPrivileges=yes
ProtectSystem=full
ProtectHome=false
PrivateTmp=yes
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
```

### `/etc/alpr/alpr-api.env`

```ini
# Python interpreter (Jetson venv)
PYTHON=/home/iks-ai2/Development/ALPR_Jetson/venv/bin/python

# Detector + OCR paths
ALPR_DET_ENGINE=/home/iks-ai2/Development/ALPR_Jetson/models/detector/yolov9-s_plate_fp16.engine
ALPR_OCR_ONNX=/home/iks-ai2/Development/ALPR_Jetson/models/ocr/cct_s_v1_global.onnx
ALPR_PLATE_CONFIG=/home/iks-ai2/Development/ALPR_Jetson/models/ocr/cct_s_v1_global_plate_config.yaml

# ONNX runtime config
ALPR_ONNX_PROVIDER=cuda     # set to "cpu" to test CPU OCR
ALPR_ONNX_GPU_MEM_MB=768

# Persistence paths
ALPR_SNAPSHOTS_DIR=/home/iks-ai2/Development/ALPR_Jetson/export/snapshots
ALPR_EVENTS_DB=/home/iks-ai2/Development/ALPR_Jetson/export/events.sqlite

# Plate + postprocess defaults
ALPR_ALLOWED_PREFIXES=A,B,D,F,E,Z,T
# Optional: ALPR_POSTPROC_CONFIG=/home/iks-ai2/Development/ALPR_Jetson/configs/ocr/postproc_indonesia.yaml

# API auth (optional but recommended when using tools/php/tes.php)
ALPR_API_TOKEN=changeme-secret-token

# Optional basic tuning
ALPR_DEFAULT_CAMERA_ID=rfid-gate
ALPR_MIN_CONF=0.5
```

If your username or repo path is different, change `User=…`, `WorkingDirectory=…`, `PYTHON=…`, and the `ALPR_*` paths accordingly.

---

## Install on Jetson

1) Create install paths and copy files

```
sudo mkdir -p /etc/alpr
sudo cp deploy/systemd/alpr-api.service /etc/systemd/system/alpr-api.service
sudo cp deploy/systemd/alpr-api.env.example /etc/alpr/alpr-api.env
```

2) Edit environment

```
sudo nano /etc/alpr/alpr-api.env
```

Set absolute paths for the detector engine plus the ONNX OCR pair:

```
ALPR_DET_ENGINE=/home/iks-ai/Development/ALPR_Jetson/models/detector/yolov9-s_plate_fp16.engine
ALPR_OCR_ONNX=/home/iks-ai/Development/ALPR_Jetson/models/ocr/cct_s_v1_global.onnx
ALPR_PLATE_CONFIG=/home/iks-ai/Development/ALPR_Jetson/models/ocr/cct_s_v1_global_plate_config.yaml
ALPR_ONNX_PROVIDER=cuda
```

You can also set `ALPR_POSTPROC_CONFIG` if you want to override post-processing costs, or uncomment the TensorRT block only if you primarily use a `.engine` OCR.

3) Point WorkingDirectory and ensure Python

The unit defaults to `WorkingDirectory=/home/iks-ai/Development/ALPR_Jetson` (user `iks-ai`).

Set the Python interpreter via `/etc/alpr/alpr-api.env` (the service reads it with `EnvironmentFile`):

```
# If your venv is named venv/ (Jetson case):
PYTHON=/home/iks-ai/Development/ALPR_Jetson/venv/bin/python
# If your venv is named .venv/:
# PYTHON=/home/iks-ai/Development/ALPR_Jetson/.venv/bin/python
```

The service ExecStart uses that variable and the correct import path with `--app-dir src`:

```
ExecStart=${PYTHON} -m uvicorn api_server.server:create_app --factory --app-dir src --host 0.0.0.0 --port 8080 --workers 1
```

Install deps once into that venv:

```
cd /home/iks-ai/Development/ALPR_Jetson
python3 -m venv venv   # or .venv
source venv/bin/activate   # or .venv/bin/activate
pip install -e . -c constraints-jetson.txt -r requirements-jetson.txt
deactivate
```

4) Reload and start

```
sudo systemctl daemon-reload
sudo systemctl enable alpr-api
sudo systemctl restart alpr-api
```

5) Check status and logs

```
systemctl status alpr-api
journalctl -u alpr-api -f
```

6) Test the endpoint

```
curl -F "image=@/path/to/image.jpg" http://127.0.0.1:8080/v1/alpr
```

## Notes

- Keep `--workers 1` to avoid loading models multiple times.
- Use the PHP bridge `ALPR_API_URL=http://<jetson-ip>:8080` so RFID taps hit the warm API.
- To stop/disable:

```
sudo systemctl stop alpr-api
sudo systemctl disable alpr-api
```

 Hardening notes
- The unit template sets `ProtectHome=false` because it runs from `/home/...`. If you move the repo to `/opt/alpr`, you can set `ProtectHome=true`.
- `ProtectSystem=full` makes `/usr` and `/etc` read-only during runtime; keep your write paths under the repo (e.g., `export/`) or `/var`.

Troubleshooting
- If startup fails with `ModuleNotFoundError: uvicorn` or `fastapi`, verify `PYTHON` points to your venv and that the venv has those packages.
- If you see import errors for `src.api_server.server`, ensure the ExecStart matches the template: use `api_server.server:create_app` with `--app-dir src`.
- If ONNX CUDA provider is unavailable on Jetson, set `ALPR_ONNX_PROVIDER=cpu` in `/etc/alpr/alpr-api.env` to get the service up, then optimize later.
