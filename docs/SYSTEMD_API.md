# Systemd Setup: ALPR API (Jetson)

This sets up the warm ALPR API as a managed service so models load once and
stay hot across requests (and reboots).

## Files in this repo

- Unit: `deploy/systemd/alpr-api.service`
- Env example: `deploy/systemd/alpr-api.env.example`

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

The unit now defaults to `WorkingDirectory=/home/iks-ai/Development/ALPR_Jetson` (user `iks-ai`) and runs:

```
/usr/bin/python3 -m uvicorn src.api_server.server:create_app --factory --host 0.0.0.0 --port 8080 --workers 1
```

Make sure the project lives at `/home/iks-ai/Development/ALPR_Jetson` (or update the unit) and `uvicorn`
is available for `/usr/bin/python3` (e.g., installed via the project deps).

4) Reload and start

```
sudo systemctl daemon-reload
sudo systemctl enable alpr-api
sudo systemctl start alpr-api
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
