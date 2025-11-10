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

Set absolute paths to your detector/OCR engines and charset. Optionally set
`ALPR_POSTPROC_CONFIG` to a YAML if you want to override postproc tuning.

3) Point WorkingDirectory and ensure Python

The unit uses `WorkingDirectory=/opt/alpr` and runs:

```
/usr/bin/python3 -m uvicorn src.api_server.server:create_app --factory --host 0.0.0.0 --port 8080 --workers 1
```

Make sure the project is at `/opt/alpr` (or update the unit) and `uvicorn`
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

