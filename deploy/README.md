Jetson Deployment (Compose)

This guide covers running the ALPR services on a Jetson Xavier NX (JetPack 5.1.5, Python 3.8) using NVIDIA L4T/DeepStream base images. It mirrors plan.md §5 (Week 2/4 ops tasks).

Prerequisites
- Jetson NX on JetPack 5.1.5 (r35.5.x), NVIDIA Container Runtime installed.
- Images pre-pulled on the device:
  - `sudo docker pull nvcr.io/nvidia/l4t-ml:r35.5.0-py3`
  - `sudo docker pull nvcr.io/nvidia/deepstream:6.4-triton-multiarch`

Layout
- Compose file: `deploy/compose.jetson.yml` mounts the repo into `/workspace` and starts services idle (`sleep infinity`).
- You install Python deps inside each container, then run the service command you want (API or OCR microservice) using the mounted code.

Start Compose
```bash
cd /path/to/repo
docker compose -f deploy/compose.jetson.yml up -d
```

Install Dependencies (once per container)
- API service shell:
  ```bash
  docker compose -f deploy/compose.jetson.yml exec alpr-api bash
  python3 -m pip install --upgrade pip
  python3 -m pip install -r requirements-jetson.txt -c constraints-jetson.txt
  exit
  ```
- OCR service shell:
  ```bash
  docker compose -f deploy/compose.jetson.yml exec alpr-ocr bash
  python3 -m pip install --upgrade pip
  python3 -m pip install -r requirements-jetson.txt -c constraints-jetson.txt
  exit
  ```

Run Services
- API (FastAPI via uvicorn factory):
  ```bash
  docker compose -f deploy/compose.jetson.yml exec alpr-api bash -lc \
    "uvicorn --host 0.0.0.0 --port 8000 --factory src.api_server.server:create_app"
  # Health check from host (adjust IP if remote):
  curl -s http://localhost:8000/healthz
  ```
- OCR microservice (example placeholder):
  ```bash
  docker compose -f deploy/compose.jetson.yml exec alpr-ocr bash -lc \
    "python3 -m alpr_jetson ocr-infer --help"
  ```

Notes
- The codebase is mounted read/write at `/workspace`; changes on host reflect live in containers.
- Prefer system OpenCV on Jetson; do not install `opencv-python` wheels.
- If enabling ONNXRuntime CUDA EP for OCR, ensure GPU memory limits are tuned (see README ‘ONNX OCR — Mode Memori’).

DeepStream Service
- `alpr-deepstream` is a placeholder in compose; the DS app and configs live under `src/deepstream_app/` and `configs/deepstream/`.
- Integrate the binary and runtime later per plan §5 (Week 2→4). For now, keep it idle or replace with your own entrypoint.

Shutdown
```bash
docker compose -f deploy/compose.jetson.yml down
```

