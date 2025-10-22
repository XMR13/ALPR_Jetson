# ALPR Jetson — YOLOv9 Detector

Repo ini menggunakan YOLOv9 (TensorRT) sebagai detektor plat utama.

Penempatan artefak model:
- ONNX: `models/detector/yolov9-s_plate.onnx`
- TensorRT FP16: `models/detector/yolov9-s_plate_fp16.engine`
- Skrip inferensi TensorRT: modul reusable `src/inference/yolov9_trt.py` dengan CLI `tools/infer_yolov9_trt.py`.

Jalankan inferensi lokal dari snapshot gambar:

```bash
python tools/infer_yolov9_trt.py \
  --engine models/detector/yolov9-s_plate_fp16.engine \
  --source data/raw/cam01/frame.jpg \
  --conf 0.4
```

Evaluasi detektor (COCO AP):

```bash
python tools/predict_yolov9_coco.py \
  --engine models/detector/yolov9-s_plate_fp16.engine \
  --coco data/processed/cam01/val/coco.json \
  --images-root /path/to/val/images \
  --output export/eval/yolov9_val_preds.json \
  --conf 0.4

python tools/eval_det_coco.py \
  --gt data/processed/cam01/val/coco.json \
  --pred export/eval/yolov9_val_preds.json
```

DeepStream:
- Konfigurasi default mengarah ke `configs/deepstream/config_infer_primary_yolov9.txt`. Jalankan smoke test:
  - `python -m alpr_jetson ds-smoke --config configs/deepstream/app_config.txt`
  - atau `bash tools/deepstream_smoke.sh configs/deepstream/app_config.txt`

Pelatihan & ekspor YOLOv9 dilakukan di lingkungan eksternal Anda. Lihat `docs/TRAINING_NOTES.md` untuk catatan ekspor ONNX → TensorRT.

Detail rencana dan tonggak: `plan.md`.

OCR (CRNN/Paddle-style) — Preproc & Service
- Preprocessing module: `src/ocr_service/preprocess.py` (grayscale + CLAHE + normalize to 32x160 by default).
- Post-processing: `src/ocr_service/postprocess.py` (regex-validasi, perbaikan karakter ambigu, majority vote).
- OCR runtime:
  - TensorRT CTC: `src/ocr_service/trt_infer.py`
  - ONNX slot-based (misal CCT-S): `src/ocr_service/onnx_infer.py` + konfigurasi YAML (`models/ocr/cct_s_v1_global_plate_config.yaml`).
- FastAPI microservice stub: `src/ocr_service/app.py` (opsional; butuh FastAPI terpasang).
- Pelatihan & ekspor OCR: lihat `docs/OCR_MODEL.md` untuk panduan PaddleOCR → ONNX → TensorRT, termasuk target akurasi dan layout model.
- Evaluasi akurasi OCR: `tools/eval_ocr.py --engine ... --charset ... --crops ... --labels ...` menghitung exact-match dan CER.

Contoh pakai OCR (lokal, jika FastAPI terpasang):
```bash
python -c "from ocr_service.app import create_app; print(bool(create_app()))"  # True jika FastAPI siap
```

Atau langsung gunakan wrapper (non-service) di Python:
```python
from ocr_service.trt_infer import OCRService
from ocr_service.preprocess import PreprocConfig
import cv2

svc = OCRService(engine_path="models/ocr/ppo_crnn_fp16.engine",
                 charset_path="models/ocr/charset.txt",
                 preproc=PreprocConfig(input_width=160, input_height=32))
img = cv2.imread("data/labeled/ocr_crops/example.jpg")
print(svc.infer_batch([img]))
```

CLI cepat untuk uji OCR (pilih backend):

```bash
# TensorRT (CTC)
python -m alpr_jetson ocr-infer \
  --engine models/ocr/ppo_crnn_fp16.engine \
  --charset models/ocr/charset.txt \
  --source data/labeled/ocr_crops/

# ONNX (slot-based)
python -m alpr_jetson ocr-infer \
  --onnx models/ocr/cct_s_v1_global.onnx \
  --plate-config models/ocr/cct_s_v1_global_plate_config.yaml \
  --source data/labeled/ocr_crops/
```

Catatan lebih rinci untuk ONNX → TensorRT OCR ada di `docs/OCR_NOTES.md`.

End-to-end detector + OCR pada folder gambar:

```bash
python -m alpr_jetson e2e \
  --det-engine models/detector/yolov9-s_plate_fp16.engine \
  --onnx models/ocr/cct_s_v1_global.onnx \
  --plate-config models/ocr/cct_s_v1_global_plate_config.yaml \
  --source data/raw/cam01/frames \
  --annotate-dir export/eval/e2e_vis
```

ONNX OCR — Mode Memori (Jetson NX)
- Maksimum performa (tanpa batas GPU):
  - Tambahkan `--onnx-provider cuda --onnx-gpu-mem-limit-mb 0` (0 = tidak ada batasan)
  - Contoh (E2E):
    - `python -m alpr_jetson e2e --det-engine ... --onnx ... --plate-config ... --onnx-provider cuda --onnx-gpu-mem-limit-mb 0 --source ...`
- Hemat memori (disarankan ketika terjadi OOM):
  - Batasi alokasi CUDA EP, mis. 512–768 MB: `--onnx-provider cuda --onnx-gpu-mem-limit-mb 512`
  - Atau pakai CPU: `--onnx-provider cpu` (lebih aman, lebih lambat)
  - Kurangi jumlah deteksi agar OCR memproses lebih sedikit crop: tingkatkan `--conf` (mis. `--conf 0.6`)
  - Hilangkan anotasi (hapus `--annotate-dir`) untuk mengurangi beban memori/I/O



API (FastAPI) — Stub Endpoints
- Server skeleton: `src/api_server/server.py` (import-safe tanpa FastAPI). Saat FastAPI tersedia, exposes:
  - `GET /healthz`, `GET /metrics`
  - `GET /v1/stream/info`, `POST /v1/hooks`, `GET /v1/events`, `WS /v1/ws`
  - Kontrak mengikuti `plan.md §9`.


Temporal Voting per Track
- Modul agregasi per-track: `src/pipeline/track_aggregator.py` — melakukan majority voting dan emisi event stabil sesuai skema di `plan.md §9`.
- Lihat uji: `tests/pipeline/test_track_aggregator.py`.
