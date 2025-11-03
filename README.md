# ALPR Jetson — YOLOv9 Detector

Repo ini menggunakan YOLOv9 (TensorRT) sebagai detektor plat utama dan menyediakan
CLI terpadu `python -m alpr_jetson` (atau `uv run python -m alpr_jetson`) untuk
uji cepat:

- **Detector-only** (`det-infer`): jalankan YOLOv9 TRT tanpa OCR, dengan opsi anotasi keluaran dan ekspor crop plat.
- **OCR-only** (`ocr-infer`): gunakan TensorRT **atau** ONNX slot-based OCR pada folder/gambar.
- **End-to-end** (`e2e`): detektor + OCR sekaligus dengan opsi teks-only / simpan anotasi.
- **End-to-end JSON** (`e2e-json`): jalankan detektor + OCR untuk satu gambar dan cetak JSON ke stdout (untuk integrasi PHP sementara).
- **Smoke tests**: `rtsp-smoke` (GStreamer) dan `ds-smoke` (DeepStream).

Contoh cepat (gunakan `PYTHONPATH=src` bila belum `pip install -e .`):

```bash
# Detector only, simpan anotasi + crop plat
python -m alpr_jetson det-infer \
  --det-engine models/detector/yolov9-s_plate_fp16.engine \
  --source data/raw/cam01/frame.jpg \
  --annotate-dir export/det_vis \
  --crop-dir export/det_crops \
  --conf 0.4

# OCR saja (TensorRT CTC)
python -m alpr_jetson ocr-infer \
  --engine models/ocr/ppo_crnn_fp16.engine \
  --charset models/ocr/charset.txt \
  --source data/labeled/ocr_crops/

# OCR slot-based ONNX (gunakan YAML PlateConfig)
python -m alpr_jetson ocr-infer \
  --onnx models/ocr/cct_s_v1_global.onnx \
  --plate-config models/ocr/cct_s_v1_global_plate_config.yaml \
  --source data/labeled/ocr_crops/ \
  --onnx-provider cuda \
  --onnx-gpu-mem-limit-mb 512
```

Penempatan artefak model:
- ONNX detektor: `models/detector/yolov9-s_plate.onnx`
- TensorRT FP16 detektor: `models/detector/yolov9-s_plate_fp16.engine`
- TensorRT / ONNX OCR: `models/ocr/…`

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
Panduan evaluasi akurasi end-to-end (CER/SER, low-confidence) ada di `docs/EVALUATION.md`.

OCR (CRNN/Paddle-style) — Preproc & Service
- Preprocessing module: `src/ocr_service/preprocess.py` (grayscale + CLAHE + normalize to 32x160 by default).
- Post-processing: `src/ocr_service/postprocess.py` (regex-validasi, perbaikan karakter ambigu, majority vote).
- OCR runtime:
  - TensorRT CTC: `src/ocr_service/trt_infer.py`
  - ONNX slot-based (misal CCT-S): `src/ocr_service/onnx_infer.py` + konfigurasi YAML (`models/ocr/cct_s_v1_global_plate_config.yaml`).
    - Rekomendasi default untuk Jetson NX (menjaga bentuk huruf dan stabil di malam hari):

```yaml
# contoh PlateConfig YAML (ONNX)
max_plate_slots: 9
alphabet: " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # pastikan pad_char ada di alphabet
pad_char: " "
img_height: 32
img_width: 160
keep_aspect_ratio: true
image_color_mode: grayscale
interpolation: area
padding_color: 144

# opsional: aktifkan hanya jika perlu
use_clahe: false                 # true untuk malam/gelap
clahe_clip: 2.0
clahe_tile: 8
clahe_brightness_gate: 110.0     # jalankan CLAHE hanya jika mean<110 (0=selalu)
auto_deskew: false               # true bila kemiringan sering >12°
deskew_threshold_deg: 12.0
```
- FastAPI microservice stub: `src/ocr_service/app.py` (opsional; butuh FastAPI terpasang).
- Pelatihan & ekspor OCR: lihat `docs/OCR_MODEL.md` untuk panduan PaddleOCR → ONNX → TensorRT, termasuk target akurasi dan layout model.
- Evaluasi akurasi OCR: `tools/eval_ocr.py --engine ... --charset ... --crops ... --labels ...` menghitung exact-match dan CER.

Dataset tools (COCO ↔ YOLO, YOLOv9)
- COCO → YOLO labels: `python tools/coco_to_yolo.py --coco <coco.json> --outdir <labels_dir>`
- Generate YOLOv9 dataset YAML: `python tools/gen_yolov9_data_yaml.py --root <yolo_root> --names plate --out configs/training/plates_yolov9.yaml`
- Verify YOLO dataset alignment: `python tools/verify_yolo_dataset.py --root <yolo_root>`
- Dataset stats:
  - COCO: `python tools/dataset_stats.py --coco <coco.json>`
  - YOLO: `python tools/dataset_stats.py --yolo-root <yolo_root>`

Training with YOLOv9
- See docs/TRAIN_YOLOV9.md for end-to-end steps (train in the YOLOv9 repo, export ONNX, build TRT, evaluate, and integrate).

Jetson containers (compose.jetson.yml)
- `deploy/compose.jetson.yml` now targets JetPack 5.1.5 via `nvcr.io/nvidia/l4t-ml:r35.5.0-py3` (Python 3.8). Pull these images on the NX before running compose:
  ```bash
  sudo docker pull nvcr.io/nvidia/l4t-ml:r35.5.0-py3
  sudo docker pull nvcr.io/nvidia/deepstream:6.4-triton-multiarch
  ```
- Both `alpr-ocr` and `alpr-api` services inherit Python 3.8 from that base; runtime assumes `python3` is available in-path.

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
  --annotate-dir export/eval/e2e_vis \
  --stats \
  --stats-file export/eval/e2e_stats.txt
```
- `--stats` menampilkan ringkasan latensi/FPS (avg/p50/p95/max) ke stderr setelah semua gambar diproses. Berlaku baik untuk satu gambar maupun direktori.
- `--stats-file <path>` (opsional) menyimpan ringkasan yang sama ke file.

Output teks saja (opsional) untuk e2e:

- Tampilkan hanya teks plat (satu per baris), tanpa output lain:

```bash
python -m alpr_jetson e2e \
  --det-engine models/detector/yolov9-s_plate_fp16.engine \
  --onnx models/ocr/cct_s_v1_global.onnx \
  --plate-config models/ocr/cct_s_v1_global_plate_config.yaml \
  --source data/raw/cam01/frames \
  --text-only
```

- Simpan semua teks plat ke file (satu baris per plat), sambil tetap menyimpan anotasi jika diminta:

```bash
python -m alpr_jetson e2e \
  --det-engine models/detector/yolov9-s_plate_fp16.engine \
  --onnx models/ocr/cct_s_v1_global.onnx \
  --plate-config models/ocr/cct_s_v1_global_plate_config.yaml \
  --source data/raw/cam01/frames \
  --annotate-dir export/eval/e2e_vis \
  --text-out export/eval/e2e_texts.txt
```

End-to-end sekali jalan (JSON untuk PHP)

```bash
# Opsi 1: panggil CLI secara langsung
python -m alpr_jetson e2e-json \
  --det-engine models/detector/yolov9-s_plate_fp16.engine \
  --onnx models/ocr/cct_s_v1_global.onnx \
  --plate-config models/ocr/cct_s_v1_global_plate_config.yaml \
  --source /path/to/frame.jpg \
  --conf 0.5
# Keluaran ke stdout adalah JSON: {status, plates[], latency_ms{det,ocr,total}}
```

Wrapper sederhana (hanya butuh path gambar)

```bash
# Default backend: ONNX OCR (ubah ke TRT dengan OCR_BACKEND=trt)
tools/alpr_e2e_json.sh /path/to/frame.jpg

# ONNX (eksplisit) dan keluaran teks saja
# Catatan TEXT_ONLY:
#  - Default hanya mencetak teks yang valid (sesuai post-proc). Jika tidak valid/No plate → exit 3 tanpa output.
#  - Opsi tambahan:
#      TEXT_MODE=raw             -> cetak ocr_raw (tanpa normalisasi)
#      TEXT_ALLOW_INVALID=1      -> cetak teks meski valid=false
#      TEXT_NO_PLATE=NO_PLATE    -> saat rc=3, tetap cetak placeholder "NO_PLATE" ke stdout
#      POSTPROC=none|indonesia   -> override post-proc
#      ALLOWED_PREFIX="B D F ..." -> batasi prefix (spasi atau koma)
TEXT_ONLY=1 OCR_BACKEND=onnx TEXT_ALLOW_INVALID=1 tools/alpr_e2e_json.sh /path/to/frame.jpg

# Simpan anotasi juga (selain JSON/teks)
ANNOTATE_DIR=export/ann tools/alpr_e2e_json.sh /path/to/frame.jpg
```
- Template PHP siap pakai: lihat `tools/php/alpr_cli_template.php` (panduan di `docs/INTEGRATION_PHP.md`).
  - Helper menjaga proses `e2e-json-stream` tetap hidup antar panggilan (latency lebih rendah). Set `USE_STREAM=0` bila perlu kembali ke mode sekali jalan, misalnya saat mengaktifkan `ANNOTATE_DIR`.

Contoh PHP (minimal) memanggil CLI dan membaca JSON:

```php
$img = '/tmp/frame.jpg';
// JSON mode (default)
$cmd = 'tools/alpr_e2e_json.sh ' . escapeshellarg($img);
$json = shell_exec($cmd);
$data = json_decode($json, true);
if (!$data) { /* tangani error */ }
// $data['status'], $data['plates'][0]['text'], dst.

// TEXT_ONLY mode (disarankan untuk Webmin)
$cmd = 'TEXT_ONLY=1 tools/alpr_e2e_json.sh ' . escapeshellarg($img);
$out = [];
$rc = 1;
exec($cmd, $out, $rc);
if ($rc === 0) {
    $plate = trim(implode("\n", $out)); // teks plat saja
} else if ($rc === 3) {
    // tidak ada plat atau teks invalid
} else {
    // error penggunaan / model / path
}
```

Streaming NDJSON (satu proses, baca path dari stdin)

```bash
python -m alpr_jetson e2e-json-stream \
  --det-engine models/detector/yolov9-s_plate_fp16.engine \
  --onnx models/ocr/cct_s_v1_global.onnx \
  --plate-config models/ocr/cct_s_v1_global_plate_config.yaml \
  --conf 0.5 < paths.txt
# paths.txt berisi satu path per baris; output: JSON line per gambar (dengan kolom "input")
```
- Setiap baris JSON sekarang menyertakan `latency_ms.iter` (durasi iterasi penuh) selain det/OCR/total.
- Ringkasan throughput dan statistik latensi (avg/p50/p95/max) dicetak ke stderr setelah stream selesai, sehingga stdout tetap NDJSON murni.

Panduan lengkap (deduplikasi, retry, kualitas OCR) ada di `docs/INTEGRATION_PHP.md`.

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

ONNX OCR — Peningkatan Kualitas
- Default yang disarankan: `keep_aspect_ratio=true`, `image_color_mode=grayscale`, `interpolation=area`.
- `use_clahe=true` hanya untuk kondisi gelap/glare; gunakan `clahe_brightness_gate` agar aktif otomatis saat gelap.
- `auto_deskew=true` bila sering ada tilt besar; akan diterapkan hanya jika |angle| ≥ `deskew_threshold_deg`.



API (FastAPI) — Stub Endpoints
- Server skeleton: `src/api_server/server.py` (import-safe tanpa FastAPI). Saat FastAPI tersedia, exposes:
- `GET /healthz`, `GET /metrics`
- `GET /v1/stream/info`, `POST /v1/hooks`, `GET /v1/events`, `WS /v1/ws`
- Synchronous test endpoint `POST /v1/alpr` (multipart upload) untuk integrasi sistem yang sudah menyimpan snapshot; lihat `docs/API.md` untuk detail kontrak.
- Kontrak endpoint lainnya mengikuti `plan.md §9`.
- Panduan integrasi PHP ↔ Jetson (flow `capture → /v1/alpr → response`) ada di `docs/INTEGRATION_TESTING.md`.


Temporal Voting per Track
- Modul agregasi per-track: `src/pipeline/track_aggregator.py` — melakukan majority voting dan emisi event stabil sesuai skema di `plan.md §9`.
- Lihat uji: `tests/pipeline/test_track_aggregator.py`.
