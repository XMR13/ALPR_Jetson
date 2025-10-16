# ALPR Jetson — YOLOv9 Detector

Repo ini menggunakan YOLOv9 (TensorRT) sebagai detektor plat utama.

Penempatan artefak model:
- ONNX: `models/detector/yolov9-s.onnx`
- TensorRT FP16: `models/detector/yolov9-s_fp16.engine` (opsional INT8: `yolov9-s_int8.engine` + `int8_calib.cache`)
- Skrip inferensi TensorRT (opsi): jadikan modul reusable di `src/inference/yolov9_trt.py` dan CLI tipis di `tools/infer_yolov9_trt.py`.

DeepStream:
- Konfigurasi default mengarah ke `configs/deepstream/config_infer_primary_yolov9.txt`. Jalankan smoke test:
  - `python -m alpr_jetson ds-smoke --config configs/deepstream/app_config.txt`
  - atau `bash tools/deepstream_smoke.sh configs/deepstream/app_config.txt`

Pelatihan & ekspor YOLOv9 dilakukan di lingkungan eksternal Anda. Lihat `docs/TRAINING_NOTES.md` untuk catatan ekspor ONNX → TensorRT.

Detail rencana dan tonggak: `plan.md`.
