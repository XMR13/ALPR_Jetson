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
