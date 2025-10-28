# Training with YOLOv9 (WongKinYiu)

This guide shows how to train the plate detector using the YOLOv9 repository
https://github.com/WongKinYiu/yolov9 and integrate the resulting model into
this project (ONNX → TensorRT → DeepStream).

## 1) Prepare the dataset

- COCO → YOLO labels:
  `python tools/coco_to_yolo.py --coco data/processed/cam01/train/coco.json --outdir /mnt/datasets/plates/labels/train`
  `python tools/coco_to_yolo.py --coco data/processed/cam01/val/coco.json   --outdir /mnt/datasets/plates/labels/val`

- Verify YOLO structure and alignment:
  `python tools/verify_yolo_dataset.py --root /mnt/datasets/plates`

- Generate YOLOv9 dataset YAML:
  `python tools/gen_yolov9_data_yaml.py --root /mnt/datasets/plates --names plate --out configs/training/plates_yolov9.yaml`

- Dataset stats (helps choose image size):
  - COCO: `python tools/dataset_stats.py --coco data/processed/cam01/train/coco.json`
  - YOLO: `python tools/dataset_stats.py --yolo-root /mnt/datasets/plates`

## 2) Train in YOLOv9 repo

> Run these steps in your separate YOLOv9 environment (not inside this repo).

```bash
git clone https://github.com/WongKinYiu/yolov9
cd yolov9
pip install -r requirements.txt

# Example with the small model and 640px input
python train.py \
  --data /absolute/path/to/configs/training/plates_yolov9.yaml \
  --cfg models/detect/yolov9-s.yaml \
  --epochs 100 --img 640 --batch 16 --device 0 \
  --project runs/plates --name yolov9s-plates

# For small plates (p50 < 40 px), consider:
#   --img 736 or 800, enable mosaic/mixup (default), and use --rect
```

Tips for accuracy
- If many boxes are < 28 px high (see dataset_stats), prefer `--img 736..800`.
- Use `--rect` (rectangular training) to reduce letterbox distortions.
- Ensure day/night balance in each epoch (shuffle or weighted sampler).
- Consider a second fine-tuning phase with reduced augmentations for the last 10–15 epochs.

## 3) Export to ONNX and build TensorRT

```bash
# Still in YOLOv9 repo
python export.py --weights runs/plates/yolov9s-plates/weights/best.pt --include onnx --simplify
# Copy ONNX into this repo under models/detector/
cp runs/plates/yolov9s-plates/weights/best.onnx /path/to/this/repo/models/detector/yolov9_plate.onnx

# Build FP16 TensorRT engine (on Jetson or compatible host with TRT 8.5.2)
cd /path/to/this/repo
bash tools/trtexec_build.sh models/detector/yolov9_plate.onnx models/detector/yolov9_plate_fp16.engine --fp16
```

## 4) Validate and integrate

- COCO eval on validation set:
  `python tools/predict_yolov9_coco.py --engine models/detector/yolov9_plate_fp16.engine --coco data/processed/cam01/val/coco.json --output export/eval/yolov9_val_preds.json`
  `python tools/eval_det_coco.py --gt data/processed/cam01/val/coco.json --pred export/eval/yolov9_val_preds.json`

- Wire the engine into the app using env:
  `export ALPR_DET_ENGINE=models/detector/yolov9_plate_fp16.engine`

## 5) Recommendations specific to this project

- Set `pre-cluster-threshold` and `nms-iou-threshold` in DeepStream configs
  consistent with observed confidence distributions (start at 0.25/0.50).
- Enforce minimum plate height/aspect gating before OCR (avoid tiny crops).
- Pair with OCR improvements (rectification, CLAHE, temporal voting) for
  end-to-end gains beyond detector AP.

