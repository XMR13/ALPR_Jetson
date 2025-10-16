# Training Notes — YOLOv9 Plate Detector

These notes capture best practices and concrete steps to maximize accuracy when training or fine-tuning YOLOv9 for Indonesian plate detection on your dataset.

## Environment
- Create a fresh venv; install `torch` (CUDA build for your GPU) and your chosen YOLOv9 implementation.
- Editable install of this repo: `pip install -e .` so `alpr_jetson` package resolves.
- Optional but recommended: TensorBoard.

## Data Due Diligence
- Validate COCO exports with `tools/coco_from_cvat.py` into a stable path.
- Run stats to understand plate sizes and adjust input/augs:
  `python tools/dataset_stats.py --coco data/processed/cam01/train/coco.json`
- Ensure train/val image roots match `file_name` entries in COCO.
- Remove or relabel noisy boxes (extreme aspect ratios, < 10 px height) before training.

## Recommended Hyperparameters (baseline)
- Input size: 640 for baseline (Jetson-friendly). Consider 736–800 if many plates are very small (p50 height < 40 px).
- Schedule: 80–100 epochs total with final 10–15 epochs no-augmentation.
- Augmentations: enable mosaic/mixup judiciously; hsv on; flip 0.5; multi-scale.
- Loss/heads: follow your YOLOv9 repo defaults; enable any final fine-tuning stage for localization.
- Batch: start with 16 per GPU (adjust to VRAM and implementation).
- Precision: FP16 mixed precision on capable GPUs.
- Seed: set a fixed seed for reproducibility (e.g., 42).

## Pretrained Weights
- Start from COCO pretrained weights for your YOLOv9 variant when available.

## Training Command
Train in your external YOLOv9 environment according to that repository’s instructions. Ensure your dataset is consistent (COCO-style recommended) and that validation sets cover day/night.

## Evaluation
- During training, ensure periodic evaluation on validation (per epoch or schedule). After training, export predictions and compute COCO AP:
  `python tools/eval_det_coco.py --gt <val_coco.json> --pred <predictions.json>`

## Export to TensorRT
- Export ONNX and build FP16 engine with `tools/trtexec_build.sh` (fill actual `trtexec` args per environment).
- Place the artifacts here:
  - `models/detector/yolov9-s.onnx`
  - `models/detector/yolov9-s_fp16.engine`

## Tips
- If val AP plate is low and many boxes < 28 px, either increase input size or add a vehicle detector to pre-crop.
- Keep day/night balanced per epoch or use class-balanced sampling if distributions diverge across days.
- Re-run stats after pruning noisy labels; small label errors have outsized impact on small objects.
