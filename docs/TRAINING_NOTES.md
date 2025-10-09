# Training Notes — YOLOX Plate Detector

These notes capture best practices and concrete steps to maximize accuracy when fine-tuning YOLOX for Indonesian plate detection on your dataset.

## Environment
- Create a fresh venv; install `torch` (CUDA build for your GPU) and `yolox`.
- Editable install of this repo: `pip install -e .` so `alpr_jetson` package resolves.
- Optional but recommended: TensorBoard.

## Data Due Diligence
- Validate COCO exports with `tools/coco_from_cvat.py` into a stable path.
- Run stats to understand plate sizes and adjust input/augs:
  `python tools/dataset_stats.py --coco data/processed/cam01/train/coco.json`
- Ensure train/val image roots match `file_name` entries in COCO.
- Remove or relabel noisy boxes (extreme aspect ratios, < 10 px height) before training.

## Recommended Hyperparameters
- Input size: 640 for baseline (Jetson-friendly). Consider 736–800 if many plates are very small (stats p50 < 40px height).
- Schedule: 80 epochs total with final 15 epochs no-augmentation.
- Augmentations: mosaic=0.7, mixup=0.05, hsv=on, flip=0.5, multi-scale (14–26).
- Loss: enable L1 during no-aug phase for better localization.
- Batch: start with 16 per GPU (adjust to your VRAM).
- Precision: FP16 mixed precision on capable GPUs.
- Seed: set a fixed seed for reproducibility (e.g., 42).

## Pretrained Weights
- Start from COCO pretrained `YOLOX_S.pth` for best convergence.
- Pass via `--ckpt` in our launcher or `YOLOX_PRETRAIN` env var.

## Commands
Example using your dataset in place:

```
python tools/train_yolox.py \
  --data-dir "D:\\RZQ\\Coding\\Datasets\\ALPR_First trial" \
  --train-ann annotations\\instances_Train.json \
  --val-ann annotations\\instances_Validation.json \
  --train-name images\\Train \
  --val-name images\\Validation \
  --batch 16 --epochs 80 --no-aug-epochs 15 --fp16 \
  --ckpt path\\to\\YOLOX_S.pth --expn yolox_s_lp --seed 42 --cache
```

Notes:
- Outputs land under `YOLOX_OUTPUTS/yolox_s_lp/` (YOLOX default) or `outputs/` depending on YOLOX version.
- The experiment file is `exps/yolox/exp_plate_yolox_s.py` and reads env overrides set by our launcher.

## Evaluation
- During training, YOLOX evaluates each epoch. After training, export predictions and compute COCO AP:
  `python tools/eval_det_coco.py --gt <val_coco.json> --pred <predictions.json>`

## Export to TensorRT
- Export ONNX and build FP16 engine with `tools/trtexec_build.sh` (fill in actual `trtexec` command per your environment).

## Tips
- If val AP plate is low and many boxes < 28 px, either increase input size or add a vehicle detector to pre-crop.
- Keep day/night balanced per epoch or use class-balanced sampling if distributions diverge across days.
- Re-run stats after pruning noisy labels; small label errors have outsized impact on small objects.

