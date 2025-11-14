# Fast-Plate-OCR Fine-Tune Notes

This repo now contains everything needed to fine-tune the upstream
[`fast-plate-ocr`](https://github.com/ankandrew/fast-plate-ocr) **cct-s** model on
Indonesian plate crops exported from `data/images_test`.

## 1. Dataset snapshots

```
data/
└── fast_plate_ocr/
    ├── train/
    │   ├── annotations.csv      # 2,414 samples
    │   └── images/*.jpg         # plate crops (copied from images_test_det)
    └── val/
        ├── annotations.csv      #   426 samples (15% split)
        └── images/*.jpg
```

Generated with:

```bash
python3 tools/gen_fast_plate_ocr_dataset.py \
  --crops-dir data/images_test_det \
  --out-csv data/fast_plate_ocr/train/annotations.csv \
  --val-csv data/fast_plate_ocr/val/annotations.csv \
  --val-ratio 0.15 \
  --seed 42 \
  --copy-images            # also copies crops into train/ and val/
```

- The script parses the plate text from the filename segment immediately following
  `CCTV<channel>_`. Filenames with an empty segment (unregistered vehicles) are
  skipped automatically.
- The `--copy-images` flag places each crop under `train/images/` or `val/images/`
  and the CSV `image_path` points at those relative paths, so the entire
  `data/fast_plate_ocr` tree is self-contained and can be zipped/moved to the
  machine where you run fast-plate-ocr training.

## 2. Plate & model configs

```
configs/fast_plate_ocr/
├── plate_indonesia.yaml        # grayscale, 48×192, max 9 chars
├── cct-s-indonesia.yaml        # cct-s backbone with extra dropout
└── augment_indonesia.yaml      # Albumentations pipeline tuned for glare/night shots
```

Key choices:

- **Plate config** mirrors Jetson preprocessing: grayscale, CLAHE-friendly, 48 px height
  for efficiency, 192 px width to keep ~4:1 aspect ratio, and alphabet restricted to
  `0-9A-Z` plus `_` for padding (max 9 symbols).
- **Model config** copies upstream `cct_s_v1` and slightly increases `mlp_dropout`
  and `head_mlp_dropout` to ward off overfitting on the ~2.8k-sample dataset.
- **Augmentations** stay channel-agnostic (they work on grayscale), inject brightness
  jitter, gamma swings, additive noise, blur/motion, coarse dropout (both dark and
  bright streaks to mimic headlights), geometric warps, downscale, and JPEG artifacts.

## 3. Training command example

Inside your `fast-plate-ocr` checkout (TensorFlow backend shown; replace with JAX/PyTorch
if preferred):

```bash
KERAS_BACKEND=tensorflow fast-plate-ocr train \
  --model-config-file /path/to/ALPR_Jetson/configs/fast_plate_ocr/cct-s-indonesia.yaml \
  --plate-config-file /path/to/ALPR_Jetson/configs/fast_plate_ocr/plate_indonesia.yaml \
  --annotations /path/to/ALPR_Jetson/data/fast_plate_ocr/train/annotations.csv \
  --val-annotations /path/to/ALPR_Jetson/data/fast_plate_ocr/val/annotations.csv \
  --augmentation-path /path/to/ALPR_Jetson/configs/fast_plate_ocr/augment_indonesia.yaml \
  --batch-size 32 \
  --epochs 30 \
  --learning-rate 3e-4 \
  --output-dir runs/cct-s-indonesia
```

Suggested monitoring:

- Target ≥95% exact-match on the validation split before exporting ONNX/TRT.
- Track CER and overfitting; with dropout + augmentations you should plateau around
  20–25 epochs. Enable early stopping once validation CER stalls.
- Keep an eye on the CSV for any lingering noisy labels (single-letter “N” samples
  etc.); cleanups here usually pay larger dividends than extra epochs.

## 4. Next steps

1. Review/spot-check `data/fast_plate_ocr/*/annotations.csv` to ensure plate text
   parsing matches expectations.
2. Copy or symlink `data/images_test_det` into your fast-plate-ocr workspace (or
   adjust the CSV paths) before launching training.
3. After training, export ONNX/TRT with `fast-plate-ocr export` and wire it into the
   Jetson OCR service (documented in `src/ocr_service/`), keeping preprocessing in sync
   with `plate_indonesia.yaml`.
