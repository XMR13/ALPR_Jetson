# OCR Model — PaddleOCR Integration

This document describes how to train, export, and deploy a license plate OCR
model for the Jetson Xavier NX (16 GB, JetPack 5.1.5, CUDA 11.4, TensorRT 8.5).
The pipeline uses PaddleOCR recognition networks, fine-tuned on Indonesian
plates, and converts them to TensorRT engines consumable by this repository.

## 1. Model Choice

- **Backbone**: PaddleOCR "PP-OCRv4" recognition (SVTR_LCNet) with CTC head.
- **Input size**: `1×3×32×160` (RGB). The repo duplicates grayscale plates to 3
  channels to match the training configuration.
- **Charset**: 36 symbols `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ`. No lowercase,
  no punctuation. The CTC blank stays internal to the model; do not add it to
  the charset file.
- **Why this model**
  - Robust to blur, glare, and perspective compared to older CRNN-only models.
  - Uses only convolutional layers and light transformers that export cleanly
    to ONNX and TensorRT 8.5.2.
  - Meets our latency target (< 2 ms per crop at FP16 on NX) while maintaining
    high accuracy on short plates.

If the exported ONNX introduces unsupported TensorRT operators, fall back to
`rec_svtrnet` or `rec_crnn` with CTC. They share the same export steps and work
with the runtime when the input/output shapes are preserved.

## 2. Directory Layout

Place artifacts in the repository after training:

```
models/
  ocr/
    ppocr_rec_fp16.engine    # TensorRT FP16 engine (Jetson-built)
    ppocr_rec_32x160.onnx    # Optional: ONNX checkpoint for reproducibility
    charset.txt              # 36 characters, one per line
data/
  ocr/
    train/
      crops/                 # Cropped training plate images
      labels.csv             # filename,text
    val/
      crops/
      labels.csv
```

The repo ignores `models/ocr/*.engine` by default. Keep the engine on the
Jetson (or package it with deployment artifacts).

## 3. Dataset Preparation

1. **Crop plates from detection annotations**
   ```bash
   python tools/crop_from_boxes.py \
     --coco data/processed/train/coco.json \
     --images data/processed/train/images \
     --outdir data/ocr/train/crops
   ```
   Repeat for validation. Ensure `labels.csv` contains `filename,text` lines.

2. **Synthetic augmentation (recommended)**
   - Generate 50k–200k synthetic Indonesian plates using realistic fonts.
   - Apply perspective jitter (±4°), motion blur, exposure changes, and mild
     noise. Save alongside real crops with labels appended to `labels.csv`.

3. **Optional rectification**
   - If you have quadrilateral annotations, warp each crop to a canonical
     rectangle before resizing to `32×160`. CLAHE remains useful afterward.

## 4. Fine-tuning with PaddleOCR

Create a PaddleOCR YAML configuration based on `rec_svtr_lcnet.yml` with these
key overrides:

```yaml
Global:
  character_dict_path: ./charset_36.txt  # 36 lines matching our charset
  use_space_char: False
  save_model_dir: ./output/rec_in_id/
  pretrained_model: ./ppocr_rec/ppocrv4_rec_latin      # Paddle checkpoint
  image_shape: [3, 32, 160]
  max_text_length: 16

Train:
  dataset:
    name: SimpleDataSet
    data_dir: ./data/ocr/train/crops
    label_file_list: [./data/ocr/train/labels.csv]
  loader:
    batch_size_per_card: 256
    num_workers: 8
    shuffle: True

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: ./data/ocr/val/crops
    label_file_list: [./data/ocr/val/labels.csv]

Optimizer:
  name: AdamW
  beta1: 0.9
  beta2: 0.999
  lr:
    learning_rate: 0.001
  regularizer:
    factor: 1.0e-4
```

Additional tips:

- Enable AMP (`use_amp: True`) for faster training. Maintain CTC loss.
- Stop training when validation exact-match stops improving (target ≥ 97%).
- Optionally distill from a heavier model (e.g., PARSeq) using KL loss; the
  student remains compatible with TensorRT.

Launch training (single GPU example):

```bash
python tools/train.py -c configs/rec/rec_svtr_lcnet.yml \
  -o Global.pretrained_model=./ppocr_rec/ppocrv4_rec_latin \
     Global.save_model_dir=./output/rec_in_id/
```

Monitor CER/exact-match on the validation split. Export the best checkpoint.

## 5. Export to ONNX

1. Save the inference model:
   ```bash
   python tools/export_model.py -c configs/rec/rec_svtr_lcnet.yml \
     -o Global.checkpoints=./output/rec_in_id/best_accuracy \
        Global.save_inference_dir=./output/rec_in_id/inference
   ```

2. Convert to ONNX (opset ≥ 11, static input shape):
   ```bash
   paddle2onnx \
     --model_dir ./output/rec_in_id/inference \
     --model_filename inference.pdmodel \
     --params_filename inference.pdiparams \
     --save_file models/ocr/ppocr_rec_32x160.onnx \
     --opset_version 11 \
     --input_shape_dict "{'x':[1,3,32,160]}"
   ```

3. Verify the ONNX graph:
   ```bash
   python - <<'PY'
   import onnx
   m = onnx.load("models/ocr/ppocr_rec_32x160.onnx")
   onnx.checker.check_model(m)
   print("Inputs:", [i.type.tensor_type.shape.dim[2].dim_value for i in m.graph.input])
   print("Outputs:", [o.type.tensor_type.shape for o in m.graph.output])
   PY
   ```
   Expect output shape `[1, T, 37]` or `[1, 37, T]`. The runtime handles either
   layout by transposing as needed.

## 6. Build TensorRT Engine (on Jetson NX)

TensorRT engines must be built on the target Jetson to guarantee kernel
compatibility.

```bash
trtexec \
  --onnx=models/ocr/ppocr_rec_32x160.onnx \
  --saveEngine=models/ocr/ppocr_rec_fp16.engine \
  --explicitBatch \
  --fp16 \
  --workspace=2048 \
  --verbose
```

Validation checklist:

- Ensure the engine reports one input (`NCHW`) and one output (`[N, T, C]` or
  `[N, C, T]`).
- Run the CLI smoke test from this repo:
  ```bash
  python -m alpr_jetson ocr-infer \
    --engine models/ocr/ppocr_rec_fp16.engine \
    --charset models/ocr/charset.txt \
    --source data/ocr/val/crops/ \
    --output export/ocr_val_preds.csv
  ```

## 7. Integrating with the Runtime

- `models/ocr/charset.txt` — Create with 36 lines (0–9, A–Z). Our runtime
  prepends the blank token automatically.
- `src/ocr_service/preprocess.py` — Now supports duplicating grayscale crops to
  3 channels when `channels=3` is selected in `PreprocConfig`.
- `src/ocr_service/trt_infer.py` — Accepts an optional `logits_layout` flag. If
  your engine emits `[N, C, T]`, set `logits_layout="NCT"` to transpose before
  decoding.
- `src/pipeline/track_aggregator.py` — Performs temporal majority voting; no
  changes required.
- `src/api_server/server.py` — Serves `/v1/events` and `/v1/ws` for downstream
  consumers once you push events into its in-memory state.

## 8. Evaluation

Use the provided `tools/eval_ocr.py` to compute exact-match and character error
rate (CER):

```bash
python tools/eval_ocr.py \
  --engine models/ocr/ppocr_rec_fp16.engine \
  --charset models/ocr/charset.txt \
  --crops data/ocr/val/crops \
  --labels data/ocr/val/labels.csv
```

Targets:

- Daytime: exact-match ≥ 97 %, CER ≤ 1.5 %
- Night/rain: exact-match ≥ 93 %, CER ≤ 3.5 %

If accuracy dips, review data balance, font diversity, and rectification. Adjust
post-processing ambiguity maps (`src/ocr_service/postprocess.py`) if certain
letters consistently flip (e.g., `P ↔ F`).

## 9. Maintenance Notes

- Rebuild the engine whenever TensorRT or CUDA versions change.
- Keep ONNX and training configs versioned alongside model metadata so the
  engine is reproducible.
- Document training runs (hyperparameters, datasets) in `plan.md` or
  `progress/*.md` per session for traceability.
- For INT8, gather ≥ 2k representative crops and run `trtexec --int8` with an
  appropriate calibration cache. Only pursue if OCR latency dominates overall
  pipeline time.

This document should stay synchronized with `plan.md` Section 5 (Week 2 tasks).
If you deviate from PaddleOCR, update both references accordingly.

