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

1. **Build OCR crops + labels.csv from CVAT** (preferred)
   - In CVAT, define label `plate` with string attribute `text`.
   - Export "CVAT for images 1.1" JSON.
   - Convert to crops + CSV:
   ```bash
   python tools/ocr_from_cvat.py \
     --json export/cvat_images_1.1.json \
     --images-dir data/raw/cam01/frames \
     --outdir data/ocr/train
   ```
   Repeat for validation with a separate export.

   Alternate (if you only have COCO + attributes):
   ```bash
   python tools/ocr_from_cvat.py \
     --json data/processed/train/coco.json \
     --images-dir data/processed/train/images \
     --outdir data/ocr/train
   ```

2. **Synthetic augmentation (recommended)**
   - Generate Indonesian-style plates with `tools/synth_plates.py`:
   ```bash
   python tools/synth_plates.py \
     --outdir data/ocr/synth \
     --count 80000 \
     --fonts-dir assets/fonts
   ```
   - Merge `data/ocr/synth/labels.csv` into your train CSV, and copy crops into
     the same `crops/` folder (or point PaddleOCR to multiple datasets).

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

## 7. Adaptive Preprocessing & Runtime Control

The Jetson runtime now adapts OCR preprocessing per crop so you don’t need to
restart when lighting or plate colors change:

- **Auto color cast** (gray-world) normalizes red/yellow LEDs and tinted IR
  light before converting to grayscale.
- **Glare/speck suppression** detects saturated hotspots per crop and either
  inpaints them or clips only the affected pixels. Tiny bright specks (screws,
  dirt) are removed automatically when detected.
- **Gamma lift** brightens very dark plates before resizing.
- **Dual-polarity** support: crops are inverted automatically when the plate is
  dark background with light text (commercial/yellow, police, etc.).
- All of the above are gated per crop; if the frame is clean daylight, nothing
  extra runs.

### Configuring on Jetson (TensorRT path)

Environment toggles (optional, defaults are auto/on):

```
export ALPR_OCR_AUTO_PREPROC=1            # master switch for per-crop adaptation
export ALPR_OCR_AUTO_COLOR=1             # gray-world color cast balancing
export ALPR_OCR_GAMMA=1                  # gamma lift for dark crops
export ALPR_OCR_AUTO_POLARITY=1          # auto invert for light-on-dark plates
export ALPR_OCR_HL_THRESHOLD=245         # glare threshold (0 = auto quantile)
export ALPR_OCR_SUPPRESS_HL=1            # always allow glare suppression
export ALPR_OCR_REMOVE_SPECKS=1          # enable speck cleanup
```

You can fine-tune these live without restarting via the API:

```
curl -X POST http://localhost:8000/v1/config/preproc \
  -H 'Content-Type: application/json' \
  -d '{
        "clahe_brightness_gate": 180,
        "suppress_highlights": true,
        "highlight_inpaint_radius": 2,
        "remove_small_bright_specks": true,
        "speck_area_px": 8,
        "gamma_correction": true,
        "gamma_dark_gate": 95,
        "auto_polarity": true
      }'
```

The handler updates the in-memory TensorRT preprocessor immediately. Use the
same endpoint to roll changes back (e.g., set `auto_polarity` to `false`).

For the ONNX fallback (workstation dev), extend your `plate_config` YAML with
matching keys (e.g., `auto_color_cast: true`, `auto_polarity: true`).

Metrics to watch:

- `/metrics` now exposes `alpr_queue_*` and request counters; add your own log
  statements when you toggle preprocessing to correlate with improved OCR.

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


### the first log is for LPR Net
&&&& RUNNING TensorRT.trtexec [TensorRT v8502] # /usr/src/tensorrt/bin/trtexec --loadEngine=/home/iks-ai2/Development/ALPR_Jetson/models/ocr/LPR_net.engine --dumpProfile --verbose
[10/21/2025-15:16:09] [I] === Model Options ===
[10/21/2025-15:16:09] [I] Format: *
[10/21/2025-15:16:09] [I] Model: 
[10/21/2025-15:16:09] [I] Output:
[10/21/2025-15:16:09] [I] === Build Options ===
[10/21/2025-15:16:09] [I] Max batch: 1
[10/21/2025-15:16:09] [I] Memory Pools: workspace: default, dlaSRAM: default, dlaLocalDRAM: default, dlaGlobalDRAM: default
[10/21/2025-15:16:09] [I] minTiming: 1
[10/21/2025-15:16:09] [I] avgTiming: 8
[10/21/2025-15:16:09] [I] Precision: FP32
[10/21/2025-15:16:09] [I] LayerPrecisions: 
[10/21/2025-15:16:09] [I] Calibration: 
[10/21/2025-15:16:09] [I] Refit: Disabled
[10/21/2025-15:16:09] [I] Sparsity: Disabled
[10/21/2025-15:16:09] [I] Safe mode: Disabled
[10/21/2025-15:16:09] [I] DirectIO mode: Disabled
[10/21/2025-15:16:09] [I] Restricted mode: Disabled
[10/21/2025-15:16:09] [I] Build only: Disabled
[10/21/2025-15:16:09] [I] Save engine: 
[10/21/2025-15:16:09] [I] Load engine: /home/iks-ai2/Development/ALPR_Jetson/models/ocr/LPR_net.engine
[10/21/2025-15:16:09] [I] Profiling verbosity: 0
[10/21/2025-15:16:09] [I] Tactic sources: Using default tactic sources
[10/21/2025-15:16:09] [I] timingCacheMode: local
[10/21/2025-15:16:09] [I] timingCacheFile: 
[10/21/2025-15:16:09] [I] Heuristic: Disabled
[10/21/2025-15:16:09] [I] Preview Features: Use default preview flags.
[10/21/2025-15:16:09] [I] Input(s)s format: fp32:CHW
[10/21/2025-15:16:09] [I] Output(s)s format: fp32:CHW
[10/21/2025-15:16:09] [I] Input build shapes: model
[10/21/2025-15:16:09] [I] Input calibration shapes: model
[10/21/2025-15:16:09] [I] === System Options ===
[10/21/2025-15:16:09] [I] Device: 0
[10/21/2025-15:16:09] [I] DLACore: 
[10/21/2025-15:16:09] [I] Plugins:
[10/21/2025-15:16:09] [I] === Inference Options ===
[10/21/2025-15:16:09] [I] Batch: 1
[10/21/2025-15:16:09] [I] Input inference shapes: model
[10/21/2025-15:16:09] [I] Iterations: 10
[10/21/2025-15:16:09] [I] Duration: 3s (+ 200ms warm up)
[10/21/2025-15:16:09] [I] Sleep time: 0ms
[10/21/2025-15:16:09] [I] Idle time: 0ms
[10/21/2025-15:16:09] [I] Streams: 1
[10/21/2025-15:16:09] [I] ExposeDMA: Disabled
[10/21/2025-15:16:09] [I] Data transfers: Enabled
[10/21/2025-15:16:09] [I] Spin-wait: Disabled
[10/21/2025-15:16:09] [I] Multithreading: Disabled
[10/21/2025-15:16:09] [I] CUDA Graph: Disabled
[10/21/2025-15:16:09] [I] Separate profiling: Disabled
[10/21/2025-15:16:09] [I] Time Deserialize: Disabled
[10/21/2025-15:16:09] [I] Time Refit: Disabled
[10/21/2025-15:16:09] [I] NVTX verbosity: 0
[10/21/2025-15:16:09] [I] Persistent Cache Ratio: 0
[10/21/2025-15:16:09] [I] Inputs:
[10/21/2025-15:16:09] [I] === Reporting Options ===
[10/21/2025-15:16:09] [I] Verbose: Enabled
[10/21/2025-15:16:09] [I] Averages: 10 inferences
[10/21/2025-15:16:09] [I] Percentiles: 90,95,99
[10/21/2025-15:16:09] [I] Dump refittable layers:Disabled
[10/21/2025-15:16:09] [I] Dump output: Disabled
[10/21/2025-15:16:09] [I] Profile: Enabled
[10/21/2025-15:16:09] [I] Export timing to JSON file: 
[10/21/2025-15:16:09] [I] Export output to JSON file: 
[10/21/2025-15:16:09] [I] Export profile to JSON file: 
[10/21/2025-15:16:09] [I] 
[10/21/2025-15:16:09] [I] === Device Information ===
[10/21/2025-15:16:09] [I] Selected Device: Xavier
[10/21/2025-15:16:09] [I] Compute Capability: 7.2
[10/21/2025-15:16:09] [I] SMs: 6
[10/21/2025-15:16:09] [I] Compute Clock Rate: 1.109 GHz
[10/21/2025-15:16:09] [I] Device Global Memory: 14885 MiB
[10/21/2025-15:16:09] [I] Shared Memory per SM: 96 KiB
[10/21/2025-15:16:09] [I] Memory Bus Width: 256 bits (ECC disabled)
[10/21/2025-15:16:09] [I] Memory Clock Rate: 1.109 GHz
[10/21/2025-15:16:09] [I] 
[10/21/2025-15:16:09] [I] TensorRT version: 8.5.2
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::BatchedNMSDynamic_TRT version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::BatchedNMS_TRT version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::BatchTilePlugin_TRT version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::Clip_TRT version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::CoordConvAC version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::CropAndResizeDynamic version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::CropAndResize version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::DecodeBbox3DPlugin version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::DetectionLayer_TRT version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::EfficientNMS_Explicit_TF_TRT version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::EfficientNMS_Implicit_TF_TRT version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::EfficientNMS_ONNX_TRT version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::EfficientNMS_TRT version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::FlattenConcat_TRT version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::GenerateDetection_TRT version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::GridAnchor_TRT version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::GridAnchorRect_TRT version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::GroupNorm version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::InstanceNormalization_TRT version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::InstanceNormalization_TRT version 2
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::LayerNorm version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::LReLU_TRT version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::MultilevelCropAndResize_TRT version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::MultilevelProposeROI_TRT version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::MultiscaleDeformableAttnPlugin_TRT version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::NMSDynamic_TRT version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::NMS_TRT version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::Normalize_TRT version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::PillarScatterPlugin version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::PriorBox_TRT version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::ProposalDynamic version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::ProposalLayer_TRT version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::Proposal version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::PyramidROIAlign_TRT version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::Region_TRT version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::Reorg_TRT version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::ResizeNearest_TRT version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::ROIAlign_TRT version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::RPROI_TRT version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::ScatterND version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::SeqLen2Spatial version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::SpecialSlice_TRT version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::SplitGeLU version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::Split version 1
[10/21/2025-15:16:09] [V] [TRT] Registered plugin creator - ::VoxelGeneratorPlugin version 1
[10/21/2025-15:16:09] [I] Engine loaded in 0.0366376 sec.
[10/21/2025-15:16:10] [I] [TRT] Loaded engine size: 28 MiB
[10/21/2025-15:16:11] [V] [TRT] Deserialization required 47495 microseconds.
[10/21/2025-15:16:11] [I] [TRT] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +27, now: CPU 0, GPU 27 (MiB)
[10/21/2025-15:16:11] [I] Engine deserialized in 1.89224 sec.
[10/21/2025-15:16:11] [V] [TRT] Total per-runner device persistent memory is 28160
[10/21/2025-15:16:11] [V] [TRT] Total per-runner host persistent memory is 68384
[10/21/2025-15:16:11] [V] [TRT] Allocated activation device memory of size 39618048
[10/21/2025-15:16:13] [I] [TRT] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +38, now: CPU 0, GPU 65 (MiB)
[10/21/2025-15:16:13] [I] Setting persistentCacheLimit to 0 bytes.
[10/21/2025-15:16:13] [V] Using enqueueV3.
[10/21/2025-15:16:13] [I] Using random values for input image_input
[10/21/2025-15:16:13] [I] Created input binding for image_input with dimensions 1x3x48x96
[10/21/2025-15:16:13] [I] Using random values for output tf_op_layer_Max
[10/21/2025-15:16:13] [I] Created output binding for tf_op_layer_Max with dimensions 1x24
[10/21/2025-15:16:13] [I] Using random values for output tf_op_layer_ArgMax
[10/21/2025-15:16:13] [I] Created output binding for tf_op_layer_ArgMax with dimensions 1x24
[10/21/2025-15:16:13] [I] Starting inference
[10/21/2025-15:16:16] [I] The e2e network timing is not reported since it is inaccurate due to the extra synchronizations when the profiler is enabled.
[10/21/2025-15:16:16] [I] To show e2e network timing report, add --separateProfileRun to profile layer timing in a separate run or remove --dumpProfile to disable the profiler.
[10/21/2025-15:16:16] [I] 
[10/21/2025-15:16:16] [I] === Profile (309 iterations ) ===
[10/21/2025-15:16:16] [I]                                                                                Layer   Time (ms)   Avg. Time (ms)   Median Time (ms)   Time %
[10/21/2025-15:16:16] [I]                                                                 [HostToDeviceCopy 0]       16.77           0.0543             0.0285      0.5
[10/21/2025-15:16:16] [I]                                                       tf_op_layer_Sum/Sum_reduce_min       12.65           0.0409             0.0308      0.4
[10/21/2025-15:16:16] [I]                  Reformatting CopyNode for Input Tensor 0 to conv1 + PWN(re_lu_clip)       17.29           0.0560             0.0420      0.6
[10/21/2025-15:16:16] [I]                                                              conv1 + PWN(re_lu_clip)       28.84           0.0933             0.0895      0.9
[10/21/2025-15:16:16] [I]                                                                 re_lu/Relu:0_pooling       14.61           0.0473             0.0427      0.5
[10/21/2025-15:16:16] [I]                                                   res2a_branch2a + PWN(re_lu_1_clip)       39.02           0.1263             0.1164      1.3
[10/21/2025-15:16:16] [I]                                                                        res2a_branch1       14.37           0.0465             0.0409      0.5
[10/21/2025-15:16:16] [I]                             res2a_branch2b + tf_op_layer_add/add + PWN(re_lu_2_clip)       39.88           0.1291             0.1253      1.3
[10/21/2025-15:16:16] [I]                                                   res2b_branch2a + PWN(re_lu_3_clip)       34.55           0.1118             0.1090      1.1
[10/21/2025-15:16:16] [I]                         res2b_branch2b + tf_op_layer_add_1/add_1 + PWN(re_lu_4_clip)       37.05           0.1199             0.1158      1.2
[10/21/2025-15:16:16] [I]                                                   res3a_branch2a + PWN(re_lu_5_clip)       21.87           0.0708             0.0673      0.7
[10/21/2025-15:16:16] [I]                                                                        res3a_branch1       11.94           0.0387             0.0380      0.4
[10/21/2025-15:16:16] [I]                         res3a_branch2b + tf_op_layer_add_2/add_2 + PWN(re_lu_6_clip)       32.26           0.1044             0.0974      1.1
[10/21/2025-15:16:16] [I]                                                   res3b_branch2a + PWN(re_lu_7_clip)       29.51           0.0955             0.0893      1.0
[10/21/2025-15:16:16] [I]                         res3b_branch2b + tf_op_layer_add_3/add_3 + PWN(re_lu_8_clip)       55.10           0.1783             0.0928      1.8
[10/21/2025-15:16:16] [I]                                                   res4a_branch2a + PWN(re_lu_9_clip)       56.36           0.1824             0.0898      1.8
[10/21/2025-15:16:16] [I]                                                                        res4a_branch1       11.87           0.0384             0.0272      0.4
[10/21/2025-15:16:16] [I]                        res4a_branch2b + tf_op_layer_add_4/add_4 + PWN(re_lu_10_clip)       44.17           0.1430             0.1301      1.4
[10/21/2025-15:16:16] [I]                                                  res4b_branch2a + PWN(re_lu_11_clip)       39.77           0.1287             0.1186      1.3
[10/21/2025-15:16:16] [I]                        res4b_branch2b + tf_op_layer_add_5/add_5 + PWN(re_lu_12_clip)       37.91           0.1227             0.1199      1.2
[10/21/2025-15:16:16] [I]                                                  res5a_branch2a + PWN(re_lu_13_clip)       53.81           0.1741             0.1695      1.8
[10/21/2025-15:16:16] [I]                                                                        res5a_branch1       11.37           0.0368             0.0361      0.4
[10/21/2025-15:16:16] [I]                        res5a_branch2b + tf_op_layer_add_6/add_6 + PWN(re_lu_14_clip)       71.28           0.2307             0.2185      2.3
[10/21/2025-15:16:16] [I]                                                  res5b_branch2a + PWN(re_lu_15_clip)      103.13           0.3337             0.2137      3.4
[10/21/2025-15:16:16] [I]                        res5b_branch2b + tf_op_layer_add_7/add_7 + PWN(re_lu_16_clip)       70.01           0.2266             0.2153      2.3
[10/21/2025-15:16:16] [I]                                               {ForeignNode[lstm_W...Max_reduce_min]}     2130.62           6.8952             6.2962     69.4
[10/21/2025-15:16:16] [I]  Reformatting CopyNode for Output Tensor 1 to {ForeignNode[lstm_W...Max_reduce_min]}       19.86           0.0643             0.0587      0.6
[10/21/2025-15:16:16] [I]                                                                        ArgMax_argmax       12.78           0.0414             0.0294      0.4
[10/21/2025-15:16:16] [I]                                                                                Total     3068.65           9.9309             9.1586    100.0
[10/21/2025-15:16:16] [I] 
&&&& PASSED TensorRT.trtexec [TensorRT v8502] # /usr/src/tensorrt/bin/trtexec --loadEngine=/home/iks-ai2/Development/ALPR_Jetson/models/ocr/LPR_net.engine --dumpProfile --verbose


### Logs for the CCT model:
&&&& RUNNING TensorRT.trtexec [TensorRT v8502] # /usr/src/tensorrt/bin/trtexec --loadEngine=/home/iks-ai2/Development/ALPR_Jetson/models/ocr/cct_s.engine --dumpProfile --verbose
[10/21/2025-15:16:50] [I] === Model Options ===
[10/21/2025-15:16:50] [I] Format: *
[10/21/2025-15:16:50] [I] Model: 
[10/21/2025-15:16:50] [I] Output:
[10/21/2025-15:16:50] [I] === Build Options ===
[10/21/2025-15:16:50] [I] Max batch: 1
[10/21/2025-15:16:50] [I] Memory Pools: workspace: default, dlaSRAM: default, dlaLocalDRAM: default, dlaGlobalDRAM: default
[10/21/2025-15:16:50] [I] minTiming: 1
[10/21/2025-15:16:50] [I] avgTiming: 8
[10/21/2025-15:16:50] [I] Precision: FP32
[10/21/2025-15:16:50] [I] LayerPrecisions: 
[10/21/2025-15:16:50] [I] Calibration: 
[10/21/2025-15:16:50] [I] Refit: Disabled
[10/21/2025-15:16:50] [I] Sparsity: Disabled
[10/21/2025-15:16:50] [I] Safe mode: Disabled
[10/21/2025-15:16:50] [I] DirectIO mode: Disabled
[10/21/2025-15:16:50] [I] Restricted mode: Disabled
[10/21/2025-15:16:50] [I] Build only: Disabled
[10/21/2025-15:16:50] [I] Save engine: 
[10/21/2025-15:16:50] [I] Load engine: /home/iks-ai2/Development/ALPR_Jetson/models/ocr/cct_s.engine
[10/21/2025-15:16:50] [I] Profiling verbosity: 0
[10/21/2025-15:16:50] [I] Tactic sources: Using default tactic sources
[10/21/2025-15:16:50] [I] timingCacheMode: local
[10/21/2025-15:16:50] [I] timingCacheFile: 
[10/21/2025-15:16:50] [I] Heuristic: Disabled
[10/21/2025-15:16:50] [I] Preview Features: Use default preview flags.
[10/21/2025-15:16:50] [I] Input(s)s format: fp32:CHW
[10/21/2025-15:16:50] [I] Output(s)s format: fp32:CHW
[10/21/2025-15:16:50] [I] Input build shapes: model
[10/21/2025-15:16:50] [I] Input calibration shapes: model
[10/21/2025-15:16:50] [I] === System Options ===
[10/21/2025-15:16:50] [I] Device: 0
[10/21/2025-15:16:50] [I] DLACore: 
[10/21/2025-15:16:50] [I] Plugins:
[10/21/2025-15:16:50] [I] === Inference Options ===
[10/21/2025-15:16:50] [I] Batch: 1
[10/21/2025-15:16:50] [I] Input inference shapes: model
[10/21/2025-15:16:50] [I] Iterations: 10
[10/21/2025-15:16:50] [I] Duration: 3s (+ 200ms warm up)
[10/21/2025-15:16:50] [I] Sleep time: 0ms
[10/21/2025-15:16:50] [I] Idle time: 0ms
[10/21/2025-15:16:50] [I] Streams: 1
[10/21/2025-15:16:50] [I] ExposeDMA: Disabled
[10/21/2025-15:16:50] [I] Data transfers: Enabled
[10/21/2025-15:16:50] [I] Spin-wait: Disabled
[10/21/2025-15:16:50] [I] Multithreading: Disabled
[10/21/2025-15:16:50] [I] CUDA Graph: Disabled
[10/21/2025-15:16:50] [I] Separate profiling: Disabled
[10/21/2025-15:16:50] [I] Time Deserialize: Disabled
[10/21/2025-15:16:50] [I] Time Refit: Disabled
[10/21/2025-15:16:50] [I] NVTX verbosity: 0
[10/21/2025-15:16:50] [I] Persistent Cache Ratio: 0
[10/21/2025-15:16:50] [I] Inputs:
[10/21/2025-15:16:50] [I] === Reporting Options ===
[10/21/2025-15:16:50] [I] Verbose: Enabled
[10/21/2025-15:16:50] [I] Averages: 10 inferences
[10/21/2025-15:16:50] [I] Percentiles: 90,95,99
[10/21/2025-15:16:50] [I] Dump refittable layers:Disabled
[10/21/2025-15:16:50] [I] Dump output: Disabled
[10/21/2025-15:16:50] [I] Profile: Enabled
[10/21/2025-15:16:50] [I] Export timing to JSON file: 
[10/21/2025-15:16:50] [I] Export output to JSON file: 
[10/21/2025-15:16:50] [I] Export profile to JSON file: 
[10/21/2025-15:16:50] [I] 
[10/21/2025-15:16:50] [I] === Device Information ===
[10/21/2025-15:16:50] [I] Selected Device: Xavier
[10/21/2025-15:16:50] [I] Compute Capability: 7.2
[10/21/2025-15:16:50] [I] SMs: 6
[10/21/2025-15:16:50] [I] Compute Clock Rate: 1.109 GHz
[10/21/2025-15:16:50] [I] Device Global Memory: 14885 MiB
[10/21/2025-15:16:50] [I] Shared Memory per SM: 96 KiB
[10/21/2025-15:16:50] [I] Memory Bus Width: 256 bits (ECC disabled)
[10/21/2025-15:16:50] [I] Memory Clock Rate: 1.109 GHz
[10/21/2025-15:16:50] [I] 
[10/21/2025-15:16:50] [I] TensorRT version: 8.5.2
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::BatchedNMSDynamic_TRT version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::BatchedNMS_TRT version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::BatchTilePlugin_TRT version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::Clip_TRT version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::CoordConvAC version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::CropAndResizeDynamic version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::CropAndResize version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::DecodeBbox3DPlugin version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::DetectionLayer_TRT version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::EfficientNMS_Explicit_TF_TRT version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::EfficientNMS_Implicit_TF_TRT version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::EfficientNMS_ONNX_TRT version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::EfficientNMS_TRT version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::FlattenConcat_TRT version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::GenerateDetection_TRT version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::GridAnchor_TRT version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::GridAnchorRect_TRT version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::GroupNorm version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::InstanceNormalization_TRT version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::InstanceNormalization_TRT version 2
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::LayerNorm version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::LReLU_TRT version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::MultilevelCropAndResize_TRT version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::MultilevelProposeROI_TRT version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::MultiscaleDeformableAttnPlugin_TRT version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::NMSDynamic_TRT version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::NMS_TRT version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::Normalize_TRT version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::PillarScatterPlugin version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::PriorBox_TRT version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::ProposalDynamic version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::ProposalLayer_TRT version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::Proposal version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::PyramidROIAlign_TRT version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::Region_TRT version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::Reorg_TRT version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::ResizeNearest_TRT version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::ROIAlign_TRT version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::RPROI_TRT version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::ScatterND version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::SeqLen2Spatial version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::SpecialSlice_TRT version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::SplitGeLU version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::Split version 1
[10/21/2025-15:16:50] [V] [TRT] Registered plugin creator - ::VoxelGeneratorPlugin version 1
[10/21/2025-15:16:50] [I] Engine loaded in 0.0204752 sec.
[10/21/2025-15:16:51] [I] [TRT] Loaded engine size: 6 MiB
[10/21/2025-15:16:52] [V] [TRT] Deserialization required 49673 microseconds.
[10/21/2025-15:16:52] [I] [TRT] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +6, now: CPU 0, GPU 6 (MiB)
[10/21/2025-15:16:52] [I] Engine deserialized in 1.75066 sec.
[10/21/2025-15:16:52] [V] [TRT] Total per-runner device persistent memory is 53248
[10/21/2025-15:16:52] [V] [TRT] Total per-runner host persistent memory is 11904
[10/21/2025-15:16:52] [V] [TRT] Allocated activation device memory of size 82112512
[10/21/2025-15:16:53] [I] [TRT] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +78, now: CPU 0, GPU 84 (MiB)
[10/21/2025-15:16:53] [I] Setting persistentCacheLimit to 0 bytes.
[10/21/2025-15:16:53] [V] Using enqueueV3.
[10/21/2025-15:16:53] [I] Using random values for input input
[10/21/2025-15:16:53] [I] Created input binding for input with dimensions 1x64x128x3
[10/21/2025-15:16:53] [I] Using random values for output Identity:0
[10/21/2025-15:16:53] [I] Created output binding for Identity:0 with dimensions 1x9x37
[10/21/2025-15:16:53] [I] Starting inference
[10/21/2025-15:16:56] [I] The e2e network timing is not reported since it is inaccurate due to the extra synchronizations when the profiler is enabled.
[10/21/2025-15:16:56] [I] To show e2e network timing report, add --separateProfileRun to profile layer timing in a separate run or remove --dumpProfile to disable the profiler.
[10/21/2025-15:16:56] [I] 
[10/21/2025-15:16:56] [I] === Profile (329 iterations ) ===
[10/21/2025-15:16:56] [I]                                                                                                                                                                                                                                                                               Layer   Time (ms)   Avg. Time (ms)   Median Time (ms)   Time %
[10/21/2025-15:16:56] [I]                                                                                                                                                                                                                                                       {ForeignNode[CCT_OCR_1/Cast]}       23.73           0.0721             0.0636      0.8
[10/21/2025-15:16:56] [I]                                                                                                                                                                                           CCT_OCR_1/rescaling_1/Cast/x:0 + (Unnamed Layer* 2) [Shuffle] + CCT_OCR_1/rescaling_1/mul       11.20           0.0341             0.0283      0.4
[10/21/2025-15:16:56] [I]                                                                                                                                                                                                                                       CCT_OCR_1/conv_stem_1/conv2d_1/convolution__7       14.91           0.0453             0.0372      0.5
[10/21/2025-15:16:56] [I]                                                                                                                                                                                                                                          CCT_OCR_1/conv_stem_1/conv2d_1/convolution       41.57           0.1263             0.1172      1.3
[10/21/2025-15:16:56] [I]                                                                                                                                                             CCT_OCR_1/conv_stem_1/conv2d_2_1/Gelu/mul/x:0 + (Unnamed Layer* 10) [Shuffle] + CCT_OCR_1/conv_stem_1/conv2d_1/Gelu/mul       11.81           0.0359             0.0353      0.4
[10/21/2025-15:16:56] [I]                                                                                                                                                                                                                                                                      Transpose__108       18.20           0.0553             0.0472      0.6
[10/21/2025-15:16:56] [I]                                                                                                                 ConstantFolding/CCT_OCR_1/transformer_block_4_1/mlp_3_1/dense_8_1/Gelu/truediv_recip:0 + (Unnamed Layer* 7) [Shuffle] + CCT_OCR_1/conv_stem_1/conv2d_1/Gelu/truediv       12.24           0.0372             0.0364      0.4
[10/21/2025-15:16:56] [I]                                                                                                                                                                                                                                                                      Transpose__106       16.48           0.0501             0.0392      0.5
[10/21/2025-15:16:56] [I]  Reformatting CopyNode for Input Tensor 0 to PWN(PWN(CCT_OCR_1/conv_stem_1/conv2d_1/Gelu/Erf, CCT_OCR_1/transformer_block_4_1/mlp_3_1/dense_7_1/Gelu/add/x:0 + (Unnamed Layer* 16) [Shuffle] + CCT_OCR_1/conv_stem_1/conv2d_1/Gelu/add), CCT_OCR_1/conv_stem_1/conv2d_1/Gelu/mul_1)       17.37           0.0528             0.0503      0.6
[10/21/2025-15:16:56] [I]  Reformatting CopyNode for Input Tensor 1 to PWN(PWN(CCT_OCR_1/conv_stem_1/conv2d_1/Gelu/Erf, CCT_OCR_1/transformer_block_4_1/mlp_3_1/dense_7_1/Gelu/add/x:0 + (Unnamed Layer* 16) [Shuffle] + CCT_OCR_1/conv_stem_1/conv2d_1/Gelu/add), CCT_OCR_1/conv_stem_1/conv2d_1/Gelu/mul_1)       16.87           0.0513             0.0499      0.5
[10/21/2025-15:16:56] [I]                                              PWN(PWN(CCT_OCR_1/conv_stem_1/conv2d_1/Gelu/Erf, CCT_OCR_1/transformer_block_4_1/mlp_3_1/dense_7_1/Gelu/add/x:0 + (Unnamed Layer* 16) [Shuffle] + CCT_OCR_1/conv_stem_1/conv2d_1/Gelu/add), CCT_OCR_1/conv_stem_1/conv2d_1/Gelu/mul_1)       32.70           0.0994             0.0956      1.1
[10/21/2025-15:16:56] [I]                                                                                                                                                                                Reformatting CopyNode for Input Tensor 0 to CCT_OCR_1/conv_stem_1/max_blur_pooling2d_1/MaxPool2d__11       18.37           0.0558             0.0546      0.6
[10/21/2025-15:16:56] [I]                                                                                                                                                                                                                            CCT_OCR_1/conv_stem_1/max_blur_pooling2d_1/MaxPool2d__11       25.83           0.0785             0.0760      0.8
[10/21/2025-15:16:56] [I]                                                                                                                                                                                                                                CCT_OCR_1/conv_stem_1/max_blur_pooling2d_1/MaxPool2d       12.92           0.0393             0.0386      0.4
[10/21/2025-15:16:56] [I]                                                                                                                                                                                                                                CCT_OCR_1/conv_stem_1/max_blur_pooling2d_1/depthwise        9.66           0.0294             0.0288      0.3
[10/21/2025-15:16:56] [I]                                                                                                                                                                                            Reformatting CopyNode for Input Tensor 0 to CCT_OCR_1/conv_stem_1/conv2d_1_2/convolution        3.96           0.0120             0.0116      0.1
[10/21/2025-15:16:56] [I]                                                                                                                                                                                                                                        CCT_OCR_1/conv_stem_1/conv2d_1_2/convolution       26.59           0.0808             0.0791      0.9
[10/21/2025-15:16:56] [I]                                                                                                                                                   CCT_OCR_1/conv_stem_1/conv2d_2_1/Gelu/mul/x:0_clone_1 + (Unnamed Layer* 25) [Shuffle] + CCT_OCR_1/conv_stem_1/conv2d_1_2/Gelu/mul        6.91           0.0210             0.0206      0.2
[10/21/2025-15:16:56] [I]                                                                                                                                                                                                                                                                      Transpose__112       12.65           0.0385             0.0379      0.4
[10/21/2025-15:16:56] [I]                                                                                                      ConstantFolding/CCT_OCR_1/transformer_block_4_1/mlp_3_1/dense_8_1/Gelu/truediv_recip:0_clone_1 + (Unnamed Layer* 23) [Shuffle] + CCT_OCR_1/conv_stem_1/conv2d_1_2/Gelu/truediv        6.69           0.0203             0.0199      0.2
[10/21/2025-15:16:56] [I]                                                                                                                                                                                                                                                                      Transpose__110       12.14           0.0369             0.0363      0.4
[10/21/2025-15:16:56] [I]                                PWN(PWN(CCT_OCR_1/conv_stem_1/conv2d_1_2/Gelu/Erf, CCT_OCR_1/transformer_block_4_1/mlp_3_1/dense_7_1/Gelu/add/x:0_clone_1 + (Unnamed Layer* 30) [Shuffle] + CCT_OCR_1/conv_stem_1/conv2d_1_2/Gelu/add), CCT_OCR_1/conv_stem_1/conv2d_1_2/Gelu/mul_1)        9.30           0.0283             0.0278      0.3
[10/21/2025-15:16:56] [I]                                                                                                                                                                                                                                    CCT_OCR_1/conv_stem_1/conv2d_2_1/convolution__26       12.09           0.0368             0.0361      0.4
[10/21/2025-15:16:56] [I]                                                                                                                                                                                                                                        CCT_OCR_1/conv_stem_1/conv2d_2_1/convolution       40.48           0.1230             0.1102      1.3
[10/21/2025-15:16:56] [I]                                                                                                                                                   CCT_OCR_1/conv_stem_1/conv2d_2_1/Gelu/mul/x:0_clone_2 + (Unnamed Layer* 37) [Shuffle] + CCT_OCR_1/conv_stem_1/conv2d_2_1/Gelu/mul        7.32           0.0222             0.0215      0.2
[10/21/2025-15:16:56] [I]                                                                                                                                                                                                                                                                      Transpose__114       15.52           0.0472             0.0435      0.5
[10/21/2025-15:16:56] [I]                                                                                                      ConstantFolding/CCT_OCR_1/transformer_block_4_1/mlp_3_1/dense_8_1/Gelu/truediv_recip:0_clone_2 + (Unnamed Layer* 35) [Shuffle] + CCT_OCR_1/conv_stem_1/conv2d_2_1/Gelu/truediv        7.74           0.0235             0.0230      0.2
[10/21/2025-15:16:56] [I]                                                                                                                                                                                                                                                                      Transpose__116       16.16           0.0491             0.0439      0.5
[10/21/2025-15:16:56] [I]                                PWN(PWN(CCT_OCR_1/conv_stem_1/conv2d_2_1/Gelu/Erf, CCT_OCR_1/transformer_block_4_1/mlp_3_1/dense_7_1/Gelu/add/x:0_clone_2 + (Unnamed Layer* 42) [Shuffle] + CCT_OCR_1/conv_stem_1/conv2d_2_1/Gelu/add), CCT_OCR_1/conv_stem_1/conv2d_2_1/Gelu/mul_1)       11.15           0.0339             0.0309      0.4
[10/21/2025-15:16:56] [I]                                                                                                                                                                                                                                    CCT_OCR_1/conv_stem_1/conv2d_3_1/convolution__30       16.21           0.0493             0.0485      0.5
[10/21/2025-15:16:56] [I]                                                                                                                                                                                            Reformatting CopyNode for Input Tensor 0 to CCT_OCR_1/conv_stem_1/conv2d_3_1/convolution        9.19           0.0279             0.0276      0.3
[10/21/2025-15:16:56] [I]                                                                                                                                                                                                                                        CCT_OCR_1/conv_stem_1/conv2d_3_1/convolution       43.62           0.1326             0.1309      1.4
[10/21/2025-15:16:56] [I]                                                                                                                                                   CCT_OCR_1/conv_stem_1/conv2d_2_1/Gelu/mul/x:0_clone_3 + (Unnamed Layer* 49) [Shuffle] + CCT_OCR_1/conv_stem_1/conv2d_3_1/Gelu/mul       10.65           0.0324             0.0320      0.3
[10/21/2025-15:16:56] [I]                                                                                                                                                                                                                                                                      Transpose__118       12.45           0.0378             0.0354      0.4
[10/21/2025-15:16:56] [I]                                                                                                      ConstantFolding/CCT_OCR_1/transformer_block_4_1/mlp_3_1/dense_8_1/Gelu/truediv_recip:0_clone_3 + (Unnamed Layer* 47) [Shuffle] + CCT_OCR_1/conv_stem_1/conv2d_3_1/Gelu/truediv        9.87           0.0300             0.0294      0.3
[10/21/2025-15:16:56] [I]                                                                                                                                                                                                                                                                      Transpose__120        9.64           0.0293             0.0282      0.3
[10/21/2025-15:16:56] [I]                                PWN(PWN(CCT_OCR_1/conv_stem_1/conv2d_3_1/Gelu/Erf, CCT_OCR_1/transformer_block_4_1/mlp_3_1/dense_7_1/Gelu/add/x:0_clone_3 + (Unnamed Layer* 54) [Shuffle] + CCT_OCR_1/conv_stem_1/conv2d_3_1/Gelu/add), CCT_OCR_1/conv_stem_1/conv2d_3_1/Gelu/mul_1)       15.01           0.0456             0.0452      0.5
[10/21/2025-15:16:56] [I]                                                                                                                                                                                             Reformatting CopyNode for Input Tensor 0 to CCT_OCR_1/patch_extractor_1/convolution__34        8.98           0.0273             0.0269      0.3
[10/21/2025-15:16:56] [I]                                                                                                                                                                                                                                         CCT_OCR_1/patch_extractor_1/convolution__34       10.44           0.0317             0.0301      0.3
[10/21/2025-15:16:56] [I]                                                                                                                                                                                                 Reformatting CopyNode for Input Tensor 0 to CCT_OCR_1/patch_extractor_1/convolution       10.17           0.0309             0.0303      0.3
[10/21/2025-15:16:56] [I]                                                                                                                                                                                                                                             CCT_OCR_1/patch_extractor_1/convolution       39.72           0.1207             0.1180      1.3
[10/21/2025-15:16:56] [I]                                                                                                                                                  {ForeignNode[CCT_OCR_1/transformer_block_6_1/multi_head_attention_5_1/Cast/x:0...CCT_OCR_1/vocab_projection_1/dense_13_1/Softmax]}     2436.45           7.4056             7.1540     78.5
[10/21/2025-15:16:56] [I]                                                                                                                                                                                                                                                                               Total     3104.96           9.4376             9.1090    100.0
[10/21/2025-15:16:56] [I] 
&&&& PASSED TensorRT.trtexec [TensorRT v8502] # /usr/src/tensorrt/bin/trtexec --loadEngine=/home/iks-ai2/Development/ALPR_Jetson/models/ocr/cct_s.engine --dumpProfile --verbose
