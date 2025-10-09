# Kaggle Training Guide – YOLOX Plate Detector

This folder contains a self-contained Kaggle Notebook to train the YOLOX‑S plate detector on a fast GPU and export artifacts to download back to your computer.

## Best Practices Built In
- COCO pretrain support: optionally fine‑tune from `YOLOX_S.pth`.
- Longer schedule: 80 epochs with a 15‑epoch no‑augmentation tail for stability.
- Strong but plate‑friendly augs: mosaic=0.7, mixup=0.05, multi‑scale (448–832), HSV, flip.
- Localization refinement: L1 loss enabled in the final stage.
- FP16 mixed precision (CUDA/AMP) for faster training on GPU.
- Per‑epoch validation and export to ONNX at the end.

## Prerequisites
- Create a Kaggle Dataset from your local data. Recommended structure (matches your current paths):
  - `annotations/instances_Train.json`
  - `annotations/instances_Validation.json`
  - `images/Train/*.jpg`
  - `images/Validation/*.jpg`
- (Optional, recommended) Create a Kaggle Dataset for pretrained weights containing `YOLOX_S.pth` (COCO). Name it e.g. `yolox-weights`.
- Start a new Kaggle Notebook with GPU (T4/A100), turn Internet ON (for `pip install`) or ensure all wheels are provided as input datasets.
- Attach your dataset(s) in the Notebook’s “Add data” panel. Note their input mount paths like `/kaggle/input/<your-dataset-slug>/`.

## How to Use (Kaggle)
1) Open the Notebook: `tools/kaggle/yolox_plate_train.ipynb` (upload to Kaggle or copy cells).
2) Set the configuration cell at the top:
   - `DATA_DIR` → `/kaggle/input/<your-dataset-slug>`
   - `TRAIN_JSON`, `VAL_JSON`, `TRAIN_NAME`, `VAL_NAME` if different.
   - `PRETRAIN_CKPT` → either a path under `/kaggle/input/yolox-weights/YOLOX_S.pth` or leave empty to train from scratch.
3) Ensure GPU is active and CUDA is available (the notebook includes a quick check):
   ```python
   import torch; print('cuda?', torch.cuda.is_available(), 'device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
   ```
4) Run all cells. The Notebook will:
    - Install dependencies (`yolox`, `pycocotools`, etc.).
    - Clone the official YOLOX repo and install it (editable).
    - Install CUDA-enabled PyTorch (tries cu121 then falls back to cu118).
    - Materialize the tuned YOLOX experiment file.
    - Print dataset stats to guide hyperparameters.
    - Train for 80 epochs with a 15‑epoch no‑aug tail (FP16 enabled).
    - Evaluate on the validation set each epoch.
    - Export ONNX and package artifacts in `/kaggle/working/export/`.
5) Download artifacts (menu: “File” → “Download All” or zip `/kaggle/working/export`).

## Outputs
- `/kaggle/working/YOLOX_outputs/<expn>/` — YOLOX logs, TensorBoard, checkpoints (`best_ckpt.pth`).
- `/kaggle/working/export/` — `yolox_s.onnx`, `metrics.json` (COCO AP parsed from eval), and `best_ckpt.pth`.

## CUDA/AMP Notes
- The notebook uses PyTorch with CUDA and enables FP16 mixed precision (`--fp16`) when launching YOLOX training. On Kaggle, this maps to the selected GPU (T4/A100).
- If `torch.cuda.is_available()` prints `False`, enable GPU from the right‑hand “Accelerator” menu and re‑run the environment setup cells.

## Validation
- The notebook runs `yolox.tools.eval` against the best checkpoint and parses COCO summary into `/kaggle/working/export/metrics.json`.
- You can also re‑run the eval cell to sanity‑check or print full logs.
- For custom thresholds or image sizes, adjust the eval command in the export cell.

## Notes
- If many plates are very small (median bbox height < 40 px), consider increasing input size to 736–800. The exp already uses strong mosaic and a no-aug tail with L1 for better localization.
- TensorRT engine builds are not available in Kaggle; export ONNX here, then build FP16 TRT on Jetson using `tools/trtexec_build.sh`.
- For fully offline runs (Internet OFF), prepackage wheels (torch, yolox, pycocotools, onnx, onnxsim) in a Kaggle dataset and install from local file paths instead of PyPI.

## Troubleshooting
- ImportError: YOLOX not found → re‑run the install cell or switch Internet ON; alternatively, attach a dataset with pre‑downloaded wheels and install via local paths.
- CUDA shows as unavailable → set Accelerator to GPU in Notebook settings; then “Restart & Run All”.
- OOM during training → lower `batch` (e.g., 8 or 4) or disable `--cache` if you added it; reduce input size or cut mixup probability.

## After Training (Jetson)
- Copy `export/yolox_s.onnx` to Jetson under `models/detector/`.
- Build FP16 TensorRT: `tools/trtexec_build.sh models/detector/yolox_s.onnx models/detector/yolox_s_fp16.engine --fp16`.
- Validate: `python tools/eval_det_coco.py --data-dir <dir> --val-ann <ann.json> --engine models/detector/yolox_s_fp16.engine`.

## Use Official YOLOX Repo + Torch CUDA (Alternative Flow)
Yes — you can use the official YOLOX repository and install dependencies there (and install a CUDA‑enabled PyTorch) inside Kaggle. Two common ways:

- Simple (git via pip):
  ```bash
  pip install --upgrade pip setuptools wheel
  # Install CUDA PyTorch matching Kaggle image (try cu121; if it fails, use cu118)
  pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio || \
  pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio

  # Install YOLOX from GitHub
  pip install "git+https://github.com/Megvii-BaseDetection/YOLOX.git@0.3.0"
  pip install pycocotools onnx onnxsim
  ```

- Full clone (editable):
  ```bash
  git clone https://github.com/Megvii-BaseDetection/YOLOX.git /kaggle/working/YOLOX
  pip install --upgrade pip setuptools wheel
  pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio || \
  pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio
  pip install -e /kaggle/working/YOLOX
  pip install pycocotools onnx onnxsim
  ```

Verify CUDA before training:
```python
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu?', torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

Run training using our tuned exp (attach this repo as a Kaggle Dataset or copy the exp file):
```bash
export YOLOX_DATA_DIR=/kaggle/input/<your-dataset-slug>
export YOLOX_TRAIN_ANN=annotations/instances_Train.json
export YOLOX_VAL_ANN=annotations/instances_Validation.json
export YOLOX_TRAIN_NAME=images/Train
export YOLOX_VAL_NAME=images/Validation

python -m yolox.tools.train \
  -f /kaggle/working/exp_plate_yolox_s.py \
  -d 1 -b 16 --fp16 -o \
  -c /kaggle/input/yolox-weights/YOLOX_S.pth
```

Export ONNX at the end (example):
```bash
python -m yolox.tools.export \
  -f /kaggle/working/exp_plate_yolox_s.py \
  -c /kaggle/working/YOLOX_outputs/plate_yolox_s/best_ckpt.pth \
  --export onnx --decode_in_inference
```

Offline variant (Internet OFF): attach a dataset containing the wheel files for torch/torchvision/torchaudio (matching CUDA), YOLOX sdist/wheel, and all other wheels (pycocotools, onnx, onnxsim) and install via `pip install <path-to-wheel.whl>` instead of PyPI/Git.
