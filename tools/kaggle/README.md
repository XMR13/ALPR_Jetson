# Kaggle Training Guide — YOLOX Plate Detector

This folder contains a self-contained Kaggle Notebook to train the YOLOX-S plate detector on a fast GPU and export artifacts to download back to your computer.

## Prerequisites
- Create a Kaggle Dataset from your local data. Recommended structure (matches your current paths):
  - `annotations/instances_Train.json`
  - `annotations/instances_Validation.json`
  - `images/Train/*.jpg`
  - `images/Validation/*.jpg`
- (Optional, recommended) Create a Kaggle Dataset for pretrained weights containing `YOLOX_S.pth` (COCO). Name it e.g. `yolox-weights`.
- Start a new Kaggle Notebook with GPU (T4/A100), turn Internet ON (for `pip install`) or ensure all wheels are provided as input datasets.
- Attach your dataset(s) in the Notebook’s “Add data” panel. Note their input mount paths like `/kaggle/input/<your-dataset-slug>/`.

## How to Use
1) Open the Notebook: `tools/kaggle/yolox_plate_train.ipynb` (upload to Kaggle or copy cells).
2) Set the configuration cell at the top:
   - `DATA_DIR` → `/kaggle/input/<your-dataset-slug>`
   - `TRAIN_JSON`, `VAL_JSON`, `TRAIN_NAME`, `VAL_NAME` if different.
   - `PRETRAIN_CKPT` → either a path under `/kaggle/input/yolox-weights/YOLOX_S.pth` or leave empty to train from scratch.
3) Run all cells. The Notebook will:
   - Install dependencies (`yolox`, `pycocotools`, etc.).
   - Materialize the tuned YOLOX experiment file.
   - Print dataset stats to guide hyperparameters.
   - Train for 80 epochs with a 15-epoch no-aug tail (FP16 enabled).
   - Evaluate on the validation set each epoch.
   - Export ONNX and package artifacts in `/kaggle/working/export/`.
4) Download artifacts (menu: “File” → “Download All” or zip `/kaggle/working/export`).

## Outputs
- `/kaggle/working/YOLOX_outputs/<expn>/` — YOLOX logs, TensorBoard, checkpoints (`best_ckpt.pth`).
- `/kaggle/working/export/` — `yolox_s.onnx`, `metrics.json`, and the best checkpoint copy.

## Notes
- If many plates are very small (median bbox height < 40 px), consider increasing input size to 736–800. The exp already uses strong mosaic and a no-aug tail with L1 for better localization.
- TensorRT engine builds are not available in Kaggle; export ONNX here, then build FP16 TRT on Jetson using `tools/trtexec_build.sh`.
- For fully offline runs (Internet OFF), prepackage wheels (torch, yolox, pycocotools, onnx, onnxsim) in a Kaggle dataset and install from local file paths instead of PyPI.

