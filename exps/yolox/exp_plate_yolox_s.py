"""YOLOX-s experiment for Indonesian license plate detection.

This experiment targets a single class (license_plate) and allows runtime
override of dataset paths via environment variables so you can train without
moving data around. Defaults are repo-local but can be changed at run time.

Env overrides (optional):
- YOLOX_DATA_DIR   → base data directory
- YOLOX_TRAIN_ANN  → path to train annotation JSON (relative to DATA_DIR or absolute)
- YOLOX_VAL_ANN    → path to val annotation JSON (relative to DATA_DIR or absolute)
- YOLOX_TRAIN_NAME → subfolder name for train images (default: train)
- YOLOX_VAL_NAME   → subfolder name for val images (default: val)

Example (Windows, using your dataset without copying):
  set YOLOX_DATA_DIR=D:\\RZQ\\Coding\\Datasets\\ALPR_First trial
  set YOLOX_TRAIN_ANN=annotations\\instances_Train.json
  set YOLOX_VAL_ANN=annotations\\instances_Validation.json
  set YOLOX_TRAIN_NAME=images\\Train
  set YOLOX_VAL_NAME=images\\Validation
  python -m yolox.tools.train -f exps/yolox/exp_plate_yolox_s.py -d 1 -b 16 --fp16 -o
"""

from __future__ import annotations

import os
from yolox.exp import Exp as _BaseExp


class Exp(_BaseExp):
    def __init__(self) -> None:
        super().__init__()

        # Model scale (YOLOX-s)
        self.depth = 0.33
        self.width = 0.50
        self.input_size = (640, 640)
        self.test_size = (640, 640)

        # Single class: license_plate
        self.num_classes = 1

        # Training schedule
        self.max_epoch = 50
        self.warmup_epochs = 3
        self.basic_lr_per_img = 0.01 / 64.0
        self.eval_interval = 1
        self.print_interval = 50
        self.data_num_workers = 4

        # Augmentations
        self.mosaic_prob = 0.5
        self.mixup_prob = 0.1
        self.hsv_prob = 1.0
        self.flip_prob = 0.5

        # Dataset paths (can be overridden via env)
        data_dir = os.getenv("YOLOX_DATA_DIR", "data/yolox/cam01")
        train_ann = os.getenv("YOLOX_TRAIN_ANN", "annotations/train.json")
        val_ann = os.getenv("YOLOX_VAL_ANN", "annotations/val.json")
        train_name = os.getenv("YOLOX_TRAIN_NAME", "train")
        val_name = os.getenv("YOLOX_VAL_NAME", "val")

        self.data_dir = data_dir
        self.train_ann = train_ann
        self.val_ann = val_ann
        self.train_name = train_name
        self.val_name = val_name

        # Run name
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

