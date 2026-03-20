"""
YOLOv8l FPB (Food Portion Benchmark) Fine-Tuning Script
=========================================================
Fine-tunes the UEC256-trained yolo.pt on the Food Portion Benchmark dataset.

FPB dataset facts:
  - 14,076 images  (9,518 train / 2,362 val / 2,196 test)
  - 138 food classes — Central Asian cuisine + international foods
  - Real tight bounding boxes (Roboflow-exported YOLO format)
  - Labels have 6 columns: class cx cy w h weight_grams  (-1 = unknown)
    → this script strips the 6th column before training

Key fine-tuning decisions:
  - freeze=0   : Unfreeze the full backbone so it can learn Central Asian
                 food textures (visually distinct from Food-101 / UEC256)
  - LR=0.0003  : Very low — this is the 3rd fine-tune stage; prevents
                 catastrophic forgetting of previous food knowledge
  - mosaic=1.0 : FPB images are single-food; mosaic synthesises multi-food
                 plate scenes — critical for multi-food detection
  - patience=8 : Early-stop quickly if plateauing (small dataset)

Data structure expected:
    Food_Portion_Benchmark/FPB_Dataset/RGB/
        data.yaml
        train/images/  train/labels/
        val/images/    val/labels/
        test/images/   test/labels/

Cache structure written:
    training_data/fpb_yolo_cache/
        images/train/   ← hard-links to FPB images (no extra disk space)
        images/val/
        labels/train/   ← cleaned 5-col YOLO labels (weight col stripped)
        labels/val/
        data.yaml       ← absolute-path yaml for Ultralytics

Usage:
    python scripts/train_yolov8_fpb.py

Resume:
    Re-run the same command — automatically resumes from last.pt if found.

Outputs:
    weights/yolo.pt                        — updated pipeline weight
    weights/yolo_uec256_backup.pt          — backup of UEC256-trained model
    training_data/fpb_yolo_cache/          — cleaned dataset cache
    runs/detect/fpb/weights/best.pt
    runs/detect/fpb/weights/last.pt

Env-var overrides:
    FPB_EPOCHS    int    25
    FPB_BATCH     int    16
    FPB_IMGSZ     int   640
    FPB_LR0       float 0.0003
    FPB_WORKERS   int     2
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).parent
ROOT_DIR    = SCRIPT_DIR.parent
FPB_DIR     = ROOT_DIR / "Food_Portion_Benchmark" / "FPB_Dataset" / "RGB"
FPB_WEIGHTS = ROOT_DIR / "Food_Portion_Benchmark" / "Model Weights" / "YOLOv8L.pt"
CACHE_DIR   = ROOT_DIR / "training_data" / "fpb_yolo_cache"
WEIGHTS_DIR = ROOT_DIR / "weights"

# Use clean COCO-pretrained YOLOv8L from FPB repo as base.
# Food-101 pseudo-box training taught the wrong thing (whole image = 1 box).
BASE_WEIGHTS   = FPB_WEIGHTS
BACKUP_WEIGHTS = WEIGHTS_DIR / "yolo_food101_backup.pt"   # keep existing backup
OUTPUT_PT      = WEIGHTS_DIR / "yoloo.pt"

RUN_NAME = "fpb"
RUNS_DIR = ROOT_DIR / "runs" / "detect" / RUN_NAME

EPOCHS  = int(os.getenv("FPB_EPOCHS",   "33"))
BATCH   = int(os.getenv("FPB_BATCH",     "8"))
IMGSZ   = int(os.getenv("FPB_IMGSZ",   "640"))
LR0     = float(os.getenv("FPB_LR0",   "0.01"))   # full training LR (COCO base → food)
WORKERS = int(os.getenv("FPB_WORKERS",   "2"))


# ---------------------------------------------------------------------------
# Step 1 — Build clean YOLO cache
#   images/train  ← hard-links (no extra disk space)
#   labels/train  ← cleaned 5-col labels (weight column stripped)
# ---------------------------------------------------------------------------

def _build_clean_cache() -> Path:
    """
    Standard Ultralytics layout:
        cache/images/train/  →  cache/labels/train/   (auto-resolved)
        cache/images/val/    →  cache/labels/val/

    FPB labels have 6 cols (class cx cy w h weight_grams); strip col 6.
    Images are hard-linked — zero extra disk space.
    """
    data_yaml = CACHE_DIR / "data.yaml"
    if data_yaml.exists():
        print(f"  FPB clean cache found at {CACHE_DIR} — skipping generation.")
        return data_yaml

    print(f"  Building FPB clean YOLO cache in {CACHE_DIR} …")
    t0 = time.time()

    # Only train + val needed for training
    splits = ["train", "val"]
    stats = {}

    for split in splits:
        src_img_dir = FPB_DIR / split / "images"
        src_lbl_dir = FPB_DIR / split / "labels"

        dst_img_dir = CACHE_DIR / "images" / split
        dst_lbl_dir = CACHE_DIR / "labels" / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        if not src_img_dir.exists():
            print(f"  [warn] image dir not found: {src_img_dir}")
            continue
        if not src_lbl_dir.exists():
            print(f"  [warn] label dir not found: {src_lbl_dir}")
            continue

        images_linked = 0
        labels_written = 0
        boxes_stripped = 0

        # Hard-link images
        for img_path in src_img_dir.glob("*.jpg"):
            dst = dst_img_dir / img_path.name
            if not dst.exists():
                try:
                    os.link(img_path, dst)
                except OSError:
                    shutil.copy2(img_path, dst)
            images_linked += 1

        # Clean + write labels
        for lbl_file in src_lbl_dir.glob("*.txt"):
            lines_in = lbl_file.read_text(encoding="utf-8").splitlines()
            lines_out = []
            for line in lines_in:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 6:
                    lines_out.append(" ".join(parts[:5]))
                    boxes_stripped += 1
                elif len(parts) == 5:
                    lines_out.append(line)
                # skip malformed (<5 col) lines

            (dst_lbl_dir / lbl_file.name).write_text(
                "\n".join(lines_out) + ("\n" if lines_out else ""),
                encoding="utf-8"
            )
            labels_written += 1

        stats[split] = (images_linked, labels_written, boxes_stripped)
        print(f"  {split:5s}: {images_linked:,} images  {labels_written:,} labels  "
              f"{boxes_stripped:,} weight-cols stripped")

    # Read nc + names from original data.yaml (single-line list format)
    orig_yaml = FPB_DIR / "data.yaml"
    nc_line = "nc: 138"
    names_line = ""
    for line in orig_yaml.read_text(encoding="utf-8").splitlines():
        if line.startswith("nc:"):
            nc_line = line.strip()
        if line.startswith("names:"):
            names_line = line.strip()

    # Write standard Ultralytics data.yaml
    yaml_content = (
        f"path: {CACHE_DIR.as_posix()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"\n"
        f"{nc_line}\n"
        f"{names_line}\n"
    )
    data_yaml.write_text(yaml_content, encoding="utf-8")

    elapsed = time.time() - t0
    print(f"  Cache built in {elapsed:.1f}s  →  {data_yaml}")
    return data_yaml


# ---------------------------------------------------------------------------
# Step 2 — Fine-tune
# ---------------------------------------------------------------------------

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"  YOLOv8l FPB Fine-Tuning  (138 food classes)")
    print(f"{'='*60}")
    if device == "cuda":
        print(f"  Device   : {device}  ({torch.cuda.get_device_name(0)})")
    else:
        print(f"  Device   : {device}")
    print(f"  Base     : {BASE_WEIGHTS}")
    print(f"  Epochs   : {EPOCHS}  |  Batch: {BATCH}  |  ImgSz: {IMGSZ}  |  LR0: {LR0}")
    print(f"  Freeze   : 0 (COCO base → full fine-tune on food classes)")
    print(f"  Mosaic   : 1.0 (synthesises multi-food plates from single-food images)")
    print(f"  Output   : {OUTPUT_PT}")
    print(f"{'='*60}\n")

    if not FPB_DIR.exists():
        print(f"ERROR: FPB dataset not found at {FPB_DIR}")
        sys.exit(1)
    if not BASE_WEIGHTS.exists():
        print(f"ERROR: Base weights not found at {BASE_WEIGHTS}")
        print("  Expected clean COCO-pretrained YOLOv8L at:")
        print(f"  {BASE_WEIGHTS}")
        sys.exit(1)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build / verify clean label cache
    print("\nPreparing FPB YOLO dataset cache …")
    data_yaml = _build_clean_cache()

    # Resume detection
    last_pt  = RUNS_DIR / "weights" / "last.pt"
    resuming = last_pt.exists()

    if resuming:
        print(f"\nResuming from {last_pt}")
        model = YOLO(str(last_pt))
    else:
        print(f"\nFine-tuning from {BASE_WEIGHTS}")
        model = YOLO(str(BASE_WEIGHTS))

    print(f"\nTraining {'(resume)' if resuming else '(fine-tune)'} …\n")

    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,

        # Full training LR — COCO base, training food detection from scratch
        lr0=LR0,
        lrf=0.01,
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=5,
        warmup_momentum=0.8,

        # Loss gains
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.0,
        mosaic=1.0,     # KEY: synthesises multi-food plates from single-food images
        mixup=0.1,      # slight blend for generalisation
        copy_paste=0.0,
        scale=0.5,
        translate=0.1,

        # Full backbone unfrozen — proper food detection training from COCO base
        freeze=0,

        # Training settings
        label_smoothing=0.1,
        workers=WORKERS,
        device=device,
        project=str(ROOT_DIR / "runs" / "detect"),
        name=RUN_NAME,
        exist_ok=True,
        resume=resuming,
        amp=True,
        patience=15,
        save_period=5,
        plots=True,
        verbose=True,
    )

    # Copy best → weights/yolo.pt
    best_pt = RUNS_DIR / "weights" / "best.pt"
    if best_pt.exists():
        shutil.copy2(best_pt, OUTPUT_PT)
        print(f"\n{'='*60}")
        print(f"  FPB fine-tuning complete!")
        print(f"  Pipeline weight  : {OUTPUT_PT}")
        print(f"  UEC256 backup    : {BACKUP_WEIGHTS}")
        print(f"  Best checkpoint  : {best_pt}")
        print(f"{'='*60}")
    else:
        print(f"\n[warn] best.pt not found at {best_pt}")


if __name__ == "__main__":
    train()
