"""
YOLOv8l UEC Food-256 Fine-Tuning Script
=========================================
Fine-tunes the Food-101-trained yolo.pt on UECFOOD256 — a dataset with
REAL bounding box annotations and multi-food plate scenes.

After this training the detector will:
  - Draw tight boxes around individual food items on plates
  - Handle multi-food scenes (multiple items per image)
  - Recognise 256 food categories

Data structure expected:
    UEC_training_data/UECFOOD256/
        category.txt              — tab-separated: id  name
        {class_id}/               — one folder per class (1–256)
            bb_info.txt           — header: img x1 y1 x2 y2 (absolute px)
            {img_id}.jpg

Usage:
    python scripts/train_yolov8_uec256.py

Resume:
    Re-run the same command — automatically resumes from last.pt if found.

Outputs:
    weights/yolo.pt                       — updated pipeline weight
    weights/yolo_food101_backup.pt        — backup of Food-101 trained model
    training_data/uec256_yolo_cache/      — generated YOLO dataset (cached)
    runs/detect/uec256/weights/best.pt
    runs/detect/uec256/weights/last.pt

Env-var overrides:
    UEC_EPOCHS    int    25
    UEC_BATCH     int     8
    UEC_IMGSZ     int   640
    UEC_LR0       float 0.001   (low — fine-tuning, not training from scratch)
    UEC_WORKERS   int     2
    UEC_VAL_SPLIT float   0.1
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).parent
ROOT_DIR    = SCRIPT_DIR.parent
UEC_DIR     = ROOT_DIR / "UEC_training_data" / "UECFOOD256"
CACHE_DIR   = ROOT_DIR / "training_data" / "uec256_yolo_cache"
WEIGHTS_DIR = ROOT_DIR / "weights"

BASE_WEIGHTS   = WEIGHTS_DIR / "yolo.pt"            # fine-tune from food101 model
BACKUP_WEIGHTS = WEIGHTS_DIR / "yolo_food101_backup.pt"
OUTPUT_PT      = WEIGHTS_DIR / "yolo.pt"            # overwrite with improved model

RUN_NAME  = "uec256"
RUNS_DIR  = ROOT_DIR / "runs" / "detect" / RUN_NAME

EPOCHS    = int(os.getenv("UEC_EPOCHS",    "30"))
BATCH     = int(os.getenv("UEC_BATCH",     "16"))
IMGSZ     = int(os.getenv("UEC_IMGSZ",    "640"))
LR0       = float(os.getenv("UEC_LR0",   "0.001"))  # low LR — fine-tuning
WORKERS   = int(os.getenv("UEC_WORKERS",    "2"))
VAL_SPLIT = float(os.getenv("UEC_VAL_SPLIT", "0.1"))
SEED      = 42


# ---------------------------------------------------------------------------
# Step 1 — Build YOLO dataset cache from UEC256
# ---------------------------------------------------------------------------

def _read_categories() -> dict[int, str]:
    """Returns {class_id: class_name} from category.txt."""
    cat_file = UEC_DIR / "category.txt"
    categories = {}
    for line in cat_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("id"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            try:
                categories[int(parts[0])] = parts[1].strip()
            except ValueError:
                continue
    return categories


def _build_dataset(categories: dict[int, str]) -> Path:
    """
    Convert UEC256 to YOLO detection format.

    Each bb_info.txt entry: img x1 y1 x2 y2  (absolute pixel coords)
    YOLO label format:       class_id cx cy w h  (normalised 0-1)

    Cached — skips regeneration if cache exists.
    """
    data_yaml = CACHE_DIR / "data.yaml"
    if data_yaml.exists():
        print(f"  UEC256 YOLO cache found at {CACHE_DIR} — skipping generation.")
        return data_yaml

    print(f"  Building UEC256 YOLO dataset in {CACHE_DIR} …")
    t0 = time.time()

    from PIL import Image as PILImage

    # class_idx = category_id - 1  (0-indexed)
    class_names = [categories[i] for i in sorted(categories.keys())]
    class_to_idx = {cid: cid - 1 for cid in categories}   # 1→0, 2→1, …

    # Collect all (img_path, class_id, x1, y1, x2, y2) tuples
    # Key: img_path (absolute) → list of (class_idx, x1, y1, x2, y2)
    img_annotations: dict[str, list] = defaultdict(list)

    class_ids = sorted([int(d.name) for d in UEC_DIR.iterdir()
                        if d.is_dir() and d.name.isdigit()])

    missing_images = 0
    total_boxes    = 0

    for cid in class_ids:
        cls_dir   = UEC_DIR / str(cid)
        bb_file   = cls_dir / "bb_info.txt"
        cls_idx   = class_to_idx.get(cid, cid - 1)

        if not bb_file.exists():
            continue

        lines = bb_file.read_text(encoding="utf-8").splitlines()
        for line in lines[1:]:     # skip header
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                img_id = int(parts[0])
                x1, y1, x2, y2 = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
            except ValueError:
                continue

            img_path = cls_dir / f"{img_id}.jpg"
            if not img_path.exists():
                missing_images += 1
                continue

            img_annotations[str(img_path)].append((cls_idx, x1, y1, x2, y2))
            total_boxes += 1

    print(f"  Found {len(img_annotations):,} images, {total_boxes:,} boxes "
          f"({missing_images} missing)")

    # Split train / val
    all_imgs = list(img_annotations.keys())
    random.seed(SEED)
    random.shuffle(all_imgs)
    val_n     = int(len(all_imgs) * VAL_SPLIT)
    val_imgs  = set(all_imgs[:val_n])
    train_imgs = set(all_imgs[val_n:])

    # Write images + labels
    splits = {"train": train_imgs, "val": val_imgs}
    images_written = {"train": 0, "val": 0}

    for split_name, split_set in splits.items():
        img_out = CACHE_DIR / "images" / split_name
        lbl_out = CACHE_DIR / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path_str in split_set:
            img_path = Path(img_path_str)
            dst_img  = img_out / f"{img_path.parent.name}__{img_path.name}"
            dst_lbl  = lbl_out / f"{img_path.parent.name}__{img_path.stem}.txt"

            # Hard-link (no extra disk space); fall back to copy
            if not dst_img.exists():
                try:
                    os.link(img_path, dst_img)
                except OSError:
                    shutil.copy2(img_path, dst_img)

            # Get image size for normalisation
            try:
                with PILImage.open(img_path) as im:
                    iw, ih = im.size   # PIL: (width, height)
            except Exception:
                continue

            lines = []
            for cls_idx, x1, y1, x2, y2 in img_annotations[img_path_str]:
                # Clamp to image bounds
                x1 = max(0, min(iw, x1))
                x2 = max(0, min(iw, x2))
                y1 = max(0, min(ih, y1))
                y2 = max(0, min(ih, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                cx = ((x1 + x2) / 2) / iw
                cy = ((y1 + y2) / 2) / ih
                bw = (x2 - x1) / iw
                bh = (y2 - y1) / ih
                lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            images_written[split_name] += 1

    # data.yaml
    yaml_lines = [
        f"path: {CACHE_DIR.as_posix()}",
        f"train: images/train",
        f"val:   images/val",
        f"",
        f"nc: {len(class_names)}",
        f"names:",
    ] + [f'  - "{n}"' for n in class_names]
    data_yaml.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    # classes.txt
    (CACHE_DIR / "classes.txt").write_text("\n".join(class_names), encoding="utf-8")

    elapsed = time.time() - t0
    print(f"  Train: {images_written['train']:,}  Val: {images_written['val']:,}  "
          f"in {elapsed:.1f}s")

    return data_yaml


# ---------------------------------------------------------------------------
# Step 2 — Fine-tune
# ---------------------------------------------------------------------------

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"  YOLOv8l UEC Food-256 Fine-Tuning")
    print(f"{'='*60}")
    if device == "cuda":
        print(f"  Device   : {device}  ({torch.cuda.get_device_name(0)})")
    else:
        print(f"  Device   : {device}")
    print(f"  Base     : {BASE_WEIGHTS}")
    print(f"  Epochs   : {EPOCHS}  |  Batch: {BATCH}  |  ImgSz: {IMGSZ}  |  LR0: {LR0}")
    print(f"  Output   : {OUTPUT_PT}")
    print(f"{'='*60}\n")

    # Verify dirs
    if not UEC_DIR.exists():
        print(f"ERROR: UEC256 data not found at {UEC_DIR}")
        sys.exit(1)
    if not BASE_WEIGHTS.exists():
        print(f"ERROR: Base weights not found at {BASE_WEIGHTS}")
        print("  Run scripts/train_yolov8_food101.py first.")
        sys.exit(1)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # Backup food101 weights before overwriting
    if BASE_WEIGHTS.exists() and not BACKUP_WEIGHTS.exists():
        shutil.copy2(BASE_WEIGHTS, BACKUP_WEIGHTS)
        print(f"  Backed up Food-101 weights → {BACKUP_WEIGHTS.name}")

    # Read categories
    categories = _read_categories()
    print(f"Classes: {len(categories)}")

    # Build / verify dataset
    print("\nPreparing UEC256 YOLO dataset …")
    data_yaml = _build_dataset(categories)

    # Resume detection
    last_pt  = RUNS_DIR / "weights" / "last.pt"
    resuming = last_pt.exists()

    if resuming:
        print(f"\nResuming from {last_pt}")
        model = YOLO(str(last_pt))
    else:
        print(f"\nFine-tuning from {BASE_WEIGHTS}")
        model = YOLO(str(BASE_WEIGHTS))

    # Fine-tune
    print(f"\nTraining {'(resume)' if resuming else '(fine-tune)'} …\n")

    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,

        # Low LR — fine-tuning, not training from scratch
        lr0=LR0,
        lrf=0.01,
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=2,           # shorter warmup for fine-tuning
        warmup_momentum=0.8,

        # Loss gains
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # Augmentation — moderate for fine-tuning
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.5,                # helps learn multi-food scenes
        mixup=0.0,
        copy_paste=0.0,
        scale=0.5,
        translate=0.1,

        # Freeze backbone (layers 0-9) — preserve Food-101 features,
        # only train neck + detection head for new 256-class layout.
        freeze=10,

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
        print(f"  Fine-tuning complete!")
        print(f"  Pipeline weight  : {OUTPUT_PT}")
        print(f"  Food-101 backup  : {BACKUP_WEIGHTS}")
        print(f"  Best checkpoint  : {best_pt}")
        print(f"{'='*60}")
    else:
        print(f"\n[warn] best.pt not found at {best_pt}")


if __name__ == "__main__":
    train()
