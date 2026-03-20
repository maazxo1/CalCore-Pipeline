"""
Open Images V7 — Food Detection Dataset Downloader
====================================================
Downloads ~50,000 food images (~10 GB) with REAL bounding box annotations.
Prioritises images containing MULTIPLE food items (plate/table scenes).
Converts to YOLO format ready for YOLOv8 fine-tuning.

Install:
    pip install fiftyone

Usage:
    python scripts/download_openimages_food.py

Resume:
    Re-run the same command — fiftyone skips already-downloaded images.

Outputs:
    training_data/openimages_food/
        images/train/    *.jpg
        images/val/      *.jpg
        labels/train/    *.txt   (YOLO: class_id cx cy w h)
        labels/val/      *.txt
        data.yaml
        classes.txt
        stats.json

Size:
    ~50,000 images ≈ 10 GB
    Change MAX_SAMPLES below to adjust (1000 images ≈ 200 MB)
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).parent
ROOT_DIR    = SCRIPT_DIR.parent
OUT_DIR     = ROOT_DIR / "training_data" / "openimages_food"

MAX_SAMPLES = int(os.getenv("OI_MAX_SAMPLES", "50000"))   # ~10 GB
VAL_SPLIT   = 0.1
SEED        = 42

# All food-related classes in Open Images V7 with bounding box annotations
FOOD_CLASSES = [
    # Prepared dishes
    "Pizza", "Hamburger", "Sushi", "Taco", "Burrito", "Hot dog",
    "Sandwich", "Salad", "Pasta", "Noodles", "Rice", "Soup",
    # Baked goods & breakfast
    "Baked goods", "Bread", "Bagel", "Pretzel", "Waffle", "Pancake",
    "Doughnut", "Muffin", "Cookie", "Croissant", "Cake", "Pastry",
    # Snacks & fast food
    "Fast food", "French fries", "Popcorn", "Potato chip", "Nachos",
    # Proteins & seafood
    "Egg (Food)", "Cheese", "Seafood", "Lobster", "Shrimp",
    "Crab", "Oyster", "Squid",
    # Fruits
    "Fruit", "Apple", "Banana", "Orange", "Strawberry", "Grape",
    "Mango", "Pineapple", "Watermelon", "Lemon", "Tomato",
    # Vegetables
    "Vegetable", "Broccoli", "Carrot", "Mushroom", "Cucumber",
    "Bell pepper", "Corn",
    # Sweets & drinks
    "Ice cream", "Chocolate", "Coffee", "Juice",
    # General
    "Food", "Dessert", "Snack",
]

# Deduplicate
seen: set = set()
FOOD_CLASSES = [c for c in FOOD_CLASSES if not (c in seen or seen.add(c))]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _export_to_yolo(dataset, split_name: str, class_to_idx: dict):
    img_out = OUT_DIR / "images" / split_name
    lbl_out = OUT_DIR / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    multi_item = 0
    class_counts: dict[str, int] = defaultdict(int)

    for sample in dataset:
        src = Path(sample.filepath)
        if not src.exists():
            skipped += 1
            continue

        dst_img = img_out / src.name
        dst_lbl = lbl_out / (src.stem + ".txt")

        # Hard-link (no extra disk); fall back to copy
        if not dst_img.exists():
            try:
                os.link(src, dst_img)
            except OSError:
                shutil.copy2(src, dst_img)

        if sample.ground_truth is None:
            dst_lbl.write_text("", encoding="utf-8")
            written += 1
            continue

        lines = []
        dets = sample.ground_truth.detections
        if len(dets) > 1:
            multi_item += 1

        for det in dets:
            if det.label not in class_to_idx:
                continue
            cls_id = class_to_idx[det.label]
            x, y, w, h = det.bounding_box      # fiftyone: top-left origin, normalised
            cx = max(0.0, min(1.0, x + w / 2))
            cy = max(0.0, min(1.0, y + h / 2))
            w  = max(0.001, min(1.0, w))
            h  = max(0.001, min(1.0, h))
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            class_counts[det.label] += 1

        dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written += 1

    print(f"  {split_name}: {written:,} images written  "
          f"({multi_item:,} with 2+ food items)  {skipped} skipped")
    return class_counts, multi_item


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    try:
        import fiftyone.zoo as foz
    except ImportError:
        print("ERROR: fiftyone not installed.\n  pip install fiftyone")
        sys.exit(1)

    train_n = int(MAX_SAMPLES * (1 - VAL_SPLIT))
    val_n   = int(MAX_SAMPLES * VAL_SPLIT)

    print(f"\n{'='*60}")
    print(f"  Open Images V7 — Food Detection Dataset")
    print(f"{'='*60}")
    print(f"  Target       : {MAX_SAMPLES:,} images  (~10 GB)")
    print(f"  Train / Val  : {train_n:,} / {val_n:,}")
    print(f"  Classes      : {len(FOOD_CLASSES)}")
    print(f"  Output       : {OUT_DIR}")
    print(f"{'='*60}\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dl_kwargs = dict(
        label_types=["detections"],
        classes=FOOD_CLASSES,
        only_matching=True,
        seed=SEED,
    )

    import fiftyone as fo

    # ── Download (resume-safe) ────────────────────────────────────────────────
    print("Downloading train split …")
    if "oi_food_train" in fo.list_datasets():
        print("  Resuming existing oi_food_train dataset …")
        train_ds = fo.load_dataset("oi_food_train")
    else:
        train_ds = foz.load_zoo_dataset(
            "open-images-v7", split="train",
            max_samples=train_n,
            dataset_name="oi_food_train",
            **dl_kwargs,
        )
    print(f"  Train: {len(train_ds):,} images")

    print("\nDownloading validation split …")
    if "oi_food_val" in fo.list_datasets():
        print("  Resuming existing oi_food_val dataset …")
        val_ds = fo.load_dataset("oi_food_val")
    else:
        val_ds = foz.load_zoo_dataset(
            "open-images-v7", split="validation",
            max_samples=val_n,
            dataset_name="oi_food_val",
            **dl_kwargs,
        )
    print(f"  Val: {len(val_ds):,} images")

    # ── Build class index from actual annotations ─────────────────────────────
    print("\nBuilding class index …")
    class_set: set[str] = set()
    for ds in (train_ds, val_ds):
        for sample in ds:
            if sample.ground_truth:
                for det in sample.ground_truth.detections:
                    class_set.add(det.label)

    class_names  = sorted(class_set)
    class_to_idx = {n: i for i, n in enumerate(class_names)}
    print(f"  {len(class_names)} classes found in annotations")

    # ── Export to YOLO ────────────────────────────────────────────────────────
    print("\nExporting to YOLO format …")
    train_counts, train_multi = _export_to_yolo(train_ds, "train", class_to_idx)
    val_counts,   val_multi   = _export_to_yolo(val_ds,   "val",   class_to_idx)

    # ── data.yaml ─────────────────────────────────────────────────────────────
    yaml_lines = [
        f"path: {OUT_DIR.as_posix()}",
        f"train: images/train",
        f"val:   images/val",
        f"",
        f"nc: {len(class_names)}",
        f"names:",
    ] + [f'  - "{n}"' for n in class_names]
    (OUT_DIR / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    (OUT_DIR / "classes.txt").write_text("\n".join(class_names), encoding="utf-8")

    # ── Stats ─────────────────────────────────────────────────────────────────
    all_counts = {k: train_counts[k] + val_counts.get(k, 0)
                  for k in set(train_counts) | set(val_counts)}
    total_imgs = len(train_ds) + len(val_ds)
    total_multi = train_multi + val_multi
    stats = {
        "total_images":     total_imgs,
        "train_images":     len(train_ds),
        "val_images":       len(val_ds),
        "multi_item_images": total_multi,
        "multi_item_pct":   f"{total_multi/max(total_imgs,1):.0%}",
        "num_classes":      len(class_names),
        "class_distribution": dict(sorted(all_counts.items(), key=lambda x: -x[1])),
    }
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # ── Disk usage ────────────────────────────────────────────────────────────
    total_gb = sum(f.stat().st_size for f in OUT_DIR.rglob("*") if f.is_file()) / 1e9

    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"  Total images     : {total_imgs:,}")
    print(f"  Multi-item images: {total_multi:,} ({total_multi/max(total_imgs,1):.0%})")
    print(f"  Classes          : {len(class_names)}")
    print(f"  Disk used        : {total_gb:.1f} GB")
    print(f"  YAML             : {OUT_DIR / 'data.yaml'}")
    print(f"\nTop 10 classes:")
    for cls, cnt in list(stats["class_distribution"].items())[:10]:
        print(f"  {cls:<30} {cnt:,} boxes")
    print(f"\nNext: python scripts/train_yolov8_openimages.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
