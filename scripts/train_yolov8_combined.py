"""
Combined Food Detection Training Script
========================================
Merges 4 food datasets into one unified training run.

Datasets merged:
  - training_data/complete food/                     (32,723 images, 214 classes)
  - training_data/allinone.v1i.yolov8/               (29,913 images,  68 classes)
  - training_data/Food Detection Dataset.v1i.yolov8/ ( 6,248 images,  77 classes)
  - training_data/openimages_food/                   (45,000 images,  55 classes)
  Total: ~113k train images, ~280 unique classes after deduplication

Strategy:
  1. Normalize all class names (lowercase, strip, remove parentheticals)
  2. Build master class list (union, deduplicated)
  3. Remap every label file's class IDs to master IDs
  4. Hard-link images into cache (zero extra disk space)
  5. One combined data.yaml → one training run

Usage:
    python scripts/train_yolov8_combined.py

Resume:
    Re-run same command — resumes from last.pt automatically.
    NOTE: If you add Open Images AFTER the cache was built, delete
          training_data/combined_cache/ first to force a rebuild.

Outputs:
    weights/yoloo.pt                   — final pipeline weight
    training_data/combined_cache/      — merged dataset cache
    runs/detect/combined/weights/best.pt
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).parent
ROOT_DIR    = SCRIPT_DIR.parent
CACHE_DIR   = ROOT_DIR / "training_data" / "combined_cache"
WEIGHTS_DIR = ROOT_DIR / "weights"

BASE_WEIGHTS = WEIGHTS_DIR / "yolo.pt"     # COCO pretrained YOLOv8l
OUTPUT_PT    = WEIGHTS_DIR / "yoloo.pt"

RUN_NAME = "combined"
RUNS_DIR = ROOT_DIR / "runs" / "detect" / RUN_NAME

EPOCHS  = int(os.getenv("COMB_EPOCHS",  "30"))
BATCH   = int(os.getenv("COMB_BATCH",    "8"))
IMGSZ   = int(os.getenv("COMB_IMGSZ",  "640"))
LR0     = float(os.getenv("COMB_LR0",  "0.003"))
WORKERS = int(os.getenv("COMB_WORKERS",  "2"))

# Roboflow datasets — (folder_name, val_subdir)
# val_subdir: the folder name used for validation split
DATASETS = [
    ("complete food",                       "valid"),
    ("allinone.v1i.yolov8",                 "valid"),
    ("Food Detection Dataset.v1i.yolov8",   "valid"),
]

# Open Images food dataset (optional — only included if downloaded)
# Run: python scripts/download_openimages_food.py  first
OI_DIR = ROOT_DIR / "training_data" / "openimages_food"

# Minimum number of boxes a class must have across the whole dataset to be kept.
# Classes below this threshold are too rare to learn reliably.
MIN_CLASS_BOXES = 30

# ---------------------------------------------------------------------------
# Class merge map — normalized_name → canonical_name
# Merges visually identical / near-identical classes to reduce confusion
# ---------------------------------------------------------------------------
MERGE_MAP: dict[str, str] = {
    # Pasta / noodle variants → pasta
    "spaghetti":                "pasta",
    "noodles":                  "pasta",
    "oil pasta":                "pasta",
    "pasta al ragu":            "pasta",
    "pasta with carbonara sauce": "pasta",
    "fried-noodle":             "pasta",
    "ramen-noodle":             "pasta",
    "soba-noodle":              "pasta",
    "udon-noodle":              "pasta",
    "tensin-noodle":            "pasta",
    "beef-noodle":              "pasta",
    # Pizza variants → pizza
    "pizza_full":               "pizza",
    "pizza_half":               "pizza",
    "pizza_slice":              "pizza",
    "middle east pizza":        "pizza",
    # Burger variants → burger
    "hamburger":                "burger",
    "middle east burger":       "burger",
    # Sandwich variants → sandwich
    "sandwiches":               "sandwich",
    "chip-butty":               "sandwich",
    # Salad variants → salad
    "middle east salad":        "salad",
    "papaya salad":             "salad",
    # Donut variants → donut
    "donuts":                   "donut",
    "doughnut":                 "donut",
    # Omelette spelling variants → omelette
    "omelet":                   "omelette",
    # Egg variants → egg
    "fried_eggs":               "egg",
    # Eggplant variants → eggplant
    "fried eggplant":           "eggplant",
    "grilled-eggplant":         "eggplant",
    # Chicken variants → chicken
    "grilled chicken":          "chicken",
    "fried_chicken":            "chicken",
    "chicken roast":            "chicken",
    # Salmon variants → salmon
    "grilled-salmon":           "salmon",
    "salmon-meuniere":          "salmon",
    # Soup variants → soup
    "middle east soup":         "soup",
    "miso-soup":                "soup",
    "potage":                   "soup",
    "tom yum":                  "soup",
    # Rice variants → rice
    "fried-rice":               "rice",
    "basil rice":               "rice",
    "chicken rice":             "rice",
    # Cabbage duplicate (lowercase)
    "cabbage":                  "cabbage",
    # Corn duplicate
    "corn":                     "corn",
    # Peas duplicate
    "peas":                     "peas",
    # Chips / fries → french fries
    "chips":                    "french fries",
    # Hot dog normalisation
    "hot dog":                  "hot dog",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_name(name: str) -> str:
    """
    Normalize class name for deduplication:
      - lowercase + strip whitespace
      - remove parenthetical qualifiers e.g. 'Egg (Food)' -> 'egg'
      - collapse internal whitespace
      - apply MERGE_MAP aliases
    """
    name = name.lower().strip()
    name = re.sub(r"\s*\(.*?\)\s*", " ", name)   # remove (Food), (Drink) etc.
    name = re.sub(r"\s+", " ", name).strip()
    return MERGE_MAP.get(name, name)              # apply merge alias


def parse_yaml_names(yaml_path: Path) -> list[str]:
    """Extract class names from any Roboflow or OI YOLO data.yaml."""
    text = yaml_path.read_text(encoding="utf-8")

    # Format 1: names: ['Apple', 'Banana', ...]  (Roboflow single-line)
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("names:"):
            val = line[len("names:"):].strip()
            if val.startswith("["):
                names = re.findall(r"'([^']+)'|\"([^\"]+)\"", val)
                return [a or b for a, b in names]
            break   # multi-line format below

    # Format 2: names:\n  - "Apple"\n  - "Banana"  (Open Images YAML list)
    names = re.findall(r'^\s*-\s+"([^"]+)"', text, re.MULTILINE)
    if names:
        return names

    # Format 3:  - Apple  (unquoted list items)
    names = re.findall(r"^\s*-\s+(.+)$", text, re.MULTILINE)
    if names:
        return [n.strip().strip("'\"") for n in names]

    return []


# ---------------------------------------------------------------------------
# Step 1 — Build master class list
# ---------------------------------------------------------------------------

def build_master_classes() -> tuple[list[str], dict[str, dict[int, int]]]:
    """
    Returns:
        master_names  : ordered list of canonical class names
        remap_tables  : {dataset_key: {local_id: master_id}}
    """
    master_norm: list[str] = []          # normalized names (dedup key)
    master_raw:  list[str] = []          # display names (first-seen wins)
    norm_to_id:  dict[str, int] = {}     # O(1) lookup

    remap_tables: dict[str, dict[int, int]] = {}

    # Collect all sources: Roboflow + Open Images (if present)
    all_sources: list[tuple[str, str | None]] = [
        (ds_folder, val_dir) for ds_folder, val_dir in DATASETS
    ]

    oi_yaml = OI_DIR / "data.yaml"
    if oi_yaml.exists():
        all_sources.append(("__openimages__", "val"))
        print(f"  Open Images found — including {OI_DIR.name}")
    else:
        print(f"  Open Images not found — skipping (run download_openimages_food.py to include)")

    for ds_key, _ in all_sources:
        yaml_path = oi_yaml if ds_key == "__openimages__" else \
                    ROOT_DIR / "training_data" / ds_key / "data.yaml"

        if not yaml_path.exists():
            print(f"  [WARN] data.yaml not found: {yaml_path} — skipping dataset")
            continue

        names = parse_yaml_names(yaml_path)
        if not names:
            print(f"  [WARN] Could not parse class names from {yaml_path} — skipping")
            continue

        remap: dict[int, int] = {}
        for local_id, raw_name in enumerate(names):
            norm = normalize_name(raw_name)
            if norm in norm_to_id:
                master_id = norm_to_id[norm]
            else:
                master_id = len(master_raw)
                master_norm.append(norm)
                master_raw.append(raw_name)
                norm_to_id[norm] = master_id
            remap[local_id] = master_id

        remap_tables[ds_key] = remap
        label = "openimages_food" if ds_key == "__openimages__" else ds_key
        print(f"  {label[:45]:45s}: {len(names):3d} classes → {len(remap)} remaps")

    print(f"\n  Master class list: {len(master_raw)} unique classes")
    return master_raw, remap_tables


# ---------------------------------------------------------------------------
# Step 2 — Build cache
# ---------------------------------------------------------------------------

def _link(src: Path, dst: Path):
    """Hard-link src → dst, fall back to copy. Skip if dst exists."""
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_cache(master_names: list[str],
                remap_tables: dict[str, dict[int, int]]) -> Path:
    """Hard-link images + write remapped labels into CACHE_DIR."""
    data_yaml = CACHE_DIR / "data.yaml"

    if data_yaml.exists():
        print(f"  Cache already exists at {CACHE_DIR}")
        print(f"  Delete that folder to force a rebuild (e.g. if you added Open Images).")
        return data_yaml

    print(f"  Building combined cache in {CACHE_DIR} …")
    t0 = time.time()

    total_images = 0
    total_labels = 0

    all_sources: list[tuple[str, str | None]] = [
        (ds_folder, val_dir) for ds_folder, val_dir in DATASETS
    ]
    if OI_DIR.exists():
        all_sources.append(("__openimages__", "val"))

    for ds_key, val_subdir in all_sources:
        is_oi = ds_key == "__openimages__"
        ds_path = OI_DIR if is_oi else ROOT_DIR / "training_data" / ds_key
        remap   = remap_tables.get(ds_key, {})

        # Determine which splits exist
        splits_to_process: list[tuple[str, Path, Path]] = []  # (cache_split, src_img, src_lbl)

        # Train split
        if is_oi:
            src_img_train = ds_path / "images" / "train"
            src_lbl_train = ds_path / "labels" / "train"
        else:
            src_img_train = ds_path / "train" / "images"
            src_lbl_train = ds_path / "train" / "labels"

        if src_img_train.exists():
            splits_to_process.append(("train", src_img_train, src_lbl_train))

        # Val split
        if val_subdir:
            if is_oi:
                src_img_val = ds_path / "images" / "val"
                src_lbl_val = ds_path / "labels" / "val"
            else:
                src_img_val = ds_path / val_subdir / "images"
                src_lbl_val = ds_path / val_subdir / "labels"

            if src_img_val.exists():
                splits_to_process.append(("val", src_img_val, src_lbl_val))

        if not splits_to_process:
            print(f"  [WARN] No valid splits found for {ds_key} — skipping")
            continue

        for cache_split, src_img, src_lbl in splits_to_process:
            dst_img = CACHE_DIR / "images" / cache_split
            dst_lbl = CACHE_DIR / "labels" / cache_split
            dst_img.mkdir(parents=True, exist_ok=True)
            dst_lbl.mkdir(parents=True, exist_ok=True)

            imgs = 0
            lbls = 0
            skipped_lbl = 0

            for img_path in src_img.iterdir():
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue

                _link(img_path, dst_img / img_path.name)
                imgs += 1

                lbl_src = src_lbl / (img_path.stem + ".txt")
                lbl_dst = dst_lbl / (img_path.stem + ".txt")

                if lbl_dst.exists():
                    lbls += 1
                    continue

                if not lbl_src.exists():
                    # Write empty label so YOLO doesn't complain
                    lbl_dst.write_text("", encoding="utf-8")
                    skipped_lbl += 1
                    continue

                lines_out = []
                for line in lbl_src.read_text(encoding="utf-8").splitlines():
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    local_id = int(parts[0])
                    master_id = remap.get(local_id)
                    if master_id is None:
                        continue  # unknown class — skip box
                    # Only take first 4 coords (handles 6-col FPB labels too)
                    try:
                        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    except ValueError:
                        continue
                    # Filter noisy boxes: must be within [0,1] and sane aspect ratio
                    if not (0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                        continue
                    if not (0.0 < cx <= 1.0 and 0.0 < cy <= 1.0):
                        continue
                    aspect = bw / bh if bh > 0 else 99
                    if aspect > 20 or aspect < 0.05:   # skip absurd boxes
                        continue
                    lines_out.append(f"{master_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                lbl_dst.write_text("\n".join(lines_out) + ("\n" if lines_out else ""),
                                   encoding="utf-8")
                lbls += 1

            total_images += imgs
            total_labels += lbls
            label = "openimages_food" if is_oi else ds_key
            warn  = f"  ({skipped_lbl} missing lbls)" if skipped_lbl else ""
            print(f"  {label[:35]:35s} {cache_split:5s}: "
                  f"{imgs:6,} images  {lbls:6,} labels{warn}")

    # ── Remove rare classes (< MIN_CLASS_BOXES boxes total) ──────────────────
    print(f"\n  Counting boxes per class (threshold: {MIN_CLASS_BOXES}) …")
    box_counts: dict[int, int] = {}
    for split in ["train", "val"]:
        lbl_dir = CACHE_DIR / "labels" / split
        if not lbl_dir.exists():
            continue
        for lbl_file in lbl_dir.iterdir():
            for line in lbl_file.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    box_counts[cid] = box_counts.get(cid, 0) + 1

    rare_ids = {cid for cid, cnt in box_counts.items() if cnt < MIN_CLASS_BOXES}
    if rare_ids:
        rare_names = [master_names[i] for i in sorted(rare_ids) if i < len(master_names)]
        print(f"  Dropping {len(rare_ids)} rare classes: {rare_names[:10]}{'...' if len(rare_names)>10 else ''}")
        # Rewrite label files removing rare class boxes
        removed_boxes = 0
        for split in ["train", "val"]:
            lbl_dir = CACHE_DIR / "labels" / split
            if not lbl_dir.exists():
                continue
            for lbl_file in lbl_dir.iterdir():
                lines = lbl_file.read_text(encoding="utf-8").splitlines()
                kept = [l for l in lines if l.strip() and int(l.split()[0]) not in rare_ids]
                if len(kept) != len([l for l in lines if l.strip()]):
                    removed_boxes += len(lines) - len(kept)
                    lbl_file.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        # Rebuild master_names without rare classes, remap IDs sequentially
        old_to_new: dict[int, int] = {}
        new_master: list[str] = []
        for old_id, name in enumerate(master_names):
            if old_id not in rare_ids:
                old_to_new[old_id] = len(new_master)
                new_master.append(name)
        master_names = new_master
        # Remap class IDs in all label files
        for split in ["train", "val"]:
            lbl_dir = CACHE_DIR / "labels" / split
            if not lbl_dir.exists():
                continue
            for lbl_file in lbl_dir.iterdir():
                lines = lbl_file.read_text(encoding="utf-8").splitlines()
                new_lines = []
                for l in lines:
                    parts = l.strip().split()
                    if len(parts) >= 5:
                        new_id = old_to_new.get(int(parts[0]))
                        if new_id is not None:
                            new_lines.append(f"{new_id} {' '.join(parts[1:])}")
                lbl_file.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")
        print(f"  Removed {removed_boxes:,} rare-class boxes. Final classes: {len(master_names)}")
    else:
        print(f"  All {len(master_names)} classes have ≥ {MIN_CLASS_BOXES} boxes. Nothing dropped.")

    # Write master data.yaml
    names_str = "[" + ", ".join(f"'{n}'" for n in master_names) + "]"
    yaml_content = (
        f"path: {CACHE_DIR.as_posix()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"\n"
        f"nc: {len(master_names)}\n"
        f"names: {names_str}\n"
    )
    data_yaml.write_text(yaml_content, encoding="utf-8")

    elapsed = time.time() - t0
    print(f"\n  Cache built in {elapsed:.1f}s")
    print(f"  Total: {total_images:,} images  {total_labels:,} labels")
    print(f"  Classes: {len(master_names)}")
    return data_yaml


# ---------------------------------------------------------------------------
# Step 3 — Train
# ---------------------------------------------------------------------------

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*65}")
    print(f"  Combined Food Detection Training")
    print(f"{'='*65}")
    if device == "cuda":
        print(f"  Device  : {torch.cuda.get_device_name(0)}")
    print(f"  Base    : {BASE_WEIGHTS}")
    print(f"  Epochs  : {EPOCHS}  |  Batch: {BATCH}  |  ImgSz: {IMGSZ}  |  LR0: {LR0}")
    print(f"  Output  : {OUTPUT_PT}")
    print(f"{'='*65}\n")

    if not BASE_WEIGHTS.exists():
        print(f"ERROR: Base weights not found at {BASE_WEIGHTS}")
        sys.exit(1)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Build master class list
    print("Step 1 — Building master class list …")
    master_names, remap_tables = build_master_classes()

    # Step 2: Build cache
    print("\nStep 2 — Building dataset cache …")
    data_yaml = build_cache(master_names, remap_tables)

    # Step 3: Train
    last_pt  = RUNS_DIR / "weights" / "last.pt"
    resuming = last_pt.exists()

    if resuming:
        print(f"\nResuming from {last_pt}")
        model = YOLO(str(last_pt))
    else:
        print(f"\nTraining from {BASE_WEIGHTS}")
        model = YOLO(str(BASE_WEIGHTS))

    n_train = sum(1 for _ in (CACHE_DIR / "images" / "train").iterdir()) \
              if (CACHE_DIR / "images" / "train").exists() else 0
    print(f"Starting training ({len(master_names)} classes, {n_train:,} train images) …\n")

    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        optimizer="SGD",

        lr0=LR0,
        lrf=0.01,
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=5,
        warmup_momentum=0.8,

        box=7.5,
        cls=1.5,
        dfl=1.5,
        close_mosaic=10,

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.0,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        scale=0.5,
        translate=0.1,

        freeze=0,
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

    best_pt = RUNS_DIR / "weights" / "best.pt"
    if best_pt.exists():
        shutil.copy2(best_pt, OUTPUT_PT)
        print(f"\n{'='*65}")
        print(f"  Training complete!")
        print(f"  Pipeline weight : {OUTPUT_PT}")
        print(f"  Best checkpoint : {best_pt}")
        print(f"{'='*65}")
    else:
        print(f"\n[WARN] best.pt not found at {best_pt}")


if __name__ == "__main__":
    train()
