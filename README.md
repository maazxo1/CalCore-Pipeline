<div align="center">

# 🍕 Food Analysis Pipeline

**Commercial-grade food recognition, weight estimation, and nutrition analysis from a single photo**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

<br/>

*Snap a photo. Get instant weight and nutrition data — no manual logging, no guessing.*

<br/>

</div>

---

## ✨ What It Does

Point a camera at any meal and the pipeline returns the **name, estimated weight (grams), and full nutrition breakdown** for every food item in the frame — in under a second on a modern GPU.

```
📸 Photo  →  🔍 Detect  →  ✂️ Segment  →  🏷️ Classify  →  📏 Depth  →  📦 Volume  →  🥗 Nutrition
```

| Input | Output |
|-------|--------|
| Any JPEG/PNG food photo | Food names, weights (g), calories, protein, fat, carbs per item |
| Up to 3 images at once | Aggregated totals across all images |
| Optional reference object | Calibrated depth scale for higher accuracy |

---

## 🧠 Pipeline Architecture

The pipeline chains **6 specialised models**, each feeding the next:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FOOD ANALYSIS PIPELINE                       │
├──────────┬──────────┬──────────┬──────────┬──────────┬─────────────┤
│  DETECT  │ SEGMENT  │ CLASSIFY │  DEPTH   │  VOLUME  │  NUTRITION  │
│          │          │          │          │          │             │
│ YOLOv8l  │ FastSAM  │ EffNet   │DepthAny- │ Voxeli-  │ USDA API +  │
│ 205-cls  │ instance │ V2-L or  │ thing V2 │ zation + │ offline     │
│ fine-    │ mask per │ CLIP     │ Metric   │ USDA     │ taxonomy    │
│ tuned on │ detection│ fallback │ Depth    │ anchoring│ lookup      │
│ 113K img │          │          │          │          │             │
└──────────┴──────────┴──────────┴──────────┴──────────┴─────────────┘
```

### Stage Details

| Stage | Model | Role |
|-------|-------|------|
| **Detection** | YOLOv8l (205 food classes) | Localises every food item with a bounding box |
| **Segmentation** | FastSAM | Generates a pixel-precise instance mask per detection |
| **Classification** | EfficientNetV2-L (primary) + CLIP ViT-L/14 (fallback) | Identifies the exact food from the masked crop |
| **Depth** | Depth Anything V2 Metric (Hypersim, ViT-L) | Produces per-pixel depth in real-world centimetres |
| **Volume** | Voxelisation + USDA correction | Integrates depth × mask area into a 3D volume estimate |
| **Nutrition** | USDA FoodData Central API + offline cache | Maps weight → calories, protein, fat, carbs |

---

## 📊 Model Accuracy

### Detector — YOLOv8l

Trained on **113,884 images** across **205 food classes** from four combined datasets.

| Metric | Score |
|--------|-------|
| mAP@50 | **0.551** |
| mAP@50-95 | **0.405** |
| Precision | 0.585 |
| Recall | 0.515 |
| Training images | 113,884 |
| Validation images | 24,626 |
| Epochs | 30 |

### Classifier — EfficientNetV2-L

Fine-tuned on **Food-101** (101,000 images, 101 classes).

| Metric | Score |
|--------|-------|
| Validation accuracy | **90.02%** |
| Training epochs | 25 |
| Training time | ~61 min/epoch on RTX 4060 |
| Fallback | CLIP ViT-L/14 zero-shot (covers foods outside Food-101) |

> **Dual-mode classification:** EfficientNet handles the 101 Food-101 classes. If confidence is below 65%, CLIP zero-shot takes over — covering regional dishes, garnishes, and anything outside the training distribution.

---

## 🎯 Detection Samples

<div align="center">

**Validation batch — model predictions**

![Val Predictions Batch 0](runs/detect/combined/val_batch0_pred.jpg)

![Val Predictions Batch 1](runs/detect/combined/val_batch1_pred.jpg)

**Training curves**

![Results](runs/detect/combined/results.png)

**PR Curve &nbsp;&nbsp;&nbsp; F1 Curve**

<img src="runs/detect/combined/BoxPR_curve.png" width="48%"/> <img src="runs/detect/combined/BoxF1_curve.png" width="48%"/>

</div>

---

## 📦 API Response Example

```json
POST /analyze
Content-Type: multipart/form-data

{
  "success": true,
  "food_items": [
    {
      "food_name": "pizza",
      "weight_g": 285.0,
      "confidence": 0.87,
      "nutrition": {
        "calories": 760,
        "protein_g": 32.1,
        "fat_g": 28.4,
        "carbs_g": 92.0,
        "fiber_g": 3.2
      },
      "bbox": [120, 80, 540, 420]
    },
    {
      "food_name": "caesar_salad",
      "weight_g": 180.0,
      "confidence": 0.79,
      "nutrition": {
        "calories": 195,
        "protein_g": 8.2,
        "fat_g": 14.1,
        "carbs_g": 10.5
      }
    }
  ],
  "count": 2,
  "total_weight_g": 465.0,
  "total_calories": 955.0
}
```

---

## ⚡ Performance

| Hardware | Inference time (single image) |
|----------|-------------------------------|
| RTX 4060 (8 GB) | ~1.2 – 2.0 s |
| RTX 3080 (10 GB) | ~0.8 – 1.4 s |
| CPU only | ~15 – 30 s |

> FP16 autocast, batched tile inference, vectorised Soft-NMS, and single-call FastSAM prompting are all enabled by default.

---

## 🗂️ Project Structure

```
pipelines/
├── main.py                       ← Single-image pipeline (CLI)
├── api_server.py                 ← FastAPI server
│
├── core/
│   ├── food_detector.py          ← YOLOv8l detector (tiled + batched)
│   ├── segment_food.py           ← FastSAM segmenter (batch prompt)
│   ├── classify.py               ← CLIP + ViT classifier
│   ├── classify_efficientnet.py  ← EfficientNetV2-L (primary)
│   ├── estimate_depth.py         ← Depth Anything V2 (metric + FP16)
│   ├── volume_calculator.py      ← Volume from mask × depth
│   ├── volume_estimator.py       ← USDA-corrected weight
│   ├── weight_guardrails.py      ← Per-food weight bounds
│   ├── food_taxonomy.py          ← Taxonomy loader
│   └── pipeline_postprocess.py  ← Deduplication + filtering
│
├── data/
│   ├── food_taxonomy.json        ← 128-entry food database
│   ├── usda_nutrition_lookup.py  ← Offline nutrition cache
│   └── extra_clip_labels.txt    ← Regional food labels
│
├── scripts/
│   ├── train_efficientnet_food101.py
│   └── train_yolov8_combined.py
│
├── Depth-Anything-V2/            ← Depth model subpackage
├── weights/                      ← Model weights (download separately)
├── samples/                      ← Test images
├── requirements.txt
└── .env.example
```

---

## 🚀 Setup & Usage

### Requirements

- Python **3.10** or **3.11**
- NVIDIA GPU **8 GB+ VRAM** (CPU fallback available, but slow)
- CUDA 12.x drivers

---

### 1 — Clone

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2 — Virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3 — Install PyTorch (pick your CUDA version)

```bash
# CUDA 12.1
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch==2.7.1 torchvision==0.22.1
```

> Find your version at [pytorch.org/get-started](https://pytorch.org/get-started/locally/)

### 4 — Install dependencies

```bash
pip install -r requirements.txt
pip install -e Depth-Anything-V2
```

### 5 — Environment variables

```bash
cp .env.example .env
# Open .env and set your USDA_API_KEY
# Free key: https://fdc.nal.usda.gov/api-key-signup
```

### 6 — Download model weights

#### 🎁 Pre-trained weights (one command)

The custom-trained YOLO and EfficientNet weights are provided as a GitHub release so you don't need to train anything yourself:

```bash
python scripts/download_weights.py
```

This downloads:

| File | Size | Description |
|------|------|-------------|
| `weights/yolo.pt` | ~84 MB | YOLOv8l fine-tuned on 205-class food dataset (113K images) |
| `weights/efficientnet_food101/best.pth` | ~451 MB | EfficientNetV2-L trained on Food-101 — **90.02% val accuracy** |

#### Third-party weights (manual download)

These are not ours to redistribute — grab them from the official sources:

| File | Source |
|------|--------|
| `weights/FastSAM.pt` | [Ultralytics Assets](https://github.com/ultralytics/assets/releases) |
| `weights/depth_anything_v2_metric_hypersim_vitl.pth` | [Depth-Anything-V2 Releases](https://github.com/DepthAnything/Depth-Anything-V2/releases) |
| `weights/depth_anything_v2_large.pth` | [Depth-Anything-V2 Releases](https://github.com/DepthAnything/Depth-Anything-V2/releases) *(optional fallback)* |

> **Graceful degradation:** no `yolo.pt` → uses pretrained YOLOv8l · no `efficientnet` → uses CLIP zero-shot · no metric depth → uses relative depth.

---

### Run — CLI

```bash
python main.py samples/food.jpg
```

Saves annotated output to `output_final.jpg` and prints results.

---

### Run — API server

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

**Analyse a single image:**
```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@your_meal.jpg"
```

**Analyse multiple images (max 3):**
```bash
curl -X POST http://localhost:8000/analyze_multi \
  -F "files=@plate1.jpg" \
  -F "files=@plate2.jpg"
```

**With a reference object for calibrated depth:**
```bash
curl -X POST "http://localhost:8000/analyze?reference_type=credit_card" \
  -F "file=@your_meal.jpg"
```

Swagger docs at **http://localhost:8000/docs**

---

### Run — Tests

```bash
python -m pytest tests/ -v
```

---

### Train the classifier (optional)

Downloads Food-101 automatically from HuggingFace. Resumes from last checkpoint.

```bash
python -u scripts/train_efficientnet_food101.py
```

---

## ⚙️ Key Configuration

All settings are in `.env`. Defaults are tuned for an RTX 4060 8 GB.

| Variable | Default | Description |
|----------|---------|-------------|
| `USDA_API_KEY` | — | Required for live nutrition lookup |
| `CLASSIFIER_BACKEND` | `auto` | `auto` / `efficientnet` / `clip` |
| `GPU_CONCURRENCY` | `1` | Parallel GPU requests (raise to `2` if > 12 GB VRAM) |
| `YOLO_CONF` | `0.10` | Detection confidence threshold |
| `FASTSAM_IMGSZ` | `640` | Segmentation input size (keep at 640 on 8 GB) |
| `EFFNET_FALLBACK_THRESHOLD` | `0.65` | Below this, CLIP fallback is triggered |

---

<div align="center">

Built with YOLOv8 · FastSAM · EfficientNet · CLIP · Depth Anything V2 · USDA FoodData Central

</div>
#   C a l C o r e - P i p e l i n e  
 