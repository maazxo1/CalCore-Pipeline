# Setup Guide — CalCore Pipeline

Complete step-by-step instructions for getting the pipeline running from scratch.

> **TL;DR** — Python 3.10/3.11 + CUDA GPU → clone → install → download weights → run.

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Clone the Repository](#2-clone-the-repository)
3. [Python Environment](#3-python-environment)
4. [Install PyTorch](#4-install-pytorch)
5. [Install Dependencies](#5-install-dependencies)
6. [Environment Variables](#6-environment-variables)
7. [Download Model Weights](#7-download-model-weights)
8. [Run the Pipeline](#8-run-the-pipeline)
9. [Run the API Server](#9-run-the-api-server)
10. [Verify Everything Works](#10-verify-everything-works)
11. [Training Your Own Models](#11-training-your-own-models-optional)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| OS | Windows 10, Ubuntu 20.04, macOS 12 | Windows 11 / Ubuntu 22.04 |
| Python | 3.10 | 3.11 |
| GPU VRAM | 6 GB (reduced settings) | 8 GB+ |
| RAM | 12 GB | 16 GB+ |
| Disk space | 4 GB (weights + deps) | 8 GB |
| CUDA | 11.8 | 12.1 or 12.4 |

> **No GPU?** The pipeline runs on CPU but expect ~15–30 s per image instead of ~1–2 s.

---

## 2. Clone the Repository

The repo uses a git submodule for Depth-Anything-V2. Use `--recurse-submodules` so it's pulled automatically:

```bash
git clone --recurse-submodules https://github.com/maazxo1/CalCore-Pipeline.git
cd CalCore-Pipeline
```

If you already cloned without the flag, run:

```bash
git submodule update --init --recursive
```

---

## 3. Python Environment

Always use a virtual environment to avoid dependency conflicts.

```bash
python -m venv venv
```

**Activate it:**

```bash
# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Windows (Command Prompt)
venv\Scripts\activate.bat

# Linux / macOS
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

---

## 4. Install PyTorch

PyTorch must be installed separately because the correct version depends on your CUDA driver.

**Check your CUDA version:**
```bash
nvidia-smi
```
Look for `CUDA Version: XX.X` in the top-right corner.

**Then install the matching build:**

```bash
# CUDA 12.1
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu124

# CUDA 11.8
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118

# CPU only (slow — not recommended for production)
pip install torch==2.7.1 torchvision==0.22.1
```

**Verify PyTorch sees your GPU:**

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected output:
```
True
NVIDIA GeForce RTX 4060
```

If you get `False`, your CUDA build doesn't match your driver — try a different `--index-url` from the [PyTorch website](https://pytorch.org/get-started/locally/).

---

## 5. Install Dependencies

```bash
# Main dependencies (FastAPI, OpenCV, Ultralytics, Transformers, timm, etc.)
pip install -r requirements.txt

# Depth Anything V2 (installs the depth_anything_v2 Python package)
pip install -e Depth-Anything-V2
```

> The `segment-anything` package installs directly from GitHub — this requires internet access and `git` to be on your PATH.

---

## 6. Environment Variables

Copy the example file and edit it:

```bash
# Linux / macOS
cp .env.example .env

# Windows
copy .env.example .env
```

Open `.env` and set your **USDA API key** — this enables live nutrition lookup. Without it the pipeline uses the offline cache (covers most common foods):

```
USDA_API_KEY=your_key_here
```

Get a free key at: https://fdc.nal.usda.gov/api-key-signup (instant, no credit card)

**All other settings have sensible defaults** — you don't need to change anything else to get started. See the [Configuration section in README](README.md#%EF%B8%8F-configuration) for the full list.

---

## 7. Download Model Weights

### Step 1 — Our pre-trained weights (automatic)

Run the download script. It pulls our trained YOLO and EfficientNet checkpoints from the GitHub release (~535 MB total):

```bash
python scripts/download_weights.py
```

Expected output:
```
============================================================
  Food Pipeline — weight downloader
============================================================

  ↓ weights/yolo.pt  (~84 MB)
  [████████████████████] 100%

  ↓ weights/efficientnet_food101/best.pth  (~451 MB)
  [████████████████████] 100%

  ✓ weights/efficientnet_food101/labels.txt already present — skipping

✅ Done.
```

### Step 2 — Third-party weights (manual)

These belong to their respective projects and must be downloaded manually. Place them directly in the `weights/` folder:

#### FastSAM

1. Go to https://github.com/ultralytics/assets/releases
2. Download `FastSAM.pt`
3. Save as `weights/FastSAM.pt`

#### Depth Anything V2 — Metric (required)

1. Go to https://github.com/DepthAnything/Depth-Anything-V2/releases
2. Download `depth_anything_v2_metric_hypersim_vitl.pth`
3. Save as `weights/depth_anything_v2_metric_hypersim_vitl.pth`

#### Depth Anything V2 — Relative (optional fallback)

1. Same releases page
2. Download `depth_anything_v2_vitl.pth`
3. Save as `weights/depth_anything_v2_large.pth`

### Final weights folder structure

```
weights/
├── yolo.pt                                        ← from download script
├── FastSAM.pt                                     ← manual
├── depth_anything_v2_metric_hypersim_vitl.pth     ← manual
├── depth_anything_v2_large.pth                    ← manual (optional)
└── efficientnet_food101/
    ├── best.pth                                   ← from download script
    └── labels.txt                                 ← from download script
```

---

## 8. Run the Pipeline

### CLI — single image

```bash
python main.py samples/food.jpg
```

Output is printed to the console and saved as `output_final.jpg`:

```
Processing: samples/food.jpg

 Step 1: Detecting food items…
   Found 2 region(s)

 Step 2–4: Segment / Classify / Depth…
   [1/2] pizza  →  87% confidence
   [2/2] caesar_salad  →  79% confidence

 Step 5–6: Volume / Nutrition…
   pizza        285g  |  760 kcal  |  P:32g  F:28g  C:92g
   caesar_salad  180g  |  195 kcal  |  P:8g   F:14g  C:11g

Total: 465g  |  955 kcal
```

**Options:**

```bash
# Custom output path
python main.py samples/food.jpg --output my_result.jpg

# Adjust detection sensitivity (lower = more detections)
python main.py samples/food.jpg --detection-conf 0.08

# Use a credit card in the photo for calibrated depth
python main.py samples/food.jpg --reference-type credit_card
```

---

## 9. Run the API Server

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

The server loads all models on startup (~10–20 s). Once you see:

```
All models loaded.
INFO:     Application startup complete.
```

it's ready. Visit **http://localhost:8000/docs** for the interactive Swagger UI.

**Test with curl:**

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@samples/food.jpg" | python -m json.tool
```

**Test with Python:**

```python
import requests

with open("samples/food.jpg", "rb") as f:
    r = requests.post("http://localhost:8000/analyze", files={"file": f})

data = r.json()
for item in data["food_items"]:
    print(f"{item['food_name']:20s}  {item['weight_g']:6.0f}g  {item['nutrition']['calories']:.0f} kcal")
```

**Production deployment** (multiple workers, no reload):

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 1
```

> Keep `--workers 1` — multiple workers each load all models into VRAM, which will OOM on most GPUs.

---

## 10. Verify Everything Works

Run the test suite:

```bash
python -m pytest tests/ -v
```

All tests should pass. If any fail, check the [Troubleshooting](#12-troubleshooting) section.

---

## 11. Training Your Own Models (optional)

### EfficientNet classifier

Downloads Food-101 automatically from HuggingFace (~3 GB). Resumes from checkpoint if interrupted.

```bash
python -u scripts/train_efficientnet_food101.py
```

- Outputs: `weights/efficientnet_food101/best.pth` + `labels.txt`
- ~61 min/epoch on RTX 4060 · 25 epochs · 90% val accuracy

### YOLO detector

See `scripts/train_yolov8_combined.py`. Requires the combined food detection datasets (~113K images).

---

## 12. Troubleshooting

### `CUDA out of memory`

Reduce input sizes in `.env`:

```env
FASTSAM_IMGSZ=480     # was 640
YOLO_IMGSZ=640        # was 800
```

Or set `YOLO_TILED=false` to disable tiled inference.

### `No module named 'depth_anything_v2'`

The submodule wasn't installed:

```bash
pip install -e Depth-Anything-V2
```

### Submodule folder is empty after cloning

```bash
git submodule update --init --recursive
```

### `ModuleNotFoundError: No module named 'clip'` or `timm`

```bash
pip install timm
# CLIP is loaded via HuggingFace transformers — already in requirements.txt
```

### Weights download fails / slow

Download manually from the [v1.0 release](https://github.com/maazxo1/CalCore-Pipeline/releases/tag/v1.0) and place files in `weights/`.

### `PIL.UnidentifiedImageError`

The image file is corrupt or not a valid JPEG/PNG. Convert it first:

```bash
python -c "from PIL import Image; Image.open('bad.jpg').save('fixed.jpg')"
```

### API returns `No food detected`

- Try lowering the detection threshold: add `?detection_conf=0.06` to the URL
- Make sure the image is well-lit and the food fills a reasonable portion of the frame
- Check the image isn't being flagged as blurry (look for `"blur_score"` in the response)

---

> Still stuck? Open an issue at https://github.com/maazxo1/CalCore-Pipeline/issues
