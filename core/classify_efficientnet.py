"""
EfficientNetV2-L Food Classifier
==================================
Drop-in replacement for CLIP FoodClassifier.
Same public interface: classify(image, bbox) -> dict.

Usage in main.py:
    Set CLASSIFIER_BACKEND=efficientnet (env var)
    Optionally set:
        EFFNET_WEIGHTS  path to fine-tuned .pth  (required for food accuracy)
        EFFNET_LABELS   path to labels .txt       (one class per line)
        EFFNET_DEVICE   auto|cuda|cpu|mps
        EFFNET_TOPK     int (default 5)

Without fine-tuned weights the model uses ImageNet-21k pretrained backbone
which has poor food specificity — provide weights for production use.

Fine-tuning:
    See train_efficientnet_food() at the bottom of this file.
    Dataset: Food-2k (https://github.com/IAIDL/RESORT) or Food-500/UEC-256.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from PIL import Image


try:
    import timm
    from timm.data import create_transform, resolve_data_config
    _TIMM_OK = True
except ImportError:
    _TIMM_OK = False

try:
    import torchvision.models as _tv_models
    from torchvision import transforms as _tv_tfm
    _TV_OK = True
except ImportError:
    _TV_OK = False


_MIN_CROP_PX = 32


def _safe_crop(image: Image.Image, bbox: Sequence[Any] | None) -> Image.Image:
    if bbox is None or len(bbox) != 4:
        return image
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    except (TypeError, ValueError):
        return image
    W, H = image.size
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if (x2 - x1) < _MIN_CROP_PX or (y2 - y1) < _MIN_CROP_PX:
        return image
    return image.crop((x1, y1, x2, y2))


class EfficientNetClassifier:
    """EfficientNetV2-L food classifier — drop-in for CLIP FoodClassifier."""

    _TIMM_MODEL = "tf_efficientnetv2_l.in21k"

    def __init__(
        self,
        weights_path: str | None = None,
        labels_path: str | None = None,
        num_classes: int = 2000,
        device: str = "auto",
    ):
        if not _TIMM_OK and not _TV_OK:
            raise ImportError("Install timm: pip install timm")

        # Device
        dev_req = os.getenv("EFFNET_DEVICE", device).lower()
        if dev_req == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(dev_req)

        self.topk = int(os.getenv("EFFNET_TOPK", "5"))
        self.conf_min = float(os.getenv("EFFNET_CONF_MIN", "0.05"))

        # Temperature scaling — divides logits before softmax.
        # Modern deep networks are overconfident; T > 1 softens the distribution
        # so that confidence scores reflect actual accuracy (calibration).
        # Fit T on your val set with: python scripts/calibrate_temperature.py
        # Typical Food-101 value after training: T ≈ 1.5–2.0
        # Default 1.0 = no change (safe until calibrated).
        self.temperature = float(os.getenv("EFFNET_TEMPERATURE", "1.0"))

        # Selective TTA — only applied when top-1 confidence falls in the
        # "gray zone" where the model is uncertain.  Outside this range TTA
        # adds latency with no meaningful gain.
        self._tta_low  = float(os.getenv("EFFNET_TTA_LOW",  "0.50"))
        self._tta_high = float(os.getenv("EFFNET_TTA_HIGH", "0.72"))

        # Labels
        self.labels: List[str] = self._load_labels(labels_path, num_classes)
        self.num_classes = len(self.labels)

        # Model
        print(f"Loading EfficientNetV2-L ({self.num_classes} classes) on {self.device} …")
        self.model = self._build_model(weights_path, self.num_classes)
        self.model.to(self.device).eval()

        # Transform
        self.transform = self._build_transform()
        print(f"  EfficientNetV2-L ready — {self.num_classes} labels.")

    # ------------------------------------------------------------------
    # Public interface (same contract as FoodClassifier)
    # ------------------------------------------------------------------

    def classify_batch(self, images: List[Image.Image]) -> List[Dict]:
        """Classify a list of images in a single GPU forward pass.

        Each image should already be cropped/masked — pass the full image as-is
        (no internal bbox cropping is done).
        Temperature scaling is applied; TTA is NOT used in batch mode (the
        per-item gray-zone check requires individual confidence scores first).
        """
        if not images:
            return []
        tensors = torch.stack([
            self.transform(img.convert("RGB")) for img in images
        ]).to(self.device)
        with torch.inference_mode():
            with torch.amp.autocast(str(self.device), enabled=(self.device.type == "cuda")):
                logits = self.model(tensors)
            probs = F.softmax(logits / self.temperature, dim=-1)
        k = min(self.topk, self.num_classes)
        top_probs_all, top_idxs_all = torch.topk(probs, k, dim=-1)
        results = []
        for i in range(len(images)):
            top_list = [
                {"label": self.labels[int(top_idxs_all[i, j])], "confidence": float(top_probs_all[i, j])}
                for j in range(k)
            ]
            results.append({
                "food_name":  top_list[0]["label"],
                "confidence": top_list[0]["confidence"],
                "top3":       top_list[:3],
            })
        return results

    def classify(
        self,
        image: Image.Image | str,
        bbox: Sequence[Any] | None = None,
    ) -> Dict:
        if isinstance(image, str):
            image = Image.open(image)
        image = image.convert("RGB")
        crop = _safe_crop(image, bbox)

        x = self.transform(crop).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            with torch.amp.autocast(str(self.device), enabled=(self.device.type == "cuda")):
                logits = self.model(x)
            probs = F.softmax(logits[0] / self.temperature, dim=-1)

        k = min(self.topk, self.num_classes)
        top_probs, top_idxs = torch.topk(probs, k)
        top_list = [
            {"label": self.labels[int(i)], "confidence": float(p)}
            for p, i in zip(top_probs, top_idxs)
        ]
        result = {
            "food_name":  top_list[0]["label"],
            "confidence": top_list[0]["confidence"],
            "top3":       top_list[:3],
        }

        # Selective TTA: only re-run with augmentations when confidence is in
        # the gray zone [_tta_low, _tta_high].  Clear wins and clear fails don't
        # benefit, but uncertain crops (the hardest cases) gain +0.5–1.5%.
        conf = result["confidence"]
        if self._tta_low <= conf <= self._tta_high:
            result = self._classify_with_tta(crop)

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _tta_augments(self, img: Image.Image) -> List[Image.Image]:
        """5 lightweight augmentations for gray-zone selective TTA."""
        from PIL import ImageEnhance
        w, h = img.size
        variants = [
            img,                                                          # original
            img.transpose(Image.FLIP_LEFT_RIGHT),                        # horizontal flip
            ImageEnhance.Brightness(img).enhance(1.10),                  # +10% brightness
            ImageEnhance.Brightness(img).enhance(0.90),                  # -10% brightness
        ]
        # 90% center crop → resize back (simulates slight scale variation)
        cw, ch = int(w * 0.90), int(h * 0.90)
        left, top = (w - cw) // 2, (h - ch) // 2
        variants.append(
            img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BILINEAR)
        )
        return variants

    def _classify_with_tta(self, crop: Image.Image) -> Dict:
        """Average softmax probabilities over 5 augmented versions of crop."""
        augments = self._tta_augments(crop)
        tensors = torch.stack([
            self.transform(aug) for aug in augments
        ]).to(self.device)

        with torch.inference_mode():
            with torch.amp.autocast(str(self.device), enabled=(self.device.type == "cuda")):
                logits = self.model(tensors)
            # Average probabilities across augmentations (not logits — Jensen's inequality)
            avg_probs = F.softmax(logits / self.temperature, dim=-1).mean(dim=0)

        k = min(self.topk, self.num_classes)
        top_probs, top_idxs = torch.topk(avg_probs, k)
        top_list = [
            {"label": self.labels[int(i)], "confidence": float(p)}
            for p, i in zip(top_probs, top_idxs)
        ]
        return {
            "food_name":  top_list[0]["label"],
            "confidence": top_list[0]["confidence"],
            "top3":       top_list[:3],
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_labels(self, labels_path: str | None, num_classes: int) -> List[str]:
        if labels_path and Path(labels_path).exists():
            lines = Path(labels_path).read_text(encoding="utf-8").splitlines()
            labels = [l.strip() for l in lines if l.strip() and not l.startswith("#")]
            print(f"  Loaded {len(labels)} labels from {labels_path}")
            return labels
        print("  WARNING: no labels file — using numeric class indices.")
        return [f"food_class_{i}" for i in range(num_classes)]

    def _build_model(self, weights_path: str | None, num_classes: int) -> torch.nn.Module:
        if _TIMM_OK:
            has_weights = weights_path and Path(weights_path).exists()
            model = timm.create_model(
                self._TIMM_MODEL,
                pretrained=not has_weights,
                num_classes=num_classes,
            )
            if has_weights:
                state = torch.load(weights_path, map_location="cpu", weights_only=True)
                if isinstance(state, dict) and "model" in state:
                    state = state["model"]
                elif isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                model.load_state_dict(state)
                print(f"  Loaded fine-tuned weights: {weights_path}")
            else:
                print(f"  WARNING: using ImageNet-21k pretrained (not food-specific).")
            return model

        # torchvision fallback
        import torch.nn as nn
        model = _tv_models.efficientnet_v2_l(weights="IMAGENET1K_V1")
        in_feat = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_feat, num_classes)
        if weights_path and Path(weights_path).exists():
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            model.load_state_dict(state)
            print(f"  Loaded fine-tuned weights: {weights_path}")
        else:
            print("  WARNING: using ImageNet-1k pretrained (not food-specific).")
        return model

    def _build_transform(self):
        if _TIMM_OK:
            cfg = resolve_data_config(self.model.pretrained_cfg)
            return create_transform(**cfg)
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize((480, 480)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


