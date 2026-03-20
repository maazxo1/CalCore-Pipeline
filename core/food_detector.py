"""
Food Detector — YOLOv8l (custom-trained on combined food dataset)
=================================================================

YOLOv8l trained on 113,884 images across 205 food classes from 4 datasets:
  complete_food, allinone, Food Detection Dataset, OpenImages food subset.

Weights: weights/yoloo.pt  (produced by scripts/train_yolov8_combined.py)
Training: 30 epochs, mAP50=0.551, Recall=0.515

The detector's job is to find WHERE food is (bounding boxes).
The classifier (EfficientNet/CLIP) determines WHAT the food is.
YOLO class labels are passed through but overridden by the classifier.

Key tunables (env vars):
  YOLO_CONF          float  0.08   Box confidence gate
  YOLO_IOU           float  0.35   NMS IoU threshold
  MIN_BBOX_AREA_PX   int    600    Drop boxes smaller than this (px²)
  YOLO_IMGSZ         int    800    Inference image size
  YOLO_DEVICE        str   auto    Force cpu / cuda / mps
  YOLO_WEIGHTS       str           Override weight path
  YOLO_TILED         bool   true   Enable tiled detection for dense plates
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from ultralytics import YOLO

from core.blur_detector import detect_blur


# ---------------------------------------------------------------------------
# Default weight path
# ---------------------------------------------------------------------------
_WEIGHTS_DIR      = Path(__file__).parent.parent / "weights"
_DEFAULT_WEIGHTS  = _WEIGHTS_DIR / "yoloo.pt"
_FALLBACK_WEIGHTS = _WEIGHTS_DIR / "yolo.pt"    # old Food-101 weights as fallback

# Tile overlap fraction for tiled detection
_TILE_OVERLAP = 0.25


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _iou(a: List[int], b: List[int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    ua = max((a[2] - a[0]) * (a[3] - a[1]), 1)
    ub = max((b[2] - b[0]) * (b[3] - b[1]), 1)
    return inter / (ua + ub - inter)


def _soft_nms(dets: List[Dict], sigma: float = 0.5, score_thresh: float = 0.05) -> List[Dict]:
    """Gaussian Soft-NMS (Bodla et al. 2017) — cross-class, vectorized.

    Unlike hard NMS, overlapping boxes have their confidence decayed by
    weight = exp(−IoU² / σ).  This preserves adjacent food items (e.g. a
    croissant touching a sausage) that a hard NMS would suppress.

    Vectorized implementation: all pairwise IoUs are precomputed as a numpy
    matrix; the greedy loop runs with O(N) numpy ops per iteration instead of
    O(N) Python object iterations, giving 3–5× speedup for N = 20–50 dets.

    Original scores are preserved in the output — decayed scores are internal.
    """
    if len(dets) <= 1:
        return list(dets)

    n      = len(dets)
    boxes  = np.array([d["bbox"] for d in dets], dtype=np.float32)   # (N, 4)
    scores = np.array([d["confidence"] for d in dets], dtype=np.float64)

    # Precompute all N×N pairwise IoUs in one vectorised pass
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter   = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    areas   = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union   = areas[:, None] + areas[None, :] - inter
    iou_mat = np.where(union > 0, inter / union, 0.0)   # (N, N)

    active   = np.ones(n, dtype=bool)
    kept_idx: List[int] = []

    while True:
        # Find best active box
        masked = np.where(active, scores, -1.0)
        best_i = int(np.argmax(masked))
        if scores[best_i] < score_thresh:
            break
        kept_idx.append(best_i)
        active[best_i] = False

        # Vectorised Gaussian decay for all remaining active boxes
        decay     = np.exp(-(iou_mat[best_i] ** 2) / sigma)
        scores   *= decay
        active   &= (scores >= score_thresh)

        if not active.any():
            break

    return [dets[i] for i in kept_idx]


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class FoodDetector:
    """Food object detector using YOLOv8l trained on 205-class combined food dataset."""

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "auto",
    ):
        # Device resolution
        requested = os.getenv("YOLO_DEVICE", device).lower()
        if requested == "auto":
            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        else:
            self._device = requested

        # Tunables
        self.conf_thresh   = float(os.getenv("YOLO_CONF",       "0.10"))
        self.iou_thresh    = float(os.getenv("YOLO_IOU",        "0.35"))
        self.min_bbox_area = int(os.getenv("MIN_BBOX_AREA_PX",  "600"))
        self.imgsz         = int(os.getenv("YOLO_IMGSZ",        "800"))
        self.tiled         = os.getenv("YOLO_TILED", "true").lower()  != "false"
        # TTA runs the model at 3 scales + horizontal flip on the full image.
        # Adds +1-2 mAP (better small-item recall) at ~3× full-image inference cost.
        # Disabled by default — enable with YOLO_TTA=true when latency allows.
        # NOTE: TTA is only applied to the full-image pass, NOT to tiles
        # (tiles already provide multi-scale analysis; double-TTA would conflict).
        self.use_tta       = os.getenv("YOLO_TTA", "false").lower() == "true"

        # Weight selection
        weight = (
            model_path
            or os.getenv("YOLO_WEIGHTS")
            or (str(_DEFAULT_WEIGHTS) if _DEFAULT_WEIGHTS.exists() else str(_FALLBACK_WEIGHTS))
        )
        print(f"Loading YOLOv8l from '{weight}' on {self._device} …")
        self._model = YOLO(weight)

        num_classes = len(self._model.names)
        print(f"  YOLOv8l ready — {num_classes} classes.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        img_cv2: np.ndarray,
        yolo_conf: float | None = None,
    ) -> List[Dict]:
        """
        Detect food in a BGR OpenCV image.

        Runs full-image detection + tiled detection on 4 overlapping quadrants.
        Tiled detection recovers small/dense items (sausages, bacon strips) that
        full-image YOLO misses due to scale and crowding.

        Returns list of dicts:
            bbox, confidence, class_name, is_food, is_container, detector_source
        """
        h, w = img_cv2.shape[:2]
        conf = yolo_conf if yolo_conf is not None else self.conf_thresh

        # ── Full-image detection ──────────────────────────────────────────
        full_dets = self._run_inference(img_cv2, conf, 0, 0, augment=self.use_tta)

        # ── Tiled detection — 2×2 overlapping tiles ───────────────────────
        tile_dets: List[Dict] = []
        if self.tiled:
            tile_dets = self._detect_tiled(img_cv2, conf)

        # ── Merge + Soft-NMS ─────────────────────────────────────────────
        all_dets = full_dets + tile_dets
        all_dets = _soft_nms(all_dets)

        # ── Sharpness filter — remove blurry background detections ────────
        all_dets = self._filter_blurry(img_cv2, all_dets)

        print(f"  YOLOv8l: {len(all_dets)} food detection(s)"
              f" (full={len(full_dets)}, tiles={len(tile_dets)})")
        return all_dets

    # ------------------------------------------------------------------
    # Tiled detection
    # ------------------------------------------------------------------

    def _detect_tiled(self, img_cv2: np.ndarray, conf: float) -> List[Dict]:
        """Split image into 2×2 overlapping tiles, run YOLO on all 4 in one batch.

        All four tiles have identical dimensions (60%×60% + 25% overlap), so
        batching them into a single YOLO forward pass avoids 3 extra GPU kernel
        launches and reduces tile inference time by ~3×.

        Overlap=25% prevents items near tile edges from being missed.
        """
        h, w = img_cv2.shape[:2]
        tile_conf = conf + 0.02  # slight noise floor to reduce background noise

        half_w = int(w * 0.60)
        half_h = int(h * 0.60)
        ox     = int(w * _TILE_OVERLAP)
        oy     = int(h * _TILE_OVERLAP)

        tiles = [
            (0,               0,               half_w + ox,     half_h + oy),   # TL
            (w - half_w - ox, 0,               w,               half_h + oy),   # TR
            (0,               h - half_h - oy, half_w + ox,     h),             # BL
            (w - half_w - ox, h - half_h - oy, w,               h),             # BR
        ]

        # Crop all valid tiles and record their offsets
        tile_imgs:    List[np.ndarray] = []
        tile_offsets: List[Tuple[int, int]] = []
        for (tx1, ty1, tx2, ty2) in tiles:
            tx1, ty1 = max(0, tx1), max(0, ty1)
            tx2, ty2 = min(w, tx2), min(h, ty2)
            img = img_cv2[ty1:ty2, tx1:tx2]
            if img.size == 0:
                continue
            tile_imgs.append(img)
            tile_offsets.append((tx1, ty1))

        if not tile_imgs:
            return []

        # Single batched YOLO call — all tiles inferred in one GPU forward pass.
        # Ultralytics auto-pads a list of equal-sized images with no overhead.
        batch_results = self._model(
            tile_imgs,
            conf=tile_conf,
            iou=self.iou_thresh,
            imgsz=self.imgsz,
            verbose=False,
            device=self._device,
            augment=False,
        )

        tile_dets: List[Dict] = []
        for r, (ox_, oy_) in zip(batch_results, tile_offsets):
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id   = int(box.cls[0])
                cls_name = r.names.get(cls_id, "unknown")
                score    = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1 += ox_; y1 += oy_; x2 += ox_; y2 += oy_
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                if not self._valid([x1, y1, x2, y2], h, w, score):
                    continue
                tile_dets.append({
                    "bbox":            [x1, y1, x2, y2],
                    "confidence":      score,
                    "class_name":      cls_name,
                    "is_food":         True,
                    "is_container":    False,
                    "detector_source": "yolov8l",
                })
        return tile_dets

    # ------------------------------------------------------------------
    # Core inference (single image or tile)
    # ------------------------------------------------------------------

    def _run_inference(
        self,
        img: np.ndarray,
        conf: float,
        offset_x: int,
        offset_y: int,
        augment: bool = False,
    ) -> List[Dict]:
        """Run YOLO on one image/tile, offset boxes by (offset_x, offset_y).

        augment=True enables YOLO TTA (3 scales + h-flip).  Only used on the
        full-image pass — tile calls always pass augment=False.
        """
        full_h = img.shape[0] + offset_y
        full_w = img.shape[1] + offset_x

        results = self._model(
            img,
            conf=conf,
            iou=self.iou_thresh,
            imgsz=self.imgsz,
            verbose=False,
            device=self._device,
            augment=augment,
        )

        dets: List[Dict] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id   = int(box.cls[0])
                cls_name = r.names.get(cls_id, "unknown")
                score    = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Offset back to full-image coordinates
                x1 += offset_x
                y1 += offset_y
                x2 += offset_x
                y2 += offset_y

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(full_w, x2), min(full_h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                if not self._valid([x1, y1, x2, y2], full_h, full_w, score):
                    continue
                dets.append({
                    "bbox":            [x1, y1, x2, y2],
                    "confidence":      score,
                    "class_name":      cls_name,
                    "is_food":         True,
                    "is_container":    False,
                    "detector_source": "yolov8l",
                })
        return dets

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _valid(self, bbox: List[int], h: int, w: int, conf: float = 0.0) -> bool:
        x1, y1, x2, y2 = bbox
        area = max(x2 - x1, 0) * max(y2 - y1, 0)
        # High-confidence detections (≥ 0.70) get a relaxed area floor so that
        # small but clearly identified items (a cherry, a sauce cup) are kept.
        min_area = int(self.min_bbox_area * 0.60) if conf >= 0.70 else self.min_bbox_area
        return (
            area >= min_area
            and area <= 0.95 * h * w
        )

    def _filter_blurry(self, img_cv2: np.ndarray, dets: List[Dict]) -> List[Dict]:
        """
        Remove detections whose bbox crop is in an out-of-focus background region.

        Uses FFT-based sharpness scoring (VolETA CVPR 2024) via blur_detector.py.
        This gives an *absolute* score (not relative to other detections), so a
        sharp on-plate strawberry is never penalised because the cake next to it
        has a higher Canny edge count.

        Scoring (detect_blur):
          Sharp   ≈ 20–40   (clear texture/edges)
          Soft    ≈ 12–18   (slightly out of focus)
          Blurry  < 10      (bokeh background)

        CONF_EXEMPT = 0.55: YOLO is confident enough → skip blur check entirely.
        BLUR_THRESHOLD = 12.0: crops scoring below this are considered background.
        """
        BLUR_THRESHOLD = 12.0   # FFT score below this → blurry background
        CONF_EXEMPT    = 0.30   # YOLO confidence at/above this → skip blur check
        # 0.30 rationale: smooth-skinned foods (strawberries, tomatoes, grapes)
        # score low on FFT even when sharp. At conf ≥ 0.30 YOLO has seen enough
        # of the item to be reliable. True bokeh background items tend to cluster
        # below 0.20 where the FFT check still applies.

        kept = []
        for d in dets:
            if d["confidence"] >= CONF_EXEMPT:
                kept.append(d)   # confident detection: trust YOLO, skip check
                continue

            x1, y1, x2, y2 = d["bbox"]
            crop = img_cv2[y1:y2, x1:x2]
            if crop.size == 0:
                # degenerate crop — keep and let downstream handle it
                kept.append(d)
                continue

            score, is_blurry = detect_blur(crop, threshold=BLUR_THRESHOLD)
            if is_blurry:
                print(f"   Blur-filtered: {d['class_name']} {d['confidence']:.2f}"
                      f" (FFT sharpness {score:.1f} < {BLUR_THRESHOLD})")
            else:
                kept.append(d)

        return kept
