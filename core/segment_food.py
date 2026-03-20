"""
Food Segmentation — FastSAM
============================

Drop-in replacement for SAM ViT-L.  FastSAM runs up to 40× faster for
typical food-blob shapes with negligible quality loss.

Design:
  • One FastSAM forward pass per image (cached by MD5 hash).
  • Per-bbox mask selection via Ultralytics predictor.prompt(bboxes=…).
  • All public methods from the old FoodSegmenter are preserved.

Key tunables (env vars):
  FASTSAM_CONF     float  0.35   Detection confidence gate
  FASTSAM_IOU      float  0.90   NMS IoU threshold
  FASTSAM_IMGSZ    int   1024   Inference image size (square)
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import FastSAM


def _img_fingerprint(img: np.ndarray) -> tuple:
    """Cheap stable cache key — replaces full-image MD5."""
    flat = img.ravel()
    n    = len(flat)
    return (
        img.shape, img.dtype.str,
        int(flat[0]), int(flat[n // 4]), int(flat[n // 2]),
        int(flat[3 * n // 4]), int(flat[-1]),
    )


class FoodSegmenter:
    """FastSAM-based food segmenter.  Same public interface as old SAM ViT-L."""

    def __init__(self, checkpoint_path: str = "weights/FastSAM.pt"):
        print("Loading FastSAM …")
        requested = os.getenv("FASTSAM_DEVICE", "auto").lower()
        if requested == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = requested
        print(f"   Using device: {self.device}")

        self._conf  = float(os.getenv("FASTSAM_CONF",  "0.35"))
        self._iou   = float(os.getenv("FASTSAM_IOU",   "0.90"))
        self._imgsz = int(os.getenv("FASTSAM_IMGSZ",  "640"))  # 1024→640 saves ~3GB VRAM on 8GB GPU
        if self._imgsz < 1024:
            print(f"   [warn] FASTSAM_IMGSZ={self._imgsz} (< 1024); mask quality reduced for small objects")

        self._model = FastSAM(checkpoint_path)

        # Full-image result cache (avoid re-running for the same image)
        self._cached_results = None
        self._cached_hash: str | None = None

        print("✅ FastSAM loaded!")

    # ------------------------------------------------------------------
    # Internal: full-image segmentation (cached)
    # ------------------------------------------------------------------

    def _get_results(self, image: np.ndarray):
        """Run FastSAM on full image; return cached results if same image."""
        img_hash = _img_fingerprint(image)
        if img_hash != self._cached_hash:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()   # free fragmented VRAM before inference
            self._cached_results = self._model(
                image,
                device=self.device,
                retina_masks=True,
                imgsz=self._imgsz,
                conf=self._conf,
                iou=self._iou,
                verbose=False,
            )
            self._cached_hash = img_hash
        return self._cached_results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_image(self, image: np.ndarray) -> None:
        """Pre-encode image (warms the cache). Same API as old SAM predictor."""
        self._get_results(image)

    def segment(
        self,
        image: np.ndarray,
        bbox: List[int],
        refine_mask: bool = True,
        use_multimask: bool = True,   # kept for API compat; FastSAM always multi
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Segment food item using bounding box prompt.

        Args:
            image:        BGR OpenCV array.
            bbox:         [x1, y1, x2, y2]
            refine_mask:  Apply morphological refinement.

        Returns:
            mask:             bool ndarray (H, W)
            segmented_image:  BGR image with non-mask pixels zeroed
            score:            confidence score (box conf from FastSAM)
        """
        h, w = image.shape[:2]
        results = self._get_results(image)

        # --- apply bbox prompt ------------------------------------------------
        try:
            prompted = self._model.predictor.prompt(
                results, bboxes=[bbox]
            )
        except Exception as e:
            print(f"   [warn] FastSAM prompt failed: {e}")
            prompted = None

        mask, score = self._extract_best_mask(prompted, bbox, h, w)

        if refine_mask and mask is not None:
            mask = self._refine_mask(mask, bbox)

        if mask is None:
            # Fallback: rectangle mask from bbox
            mask = np.zeros((h, w), dtype=bool)
            x1, y1, x2, y2 = bbox
            mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = True
            score = 0.0

        segmented_image = image.copy()
        segmented_image[~mask] = 0
        return mask, segmented_image, score

    def segment_batch(
        self,
        image: np.ndarray,
        bboxes: List[List[int]],
        refine_mask: bool = True,
    ) -> List[Tuple[np.ndarray, float]]:
        """Segment all bboxes in a single FastSAM prompt call.

        Returns a list of (mask, seg_score) — one per input bbox.
        Using a single prompt() call saves N-1 Python→C++ round-trips.
        Falls back to individual calls if the batch prompt raises.
        """
        if not bboxes:
            return []
        h, w = image.shape[:2]
        results = self._get_results(image)

        try:
            prompted = self._model.predictor.prompt(results, bboxes=bboxes)
        except Exception as e:
            print(f"   [warn] FastSAM batch prompt failed ({e}); falling back")
            prompted = None

        output: List[Tuple[np.ndarray, float]] = []
        for bbox in bboxes:
            if prompted is not None:
                mask, score = self._extract_best_mask(prompted, bbox, h, w)
            else:
                _, _, score = self.segment(image, bbox, refine_mask=False)
                mask = None

            if refine_mask and mask is not None:
                mask = self._refine_mask(mask, bbox)
            if mask is None:
                mask = np.zeros((h, w), dtype=bool)
                x1, y1, x2, y2 = bbox
                mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = True
                score = 0.0
            output.append((mask, score))
        return output

    def segment_with_points(
        self,
        image: np.ndarray,
        bbox: List[int],
        add_center_point: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """API-compat wrapper — delegates to segment() (FastSAM uses bbox natively)."""
        return self.segment(image, bbox)

    # ------------------------------------------------------------------
    # Mask utilities (unchanged from old SAM implementation)
    # ------------------------------------------------------------------

    def get_mask_contour(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Extract largest contour from mask."""
        if mask.dtype == bool:
            mask_u8 = mask.astype(np.uint8) * 255
        elif mask.dtype.kind == 'f':
            # float mask (0.0–1.0): threshold at 0.5 to avoid truncation artifacts
            mask_u8 = (mask >= 0.5).astype(np.uint8) * 255
        else:
            mask_u8 = mask.astype(np.uint8)
            if mask_u8.max() <= 1:
                mask_u8 = mask_u8 * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return max(contours, key=cv2.contourArea) if contours else None

    def calculate_mask_area(self, mask: np.ndarray) -> int:
        """Pixel area of mask."""
        return int(np.sum(mask > 0 if mask.dtype != bool else mask))

    def get_mask_bbox(self, mask: np.ndarray) -> List[int]:
        """Tight bounding box around mask [x, y, x+w, y+h]."""
        mask_u8 = mask.astype(np.uint8) * 255 if mask.dtype == bool else mask
        coords = cv2.findNonZero(mask_u8)
        if coords is None:
            return [0, 0, 0, 0]
        x, y, bw, bh = cv2.boundingRect(coords)
        return [x, y, x + bw, y + bh]

    def erode_mask_edges(self, mask: np.ndarray, pixels: int = 5) -> np.ndarray:
        """Erode mask edges to avoid depth edge-bleeding."""
        mask_u8 = (mask * 255).astype(np.uint8) if mask.dtype == bool else mask
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (pixels * 2 + 1, pixels * 2 + 1)
        )
        eroded = cv2.erode(mask_u8, k, iterations=1)
        if cv2.countNonZero(eroded) == 0:
            return mask_u8 > 127
        return eroded > 127

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_best_mask(
        self,
        prompted,
        bbox: List[int],
        h: int,
        w: int,
    ) -> Tuple[Optional[np.ndarray], float]:
        """Pick the best mask from prompted FastSAM results."""
        if prompted is None or len(prompted) == 0:
            return None, 0.0

        result = prompted[0]
        if result.masks is None or result.masks.data.shape[0] == 0:
            return None, 0.0

        masks_tensor = result.masks.data      # (N, H, W) float/bool
        scores_raw   = result.boxes.conf if result.boxes is not None else None

        # Resize masks to original image resolution if needed
        mh, mw = masks_tensor.shape[1], masks_tensor.shape[2]
        if mh != h or mw != w:
            masks_np = masks_tensor.cpu().numpy()
            resized = np.stack([
                cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                for m in masks_np
            ], axis=0).astype(bool)
        else:
            resized = masks_tensor.cpu().numpy().astype(bool)

        # Select best mask:
        #   combined = raw_score × size_weight × in_bbox_fraction × centroid_weight
        # centroid_weight: penalise masks whose centroid is outside the bbox — this
        # prevents FastSAM from picking a large background object (e.g. a plant or
        # tablecloth) whose area happens to overlap the bbox but is centred elsewhere.
        x1, y1, x2, y2 = bbox
        # Normalise flipped coords (x1>x2 or y1>y2 can happen with noisy detections)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        bbox_area  = max((x2 - x1) * (y2 - y1), 1)
        cx_b = (x1 + x2) / 2.0   # bbox centre x
        cy_b = (y1 + y2) / 2.0   # bbox centre y
        bw = max(x2 - x1, 1)
        bh = max(y2 - y1, 1)

        best_mask, best_score = None, -1.0

        for i, mask in enumerate(resized):
            raw_score = float(scores_raw[i]) if (scores_raw is not None and i < len(scores_raw)) else 0.5
            mask_area = int(mask.sum())
            if mask_area == 0:
                continue

            fill = mask_area / bbox_area
            # Penalise masks much larger or smaller than the bbox
            size_w = 1.0 if 0.10 <= fill <= 3.0 else 0.5

            # Fraction of mask pixels that fall inside the bbox
            roi = mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            in_bbox = float(roi.sum()) / (mask_area + 1e-6)

            # Centroid of mask vs centroid of bbox (normalised distance)
            ys, xs = np.nonzero(mask)
            cx_m = float(xs.mean())
            cy_m = float(ys.mean())
            dist_norm = ((cx_m - cx_b) / bw) ** 2 + ((cy_m - cy_b) / bh) ** 2
            centroid_w = float(np.exp(-2.0 * dist_norm))   # 1.0 at centre, ~0.14 at 1 bbox-width away

            combined = raw_score * size_w * (0.5 + 0.3 * min(in_bbox, 1.0) + 0.2 * centroid_w)
            if combined > best_score:
                best_score = combined
                best_mask  = mask

        return best_mask, best_score

    def _refine_mask(self, mask: np.ndarray, bbox: List[int]) -> np.ndarray:
        """Close holes, remove noise, keep largest connected component."""
        mask_u8 = mask.astype(np.uint8) * 255
        x1, y1, x2, y2 = bbox
        ks = max(3, int(max(x2 - x1, y2 - y1) * 0.01))
        if ks % 2 == 0:
            ks += 1
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k, iterations=2)
        opened  = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN,  k, iterations=1)
        if cv2.countNonZero(opened) > 0:
            mask_u8 = opened
        return self._keep_largest_component(mask_u8) > 127

    def _keep_largest_component(self, mask_u8: np.ndarray) -> np.ndarray:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        if n <= 1:
            return mask_u8
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        out = np.zeros_like(mask_u8)
        out[labels == largest] = 255
        return out
