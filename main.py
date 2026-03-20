"""
Food Analysis Pipeline — Commercial Grade
==========================================
detect → segment → classify → depth → volume → nutrition

Architecture:
  Detection   YOLOv8l (205-class combined dataset) — bounding-box proposals only; class labels ignored
  Segment     FastSAM  — precise food mask within each bbox (up to 40× faster than SAM ViT-L)
  Classify    EfficientNetV2-L (primary) + CLIP ViT-L/14 (fallback) on masked food crop
  Depth       Depth Anything V2 metric (hypersim/vitl) — run ONCE per image
  Volume      Voxelization height-field with USDA anchor correction
  Nutrition   USDA FoodData Central + offline fallback

Order rationale:
  Segment before classify so the classifier sees only the food pixels (mask applied),
  not background noise (plate rim, tablecloth, other items).  YOLO class labels are
  not used for the final label — the classifier always determines the food identity.
"""

import cv2
import os
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
from typing import List, Dict, Optional

from core.food_detector import FoodDetector
from core.classify import FoodClassifier
from core.classify_efficientnet import EfficientNetClassifier
from core.segment_food import FoodSegmenter
from core.estimate_depth import DepthEstimator
from core.volume_calculator import VolumeCalculator
from core.volume_estimator import aggregate_multi_image_volumes
from core.blur_detector import detect_blur
from core.food_taxonomy import get_taxonomy
from core.pipeline_postprocess import (
    dedupe_items,
    filter_failed_items,
    group_items_across_images,
    mask_iou,
)
from data.usda_nutrition_lookup import get_nutrition_info, get_typical_serving_weight


# ============================================================================
# Module-level singletons
# ============================================================================

_detector: Optional[FoodDetector] = None
_classifier: Optional[FoodClassifier] = None          # primary classifier
_clip_classifier: Optional[FoodClassifier] = None     # CLIP fallback (dual-mode only)
_segmenter: Optional[FoodSegmenter] = None
_depth_estimator: Optional[DepthEstimator] = None
_volume_calculator: Optional[VolumeCalculator] = None

# Confidence threshold below which primary (EfficientNet) defers to CLIP fallback
_EFFNET_FALLBACK_THRESHOLD = float(os.getenv("EFFNET_FALLBACK_THRESHOLD", "0.65"))

_METRIC_CHECKPOINT  = "weights/depth_anything_v2_metric_hypersim_vitl.pth"
_RELATIVE_CHECKPOINT = "weights/depth_anything_v2_large.pth"

# Post-processing thresholds
_MIN_MASK_COVERAGE      = float(os.getenv("MIN_MASK_COVERAGE",      "0.10"))
_MIN_SEG_SCORE          = float(os.getenv("MIN_SEG_SCORE",          "0.35"))
_MIN_BBOX_RELATIVE_AREA = float(os.getenv("MIN_BBOX_RELATIVE_AREA", "0.005"))  # 0.5% — was 2%
_MASK_IOU_DEDUP         = float(os.getenv("MASK_IOU_DEDUP",         "0.60"))
_DEDUP_SAME_LABEL_IOU   = float(os.getenv("DEDUP_SAME_LABEL_IOU",   "0.35"))
_DEDUP_SAME_LABEL_SIM   = float(os.getenv("DEDUP_SAME_LABEL_SIM",   "0.62"))
_DEDUP_CROSS_LABEL_IOU  = float(os.getenv("DEDUP_CROSS_LABEL_IOU",  "0.88"))


# ============================================================================
# Model loading
# ============================================================================

def load_models() -> None:
    """Load all pipeline models into singletons (idempotent)."""
    global _detector, _classifier, _clip_classifier, _segmenter, _depth_estimator, _volume_calculator

    if all(x is not None for x in (_detector, _classifier, _segmenter, _depth_estimator, _volume_calculator)):
        return

    print("\n" + "=" * 70)
    print("Loading pipeline models…")
    print("=" * 70)

    _detector = FoodDetector()

    _cls_backend    = os.getenv("CLASSIFIER_BACKEND", "clip").strip().lower()
    _effnet_weights = os.getenv("EFFNET_WEIGHTS", "weights/efficientnet_food101/best.pth")
    _effnet_labels  = os.getenv("EFFNET_LABELS",  "weights/efficientnet_food101/labels.txt")
    _effnet_ready   = Path(_effnet_weights).exists() and Path(_effnet_labels).exists()

    if _cls_backend == "efficientnet" or (_cls_backend == "auto" and _effnet_ready):
        # Dual-mode: EfficientNet primary + CLIP fallback for low-confidence crops
        print("Classifier mode: EfficientNet primary + CLIP fallback")
        _classifier      = EfficientNetClassifier(
            weights_path=_effnet_weights,
            labels_path=_effnet_labels,
        )
        _clip_classifier = FoodClassifier("nateraw/food")
    else:
        # CLIP-only (default until EfficientNet weights exist)
        if _cls_backend == "efficientnet" and not _effnet_ready:
            print(f"WARNING: EfficientNet weights not found at {_effnet_weights}")
            print("         Run: python scripts/train_efficientnet_food101.py")
        _classifier      = FoodClassifier("nateraw/food")
        _clip_classifier = None

    _segmenter = FoodSegmenter(
        checkpoint_path=os.getenv("FASTSAM_WEIGHTS", "weights/FastSAM.pt"),
    )

    if Path(_METRIC_CHECKPOINT).exists():
        print(f"\nUsing METRIC depth: {_METRIC_CHECKPOINT}")
        _depth_estimator = DepthEstimator(
            model_size="large",
            checkpoint_path=_METRIC_CHECKPOINT,
            use_metric=True,
            metric_dataset="hypersim",
            max_depth=5.0,
        )
    else:
        print(f"\nMetric checkpoint not found — using relative depth.")
        _depth_estimator = DepthEstimator(
            model_size="large",
            checkpoint_path=_RELATIVE_CHECKPOINT,
            use_metric=False,
        )
    _depth_estimator.load_model()

    _volume_calculator = VolumeCalculator()
    print(f"\nAll models loaded!")


def _get_models():
    if _detector is None:
        load_models()
    return _detector, _classifier, _clip_classifier, _segmenter, _depth_estimator, _volume_calculator


def _classify_with_fallback(
    classifier,
    clip_classifier,
    image,
    bbox,
    fallback_threshold: float = None,
) -> dict:
    """Run primary classifier; fall back to CLIP if confidence is too low.

    When EfficientNet is primary (closed-set, 101 classes), CLIP covers foods
    outside Food-101 and boosts confidence on ambiguous crops.
    The higher-confidence result wins.
    """
    threshold = _EFFNET_FALLBACK_THRESHOLD if fallback_threshold is None else fallback_threshold
    primary = classifier.classify(image, bbox)

    if clip_classifier is None:
        return primary  # single-classifier mode

    primary_conf = float(primary.get("confidence", 0.0))

    if primary_conf >= threshold:
        primary["classifier_mode"] = "primary"
        return primary

    # Primary uncertain — run CLIP fallback
    fallback = clip_classifier.classify(image, bbox)
    fallback_conf = float(fallback.get("confidence", 0.0))

    if fallback_conf > primary_conf:
        fallback["classifier_mode"] = "clip_fallback"
        return fallback

    primary["classifier_mode"] = "primary_low_conf"
    return primary


def _classify_batch_with_fallback(
    classifier,
    clip_classifier,
    masked_pils: List[Image.Image],
    fallback_threshold: float = None,
) -> List[dict]:
    """Batch version of _classify_with_fallback.

    Runs EfficientNet on all crops in one forward pass.
    Items where confidence < threshold are sent to CLIP in a second batch pass.
    Avoids N separate GPU forward passes — biggest single speed-up in the pipeline.
    """
    if not masked_pils:
        return []

    threshold = _EFFNET_FALLBACK_THRESHOLD if fallback_threshold is None else fallback_threshold

    # Primary batch pass
    if hasattr(classifier, "classify_batch"):
        primary_results = classifier.classify_batch(masked_pils)
    else:
        primary_results = [classifier.classify(img, None) for img in masked_pils]

    if clip_classifier is None:
        for r in primary_results:
            r.setdefault("classifier_mode", "primary")
        return primary_results

    # Identify which items need CLIP fallback
    clip_needed = [
        i for i, r in enumerate(primary_results)
        if float(r.get("confidence", 0.0)) < threshold
    ]

    if not clip_needed:
        for r in primary_results:
            r["classifier_mode"] = "primary"
        return primary_results

    # CLIP batch pass for low-confidence items
    clip_imgs    = [masked_pils[i] for i in clip_needed]
    clip_bboxes  = [None] * len(clip_imgs)
    if hasattr(clip_classifier, "classify_batch"):
        clip_results = clip_classifier.classify_batch(clip_imgs, clip_bboxes)
    else:
        clip_results = [clip_classifier.classify(img, None) for img in clip_imgs]

    final = list(primary_results)
    for j, i in enumerate(clip_needed):
        prim = primary_results[i]
        clip = clip_results[j]
        if clip.get("gate_rejected"):
            # Food gate fired on this item — propagate so _pick_label rejects it
            final[i] = clip
            final[i]["classifier_mode"] = "clip_gate_reject"
            continue
        prim_conf = float(prim.get("confidence", 0.0))
        clip_conf = float(clip.get("confidence", 0.0))
        if clip_conf > prim_conf:
            final[i] = clip
            final[i]["classifier_mode"] = "clip_fallback"
        else:
            final[i] = prim
            final[i]["classifier_mode"] = "primary_low_conf"

    return final


# ============================================================================
# Label resolution (replaces label_fusion + label_mapper chain)
# ============================================================================

def _pick_label(
    detection: Dict,
    classification: Dict,
    taxonomy,
    cls_conf_min: float = 0.65,
) -> Dict:
    """
    4-path label resolution.

    Priority:
      0. Detector strong override (conf ≥ 0.80, taxonomy resolves) → detector wins
         BEFORE CLIP.  Prevents CLIP from overriding a high-confidence detector
         result with a spurious label caused by a wide crop (e.g. egg bbox that
         also contains toast → CLIP picks "sandwich").
      1. CLIP confident (conf ≥ cls_conf_min, margin ≥ 0.10) → classifier wins
      2. Detector confident (conf ≥ 0.25, taxonomy resolves) → detector wins
      3. CLIP weak-pass — tiered margin requirements:
           solo high  (cls_conf ≥ 0.73): margin ≥ 0.07  (catches 74-78% CLIP hits)
           solo mid   (cls_conf ≥ 0.70): margin ≥ 0.10
           det-assist (det_conf ≥ 0.10): margin ≥ 0.10, cls_conf ≥ cls_conf_min-0.10
      4. Reject
    """
    cls_name = str(classification.get("food_name", "")).strip()
    cls_conf = float(classification.get("confidence", 0.0))
    top3     = classification.get("top3", [])
    top2_conf = float(top3[1]["confidence"]) if len(top3) > 1 else 0.0
    cls_margin = cls_conf - top2_conf

    det_name = str(detection.get("class_name", "")).strip()
    det_conf = float(detection.get("confidence", 0.0))

    cls_res = taxonomy.resolve_label(cls_name)
    det_res = taxonomy.resolve_label(det_name)

    # Hard-pair margin — when top-1 and top-2 classifier labels form a known
    # visually-confusable pair, require a higher confidence margin before
    # accepting the result (0.15 instead of the default 0.10).
    _top2_label  = str(top3[1]["label"]).strip().lower() if len(top3) > 1 else ""
    _cls_can_low = cls_res.canonical_name.lower() if cls_res.resolved else cls_name.lower()
    _req_margin  = (
        _HARD_PAIR_MARGIN
        if frozenset({_cls_can_low, _top2_label}) in _HARD_CLASSIFICATION_PAIRS
        else 0.10
    )

    # Gate rejection — non-food crop flagged by CLIP binary gate
    if classification.get("gate_rejected"):
        return {
            "accepted":           False,
            "food_name":          det_name,
            "canonical_id":       "",
            "canonical_name":     "",
            "canonical_category": "default_unknown",
            "confidence":         0.0,
            "top3":               [],
            "source":             "none",
            "reason_code":        "nonfood_gate_reject",
            "predicted_label":    det_name,
        }

    # Path 0: Detector strong override (≥ 0.80) — fires before classifier.
    # Our trained YOLOv8l is reliable at high confidence on specific food labels.
    # Lookalike exception: for visually ambiguous pairs the classifier decides.
    if det_res.resolved and det_conf >= 0.80:
        _det_can  = det_res.canonical_name.lower()
        _cls_can  = cls_res.canonical_name.lower() if cls_res.resolved else ""
        _is_lookalike = frozenset({_det_can, _cls_can}) in _LOOKALIKE_PAIRS
        if not (_is_lookalike and cls_res.resolved and cls_conf >= 0.60):
            return {
                "accepted":           True,
                "food_name":          _det_can,
                "canonical_id":       det_res.canonical_id,
                "canonical_name":     det_res.canonical_name,
                "canonical_category": det_res.category,
                "confidence":         det_conf,
                "top3":               [{"label": _det_can, "confidence": det_conf}],
                "source":             "detector",
                "reason_code":        "detector_strong_override",
                "predicted_label":    det_name,
            }

    # Path 1a: Detector high-confidence (≥ 0.70) wins BEFORE classifier.
    # At ≥ 70% confidence, YOLO's food-specific training is very reliable.
    # Below 0.70, YOLO labels fall through to EfficientNet/CLIP which handle
    # fine-grained classification (e.g. bacon crop inside a "Bread" bbox).
    if det_res.resolved and det_conf >= 0.70:
        return {
            "accepted":           True,
            "food_name":          det_res.canonical_name.lower(),
            "canonical_id":       det_res.canonical_id,
            "canonical_name":     det_res.canonical_name,
            "canonical_category": det_res.category,
            "confidence":         det_conf,
            "top3":               [{"label": det_res.canonical_name.lower(), "confidence": det_conf}],
            "source":             "detector",
            "reason_code":        "detector_pass",
            "predicted_label":    det_name,
        }

    # Path 1b: Classifier clear winner (conf ≥ cls_conf_min, margin ≥ req_margin)
    # Fires when YOLO is uncertain (< 0.70) — classifier takes over.
    # Hard pairs require _req_margin=0.15 instead of 0.10.
    if cls_res.resolved and cls_conf >= cls_conf_min and cls_margin >= _req_margin:
        return {
            "accepted":           True,
            "food_name":          cls_res.canonical_name.lower(),
            "canonical_id":       cls_res.canonical_id,
            "canonical_name":     cls_res.canonical_name,
            "canonical_category": cls_res.category,
            "confidence":         cls_conf,
            "top3":               top3,
            "source":             "classifier",
            "reason_code":        "classifier_pass",
            "predicted_label":    cls_name,
        }

    # Path 2: Detector low-confidence fallback (min 0.35 — below this YOLO precision ~30%)
    if det_res.resolved and det_conf >= 0.35:
        return {
            "accepted":           True,
            "food_name":          det_res.canonical_name.lower(),
            "canonical_id":       det_res.canonical_id,
            "canonical_name":     det_res.canonical_name,
            "canonical_category": det_res.category,
            "confidence":         det_conf,
            "top3":               [{"label": det_res.canonical_name.lower(), "confidence": det_conf}],
            "source":             "detector",
            "reason_code":        "detector_pass",
            "predicted_label":    det_name,
        }

    # Path 3: CLIP weak pass — tiered margins so high-confidence CLIP results
    # aren't wrongly rejected just because top-2 is close.
    # solo_high: cls_conf ≥ 0.73 needs only 0.07 margin (catches 74-78% rejects)
    # solo_mid:  cls_conf ≥ 0.70 needs 0.10 margin
    # det_bar:   detector assists → lower cls threshold, margin ≥ 0.10
    _has_det   = det_res.resolved and det_conf >= 0.10
    _solo_high = cls_conf >= 0.73 and cls_margin >= max(0.07, _req_margin - 0.05)
    _solo_mid  = cls_conf >= 0.70 and cls_margin >= _req_margin
    _det_bar   = _has_det and cls_conf >= cls_conf_min - 0.10 and cls_margin >= _req_margin
    if cls_res.resolved and (_solo_high or _solo_mid or _det_bar):
        return {
            "accepted":           True,
            "food_name":          cls_res.canonical_name.lower(),
            "canonical_id":       cls_res.canonical_id,
            "canonical_name":     cls_res.canonical_name,
            "canonical_category": cls_res.category,
            "confidence":         cls_conf,
            "top3":               top3,
            "source":             "classifier_weak",
            "reason_code":        "classifier_weak_pass",
            "predicted_label":    cls_name,
        }

    # Reject
    return {
        "accepted":           False,
        "food_name":          det_name or cls_name,
        "canonical_id":       "",
        "canonical_name":     "",
        "canonical_category": "default_unknown",
        "confidence":         max(cls_conf, det_conf),
        "top3":               top3,
        "source":             "none",
        "reason_code":        "low_confidence_reject",
        "predicted_label":    cls_name or det_name,
    }


# Generic / category-level YOLO class names that should NOT override the classifier.
# These are broad labels (from the allinone dataset) that are correct detections
# but too vague to use as final food labels.
_GENERIC_YOLO_CLASSES = {
    "food", "fruit", "vegetable", "baked goods", "fast food",
    "dessert", "seafood", "snack", "pastry",
}

# Visually ambiguous food pairs — for these, the classifier decides even when
# YOLO fires at high confidence (path 0 / path 1a).
# Defined once at module level (not per-call) for efficiency.
_LOOKALIKE_PAIRS = {
    frozenset({"bagel", "donut"}),
    frozenset({"sausage", "croissant"}),
    frozenset({"muffin", "cupcake"}),
    frozenset({"hot dog", "sausage"}),
    frozenset({"spring roll", "sausage"}),
}

# Hard classification pairs — foods that are visually very similar and cause
# systematic classifier confusion (from Food-101 confusion matrix analysis).
# When the top-1 / top-2 labels form one of these pairs, a higher confidence
# margin is required before accepting the label (0.15 vs the default 0.10).
# This prevents a sausage → pork chop flip just because they both have
# grill marks and the model was marginally more confident on the wrong one.
_HARD_CLASSIFICATION_PAIRS = {
    frozenset({"steak",      "pork chop"}),
    frozenset({"pho",        "ramen"}),
    frozenset({"dumplings",  "ravioli"}),
    frozenset({"croissant",  "spring roll"}),
    frozenset({"apple pie",  "baklava"}),
    frozenset({"muffin",     "cupcake"}),
    frozenset({"hot dog",    "sausage"}),
    frozenset({"pancakes",   "flatbread"}),
    frozenset({"fried egg",  "egg"}),
    frozenset({"steak",      "grilled salmon"}),
}
_HARD_PAIR_MARGIN = 0.15   # required margin when top-1/top-2 form a hard pair


# ============================================================================
# Calibration helper
# ============================================================================

def _compute_calibration_scale(
    reference_type: Optional[str],
    reference_size_cm: Optional[float],
) -> float:
    if reference_size_cm is None:
        return 1.0
    try:
        value = float(reference_size_cm)
    except (TypeError, ValueError):
        return 1.0
    if value <= 0:
        return 1.0
    ref = (reference_type or "").strip().lower()
    if ref == "manual_scale":
        return float(np.clip(value, 0.5, 2.0))
    defaults = {"plate": 26.0, "card": 8.56}
    base = defaults.get(ref)
    if not base:
        return 1.0
    return float(np.clip(value / base, 0.5, 2.0))


# ============================================================================
# Masked-crop helper
# ============================================================================

def _make_masked_crop(img_cv2: np.ndarray, mask: np.ndarray, bbox: List[int]) -> Image.Image:
    """Apply food mask to a bbox crop (zero out background), return as PIL RGB.

    Only copies the bbox slice — avoids a full-image copy per detection.
    """
    h, w = img_cv2.shape[:2]
    x1, y1 = max(0, bbox[0]), max(0, bbox[1])
    x2, y2 = min(w, bbox[2]), min(h, bbox[3])
    crop = img_cv2[y1:y2, x1:x2].copy()
    # Guard against mask shape mismatch (e.g. FastSAM returned different resolution)
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    crop[~mask[y1:y2, x1:x2]] = 0
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


# ============================================================================
# Single-image pipeline
# ============================================================================

def process_food_image(
    image_path: str,
    output_path: Optional[str] = "output_final.jpg",
    detection_conf: float = 0.10,
    classification_conf: float = 0.60,
    reference_type: Optional[str] = None,
    reference_size_cm: Optional[float] = None,
    reject_low_quality: bool = False,
) -> List[Dict]:
    """Run the full food analysis pipeline on a single image."""
    detector, classifier, clip_classifier, segmenter, depth_estimator, volume_calculator = _get_models()
    taxonomy = get_taxonomy()
    calibration_scale = _compute_calibration_scale(reference_type, reference_size_cm)

    print(f"\nProcessing: {image_path}")
    try:
        _pil_raw = Image.open(image_path)
        img_pil  = ImageOps.exif_transpose(_pil_raw)
        img_cv2  = cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"  Failed to open image '{image_path}': {e}")
        return []

    # Blur gate
    blur_score, is_blurry = detect_blur(img_cv2)
    if is_blurry:
        print(f"  Image too blurry (score {blur_score:.1f}). Skipping.")
        return []

    # ------------------------------------------------------------------
    # Step 1: Detect
    # ------------------------------------------------------------------
    print("\n Step 1: Detecting food items…")
    detections = detector.detect(img_cv2, yolo_conf=detection_conf)
    print(f"   Found {len(detections)} region(s)")

    if not detections:
        print("   No food items detected!")
        return []

    # ------------------------------------------------------------------
    # Step 2: Depth
    # ------------------------------------------------------------------
    print("\n Step 2: Estimating depth…")
    if depth_estimator.use_metric_original:
        depth_map, depth_colored = depth_estimator.estimate_depth_metric(img_cv2)
        print(f"   Metric depth: {depth_map.min():.1f}–{depth_map.max():.1f} cm")
    else:
        depth_map, depth_colored = depth_estimator.estimate_depth(img_cv2)
        print(f"   Relative depth: {depth_map.min():.3f}–{depth_map.max():.3f}")

    if output_path:
        cv2.imwrite("depth_visualization.jpg", depth_colored)

    # ------------------------------------------------------------------
    # Step 3: Segment → Batch-classify → Volume → Nutrition
    # ------------------------------------------------------------------
    print(f"\n Step 3: Analysing {len(detections)} region(s)…")
    print("-" * 70)

    img_h, img_w = img_cv2.shape[:2]

    # Filter containers (plate, bowl, candle, …) before any heavy inference.
    food_detections = [d for d in detections if not d.get("is_container")]
    skipped_containers = len(detections) - len(food_detections)
    if skipped_containers:
        print(f"   Skipped {skipped_containers} container detection(s) (plate/bowl/candle/…)")

    # Pre-warm FastSAM cache — single full-image forward pass; all prompt calls
    # below will hit the cache instead of re-running FastSAM inference.
    if food_detections:
        segmenter.set_image(img_cv2)

    # ── Pass 1: Filter by bbox size, then batch-segment all candidates ────
    img_area = img_h * img_w

    # Size filter first (cheap) — skip sub-region boxes before segmentation
    candidate_dets = []
    for i, detection in enumerate(food_detections, 1):
        bbox = detection["bbox"]
        x1b, y1b, x2b, y2b = bbox
        bbox_area = max((x2b - x1b) * (y2b - y1b), 1)
        if bbox_area / img_area < _MIN_BBOX_RELATIVE_AREA:
            print(f"\n  Object {i}: Skipped (bbox too small: {bbox_area/img_area:.1%})")
        else:
            candidate_dets.append((i, detection, bbox, bbox_area))

    # Batch segmentation — single prompt() call for all candidate bboxes
    valid_items = []   # (detection, bbox, mask, mask_area_px, seg_score)
    if candidate_dets:
        all_bboxes = [bbox for _, _, bbox, _ in candidate_dets]
        print(f"\n  Segmenting {len(all_bboxes)} candidate(s) in batch…")
        batch_seg = segmenter.segment_batch(img_cv2, all_bboxes)
        for (i, detection, bbox, bbox_area), (mask, seg_score) in zip(candidate_dets, batch_seg):
            mask_area_px = segmenter.calculate_mask_area(mask)
            print(f"   Object {i}: mask {mask_area_px:,} px, seg_score {seg_score:.2f}")

            if seg_score < _MIN_SEG_SCORE:
                print(f"   Skipped (seg score {seg_score:.2f} < {_MIN_SEG_SCORE})")
                continue

            mask_coverage = mask_area_px / bbox_area
            if mask_coverage < _MIN_MASK_COVERAGE:
                print(f"   Skipped (mask coverage {mask_coverage:.1%} < {_MIN_MASK_COVERAGE:.0%})")
                continue

            valid_items.append((detection, bbox, mask, mask_area_px, seg_score))

    # Mask-IoU dedup — drop items whose segmentation overlaps an already-kept
    # mask by > _MASK_IOU_DEDUP. Catches the case where multiple YOLO boxes all
    # produce the same FastSAM full-image mask for a large food item (e.g. a
    # whole cake viewed from the side).  Sort by seg_score first so we keep
    # the highest-confidence segmentation.
    valid_items.sort(key=lambda x: x[4], reverse=True)
    deduped_items: list = []
    for item in valid_items:
        item_mask = item[2]
        if any(mask_iou(item_mask, kept[2]) >= _MASK_IOU_DEDUP for kept in deduped_items):
            print(f"   Mask-deduped: {item[3]:,} px (overlaps existing mask ≥ {_MASK_IOU_DEDUP:.0%})")
            continue
        deduped_items.append(item)
    valid_items = deduped_items

    if not valid_items:
        results = []
    else:
        # ── Pass 2: Batch-classify all masked crops in one GPU call ───────
        masked_pils = [_make_masked_crop(img_cv2, m, b) for _, b, m, _, _ in valid_items]
        print(f"\n  Classifying {len(masked_pils)} crop(s) in batch…")
        classifications = _classify_batch_with_fallback(classifier, clip_classifier, masked_pils)

        # ── Pass 3: Label pick → Volume → Nutrition ───────────────────────
        results = []

        for (detection, bbox, mask, mask_area_px, seg_score), classification in zip(valid_items, classifications):
            # Pass real YOLO detection so paths 0 & 2 can fire for specific labels.
            # Generic category labels (Food, Fruit, …) are suppressed — they are
            # valid detections but too vague to use as the final food name.
            _det_name_lower = detection.get("class_name", "").lower()
            det_for_label = (
                {"class_name": "", "confidence": 0.0}
                if _det_name_lower in _GENERIC_YOLO_CLASSES
                else detection
            )
            label_choice = _pick_label(det_for_label, classification, taxonomy, cls_conf_min=classification_conf)

            if not label_choice["accepted"]:
                reason = label_choice.get("reason_code", "rejected")
                conf   = float(label_choice.get("confidence", 0.0))
                print(f"   Skipped ({reason}, conf {conf:.1%})")
                continue

            top1_conf       = float(label_choice["confidence"])
            food_name       = label_choice["food_name"]
            canonical_id    = label_choice.get("canonical_id", "")
            canonical_name  = label_choice.get("canonical_name", food_name)
            canonical_cat   = label_choice.get("canonical_category", "default_unknown")
            predicted_label = label_choice.get("predicted_label", "")
            contour         = segmenter.get_mask_contour(mask)

            print(f"\n{'='*70}")
            print(f" FOOD ITEM #{len(results) + 1}: {food_name.upper()}")
            print(f"   Source: {label_choice['source']} ({top1_conf:.1%})")

            # Volume & weight
            tax_food = taxonomy.get_food(canonical_id) if canonical_id else None
            usda_weight_g = get_typical_serving_weight(food_name)
            if usda_weight_g is None and tax_food and tax_food.typical_serving_g:
                usda_weight_g = float(tax_food.typical_serving_g)

            print("   Calculating volume & weight…")
            volume_result = volume_calculator.calculate_volume_from_mask_and_depth(
                mask=mask,
                depth_map=depth_map,
                food_name=food_name,
                usda_weight_g=usda_weight_g,
                image_resolution=(img_h, img_w),
                canonical_id=canonical_id or None,
                category=canonical_cat,
                calibration_scale=calibration_scale,
                reject_low_quality=reject_low_quality,
                is_metric=(
                    depth_estimator.use_metric_original
                    and not depth_estimator._last_metric_failed
                ),
            )

            vol_ml       = volume_result["volume_ml"]
            weight_g     = volume_result["estimated_weight_g"]
            weight_low_g = volume_result.get("weight_low_g", weight_g)
            weight_high_g= volume_result.get("weight_high_g", weight_g)
            method       = volume_result["method"]
            vol_conf     = volume_result["confidence"]

            _bad_methods = {"failed", "usda_default_fallback"}
            if str(method).strip().lower() in _bad_methods or vol_ml <= 0 or weight_g <= 0:
                print(f"   Skipped (invalid estimate: {method}, vol={vol_ml:.1f}, w={weight_g:.1f})")
                continue

            print(f"   Volume: {vol_ml:.1f} ml")
            print(f"   Weight: {weight_g:.1f} g ({weight_low_g:.1f}–{weight_high_g:.1f} g)")
            print(f"   Method: {method} ({vol_conf:.1%})")

            print("   Nutrition…")
            _tax_queries = list(tax_food.usda_queries) if tax_food else []
            nutrition = get_nutrition_info(
                food_name,
                weight_g,
                canonical_id=canonical_id or None,
                category=canonical_cat,
                usda_queries=_tax_queries or None,
            )
            if not nutrition or nutrition.get("calories", 0) == 0:
                nutrition = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0, "fiber": 0}
                print("   (nutrition unavailable)")
            else:
                print(f"   {nutrition['calories']:.0f} kcal | "
                      f"P:{nutrition['protein']:.1f}g C:{nutrition['carbs']:.1f}g F:{nutrition['fat']:.1f}g")

            results.append({
                "food_name":                 food_name,
                "predicted_label":           predicted_label,
                "canonical_id":              canonical_id,
                "canonical_name":            canonical_name,
                "canonical_category":        canonical_cat,
                "reason_code":               label_choice.get("reason_code", ""),
                "bbox":                      bbox,
                "classification_confidence": top1_conf,
                "top3":                      label_choice["top3"],
                "label_source":              label_choice["source"],
                "mask":                      mask,
                "contour":                   contour,
                "mask_area_px":              mask_area_px,
                "volume_ml":                 vol_ml,
                "weight_g":                  weight_g,
                "weight_low_g":              weight_low_g,
                "weight_high_g":             weight_high_g,
                "weight_method":             method,
                "weight_confidence":         vol_conf,
                "quality_flags":             volume_result.get("quality_flags", []),
                "nutrition":                 nutrition,
                "volume_result":             volume_result,
            })

    # Post-processing
    results, dropped_failed = filter_failed_items(results)
    if dropped_failed:
        print(f"\n   Filtered {dropped_failed} failed estimate(s).")

    results, dropped_dupes = dedupe_items(
        results,
        same_label_iou_threshold=_DEDUP_SAME_LABEL_IOU,
        same_label_similarity_threshold=_DEDUP_SAME_LABEL_SIM,
        cross_label_iou_threshold=_DEDUP_CROSS_LABEL_IOU,
    )
    if dropped_dupes:
        print(f"   Filtered {dropped_dupes} overlapping duplicate(s).")

    results = _group_same_food_items(results)

    print("-" * 70)

    if output_path and results:
        overlay = create_visualization(img_cv2, results)
        cv2.imwrite(output_path, overlay)
        print(f"\n Saved: {output_path}")

    _print_meal_summary(results)
    return results


# ============================================================================
# Multi-image pipeline
# ============================================================================

def process_food_images(
    image_paths: List[str],
    output_path: Optional[str] = "output_final.jpg",
    detection_conf: float = 0.10,
    classification_conf: float = 0.60,
    reference_type: Optional[str] = None,
    reference_size_cm: Optional[float] = None,
    reject_low_quality: bool = False,
) -> List[Dict]:
    """Process 1–3 images and aggregate volumes via median."""
    if not image_paths:
        raise ValueError("At least one image path is required")

    if len(image_paths) == 1:
        return process_food_image(
            image_paths[0], output_path, detection_conf, classification_conf,
            reference_type=reference_type, reference_size_cm=reference_size_cm,
            reject_low_quality=reject_low_quality,
        )

    print("\n" + "=" * 70)
    print(f"MULTI-IMAGE FOOD ANALYSIS ({len(image_paths)} images)")
    print("=" * 70)

    all_results: List[List[Dict]] = []
    for i, path in enumerate(image_paths, 1):
        print(f"\n{'─'*70}")
        print(f"Image {i}/{len(image_paths)}: {path}")
        print("─" * 70)
        res = process_food_image(
            path, output_path=None,
            detection_conf=detection_conf,
            classification_conf=classification_conf,
            reference_type=reference_type,
            reference_size_cm=reference_size_cm,
            reject_low_quality=reject_low_quality,
        )
        all_results.append(res)

    print(f"\n{'='*70}")
    print("Aggregating volumes across images…")
    aggregated = _aggregate_results_across_images(all_results)

    if output_path and aggregated:
        img_cv2 = cv2.imread(image_paths[-1])
        if img_cv2 is not None:
            cv2.imwrite(output_path, create_visualization(img_cv2, aggregated))
            print(f"\n Saved: {output_path}")

    print("\n" + "=" * 70)
    print("AGGREGATED MEAL SUMMARY")
    print("=" * 70)
    _print_meal_summary(aggregated)
    return aggregated


def _aggregate_results_across_images(all_image_results: List[List[Dict]]) -> List[Dict]:
    grouped_items = group_items_across_images(all_image_results)
    aggregated = []
    for items in grouped_items:
        if len(items) == 1:
            item = items[0].copy()
            item["image_count"] = 1
            aggregated.append(item)
            continue
        food_name = items[0]["food_name"]
        volume_results = [item["volume_result"] for item in items]
        agg_vol = aggregate_multi_image_volumes(volume_results)
        agg_weight = agg_vol["estimated_weight_g"]
        nutrition = get_nutrition_info(
            food_name, agg_weight,
            canonical_id=items[0].get("canonical_id"),
            category=items[0].get("canonical_category"),
        )
        if not nutrition or nutrition.get("calories", 0) == 0:
            nutrition = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0, "fiber": 0}
        agg_item = items[0].copy()
        agg_item.update({
            "volume_ml":        agg_vol["volume_ml"],
            "weight_g":         agg_weight,
            "weight_low_g":     agg_vol.get("weight_low_g", agg_weight),
            "weight_high_g":    agg_vol.get("weight_high_g", agg_weight),
            "weight_method":    agg_vol.get("method", "multi_image_aggregate"),
            "weight_confidence":agg_vol.get("confidence", 0.0),
            "nutrition":        nutrition,
            "volume_result":    agg_vol,
            "image_count":      len(items),
        })
        aggregated.append(agg_item)
    return aggregated


# ============================================================================
# Helpers
# ============================================================================

def _group_same_food_items(results: List[Dict]) -> List[Dict]:
    """Merge multiple detections of the same food: Sausage + Sausage → Sausage ×2."""
    from collections import defaultdict
    groups: Dict[str, List[Dict]] = defaultdict(list)
    no_cid: List[Dict] = []
    first_seen: Dict[str, int] = {}

    for item in results:
        cid = str(item.get("canonical_id", "")).strip()
        if cid:
            if cid not in first_seen:
                first_seen[cid] = len(groups)
            groups[cid].append(item)
        else:
            no_cid.append(item)

    out: List[Optional[Dict]] = [None] * len(groups)
    for cid, items in groups.items():
        slot = first_seen[cid]
        if len(items) == 1:
            entry = items[0].copy()
            entry.setdefault("count", 1)
            out[slot] = entry
        else:
            count = len(items)
            best = max(items, key=lambda x: x.get("weight_confidence", 0))
            total_weight  = sum(x["weight_g"] for x in items)
            total_wl      = sum(x.get("weight_low_g",  x["weight_g"]) for x in items)
            total_wh      = sum(x.get("weight_high_g", x["weight_g"]) for x in items)
            total_vol     = sum(x["volume_ml"] for x in items)
            combined_nut  = get_nutrition_info(
                best["food_name"], total_weight,
                canonical_id=cid or None,
                category=best.get("canonical_category"),
            )
            if not combined_nut or combined_nut.get("calories", 0) == 0:
                pn = best["nutrition"]
                scale = total_weight / max(best["weight_g"], 1.0)
                combined_nut = {k: v * scale for k, v in pn.items()}
            entry = best.copy()
            entry.update({
                "count":       count,
                "weight_g":    total_weight,
                "weight_low_g":total_wl,
                "weight_high_g":total_wh,
                "volume_ml":   total_vol,
                "nutrition":   combined_nut,
            })
            out[slot] = entry

    return [x for x in out if x is not None] + no_cid


def _print_meal_summary(results: List[Dict]) -> None:
    if not results:
        print("\n  No food items detected.")
        return

    print(f"\n{'='*70}")
    print("MEAL SUMMARY")
    print("=" * 70)
    print(f"\n  Foods detected: {len(results)}")

    total_cal = sum(r["nutrition"].get("calories", 0) for r in results)
    total_prot= sum(r["nutrition"].get("protein",  0) for r in results)
    total_carb= sum(r["nutrition"].get("carbs",    0) for r in results)
    total_fat = sum(r["nutrition"].get("fat",      0) for r in results)
    total_wt  = sum(r["weight_g"] for r in results)

    print("\n Individual items:")
    for i, r in enumerate(results, 1):
        n = r["nutrition"]
        count = r.get("count", 1)
        tag   = f" ×{count}" if count > 1 else ""
        imgs  = f" ({r['image_count']} imgs)" if r.get("image_count", 1) > 1 else ""
        print(f"\n  {i}. {r['food_name'].title()}{tag}{imgs}")
        print(f"     Weight: {r['weight_g']:.0f}g | Calories: {n.get('calories',0):.0f} kcal")
        print(f"     P: {n.get('protein',0):.1f}g | C: {n.get('carbs',0):.1f}g | F: {n.get('fat',0):.1f}g")
        cls_conf = r.get('classification_confidence', 0.0)
        print(f"     Classification: {cls_conf:.0%} ({r.get('label_source','?')}) | Volume: {r['weight_confidence']:.0%} ({r['weight_method']})")

    print(f"\n{'='*70}")
    print(" TOTAL MEAL")
    print("=" * 70)
    print(f"   Total weight:   {total_wt:.0f} g")
    print(f"   Total calories: {total_cal:.0f} kcal")
    print(f"   Total protein:  {total_prot:.1f} g")
    print(f"   Total carbs:    {total_carb:.1f} g")
    print(f"   Total fat:      {total_fat:.1f} g")
    total_macros = total_prot * 4 + total_carb * 4 + total_fat * 9
    if total_macros > 0:
        print(f"\n   Macro split:")
        print(f"     Protein: {total_prot*4/total_macros*100:.0f}%")
        print(f"     Carbs:   {total_carb*4/total_macros*100:.0f}%")
        print(f"     Fat:     {total_fat*9/total_macros*100:.0f}%")
    print("\n" + "=" * 70)
    print(" Analysis complete!")
    print("=" * 70)


def create_visualization(img_cv2: np.ndarray, results: List[Dict]) -> np.ndarray:
    overlay = img_cv2.copy()
    img_h, img_w = img_cv2.shape[:2]
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
    ]
    LABEL_H, LABEL_W = 80, 285

    for i, r in enumerate(results):
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = r["bbox"]
        n = r["nutrition"]

        mask_overlay = np.zeros_like(img_cv2)
        mask_overlay[r["mask"]] = color
        overlay = cv2.addWeighted(overlay, 1, mask_overlay, 0.3, 0)

        if r.get("contour") is not None:
            cv2.drawContours(overlay, [r["contour"]], -1, color, 3)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        lx = max(0, min(x1, img_w - LABEL_W))
        if y1 >= LABEL_H:
            ly1, ly2 = y1 - LABEL_H, y1
            ty_name, ty_info1, ty_info2, ty_dot = y1-60, y1-35, y1-15, y1-40
        else:
            ly1, ly2 = y2, min(y2 + LABEL_H, img_h)
            ty_name, ty_info1, ty_info2, ty_dot = y2+20, y2+45, y2+65, y2+40

        cv2.rectangle(overlay, (lx, ly1), (lx + LABEL_W, ly2), color, -1)

        count     = r.get("count", 1)
        count_tag = f" \u00d7{count}" if count > 1 else ""
        imgs_tag  = f" {r['image_count']}img" if r.get("image_count", 1) > 1 else ""
        label = f"#{i} {r['food_name'].title()}{count_tag}{imgs_tag}"
        cv2.putText(overlay, label, (lx+5, ty_name), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        info1 = f"{r['weight_g']:.0f}g | {n.get('calories',0):.0f} kcal"
        cv2.putText(overlay, info1, (lx+5, ty_info1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        info2 = f"P:{n.get('protein',0):.0f}g  C:{n.get('carbs',0):.0f}g  F:{n.get('fat',0):.0f}g"
        cv2.putText(overlay, info2, (lx+5, ty_info2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

        dot_color = (0, 255, 0) if r.get("classification_confidence", 0) >= 0.70 else (0, 165, 255)
        cv2.circle(overlay, (lx + LABEL_W - 20, ty_dot), 8, dot_color, -1)

    return overlay


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = sys.argv[1:]
    if not args:
        image_paths = ["samples/food_2.jpg"]
        output_path = "output_final.jpg"
    elif len(args) == 1 and not args[0].startswith("-"):
        image_paths = [args[0]]
        output_path = "output_final.jpg"
    else:
        import os as _os
        if (
            args[-1].endswith((".jpg", ".jpeg", ".png"))
            and len(args) > 1
            and not _os.path.exists(args[-1])
        ):
            image_paths = args[:-1]
            output_path = args[-1]
        else:
            image_paths = args
            output_path = "output_final.jpg"

    print("\n" + "=" * 70)
    print("  FOOD ANALYSIS SYSTEM — COMMERCIAL GRADE")
    print("=" * 70)
    print(f"\nInput:  {image_paths}")
    print(f"Output: {output_path}\n")

    try:
        results = process_food_images(
            image_paths=image_paths,
            output_path=output_path,
        )
        print("\n SUCCESS! Check the output files.")
    except Exception as e:
        import traceback
        print(f"\n ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
