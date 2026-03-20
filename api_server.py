"""
FastAPI server for the food analysis pipeline.

Pipeline (via main.py):
  detect (YOLOv8l 205-class) → segment (FastSAM) → classify (EfficientNet/CLIP)
  → depth (DepthAnything V2 metric) → volume → nutrition (USDA)
"""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional
import io
import json
import os
import threading
import traceback

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image, ImageOps

import main as _pipeline                              # shared models + helpers
from core.blur_detector import detect_blur
from core.food_taxonomy import get_taxonomy
from core.pipeline_postprocess import dedupe_items, filter_failed_items, group_items_across_images, mask_iou
from core.volume_estimator import aggregate_multi_image_volumes
from data.usda_nutrition_lookup import (
    _has_usda_key,
    get_nutrition_info,
    get_typical_serving_weight,
    get_usda_cache_status,
)


# ---------------------------------------------------------------------------
# Pipeline-level counters + concurrency control
# ---------------------------------------------------------------------------

# _pipeline_lock is a no-op context manager: the old threading.RLock was a
# C-3 bottleneck (serialised every async request including health checks).
# GPU serialisation is now handled by _gpu_semaphore in the async endpoint,
# which suspends waiting requests in the async queue without blocking threads.
# Set GPU_CONCURRENCY=2 only if VRAM comfortably fits two concurrent requests.
_pipeline_lock: contextlib.AbstractContextManager = contextlib.nullcontext()
_gpu_semaphore: asyncio.Semaphore | None = None   # created in lifespan
_pipeline_counters: Dict[str, int] = {
    "detected_raw": 0,
    "accepted_items": 0,
    "rejected_items": 0,
    "failed_estimates": 0,
    "deduped_items": 0,
    "usda_api_hits": 0,
}


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _log_event(event: str, **fields: object) -> None:
    payload = {"event": event, **fields}
    try:
        print(json.dumps(payload, ensure_ascii=True))
    except Exception:
        print(f"[event={event}] {fields}")


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineSettings:
    detection_conf_default: float
    classification_conf_default: float
    classification_margin_min: float
    min_bbox_relative_area: float
    mask_iou_dedup: float
    detector_label_conf_min: float
    container_conf_min: float
    min_mask_area_px: int
    min_mask_coverage: float
    min_seg_score: float
    blur_threshold: float
    allowed_origins: List[str]
    cors_allow_credentials: bool
    reject_low_quality: bool
    dedup_same_label_iou: float
    dedup_same_label_similarity: float
    dedup_cross_label_iou: float


def _load_settings() -> PipelineSettings:
    origins_raw = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173",
    )
    origins = [x.strip() for x in origins_raw.split(",") if x.strip()]
    if not origins:
        origins = ["http://localhost:3000"]

    return PipelineSettings(
        detection_conf_default=_clamp01(_env_float("DETECTION_CONF_DEFAULT", 0.20)),
        min_bbox_relative_area=max(0.0, _env_float("MIN_BBOX_RELATIVE_AREA", 0.02)),
        mask_iou_dedup=max(0.0, _env_float("MASK_IOU_DEDUP", 0.60)),
        classification_conf_default=_clamp01(_env_float("CLASSIFICATION_CONF_DEFAULT", 0.60)),
        classification_margin_min=max(0.0, _env_float("CLASSIFICATION_MARGIN_MIN", 0.15)),
        detector_label_conf_min=_clamp01(_env_float("DETECTOR_LABEL_CONF_MIN", 0.35)),
        container_conf_min=_clamp01(_env_float("CONTAINER_CONF_MIN", 0.65)),
        min_mask_area_px=max(1, _env_int("MIN_MASK_AREA_PX", 1500)),
        min_mask_coverage=max(0.0, _env_float("MIN_MASK_COVERAGE", 0.12)),
        min_seg_score=max(0.0, _env_float("MIN_SEG_SCORE", 0.45)),
        blur_threshold=_env_float("BLUR_THRESHOLD", 10.0),
        allowed_origins=origins,
        cors_allow_credentials=_env_bool("CORS_ALLOW_CREDENTIALS", True),
        reject_low_quality=_env_bool("REJECT_LOW_QUALITY_ESTIMATES", True),
        dedup_same_label_iou=_env_float("DEDUP_SAME_LABEL_IOU", 0.45),
        dedup_same_label_similarity=_env_float("DEDUP_SAME_LABEL_SIM", 0.62),
        dedup_cross_label_iou=_env_float("DEDUP_CROSS_LABEL_IOU", 0.88),
    )


SETTINGS = _load_settings()


# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _gpu_semaphore
    concurrency = int(os.getenv("GPU_CONCURRENCY", "1"))
    _gpu_semaphore = asyncio.Semaphore(concurrency)
    print(f"Loading pipeline models (GPU_CONCURRENCY={concurrency})...")
    _pipeline.load_models()          # initialises FoodDetector, CLIP, SAM, Depth, Volume
    print("All models loaded.")
    yield


app = FastAPI(title="Food Analysis API", version="4.0", lifespan=lifespan)

_allow_credentials = SETTINGS.cors_allow_credentials
if "*" in SETTINGS.allowed_origins and _allow_credentials:
    print("WARNING: wildcard CORS origin with credentials is invalid. Disabling credentials.")
    _allow_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=SETTINGS.allowed_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Core analysis (shared by /analyze and /analyze_multi)
# ---------------------------------------------------------------------------

def _analyze_image_array(
    img_cv2: np.ndarray,
    img_pil: Image.Image,
    detection_conf: float,
    classification_conf: float,
    reference_type: Optional[str] = None,
    reference_size_cm: Optional[float] = None,
) -> tuple:  # (List[Dict], Dict) — results + per-request diagnostics
    diag: Dict = {
        "detected_raw": 0,
        "label_rejected": 0,
        "container_rejected": 0,
        "mask_too_small": 0,
        "mask_coverage_low": 0,
        "volume_failed": 0,
        "deduped": 0,
        "label_source_counts": {},
        "volume_method_counts": {},
        "nutrition_source_counts": {},
        "weight_confidence_counts": {"high": 0, "medium": 0, "low": 0},
    }

    with _pipeline_lock:
        detector, classifier, clip_classifier, segmenter, depth_estimator, volume_calculator = (
            _pipeline._get_models()
        )
        taxonomy = get_taxonomy()
        calibration_scale = _pipeline._compute_calibration_scale(reference_type, reference_size_cm)

        # Detection: YOLOv8l (205-class combined dataset)
        detections = detector.detect(img_cv2, yolo_conf=detection_conf)
        diag["detected_raw"] = len(detections)
        _pipeline_counters["detected_raw"] += len(detections)
        _log_event("detections", count=len(detections), conf=detection_conf)
        if not detections:
            return [], diag

        # Depth estimation
        if depth_estimator.use_metric_original:
            depth_map, _ = depth_estimator.estimate_depth_metric(img_cv2)
        else:
            depth_map, _ = depth_estimator.estimate_depth(img_cv2)

        img_h, img_w = img_cv2.shape[:2]
        results: List[Dict] = []
        failed_estimates = 0

        # Filter containers (plate, bowl, candle, …) before heavy inference.
        food_detections = [d for d in detections if not d.get("is_container")]
        n_containers = len(detections) - len(food_detections)
        if n_containers:
            diag["container_rejected"] += n_containers
            _pipeline_counters["rejected_items"] += n_containers

        # Pre-warm FastSAM cache — single GPU forward pass; all prompts below hit cache.
        if food_detections:
            segmenter.set_image(img_cv2)

        # ── Pass 1: Size filter (cheap), then batch-segment all candidates ──
        img_area = img_h * img_w
        candidate_dets = []
        for detection in food_detections:
            bbox = detection["bbox"]
            x1b, y1b, x2b, y2b = bbox
            bbox_area = max((x2b - x1b) * (y2b - y1b), 1)
            if bbox_area / img_area < SETTINGS.min_bbox_relative_area:
                _pipeline_counters["rejected_items"] += 1
            else:
                candidate_dets.append((detection, bbox, bbox_area))

        valid_items = []
        if candidate_dets:
            all_bboxes = [bbox for _, bbox, _ in candidate_dets]
            batch_seg = segmenter.segment_batch(img_cv2, all_bboxes)
            for (detection, bbox, bbox_area), (mask, seg_score) in zip(candidate_dets, batch_seg):
                if seg_score < SETTINGS.min_seg_score:
                    diag["mask_too_small"] += 1
                    _pipeline_counters["rejected_items"] += 1
                    continue
                mask_area_px = segmenter.calculate_mask_area(mask)
                if mask_area_px < SETTINGS.min_mask_area_px:
                    diag["mask_too_small"] += 1
                    _pipeline_counters["rejected_items"] += 1
                    continue
                if mask_area_px / bbox_area < SETTINGS.min_mask_coverage:
                    diag["mask_coverage_low"] += 1
                    _pipeline_counters["rejected_items"] += 1
                    continue
                valid_items.append((detection, bbox, mask, mask_area_px, seg_score))

        # Mask-IoU dedup — keep highest seg_score mask when two overlap ≥ threshold.
        valid_items.sort(key=lambda x: x[4], reverse=True)
        deduped: list = []
        for item in valid_items:
            if any(mask_iou(item[2], k[2]) >= SETTINGS.mask_iou_dedup for k in deduped):
                _pipeline_counters["rejected_items"] += 1
                continue
            deduped.append(item)
        valid_items = deduped

        # ── Pass 2: Batch-classify all masked crops in one GPU call ──────────
        if valid_items:
            masked_pils = [_pipeline._make_masked_crop(img_cv2, m, b) for _, b, m, _, _ in valid_items]
            classifications = _pipeline._classify_batch_with_fallback(
                classifier, clip_classifier, masked_pils
            )
        else:
            classifications = []

        # ── Pass 3: Label pick → volume → nutrition ───────────────────────────
        for (detection, bbox, mask, mask_area_px, _seg_score), classification in zip(valid_items, classifications):
            # Suppress generic YOLO class names (e.g. "food", "fruit") that are
            # too vague to use as final labels, but pass specific names through
            # so Path 0/1 strong-detector overrides can fire in the API too.
            _det_name_lower = detection.get("class_name", "").lower()
            det_for_label = (
                {"class_name": "", "confidence": 0.0}
                if _det_name_lower in _pipeline._GENERIC_YOLO_CLASSES
                else detection
            )
            label_choice = _pipeline._pick_label(
                det_for_label, classification, taxonomy,
                cls_conf_min=classification_conf,
            )
            if not label_choice["accepted"]:
                diag["label_rejected"] += 1
                _pipeline_counters["rejected_items"] += 1
                continue

            top1_conf = float(label_choice["confidence"])

            src = label_choice.get("source", "unknown")
            diag["label_source_counts"][src] = diag["label_source_counts"].get(src, 0) + 1

            canonical_id       = label_choice.get("canonical_id", "")
            canonical_name     = label_choice.get("canonical_name", label_choice["food_name"])
            canonical_category = label_choice.get("canonical_category", "default_unknown")
            predicted_label    = label_choice.get("predicted_label", "")
            food_name          = canonical_name.lower()
            tax_food           = taxonomy.get_food(canonical_id) if canonical_id else None

            # Volume & weight
            usda_weight_g = get_typical_serving_weight(food_name)
            if usda_weight_g is None and tax_food and tax_food.typical_serving_g:
                usda_weight_g = float(tax_food.typical_serving_g)

            volume_result = volume_calculator.calculate_volume_from_mask_and_depth(
                mask=mask,
                depth_map=depth_map,
                food_name=food_name,
                usda_weight_g=usda_weight_g,
                image_resolution=(img_h, img_w),
                canonical_id=canonical_id or None,
                category=canonical_category,
                calibration_scale=calibration_scale,
                reject_low_quality=SETTINGS.reject_low_quality,
                is_metric=(
                    depth_estimator.use_metric_original
                    and not depth_estimator._last_metric_failed
                ),
            )

            method    = str(volume_result.get("method", ""))
            weight_g  = float(volume_result.get("estimated_weight_g", 0.0))
            volume_ml = float(volume_result.get("volume_ml", 0.0))
            _failed_methods = {"failed", "usda_default_fallback"}
            if method.lower() in _failed_methods or weight_g <= 0 or volume_ml <= 0:
                failed_estimates += 1
                diag["volume_failed"] += 1
                continue

            method_key = method.split("+")[0].strip() if "+" in method else method
            diag["volume_method_counts"][method_key] = (
                diag["volume_method_counts"].get(method_key, 0) + 1
            )

            # Nutrition
            _tax_queries = list(tax_food.usda_queries) if tax_food else []
            nutrition = get_nutrition_info(
                food_name,
                weight_g,
                canonical_id=canonical_id or None,
                category=canonical_category,
                usda_queries=_tax_queries or None,
            )
            nsrc = (nutrition or {}).get("nutrition_source", "Unknown") or "Unknown"
            diag["nutrition_source_counts"][nsrc] = (
                diag["nutrition_source_counts"].get(nsrc, 0) + 1
            )
            if nsrc in {"USDA API", "USDA cache"}:
                _pipeline_counters["usda_api_hits"] += 1
            if not nutrition or nutrition.get("calories", 0) == 0:
                nutrition = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0, "fiber": 0}

            wc = float(volume_result.get("confidence", 0.0))
            if wc >= 0.75:
                diag["weight_confidence_counts"]["high"] += 1
            elif wc >= 0.55:
                diag["weight_confidence_counts"]["medium"] += 1
            else:
                diag["weight_confidence_counts"]["low"] += 1

            results.append({
                "food_name":          food_name,
                "predicted_label":    predicted_label,
                "canonical_id":       canonical_id,
                "canonical_name":     canonical_name,
                "canonical_category": canonical_category,
                "reason_code":        label_choice.get("reason_code", ""),
                "confidence":         round(top1_conf, 3),
                "top3":               label_choice["top3"],
                "label_source":       label_choice["source"],
                "bbox":               bbox,
                "mask_area_px":       int(mask_area_px),
                "volume_ml":          round(volume_ml, 1),
                "weight_g":           round(weight_g, 1),
                "weight_low_g":       round(float(volume_result.get("weight_low_g",  weight_g)), 1),
                "weight_high_g":      round(float(volume_result.get("weight_high_g", weight_g)), 1),
                "method":             method,
                "weight_confidence":  round(wc, 3),
                "quality_flags":      volume_result.get("quality_flags", []),
                "nutrition":          nutrition,
                "nutrition_source":   nsrc,
                "volume_result":      volume_result,
            })

        results, dropped_failed = filter_failed_items(results)
        failed_estimates += dropped_failed
        diag["volume_failed"] += dropped_failed
        results, dropped_dupes = dedupe_items(
            results,
            same_label_iou_threshold=SETTINGS.dedup_same_label_iou,
            same_label_similarity_threshold=SETTINGS.dedup_same_label_similarity,
            cross_label_iou_threshold=SETTINGS.dedup_cross_label_iou,
        )
        diag["deduped"] = dropped_dupes

        _pipeline_counters["failed_estimates"] += failed_estimates
        _pipeline_counters["deduped_items"]    += dropped_dupes
        _pipeline_counters["accepted_items"]   += len(results)
        _log_event(
            "postprocess",
            accepted=len(results),
            failed=failed_estimates,
            deduped=dropped_dupes,
        )
        return results, diag


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------

def _to_api_items(results: List[Dict]) -> List[Dict]:
    _skip = {"volume_result"}
    return [{k: v for k, v in row.items() if k not in _skip} for row in results]


def _build_pipeline_diagnostics(diag: Dict, final_accepted: int) -> Dict:
    def pct(n: int, total: int) -> float:
        return round(100.0 * n / max(total, 1), 1)

    detected  = diag.get("detected_raw", 0)
    label_pass = detected - diag.get("label_rejected", 0)
    seg_pass   = label_pass - (
        diag.get("container_rejected", 0)
        + diag.get("mask_too_small", 0)
        + diag.get("mask_coverage_low", 0)
    )
    lsc       = diag.get("label_source_counts", {})
    lsc_total = max(sum(lsc.values()), 1)
    vmc       = diag.get("volume_method_counts", {})
    nsc       = diag.get("nutrition_source_counts", {})
    nsc_total = max(sum(nsc.values()), 1)
    wcc       = diag.get("weight_confidence_counts", {"high": 0, "medium": 0, "low": 0})
    wcc_total = max(sum(wcc.values()), 1)

    return {
        "detection": {
            "raw_detections":  detected,
            "label_pass":      label_pass,
            "label_rejected":  diag.get("label_rejected", 0),
            "pass_rate_pct":   pct(label_pass, detected),
        },
        "segmentation": {
            "container_rejected": diag.get("container_rejected", 0),
            "mask_too_small":     diag.get("mask_too_small", 0),
            "mask_coverage_low":  diag.get("mask_coverage_low", 0),
            "passed":             seg_pass,
        },
        "classification": {
            "breakdown": {
                k: {"count": v, "pct": pct(v, lsc_total)}
                for k, v in sorted(lsc.items(), key=lambda x: -x[1])
            },
        },
        "volume": {
            "methods": {
                k: {"count": v, "pct": pct(v, max(seg_pass, 1))}
                for k, v in sorted(vmc.items(), key=lambda x: -x[1])
            },
            "failed": diag.get("volume_failed", 0),
        },
        "nutrition": {
            "sources": {
                k: {"count": v, "pct": pct(v, nsc_total)}
                for k, v in sorted(nsc.items(), key=lambda x: -x[1])
            },
        },
        "weight_confidence": {
            "high_pct":   pct(wcc.get("high",   0), wcc_total),
            "medium_pct": pct(wcc.get("medium", 0), wcc_total),
            "low_pct":    pct(wcc.get("low",    0), wcc_total),
            "counts":     dict(wcc),
        },
        "overall": {
            "detected_raw":       detected,
            "final_accepted":     final_accepted,
            "deduped":            diag.get("deduped", 0),
            "acceptance_rate_pct": pct(final_accepted, max(detected, 1)),
        },
    }


def _merge_diags(diags: List[Dict]) -> Dict:
    merged: Dict = {
        "detected_raw": 0,
        "label_rejected": 0,
        "container_rejected": 0,
        "mask_too_small": 0,
        "mask_coverage_low": 0,
        "volume_failed": 0,
        "deduped": 0,
        "label_source_counts": {},
        "volume_method_counts": {},
        "nutrition_source_counts": {},
        "weight_confidence_counts": {"high": 0, "medium": 0, "low": 0},
    }
    for d in diags:
        for key in (
            "detected_raw", "label_rejected", "container_rejected",
            "mask_too_small", "mask_coverage_low", "volume_failed", "deduped",
        ):
            merged[key] += d.get(key, 0)
        for sub in ("label_source_counts", "volume_method_counts", "nutrition_source_counts"):
            for k, v in d.get(sub, {}).items():
                merged[sub][k] = merged[sub].get(k, 0) + v
        for tier in ("high", "medium", "low"):
            merged["weight_confidence_counts"][tier] += (
                d.get("weight_confidence_counts", {}).get(tier, 0)
            )
    return merged


def _aggregate_api_results(all_results: List[List[Dict]]) -> List[Dict]:
    taxonomy = get_taxonomy()
    grouped  = group_items_across_images(all_results)

    aggregated: List[Dict] = []
    for items in grouped:
        if len(items) == 1:
            item = items[0].copy()
            item["image_count"] = 1
            aggregated.append(item)
            continue

        agg_vol    = aggregate_multi_image_volumes([x["volume_result"] for x in items])
        agg_weight = float(agg_vol.get("estimated_weight_g", 0.0))
        agg_volume = float(agg_vol.get("volume_ml", 0.0))
        if agg_weight <= 0 or agg_volume <= 0:
            _pipeline_counters["failed_estimates"] += 1
            continue

        food_name          = items[0]["food_name"]
        predicted_label    = items[0].get("predicted_label", "")
        canonical_id       = items[0].get("canonical_id")
        canonical_category = items[0].get("canonical_category")
        tax_food           = taxonomy.get_food(canonical_id) if canonical_id else None
        _tax_queries: list = list(tax_food.usda_queries) if tax_food else []
        if predicted_label and predicted_label.lower() not in {
            food_name, *[q.lower() for q in _tax_queries]
        }:
            _tax_queries = [predicted_label] + _tax_queries
        nutrition = get_nutrition_info(
            predicted_label or food_name,
            agg_weight,
            canonical_id=canonical_id,
            category=canonical_category,
            usda_queries=_tax_queries or None,
        )
        if not nutrition or nutrition.get("calories", 0) == 0:
            nutrition = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0, "fiber": 0}

        out = items[0].copy()
        out["volume_ml"]        = round(agg_volume, 1)
        out["weight_g"]         = round(agg_weight, 1)
        out["weight_low_g"]     = round(float(agg_vol.get("weight_low_g",  agg_weight)), 1)
        out["weight_high_g"]    = round(float(agg_vol.get("weight_high_g", agg_weight)), 1)
        out["method"]           = agg_vol.get("method", "multi_image_aggregate")
        out["weight_confidence"] = round(float(agg_vol.get("confidence", 0.0)), 3)
        out["nutrition"]        = nutrition
        out["volume_result"]    = agg_vol
        out["image_count"]      = len(items)
        aggregated.append(out)

    aggregated, dropped_failed = filter_failed_items(aggregated)
    aggregated, dropped_dupes  = dedupe_items(
        aggregated,
        same_label_iou_threshold=SETTINGS.dedup_same_label_iou,
        same_label_similarity_threshold=SETTINGS.dedup_same_label_similarity,
        cross_label_iou_threshold=SETTINGS.dedup_cross_label_iou,
    )
    _pipeline_counters["failed_estimates"] += dropped_failed
    _pipeline_counters["deduped_items"]    += dropped_dupes
    return aggregated


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {"status": "ok", "message": "Food Analysis API v4.0"}


@app.get("/health")
async def health():
    cache_status = get_usda_cache_status()
    accepted  = max(_pipeline_counters.get("accepted_items", 0), 1)
    hit_rate  = float(_pipeline_counters.get("usda_api_hits", 0)) / accepted
    det, _, _, _, depth_est, _ = _pipeline._get_models()
    return {
        "status":           "ok",
        "models_loaded":    det is not None,
        "detector_backend": "yolov8l_combined",
        "metric_depth":     depth_est.use_metric_original if depth_est else False,
        "taxonomy_version": get_taxonomy().version,
        "usda_available":   _has_usda_key(),
        "usda_cache_status": cache_status,
        "pipeline_counters": dict(_pipeline_counters),
        "usda_api_hit_rate": round(hit_rate, 3),
    }


@app.post("/analyze")
async def analyze_food(
    file: UploadFile = File(...),
    detection_conf: float = SETTINGS.detection_conf_default,
    classification_conf: float = SETTINGS.classification_conf_default,
    reference_type: Optional[str] = None,
    reference_size_cm: Optional[float] = None,
):
    try:
        detection_conf      = _clamp01(detection_conf)
        classification_conf = _clamp01(classification_conf)

        contents = await file.read()
        nparr    = np.frombuffer(contents, np.uint8)
        raw_cv2  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if raw_cv2 is None:
            raise HTTPException(status_code=400, detail="Invalid or unreadable image.")

        # Apply EXIF orientation so cv2 and PIL share the same orientation.
        img_pil = ImageOps.exif_transpose(Image.open(io.BytesIO(contents)))
        img_cv2 = cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

        blur_score, is_blurry = detect_blur(img_cv2, threshold=SETTINGS.blur_threshold)
        if is_blurry:
            return JSONResponse({
                "success":    False,
                "food_items": [],
                "blur_score": round(float(blur_score), 2),
                "message":    "Image is too blurry. Please retake a sharper photo.",
            })

        sem = _gpu_semaphore or asyncio.Semaphore(1)
        async with sem:
            results, diag = await run_in_threadpool(
                _analyze_image_array,
                img_cv2, img_pil, detection_conf, classification_conf,
                reference_type, reference_size_cm,
            )
        if not results:
            return JSONResponse({
                "success":              False,
                "food_items":           [],
                "pipeline_diagnostics": _build_pipeline_diagnostics(diag, 0),
                "message":              "No food detected or confidence too low.",
            })

        output_items   = _to_api_items(results)
        total_weight   = sum(float(r["weight_g"]) for r in results)
        total_calories = sum(float(r["nutrition"].get("calories", 0.0)) for r in results)
        return JSONResponse({
            "success":              True,
            "food_items":           output_items,
            "count":                len(output_items),
            "total_weight_g":       round(total_weight, 1),
            "total_calories":       round(total_calories, 1),
            "reference_type":       reference_type,
            "reference_size_cm":    reference_size_cm,
            "pipeline_diagnostics": _build_pipeline_diagnostics(diag, len(output_items)),
            "message":              f"Found {len(output_items)} food item(s).",
        })
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/analyze_multi")
async def analyze_food_multi(
    files: List[UploadFile] = File(...),
    detection_conf: float = SETTINGS.detection_conf_default,
    classification_conf: float = SETTINGS.classification_conf_default,
    reference_type: Optional[str] = None,
    reference_size_cm: Optional[float] = None,
):
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")
    if len(files) > 3:
        raise HTTPException(status_code=400, detail="Maximum 3 files per request.")

    try:
        detection_conf      = _clamp01(detection_conf)
        classification_conf = _clamp01(classification_conf)

        all_results: List[List[Dict]] = []
        all_diags:   List[Dict]       = []
        blurry_count  = 0
        invalid_count = 0

        for upload in files:
            contents = await upload.read()
            nparr    = np.frombuffer(contents, np.uint8)
            raw_cv2  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if raw_cv2 is None:
                invalid_count += 1
                continue

            img_pil = ImageOps.exif_transpose(Image.open(io.BytesIO(contents)))
            img_cv2 = cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

            _, is_blurry = detect_blur(img_cv2, threshold=SETTINGS.blur_threshold)
            if is_blurry:
                blurry_count += 1
                continue

            sem = _gpu_semaphore or asyncio.Semaphore(1)
            async with sem:
                results, img_diag = await run_in_threadpool(
                    _analyze_image_array,
                    img_cv2, img_pil, detection_conf, classification_conf,
                    reference_type, reference_size_cm,
                )
            all_results.append(results)
            all_diags.append(img_diag)

        merged_diag = _merge_diags(all_diags)

        if blurry_count + invalid_count == len(files):
            return JSONResponse({
                "success":              False,
                "food_items":           [],
                "images_invalid":       invalid_count,
                "images_skipped_blurry": blurry_count,
                "pipeline_diagnostics": _build_pipeline_diagnostics(merged_diag, 0),
                "message":              "All images were too blurry or unreadable.",
            })
        if not any(all_results):
            return JSONResponse({
                "success":              False,
                "food_items":           [],
                "pipeline_diagnostics": _build_pipeline_diagnostics(merged_diag, 0),
                "message":              "No food detected in any of the provided images.",
            })

        aggregated     = _aggregate_api_results(all_results)
        output_items   = _to_api_items(aggregated)
        total_weight   = sum(float(r["weight_g"]) for r in aggregated)
        total_calories = sum(float(r["nutrition"].get("calories", 0.0)) for r in aggregated)
        sharp_count    = len(files) - blurry_count - invalid_count
        skipped_notes  = []
        if blurry_count:
            skipped_notes.append(f"{blurry_count} blurry")
        if invalid_count:
            skipped_notes.append(f"{invalid_count} unreadable")
        skip_suffix = f" ({', '.join(skipped_notes)} image(s) skipped)." if skipped_notes else "."
        return JSONResponse({
            "success":               True,
            "food_items":            output_items,
            "count":                 len(output_items),
            "images_processed":      sharp_count,
            "images_skipped_blurry": blurry_count,
            "images_invalid":        invalid_count,
            "total_weight_g":        round(total_weight, 1),
            "total_calories":        round(total_calories, 1),
            "reference_type":        reference_type,
            "reference_size_cm":     reference_size_cm,
            "pipeline_diagnostics":  _build_pipeline_diagnostics(merged_diag, len(output_items)),
            "message":               f"Found {len(output_items)} food item(s) from {sharp_count} image(s){skip_suffix}",
        })
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
