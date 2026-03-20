"""
Shared post-processing helpers for pipeline outputs.

These helpers are intentionally model-agnostic and operate on the plain
dict structures produced by main.py and api_server.py.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np


def mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    """Pixel-wise IoU between two boolean masks."""
    inter = int((m1 & m2).sum())
    union = int((m1 | m2).sum())
    return inter / union if union > 0 else 0.0


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _label_key(item: Dict) -> str:
    """Grouping key that combines canonical_id + specific food_name.

    Using canonical_id alone would merge two distinct dishes that share the
    same canonical bucket (e.g. "biryani" and "plain rice" both map to
    canonical_id="rice").  The composite key keeps them separate while still
    allowing the same dish seen from different angles to group together.
    """
    cid = str(item.get("canonical_id", "")).strip().lower()
    name = str(item.get("food_name", "")).strip().lower()
    if cid and name and name != cid:
        return f"{cid}:{name}"
    # Return empty string for truly unknown items so they are never treated as
    # "same label" and only dropped by the stricter cross-label IoU threshold.
    return cid or name or ""


def _bbox_tuple(bbox: Sequence[object]) -> Tuple[float, float, float, float] | None:
    if not bbox or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def bbox_iou(box1: Sequence[object], box2: Sequence[object]) -> float:
    a = _bbox_tuple(box1)
    b = _bbox_tuple(box2)
    if a is None or b is None:
        return 0.0

    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _bbox_center_and_area(bbox: Sequence[object]) -> Tuple[float, float, float] | None:
    b = _bbox_tuple(bbox)
    if b is None:
        return None
    cx = 0.5 * (b[0] + b[2])
    cy = 0.5 * (b[1] + b[3])
    area = (b[2] - b[0]) * (b[3] - b[1])
    return cx, cy, area


def _item_rank_score(item: Dict) -> Tuple[float, float, float, float]:
    # Prefer items with stronger weight/classification confidence and larger mask.
    return (
        _safe_float(item.get("weight_confidence", item.get("confidence", 0.0))),
        _safe_float(item.get("classification_confidence", 0.0)),
        _safe_float(item.get("confidence", 0.0)),
        _safe_float(item.get("mask_area_px", 0.0)),
    )


def is_failed_item(item: Dict) -> bool:
    volume_result = item.get("volume_result")
    if isinstance(volume_result, dict):
        method = str(volume_result.get("method", "")).strip().lower()
        if method in {"failed", "usda_default_fallback"}:
            return True
        if _safe_float(volume_result.get("estimated_weight_g"), 0.0) <= 0.0:
            return True
        if _safe_float(volume_result.get("volume_ml"), 0.0) <= 0.0:
            return True

    if _safe_float(item.get("weight_g"), 0.0) <= 0.0:
        return True
    if _safe_float(item.get("volume_ml"), 0.0) <= 0.0:
        return True
    return False


def filter_failed_items(items: List[Dict]) -> Tuple[List[Dict], int]:
    kept: List[Dict] = []
    dropped = 0
    for item in items:
        if is_failed_item(item):
            dropped += 1
            continue
        kept.append(item)
    return kept, dropped


def dedupe_items(
    items: List[Dict],
    *,
    same_label_iou_threshold: float = 0.45,
    same_label_similarity_threshold: float = 0.62,
    cross_label_iou_threshold: float = 0.88,
) -> Tuple[List[Dict], int]:
    if len(items) <= 1:
        return items[:], 0

    ordered = sorted(items, key=_item_rank_score, reverse=True)
    kept: List[Dict] = []
    dropped = 0

    for item in ordered:
        item_bbox = item.get("bbox", [])
        item_key = _label_key(item)
        dominated = False

        item_cid = str(item.get("canonical_id", "")).strip().lower()
        for chosen in kept:
            chosen_bbox = chosen.get("bbox", [])
            iou = bbox_iou(item_bbox, chosen_bbox)
            chosen_key = _label_key(chosen)
            # Same label if composite key matches OR if both share the same
            # canonical_id (handles cross-detector aliases like YOLO "hot dog"
            # and GDINO "sausage" which both resolve to canonical_id="sausage").
            chosen_cid = str(chosen.get("canonical_id", "")).strip().lower()
            same_canonical = bool(item_cid and chosen_cid and item_cid == chosen_cid)
            same_label = bool(item_key and chosen_key and item_key == chosen_key) or same_canonical
            similarity = _similarity_for_grouping(item, chosen)
            if (same_label and (iou >= same_label_iou_threshold or similarity >= same_label_similarity_threshold)) or (
                not same_label and iou >= cross_label_iou_threshold
            ):
                dominated = True
                break

        if dominated:
            dropped += 1
            continue
        kept.append(item)

    return kept, dropped


def _similarity_for_grouping(item: Dict, anchor: Dict) -> float:
    iou = bbox_iou(item.get("bbox", []), anchor.get("bbox", []))
    a = _bbox_center_and_area(item.get("bbox", []))
    b = _bbox_center_and_area(anchor.get("bbox", []))
    if a is None or b is None:
        return iou

    dx = a[0] - b[0]
    dy = a[1] - b[1]
    distance = math.hypot(dx, dy)
    # Normalise by a rough per-object scale so large items allow larger drift.
    scale = math.sqrt(max(a[2], b[2], 1.0))
    center_score = max(0.0, 1.0 - (distance / (2.5 * scale)))
    area_score = min(a[2], b[2]) / max(a[2], b[2], 1.0)
    return 0.55 * iou + 0.30 * center_score + 0.15 * area_score


def group_items_across_images(
    all_image_results: List[List[Dict]],
    *,
    similarity_threshold: float = 0.33,
) -> List[List[Dict]]:
    """
    Group same-food items across images with similarity matching.

    This avoids strict "occurrence index" coupling, which is brittle when
    detection order changes between views.
    """
    groups: List[Dict] = []

    for image_index, image_items in enumerate(all_image_results):
        for item in image_items:
            key = _label_key(item)
            if not key:
                # Unknown labels are not safe to merge across views.
                groups.append({"key": f"unknown_{len(groups)}", "images": {image_index}, "items": [item]})
                continue

            best_group = None
            best_score = -1.0
            for group in groups:
                if group["key"] != key:
                    continue
                if image_index in group["images"]:
                    continue
                anchor = group["items"][0]
                score = _similarity_for_grouping(item, anchor)
                if score > best_score:
                    best_score = score
                    best_group = group

            if best_group is not None and best_score >= similarity_threshold:
                best_group["items"].append(item)
                best_group["images"].add(image_index)
            else:
                groups.append({"key": key, "images": {image_index}, "items": [item]})

    return [g["items"] for g in groups]
