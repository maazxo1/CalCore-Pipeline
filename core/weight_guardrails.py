"""
Weight guardrails for safety-first single-image estimates.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from core.food_taxonomy import get_taxonomy


def derive_weight_range(weight_g: float, confidence: float) -> Dict[str, float]:
    # Wider interval for lower confidence.
    conf = float(np.clip(confidence, 0.0, 1.0))
    spread = 0.45 - 0.30 * conf  # 45% at low conf -> 15% at high conf
    low = float(max(0.0, weight_g * (1.0 - spread)))
    high = float(max(low, weight_g * (1.0 + spread)))
    return {"weight_low_g": low, "weight_high_g": high}


def quality_flags(
    original_weight_g: float,
    bounded_weight_g: float,
    confidence: float,
    guardrail_applied: bool,
) -> list[str]:
    flags: list[str] = []
    if guardrail_applied:
        flags.append("guardrail_clamped")
        if original_weight_g > 0 and bounded_weight_g <= original_weight_g * 0.5:
            flags.append("severe_clamp")
    if confidence < 0.55:
        flags.append("low_confidence")
    if guardrail_applied and confidence < 0.55:
        flags.append("low_quality")
    return flags


def apply_weight_bounds(
    weight_g: float,
    *,
    canonical_id: Optional[str] = None,
    category: Optional[str] = None,
    typical_serving_g: Optional[float] = None,
    confidence: float = 0.0,
    calibration_used: bool = False,
    reject_low_quality: bool = False,
) -> Dict:
    taxonomy = get_taxonomy()
    food = taxonomy.get_food(canonical_id or "")

    category_name = category or (food.category if food else "default_unknown")
    serving = typical_serving_g if typical_serving_g else (food.typical_serving_g if food else None)

    min_w = taxonomy.get_default_min_weight()
    global_cap = taxonomy.get_default_cap(with_calibration=calibration_used)

    if food and food.max_weight_g is not None:
        # Taxonomy entry has an explicit per-food ceiling — use it.  This lets
        # whole-item foods (whole cake, full pizza, large watermelon) be weighed
        # correctly without inflating every category cap.
        max_w = min(float(food.max_weight_g), global_cap)
    else:
        max_w = taxonomy.get_category_cap(category_name)
        if serving and serving > 0:
            max_w = min(max_w if max_w > 0 else 1e9, 4.00 * float(serving))
        max_w = min(max_w, global_cap)

    if food and food.min_weight_g is not None:
        min_w = max(min_w, float(food.min_weight_g))
    elif serving and serving > 0:
        min_w = max(min_w, 0.30 * float(serving))

    original = float(max(0.0, weight_g))
    bounded = float(np.clip(original, min_w, max_w))
    guardrail_applied = bool(abs(bounded - original) > 1e-6)

    ranges = derive_weight_range(bounded, confidence)
    flags = quality_flags(original, bounded, confidence, guardrail_applied)
    rejected = bool(reject_low_quality and "low_quality" in flags)

    return {
        "estimated_weight_g": bounded,
        "weight_low_g": ranges["weight_low_g"],
        "weight_high_g": ranges["weight_high_g"],
        "guardrail_applied": guardrail_applied,
        "quality_flags": flags,
        "rejected": rejected,
        "guardrail_min_g": float(min_w),
        "guardrail_max_g": float(max_w),
    }

