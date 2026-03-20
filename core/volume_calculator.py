"""
Volume Calculator - Dispatcher / Wrapper.

Chooses the best available volume calculation method:
  1) Metric-depth voxelization (preferred)
  2) Relative-depth enhanced fallback

Adds commercial safety guardrails:
  - bounded single-item weights
  - confidence-aware weight ranges
  - quality flags for downstream UX/monitoring
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from core.enhanced_fallback_system import EnhancedFallbackSystem
from core.volume_estimator import VolumeEstimator
from core.weight_guardrails import apply_weight_bounds
from data.food_dimensions_database import FoodDimensionsDatabase
from data.usda_nutrition_lookup import get_food_density, get_typical_serving_weight


_METRIC_DEPTH_THRESHOLD_CM = 5.0


class VolumeCalculator:
    def __init__(self):
        self.dimensions_db = FoodDimensionsDatabase()
        self.fallback = EnhancedFallbackSystem()
        self.voxel_estimator = VolumeEstimator()

        self._ref_area_px = 40_000
        self._ref_weight_g = 150.0
        self._ref_resolution = (640, 480)

    def calculate_volume_from_mask_and_depth(
        self,
        mask: np.ndarray,
        depth_map: np.ndarray,
        food_name: str,
        usda_weight_g: Optional[float] = None,
        image_resolution: Optional[Tuple[int, int]] = None,
        *,
        canonical_id: Optional[str] = None,
        category: Optional[str] = None,
        calibration_scale: float = 1.0,
        reject_low_quality: bool = False,
        is_metric: Optional[bool] = None,
    ) -> Dict:
        mask_bool = mask.astype(bool)
        mask_area_px = int(np.sum(mask_bool))
        if mask_area_px == 0:
            return self._empty_result("No mask area")

        # Prefer caller-supplied flag; fall back to heuristic from depth range.
        if is_metric is None:
            is_metric = float(depth_map.max()) > _METRIC_DEPTH_THRESHOLD_CM
        scale = float(np.clip(calibration_scale, 0.5, 2.0))

        if is_metric:
            return self._calculate_voxelization(
                mask_bool=mask_bool,
                depth_cm=depth_map,
                food_name=food_name,
                usda_weight_g=usda_weight_g,
                canonical_id=canonical_id,
                category=category,
                calibration_scale=scale,
                reject_low_quality=reject_low_quality,
            )
        return self._calculate_fallback(
            mask_bool=mask_bool,
            mask_area_px=mask_area_px,
            depth_map=depth_map,
            food_name=food_name,
            usda_weight_g=usda_weight_g,
            image_resolution=image_resolution,
            canonical_id=canonical_id,
            category=category,
            calibration_scale=scale,
            reject_low_quality=reject_low_quality,
        )

    def _apply_guardrails(
        self,
        *,
        weight_g: float,
        confidence: float,
        canonical_id: Optional[str],
        category: Optional[str],
        usda_weight_g: Optional[float],
        calibration_scale: float,
        reject_low_quality: bool,
    ) -> Dict:
        bounded = apply_weight_bounds(
            weight_g=weight_g,
            canonical_id=canonical_id,
            category=category,
            typical_serving_g=usda_weight_g,
            confidence=confidence,
            calibration_used=abs(calibration_scale - 1.0) > 1e-6,
            reject_low_quality=reject_low_quality,
        )
        out_conf = float(confidence)
        if bounded["guardrail_applied"]:
            out_conf *= 0.85
        return {
            **bounded,
            "confidence": float(np.clip(out_conf, 0.0, 0.95)),
        }

    def _calculate_voxelization(
        self,
        *,
        mask_bool: np.ndarray,
        depth_cm: np.ndarray,
        food_name: str,
        usda_weight_g: Optional[float],
        canonical_id: Optional[str],
        category: Optional[str],
        calibration_scale: float,
        reject_low_quality: bool,
    ) -> Dict:
        result = self.voxel_estimator.estimate_volume(depth_cm, mask_bool, food_name)

        # Optional scale calibration: volume grows as scale³, but weight already
        # incorporates density so it only scales linearly with calibration.
        volume_scale = calibration_scale ** 3
        volume_ml = float(result["volume_ml"]) * volume_scale
        estimated_weight_g = float(result["estimated_weight_g"]) * calibration_scale
        # NOTE: no usda_weight_g clamp here — VolumeEstimator already handles USDA
        # anchoring internally.  Final bounds are applied by _apply_guardrails
        # using the per-food taxonomy max_weight_g (e.g. 3500g for a whole cake).

        bounded = self._apply_guardrails(
            weight_g=estimated_weight_g,
            confidence=float(result.get("confidence", 0.0)),
            canonical_id=canonical_id,
            category=category,
            usda_weight_g=usda_weight_g,
            calibration_scale=calibration_scale,
            reject_low_quality=reject_low_quality,
        )
        if bounded["rejected"]:
            return self._empty_result("Rejected by low-quality guardrail")

        return {
            "volume_ml": volume_ml,
            "estimated_weight_g": bounded["estimated_weight_g"],
            "weight_low_g": bounded["weight_low_g"],
            "weight_high_g": bounded["weight_high_g"],
            "guardrail_applied": bounded["guardrail_applied"],
            "quality_flags": bounded["quality_flags"],
            "method": result.get("method", "voxelization"),
            "confidence": bounded["confidence"],
            "size_ratio": result.get("correction_factor", 1.0),
            "dimensions_used": False,
            "volume_raw_ml": float(result.get("volume_raw_ml", 0.0)) * volume_scale,
            "hf_volume_ml": float(result.get("hf_volume_ml", 0.0)) * volume_scale,
            "correction_factor": result.get("correction_factor", 1.0),
            "voxel_count": result.get("voxel_count", 0),
            "usda_typical_ml": result.get("usda_typical_ml"),
            "usda_typical_g": result.get("usda_typical_g"),
            "density_g_ml": result.get("density_g_ml", 0.75),
            "pixel_to_cm": result.get("pixel_to_cm", 0.0),
            "median_depth_cm": result.get("median_depth_cm", 0.0),
            "multi_object_flag": result.get("multi_object_flag", False),
            "calibration_scale": calibration_scale,
            **({"error": result["error"]} if "error" in result else {}),
        }

    def _calculate_fallback(
        self,
        *,
        mask_bool: np.ndarray,
        mask_area_px: int,
        depth_map: np.ndarray,
        food_name: str,
        usda_weight_g: Optional[float],
        image_resolution: Optional[Tuple[int, int]],
        canonical_id: Optional[str],
        category: Optional[str],
        calibration_scale: float,
        reject_low_quality: bool,
    ) -> Dict:
        if image_resolution is None:
            image_resolution = depth_map.shape[:2]
        img_h, img_w = image_resolution

        ref_w, ref_h = self._ref_resolution
        resolution_scale = (img_w * img_h) / (ref_w * ref_h)
        adjusted_ref_area = self._ref_area_px * resolution_scale

        masked_depth = depth_map[mask_bool]
        if len(masked_depth) == 0:
            return self._empty_result("No depth data in mask")

        depth_min = float(np.percentile(masked_depth, 5))
        depth_max = float(np.percentile(masked_depth, 95))
        depth_range = depth_max - depth_min

        if usda_weight_g is None:
            usda_weight_g = get_typical_serving_weight(food_name)
        if usda_weight_g is None:
            dims = self.dimensions_db.get_dimensions(food_name)
            if dims:
                usda_weight_g = dims.get("typical_weight_g")

        dims = self.dimensions_db.get_dimensions(food_name)
        if dims:
            return self._calc_with_dimensions(
                mask_area_px=mask_area_px,
                depth_range=depth_range,
                food_name=food_name,
                dims=dims,
                usda_weight_g=usda_weight_g,
                adjusted_ref_area=adjusted_ref_area,
                canonical_id=canonical_id,
                category=category,
                calibration_scale=calibration_scale,
                reject_low_quality=reject_low_quality,
            )
        return self._calc_enhanced_fallback(
            mask_area_px=mask_area_px,
            depth_range=depth_range,
            food_name=food_name,
            usda_weight_g=usda_weight_g,
            adjusted_ref_area=adjusted_ref_area,
            canonical_id=canonical_id,
            category=category,
            calibration_scale=calibration_scale,
            reject_low_quality=reject_low_quality,
        )

    def _calc_with_dimensions(
        self,
        *,
        mask_area_px: float,
        depth_range: float,
        food_name: str,
        dims: Dict,
        usda_weight_g: Optional[float],
        adjusted_ref_area: float,
        canonical_id: Optional[str],
        category: Optional[str],
        calibration_scale: float,
        reject_low_quality: bool,
    ) -> Dict:
        theoretical_vol = self.dimensions_db.calculate_theoretical_volume(food_name)
        if not theoretical_vol:
            return self._calc_enhanced_fallback(
                mask_area_px=mask_area_px,
                depth_range=depth_range,
                food_name=food_name,
                usda_weight_g=usda_weight_g,
                adjusted_ref_area=adjusted_ref_area,
                canonical_id=canonical_id,
                category=category,
                calibration_scale=calibration_scale,
                reject_low_quality=reject_low_quality,
            )

        size_ratio = float(np.clip(np.sqrt(mask_area_px / adjusted_ref_area), 0.3, 3.0))
        estimated_vol = float(theoretical_vol * (size_ratio ** 3))
        if depth_range > 0.05:
            depth_factor = float(np.clip(depth_range / 0.25, 0.5, 2.0))
            estimated_vol *= depth_factor
        estimated_vol = float(np.clip(estimated_vol, 10.0, 5000.0))

        if usda_weight_g:
            typical_weight = dims.get("typical_weight_g", usda_weight_g)
            if typical_weight and typical_weight > 1000:
                area_ratio = mask_area_px / adjusted_ref_area
                weight_g = float(np.clip(typical_weight * area_ratio, 100, 10_000))
            else:
                typical_area = adjusted_ref_area * (
                    (typical_weight or self._ref_weight_g) / self._ref_weight_g
                ) ** (2 / 3)
                area_ratio = mask_area_px / typical_area
                weight_scale = area_ratio ** 1.5
                weight_g = float(np.clip((typical_weight or self._ref_weight_g) * weight_scale, 1, 2000))
            confidence = 0.65
            method = "dimensions_db + usda_weight (relative_depth)"
        else:
            density = get_food_density(food_name)
            weight_g = float(np.clip(estimated_vol * density, 5, 2000))
            confidence = 0.55
            method = "dimensions_db + density (relative_depth)"

        volume_scale = calibration_scale ** 3
        estimated_vol *= volume_scale
        weight_g *= calibration_scale  # linear: weight already encodes density

        bounded = self._apply_guardrails(
            weight_g=weight_g,
            confidence=confidence,
            canonical_id=canonical_id,
            category=category,
            usda_weight_g=usda_weight_g,
            calibration_scale=calibration_scale,
            reject_low_quality=reject_low_quality,
        )
        if bounded["rejected"]:
            return self._empty_result("Rejected by low-quality guardrail")

        return {
            "volume_ml": estimated_vol,
            "estimated_weight_g": bounded["estimated_weight_g"],
            "weight_low_g": bounded["weight_low_g"],
            "weight_high_g": bounded["weight_high_g"],
            "guardrail_applied": bounded["guardrail_applied"],
            "quality_flags": bounded["quality_flags"],
            "method": method,
            "confidence": bounded["confidence"],
            "size_ratio": size_ratio,
            "dimensions_used": True,
            "calibration_scale": calibration_scale,
        }

    def _calc_enhanced_fallback(
        self,
        *,
        mask_area_px: float,
        depth_range: float,
        food_name: str,
        usda_weight_g: Optional[float],
        adjusted_ref_area: float,
        canonical_id: Optional[str],
        category: Optional[str],
        calibration_scale: float,
        reject_low_quality: bool,
    ) -> Dict:
        fallback_result = self.fallback.estimate_properties(
            food_name,
            mask_area_px,
            depth_range,
            adjusted_ref_area=adjusted_ref_area,
        )

        if usda_weight_g:
            area_ratio = mask_area_px / adjusted_ref_area
            weight_scale = area_ratio ** 1.5
            weight_g = float(np.clip(usda_weight_g * weight_scale, 5, 2000))
            confidence = 0.55
            method = "enhanced_fallback + usda_weight"
        else:
            weight_g = float(np.clip(fallback_result["weight_g"], 5, 2000))
            confidence = float(fallback_result["confidence"])
            method = f"enhanced_fallback ({confidence:.0%})"

        volume_scale = calibration_scale ** 3
        volume_ml = float(fallback_result["volume_ml"]) * volume_scale
        weight_g *= calibration_scale  # linear: weight already encodes density

        bounded = self._apply_guardrails(
            weight_g=weight_g,
            confidence=confidence,
            canonical_id=canonical_id,
            category=category,
            usda_weight_g=usda_weight_g,
            calibration_scale=calibration_scale,
            reject_low_quality=reject_low_quality,
        )
        if bounded["rejected"]:
            return self._empty_result("Rejected by low-quality guardrail")

        similar_foods = self.fallback.suggest_similar_foods(food_name, top_n=3)
        return {
            "volume_ml": volume_ml,
            "estimated_weight_g": bounded["estimated_weight_g"],
            "weight_low_g": bounded["weight_low_g"],
            "weight_high_g": bounded["weight_high_g"],
            "guardrail_applied": bounded["guardrail_applied"],
            "quality_flags": bounded["quality_flags"],
            "method": method,
            "confidence": bounded["confidence"],
            "size_ratio": float(np.sqrt(mask_area_px / adjusted_ref_area)),
            "dimensions_used": False,
            "density_g_ml": fallback_result["density_g_ml"],
            "fallback_methods": fallback_result["methods_used"],
            "similar_foods": similar_foods,
            "confidence_level": self.fallback.get_confidence_level(confidence),
            "calibration_scale": calibration_scale,
        }

    @staticmethod
    def _empty_result(reason: str) -> Dict:
        return {
            "volume_ml": 0.0,
            "estimated_weight_g": 0.0,
            "weight_low_g": 0.0,
            "weight_high_g": 0.0,
            "guardrail_applied": False,
            "quality_flags": ["failed"],
            "method": "failed",
            "confidence": 0.0,
            "size_ratio": 0.0,
            "dimensions_used": False,
            "error": reason,
        }

