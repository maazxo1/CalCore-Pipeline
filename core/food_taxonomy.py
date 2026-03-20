"""
Canonical food taxonomy loader and resolver.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional


_TAXONOMY_PATH = Path(__file__).resolve().parents[1] / "data" / "food_taxonomy.json"


def _normalize(text: str) -> str:
    text = (text or "").strip().lower().replace("_", " ")
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass(frozen=True)
class CanonicalFood:
    id: str
    display_name: str
    aliases: List[str]
    category: str
    density_g_ml: float
    typical_serving_g: Optional[float]
    min_weight_g: Optional[float]
    max_weight_g: Optional[float]
    usda_queries: List[str]


@dataclass(frozen=True)
class ResolutionResult:
    resolved: bool
    canonical_id: str
    canonical_name: str
    category: str
    score: float
    reason: str


class FoodTaxonomy:
    def __init__(self, data: Dict):
        self.version = str(data.get("version", "unknown"))
        self.global_cfg = dict(data.get("global", {}))
        self.category_caps_g = dict(data.get("category_caps_g", {}))
        self.category_fallback_nutrition = dict(data.get("category_fallback_nutrition_per100g", {}))

        self.foods_by_id: Dict[str, CanonicalFood] = {}
        self.alias_to_id: Dict[str, str] = {}

        for item in data.get("foods", []):
            food = CanonicalFood(
                id=str(item["id"]),
                display_name=str(item["display_name"]),
                aliases=[_normalize(a) for a in item.get("aliases", []) if _normalize(a)],
                category=str(item.get("category", "default_unknown")),
                density_g_ml=float(item.get("density_g_ml", 0.75)),
                typical_serving_g=(
                    float(item["typical_serving_g"])
                    if item.get("typical_serving_g") is not None
                    else None
                ),
                min_weight_g=(
                    float(item["min_weight_g"])
                    if item.get("min_weight_g") is not None
                    else None
                ),
                max_weight_g=(
                    float(item["max_weight_g"])
                    if item.get("max_weight_g") is not None
                    else None
                ),
                usda_queries=[str(q) for q in item.get("usda_queries", []) if str(q).strip()],
            )
            self.foods_by_id[food.id] = food
            for alias in food.aliases + [_normalize(food.display_name), _normalize(food.id)]:
                if alias and alias not in self.alias_to_id:
                    self.alias_to_id[alias] = food.id

        self._aliases_sorted = sorted(self.alias_to_id.keys(), key=len, reverse=True)

    def get_food(self, canonical_id: str) -> Optional[CanonicalFood]:
        return self.foods_by_id.get(canonical_id)

    def get_category_cap(self, category: str) -> float:
        return float(self.category_caps_g.get(category, self.category_caps_g.get("default_unknown", 450)))

    def get_category_fallback_nutrition(self, category: str) -> Dict:
        return dict(self.category_fallback_nutrition.get(category, self.category_fallback_nutrition.get("default_unknown", {})))

    def get_default_cap(self, with_calibration: bool = False) -> float:
        if with_calibration:
            return float(self.global_cfg.get("single_item_cap_g_with_calibration", 1800))
        return float(self.global_cfg.get("single_item_cap_g", 1200))

    def get_default_min_weight(self) -> float:
        return float(self.global_cfg.get("default_min_weight_g", 5))

    def resolve_label(self, label: str) -> ResolutionResult:
        norm = _normalize(label)
        if not norm:
            return ResolutionResult(False, "", "", "default_unknown", 0.0, "empty_label")

        if norm in self.alias_to_id:
            canonical_id = self.alias_to_id[norm]
            food = self.foods_by_id[canonical_id]
            return ResolutionResult(True, food.id, food.display_name, food.category, 1.0, "exact_alias")

        for alias in self._aliases_sorted:
            # Guard 1: skip trivially short aliases (e.g. "tea" ⊂ "steak").
            if len(alias) < 4:
                continue
            if alias in norm or norm in alias:
                # Guard 2: lengths must be within 2× of each other so a
                # 3-char alias can't claim a 20-char query.
                ratio = min(len(alias), len(norm)) / max(len(alias), len(norm), 1)
                if ratio < 0.50:
                    continue
                canonical_id = self.alias_to_id[alias]
                food = self.foods_by_id[canonical_id]
                # Partial/substring match is weaker than exact.
                return ResolutionResult(True, food.id, food.display_name, food.category, 0.72, "partial_alias")

        return ResolutionResult(False, "", "", "default_unknown", 0.0, "no_match")

    def coarse_for_category(self, category: str) -> ResolutionResult:
        fallback_map = {
            "egg": "egg",
            "bread_pastry": "bread",
            "fruit": "fruit",
            "sandwich_wrap": "sandwich",
            "rice_pasta_curry_bowl": "rice",
            "salad": "salad",
            "default_unknown": "default_unknown",
        }
        canonical_id = fallback_map.get(category, "default_unknown")
        food = self.foods_by_id.get(canonical_id)
        if not food:
            food = self.foods_by_id.get("default_unknown")
        if not food:
            return ResolutionResult(False, "", "", "default_unknown", 0.0, "no_coarse_fallback")
        return ResolutionResult(True, food.id, food.display_name, food.category, 0.60, "coarse_category_fallback")


@lru_cache(maxsize=1)
def get_taxonomy() -> FoodTaxonomy:
    with _TAXONOMY_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return FoodTaxonomy(data)

