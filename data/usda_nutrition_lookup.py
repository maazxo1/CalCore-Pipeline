"""
USDA FoodData Central – Canonical Nutrition & Density Module
=============================================================
Single source of truth for nutrition lookup and density data.
usda_nutrition_lookup_api.py now re-exports everything from here.

Fixes applied:
  - API key read from USDA_API_KEY env variable; hardcoded key is fallback only
  - kJ/kcal disambiguation: filter by nutrient unit name, not value threshold
  - USDA result selection: prefer Foundation > SR Legacy > Survey > Branded
  - get_food_density() no longer instantiates a full class per call
  - get_typical_serving_weight() partial match uses longest-wins strategy
  - Salad fallback nutrition corrected (was 15 kcal/100g – plain lettuce only)
  - Nutrition cache stored per-food-name per-100g; scaled at call time
"""

import os
from pathlib import Path
import re
import sqlite3
import threading
import time
import requests
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load .env from project root if present.
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

# ---------------------------------------------------------------------------
# API configuration – set USDA_API_KEY env var in production
# ---------------------------------------------------------------------------
USDA_API_KEY: str = os.environ.get("USDA_API_KEY", "").strip()
USDA_BASE_URL = "https://api.nal.usda.gov/fdc/v1"


def _get_usda_key() -> str:
    """
    Return USDA API key with runtime env precedence.

    This keeps behavior dynamic if the environment variable is rotated
    after module import (e.g., container secret refresh + process reload).
    """
    return os.environ.get("USDA_API_KEY", USDA_API_KEY).strip()


def _has_usda_key() -> bool:
    return bool(_get_usda_key())


def _redact_api_key_from_text(text: str, api_key: Optional[str] = None) -> str:
    """Remove USDA API key values from logs/exceptions."""
    safe = str(text)
    key = (api_key or _get_usda_key() or "").strip()
    if key:
        safe = safe.replace(key, "***")
    safe = re.sub(r"(api_key=)[^&\s]+", r"\1***", safe)
    return safe


def get_usda_cache_status() -> Dict:
    exists = _CACHE_DB_PATH.exists()
    size_bytes = _CACHE_DB_PATH.stat().st_size if exists else 0
    return {
        "cache_db_path": str(_CACHE_DB_PATH),
        "cache_db_exists": bool(exists),
        "cache_db_size_bytes": int(size_bytes),
        "cache_ttl_seconds": int(_CACHE_TTL_SECONDS),
    }


def _init_sqlite_cache() -> None:
    with _CACHE_LOCK:
        conn = sqlite3.connect(_CACHE_DB_PATH)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS nutrition_cache (
                    cache_key TEXT PRIMARY KEY,
                    fdc_id INTEGER,
                    description TEXT,
                    payload_json TEXT NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()


def _cache_get(cache_key: str) -> Optional[Dict]:
    _init_sqlite_cache()
    with _CACHE_LOCK:
        conn = sqlite3.connect(_CACHE_DB_PATH)
        try:
            row = conn.execute(
                """
                SELECT fdc_id, description, payload_json, updated_at
                FROM nutrition_cache
                WHERE cache_key = ?
                """,
                (cache_key,),
            ).fetchone()
        finally:
            conn.close()
    if not row:
        return None
    fdc_id, description, payload_json, updated_at = row
    # Tiered TTL: weak matches (confidence < 0.70) expire in 4 h instead of 14 days.
    try:
        import json as _json
        _conf_check = _json.loads(payload_json or "{}").get("nutrition_confidence", 1.0)
    except Exception:
        _conf_check = 1.0
    effective_ttl = _CACHE_TTL_SECONDS if _conf_check >= 0.70 else (4 * 3600)
    if int(time.time()) - int(updated_at) > effective_ttl:
        return None
    try:
        import json
        payload = json.loads(payload_json)
    except Exception:
        return None
    if fdc_id:
        payload["fdc_id"] = int(fdc_id)
    if description:
        payload["usda_description"] = description
    return payload


def _cache_put(cache_key: str, payload: Dict) -> None:
    _init_sqlite_cache()
    fdc_id = payload.get("fdc_id")
    description = payload.get("usda_description")
    import json
    payload_json = json.dumps(payload, ensure_ascii=True)
    with _CACHE_LOCK:
        conn = sqlite3.connect(_CACHE_DB_PATH)
        try:
            conn.execute(
                """
                INSERT INTO nutrition_cache(cache_key, fdc_id, description, payload_json, updated_at)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    fdc_id=excluded.fdc_id,
                    description=excluded.description,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (cache_key, fdc_id, description, payload_json, int(time.time())),
            )
            conn.commit()
        finally:
            conn.close()

# Priority order for USDA data types (higher = more reliable)
_DATA_TYPE_PRIORITY = {
    "Foundation": 4,
    "SR Legacy":  3,
    "Survey (FNDDS)": 2,
    "Branded":    1,
}

# USDA nutrient IDs for energy in kcal (avoids kJ confusion)
_ENERGY_KCAL_IDS = {1008, 2047}

# Cache stores per-100 g values; key = normalised food name
_nutrition_cache_per100g: Dict[str, Dict] = {}

# Persistent sqlite cache for USDA results.
_CACHE_DB_PATH = Path(__file__).resolve().parent / "usda_cache.sqlite"
_CACHE_TTL_SECONDS = 14 * 24 * 60 * 60
_CACHE_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# FAO Density Database (g/ml)  – used by volume_calculator.py
# ---------------------------------------------------------------------------
_DENSITY_DB: Dict[str, float] = {
    # Fruits
    "apple": 0.64, "banana": 0.94, "orange": 0.87, "strawberry": 0.60,
    "watermelon": 0.96, "grapes": 0.61, "pineapple": 0.54, "mango": 0.64,
    "pear": 0.59, "peach": 0.61, "avocado": 0.97, "lemon": 0.80,
    "cherry": 0.90, "blueberry": 0.60, "blackberry": 0.60, "kiwi": 0.90,

    # Vegetables
    "tomato": 0.96, "lettuce": 0.55, "cucumber": 0.97, "carrot": 0.64,
    "broccoli": 0.55, "cauliflower": 0.44, "spinach": 0.38, "potato": 0.81,
    "onion": 0.56, "bell_pepper": 0.54, "corn": 0.72, "asparagus": 0.62,
    "mushroom": 0.50, "cabbage": 0.60,

    # Grains (cooked)
    "rice": 0.96, "pasta": 0.92, "bread": 0.26, "garlic_bread": 0.26, "bagel": 0.35,
    "cereal": 0.30, "oatmeal": 0.85, "quinoa": 0.85, "naan": 0.55,
    "roti": 0.45, "chapati": 0.45, "paratha": 0.60, "pita": 0.40,
    "tortilla": 0.60, "pancake": 0.50, "waffle": 0.50,

    # Proteins
    "chicken_breast": 1.05, "chicken": 1.05, "beef": 1.04, "steak": 1.04,
    "pork": 1.03, "pork_chop": 1.03, "lamb": 1.04, "turkey": 1.04, "duck": 1.05,
    "bacon": 0.95, "ham": 1.03,
    "fish": 1.04, "salmon": 1.05, "tuna": 1.08, "crab": 1.03, "lobster": 1.04,
    "egg": 1.03, "boiled_egg": 1.03, "fried_egg": 1.00, "scrambled_egg": 0.95,
    "tofu": 1.03, "shrimp": 1.06,

    # Dairy
    "milk": 1.03, "yogurt": 1.05, "cheese": 1.15, "butter": 0.91,
    "cream": 1.01, "ice_cream": 0.56,

    # Fast Food
    "pizza": 0.65, "burger": 0.85, "fries": 0.35, "sandwich": 0.55,
    "hot_dog": 0.88, "taco": 0.75, "burrito": 0.85, "sausage": 0.88,
    "baked_beans": 0.84, "hash_brown": 0.62,

    # Snacks & Sweets
    "chips": 0.30, "popcorn": 0.17, "crackers": 0.28, "cookies": 0.52,
    "cake": 0.60, "donut": 0.48, "chocolate": 1.26, "brownie": 0.75,
    "muffin": 0.55, "pie": 0.75,

    # South Asian
    "biryani": 0.90, "daal": 0.95, "samosa": 0.70, "chicken_curry": 0.92,
    "butter_chicken": 0.94, "tikka_masala": 0.91, "palak_paneer": 0.88,
    "gulab_jamun": 1.00, "paneer": 1.20,
    "dosa": 0.35, "idli": 0.65, "pakora": 0.70, "jalebi": 1.10, "kheer": 1.02,

    # East / Southeast Asian
    "gyoza": 0.85, "tempura": 0.70, "takoyaki": 0.90, "okonomiyaki": 0.82,
    "onigiri": 0.95, "mochi": 0.95, "bao": 0.55, "char_siu": 1.05,
    "peking_duck": 1.00, "egg_tart": 0.90,
    "satay": 1.00, "rendang": 1.05, "nasi_goreng": 0.92, "laksa": 0.88,
    "poke_bowl": 0.88, "acai_bowl": 0.82, "granola": 0.45,

    # Salads & Soups
    "salad": 0.50, "caesar_salad": 0.45, "greek_salad": 0.68,
    "soup": 0.98, "ramen": 0.82, "pho": 0.85,

    # Sushi
    "sushi": 0.90, "sushi_roll": 0.90,

    # Beverages / condiments
    "juice": 1.04, "soda": 1.04, "coffee": 1.00, "tea": 1.00,
    "ketchup": 1.14, "mayonnaise": 0.91, "mustard": 1.08, "olive_oil": 0.92,

    "default": 0.75,
}

# Name aliases → canonical density key
_DENSITY_ALIASES: Dict[str, str] = {
    "french fries": "fries",
    "french_fries": "fries",
    "cooked rice": "rice",
    "white rice": "rice",
    "brown rice": "rice",
    "hamburger": "burger",
    "cheeseburger": "burger",
    "grilled chicken": "chicken",
    "chicken breast": "chicken_breast",
    "baked potato": "potato",
}


# ---------------------------------------------------------------------------
# USDA Typical Serving Weights
# ---------------------------------------------------------------------------
_TYPICAL_SERVINGS: Dict[str, float] = {
    # Fruits
    "apple": 182, "banana": 118, "orange": 131, "strawberry": 36,
    "blueberry": 148, "blackberry": 144, "mango": 165, "grapes": 92,
    "watermelon": 280, "pineapple": 82, "pear": 178, "peach": 150,
    "avocado": 150, "kiwi": 75, "cherry": 68, "lemon": 108,

    # Vegetables
    "broccoli": 91, "carrot": 78, "tomato": 123, "potato": 173,
    "lettuce": 47, "cucumber": 119, "bell_pepper": 119, "onion": 110,
    "spinach": 30, "corn": 90, "asparagus": 90, "mushroom": 70,
    "cabbage": 150,

    # Fast Food
    "pizza": 107, "burger": 220, "sandwich": 150, "hot_dog": 76,
    "french fries": 117, "fries": 117, "taco": 85, "burrito": 220,
    "sausage": 65, "hash brown": 85, "hash_brown": 85,
    "baked beans": 253, "baked_beans": 253,

    # Grains
    "rice": 197, "pasta": 140, "bread": 29, "bagel": 89,
    "naan": 90, "roti": 40, "chapati": 40, "paratha": 65,
    "oatmeal": 234, "cereal": 40, "pancake": 38, "waffle": 75,

    # Proteins
    "chicken breast": 174, "chicken_breast": 174, "chicken": 174, "steak": 226,
    "beef": 226, "pork": 175, "pork chop": 200, "pork_chop": 200,
    "lamb": 175, "turkey": 175, "duck": 155,
    "sausage": 85, "bacon": 28, "ham": 175,
    "salmon": 178, "tuna": 140, "fish": 140, "shrimp": 90, "crab": 85, "lobster": 85,
    "egg": 50, "boiled egg": 50, "boiled_egg": 50,
    "fried egg": 46, "fried_egg": 46, "scrambled egg": 100, "scrambled_egg": 100,
    "tofu": 130,

    # Dairy
    "yogurt": 245, "cheese": 28, "milk": 244, "ice_cream": 74,

    # Baked Goods
    "donut": 52, "muffin": 113, "cookie": 40, "cake": 80,
    "brownie": 56, "pie": 113,

    # South Asian
    "biryani": 270, "daal": 245, "samosa": 70,
    "chicken curry": 250, "chicken_curry": 250,
    "butter chicken": 250, "butter_chicken": 250,
    "tikka masala": 250, "tikka_masala": 250,
    "palak paneer": 240, "palak_paneer": 240,
    "gulab jamun": 60, "gulab_jamun": 60,
    "dosa": 100, "idli": 50, "pakora": 60, "jalebi": 80, "kheer": 245,

    # East / Southeast Asian
    "gyoza": 100, "tempura": 120, "takoyaki": 100, "okonomiyaki": 180,
    "onigiri": 110, "mochi": 60, "bao": 80, "char siu": 160, "char_siu": 160,
    "peking duck": 140, "peking_duck": 140, "egg tart": 60, "egg_tart": 60,
    "satay": 90, "rendang": 150, "nasi goreng": 270, "nasi_goreng": 270,
    "laksa": 440,

    # Trending / health
    "poke bowl": 330, "poke_bowl": 330,
    "acai bowl": 300, "acai_bowl": 300,
    "granola": 60,

    # Prepared
    "salad": 100, "soup": 245,
}


# ---------------------------------------------------------------------------
# USDAFoodDatabase class (kept for backward compatibility with api_server.py)
# ---------------------------------------------------------------------------

class USDAFoodDatabase:
    """USDA FoodData Central API interface."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = (api_key or _get_usda_key() or "").strip()
        self.base_url = USDA_BASE_URL
        self.food_densities = _DENSITY_DB
        self.name_mappings = _DENSITY_ALIASES

    def search_food(self, query: str, page_size: int = 5) -> List[Dict]:
        """Search USDA for foods; returns list with fdc_id, description, data_type."""
        if not self.api_key:
            return []

        url = f"{self.base_url}/foods/search"
        params = {
            "api_key": self.api_key,
            "query":    query,
            "pageSize": page_size,
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for food in data.get("foods", []):
                results.append({
                    "fdc_id":      food.get("fdcId"),
                    "description": food.get("description"),
                    "data_type":   food.get("dataType"),
                    "brand":       food.get("brandOwner", "N/A"),
                })
            return results
        except requests.exceptions.RequestException as e:
            print(f"[USDA] Search error: {_redact_api_key_from_text(e, self.api_key)}")
            return []

    def get_food_details(self, fdc_id: int) -> Optional[Dict]:
        """Return nutrient dict per 100 g for a given FDC ID."""
        if not self.api_key:
            return None

        url = f"{self.base_url}/food/{fdc_id}"
        params = {"api_key": self.api_key, "format": "full"}
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            nutrients = _parse_nutrients(data.get("foodNutrients", []))

            return {
                "fdc_id":          fdc_id,
                "description":     data.get("description"),
                "nutrients":       nutrients,
                "portion_size_g":  100,
            }
        except requests.exceptions.RequestException as e:
            print(f"[USDA] Details error: {_redact_api_key_from_text(e, self.api_key)}")
            return None

    def get_density(self, food_name: str) -> float:
        return get_food_density(food_name)

    def volume_to_weight(self, volume_ml: float, food_name: str) -> float:
        return volume_ml * self.get_density(food_name)

    def get_nutrition_for_volume(
        self, food_name: str, volume_ml: float
    ) -> Optional[Dict]:
        """Convenience: volume → weight → nutrition."""
        weight_g = self.volume_to_weight(volume_ml, food_name)
        nutrition = get_nutrition_info(food_name, weight_g)
        return {
            "food_name":    food_name,
            "volume_ml":    volume_ml,
            "weight_g":     weight_g,
            "density_g_ml": self.get_density(food_name),
            **nutrition,
        }


# ---------------------------------------------------------------------------
# Standalone helper functions
# ---------------------------------------------------------------------------

def get_food_density(food_name: str) -> float:
    """
    Return FAO density (g/ml) for a food.
    No class instantiation; pure dict lookup with partial match fallback.
    """
    key = food_name.lower().strip()

    # Resolve alias
    key = _DENSITY_ALIASES.get(key, key)
    key = key.replace(" ", "_")

    if key in _DENSITY_DB:
        return _DENSITY_DB[key]

    # Longest partial match
    best_key, best_len = None, 0
    for k in _DENSITY_DB:
        if k == "default":
            continue
        if k in key or key in k:
            if len(k) > best_len:
                best_key, best_len = k, len(k)

    if best_key:
        return _DENSITY_DB[best_key]

    return _DENSITY_DB["default"]


def get_typical_serving_weight(food_name: str) -> Optional[float]:
    """
    Return typical USDA serving weight (g).
    Uses longest-match strategy to avoid short keys overriding specific ones
    (e.g. "chicken" must not shadow "chicken tikka masala").
    Normalises spaces↔underscores so "hot dog" matches "hot_dog" and vice versa.
    """
    key = food_name.lower().strip()
    key_under = key.replace(' ', '_')
    key_space  = key.replace('_', ' ')

    # Exact match — try all three normalisation forms
    for k in (key, key_under, key_space):
        if k in _TYPICAL_SERVINGS:
            return _TYPICAL_SERVINGS[k]

    # Longest partial match wins (compare normalised forms to DB keys)
    best_val, best_len = None, 0
    for k, v in _TYPICAL_SERVINGS.items():
        k_norm = k.replace('_', ' ')
        if k_norm in key_space or key_space in k_norm:
            if len(k) > best_len:
                best_val, best_len = v, len(k)

    return best_val


def _cache_key_for_food(food_name: str, canonical_id: Optional[str]) -> str:
    specific = food_name.lower().strip().replace(" ", "_")
    base = (canonical_id or specific).replace(" ", "_")
    # Keep canonical_id as namespace prefix but include specific label so
    # "biryani" and "plain rice" (both canonical_id="rice") get separate entries.
    return f"{base}::{specific}" if specific != base else base


def _category_fallback_profile(category: Optional[str]) -> Optional[Dict]:
    try:
        from core.food_taxonomy import get_taxonomy
        taxonomy = get_taxonomy()
        if category:
            profile = taxonomy.get_category_fallback_nutrition(category)
            if profile:
                return profile
    except Exception:
        pass
    return None


def _nutrition_confidence_from_datatype(dtype: str) -> float:
    return {
        "Foundation": 0.95,
        "SR Legacy": 0.90,
        "Survey (FNDDS)": 0.80,
        "Branded": 0.65,
    }.get(dtype, 0.60)


_CATEGORY_BANNED_TERMS = {
    "egg": {
        "bagel", "sandwich", "biscuit", "burger", "pizza",
        "salad", "wrap", "taco", "benedict", "breakfast",
        "burrito", "toast",
        # Exclude animal parts that share cooking terms (e.g. "Chicken, feet, boiled")
        "chicken", "feet", "pork", "beef", "turkey", "duck",
        "fish", "meat", "veal", "lamb",
    },
    # Fruits — prevent cross-contamination with preparations & unrelated foods
    "banana":  {"pepper", "dehydrated", "dried", "chips", "powder", "bread"},
    "apple":   {"crisp", "pie", "juice", "cider", "sauce", "butter"},
    "orange":  {"juice", "peel", "extract", "drink"},
    "grape":   {"juice", "wine", "raisin", "drink"},
    "mango":   {"powder", "extract", "dried", "chutney"},
    # Proteins — prevent broth/seasoning/processed forms
    "chicken": {"broth", "stock", "powder", "seasoning", "flavor", "soup"},
    "salmon":  {"flavor", "seasoning", "powder", "oil"},
    "beef":    {"broth", "stock", "flavor", "jerky", "powder"},
    "steak":   {"sauce", "seasoning"},
    # Grains — prevent dirty/prepared/processed forms
    "rice":    {"dirty", "pudding", "cake", "wine", "flour", "syrup"},
    "corn":    {"starch", "flour", "syrup", "oil"},
    "oat":     {"flour"},
    # Vegetables
    "potato":  {"chips", "powder", "starch", "flour"},
    "tomato":  {"juice", "paste", "sauce", "puree", "soup"},
    # Bread — prevent crumbs/pudding confusion
    "bread":   {"crumbs", "pudding"},
}


# Expected calorie range per 100 g for each food category.
# Results outside the range are rejected before caching so implausible
# matches (e.g. "dehydrated banana powder" at 356 kcal/100g for category
# "fruit") never pollute the cache.
_CALORIE_BOUNDS_PER100G: Dict[str, tuple] = {
    "fruit":                 (20,  160),
    "vegetable":             (10,  160),
    "bread_pastry":          (180, 680),
    "rice_pasta_curry_bowl": (80,  480),
    "protein":               (80,  420),
    "egg":                   (100, 260),
    "snack":                 (180, 780),
    "drink":                 (0,   250),
}


def _is_usda_candidate_compatible(
    description: str,
    *,
    canonical_id: Optional[str],
    category: Optional[str],
) -> bool:
    text = (description or "").lower()
    banned = set()
    if category:
        banned |= _CATEGORY_BANNED_TERMS.get(category, set())
    if canonical_id:
        banned |= _CATEGORY_BANNED_TERMS.get(canonical_id, set())
    if banned and any(term in text for term in banned):
        return False
    return True


def get_nutrition_info(
    food_name: str,
    weight_g: float,
    *,
    canonical_id: Optional[str] = None,
    category: Optional[str] = None,
    usda_queries: Optional[List[str]] = None,
) -> Dict:
    """
    Return nutrition for food_name at weight_g.

    Strategy:
        1. If food_name result already cached (per 100 g), scale and return.
        2. Call USDA search → pick best data-type result.
        3. Fetch nutrient detail, parse kcal correctly (by unit, not threshold).
        4. Cache per-100-g; fall back to offline DB on any failure.
    """
    cache_key = _cache_key_for_food(food_name, canonical_id)

    # ── 1. Serve from cache ───────────────────────────────────────────────
    if cache_key in _nutrition_cache_per100g:
        cached_mem = _nutrition_cache_per100g[cache_key]
        if _is_usda_candidate_compatible(
            str(cached_mem.get("usda_description", "")),
            canonical_id=canonical_id,
            category=category,
        ):
            return _with_macro_sanity(_scale_nutrition(cached_mem, weight_g))

    cached = _cache_get(cache_key)
    if cached:
        if _is_usda_candidate_compatible(
            str(cached.get("usda_description", "")),
            canonical_id=canonical_id,
            category=category,
        ):
            _nutrition_cache_per100g[cache_key] = cached
            out = _scale_nutrition(cached, weight_g)
            out["nutrition_source"] = "USDA cache"
            return _with_macro_sanity(out)

    if not _has_usda_key():
        return _with_macro_sanity(_nutrition_from_fallback(food_name, weight_g, category=category))

    # ── 2. USDA Search ────────────────────────────────────────────────────
    api_key = _get_usda_key()
    queries = [q for q in (usda_queries or []) if str(q).strip()]
    if not queries:
        queries = [food_name]
    if food_name not in queries:
        queries.append(food_name)

    try:
        candidate_pool: List[tuple[int, str, Dict]] = []

        for query in queries:
            search_url = f"{USDA_BASE_URL}/foods/search"
            search_params = {
                "api_key": api_key,
                "query": query,
                "pageSize": 10,
            }
            resp = requests.get(search_url, params=search_params, timeout=10)
            resp.raise_for_status()
            foods = resp.json().get("foods", [])
            if not foods:
                continue
            ranked = sorted(
                foods,
                key=lambda food: _score_usda_candidate(food, query),
                reverse=True,
            )
            for candidate in ranked[:5]:
                description = str(candidate.get("description", ""))
                if not _is_usda_candidate_compatible(
                    description,
                    canonical_id=canonical_id,
                    category=category,
                ):
                    continue
                score = _score_usda_candidate(candidate, query)
                candidate_pool.append((score, query, candidate))

        if not candidate_pool:
            raise ValueError("No foods found")

        candidate_pool.sort(key=lambda x: x[0], reverse=True)
        last_error: Optional[str] = None
        for _score, best_query, best in candidate_pool:
            try:
                fdc_id = int(best["fdcId"])
                description = str(best.get("description", ""))
                dtype = str(best.get("dataType", ""))
                print(f"   [USDA API] Matched ({best_query}): {description} [{dtype}]")

                detail_url = f"{USDA_BASE_URL}/food/{fdc_id}"
                detail_resp = requests.get(
                    detail_url,
                    params={"api_key": api_key, "format": "full"},
                    timeout=10,
                )
                detail_resp.raise_for_status()
                detail_data = detail_resp.json()

                per100g = _parse_nutrients(detail_data.get("foodNutrients", []))

                # Calorie sanity: reject physiologically implausible matches
                # before caching (e.g. "banana" → 356 kcal/100g dehydrated powder).
                cal_per100g = float(per100g.get("calories", 0) or 0)
                cat_bounds = _CALORIE_BOUNDS_PER100G.get(category or "")
                if cat_bounds and cal_per100g > 0:
                    if not (cat_bounds[0] <= cal_per100g <= cat_bounds[1]):
                        print(
                            f"   [USDA API] Skipping {description!r}: "
                            f"{cal_per100g:.0f} kcal/100g outside bounds "
                            f"{cat_bounds} for category {category!r}"
                        )
                        continue   # try next candidate

                per100g["source"] = "USDA API"
                per100g["nutrition_source"] = "USDA API"
                per100g["usda_description"] = description
                per100g["fdc_id"] = fdc_id
                per100g["nutrition_confidence"] = _nutrition_confidence_from_datatype(dtype)

                # Only persist to cache when confidence is high enough.
                # Low-confidence matches are returned for this request only.
                if per100g.get("nutrition_confidence", 0) >= 0.55:
                    _nutrition_cache_per100g[cache_key] = per100g
                    _cache_put(cache_key, per100g)
                return _with_macro_sanity(_scale_nutrition(per100g, weight_g))
            except Exception as candidate_error:
                last_error = _redact_api_key_from_text(candidate_error, api_key)
                continue

        raise RuntimeError(last_error or "All USDA candidates failed")

    except Exception as e:
        print(
            "   [USDA API] Failed: "
            f"{_redact_api_key_from_text(e, api_key)} -> using offline fallback"
        )
        return _with_macro_sanity(_nutrition_from_fallback(food_name, weight_g, category=category))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Terms that signal a processed/industrial form — penalised unless the query
# itself asks for that form (e.g. "dried mango" would contain "dried").
_USDA_PROCESSING_PENALTY_TERMS: frozenset = frozenset({
    "dehydrated", "dried", "powder", "concentrate", "extract",
    "freeze-dried", "flakes", "mix", "paste", "broth", "stock",
    "flavored", "seasoning", "crumbs",
})
# Terms that indicate a composite preparation rather than the raw ingredient.
_USDA_PREPARATION_PENALTY_TERMS: frozenset = frozenset({
    "crisp", "pie", "cake", "casserole", "stew", "sauce",
    "pudding", "candy",
})
# Terms that indicate a minimally-processed, whole form — preferred for
# generic food queries.
_USDA_RAW_BONUS_TERMS: frozenset = frozenset({
    "raw", "fresh", "whole", "plain",
})


def _score_usda_candidate(food: Dict, query: str) -> int:
    """Score a USDA candidate result for relevance to *query*.

    Base: data-type priority * 100  +  word overlap * 10  +  exact match.
    Additions (when query does not itself request processed form):
      +30  if description signals a raw/whole/fresh form
      −80  per processing-term hit  (dried, powder, broth, …)
      −200 per preparation-term hit (crisp, pie, pudding, …)
    """
    query_lower = str(query).lower()
    query_words = set(query_lower.split())
    dtype = str(food.get("dataType", ""))
    desc = str(food.get("description", "")).lower()
    # Strip punctuation before word-overlap so "Egg, whole" desc_word "egg," matches "egg"
    desc_words = set(re.sub(r"[^\w\s]", "", desc).split())

    type_score = _DATA_TYPE_PRIORITY.get(dtype, 0)
    overlap = len(query_words & desc_words)
    exact = 1 if query_lower in desc else 0
    score = type_score * 100 + overlap * 10 + exact

    # Only apply adjustments when the query is a plain food name (no modifiers).
    query_has_processing = any(t in query_lower for t in _USDA_PROCESSING_PENALTY_TERMS)
    if not query_has_processing:
        # Reward raw/whole form
        if any(t in desc_words for t in _USDA_RAW_BONUS_TERMS):
            score += 30
        # Penalise industrial processing terms
        for term in _USDA_PROCESSING_PENALTY_TERMS:
            if term in desc:
                score -= 80
        # Heavily penalise preparation/composite terms
        for term in _USDA_PREPARATION_PENALTY_TERMS:
            if term in desc:
                score -= 200

    return score


def _pick_best_result(foods: List[Dict], query: str) -> Dict:
    """
    Select the most relevant USDA search result.

    Scoring:
        - Data type priority  (Foundation > SR Legacy > Survey > Branded)
        - Word-overlap between query and food description
    """
    return max(foods, key=lambda food: _score_usda_candidate(food, query))


def _parse_nutrients(food_nutrients: List[Dict]) -> Dict:
    """
    Parse USDA foodNutrients list.

    Energy is extracted by unit name ('kcal') so kJ values are never
    mistaken for kcal, regardless of their numeric magnitude.
    """
    energy_kcal = 0.0
    energy_kj   = 0.0
    protein     = 0.0
    carbs       = 0.0
    fat         = 0.0
    fiber       = 0.0

    for item in food_nutrients:
        nutrient = item.get("nutrient", {})
        name     = nutrient.get("name", "")
        unit     = nutrient.get("unitName", "").lower()
        nid      = nutrient.get("id", 0)
        amount   = float(item.get("amount", 0) or 0)

        if name == "Energy":
            if unit == "kcal" or nid in _ENERGY_KCAL_IDS:
                energy_kcal = amount
            elif unit == "kj":
                energy_kj = amount
        elif name == "Protein":
            protein = amount
        elif name == "Carbohydrate, by difference":
            carbs = amount
        elif name == "Total lipid (fat)":
            fat = amount
        elif name == "Fiber, total dietary":
            fiber = amount

    # Use kcal; only convert kJ if no kcal entry found
    calories = energy_kcal if energy_kcal > 0 else (energy_kj / 4.184)

    # Fallback: compute from macros if USDA didn't return an energy nutrient
    if calories == 0 and (protein > 0 or carbs > 0 or fat > 0):
        calories = protein * 4.0 + carbs * 4.0 + fat * 9.0

    return {
        "calories": calories,
        "protein":  protein,
        "carbs":    carbs,
        "fat":      fat,
        "fiber":    fiber,
    }


def _scale_nutrition(per100g: Dict, weight_g: float) -> Dict:
    """Scale per-100 g values to actual weight."""
    scale = weight_g / 100.0
    cal_per100g = float(per100g.get("calories", 0) or 0)
    if cal_per100g == 0:
        p = float(per100g.get("protein", 0) or 0)
        c = float(per100g.get("carbs", 0) or 0)
        f = float(per100g.get("fat", 0) or 0)
        if p > 0 or c > 0 or f > 0:
            cal_per100g = p * 4.0 + c * 4.0 + f * 9.0
    return {
        "calories": cal_per100g * scale,
        "protein":  per100g.get("protein",  0) * scale,
        "carbs":    per100g.get("carbs",    0) * scale,
        "fat":      per100g.get("fat",      0) * scale,
        "fiber":    per100g.get("fiber",    0) * scale,
        "source":   per100g.get("source",   "cache"),
        "nutrition_source": per100g.get("nutrition_source", per100g.get("source", "cache")),
        **({"fdc_id": per100g["fdc_id"]} if "fdc_id" in per100g else {}),
        **({"nutrition_confidence": per100g["nutrition_confidence"]}
           if "nutrition_confidence" in per100g else {}),
        **({"usda_description": per100g["usda_description"]}
           if "usda_description" in per100g else {}),
    }


def _with_macro_sanity(nutrition: Dict) -> Dict:
    cal = float(nutrition.get("calories", 0) or 0)
    protein = float(nutrition.get("protein", 0) or 0)
    carbs = float(nutrition.get("carbs", 0) or 0)
    fat = float(nutrition.get("fat", 0) or 0)
    macro_cal = protein * 4.0 + carbs * 4.0 + fat * 9.0
    if cal > 0:
        delta_ratio = abs(macro_cal - cal) / max(cal, 1e-6)
        if delta_ratio > 0.35:
            flags = list(nutrition.get("quality_flags", []))
            flags.append("macro_kcal_mismatch")
            nutrition["quality_flags"] = flags
    return nutrition


# ---------------------------------------------------------------------------
# Offline fallback nutrition database (per 100 g)
# ---------------------------------------------------------------------------

_OFFLINE_NUTRITION: Dict[str, Dict] = {
    # Fruits
    "apple":      {"calories": 52,  "protein": 0.3, "carbs": 14,   "fat": 0.2, "fiber": 2.4},
    "banana":     {"calories": 89,  "protein": 1.1, "carbs": 23,   "fat": 0.3, "fiber": 2.6},
    "orange":     {"calories": 47,  "protein": 0.9, "carbs": 12,   "fat": 0.1, "fiber": 2.4},
    "strawberry": {"calories": 32,  "protein": 0.7, "carbs": 8,    "fat": 0.3, "fiber": 2.0},
    "watermelon": {"calories": 30,  "protein": 0.6, "carbs": 8,    "fat": 0.2, "fiber": 0.4},
    "mango":      {"calories": 60,  "protein": 0.8, "carbs": 15,   "fat": 0.4, "fiber": 1.6},
    "avocado":    {"calories": 160, "protein": 2.0, "carbs": 9,    "fat": 15,  "fiber": 7.0},
    "grapes":     {"calories": 69,  "protein": 0.7, "carbs": 18,   "fat": 0.2, "fiber": 0.9},

    # Vegetables
    "broccoli":   {"calories": 34,  "protein": 2.8, "carbs": 7,    "fat": 0.4, "fiber": 2.6},
    "carrot":     {"calories": 41,  "protein": 0.9, "carbs": 10,   "fat": 0.2, "fiber": 2.8},
    "tomato":     {"calories": 18,  "protein": 0.9, "carbs": 4,    "fat": 0.2, "fiber": 1.2},
    "potato":     {"calories": 77,  "protein": 2.0, "carbs": 17,   "fat": 0.1, "fiber": 2.2},
    "lettuce":    {"calories": 15,  "protein": 1.4, "carbs": 3,    "fat": 0.2, "fiber": 1.3},
    "spinach":    {"calories": 23,  "protein": 2.9, "carbs": 3.6,  "fat": 0.4, "fiber": 2.2},
    "mushroom":   {"calories": 22,  "protein": 3.1, "carbs": 3.3,  "fat": 0.3, "fiber": 1.0},
    "baked_beans": {"calories": 94, "protein": 5.0, "carbs": 18,   "fat": 0.4, "fiber": 4.0},
    "hash_brown":  {"calories": 265,"protein": 3.1, "carbs": 35,   "fat": 13,  "fiber": 2.5},

    # Grains (cooked)
    "rice":       {"calories": 130, "protein": 2.7, "carbs": 28,   "fat": 0.3, "fiber": 0.4},
    "pasta":      {"calories": 131, "protein": 5.0, "carbs": 25,   "fat": 1.1, "fiber": 1.8},
    "bread":      {"calories": 265, "protein": 9.0, "carbs": 49,   "fat": 3.2, "fiber": 2.7},
    "bagel":      {"calories": 257, "protein": 10,  "carbs": 50,   "fat": 1.7, "fiber": 2.3},
    "oatmeal":    {"calories": 71,  "protein": 2.5, "carbs": 12,   "fat": 1.5, "fiber": 1.7},

    # Proteins
    "chicken":         {"calories": 165, "protein": 31,  "carbs": 0,   "fat": 3.6, "fiber": 0},
    "chicken breast":  {"calories": 165, "protein": 31,  "carbs": 0,   "fat": 3.6, "fiber": 0},
    "chicken_breast":  {"calories": 165, "protein": 31,  "carbs": 0,   "fat": 3.6, "fiber": 0},
    "steak":           {"calories": 271, "protein": 25,  "carbs": 0,   "fat": 19,  "fiber": 0},
    "beef":            {"calories": 250, "protein": 26,  "carbs": 0,   "fat": 15,  "fiber": 0},
    "pork":            {"calories": 242, "protein": 27,  "carbs": 0,   "fat": 14,  "fiber": 0},
    "pork chop":       {"calories": 231, "protein": 25,  "carbs": 0,   "fat": 14,  "fiber": 0},
    "pork_chop":       {"calories": 231, "protein": 25,  "carbs": 0,   "fat": 14,  "fiber": 0},
    "lamb":            {"calories": 294, "protein": 25,  "carbs": 0,   "fat": 21,  "fiber": 0},
    "turkey":          {"calories": 189, "protein": 29,  "carbs": 0,   "fat": 7,   "fiber": 0},
    "duck":            {"calories": 337, "protein": 19,  "carbs": 0,   "fat": 28,  "fiber": 0},
    "sausage":         {"calories": 301, "protein": 14,  "carbs": 1.5, "fat": 26,  "fiber": 0},
    "bacon":           {"calories": 541, "protein": 37,  "carbs": 1.4, "fat": 42,  "fiber": 0},
    "ham":             {"calories": 145, "protein": 21,  "carbs": 1.5, "fat": 6,   "fiber": 0},
    "fish":            {"calories": 206, "protein": 22,  "carbs": 0,   "fat": 12,  "fiber": 0},
    "salmon":          {"calories": 208, "protein": 20,  "carbs": 0,   "fat": 13,  "fiber": 0},
    "tuna":            {"calories": 132, "protein": 28,  "carbs": 0,   "fat": 1,   "fiber": 0},
    "crab":            {"calories": 87,  "protein": 18,  "carbs": 0,   "fat": 1.5, "fiber": 0},
    "lobster":         {"calories": 89,  "protein": 19,  "carbs": 0.5, "fat": 0.9, "fiber": 0},
    "egg":             {"calories": 155, "protein": 13,  "carbs": 1.1, "fat": 11,  "fiber": 0},
    "boiled egg":      {"calories": 155, "protein": 13,  "carbs": 1.1, "fat": 11,  "fiber": 0},
    "boiled_egg":      {"calories": 155, "protein": 13,  "carbs": 1.1, "fat": 11,  "fiber": 0},
    "fried egg":       {"calories": 196, "protein": 14,  "carbs": 0.4, "fat": 15,  "fiber": 0},
    "fried_egg":       {"calories": 196, "protein": 14,  "carbs": 0.4, "fat": 15,  "fiber": 0},
    "scrambled egg":   {"calories": 149, "protein": 10,  "carbs": 1.6, "fat": 11,  "fiber": 0},
    "scrambled_egg":   {"calories": 149, "protein": 10,  "carbs": 1.6, "fat": 11,  "fiber": 0},
    "shrimp":          {"calories": 99,  "protein": 24,  "carbs": 0.2, "fat": 0.3, "fiber": 0},
    "tofu":            {"calories": 76,  "protein": 8.0, "carbs": 1.9, "fat": 4.8, "fiber": 0.3},

    # Dairy
    "yogurt":     {"calories": 59,  "protein": 3.5, "carbs": 5.0, "fat": 3.3, "fiber": 0},
    "cheese":     {"calories": 402, "protein": 25,  "carbs": 1.3, "fat": 33,  "fiber": 0},
    "milk":       {"calories": 61,  "protein": 3.2, "carbs": 4.8, "fat": 3.3, "fiber": 0},
    "ice_cream":  {"calories": 207, "protein": 3.5, "carbs": 24,  "fat": 11,  "fiber": 0.7},

    # Fast Food
    "pizza":       {"calories": 266, "protein": 11,  "carbs": 33, "fat": 10, "fiber": 2.5},
    "burger":      {"calories": 295, "protein": 17,  "carbs": 24, "fat": 15, "fiber": 1.5},
    "hamburger":   {"calories": 295, "protein": 17,  "carbs": 24, "fat": 15, "fiber": 1.5},
    "sandwich":    {"calories": 250, "protein": 12,  "carbs": 30, "fat": 9,  "fiber": 2.0},
    "hot dog":     {"calories": 290, "protein": 10,  "carbs": 23, "fat": 17, "fiber": 0.8},
    "hot_dog":     {"calories": 290, "protein": 10,  "carbs": 23, "fat": 17, "fiber": 0.8},
    "fries":       {"calories": 312, "protein": 3.4, "carbs": 41, "fat": 15, "fiber": 3.8},
    "french fries":{"calories": 312, "protein": 3.4, "carbs": 41, "fat": 15, "fiber": 3.8},
    "taco":        {"calories": 226, "protein": 9,   "carbs": 20, "fat": 12, "fiber": 3.0},
    "burrito":     {"calories": 206, "protein": 9,   "carbs": 29, "fat": 6,  "fiber": 3.0},

    # Baked Goods
    "donut":       {"calories": 452, "protein": 5,   "carbs": 51, "fat": 25, "fiber": 1.4},
    "muffin":      {"calories": 377, "protein": 6,   "carbs": 51, "fat": 17, "fiber": 1.5},
    "cookie":      {"calories": 502, "protein": 5.7, "carbs": 64, "fat": 25, "fiber": 2.0},
    "cake":        {"calories": 257, "protein": 4,   "carbs": 42, "fat": 9,  "fiber": 0.8},
    "brownie":     {"calories": 415, "protein": 5,   "carbs": 55, "fat": 21, "fiber": 2.0},
    "pie":         {"calories": 237, "protein": 2.5, "carbs": 34, "fat": 11, "fiber": 1.5},
    "pancake":     {"calories": 227, "protein": 6,   "carbs": 40, "fat": 5,  "fiber": 1.5},

    # Salads (corrected – NOT plain lettuce)
    "salad":         {"calories": 75,  "protein": 3,   "carbs": 8,  "fat": 3,  "fiber": 2.5},
    "caesar_salad":  {"calories": 150, "protein": 5,   "carbs": 8,  "fat": 11, "fiber": 1.5},
    "greek_salad":   {"calories": 100, "protein": 3,   "carbs": 7,  "fat": 7,  "fiber": 2.0},

    # Soups
    "soup":        {"calories": 38,  "protein": 2,   "carbs": 6,  "fat": 1,  "fiber": 1.0},

    # South Asian
    "biryani":       {"calories": 150, "protein": 8,   "carbs": 20, "fat": 4,  "fiber": 1.0},
    "roti":          {"calories": 297, "protein": 8,   "carbs": 60, "fat": 2.8,"fiber": 2.7},
    "chapati":       {"calories": 297, "protein": 8,   "carbs": 60, "fat": 2.8,"fiber": 2.7},
    "naan":          {"calories": 262, "protein": 9,   "carbs": 46, "fat": 5,  "fiber": 2.0},
    "paratha":       {"calories": 320, "protein": 6,   "carbs": 45, "fat": 12, "fiber": 2.0},
    "daal":          {"calories": 116, "protein": 9,   "carbs": 20, "fat": 0.4,"fiber": 8.0},
    "chicken curry": {"calories": 112, "protein": 10,  "carbs": 4.4,"fat": 5.8,"fiber": 1.2},
    "butter chicken":{"calories": 141, "protein": 10,  "carbs": 5,  "fat": 9,  "fiber": 1.0},
    "samosa":        {"calories": 262, "protein": 4.6, "carbs": 24, "fat": 17, "fiber": 2.4},
    "tikka masala":  {"calories": 119, "protein": 10,  "carbs": 5,  "fat": 6.5,"fiber": 1.2},
    "palak paneer":  {"calories": 120, "protein": 6,   "carbs": 8,  "fat": 7,  "fiber": 2.0},
    "gulab jamun":   {"calories": 387, "protein": 5,   "carbs": 55, "fat": 16, "fiber": 0.5},
    "dosa":          {"calories": 133, "protein": 4,   "carbs": 22, "fat": 3.5,"fiber": 1.5},
    "idli":          {"calories": 58,  "protein": 2,   "carbs": 12, "fat": 0.2,"fiber": 0.6},
    "pakora":        {"calories": 260, "protein": 5,   "carbs": 25, "fat": 15, "fiber": 2.0},
    "jalebi":        {"calories": 330, "protein": 3,   "carbs": 65, "fat": 7,  "fiber": 0.2},
    "kheer":         {"calories": 135, "protein": 4,   "carbs": 22, "fat": 4,  "fiber": 0.2},

    # East / Southeast Asian
    "gyoza":         {"calories": 195, "protein": 8,   "carbs": 22, "fat": 8,  "fiber": 1.5},
    "tempura":       {"calories": 230, "protein": 10,  "carbs": 20, "fat": 12, "fiber": 0.8},
    "takoyaki":      {"calories": 200, "protein": 10,  "carbs": 22, "fat": 8,  "fiber": 0.5},
    "okonomiyaki":   {"calories": 185, "protein": 9,   "carbs": 18, "fat": 8,  "fiber": 1.5},
    "onigiri":       {"calories": 170, "protein": 4,   "carbs": 36, "fat": 1,  "fiber": 0.5},
    "mochi":         {"calories": 250, "protein": 3,   "carbs": 55, "fat": 1,  "fiber": 0.2},
    "bao":           {"calories": 200, "protein": 8,   "carbs": 30, "fat": 6,  "fiber": 1.0},
    "char siu":      {"calories": 270, "protein": 22,  "carbs": 12, "fat": 14, "fiber": 0.3},
    "char_siu":      {"calories": 270, "protein": 22,  "carbs": 12, "fat": 14, "fiber": 0.3},
    "peking duck":   {"calories": 337, "protein": 19,  "carbs": 0,  "fat": 28, "fiber": 0.0},
    "peking_duck":   {"calories": 337, "protein": 19,  "carbs": 0,  "fat": 28, "fiber": 0.0},
    "egg tart":      {"calories": 260, "protein": 6,   "carbs": 28, "fat": 14, "fiber": 0.5},
    "egg_tart":      {"calories": 260, "protein": 6,   "carbs": 28, "fat": 14, "fiber": 0.5},
    "satay":         {"calories": 215, "protein": 18,  "carbs": 6,  "fat": 13, "fiber": 0.5},
    "rendang":       {"calories": 195, "protein": 18,  "carbs": 4,  "fat": 12, "fiber": 1.0},
    "nasi goreng":   {"calories": 148, "protein": 6,   "carbs": 24, "fat": 3.5,"fiber": 1.0},
    "nasi_goreng":   {"calories": 148, "protein": 6,   "carbs": 24, "fat": 3.5,"fiber": 1.0},
    "laksa":         {"calories": 68,  "protein": 5,   "carbs": 8,  "fat": 2,  "fiber": 0.8},

    # Trending / health bowls
    "poke bowl":     {"calories": 120, "protein": 10,  "carbs": 15, "fat": 3,  "fiber": 2.0},
    "poke_bowl":     {"calories": 120, "protein": 10,  "carbs": 15, "fat": 3,  "fiber": 2.0},
    "acai bowl":     {"calories": 200, "protein": 4,   "carbs": 35, "fat": 6,  "fiber": 5.0},
    "acai_bowl":     {"calories": 200, "protein": 4,   "carbs": 35, "fat": 6,  "fiber": 5.0},
    "granola":       {"calories": 471, "protein": 10,  "carbs": 64, "fat": 20, "fiber": 5.0},

    # Default
    "default":     {"calories": 150, "protein": 5,   "carbs": 20, "fat": 5,  "fiber": 2.0},
}


def _nutrition_from_fallback(food_name: str, weight_g: float, category: Optional[str] = None) -> Dict:
    """Offline fallback: longest-match lookup → scale to weight_g.

    Priority: specific food lookup FIRST, then category profile, then default.
    Category profile is only used when no specific food entry exists — prevents
    a generic category average (e.g. 75 kcal/100g "salad") from overriding a
    correct specific entry (e.g. 150 kcal/100g "caesar_salad").
    """
    key = food_name.lower().strip()
    used_category_profile = False

    if key in _OFFLINE_NUTRITION:
        per100g = _OFFLINE_NUTRITION[key]
    else:
        # Longest partial match in the specific food table.
        best_key, best_len = None, 0
        for k in _OFFLINE_NUTRITION:
            if k == "default":
                continue
            if k in key or key in k:
                if len(k) > best_len:
                    best_key, best_len = k, len(k)

        if best_key:
            per100g = _OFFLINE_NUTRITION[best_key]
        else:
            # No specific match — try category average, then default.
            category_profile = _category_fallback_profile(category)
            if category_profile:
                per100g = category_profile
                used_category_profile = True
            else:
                per100g = _OFFLINE_NUTRITION["default"]

    result = _scale_nutrition(per100g, weight_g)
    result["source"] = "Offline Fallback"
    result["nutrition_source"] = "Offline Fallback"
    result["nutrition_confidence"] = 0.45 if used_category_profile else 0.35
    return result


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Density lookup ===")
    for food in ["chicken breast", "pizza", "biryani", "gingerbread", "caesar_salad"]:
        print(f"  {food:25s}: {get_food_density(food):.3f} g/ml")

    print("\n=== Typical serving weights ===")
    for food in ["chicken tikka masala", "biryani", "french fries", "apple", "salad"]:
        print(f"  {food:30s}: {get_typical_serving_weight(food)} g")

    print("\n=== Nutrition (offline) ===")
    n = _nutrition_from_fallback("caesar_salad", 150)
    print(f"  Caesar salad 150g: {n['calories']:.1f} kcal  (was 15 kcal/100g, fixed to 150)")
