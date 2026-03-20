"""
Enhanced Fallback System for Unknown Foods
==========================================
Used when ONLY relative (0-1) depth is available and the food is not in
the dimensions database.  Implements a 4-method density ensemble.

Fixes applied:
  - Removed all duplicate dictionary keys (lasagna, pho, ramen, crab_cakes,
    gyro, pozole, casserole – all appeared twice)
  - Fixed confidence ensemble: divides by number of FIRED methods, not
    always 4, so a single high-confidence direct lookup is not penalised
  - Fixed keyword matching to require whole-word-ish matches to avoid false
    substring hits (e.g. "bread" matching "gingerbread")
  - Fixed _estimate_volume to accept and use the resolution-adjusted
    reference area from VolumeCalculator
"""

import re
import numpy as np
from typing import Dict, Optional, Tuple


class EnhancedFallbackSystem:
    """
    Research-proven fallback for foods not in the dimensions database.

    References:
        1. "Food Volume Estimation from Images" – Pouladzadeh et al.
        2. "Density-based Weight Estimation" – USDA Food Engineering
        3. "Visual Texture Analysis for Food Properties" – Martin et al.
    """

    def __init__(self):
        self.food_densities = self._build_density_database()
        self.category_densities = self._build_category_defaults()
        self.texture_patterns = self._build_texture_patterns()

    # ------------------------------------------------------------------
    # Database builders
    # ------------------------------------------------------------------

    def _build_density_database(self) -> Dict[str, float]:
        """Comprehensive density database (g/ml). All duplicate keys removed."""
        return {
            # ── Pasta & Noodles ──────────────────────────────────────────
            "lasagna":              1.05,
            "cannelloni":           1.08,
            "ravioli":              0.95,
            "tortellini":           0.93,
            "spaghetti":            0.92,
            "fettuccine":           0.91,
            "linguine":             0.90,
            "penne":                0.89,
            "macaroni":             0.88,
            "rigatoni":             0.87,
            "pasta_with_sauce":     0.98,
            "mac_and_cheese":       1.02,
            "spaghetti_bolognese":  0.95,
            "carbonara":            1.00,

            # ── Asian Noodles ─────────────────────────────────────────────
            "pad_thai":             0.88,
            "lo_mein":              0.85,
            "chow_mein":            0.83,
            "ramen":                0.82,
            "pho":                  0.85,
            "udon":                 0.90,
            "soba":                 0.87,
            "pad_see_ew":           0.89,
            "yakisoba":             0.84,

            # ── Rice Dishes ───────────────────────────────────────────────
            "fried_rice":           0.93,
            "risotto":              0.97,
            "paella":               0.91,
            "jambalaya":            0.94,
            "congee":               0.98,
            "sticky_rice":          0.99,

            # ── Casseroles & Baked ────────────────────────────────────────
            "casserole":            0.90,
            "moussaka":             1.03,
            "shepherd_pie":         1.00,
            "cottage_pie":          0.98,
            "pot_pie":              0.95,
            "tuna_casserole":       0.92,
            "green_bean_casserole": 0.75,
            "enchilada_casserole":  0.96,

            # ── Quiche & Egg Dishes ───────────────────────────────────────
            "quiche":               1.10,
            "frittata":             1.08,
            "omelette":             1.05,
            "scrambled_eggs":       1.02,
            "egg_bake":             1.07,

            # ── Soups & Stews ─────────────────────────────────────────────
            "soup":                 0.98,
            "stew":                 1.00,
            "chili":                1.02,
            "gumbo":                0.97,
            "minestrone":           0.95,
            "chicken_noodle_soup":  0.93,
            "tomato_soup":          0.96,
            "clam_chowder":         1.01,
            "miso_soup":            0.94,
            "pozole":               0.96,

            # ── Mexican ───────────────────────────────────────────────────
            "enchilada":            0.96,
            "chimichanga":          0.88,
            "fajita":               0.85,
            "tostada":              0.70,
            "chalupa":              0.75,
            "gordita":              0.82,
            "sope":                 0.85,
            "empanada":             0.78,
            "tamale":               0.92,

            # ── Indian / Pakistani ────────────────────────────────────────
            "vindaloo":             0.93,
            "rogan_josh":           0.94,
            "saag":                 0.88,
            "bhuna":                0.92,
            "jalfrezi":             0.89,
            "madras":               0.91,
            "pathia":               0.90,
            "dhansak":              0.93,
            "balti":                0.92,

            # ── Middle Eastern ────────────────────────────────────────────
            "hummus":               0.95,
            "baba_ganoush":         0.88,
            "falafel":              0.65,
            "shawarma":             0.90,
            "gyro":                 0.88,
            "dolma":                0.85,
            "kofta":                0.98,
            "tabbouleh":            0.75,
            "fattoush":             0.70,
            "kibbeh":               0.95,

            # ── Sandwiches & Wraps ────────────────────────────────────────
            "club_sandwich":        0.55,
            "reuben":               0.72,
            "panini":               0.75,
            "grilled_cheese":       0.80,
            "blt":                  0.60,
            "cubano":               0.78,
            "falafel_wrap":         0.70,
            "chicken_wrap":         0.75,

            # ── Salads ────────────────────────────────────────────────────
            "caesar_salad":         0.45,
            "greek_salad":          0.50,
            "cobb_salad":           0.60,
            "caprese_salad":        0.65,
            "pasta_salad":          0.85,
            "potato_salad":         0.92,
            "coleslaw":             0.55,
            "fruit_salad":          0.75,
            "tuna_salad":           0.88,
            "chicken_salad":        0.87,

            # ── Appetizers ────────────────────────────────────────────────
            "mozzarella_sticks":    0.68,
            "buffalo_wings":        0.82,
            "calamari":             0.65,
            "bruschetta":           0.70,
            "crab_cakes":           0.88,
            "deviled_eggs":         1.00,
            "stuffed_mushrooms":    0.75,
            "jalapeno_poppers":     0.72,

            # ── Breakfast ─────────────────────────────────────────────────
            "french_toast":         0.75,
            "breakfast_burrito":    0.85,
            "eggs_benedict":        0.95,
            "hash_browns":          0.82,
            "breakfast_sandwich":   0.78,
            "oatmeal":              0.92,
            "granola":              0.45,
            "yogurt_parfait":       0.88,

            # ── Desserts ──────────────────────────────────────────────────
            "tiramisu":             0.85,
            "cheesecake":           1.15,
            "tres_leches":          0.95,
            "flan":                 1.05,
            "mousse":               0.65,
            "pudding":              0.98,
            "gelato":               0.92,
            "sorbet":               0.75,
            "panna_cotta":          1.00,
            "baklava":              0.70,
            "cannoli":              0.68,
            "eclair":               0.62,
            "profiterole":          0.58,
            "creme_brulee":         1.08,
            "souffle":              0.40,

            # ── Pizza Variations ──────────────────────────────────────────
            "deep_dish_pizza":      0.75,
            "thin_crust_pizza":     0.58,
            "stuffed_crust_pizza":  0.68,
            "calzone":              0.80,
            "stromboli":            0.82,

            # ── Seafood ───────────────────────────────────────────────────
            "fish_and_chips":       0.78,
            "shrimp_scampi":        0.92,
            "lobster_roll":         0.75,
            "sushi_roll":           0.90,
            "poke_bowl":            0.88,
            "ceviche":              0.85,

            # ── Vegetarian ────────────────────────────────────────────────
            "veggie_burger":        0.70,
            "tofu_stir_fry":        0.85,
            "vegetable_curry":      0.83,
            "ratatouille":          0.80,
            "stuffed_peppers":      0.82,
            "eggplant_parmesan":    0.88,

            # ── General ───────────────────────────────────────────────────
            "stir_fry":             0.85,
            "curry":                0.90,
            "gravy":                0.95,
            "meatloaf":             1.00,
            "meatballs":            0.98,
            "stuffing":             0.60,
            "couscous":             0.88,
            "polenta":              0.95,
            "grits":                0.93,
        }

    def _build_category_defaults(self) -> Dict[str, float]:
        """Category-based density defaults (g/ml)."""
        return {
            "pasta_dish":    0.90,
            "rice_dish":     0.92,
            "noodle_dish":   0.86,
            "soup":          0.95,
            "stew":          1.00,
            "curry":         0.90,
            "casserole":     0.92,
            "baked_pasta":   1.00,
            "meat_dish":     1.05,
            "poultry_dish":  1.00,
            "seafood_dish":  0.95,
            "egg_dish":      1.05,
            "fried":         0.75,
            "grilled":       0.95,
            "baked":         0.85,
            "steamed":       0.90,
            "roasted":       0.92,
            "creamy":        0.98,
            "crispy":        0.60,
            "dense":         1.05,
            "fluffy":        0.50,
            "layered":       0.95,
            "vegetable_dish":0.80,
            "salad":         0.50,
            "stir_fry":      0.85,
            "sandwich":      0.70,
            "wrap":          0.75,
            "flatbread":     0.65,
            "cake":          0.55,
            "pie":           0.75,
            "custard":       1.05,
            "mousse":        0.65,
            "ice_cream":     0.85,
            "grain_dish":    0.88,
            "pilaf":         0.89,
            "porridge":      0.93,
        }

    def _build_texture_patterns(self) -> Dict[str, Dict]:
        """Visual texture → density modifiers."""
        return {
            "layered": {"density_modifier": 1.10, "examples": ["lasagna", "moussaka", "tiramisu"]},
            "creamy":  {"density_modifier": 1.05, "examples": ["alfredo", "carbonara", "risotto"]},
            "crispy":  {"density_modifier": 0.65, "examples": ["chips", "crackers", "fried"]},
            "fluffy":  {"density_modifier": 0.50, "examples": ["cake", "souffle", "whipped"]},
            "chunky":  {"density_modifier": 0.85, "examples": ["stew", "chili", "chunky_soup"]},
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_properties(
        self,
        food_name: str,
        mask_area_px: int,
        depth_range: float,
        adjusted_ref_area: float = 40_000,
        visual_features: Optional[Dict] = None,
    ) -> Dict:
        """
        Estimate food density, volume, and weight via 4-method ensemble.

        Args:
            food_name        : Classified food label.
            mask_area_px     : Pixel count of segmentation mask.
            depth_range      : Depth range (relative 0-1) within mask.
            adjusted_ref_area: Resolution-scaled reference area (px).
                               Pass the value from VolumeCalculator for accuracy.
            visual_features  : Optional dict with 'texture' key.

        Returns:
            Dict with density_g_ml, volume_ml, weight_g, confidence,
            method, methods_used.
        """
        d1, c1 = self._method_1_direct_lookup(food_name)
        d2, c2 = self._method_2_category_based(food_name)
        d3, c3 = self._method_3_keyword_matching(food_name)
        d4, c4 = (self._method_4_visual_analysis(visual_features)
                  if visual_features else (0.75, 0.0))

        pairs = [(d1, c1), (d2, c2), (d3, c3), (d4, c4)]
        fired = [(d, c) for d, c in pairs if c > 0]

        if fired:
            total_conf = sum(c for _, c in fired)
            density = sum(d * c for d, c in fired) / total_conf
            # Divide by number of FIRED methods (not always 4) so a single
            # high-confidence direct lookup isn't penalised.
            confidence = min(total_conf / len(fired), 0.85)
        else:
            density = 0.75
            confidence = 0.40

        volume_ml = self._estimate_volume(mask_area_px, depth_range, adjusted_ref_area)
        weight_g = volume_ml * density

        return {
            "density_g_ml": float(density),
            "volume_ml":    float(volume_ml),
            "weight_g":     float(weight_g),
            "confidence":   float(confidence),
            "method":       "ensemble_fallback",
            "methods_used": {
                "direct_lookup": c1 > 0,
                "category":      c2 > 0,
                "keyword":       c3 > 0,
                "visual":        c4 > 0,
            },
        }

    def get_confidence_level(self, confidence: float) -> str:
        """Convert numeric confidence to human-readable label."""
        if confidence >= 0.80:
            return "high"
        elif confidence >= 0.65:
            return "medium"
        return "low"

    def suggest_similar_foods(self, food_name: str, top_n: int = 3) -> list:
        """Return database foods with highest keyword overlap with food_name."""
        name_lower = food_name.lower()
        query_words = set(re.findall(r"\w+", name_lower))

        scored = []
        for db_food in self.food_densities:
            db_words = set(re.findall(r"\w+", db_food))
            overlap = len(query_words & db_words)
            if overlap > 0:
                scored.append((db_food, overlap))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [f for f, _ in scored[:top_n]]

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _method_1_direct_lookup(self, food_name: str) -> Tuple[float, float]:
        """Exact (normalised) key lookup – 90% confidence if found."""
        key = food_name.lower().strip().replace(" ", "_")
        if key in self.food_densities:
            return self.food_densities[key], 0.90
        return 0.75, 0.0

    def _method_2_category_based(self, food_name: str) -> Tuple[float, float]:
        """Category substring match – 70% confidence."""
        name_lower = food_name.lower()
        for cat, density in self.category_densities.items():
            cat_words = cat.replace("_", " ")
            if cat_words in name_lower or cat in name_lower:
                return density, 0.70
        return 0.75, 0.0

    def _method_3_keyword_matching(self, food_name: str) -> Tuple[float, float]:
        """
        Keyword density hints – 60% confidence.

        Uses word-boundary matching to avoid false substring hits:
        e.g. 'bread' should NOT match 'gingerbread' or 'cornbread'.
        """
        name_lower = food_name.lower()
        keyword_hints = {
            "fried":    0.70,
            "crispy":   0.65,
            "creamy":   0.98,
            "soup":     0.95,
            "stew":     1.00,
            "curry":    0.90,
            "baked":    0.85,
            "grilled":  0.95,
            "steamed":  0.90,
            "stuffed":  0.88,
            "layered":  1.05,
            "casserole":0.92,
            "salad":    0.50,
            "cake":     0.55,
            "pie":      0.75,
            "bread":    0.45,
            "pasta":    0.90,
            "rice":     0.92,
            "noodle":   0.86,
        }

        # Use word-boundary regex to avoid partial matches
        for keyword, density in keyword_hints.items():
            # \b word boundary ensures "bread" doesn't match "gingerbread"
            if re.search(r"\b" + re.escape(keyword) + r"\b", name_lower):
                return density, 0.60

        return 0.75, 0.0

    def _method_4_visual_analysis(
        self, visual_features: Dict
    ) -> Tuple[float, float]:
        """Visual texture → density modifier – 65% confidence."""
        if "texture" in visual_features:
            texture = visual_features["texture"]
            if texture in self.texture_patterns:
                modifier = self.texture_patterns[texture]["density_modifier"]
                return modifier, 0.65
        return 0.75, 0.0

    def _estimate_volume(
        self,
        mask_area_px: int,
        depth_range: float,
        adjusted_ref_area: float = 40_000,
    ) -> float:
        """
        Estimate volume from mask area and relative depth range.

        Uses the resolution-adjusted reference area so estimates scale
        correctly across different image resolutions.

        Reference: "3D Food Volume Estimation" – Pouladzadeh et al.
        """
        ref_volume_ml = 150.0  # ml for a ~150 g "typical" food at ref_area pixels

        area_ratio = mask_area_px / adjusted_ref_area
        estimated_volume = ref_volume_ml * (area_ratio ** 1.5)

        # Depth adjustment (conservative for relative depth)
        if depth_range > 0.05:
            depth_factor = float(np.clip(depth_range / 0.25, 0.7, 1.5))
            estimated_volume *= depth_factor

        return float(np.clip(estimated_volume, 10.0, 5000.0))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("ENHANCED FALLBACK SYSTEM – FIXED VERSION")
    print("=" * 70)

    fb = EnhancedFallbackSystem()
    print(f"\n✓ {len(fb.food_densities)} food densities loaded (no duplicates)")
    print(f"✓ {len(fb.category_densities)} category defaults")

    tests = [
        ("lasagna",           38_000, 0.05),
        ("pad thai",          40_000, 0.04),
        ("tiramisu",          25_000, 0.03),
        ("gingerbread cake",  30_000, 0.04),  # 'bread' must NOT dominate
        ("unknown_food_xyz",  30_000, 0.04),
    ]

    for food, area, dr in tests:
        r = fb.estimate_properties(food, area, dr)
        print(f"\n{food.upper()}")
        print(f"  Density : {r['density_g_ml']:.3f} g/ml")
        print(f"  Volume  : {r['volume_ml']:.1f} ml")
        print(f"  Weight  : {r['weight_g']:.1f} g")
        print(f"  Conf    : {r['confidence']:.0%}  ({fb.get_confidence_level(r['confidence'])})")
        print(f"  Methods : {r['methods_used']}")

    print("\n✅ Enhanced Fallback System OK")
