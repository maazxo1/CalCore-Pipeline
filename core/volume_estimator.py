"""
Volume Estimator - Voxelization + Height-Field Integration with USDA Correction
================================================================================
Algorithm (updated with techniques from Alex et al. point_cloud_utils.py):

1.  Extract masked 3D points from metric depth + segmentation mask
2.  Compute pixel→physical scaling using proper camera focal length
    (F = width / (2·tan(FOV/2)) for 70° FOV, replacing old FOV_FACTOR=2.0
     which implied 90° and overcounted XY area by ~2×)
3.  Convert pixel coordinates to 3D world coordinates (cm)
4.  Statistical Outlier Removal (SOR, PCL-style) – adapted from Alex et al.
    Removes noisy boundary / reflective-surface depth pixels
5a. Height-field Delaunay integration (PRIMARY) – adapted from Alex et al.
    pc_to_volume():  V = Σ triangle_area × mean_height(3 vertices)
    Geometrically correct for monocular depth (surface-only 3D capture)
5b. Voxelization (COMPLEMENT) – kept as a cross-check / blend
6.  Apply USDA typical-volume correction factor
7.  Convert corrected volume to weight using FAO densities

Expected Accuracy:
    Without correction : 40-50% MAPE
    With USDA correction: 25-35% MAPE (target)
    Per CalcCore spec   : simple fruits 20-25%, complex foods 35-45%

Multi-image aggregation:
    Use aggregate_multi_image_volumes() to combine results from 3 photos
    of the same food. Takes median volume → more robust than single-image.
"""

import numpy as np
from scipy.spatial import cKDTree, Delaunay
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# USDA Typical Serving Volumes & Weights
# Source: USDA FNDDS (Food and Nutrient Database for Dietary Studies)
# These are used as correction anchors, NOT as final values.
# ---------------------------------------------------------------------------
USDA_TYPICAL_VOLUMES: Dict[str, Dict] = {
    # ── Fruits ────────────────────────────────────────────────────────────
    "apple":       {"volume_ml": 182,  "weight_g": 182,  "serving": "1 medium (182g)"},
    "banana":      {"volume_ml": 126,  "weight_g": 118,  "serving": "1 medium (118g)"},
    "orange":      {"volume_ml": 151,  "weight_g": 131,  "serving": "1 medium (131g)"},
    "strawberry":  {"volume_ml": 60,   "weight_g": 36,   "serving": "2 large (36g)"},
    "blueberry":   {"volume_ml": 148,  "weight_g": 148,  "serving": "1 cup (148g)"},
    "blackberry":  {"volume_ml": 144,  "weight_g": 144,  "serving": "1 cup (144g)"},
    "mango":       {"volume_ml": 250,  "weight_g": 165,  "serving": "1 cup sliced"},
    "grapes":      {"volume_ml": 150,  "weight_g": 92,   "serving": "1 cup (92g)"},
    "watermelon":  {"volume_ml": 280,  "weight_g": 280,  "serving": "1 cup diced"},
    "pineapple":   {"volume_ml": 165,  "weight_g": 82,   "serving": "1/2 cup chunks"},
    "pear":        {"volume_ml": 178,  "weight_g": 178,  "serving": "1 medium (178g)"},
    "peach":       {"volume_ml": 150,  "weight_g": 150,  "serving": "1 medium (150g)"},
    "avocado":     {"volume_ml": 155,  "weight_g": 150,  "serving": "1/2 avocado"},
    "lemon":       {"volume_ml": 108,  "weight_g": 108,  "serving": "1 medium"},
    "cherry":      {"volume_ml": 68,   "weight_g": 68,   "serving": "10 cherries"},
    "kiwi":        {"volume_ml": 75,   "weight_g": 75,   "serving": "1 medium"},

    # ── Vegetables ───────────────────────────────────────────────────────
    "broccoli":    {"volume_ml": 165,  "weight_g": 91,   "serving": "1 cup florets"},
    "carrot":      {"volume_ml": 122,  "weight_g": 78,   "serving": "1 medium"},
    "tomato":      {"volume_ml": 123,  "weight_g": 123,  "serving": "1 medium"},
    "potato":      {"volume_ml": 213,  "weight_g": 173,  "serving": "1 medium"},
    "lettuce":     {"volume_ml": 85,   "weight_g": 47,   "serving": "1 cup shredded"},
    "cucumber":    {"volume_ml": 119,  "weight_g": 119,  "serving": "1/2 cup sliced"},
    "bell_pepper": {"volume_ml": 149,  "weight_g": 119,  "serving": "1 medium"},
    "onion":       {"volume_ml": 196,  "weight_g": 110,  "serving": "1 medium"},
    "spinach":     {"volume_ml": 30,   "weight_g": 30,   "serving": "1 cup raw"},
    "corn":        {"volume_ml": 154,  "weight_g": 154,  "serving": "1 ear"},
    "asparagus":   {"volume_ml": 90,   "weight_g": 90,   "serving": "6 spears"},
    "mushroom":    {"volume_ml": 70,   "weight_g": 70,   "serving": "1 cup sliced"},
    "baked_beans": {"volume_ml": 200,  "weight_g": 210,  "serving": "1/2 cup / small can"},
    "hash_brown":  {"volume_ml": 80,   "weight_g": 68,   "serving": "1 patty"},

    # ── Grains & Breads ──────────────────────────────────────────────────
    "rice":        {"volume_ml": 205,  "weight_g": 197,  "serving": "1 cup cooked"},
    "pasta":       {"volume_ml": 140,  "weight_g": 140,  "serving": "1 cup cooked"},
    "bread":       {"volume_ml": 112,  "weight_g": 29,   "serving": "1 slice"},
    "bagel":       {"volume_ml": 254,  "weight_g": 89,   "serving": "1 medium (3.5in)"},
    "naan":        {"volume_ml": 100,  "weight_g": 90,   "serving": "1 piece"},
    "roti":        {"volume_ml": 55,   "weight_g": 40,   "serving": "1 piece"},
    "chapati":     {"volume_ml": 55,   "weight_g": 40,   "serving": "1 piece"},
    "paratha":     {"volume_ml": 108,  "weight_g": 65,   "serving": "1 piece"},
    "pita":        {"volume_ml": 60,   "weight_g": 28,   "serving": "1 pita"},
    "tortilla":    {"volume_ml": 45,   "weight_g": 45,   "serving": "1 medium"},
    "oatmeal":     {"volume_ml": 234,  "weight_g": 234,  "serving": "1 cup cooked"},
    "cereal":      {"volume_ml": 250,  "weight_g": 40,   "serving": "1 cup dry"},

    # ── Fast Food ─────────────────────────────────────────────────────────
    "pizza":       {"volume_ml": 165,  "weight_g": 107,  "serving": "1 slice regular"},
    "burger":      {"volume_ml": 259,  "weight_g": 220,  "serving": "1 medium burger"},
    "hot_dog":     {"volume_ml": 86,   "weight_g": 76,   "serving": "1 hot dog (bun)"},
    "fries":       {"volume_ml": 335,  "weight_g": 117,  "serving": "medium serving"},
    "taco":        {"volume_ml": 113,  "weight_g": 85,   "serving": "1 hard shell taco"},
    "sandwich":    {"volume_ml": 273,  "weight_g": 150,  "serving": "1 sandwich"},
    "burrito":     {"volume_ml": 250,  "weight_g": 220,  "serving": "1 burrito"},
    "donut":       {"volume_ml": 108,  "weight_g": 52,   "serving": "1 glazed donut"},

    # ── Proteins ─────────────────────────────────────────────────────────
    "chicken_breast": {"volume_ml": 166, "weight_g": 174, "serving": "1 breast cooked"},
    "chicken":     {"volume_ml": 166,  "weight_g": 174,  "serving": "3oz portion"},
    "steak":       {"volume_ml": 217,  "weight_g": 226,  "serving": "8oz steak"},
    "beef":        {"volume_ml": 217,  "weight_g": 226,  "serving": "8oz portion"},
    "pork":        {"volume_ml": 170,  "weight_g": 175,  "serving": "6oz portion"},
    "pork_chop":   {"volume_ml": 195,  "weight_g": 200,  "serving": "1 chop (200g)"},
    "lamb":        {"volume_ml": 170,  "weight_g": 175,  "serving": "6oz portion"},
    "turkey":      {"volume_ml": 170,  "weight_g": 175,  "serving": "6oz portion"},
    "duck":        {"volume_ml": 150,  "weight_g": 155,  "serving": "5oz portion"},
    "sausage":     {"volume_ml": 86,   "weight_g": 85,   "serving": "1 link cooked (85g)"},
    "bacon":       {"volume_ml": 40,   "weight_g": 28,   "serving": "2 strips cooked"},
    "ham":         {"volume_ml": 170,  "weight_g": 175,  "serving": "6oz slice"},
    "salmon":      {"volume_ml": 170,  "weight_g": 178,  "serving": "6oz fillet"},
    "tuna":        {"volume_ml": 130,  "weight_g": 140,  "serving": "5oz can drained"},
    "fish":        {"volume_ml": 140,  "weight_g": 140,  "serving": "5oz fillet"},
    "shrimp":      {"volume_ml": 85,   "weight_g": 90,   "serving": "6 large shrimp"},
    "crab":        {"volume_ml": 85,   "weight_g": 85,   "serving": "3oz portion"},
    "lobster":     {"volume_ml": 85,   "weight_g": 85,   "serving": "3oz portion"},
    "egg":         {"volume_ml": 48,   "weight_g": 50,   "serving": "1 large egg"},
    "boiled_egg":  {"volume_ml": 48,   "weight_g": 50,   "serving": "1 large egg"},
    "fried_egg":   {"volume_ml": 48,   "weight_g": 46,   "serving": "1 large egg"},
    "scrambled_egg": {"volume_ml": 110, "weight_g": 100, "serving": "2 eggs scrambled"},
    "tofu":        {"volume_ml": 126,  "weight_g": 130,  "serving": "3oz piece"},

    # ── Dairy ─────────────────────────────────────────────────────────────
    "yogurt":      {"volume_ml": 245,  "weight_g": 245,  "serving": "1 cup (8oz)"},
    "cheese":      {"volume_ml": 30,   "weight_g": 28,   "serving": "1oz slice"},
    "ice_cream":   {"volume_ml": 132,  "weight_g": 74,   "serving": "1/2 cup"},
    "milk":        {"volume_ml": 244,  "weight_g": 244,  "serving": "1 cup"},

    # ── Baked Goods & Desserts ───────────────────────────────────────────
    "cake":        {"volume_ml": 145,  "weight_g": 80,   "serving": "1 slice"},
    "muffin":      {"volume_ml": 200,  "weight_g": 113,  "serving": "1 large muffin"},
    "cookie":      {"volume_ml": 77,   "weight_g": 40,   "serving": "2 medium cookies"},
    "brownie":     {"volume_ml": 90,   "weight_g": 56,   "serving": "1 brownie"},
    "pie":         {"volume_ml": 150,  "weight_g": 113,  "serving": "1 slice"},
    "pancake":     {"volume_ml": 77,   "weight_g": 38,   "serving": "1 medium pancake"},
    "waffle":      {"volume_ml": 75,   "weight_g": 75,   "serving": "1 waffle"},

    # ── South Asian ───────────────────────────────────────────────────────
    "biryani":        {"volume_ml": 300, "weight_g": 270, "serving": "1 cup"},
    "daal":           {"volume_ml": 245, "weight_g": 245, "serving": "1 cup"},
    "samosa":         {"volume_ml": 100, "weight_g": 70,  "serving": "1 samosa"},
    "chicken_curry":  {"volume_ml": 245, "weight_g": 250, "serving": "1 cup"},
    "butter_chicken": {"volume_ml": 245, "weight_g": 250, "serving": "1 cup"},
    "tikka_masala":   {"volume_ml": 245, "weight_g": 250, "serving": "1 cup"},
    "palak_paneer":   {"volume_ml": 245, "weight_g": 240, "serving": "1 cup"},
    "paneer":         {"volume_ml": 100, "weight_g": 100, "serving": "1 piece"},
    "gulab_jamun":    {"volume_ml": 60,  "weight_g": 60,  "serving": "1 piece"},
    "dosa":           {"volume_ml": 120, "weight_g": 100, "serving": "1 dosa"},
    "idli":           {"volume_ml": 60,  "weight_g": 50,  "serving": "1 idli"},
    "pakora":         {"volume_ml": 80,  "weight_g": 60,  "serving": "3 pieces"},
    "jalebi":         {"volume_ml": 80,  "weight_g": 80,  "serving": "1 serving"},
    "kheer":          {"volume_ml": 245, "weight_g": 245, "serving": "1 cup"},

    # ── East Asian ───────────────────────────────────────────────────────
    "gyoza":          {"volume_ml": 120, "weight_g": 100, "serving": "5 pieces"},
    "tempura":        {"volume_ml": 150, "weight_g": 120, "serving": "1 serving"},
    "takoyaki":       {"volume_ml": 120, "weight_g": 100, "serving": "6 pieces"},
    "okonomiyaki":    {"volume_ml": 200, "weight_g": 180, "serving": "1 piece"},
    "onigiri":        {"volume_ml": 120, "weight_g": 110, "serving": "1 onigiri"},
    "mochi":          {"volume_ml": 80,  "weight_g": 60,  "serving": "1 piece"},
    "bao":            {"volume_ml": 100, "weight_g": 80,  "serving": "1 bao"},
    "char_siu":       {"volume_ml": 150, "weight_g": 160, "serving": "3oz portion"},
    "peking_duck":    {"volume_ml": 150, "weight_g": 140, "serving": "1 serving"},
    "egg_tart":       {"volume_ml": 70,  "weight_g": 60,  "serving": "1 tart"},

    # ── Southeast Asian ──────────────────────────────────────────────────
    "satay":          {"volume_ml": 100, "weight_g": 90,  "serving": "3 skewers"},
    "rendang":        {"volume_ml": 150, "weight_g": 150, "serving": "1 serving"},
    "nasi_goreng":    {"volume_ml": 300, "weight_g": 270, "serving": "1 plate"},
    "laksa":          {"volume_ml": 450, "weight_g": 440, "serving": "1 bowl"},
    "poke_bowl":      {"volume_ml": 350, "weight_g": 330, "serving": "1 bowl"},

    # ── Trending / health ────────────────────────────────────────────────
    "acai_bowl":      {"volume_ml": 350, "weight_g": 300, "serving": "1 bowl"},
    "granola":        {"volume_ml": 120, "weight_g": 60,  "serving": "1/2 cup"},

    # ── Salads ────────────────────────────────────────────────────────────
    "salad":          {"volume_ml": 200, "weight_g": 100,  "serving": "1 cup mixed"},
    "caesar_salad":   {"volume_ml": 200, "weight_g": 100,  "serving": "1 cup"},
    "greek_salad":    {"volume_ml": 220, "weight_g": 150,  "serving": "1 cup"},

    # ── Soups ─────────────────────────────────────────────────────────────
    "soup":        {"volume_ml": 245,  "weight_g": 245,  "serving": "1 cup"},
    "ramen":       {"volume_ml": 500,  "weight_g": 490,  "serving": "1 bowl"},
    "pho":         {"volume_ml": 500,  "weight_g": 490,  "serving": "1 bowl"},

    # ── Sushi ─────────────────────────────────────────────────────────────
    "sushi":       {"volume_ml": 30,   "weight_g": 28,   "serving": "1 piece"},
    "sushi_roll":  {"volume_ml": 180,  "weight_g": 170,  "serving": "6 pieces"},

    # ── Snacks ────────────────────────────────────────────────────────────
    "chips":       {"volume_ml": 250,  "weight_g": 28,   "serving": "1oz (1 bag)"},
    "popcorn":     {"volume_ml": 750,  "weight_g": 28,   "serving": "3 cups"},

    # ── Default (unknown food) ────────────────────────────────────────────
    "default":     {"volume_ml": 200,  "weight_g": 150,  "serving": "1 serving"},
}


# ---------------------------------------------------------------------------
# FAO Density Database (g/ml)
# Source: FAO Food Density Tables + USDA Food Engineering Handbook
# ---------------------------------------------------------------------------
FAO_DENSITIES: Dict[str, float] = {
    # Fruits
    "apple": 0.64, "banana": 0.94, "orange": 0.87, "strawberry": 0.60,
    "blueberry": 0.60, "blackberry": 0.60, "watermelon": 0.96,
    "grapes": 0.61, "pineapple": 0.54, "mango": 0.64, "pear": 0.59,
    "peach": 0.61, "avocado": 0.97, "lemon": 0.80, "cherry": 0.90, "kiwi": 0.90,

    # Vegetables
    "tomato": 0.96, "lettuce": 0.55, "cucumber": 0.97, "carrot": 0.64,
    "broccoli": 0.55, "cauliflower": 0.44, "spinach": 0.38, "potato": 0.81,
    "onion": 0.56, "bell_pepper": 0.54, "corn": 0.72, "asparagus": 0.62,
    "mushroom": 0.50, "cabbage": 0.60,
    "baked_beans": 1.05, "hash_brown": 0.85,

    # Grains (cooked)
    "rice": 0.96, "pasta": 0.92, "bread": 0.26, "garlic_bread": 0.26, "bagel": 0.35,
    "oatmeal": 0.85, "naan": 0.55, "roti": 0.45, "chapati": 0.45,
    "paratha": 0.60, "pita": 0.40, "tortilla": 0.60, "cereal": 0.16,
    "pancake": 0.50, "waffle": 0.50, "biryani": 0.90,

    # Proteins
    "chicken": 1.05, "chicken_breast": 1.05, "beef": 1.04, "steak": 1.04,
    "pork": 1.03, "pork_chop": 1.03, "lamb": 1.04, "turkey": 1.04, "duck": 1.05,
    "sausage": 1.00, "bacon": 0.95, "ham": 1.03,
    "fish": 1.04, "salmon": 1.05, "tuna": 1.08, "crab": 1.03, "lobster": 1.04,
    "egg": 1.03, "boiled_egg": 1.03, "fried_egg": 1.00, "scrambled_egg": 0.95,
    "tofu": 1.03, "shrimp": 1.06,

    # Dairy
    "milk": 1.03, "yogurt": 1.05, "cheese": 1.15, "butter": 0.91,
    "ice_cream": 0.56, "cream": 1.01,

    # Fast Food
    "pizza": 0.65, "burger": 0.85, "fries": 0.35, "sandwich": 0.55,
    "hot_dog": 0.88, "taco": 0.75, "burrito": 0.85,

    # Baked Goods
    "cake": 0.60, "donut": 0.48, "muffin": 0.55, "cookie": 0.52,
    "brownie": 0.75, "pie": 0.75,

    # Snacks
    "chips": 0.30, "popcorn": 0.17, "chocolate": 1.26,

    # South Asian
    "daal": 0.95, "samosa": 0.70, "chicken_curry": 0.92,
    "butter_chicken": 0.94, "tikka_masala": 0.91, "palak_paneer": 0.88,
    "gulab_jamun": 1.00, "paneer": 1.20,
    "dosa": 0.35, "idli": 0.65, "pakora": 0.70, "jalebi": 1.10, "kheer": 1.02,

    # East / Southeast Asian
    "gyoza": 0.85, "tempura": 0.70, "takoyaki": 0.90, "okonomiyaki": 0.82,
    "onigiri": 0.95, "mochi": 0.95, "bao": 0.55, "char_siu": 1.05,
    "peking_duck": 1.00, "egg_tart": 0.90,
    "satay": 1.00, "rendang": 1.05, "nasi_goreng": 0.92, "laksa": 0.88,

    # Trending / health
    "poke_bowl": 0.88, "acai_bowl": 0.82, "granola": 0.45,

    # Salads & Soups
    "salad": 0.50, "caesar_salad": 0.45, "greek_salad": 0.68,
    "soup": 0.98, "ramen": 0.82, "pho": 0.85,

    # Sushi
    "sushi": 0.90, "sushi_roll": 0.90,

    # Default
    "default": 0.75,
}


class VolumeEstimator:
    """
    Food volume estimator combining height-field Delaunay integration and
    voxelization with USDA correction.

    Requires metric depth in centimetres from Depth Anything V2 Metric model.

    Key improvements over naive voxelization:
      - Proper camera focal-length transform (70° FOV, not 90°)
      - SOR noise filter on 3D points (PCL-style, from Alex et al.)
      - Height-field Delaunay integration (from Alex et al. pc_to_volume)

    Usage:
        estimator = VolumeEstimator()
        result = estimator.estimate_volume(depth_cm, mask, "apple")
        print(result['volume_ml'], result['estimated_weight_g'])
    """

    # Camera model
    CAMERA_FOV_DEG  = 70.0    # Typical smartphone rear camera (60–70°).
    #   FOV_FACTOR=2.0 (old) implied 90° → overcounted XY by (2/1.4)²≈2×.

    # Voxelization grid
    VOXEL_SIZE_CM   = 0.5     # cm per voxel side

    # Quality gates
    MIN_MASK_PIXELS     = 100
    MAX_DEPTH_RANGE_CM  = 50.0   # Flag if depth range > 50 cm (multiple objects?)
    MAX_POINTS          = 50_000  # Downsample above this for O(N) performance

    # Statistical Outlier Removal (PCL-style, adapted from Alex et al.)
    SOR_K_NEIGHBORS = 10    # Number of nearest neighbours
    SOR_STD_MULT    = 2.0   # Outlier threshold: mean + k·std

    # Height-field Delaunay integration (adapted from Alex et al. pc_to_volume)
    MAX_DELAUNAY_PTS = 8_000  # Sub-sample before triangulation for speed

    # USDA anchoring guardrails (known foods only).
    # MIN raised to 0.10 now that max_depth is correctly set to 5 m (not 20 m),
    # which eliminates the ~64× raw volume overestimate that previously required
    # the 0.003 floor.  Correction factors persistently below 0.10 now indicate
    # a genuinely unusual scene rather than a model calibration artifact.
    USDA_CORRECTION_FACTOR_MIN = 0.10
    USDA_CORRECTION_FACTOR_MAX = 5.0
    USDA_VOLUME_MIN_MULT = 0.25
    # Volume/weight MAX raised to 20× — a whole cake is ~15 slices of 100g each.
    # The taxonomy max_weight_g (e.g. 3000g for cake) acts as the final ceiling
    # via weight_guardrails, so raising this multiplier only unlocks large items
    # whose taxonomy explicitly allows high max weights.
    USDA_VOLUME_MAX_MULT = 20.0
    USDA_WEIGHT_MIN_MULT = 0.25
    USDA_WEIGHT_MAX_MULT = 20.0

    def __init__(self, voxel_size_cm: float = 0.5):
        self.voxel_size = voxel_size_cm
        self.usda_volumes = USDA_TYPICAL_VOLUMES
        self.densities = FAO_DENSITIES
        # Seeded RNG for deterministic downsampling (same input → same output).
        self._rng = np.random.default_rng(42)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_volume(
        self,
        depth_cm: np.ndarray,
        mask: np.ndarray,
        food_name: str,
    ) -> Dict:
        """
        Estimate food volume using height-field integration + voxelization.

        Args:
            depth_cm : (H, W) numpy array, depth in centimetres (> 0).
                       NaN / inf are filtered automatically.
            mask     : (H, W) boolean or uint8. True/non-zero = food pixel.
            food_name: Food classification label (e.g. "apple", "pizza").

        Returns:
            Dict with keys:
                volume_ml          – corrected volume in ml
                volume_raw_ml      – uncorrected raw volume
                hf_volume_ml       – height-field Delaunay volume (pre-blend)
                estimated_weight_g – weight in grams
                correction_factor  – USDA correction applied (1.0 = no correction)
                confidence         – 0.0–0.90
                method             – description of calculation path
                voxel_count        – unique voxels occupied
                points_count       – 3-D points after SOR filter
                usda_typical_ml    – USDA reference volume (or None)
                usda_typical_g     – USDA reference weight (or None)
                density_g_ml       – FAO density used
                pixel_to_cm        – physical scale (cm per pixel at food depth)
                median_depth_cm    – median depth of food pixels
                multi_object_flag  – True if depth range suggests >1 object
                dimensions_used    – always False (geometry-based)
                error              – only present when calculation failed
        """
        food_key = _normalize_name(food_name)
        mask_bool = mask.astype(bool)

        # ── Validate mask ─────────────────────────────────────────────────
        if mask_bool.sum() < self.MIN_MASK_PIXELS:
            return self._error_result("Mask too small", food_key)

        h, w = depth_cm.shape

        # ── Filter to valid depth within mask ─────────────────────────────
        masked_depth = depth_cm[mask_bool]
        valid_filter = np.isfinite(masked_depth) & (masked_depth > 0)

        if valid_filter.sum() < self.MIN_MASK_PIXELS:
            return self._error_result("Insufficient valid depth data", food_key)

        ys, xs = np.where(mask_bool)
        valid_ys = ys[valid_filter]
        valid_xs = xs[valid_filter]
        valid_zs = masked_depth[valid_filter]

        # ── Flag suspicious depth range ───────────────────────────────────
        depth_range = float(valid_zs.max() - valid_zs.min())
        multi_object_flag = depth_range > self.MAX_DEPTH_RANGE_CM

        # ── Step 1: Proper focal length (physically accurate) ─────────────
        # F = w / (2·tan(FOV/2)).  For 70°: F ≈ 0.714·w.
        # Old FOV_FACTOR=2.0 → implied 90°, overcounted X/Y by ~(2/1.4)≈1.43×
        # → volume overcounted by ≈2× before USDA correction.
        F = w / (2.0 * np.tan(np.radians(self.CAMERA_FOV_DEG / 2.0)))
        median_depth_cm = float(np.median(valid_zs))
        if median_depth_cm <= 0:
            return self._error_result("Invalid median depth (≤ 0)", food_key)

        pixel_to_cm = median_depth_cm / F    # cm per pixel at food depth

        # ── Step 2: 3-D world coordinates ─────────────────────────────────
        cx, cy = w / 2.0, h / 2.0
        X = (valid_xs.astype(np.float64) - cx) / F * valid_zs.astype(np.float64)
        Y = (valid_ys.astype(np.float64) - cy) / F * valid_zs.astype(np.float64)
        Z = valid_zs.astype(np.float64)

        # ── Step 3: SOR – Statistical Outlier Removal ─────────────────────
        # Removes depth noise at food edges / reflective surfaces.
        # Adapted from Alex et al. (point_cloud_utils.sor_filter).
        points_3d = np.stack([X, Y, Z], axis=1)
        if len(points_3d) > 3 * self.SOR_K_NEIGHBORS:
            points_3d, _ = self._sor_filter(
                points_3d, k=self.SOR_K_NEIGHBORS, n_std=self.SOR_STD_MULT
            )
            X = points_3d[:, 0]
            Y = points_3d[:, 1]
            Z = points_3d[:, 2]

        n_pts = len(X)
        if n_pts < 10:
            return self._error_result("Too few points after SOR filter", food_key)

        # ── Step 4: Downsample for performance ────────────────────────────
        if n_pts > self.MAX_POINTS:
            idx = self._rng.choice(n_pts, self.MAX_POINTS, replace=False)
            X, Y, Z = X[idx], Y[idx], Z[idx]
            n_pts = self.MAX_POINTS

        # ── Step 5a: Height-field Delaunay volume (primary) ───────────────
        # Adapted from Alex et al. (point_cloud_utils.pc_to_volume).
        # Height above base = reference_depth − food_depth, where
        # reference_depth ≈ 95th-percentile depth in food mask (deepest
        # visible food pixels = where food meets plate / bowl rim).
        z_base  = float(np.percentile(Z, 95))
        heights = np.clip(z_base - Z, 0.0, None)   # cm above reference plane
        hf_volume_ml = self._height_field_volume(
            np.stack([X, Y], axis=1), heights, self.MAX_DELAUNAY_PTS,
            rng=self._rng,
        )

        # ── Step 5b: Voxelization (complementary) ─────────────────────────
        vx = np.floor(X / self.voxel_size).astype(np.int32)
        vy = np.floor(Y / self.voxel_size).astype(np.int32)
        vz = np.floor(Z / self.voxel_size).astype(np.int32)
        unique_voxels = np.unique(np.stack([vx, vy, vz], axis=1), axis=0)
        n_voxels = len(unique_voxels)
        voxel_volume_ml = float(n_voxels * self.voxel_size ** 3)

        # ── Step 6: Blend methods ─────────────────────────────────────────
        # Height-field is geometrically correct for surface-sampled depth.
        # Voxelization serves as a robustness cross-check.
        if hf_volume_ml > 1.0:
            raw_volume_ml = 0.7 * hf_volume_ml + 0.3 * voxel_volume_ml
            method_suffix = "height_field+voxel"
        else:
            # Flat foods (soup surface, tortilla) have near-zero height variation;
            # fall back to voxelization and let USDA correction do the work.
            raw_volume_ml = voxel_volume_ml
            method_suffix = "voxel_only"

        # ── Step 7: USDA correction ───────────────────────────────────────
        corrected_volume_ml, correction_factor, usda_entry = (
            self._apply_usda_correction(raw_volume_ml, food_key)
        )

        # ── Step 8: Confidence ────────────────────────────────────────────
        confidence = self._calc_confidence(
            correction_factor, usda_entry is not None, multi_object_flag
        )

        # ── Step 9: Weight ────────────────────────────────────────────────
        density = self._get_density(food_key)
        weight_from_volume = corrected_volume_ml * density

        if usda_entry:
            usda_weight = float(usda_entry["weight_g"])
            if correction_factor < 0.10:
                # Depth model is severely overestimating (>10× off); volume-
                # derived weight is unreliable.  Use USDA weight as anchor
                # directly instead of scaling it by the tiny correction factor.
                anchor_weight = usda_weight
            elif correction_factor == 1.0 and weight_from_volume > usda_weight * 5:
                # Large/whole item path (raw_factor was below MIN): depth volume is
                # trusted directly.  The USDA single-serving anchor would drag the
                # estimate toward one serving; use volume-derived weight entirely.
                weight_g = weight_from_volume
                anchor_weight = None
            else:
                anchor_weight = usda_weight * correction_factor
            if anchor_weight is not None:
                weight_g = 0.6 * weight_from_volume + 0.4 * anchor_weight
        else:
            weight_g = weight_from_volume

        if usda_entry:
            usda_weight = float(usda_entry["weight_g"])
            min_w = usda_weight * self.USDA_WEIGHT_MIN_MULT
            max_w = usda_weight * self.USDA_WEIGHT_MAX_MULT
            weight_g = float(np.clip(weight_g, min_w, max_w))
        else:
            # Unknown foods get a conservative hard cap; final bounds are applied
            # downstream by weight_guardrails in VolumeCalculator.
            weight_g = float(np.clip(weight_g, 5.0, 1200.0))

        return {
            "volume_ml":           float(corrected_volume_ml),
            "volume_raw_ml":       float(raw_volume_ml),
            "hf_volume_ml":        float(hf_volume_ml),
            "estimated_weight_g":  weight_g,
            "correction_factor":   float(correction_factor),
            "confidence":          float(confidence),
            "method":              f"voxelization_{method_suffix}",
            "voxel_count":         int(n_voxels),
            "points_count":        int(n_pts),
            "usda_typical_ml":     float(usda_entry["volume_ml"]) if usda_entry else None,
            "usda_typical_g":      float(usda_entry["weight_g"])  if usda_entry else None,
            "density_g_ml":        float(density),
            "pixel_to_cm":         float(pixel_to_cm),
            "median_depth_cm":     float(median_depth_cm),
            "multi_object_flag":   bool(multi_object_flag),
            "dimensions_used":     False,
        }

    # ------------------------------------------------------------------
    # New methods from Alex et al. (point_cloud_utils.py)
    # ------------------------------------------------------------------

    @staticmethod
    def _sor_filter(
        points: np.ndarray,
        k: int = 10,
        n_std: float = 2.0,
    ):
        """Statistical Outlier Removal – PCL-style (adapted from Alex et al.).

        Each point's mean k-NN distance is computed; points whose mean distance
        exceeds (global_mean + n_std × global_std) are marked as outliers.

        Args:
            points: (N, 3) array of 3-D points.
            k:      Number of nearest neighbours.
            n_std:  Std-multiplier for the inlier threshold.

        Returns:
            Tuple (inlier_points, bool_mask).
        """
        n = len(points)
        k_actual = min(k + 1, n)   # +1: cKDTree query includes the point itself

        kdtree = cKDTree(points)
        distances, _ = kdtree.query(points, k=k_actual)

        # Column 0 is self-distance (always 0); use columns 1:
        mean_dists = (
            distances[:, 1:].mean(axis=1) if k_actual > 1 else distances[:, 0]
        )

        threshold = mean_dists.mean() + n_std * mean_dists.std()
        sor_mask  = mean_dists <= threshold

        return points[sor_mask], sor_mask

    @staticmethod
    def _height_field_volume(
        xy_points: np.ndarray,
        heights: np.ndarray,
        max_pts: int = 8_000,
        rng: "np.random.Generator | None" = None,
    ) -> float:
        """Integrate food volume as height × footprint using Delaunay mesh.

        Adapted from Alex et al. point_cloud_utils.pc_to_volume().

        For each Delaunay triangle in the food's XY footprint, volume
        contribution = triangle_area × mean_height(3 vertices).
        Geometrically correct for monocular depth maps (surface-only 3-D).

        Long "boundary-spanning" triangles (edge > 75% of food extent) are
        discarded to handle concave or partially visible food regions.

        Args:
            xy_points: (N, 2) XY coordinates in cm.
            heights:   (N,)  heights above reference plane in cm.
            max_pts:   Sub-sample to this many points before triangulation.

        Returns:
            Volume in ml (1 cm³ = 1 ml).
        """
        pos_mask = heights > 0.0
        if pos_mask.sum() < 4:
            return 0.0

        xy = xy_points[pos_mask]
        h  = heights[pos_mask]

        # Sub-sample for Delaunay speed (O(N log N))
        if len(xy) > max_pts:
            _rng = rng if rng is not None else np.random.default_rng(42)
            idx = _rng.choice(len(xy), max_pts, replace=False)
            xy  = xy[idx]
            h   = h[idx]

        try:
            tri = Delaunay(xy)
        except Exception:
            return 0.0

        simplices = tri.simplices                          # (T, 3) vertex indices

        v1 = xy[simplices[:, 0]]                          # (T, 2)
        v2 = xy[simplices[:, 1]]
        v3 = xy[simplices[:, 2]]

        # Edge lengths – vectorised (Alex et al. use the same Heron approach)
        a = np.linalg.norm(v1 - v2, axis=1)
        b = np.linalg.norm(v2 - v3, axis=1)
        c = np.linalg.norm(v3 - v1, axis=1)

        # Reject boundary-spanning / degenerate triangles
        xy_extent = float(np.max(np.ptp(xy, axis=0)))
        max_edge  = max(xy_extent * 0.75, 1.0)           # at least 1 cm
        keep      = np.maximum(np.maximum(a, b), c) <= max_edge
        a, b, c   = a[keep], b[keep], c[keep]
        s_keep    = simplices[keep]

        # Heron's formula for triangle area – vectorised
        s  = (a + b + c) / 2.0
        s2 = s * (s - a) * (s - b) * (s - c)
        pos = s2 > 0.0
        a, b, c, s, s2, s_keep = a[pos], b[pos], c[pos], s[pos], s2[pos], s_keep[pos]
        areas = np.sqrt(s2)

        # Mean height per triangle
        avg_h = (h[s_keep[:, 0]] + h[s_keep[:, 1]] + h[s_keep[:, 2]]) / 3.0

        total_volume = float(np.sum(areas * avg_h))   # cm³ = ml
        return total_volume

    # ------------------------------------------------------------------
    # Internal helpers (unchanged)
    # ------------------------------------------------------------------

    def _apply_usda_correction(
        self, raw_volume: float, food_key: str
    ) -> Tuple[float, float, Optional[Dict]]:
        """Apply USDA typical-volume correction.

        Returns (corrected_volume, bounded_factor, usda_entry_or_None).
        """
        usda_entry = _lookup_usda(food_key, self.usda_volumes)

        if usda_entry is None or raw_volume <= 0:
            return raw_volume, 1.0, None

        raw_factor = usda_entry["volume_ml"] / raw_volume

        if raw_factor < self.USDA_CORRECTION_FACTOR_MIN:
            # raw_volume >> USDA single-serving typical: this is a large or whole item
            # (e.g. a whole cake whose depth voxelization reads ~4,700 mL vs 167 mL for
            # one slice).  Applying the floor correction would crush an accurate depth
            # estimate to 10% of reality.  Instead, trust the raw depth volume directly
            # and let downstream weight_guardrails / taxonomy max_weight_g bound it.
            corrected = raw_volume
            bounded = 1.0
        else:
            bounded = float(
                np.clip(
                    raw_factor,
                    self.USDA_CORRECTION_FACTOR_MIN,
                    self.USDA_CORRECTION_FACTOR_MAX,
                )
            )
            corrected = raw_volume * bounded

        min_vol = float(usda_entry["volume_ml"]) * self.USDA_VOLUME_MIN_MULT
        max_vol = float(usda_entry["volume_ml"]) * self.USDA_VOLUME_MAX_MULT
        corrected = float(np.clip(corrected, min_vol, max_vol))
        return corrected, bounded, usda_entry

    def _calc_confidence(
        self,
        correction_factor: float,
        has_usda: bool,
        multi_object: bool,
    ) -> float:
        """Confidence based on USDA correction deviation from 1.0."""
        if not has_usda:
            base = 0.45
        else:
            deviation = abs(1.0 - correction_factor)
            base = float(max(0.25, 0.80 - deviation * 0.5))

        if multi_object:
            base *= 0.75

        return float(min(base, 0.90))

    def _get_density(self, food_key: str) -> float:
        """Lookup FAO density; longest partial match if exact match fails."""
        if food_key in self.densities:
            return self.densities[food_key]

        best_key, best_len = None, 0
        for k in self.densities:
            if k == "default":
                continue
            if k in food_key or food_key in k:
                if len(k) > best_len:
                    best_key, best_len = k, len(k)

        return self.densities[best_key] if best_key else self.densities["default"]

    def _error_result(self, reason: str, food_key: str) -> Dict:
        """Return a safe default result when calculation cannot proceed."""
        usda_entry = _lookup_usda(food_key, self.usda_volumes)
        density    = self._get_density(food_key)

        default_vol = float(usda_entry["volume_ml"]) if usda_entry else 200.0
        default_wgt = float(usda_entry["weight_g"])  if usda_entry else 150.0

        return {
            "volume_ml":           default_vol,
            "volume_raw_ml":       0.0,
            "hf_volume_ml":        0.0,
            "estimated_weight_g":  default_wgt,
            "correction_factor":   1.0,
            "confidence":          0.20,
            "method":              "usda_default_fallback",
            "voxel_count":         0,
            "points_count":        0,
            "usda_typical_ml":     default_vol,
            "usda_typical_g":      default_wgt,
            "density_g_ml":        float(density),
            "pixel_to_cm":         0.0,
            "median_depth_cm":     0.0,
            "multi_object_flag":   False,
            "dimensions_used":     False,
            "error":               reason,
        }


# ---------------------------------------------------------------------------
# Multi-image aggregation
# ---------------------------------------------------------------------------

def aggregate_multi_image_volumes(results_list: List[Dict]) -> Dict:
    """
    Aggregate VolumeEstimator results from 2–3 images of the same food.

    Strategy:
        - Median volume → robust against outlier images
        - Consistency bonus added to confidence when images agree closely
        - Falls back to single result if only one valid estimate exists

    Args:
        results_list: List of dicts returned by VolumeEstimator.estimate_volume()

    Returns:
        Single aggregated result dict with 'n_images_used' key added.
    """
    if not results_list:
        return {}
    if len(results_list) == 1:
        r = results_list[0].copy()
        r["n_images_used"] = 1
        return r

    # Prefer results from actual voxelization (not fallback)
    valid = [r for r in results_list if r.get("voxel_count", 0) > 0]
    if not valid:
        valid = results_list

    volumes = np.array([r["volume_ml"]          for r in valid], dtype=np.float64)
    weights = np.array([r["estimated_weight_g"] for r in valid], dtype=np.float64)
    lows    = np.array([r.get("weight_low_g", r["estimated_weight_g"]) for r in valid], dtype=np.float64)
    highs   = np.array([r.get("weight_high_g", r["estimated_weight_g"]) for r in valid], dtype=np.float64)
    confs   = np.array([r["confidence"]         for r in valid], dtype=np.float64)

    agg_volume = float(np.median(volumes))
    agg_weight = float(np.median(weights))

    # Consistency bonus: low CV → images agree → higher confidence
    cv = float(np.std(volumes) / (np.mean(volumes) + 1e-6))
    consistency_bonus = float(max(0.0, 0.10 * (1.0 - min(cv, 1.0))))
    agg_conf = float(min(0.95, float(np.mean(confs)) + consistency_bonus))

    # Build output from the highest-confidence individual result
    best = max(valid, key=lambda r: r["confidence"])
    out  = best.copy()
    out["volume_ml"]          = agg_volume
    out["estimated_weight_g"] = agg_weight
    out["weight_low_g"]       = float(np.median(lows))
    out["weight_high_g"]      = float(np.median(highs))
    out["confidence"]         = agg_conf
    out["method"]             = f"voxelization_multi_image_{len(valid)}imgs"
    out["n_images_used"]      = len(valid)
    out["volume_per_image"]   = volumes.tolist()

    return out


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _normalize_name(food_name: str) -> str:
    """Lowercase, strip, spaces/hyphens → underscores."""
    return food_name.lower().strip().replace(" ", "_").replace("-", "_")


def _lookup_usda(
    food_key: str, usda_volumes: Dict[str, Dict]
) -> Optional[Dict]:
    """
    Look up USDA typical volume for a food.

    Priority:
    1. Exact match
    2. Match ignoring underscores
    3. Longest partial substring match (more specific key wins)
    """
    if food_key in usda_volumes:
        return usda_volumes[food_key]

    no_under = food_key.replace("_", "")
    for k, v in usda_volumes.items():
        if k.replace("_", "") == no_under:
            return v

    best_val, best_len = None, 0
    for k, v in usda_volumes.items():
        if k == "default":
            continue
        if k in food_key or food_key in k:
            if len(k) > best_len:
                best_val, best_len = v, len(k)

    return best_val


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("VolumeEstimator – Self Test")
    print("=" * 65)

    estimator = VolumeEstimator()

    # Synthetic apple: hemisphere depth profile.
    # Camera at 0 cm; food surface ranges from 50 cm (top) to 55 cm (edge).
    # Depth variation exercises the height-field integration path.
    H, W = 480, 640
    cy_c, cx_c = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    radius_px = 80
    dist_sq = (xx - cx_c) ** 2 + (yy - cy_c) ** 2
    inside = dist_sq < radius_px ** 2

    depth_map = np.full((H, W), 65.0, dtype=np.float32)   # background 65 cm
    # Hemisphere: centre at 50 cm, edge at 55 cm (5 cm height profile)
    depth_map[inside] = (
        55.0 - 5.0 * np.sqrt(np.maximum(0.0, 1.0 - dist_sq[inside] / radius_px ** 2))
    )

    result = estimator.estimate_volume(depth_map, inside, "apple")

    print(f"\nFood     : apple (synthetic hemisphere)")
    print(f"HF vol   : {result['hf_volume_ml']:.1f} ml  (height-field Delaunay)")
    print(f"Raw vol  : {result['volume_raw_ml']:.1f} ml  (blended)")
    print(f"Corr vol : {result['volume_ml']:.1f} ml   (USDA typical ≈ 182 ml)")
    print(f"Weight   : {result['estimated_weight_g']:.1f} g  (USDA typical ≈ 182 g)")
    print(f"Conf     : {result['confidence']:.0%}")
    print(f"Factor   : {result['correction_factor']:.3f}")
    print(f"Voxels   : {result['voxel_count']:,}")
    print(f"px/cm    : {result['pixel_to_cm']:.4f}")
    print(f"Method   : {result['method']}")
    print("\n[OK] VolumeEstimator OK")
