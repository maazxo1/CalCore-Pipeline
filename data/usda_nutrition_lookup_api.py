"""
DEPRECATED — superseded by usda_nutrition_lookup.py
=====================================================
This module is kept only for backward compatibility with any code that
imports from here directly.  All logic now lives in usda_nutrition_lookup.py.

The USDA API key is no longer hardcoded; set the USDA_API_KEY environment
variable instead (see usda_nutrition_lookup.py).
"""

import warnings

warnings.warn(
    "usda_nutrition_lookup_api.py is deprecated. "
    "Import from usda_nutrition_lookup instead.",
    DeprecationWarning,
    stacklevel=2,
)

from data.usda_nutrition_lookup import (  # noqa: E402
    USDAFoodDatabase,
    get_nutrition_info,
    get_food_density,
    get_typical_serving_weight,
)

# Legacy class alias (old code used USDANutritionLookup)
USDANutritionLookup = USDAFoodDatabase

__all__ = [
    "USDAFoodDatabase",
    "USDANutritionLookup",
    "get_nutrition_info",
    "get_food_density",
    "get_typical_serving_weight",
]
