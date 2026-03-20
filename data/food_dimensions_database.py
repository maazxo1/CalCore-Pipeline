"""
Food Dimensions Database - JSON-Based (Expandable)
Supports multiple regions: Western, South Asian, Middle Eastern, etc.
Easy to update without changing code!

Version: 2.0 - Production Ready
Coverage: 150+ foods (Western + South Asian + more)
"""

import json
import re
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List


class FoodDimensionsDatabase:
    """
    JSON-based food dimensions database
    Easily expandable - just edit the JSON file!
    """
    
    def __init__(self, json_path: Optional[str] = None):
        """
        Initialize database from JSON file
        
        Args:
            json_path: Path to JSON database file (optional)
        """
        if json_path is None:
            # Default: look for foods_database.json in same directory
            json_path = Path(__file__).parent / "foods_database.json"
        
        self.json_path = Path(json_path)
        self.dimensions = {}
        self.metadata = {}
        self._normalized_index: Dict[str, str] = {}
        self._compact_index: Dict[str, str] = {}
        
        # Load database
        self._load_database()
        
        # Shape calculation methods
        self.shape_formulas = {
            'spheroid': self._calc_spheroid,
            'cylindrical': self._calc_cylindrical,
            'rectangular': self._calc_rectangular,
            'disk': self._calc_disk,
            'ellipsoid': self._calc_ellipsoid,
            'conical': self._calc_conical,
            'torus': self._calc_torus,
            'mound': self._calc_mound,
            'pile': self._calc_pile,
            'triangular': self._calc_triangular,
            'folded': self._calc_folded,
            'twisted': self._calc_twisted,
            'irregular': self._calc_irregular,
        }
    
    def _load_database(self):
        """Load food database from JSON file"""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.metadata = data.get('metadata', {})
            self.dimensions = data.get('foods', {})
            
            # Remove comment keys (starting with _)
            self.dimensions = {
                k: v for k, v in self.dimensions.items() 
                if not k.startswith('_')
            }
            self._rebuild_indexes()
            
            print(f"[OK] Loaded {len(self.dimensions)} foods from database")
            print(f"[OK] Regions: {', '.join(self.metadata.get('regions', ['unknown']))}")
            
        except FileNotFoundError:
            print(f"⚠️  Database file not found: {self.json_path}")
            print("   Using fallback minimal database")
            self._create_fallback_database()
        except json.JSONDecodeError as e:
            print(f"⚠️  Error parsing JSON: {e}")
            self._create_fallback_database()
    
    def _create_fallback_database(self):
        """Create minimal fallback if JSON load fails"""
        self.dimensions = {
            'apple': {
                'diameter': 7.5, 'height': 7.0, 'shape': 'spheroid',
                'correction_factor': 0.92, 'typical_weight_g': 182
            },
            'rice': {
                'diameter': 11.0, 'height': 3.5, 'shape': 'mound',
                'correction_factor': 0.95, 'typical_weight_g': 158
            }
        }
        self.metadata = {'version': '1.0-fallback', 'total_foods': 2}
        self._rebuild_indexes()

    @staticmethod
    def _normalize_food_key(name: str) -> str:
        """Normalize key so space/underscore/hyphen variants resolve consistently."""
        tokens = re.findall(r"[a-z0-9]+", name.lower())
        return "_".join(tokens)

    def _rebuild_indexes(self) -> None:
        """Build fast lookup indexes for normalized and compact food keys."""
        self._normalized_index = {}
        self._compact_index = {}

        for original_key in self.dimensions.keys():
            norm_key = self._normalize_food_key(original_key)
            compact_key = norm_key.replace("_", "")

            existing_norm = self._normalized_index.get(norm_key)
            if existing_norm is None or len(original_key) > len(existing_norm):
                self._normalized_index[norm_key] = original_key

            existing_compact = self._compact_index.get(compact_key)
            if existing_compact is None or len(original_key) > len(existing_compact):
                self._compact_index[compact_key] = original_key
    
    # ==================== VOLUME CALCULATION METHODS ====================
    
    def _calc_spheroid(self, dims: Dict) -> float:
        """Sphere/ellipsoid"""
        d = dims.get('diameter', 0)
        h = dims.get('height', d)
        return (4/3) * np.pi * (d/2)**2 * (h/2)
    
    def _calc_cylindrical(self, dims: Dict) -> float:
        """Cylinder"""
        d = dims.get('diameter', 0)
        h = dims.get('height', dims.get('length', 0))
        return np.pi * (d/2)**2 * h
    
    def _calc_rectangular(self, dims: Dict) -> float:
        """Box"""
        l = dims.get('length', 0)
        w = dims.get('width', 0)
        h = dims.get('height', 0)
        return l * w * h
    
    def _calc_disk(self, dims: Dict) -> float:
        """Flat cylinder"""
        d = dims.get('diameter', 0)
        h = dims.get('height', 0)
        return np.pi * (d/2)**2 * h
    
    def _calc_ellipsoid(self, dims: Dict) -> float:
        """Ellipsoid"""
        l = dims.get('length', dims.get('diameter', 0))
        d = dims.get('diameter', l)
        return (4/3) * np.pi * (l/2) * (d/2) * (d/2)
    
    def _calc_conical(self, dims: Dict) -> float:
        """Cone"""
        d = dims.get('diameter', 0)
        h = dims.get('height', 0)
        return (1/3) * np.pi * (d/2)**2 * h
    
    def _calc_torus(self, dims: Dict) -> float:
        """Donut/ring shape. V = 2π²·R·r² where R=centre radius, r=tube radius."""
        d = dims.get('diameter', 0)   # outer diameter
        h = dims.get('height', 0)     # tube diameter (height of ring)
        tube_radius = h / 2           # r
        centre_radius = max(d / 2 - tube_radius, tube_radius * 0.5)  # R (guard against ≤0)
        return 2 * np.pi ** 2 * centre_radius * tube_radius ** 2
    
    def _calc_mound(self, dims: Dict) -> float:
        """Hemispherical mound"""
        d = dims.get('diameter', 0)
        h = dims.get('height', 0)
        return (2/3) * np.pi * (d/2)**2 * h
    
    def _calc_pile(self, dims: Dict) -> float:
        """Pyramid-like pile"""
        l = dims.get('length', dims.get('diameter', 0))
        w = dims.get('width', l)
        h = dims.get('height', 0)
        return 0.5 * l * w * h
    
    def _calc_triangular(self, dims: Dict) -> float:
        """Triangular prism"""
        l = dims.get('length', 0)
        w = dims.get('width', 0)
        h = dims.get('height', 0)
        return 0.5 * l * w * h
    
    def _calc_folded(self, dims: Dict) -> float:
        """Folded (taco-like)"""
        l = dims.get('length', dims.get('diameter', 0))
        w = dims.get('width', l * 0.8)
        h = dims.get('height', 0)
        return 0.5 * l * w * h
    
    def _calc_twisted(self, dims: Dict) -> float:
        """Twisted (cinnamon roll, jalebi)"""
        d = dims.get('diameter', 0)
        h = dims.get('height', 0)
        return 0.6 * np.pi * (d/2)**2 * h
    
    def _calc_irregular(self, dims: Dict) -> float:
        """Irregular shape"""
        return self._calc_spheroid(dims)
    
    # ==================== PUBLIC METHODS ====================
    
    def get_dimensions(self, food_name: str) -> Optional[Dict]:
        """
        Get physical dimensions for a food item
        
        Args:
            food_name: Name of food (e.g., 'biryani', 'apple', 'roti')
            
        Returns:
            Dictionary with dimensions or None if not found
        """
        if not food_name:
            return None

        # Direct original-key lookup first
        raw_key = food_name.lower().strip()
        if raw_key in self.dimensions:
            return self.dimensions[raw_key].copy()

        # Normalized lookups: "hot dog" / "hot_dog" / "hot-dog" all map
        norm_key = self._normalize_food_key(food_name)
        mapped = self._normalized_index.get(norm_key)
        if mapped:
            return self.dimensions[mapped].copy()

        compact_key = norm_key.replace("_", "")
        mapped = self._compact_index.get(compact_key)
        if mapped:
            return self.dimensions[mapped].copy()

        # Partial match fallback – longest normalized key wins
        best_key, best_len = None, 0
        for key in self.dimensions:
            key_norm = self._normalize_food_key(key)
            if key_norm in norm_key or norm_key in key_norm:
                if len(key_norm) > best_len:
                    best_key, best_len = key, len(key_norm)

        if best_key:
            return self.dimensions[best_key].copy()

        return None
    
    def calculate_theoretical_volume(self, food_name: str) -> Optional[float]:
        """
        Calculate theoretical volume based on typical dimensions
        
        Args:
            food_name: Name of food
            
        Returns:
            Volume in cm³ (ml) or None if not found
        """
        dims = self.get_dimensions(food_name)
        if not dims:
            return None
        
        shape = dims.get('shape', 'irregular')
        
        if shape not in self.shape_formulas:
            return None
        
        calc_method = self.shape_formulas[shape]
        
        try:
            volume = calc_method(dims)
            correction = dims.get('correction_factor', 1.0)
            corrected_volume = volume * correction
            return float(corrected_volume)
        except Exception as e:
            print(f"Error calculating volume for {food_name}: {e}")
            return None
    
    def get_correction_factor(self, food_name: str) -> float:
        """Get correction factor for a food"""
        dims = self.get_dimensions(food_name)
        return dims.get('correction_factor', 0.75) if dims else 0.75
    
    def list_all_foods(self) -> List[str]:
        """Get list of all foods in database"""
        return sorted(self.dimensions.keys())
    
    def list_foods_by_region(self, region: str) -> List[str]:
        """Get foods for a specific region"""
        return sorted([
            name for name, dims in self.dimensions.items()
            if dims.get('region', '').lower() == region.lower()
        ])
    
    def list_foods_by_category(self, category: str) -> List[str]:
        """Get foods by category"""
        return sorted([
            name for name, dims in self.dimensions.items()
            if dims.get('category', '').lower() == category.lower()
        ])
    
    def get_food_count(self) -> int:
        """Get total number of foods in database"""
        return len(self.dimensions)
    
    def get_metadata(self) -> Dict:
        """Get database metadata"""
        return self.metadata.copy()
    
    def add_food(self, name: str, dimensions: Dict) -> bool:
        """
        Add a new food to the database (runtime only, not persisted)
        
        Args:
            name: Food name
            dimensions: Dictionary with dimensions
            
        Returns:
            True if added successfully
        """
        try:
            self.dimensions[name.lower().replace(' ', '_')] = dimensions
            self._rebuild_indexes()
            return True
        except Exception as e:
            print(f"Error adding food: {e}")
            return False


# ==================== TESTING ====================

if __name__ == "__main__":
    print("=" * 70)
    print("FOOD DIMENSIONS DATABASE - JSON-BASED TEST")
    print("=" * 70)
    
    db = FoodDimensionsDatabase()
    
    print(f"\n📊 Database Statistics:")
    print(f"   Total foods: {db.get_food_count()}")
    print(f"   Version: {db.get_metadata().get('version', 'unknown')}")
    print(f"   Regions: {', '.join(db.get_metadata().get('regions', []))}")
    
    # Test South Asian foods
    print("\n" + "=" * 70)
    print("SOUTH ASIAN FOODS TEST")
    print("=" * 70)
    
    south_asian_foods = [
        'biryani', 'roti', 'naan', 'daal', 'samosa',
        'chicken_curry', 'butter_chicken', 'gulab_jamun'
    ]
    
    for food in south_asian_foods:
        dims = db.get_dimensions(food)
        vol = db.calculate_theoretical_volume(food)
        
        if dims and vol:
            print(f"\n{food.upper().replace('_', ' ')}")
            print(f"  Weight: {dims.get('typical_weight_g')}g")
            print(f"  Volume: {vol:.1f}ml")
            print(f"  Shape: {dims.get('shape')}")
            print(f"  ✓ SUCCESS")
        else:
            print(f"\n{food.upper()}: ✗ NOT FOUND")
    
    # Test Western foods still work
    print("\n" + "=" * 70)
    print("WESTERN FOODS TEST (Backwards Compatible)")
    print("=" * 70)
    
    western_foods = ['apple', 'burger', 'pizza']
    
    for food in western_foods:
        dims = db.get_dimensions(food)
        if dims:
            print(f"✓ {food}: {dims.get('typical_weight_g')}g")
        else:
            print(f"✗ {food}: NOT FOUND")
    
    print("\n" + "=" * 70)
    print(f"✅ {db.get_food_count()} foods loaded successfully!")
    print("=" * 70)
