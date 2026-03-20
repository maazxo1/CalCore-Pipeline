"""
Depth Estimation Module using Depth Anything V2 (FIXED VERSION)

Supports TWO modes:
1. RELATIVE depth (default) - normalized 0-1, good for visualization
2. METRIC depth - actual distance in meters, required for volume estimation

FIXES:
- Detects when metric depth returns all zeros
- Automatic fallback to relative depth
- Better max_depth defaults for food photography
- Warning messages for debugging
"""

import cv2
import os
import torch
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Tuple, Optional

# Add Depth-Anything-V2 to path
depth_anything_path = Path("Depth-Anything-V2")
if depth_anything_path.exists():
    sys.path.insert(0, str(depth_anything_path))
    # Also add metric_depth subdirectory
    metric_depth_path = depth_anything_path / "metric_depth"
    if metric_depth_path.exists():
        sys.path.insert(0, str(metric_depth_path))

from depth_anything_v2.dpt import DepthAnythingV2


def _img_fingerprint(img: np.ndarray) -> tuple:
    """Cheap stable cache key — replaces full-image MD5 (~12 ms saved per call).

    Samples 5 pixel values spread across the flattened array.  Collision
    probability is negligible for typical pipeline usage (same-session caching).
    """
    flat = img.ravel()
    n    = len(flat)
    return (
        img.shape, img.dtype.str,
        int(flat[0]), int(flat[n // 4]), int(flat[n // 2]),
        int(flat[3 * n // 4]), int(flat[-1]),
    )


class DepthEstimator:
    """
    Depth Anything V2 wrapper supporting both relative and metric depth
    WITH AUTOMATIC FALLBACK FOR METRIC DEPTH FAILURES
    """

    def __init__(self,
                 model_size: str = "large",
                 checkpoint_path: Optional[str] = None,
                 use_metric: bool = False,
                 metric_dataset: str = "hypersim",  # 'hypersim' for indoor, 'vkitti' for outdoor
                 max_depth: float = 5.0):  # CHANGED: 5m default for food (was 20m)
        """
        Initialize Depth Anything V2

        Args:
            model_size: 'small', 'base', or 'large'
            checkpoint_path: Path to model checkpoint (auto-detected if None)
            use_metric: If True, use metric depth model (outputs meters)
            metric_dataset: 'hypersim' for indoor scenes, 'vkitti' for outdoor
            max_depth: Maximum depth in meters (5m for table-top, 20m for rooms)
        """
        self.model_size = model_size
        self.checkpoint_path = checkpoint_path
        self.use_metric = use_metric
        self.use_metric_original = use_metric  # Remember original choice
        self.metric_dataset = metric_dataset
        self.max_depth = max_depth
        _dev = os.getenv("DEPTH_DEVICE", "auto").lower()
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if _dev == "auto" else _dev
        self.model = None
        self._cached_depth = None
        self._cached_image_hash = None
        self._last_metric_failed = False
        self._last_output_is_metric = False

    @staticmethod
    def _candidate_checkpoints(path: Path, use_metric: bool) -> list[Path]:
        """Return checkpoint candidates for renamed weight files and alt dirs."""
        candidates = [path]

        # Local rename compatibility: some repos use "large" instead of "vitl".
        if not use_metric and path.name == "depth_anything_v2_vitl.pth":
            candidates.append(path.with_name("depth_anything_v2_large.pth"))

        metric_dir = Path("Depth-Anything-V2/metric_depth/checkpoints")
        for candidate in list(candidates):
            candidates.append(metric_dir / candidate.name)

        # Deduplicate while preserving order
        deduped = []
        seen = set()
        for candidate in candidates:
            key = str(candidate)
            if key not in seen:
                seen.add(key)
                deduped.append(candidate)
        return deduped

    def load_model(self) -> None:
        """Load the depth model"""
        if self.use_metric:
            print(f"Loading Depth Anything V2 METRIC ({self.metric_dataset})...")
        else:
            print("Loading Depth Anything V2 (relative)...")

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }

        encoder_map = {'small': 'vits', 'base': 'vitb', 'large': 'vitl'}
        encoder = encoder_map[self.model_size]

        # Determine checkpoint path
        if self.checkpoint_path is None:
            if self.use_metric:
                # Metric model checkpoint
                self.checkpoint_path = f"weights/depth_anything_v2_metric_{self.metric_dataset}_{encoder}.pth"
            else:
                # Relative depth checkpoint
                self.checkpoint_path = f"weights/depth_anything_v2_{encoder}.pth"

        print(f"   Using device: {self.device}")
        print(f"   Checkpoint: {self.checkpoint_path}")

        # Build model config
        config = model_configs[encoder].copy()

        # The metric class (from metric_depth/depth_anything_v2/dpt.py, highest
        # priority in sys.path) accepts max_depth and uses it as a scale factor:
        #   depth = sigmoid_head_output * max_depth  (in meters)
        # Hypersim was trained with max_depth=20, but for close-up food photography
        # (~30-60 cm) we use self.max_depth (default 5 m) for correct scaling.
        # Only inject when use_metric=True; the relative class has no max_depth param.
        if self.use_metric:
            config['max_depth'] = self.max_depth

        self.model = DepthAnythingV2(**config)

        # Load weights
        checkpoint = Path(self.checkpoint_path)
        resolved = None
        for candidate in self._candidate_checkpoints(checkpoint, use_metric=self.use_metric):
            if candidate.exists():
                resolved = candidate
                break
        if resolved is None:
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        self.checkpoint_path = str(resolved)

        self.model.load_state_dict(
            torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        )
        self.model = self.model.to(self.device).eval()

        if self.use_metric:
            print(f"✅ Depth Anything V2 METRIC loaded (max_depth={self.max_depth}m)")
        else:
            print("✅ Depth Anything V2 loaded (relative depth)")

    def estimate_depth(self,
                       image: np.ndarray,
                       apply_filter: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate depth map from image.

        Returns:
            depth_map: If metric output is valid, depth in METERS.
                       Otherwise normalized relative depth 0-1.
            depth_colored: colored visualization
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Cheap fingerprint — avoids full-image MD5 hashing (~12 ms on 4K images)
        image_hash = _img_fingerprint(image)
        if image_hash == self._cached_image_hash and self._cached_depth is not None:
            depth = self._cached_depth.copy()
            is_metric_output = self._last_output_is_metric
            self._last_metric_failed = not is_metric_output and self.use_metric
        else:
            # NOTE: dpt.py's infer_image/image2tensor already converts BGR->RGB internally,
            # so we pass raw BGR input.
            print(
                "   Running depth inference (first run compiles CUDA kernels, please wait)...",
                flush=True,
            )
            # FP16 autocast: 1.5–2× faster on CUDA with negligible quality loss.
            # Depth values are 0–5 m; FP16 has ~1 mm precision at 5 m — sufficient.
            _use_fp16 = (self.device == "cuda")
            with torch.no_grad():
                with torch.amp.autocast(
                    device_type=("cuda" if _use_fp16 else "cpu"),
                    dtype=torch.float16,
                    enabled=_use_fp16,
                ):
                    raw_depth = self.model.infer_image(image)
            print("   Depth inference complete.", flush=True)

            is_metric_output = bool(self.use_metric)
            self._last_metric_failed = False
            depth = raw_depth

            # Validate metric output per image; do not mutate global mode.
            if self.use_metric:
                depth_range = float(depth.max() - depth.min())
                invalid_metric = False

                if depth.max() == 0 and depth.min() == 0:
                    print("   WARNING: Metric depth returned all zeros. Using relative fallback for this image.")
                    invalid_metric = True
                elif depth_range < 1e-6:
                    print(f"   WARNING: Metric depth has no variance (range={depth_range:.2e}). Using relative fallback.")
                    invalid_metric = True
                elif np.isnan(depth).any() or np.isinf(depth).any():
                    print("   WARNING: Metric depth contains NaN/Inf. Using relative fallback for this image.")
                    invalid_metric = True
                else:
                    # Sanity check: food is never more than 3 m from the camera.
                    # If the 5th percentile already exceeds that, the model is
                    # returning room-scale values for a close-up shot.
                    _p5 = float(np.percentile(depth, 5))
                    if _p5 > 3.0:
                        print(f"   WARNING: Metric depth suspiciously large (p5={_p5:.1f} m). Using relative fallback.")
                        invalid_metric = True

                if invalid_metric:
                    self._last_metric_failed = True
                    is_metric_output = False
                else:
                    depth = np.clip(depth, 0, self.max_depth)

            # Relative output (native relative mode or per-image metric failure)
            if not is_metric_output:
                depth_min = float(depth.min())
                depth_max = float(depth.max())
                if depth_max - depth_min < 1e-8:
                    print("   WARNING: Depth map has no variance.")
                    depth = np.zeros_like(depth)
                else:
                    depth = (depth - depth_min) / (depth_max - depth_min + 1e-8)

            # Cache result
            self._cached_depth = depth.copy()
            self._cached_image_hash = image_hash
            self._last_output_is_metric = is_metric_output

        # Optional bilateral filtering (only for relative output)
        if apply_filter and not is_metric_output:
            depth = self._bilateral_filter_depth(depth)

        depth_colored = self.colorize_depth(depth, is_metric=is_metric_output)
        return depth, depth_colored

    def estimate_depth_metric(self,
                              image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get depth in centimeters when metric output is valid.

        If a single image fails metric validation, this returns relative depth
        (0-1) in depth_cm to force safe fallback behavior downstream.
        """
        if not self.use_metric_original:
            raise RuntimeError("Metric depth requires use_metric=True in constructor")

        depth_m, depth_colored = self.estimate_depth(image, apply_filter=False)

        if self._last_metric_failed:
            print("   WARNING: Metric depth failed for this image; using relative fallback output.")
            depth_cm = depth_m
        else:
            depth_cm = depth_m * 100.0

        return depth_cm, depth_colored

    def get_height_from_depth(self,
                              depth_map: np.ndarray,
                              mask: np.ndarray,
                              base_percentile: float = 95) -> np.ndarray:
        """
        Convert depth map to height map for a food item

        The food sits ON a surface (plate/table). The surface is the "base"
        and food height = base_depth - food_depth (food is closer to camera)

        Args:
            depth_map: Depth in cm (from estimate_depth_metric) or relative (0-1)
            mask: Binary mask of food item
            base_percentile: Percentile to use as base surface (default 95 = deepest points)

        Returns:
            height_map: Height above surface (in same units as depth_map)
        """
        masked_depth = depth_map[mask]

        if len(masked_depth) == 0:
            return np.zeros_like(depth_map)

        # The base (plate/table) is at the MAXIMUM depth (farthest from camera)
        # Food surface is at MINIMUM depth (closest to camera)
        base_depth = np.percentile(masked_depth, base_percentile)

        # Height = base - depth (positive where food rises above base)
        height_map = np.zeros_like(depth_map)
        height_map[mask] = np.maximum(base_depth - depth_map[mask], 0)

        return height_map

    def _bilateral_filter_depth(self, depth: np.ndarray) -> np.ndarray:
        """Apply bilateral filter to smooth depth while preserving edges.

        Operates directly on float32 to avoid the precision loss caused by
        quantizing to uint8 (only 256 discrete levels).  For a normalized
        0-1 depth map sigmaColor=0.05 means "blend pixels whose depth values
        differ by less than ~5% of the full range".
        """
        depth_f32 = depth.astype(np.float32)
        filtered = cv2.bilateralFilter(depth_f32, d=9, sigmaColor=0.05, sigmaSpace=75)
        return filtered

    def colorize_depth(self, depth: np.ndarray, is_metric: bool = False) -> np.ndarray:
        """Create colored visualization of depth map."""
        if is_metric:
            depth_normalized = np.clip(depth / self.max_depth, 0, 1)
            depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        else:
            depth_uint8 = (depth * 255).astype(np.uint8)

        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
        return depth_colored

    def estimate_depth_for_mask(self,
                                image: np.ndarray,
                                mask: np.ndarray,
                                erode_edges: bool = True,
                                erode_pixels: int = 3) -> Dict:
        """Get depth statistics for masked region"""
        depth_map, _ = self.estimate_depth(image)

        if erode_edges and erode_pixels > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (erode_pixels * 2 + 1, erode_pixels * 2 + 1)
            )
            mask_uint8 = mask.astype(np.uint8) * 255 if mask.dtype == bool else mask
            eroded_mask = cv2.erode(mask_uint8, kernel, iterations=1) > 127
            sample_mask = eroded_mask
        else:
            sample_mask = mask

        masked_depth = depth_map[sample_mask]

        if len(masked_depth) > 0:
            mean_depth = float(np.mean(masked_depth))
            median_depth = float(np.median(masked_depth))
            std_depth = float(np.std(masked_depth))
            min_depth = float(np.percentile(masked_depth, 2))
            max_depth = float(np.percentile(masked_depth, 98))
            depth_range = max_depth - min_depth
        else:
            mean_depth = median_depth = std_depth = min_depth = max_depth = depth_range = 0.0

        return {
            'mean_depth': mean_depth,
            'median_depth': median_depth,
            'std_depth': std_depth,
            'min_depth': min_depth,
            'max_depth': max_depth,
            'depth_range': depth_range,
            'depth_map': depth_map,
            'is_metric': self._last_output_is_metric
        }

    def get_relative_depth(self, depth_map: np.ndarray, bbox: list) -> Dict:
        """Get depth statistics for a bounding box region"""
        x1, y1, x2, y2 = bbox
        region_depth = depth_map[y1:y2, x1:x2]

        if region_depth.size == 0:
            return {'mean': 0, 'median': 0, 'min': 0, 'max': 0, 'std': 0}

        return {
            'mean': float(np.mean(region_depth)),
            'median': float(np.median(region_depth)),
            'min': float(np.min(region_depth)),
            'max': float(np.max(region_depth)),
            'std': float(np.std(region_depth))
        }

    def create_depth_overlay(self,
                             image: np.ndarray,
                             depth_colored: np.ndarray,
                             alpha: float = 0.5) -> np.ndarray:
        """Create an overlay of depth visualization on original image"""
        if depth_colored.shape[:2] != image.shape[:2]:
            depth_colored = cv2.resize(depth_colored, (image.shape[1], image.shape[0]))
        return cv2.addWeighted(image, 1 - alpha, depth_colored, alpha, 0)


if __name__ == "__main__":
    import sys

    print("Testing DepthEstimator (FIXED VERSION)...")
    print("="*50)

    # Check for metric model
    metric_checkpoint = Path("weights/depth_anything_v2_metric_hypersim_vitl.pth")
    alt_metric = Path("Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth")

    if metric_checkpoint.exists() or alt_metric.exists():
        print("Metric model found - testing METRIC depth with AUTO-FALLBACK")
        estimator = DepthEstimator(
            model_size="large",
            use_metric=True,
            metric_dataset="hypersim",
            max_depth=5.0  # Better for food photos
        )
    else:
        print("Metric model not found - testing RELATIVE depth")
        print(f"To use metric depth, download from:")
        print("https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large")
        estimator = DepthEstimator(
            model_size="large",
            checkpoint_path="weights/depth_anything_v2_vitl.pth",
            use_metric=False
        )

    estimator.load_model()

    # Test on image
    img = cv2.imread("samples/food.jpg")
    
    if img is not None:
        print("\nTesting depth estimation...")
        
        if estimator.use_metric_original:
            depth_cm, depth_colored = estimator.estimate_depth_metric(img)
            
            if estimator._last_metric_failed:
                print(f"\n✅ Fallback successful - Relative depth range: {depth_cm.min():.3f} - {depth_cm.max():.3f}")
            else:
                print(f"\n✅ Metric depth range: {depth_cm.min():.1f} - {depth_cm.max():.1f} cm")
        else:
            depth, depth_colored = estimator.estimate_depth(img)
            print(f"\n✅ Relative depth range: {depth.min():.3f} - {depth.max():.3f}")

        cv2.imwrite("depth_test_fixed.jpg", depth_colored)
        print("Saved depth_test_fixed.jpg")
    else:
        print("Could not load food.jpg")
