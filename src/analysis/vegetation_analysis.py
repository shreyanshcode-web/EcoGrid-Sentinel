#!/usr/bin/env python3
"""
Vegetation Analysis Script (Stage 2)

Computes NDVI = (B8 - B4) / (B8 + B4) from Sentinel-2 bands,
creates vegetation mask via configurable threshold (default 0.3),
applies morphological operations for denoising.

Known Limitations:
1. Sentinel-2 is 10m resolution — cannot resolve individual trees near conductors.
   Risk scores are corridor-segment-level, not tree-level.
2. No canopy height data — NDVI/vegetation fraction is a proxy, not direct
   measurement of fall/contact risk.
3. U-Net segmentation deferred to v2 — this uses threshold-based masking only.
   A stub function is provided for future U-Net upgrade.
4. Land cover data not guaranteed — flag when NDVI-high areas might be cropland.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
import rasterio.warp
from scipy import ndimage
from skimage.morphology import disk, opening, closing
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# Default configuration constants
DEFAULT_NDVI_THRESHOLD_LOW = 0.3    # Below this: bare ground/water
DEFAULT_NDVI_THRESHOLD_HIGH = 0.7   # Above this: dense vegetation
DEFAULT_OPENING_RADIUS = 2          # Morphological opening kernel radius
DEFAULT_CLOSING_RADIUS = 3          # Morphological closing kernel radius
MIN_VEGETATION_PATCH_PIXELS = 10    # Minimum patch size (in pixels) to keep


class VegetationAnalyzer:
    """Compute NDVI and create vegetation masks from Sentinel-2 bands."""

    def __init__(
        self,
        ndvi_threshold_low: float = DEFAULT_NDVI_THRESHOLD_LOW,
        ndvi_threshold_high: float = DEFAULT_NDVI_THRESHOLD_HIGH,
        opening_radius: int = DEFAULT_OPENING_RADIUS,
        closing_radius: int = DEFAULT_CLOSING_RADIUS,
        min_patch_pixels: int = MIN_VEGETATION_PATCH_PIXELS,
    ):
        self.ndvi_threshold_low = ndvi_threshold_low
        self.ndvi_threshold_high = ndvi_threshold_high
        self.opening_radius = opening_radius
        self.closing_radius = closing_radius
        self.min_patch_pixels = min_patch_pixels

    def compute_ndvi(self, nir_path: str, red_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Compute NDVI from NIR (B8) and Red (B4) bands.

        NDVI = (B8 - B4) / (B8 + B4)

        Returns:
            (ndvi_array, metadata_dict)
        """
        with rasterio.open(nir_path) as nir_src, rasterio.open(red_path) as red_src:
            nir = nir_src.read(1).astype(np.float32)
            red = red_src.read(1).astype(np.float32)
            transform = nir_src.transform
            crs = nir_src.crs
            profile = nir_src.profile.copy()

        # Compute NDVI with safe division
        denominator = nir + red
        denominator[denominator == 0] = np.nan  # Avoid division by zero
        ndvi = (nir - red) / denominator

        # Handle NaN (no data)
        ndvi = np.nan_to_num(ndvi, nan=0.0)
        ndvi = np.clip(ndvi, -1.0, 1.0)  # NDVI range is [-1, 1]

        meta = {
            "crs": str(crs),
            "transform": list(transform),
            "shape": ndvi.shape,
            "ndvi_min": float(np.min(ndvi)),
            "ndvi_max": float(np.max(ndvi)),
            "ndvi_mean": float(np.mean(ndvi[ndvi != 0])),
            "threshold_low": self.ndvi_threshold_low,
            "threshold_high": self.ndvi_threshold_high,
        }

        return ndvi, meta

    def create_vegetation_mask(
        self, ndvi: np.ndarray, apply_morphology: bool = True
    ) -> np.ndarray:
        """
        Create binary vegetation mask from NDVI using threshold.

        Args:
            ndvi: NDVI array (float32, range [-1, 1])
            apply_morphology: Whether to apply morphological opening/closing

        Returns:
            Binary mask: 1 = vegetation, 0 = non-vegetation
        """
        # Threshold-based mask (simple for v1; U-Net deferred to v2)
        mask = (ndvi >= self.ndvi_threshold_low).astype(np.uint8)

        if apply_morphology and np.sum(mask) > 0:
            # Morphological opening: remove small isolated vegetation patches
            selem_open = disk(self.opening_radius)
            mask = opening(mask, selem_open).astype(np.uint8)

            # Morphological closing: fill small gaps within vegetation
            selem_close = disk(self.closing_radius)
            mask = closing(mask, selem_close).astype(np.uint8)

        return mask

    def remove_small_patches(
        self, mask: np.ndarray, min_pixels: Optional[int] = None
    ) -> np.ndarray:
        """Remove vegetation patches smaller than min_pixels."""
        if min_pixels is None:
            min_pixels = self.min_patch_pixels

        # Label connected components
        labeled, num_features = ndimage.label(mask)
        print(f"  Found {num_features} vegetation patches")

        # Remove small patches
        cleaned_mask = np.zeros_like(mask)
        kept_patches = 0
        for i in range(1, num_features + 1):
            patch = labeled == i
            if np.sum(patch) >= min_pixels:
                cleaned_mask[patch] = 1
                kept_patches += 1

        print(f"  Kept {kept_patches} patches (min size: {min_pixels} pixels)")
        return cleaned_mask

    def compute_vegetation_fraction(
        self, mask: np.ndarray, window_size: int = 100
    ) -> np.ndarray:
        """
        Compute vegetation fraction using a sliding window.
        Useful for per-corridor-segment feature engineering.
        """
        # Use uniform filter for efficient sliding window average
        fraction = ndimage.uniform_filter(mask.astype(np.float32), size=window_size)
        return fraction

    def create_vegetation_class_map(self, ndvi: np.ndarray) -> np.ndarray:
        """
        Create multi-class vegetation map:
        0 = Bare ground / Water
        1 = Sparse vegetation (NDVI 0.3-0.5)
        2 = Moderate vegetation (NDVI 0.5-0.7)
        3 = Dense vegetation (NDVI > 0.7)

        Note: Without land cover data, we cannot distinguish cropland vs.
        woody vegetation. This is flagged as a known limitation.
        """
        class_map = np.zeros_like(ndvi, dtype=np.uint8)
        class_map[(ndvi >= 0.3) & (ndvi < 0.5)] = 1
        class_map[(ndvi >= 0.5) & (ndvi < 0.7)] = 2
        class_map[ndvi >= 0.7] = 3
        return class_map

    @staticmethod
    def stub_unet_segmentation():
        """
        STUB: U-Net segmentation for vegetation classification.
        Deferred to v2. Leave this as documented upgrade path.

        v2 TODOs:
        - Train on aerial imagery + vegetation labels
        - Fine-tune for transmission corridor vegetation
        - Output individual tree instance masks
        - Integrate with LiDAR canopy height if available
        """
        raise NotImplementedError(
            "U-Net segmentation is deferred to v2. "
            "See stub_unet_segmentation() for upgrade path documentation."
        )

    def analyze_tile(
        self, tile_id: str, bands_dir: str, output_dir: str
    ) -> Dict:
        """Run full vegetation analysis on a single tile."""
        bands_path = Path(bands_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Find NIR (B8) and Red (B4) bands
        nir_pattern = f"*{tile_id}*B08.tif"
        red_pattern = f"*{tile_id}*B04.tif"

        nir_files = list(bands_path.glob(nir_pattern))
        red_files = list(bands_path.glob(red_pattern))

        if not nir_files or not red_files:
            raise FileNotFoundError(f"Cannot find B08 or B04 for tile {tile_id}")

        nir_path = str(nir_files[0])
        red_path = str(red_files[0])

        # Compute NDVI
        print(f"Computing NDVI for {tile_id}...")
        ndvi, ndvi_meta = self.compute_ndvi(nir_path, red_path)

        # Create vegetation mask
        print(f"Creating vegetation mask...")
        veg_mask = self.create_vegetation_mask(ndvi)

        # Remove small patches
        veg_mask_clean = self.remove_small_patches(veg_mask)

        # Create class map
        class_map = self.create_vegetation_class_map(ndvi)

        # Save NDVI raster
        ndvi_path = out_path / f"{tile_id}_ndvi.tif"
        profile = rasterio.open(nir_path).profile.copy()
        profile.update(
            driver="GTiff",
            height=ndvi.shape[0],
            width=ndvi.shape[1],
            count=1,
            dtype=ndvi.dtype,
            compress="lzw",
        )
        with rasterio.open(ndvi_path, "w", **profile) as dst:
            dst.write(ndvi, 1)

        # Save vegetation mask
        mask_path = out_path / f"{tile_id}_vegmask.tif"
        mask_profile = profile.copy()
        mask_profile.update(dtype=veg_mask_clean.dtype)
        with rasterio.open(mask_path, "w", **mask_profile) as dst:
            dst.write(veg_mask_clean, 1)

        # Save class map
        class_path = out_path / f"{tile_id}_vegclasses.tif"
        class_profile = profile.copy()
        class_profile.update(dtype=class_map.dtype)
        with rasterio.open(class_path, "w", **class_profile) as dst:
            dst.write(class_map, 1)

        # Create NDVI visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # NDVI
        im1 = axes[0].imshow(
            ndvi, cmap="RdYlGn", vmin=-0.2, vmax=1.0, aspect="auto"
        )
        axes[0].set_title(f"NDVI - {tile_id}")
        plt.colorbar(im1, ax=axes[0])

        # Vegetation mask
        im2 = axes[1].imshow(veg_mask_clean, cmap="Greens", aspect="auto")
        axes[1].set_title("Vegetation Mask (Cleaned)")
        plt.colorbar(im2, ax=axes[1])

        # Class map
        im3 = axes[2].imshow(class_map, cmap="YlGn", vmin=0, vmax=3, aspect="auto")
        axes[2].set_title("Vegetation Classes")
        plt.colorbar(im3, ax=axes[2])

        plt.tight_layout()
        fig_path = out_path / f"{tile_id}_veg_analysis.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()

        # Compute statistics
        veg_stats = {
            "tile_id": tile_id,
            "ndvi_mean": float(np.mean(ndvi[ndvi != 0])),
            "ndvi_std": float(np.std(ndvi[ndvi != 0])),
            "vegetation_fraction": float(np.mean(veg_mask_clean)),
            "vegetation_pixels": int(np.sum(veg_mask_clean)),
            "total_pixels": int(veg_mask_clean.size),
            "sparse_veg_pixels": int(np.sum(class_map == 1)),
            "moderate_veg_pixels": int(np.sum(class_map == 2)),
            "dense_veg_pixels": int(np.sum(class_map == 3)),
            "threshold_low": self.ndvi_threshold_low,
            "threshold_high": self.ndvi_threshold_high,
            "ndvi_path": str(ndvi_path),
            "mask_path": str(mask_path),
            "class_path": str(class_path),
            "viz_path": str(fig_path),
        }

        # Save stats
        stats_path = out_path / f"{tile_id}_vegstats.json"
        with open(stats_path, "w") as f:
            json.dump(veg_stats, f, indent=2)

        return veg_stats


def main():
    parser = argparse.ArgumentParser(
        description="Vegetation Analysis (Stage 2): NDVI computation and vegetation masking"
    )
    parser.add_argument("--tile-id", required=True, help="Tile ID to process")
    parser.add_argument("--bands-dir", required=True, help="Directory containing band GeoTIFFs")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--ndvi-threshold",
        type=float,
        default=DEFAULT_NDVI_THRESHOLD_LOW,
        help=f"NDVI threshold for vegetation (default: {DEFAULT_NDVI_THRESHOLD_LOW})",
    )
    parser.add_argument(
        "--min-patch-size",
        type=int,
        default=MIN_VEGETATION_PATCH_PIXELS,
        help=f"Minimum vegetation patch size in pixels (default: {MIN_VEGETATION_PATCH_PIXELS})",
    )

    args = parser.parse_args()

    analyzer = VegetationAnalyzer(
        ndvi_threshold_low=args.ndvi_threshold,
        min_patch_pixels=args.min_patch_size,
    )

    stats = analyzer.analyze_tile(args.tile_id, args.bands_dir, args.output_dir)
    print(f"\nDone! Analysis for {args.tile_id}:")
    print(f"  Vegetation fraction: {stats['vegetation_fraction']:.3f}")
    print(f"  NDVI mean: {stats['ndvi_mean']:.3f}")
    print(f"  Dense vegetation pixels: {stats['dense_veg_pixels']}")


if __name__ == "__main__":
    main()