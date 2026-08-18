#!/usr/bin/env python3
"""
Temporal Analysis Script (Stage 4)

Computes NDVI change between two dates for the same AOI and calculates
simple linear growth rate (% change / days).

Known Limitations:
1. Rolling stats and seasonal trend modeling are deferred to v2.
2. Only two-date comparison is implemented (no time series).
3. Assumes same spatial extent and resolution for both dates.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import rasterio
import rasterio.warp
from scipy import ndimage
import matplotlib.pyplot as plt


class TemporalAnalyzer:
    """Analyze NDVI changes between two time periods."""

    def __init__(self):
        pass

    def load_ndvi(self, ndvi_path: str) -> Tuple[np.ndarray, rasterio.transform.Affine]:
        """Load NDVI raster and return array + transform."""
        with rasterio.open(ndvi_path) as src:
            ndvi = src.read(1)
            transform = src.transform
        return ndvi, transform

    def align_rasters(
        self, ndvi1: np.ndarray, ndvi2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Align two rasters to same shape by padding/cropping."""
        # Get minimum dimensions
        min_h = min(ndvi1.shape[0], ndvi2.shape[0])
        min_w = min(ndvi1.shape[1], ndvi2.shape[1])

        # Crop both to same size
        ndvi1_aligned = ndvi1[:min_h, :min_w]
        ndvi2_aligned = ndvi2[:min_h, :min_w]

        return ndvi1_aligned, ndvi2_aligned

    def compute_ndvi_change(
        self, ndvi_early: np.ndarray, ndvi_late: np.ndarray
    ) -> Dict:
        """
        Compute NDVI change between two dates.

        Returns:
            Dictionary with change statistics and arrays
        """
        # Align rasters
        ndvi1, ndvi2 = self.align_rasters(ndvi_early, ndvi_late)

        # Absolute change
        change = ndvi2 - ndvi1

        # Relative change (percent)
        with np.errstate(divide="ignore", invalid="ignore"):
            relative_change = np.where(
                ndvi1 != 0, ((ndvi2 - ndvi1) / ndvi1) * 100, 0
            )

        # Binary change (significant increase/decrease)
        threshold = 0.1  # 10% change threshold
        significant_increase = change > threshold
        significant_decrease = change < -threshold
        stable = ~significant_increase & ~significant_decrease

        stats = {
            "mean_change": float(np.mean(change)),
            "std_change": float(np.std(change)),
            "max_increase": float(np.max(change)),
            "max_decrease": float(np.min(change)),
            "pct_increased": float(np.mean(change > 0) * 100),
            "pct_decreased": float(np.mean(change < 0) * 100),
            "pct_stable": float(np.mean(stable) * 100),
            "pct_significant_increase": float(np.mean(significant_increase) * 100),
            "pct_significant_decrease": float(np.mean(significant_decrease) * 100),
        }

        return {
            "change": change,
            "relative_change": relative_change,
            "significant_increase": significant_increase,
            "significant_decrease": significant_decrease,
            "stats": stats,
        }

    def compute_growth_rate(
        self, ndvi_early: np.ndarray, ndvi_late: np.ndarray, days_between: int
    ) -> Dict:
        """
        Compute simple linear growth rate (% change / days).

        Args:
            ndvi_early: NDVI array from earlier date
            ndvi_late: NDVI array from later date
            days_between: Number of days between observations

        Returns:
            Dictionary with growth rate statistics and array
        """
        if days_between <= 0:
            raise ValueError("days_between must be positive")

        # Align rasters
        ndvi1, ndvi2 = self.align_rasters(ndvi_early, ndvi_late)

        # Compute absolute change
        absolute_change = ndvi2 - ndvi1

        # Growth rate: % change per day
        with np.errstate(divide="ignore", invalid="ignore"):
            growth_rate = np.where(
                ndvi1 != 0,
                ((ndvi2 - ndvi1) / ndvi1) / days_between * 100,
                0,
            )

        # Classify growth rates
        fast_growth = growth_rate > 0.1  # > 0.1% per day
        slow_growth = (growth_rate > 0.01) & (growth_rate <= 0.1)
        stable = (growth_rate >= -0.01) & (growth_rate <= 0.01)
        declining = growth_rate < -0.01

        stats = {
            "mean_growth_rate_pct_per_day": float(np.mean(growth_rate)),
            "std_growth_rate": float(np.std(growth_rate)),
            "max_growth_rate": float(np.max(growth_rate)),
            "min_growth_rate": float(np.min(growth_rate)),
            "pct_fast_growth": float(np.mean(fast_growth) * 100),
            "pct_slow_growth": float(np.mean(slow_growth) * 100),
            "pct_stable": float(np.mean(stable) * 100),
            "pct_declining": float(np.mean(declining) * 100),
            "days_between": days_between,
        }

        return {
            "growth_rate": growth_rate,
            "absolute_change": absolute_change,
            "stats": stats,
        }

    def create_change_visualization(
        self,
        ndvi_early: np.ndarray,
        ndvi_late: np.ndarray,
        change_result: Dict,
        growth_result: Dict,
        output_path: str,
        tile_id: str = "",
    ):
        """Create visualization of temporal changes."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Row 1: NDVI images
        im1 = axes[0, 0].imshow(ndvi_early, cmap="RdYlGn", vmin=-0.2, vmax=1.0)
        axes[0, 0].set_title("NDVI - Early Date")
        plt.colorbar(im1, ax=axes[0, 0])

        im2 = axes[0, 1].imshow(ndvi_late, cmap="RdYlGn", vmin=-0.2, vmax=1.0)
        axes[0, 1].set_title("NDVI - Late Date")
        plt.colorbar(im2, ax=axes[0, 1])

        im3 = axes[0, 2].imshow(
            change_result["change"], cmap="RdYlGn", vmin=-0.5, vmax=0.5
        )
        axes[0, 2].set_title("NDVI Change")
        plt.colorbar(im3, ax=axes[0, 2])

        # Row 2: Growth and classification
        im4 = axes[1, 0].imshow(
            growth_result["growth_rate"], cmap="RdYlGn", vmin=-0.5, vmax=0.5
        )
        axes[1, 0].set_title("Growth Rate (%/day)")
        plt.colorbar(im4, ax=axes[1, 0])

        # Change classification
        change_class = np.zeros_like(change_result["change"])
        change_class[change_result["significant_increase"]] = 2
        change_class[change_result["significant_decrease"]] = -2
        change_class[~change_result["significant_increase"] & ~change_result["significant_decrease"]] = 0

        im5 = axes[1, 1].imshow(change_class, cmap="RdYlGn", vmin=-2, vmax=2)
        axes[1, 1].set_title("Change Classification")

        # Histogram of change values
        change_flat = change_result["change"].flatten()
        change_flat = change_flat[change_flat != 0]
        if len(change_flat) > 0:
            axes[1, 2].hist(change_flat, bins=50, color="steelblue", alpha=0.7)
            axes[1, 2].axvline(x=0, color="red", linestyle="--", label="No Change")
            axes[1, 2].set_xlabel("NDVI Change")
            axes[1, 2].set_ylabel("Pixel Count")
            axes[1, 2].set_title("Distribution of NDVI Changes")
            axes[1, 2].legend()

        plt.suptitle(f"Temporal Analysis - {tile_id}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    def analyze(
        self,
        ndvi_early_path: str,
        ndvi_late_path: str,
        days_between: int,
        output_dir: str,
        tile_id: str = "",
    ) -> Dict:
        """Run full temporal analysis."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loading NDVI rasters...")
        ndvi_early, _ = self.load_ndvi(ndvi_early_path)
        ndvi_late, _ = self.load_ndvi(ndvi_late_path)

        print(f"Computing NDVI change ({days_between} days)...")
        change_result = self.compute_ndvi_change(ndvi_early, ndvi_late)

        print(f"Computing growth rate...")
        growth_result = self.compute_growth_rate(ndvi_early, ndvi_late, days_between)

        # Create visualization
        viz_path = output_dir / f"{tile_id}_temporal_analysis.png"
        self.create_change_visualization(
            ndvi_early, ndvi_late, change_result, growth_result, str(viz_path), tile_id
        )

        # Combine stats
        combined_stats = {
            "tile_id": tile_id,
            "change_stats": change_result["stats"],
            "growth_stats": growth_result["stats"],
            "visualization": str(viz_path),
        }

        # Save stats
        stats_path = output_dir / f"{tile_id}_temporal_stats.json"
        with open(stats_path, "w") as f:
            json.dump(combined_stats, f, indent=2)

        print(f"\nTemporal Analysis Complete:")
        print(f"  Mean change: {change_result['stats']['mean_change']:.4f}")
        print(f"  Mean growth rate: {growth_result['stats']['mean_growth_rate_pct_per_day']:.4f}%/day")
        print(f"  Saved to: {output_dir}")

        return combined_stats


def main():
    parser = argparse.ArgumentParser(
        description="Temporal Analysis (Stage 4): NDVI change and growth rate computation"
    )
    parser.add_argument(
        "--ndvi-early", required=True, help="Path to NDVI raster (earlier date)"
    )
    parser.add_argument(
        "--ndvi-late", required=True, help="Path to NDVI raster (later date)"
    )
    parser.add_argument(
        "--days-between", type=int, required=True, help="Number of days between observations"
    )
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--tile-id", default="", help="Tile ID for labeling")

    args = parser.parse_args()

    analyzer = TemporalAnalyzer()
    analyzer.analyze(
        ndvi_early_path=args.ndvi_early,
        ndvi_late_path=args.ndvi_late,
        days_between=args.days_between,
        output_dir=args.output_dir,
        tile_id=args.tile_id,
    )


if __name__ == "__main__":
    main()