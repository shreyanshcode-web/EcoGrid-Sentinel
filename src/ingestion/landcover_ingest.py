#!/usr/bin/env python3
"""
ESA WorldCover Land Cover Ingestion

Official source: https://esa-worldcover.org/en/data-access
Provider: ESA (European Space Agency)

ESA WorldCover provides 10 m global land cover products for 2020 and 2021.
Relevant classes for transmission corridor vegetation risk:
- 10: Tree cover
- 20: Shrubland
- 30: Grassland
- 40: Cropland
- 50: Built-up
- 60: Bare / sparse vegetation
- 70: Snow and ice
- 80: Permanent water bodies
- 90: Herbaceous wetland
- 95: Mangroves
- 100: Moss and lichen

IMPORTANT: Do not treat every high-NDVI pixel as tree.
Use WorldCover to distinguish woody vegetation from crops.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
import rasterio.warp
from rasterio.enums import Resampling
from shapely.geometry import box, shape
import geopandas as gpd
from tqdm import tqdm


# ESA WorldCover class mapping
WORLDCOVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}

# Classes considered "woody vegetation" for transmission corridor risk
WOODY_VEGETATION_CLASSES = {10, 20, 95}  # Tree cover, Shrubland, Mangroves

# Classes that are vegetation but NOT woody (high NDVI but not tree risk)
NON_WOODY_VEGETATION_CLASSES = {30, 40, 90}  # Grassland, Cropland, Herbaceous wetland


class WorldCoverIngestor:
    """Download and process ESA WorldCover land cover data for AOI."""

    def __init__(
        self,
        aoi_geojson: str,
        output_dir: str,
        year: int = 2021,
        target_crs: str = "EPSG:4326",
    ):
        """
        Initialize WorldCover ingestor.

        Args:
            aoi_geojson: Path to AOI GeoJSON file
            output_dir: Output directory
            year: WorldCover year (2020 or 2021)
            target_crs: Target CRS (default WGS84)
        """
        self.aoi_path = Path(aoi_geojson)
        self.output_dir = Path(output_dir)
        self.year = year
        self.target_crs = target_crs

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "worldcover").mkdir(exist_ok=True)

        # Load AOI
        self.aoi_gdf = gpd.read_file(self.aoi_path)
        if self.aoi_gdf.crs is None:
            raise ValueError("AOI GeoJSON must have a CRS defined")
        self.aoi_gdf = self.aoi_gdf.to_crs(self.target_crs)
        self.aoi_bounds = self.aoi_gdf.total_bounds
        self.aoi_geometry = self.aoi_gdf.geometry.unary_union

        # WorldCover STAC catalog (via Microsoft Planetary Computer - mirrors ESA data)
        # Canonical ESA WorldCover access: https://esa-worldcover.org/en/data-access
        # For programmatic access, we can use the STAC catalog
        self.stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1"

    def get_worldcover_asset_urls(self) -> List[str]:
        """Get WorldCover asset URLs for AOI tiles."""
        import pystac_client
        import planetary_computer as pc

        catalog = pystac_client.Client.open(
            self.stac_url,
            modifier=pc.sign_inplace,
        )

        search = catalog.search(
            collections=["esa-worldcover"],
            intersects=self.aoi_geometry,
        )

        items = list(search.items())
        return [pc.sign(item.assets["map"].href) for item in items]

    def download_and_clip(self, asset_url: str, output_path: Path) -> Dict:
        """Download WorldCover tile and clip to AOI."""
        with rasterio.open(asset_url) as src:
            # Calculate output transform and shape for reprojection
            if src.crs.to_string() != self.target_crs:
                transform, width, height = rasterio.warp.calculate_default_transform(
                    src.crs, self.target_crs, src.width, src.height, *src.bounds
                )
            else:
                transform = src.transform
                width, height = src.width, src.height

            # Read and reproject/clip to AOI
            aoi_bounds_src = rasterio.warp.transform_bounds(
                self.target_crs, src.crs, *self.aoi_bounds
            )
            window = rasterio.windows.from_bounds(
                *aoi_bounds_src, transform=src.transform
            )
            window = window.round_offsets().round_shape()

            data = src.read(1, window=window)
            clipped_transform = rasterio.windows.transform(window, src.transform)

            # Reproject if needed
            if src.crs.to_string() != self.target_crs:
                out_data = np.empty((height, width), dtype=src.dtypes[0])
                rasterio.warp.reproject(
                    source=data,
                    destination=out_data,
                    src_transform=clipped_transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=self.target_crs,
                    resampling=Resampling.nearest,
                )
                data = out_data[:data.shape[0], :data.shape[1]]
                final_transform = transform
            else:
                final_transform = clipped_transform

            # Save clipped WorldCover
            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                height=data.shape[0],
                width=data.shape[1],
                count=1,
                dtype=data.dtype,
                crs=self.target_crs,
                transform=final_transform,
                compress="lzw",
            )

            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(data, 1)

            return {
                "path": str(output_path),
                "shape": data.shape,
                "crs": self.target_crs,
                "transform": list(final_transform),
                "unique_classes": np.unique(data).tolist(),
                "class_counts": {int(k): int(v) for k, v in zip(*np.unique(data, return_counts=True))},
            }

    def compute_woody_vegetation_mask(self, worldcover_path: Path) -> Tuple[np.ndarray, Dict]:
        """Create binary mask of woody vegetation from WorldCover."""
        with rasterio.open(worldcover_path) as src:
            data = src.read(1)
            transform = src.transform
            crs = src.crs

        # Create woody vegetation mask
        woody_mask = np.isin(data, list(WOODY_VEGETATION_CLASSES)).astype(np.uint8)
        non_woody_veg_mask = np.isin(data, list(NON_WOODY_VEGETATION_CLASSES)).astype(np.uint8)

        stats = {
            "woody_pixels": int(np.sum(woody_mask)),
            "non_woody_veg_pixels": int(np.sum(non_woody_veg_mask)),
            "total_pixels": int(data.size),
            "woody_fraction": float(np.mean(woody_mask)),
            "non_woody_veg_fraction": float(np.mean(non_woody_veg_mask)),
            "classes_present": {int(k): WORLDCOVER_CLASSES.get(int(k), "Unknown") for k in np.unique(data)},
        }

        return woody_mask, stats

    def save_woody_mask(self, woody_mask: np.ndarray, reference_path: Path, output_path: Path):
        """Save woody vegetation mask as GeoTIFF."""
        with rasterio.open(reference_path) as src:
            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                dtype=woody_mask.dtype,
                count=1,
                compress="lzw",
            )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(woody_mask, 1)

    def run(self) -> Dict:
        """Run WorldCover ingestion for AOI."""
        print(f"Fetching ESA WorldCover {self.year} for AOI...")
        print(f"  AOI bounds: {self.aoi_bounds}")

        asset_urls = self.get_worldcover_asset_urls()
        print(f"Found {len(asset_urls)} WorldCover tiles")

        if not asset_urls:
            print("  No WorldCover tiles found for AOI")
            return {"tiles": [], "woody_mask": None}

        results = []
        woody_masks = []

        for i, url in enumerate(tqdm(asset_urls, desc="Processing WorldCover tiles")):
            tile_path = self.output_dir / "worldcover" / f"worldcover_{self.year}_tile_{i}.tif"
            meta = self.download_and_clip(url, tile_path)
            results.append(meta)

            # Compute woody vegetation mask
            woody_mask, stats = self.compute_woody_vegetation_mask(tile_path)
            woody_masks.append(woody_mask)

            # Save woody mask
            woody_path = self.output_dir / "worldcover" / f"woody_vegetation_tile_{i}.tif"
            self.save_woody_mask(woody_mask, tile_path, woody_path)

            print(f"  Tile {i}: {stats['woody_fraction']*100:.1f}% woody vegetation")

        # Combine masks if multiple tiles
        if len(woody_masks) > 1:
            combined = np.maximum.reduce(woody_masks)
            combined_path = self.output_dir / "worldcover" / "woody_vegetation_combined.tif"
            self.save_woody_mask(combined, Path(results[0]["path"]), combined_path)
            combined_stats = {
                "woody_pixels": int(np.sum(combined)),
                "total_pixels": int(combined.size),
                "woody_fraction": float(np.mean(combined)),
            }
        else:
            combined = woody_masks[0] if woody_masks else None
            combined_stats = stats if woody_masks else {}

        # Save summary
        summary = {
            "year": self.year,
            "aoi": str(self.aoi_path),
            "tiles_processed": len(results),
            "tile_details": results,
            "combined_woody_stats": combined_stats,
            "data_source": "ESA WorldCover",
            "data_source_url": "https://esa-worldcover.org/en/data-access",
            "classes": WORLDCOVER_CLASSES,
            "woody_classes": list(WOODY_VEGETATION_CLASSES),
            "non_woody_veg_classes": list(NON_WOODY_VEGETATION_CLASSES),
        }

        summary_path = self.output_dir / "worldcover" / "worldcover_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nWorldCover processing complete.")
        print(f"  Combined woody vegetation fraction: {combined_stats.get('woody_fraction', 0)*100:.1f}%")
        print(f"  Summary saved to: {summary_path}")

        return summary


def main():
    parser = argparse.ArgumentParser(
        description="ESA WorldCover Land Cover Ingestion"
    )
    parser.add_argument("--aoi", required=True, help="Path to AOI GeoJSON file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--year", type=int, default=2021, choices=[2020, 2021], help="WorldCover year (default: 2021)")
    parser.add_argument("--target-crs", default="EPSG:4326", help="Target CRS (default: EPSG:4326)")

    args = parser.parse_args()

    ingestor = WorldCoverIngestor(
        aoi_geojson=args.aoi,
        output_dir=args.output_dir,
        year=args.year,
        target_crs=args.target_crs,
    )

    ingestor.run()


if __name__ == "__main__":
    main()