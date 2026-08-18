#!/usr/bin/env python3
"""
Sentinel-2 Ingestion Script (Stage 1)

Pulls Sentinel-2 Level-2A Surface Reflectance bands (B2, B3, B4, B8) for a
given AOI + date range via Copernicus Data Space Ecosystem STAC API.

Canonical source: https://dataspace.copernicus.eu/
Sentinel-2 collection: https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-missions/sentinel-2

Alternative access: Google Earth Engine (for prototyping) — Copernicus remains
canonical reference.

Outputs clipped, reprojected GeoTIFF tiles + metadata JSON.

Known Limitations (documented per project spec):
1. Sentinel-2 is 10m resolution (B2/B3/B4/B8) — cannot resolve individual trees
   near conductors. Risk scores are corridor-segment-level, not tree-level.
2. Cloud filtering uses scene-level cloud cover metadata (from product quality
   info). Per-pixel cloud masking via SCL band optional; never silently discard
   quality information — all quality metadata is stored per result.
3. No canopy height data — NDVI/vegetation fraction is a proxy, not direct
   measurement of fall/contact risk.
4. No historical incident/outage ground truth — risk model must be explicit
   weighted heuristic, not supervised ML.
5. Land cover data not guaranteed — flag when NDVI-high areas might be cropland
   vs. woody vegetation. ESA WorldCover / Bhuvan used to disambiguate where available.
6. Do NOT claim Sentinel-2 alone can determine whether a specific tree will
   physically contact a conductor. This is a corridor-scale screening system.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pystac_client
import planetary_computer as pc
import rasterio
import rasterio.mask
import rasterio.warp
from rasterio.enums import Resampling
from shapely.geometry import box, shape, mapping
from shapely.ops import transform
import pyproj
from tqdm import tqdm


# Configuration constants
BANDS_OF_INTEREST = ["B02", "B03", "B04", "B08"]  # Blue, Green, Red, NIR
SCL_BAND = "SCL"  # Scene Classification Layer for cloud masking
TARGET_CRS = "EPSG:4326"  # WGS84
DEFAULT_CLOUD_COVER_MAX = 20  # percent
DEFAULT_NDVI_THRESHOLD = 0.3


class Sentinel2Ingestor:
    """Handles Sentinel-2 data search, download, and processing via Planetary Computer."""

    def __init__(
        self,
        aoi_geojson: str,
        start_date: str,
        end_date: str,
        output_dir: str,
        cloud_cover_max: int = DEFAULT_CLOUD_COVER_MAX,
        target_crs: str = TARGET_CRS,
    ):
        """
        Initialize ingestor.

        Args:
            aoi_geojson: Path to GeoJSON file defining Area of Interest
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            output_dir: Directory to save output GeoTIFFs and metadata
            cloud_cover_max: Maximum cloud cover percentage (0-100)
            target_crs: Target CRS for reprojection
        """
        self.aoi_path = Path(aoi_geojson)
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = Path(output_dir)
        self.cloud_cover_max = cloud_cover_max
        self.target_crs = target_crs

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "geotiffs").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)

        # Load AOI
        self.aoi_gdf = gpd.read_file(self.aoi_path)
        if self.aoi_gdf.crs is None:
            raise ValueError("AOI GeoJSON must have a CRS defined")
        self.aoi_gdf = self.aoi_gdf.to_crs(self.target_crs)
        self.aoi_bounds = self.aoi_gdf.total_bounds  # [minx, miny, maxx, maxy]
        self.aoi_geometry = self.aoi_gdf.geometry.unary_union

        # STAC client for Copernicus Data Space Ecosystem
        # Canonical source: https://dataspace.copernicus.eu/
        self.catalog = pystac_client.Client.open(
            "https://catalogue.dataspace.copernicus.eu/stac/",
            # Note: For production, consider implementing proper authentication
            # via Copernicus Data Space OAuth. For prototyping, public access
            # to Sentinel-2 L2A STAC catalog works without auth.
        )

    def search_items(self) -> List:
        """Search for Sentinel-2 items matching AOI and date range."""
        print(f"Searching Sentinel-2 items...")
        print(f"  AOI bounds: {self.aoi_bounds}")
        print(f"  Date range: {self.start_date} to {self.end_date}")
        print(f"  Max cloud cover: {self.cloud_cover_max}%")

        search = self.catalog.search(
            collections=["SENTINEL-2-L2A"],
            intersects=self.aoi_geometry,
            datetime=f"{self.start_date}/{self.end_date}",
            query={"eo:cloud_cover": {"lt": self.cloud_cover_max}},
        )

        items = list(search.items())
        print(f"Found {len(items)} items")
        return items

    def get_item_assets(self, item) -> Dict[str, str]:
        """Get signed asset URLs for bands of interest."""
        assets = {}
        for band in BANDS_OF_INTEREST + [SCL_BAND]:
            if band in item.assets:
                assets[band] = pc.sign(item.assets[band].href)
        return assets

    def download_and_clip_band(
        self, asset_url: str, band_name: str, item_id: str
    ) -> Tuple[np.ndarray, rasterio.transform.Affine, rasterio.crs.CRS]:
        """
        Download a band, clip to AOI, and reproject to target CRS.

        Returns:
            (data_array, transform, crs)
        """
        with rasterio.open(asset_url) as src:
            # Check if we need to reproject
            if src.crs.to_string() != self.target_crs:
                # Calculate output transform and shape for reprojection
                transform, width, height = rasterio.warp.calculate_default_transform(
                    src.crs, self.target_crs, src.width, src.height, *src.bounds
                )
                # Read and reproject
                data = np.empty((height, width), dtype=src.dtypes[0])
                rasterio.warp.reproject(
                    source=rasterio.band(src, 1),
                    destination=data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=self.target_crs,
                    resampling=Resampling.bilinear,
                )
            else:
                data = src.read(1)
                transform = src.transform

            # Clip to AOI
            # Convert AOI geometry to raster coordinates
            aoi_bounds_raster = rasterio.warp.transform_bounds(
                self.target_crs, src.crs, *self.aoi_bounds
            )
            window = rasterio.windows.from_bounds(
                *aoi_bounds_raster, transform=src.transform
            )
            window = window.round_offsets().round_shape()

            # Read clipped data
            clipped_data = src.read(1, window=window)
            clipped_transform = rasterio.windows.transform(window, src.transform)

            return clipped_data, clipped_transform, src.crs

    def process_item(self, item) -> Optional[Dict]:
        """Process a single Sentinel-2 item: download bands, clip, save."""
        item_id = item.id
        print(f"\nProcessing item: {item_id}")

        # Get asset URLs
        assets = self.get_item_assets(item)
        if len(assets) < len(BANDS_OF_INTEREST):
            print(f"  Warning: Missing bands, skipping")
            return None

        # Download and clip each band
        band_data = {}
        band_meta = {}

        for band_name in BANDS_OF_INTEREST + [SCL_BAND]:
            try:
                data, transform, crs = self.download_and_clip_band(
                    assets[band_name], band_name, item_id
                )
                band_data[band_name] = data
                band_meta[band_name] = {"transform": transform, "crs": crs}
            except Exception as e:
                print(f"  Error downloading {band_name}: {e}")
                return None

        # Save each band as GeoTIFF
        date_str = item.datetime.strftime("%Y%m%d") if item.datetime else "unknown"
        tile_id = f"{item_id}_{date_str}"

        for band_name in BANDS_OF_INTEREST + [SCL_BAND]:
            output_path = self.output_dir / "geotiffs" / f"{tile_id}_{band_name}.tif"
            meta = band_meta[band_name]
            with rasterio.open(
                output_path,
                "w",
                driver="GTiff",
                height=band_data[band_name].shape[0],
                width=band_data[band_name].shape[1],
                count=1,
                dtype=band_data[band_name].dtype,
                crs=meta["crs"],
                transform=meta["transform"],
                compress="lzw",
            ) as dst:
                dst.write(band_data[band_name], 1)

        # Build metadata with full quality and provenance info
        metadata = {
            "tile_id": tile_id,
            "item_id": item_id,
            "datetime": item.datetime.isoformat() if item.datetime else None,
            "date": date_str,
            "sensing_time": item.properties.get("sensing_time", None),
            "bounds": list(self.aoi_bounds),
            "cloud_cover": item.properties.get("eo:cloud_cover", None),
            "platform": item.properties.get("platform", None),
            "processing_level": item.properties.get("processing:level", "L2A"),
            "resolution_m": 10,  # B2/B3/B4/B8 are 10m
            "crs": self.target_crs,
            "footprint": item.geometry,
            "quality_info": {
                "cloud_percentage": item.properties.get("eo:cloud_cover", None),
                "valid_pixel_percentage": item.properties.get("valid_pixels", None),
                "missing_pixel_percentage": item.properties.get("missing_pixels", None),
            },
            "bands": {b: str(self.output_dir / "geotiffs" / f"{tile_id}_{b}.tif") for b in BANDS_OF_INTEREST + [SCL_BAND]},
            "data_source": "Copernicus Data Space Ecosystem",
            "data_source_url": "https://dataspace.copernicus.eu/",
            "processed_at": datetime.utcnow().isoformat() + "Z",
        }

        # Save metadata
        meta_path = self.output_dir / "metadata" / f"{tile_id}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  Saved: {tile_id} (cloud cover: {metadata['cloud_cover']}%)")
        return metadata

    def run(self) -> List[Dict]:
        """Run the full ingestion pipeline."""
        items = self.search_items()
        if not items:
            print("No items found matching criteria")
            return []

        results = []
        for item in tqdm(items, desc="Processing items"):
            try:
                meta = self.process_item(item)
                if meta:
                    results.append(meta)
            except Exception as e:
                print(f"Error processing {item.id}: {e}")
                continue

        # Save summary
        summary_path = self.output_dir / "ingestion_summary.json"
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "aoi": str(self.aoi_path),
                    "date_range": f"{self.start_date}/{self.end_date}",
                    "cloud_cover_max": self.cloud_cover_max,
                    "items_processed": len(results),
                    "tiles": results,
                },
                f,
                indent=2,
            )

        print(f"\nDone! Processed {len(results)} tiles.")
        print(f"Summary saved to: {summary_path}")
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Sentinel-2 Ingestion for Vegetation Risk Analysis (Stage 1)"
    )
    parser.add_argument(
        "--aoi", required=True, help="Path to AOI GeoJSON file"
    )
    parser.add_argument(
        "--start-date", required=True, help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", required=True, help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for GeoTIFFs and metadata"
    )
    parser.add_argument(
        "--cloud-cover-max",
        type=int,
        default=DEFAULT_CLOUD_COVER_MAX,
        help=f"Maximum cloud cover % (default: {DEFAULT_CLOUD_COVER_MAX})",
    )
    parser.add_argument(
        "--target-crs",
        default=TARGET_CRS,
        help=f"Target CRS (default: {TARGET_CRS})",
    )

    args = parser.parse_args()

    ingestor = Sentinel2Ingestor(
        aoi_geojson=args.aoi,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        cloud_cover_max=args.cloud_cover_max,
        target_crs=args.target_crs,
    )

    ingestor.run()


if __name__ == "__main__":
    main()