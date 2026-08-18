#!/usr/bin/env python3
"""
NASA GEDI Canopy Height Ingestion

Official source: https://www.earthdata.nasa.gov/data/catalog/lpcloud-gedi02-a-002
NASA Earthdata GEDI L2A provides elevation and height metrics including:
- canopy top height (canopy_top_height)
- relative-height metrics (RH metrics RH25, RH50, RH75, RH95, etc.)
- canopy cover

GEDI L2B provides canopy-cover and vertical-profile metrics.

Google Earth Engine access (rasterized):
https://developers.google.com/earth-engine/datasets/catalog/LARSE_GEDI_GEDI02_A_002_MONTHLY

IMPORTANT: GEDI has approximately 25 m footprint and is SPARSE. Missing GEDI
coverage must be represented explicitly. Do NOT treat GEDI as continuous
high-resolution canopy height everywhere. Do not invent canopy height where
GEDI data are unavailable.

DEM elevation is NOT canopy height — GEDI is the source for canopy height.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
import rasterio.warp
from rasterio.enums import Resampling
from shapely.geometry import box, shape
import geopandas as gpd
from tqdm import tqdm


# GEDI L2A relative height metrics (RH) and canopy metrics
GEDI_METRICS = {
    "canopy_top_height": "Direct canopy top height (m)",
    "rh_25": "Relative height 25th percentile (m)",
    "rh_50": "Relative height 50th percentile (m) - median canopy height",
    "rh_75": "Relative height 75th percentile (m)",
    "rh_95": "Relative height 95th percentile (m)",
    "canopy_cover": "Canopy cover fraction (L2B)",
    "quality_flag": "GEDI quality flag",
}

# GEDI footprint approximate size (meters)
GEDI_FOOTPRINT_M = 25

# Minimum quality flag value for reliable data
GEDI_QUALITY_THRESHOLD = 1


class GEDIIngestor:
    """Download and process NASA GEDI canopy height data for AOI."""

    def __init__(
        self,
        aoi_geojson: str,
        output_dir: str,
        year: int = 2021,
        target_crs: str = "EPSG:4326",
        quality_threshold: int = GEDI_QUALITY_THRESHOLD,
    ):
        """
        Initialize GEDI ingestor.

        Args:
            aoi_geojson: Path to AOI GeoJSON file
            output_dir: Output directory
            year: Year for GEDI data (GEDI launched 2019, data from 2019+)
            target_crs: Target CRS (default WGS84)
            quality_threshold: Minimum quality flag value (1=best, 0=with caveats)
        """
        self.aoi_path = Path(aoi_geojson)
        self.output_dir = Path(output_dir)
        self.year = year
        self.target_crs = target_crs
        self.quality_threshold = quality_threshold

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "gedi").mkdir(exist_ok=True)

        # Load AOI
        self.aoi_gdf = gpd.read_file(self.aoi_path)
        if self.aoi_gdf.crs is None:
            raise ValueError("AOI GeoJSON must have a CRS defined")
        self.aoi_gdf = self.aoi_gdf.to_crs(self.target_crs)
        self.aoi_bounds = self.aoi_gdf.total_bounds
        self.aoi_geometry = self.aoi_gdf.geometry.unary_union

        # GEDI L2A rasterized product on Google Earth Engine
        # Canonical: https://www.earthdata.nasa.gov/data/catalog/lpcloud-gedi02-a-002
        # GEE: https://developers.google.com/earth-engine/datasets/catalog/LARSE_GEDI_GEDI02_A_002_MONTHLY
        self.gee_catalog_url = "https://earthengine.googleapis.com/api/metadata/v1/catalog/LARSE/GEDI/GEDI02_A_002_MONTHLY"

    def get_gedi_asset_urls(self) -> List[str]:
        """Get GEDI L2A rasterized asset URLs for AOI (from Earth Engine catalog)."""
        import ee

        try:
            ee.Initialize()
        except Exception:
            print("  Earth Engine authentication required for GEDI access.")
            print("  Run: earthengine authenticate")
            return []

        # GEDI L2A rasterized collection
        collection = ee.ImageCollection("LARSE/GEDI/GEDI02_A_002_MONTHLY")
        aoi_ee = ee.Geometry(self.aoi_geometry.__geo_interface__)

        # Filter by date and bounds
        filtered = collection.filterBounds(aoi_ee).filterDate(
            f"{self.year}-01-01", f"{self.year}-12-31"
        )

        # Get available images
        images = filtered.toList(filtered.size())
        count = images.size().getInfo()

        urls = []
        for i in range(count):
            img = ee.Image(images.get(i))
            # Get download URL for canopy top height band
            url = img.select("canopy_top_height").getDownloadURL(
                {"scale": 25, "region": aoi_ee, "crs": self.target_crs}
            )
            urls.append(url)

        return urls

    def process_gedi_image(
        self, image_path: Path, band: str = "canopy_top_height"
    ) -> Dict:
        """Process a single GEDI image (already downloaded)."""
        # This assumes the image was downloaded via GEE export
        # For MVP, we'll process the relative height metrics
        pass

    def create_gedi_coverage_map(
        self, gedi_data: np.ndarray, gedi_quality: np.ndarray, transform, crs
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Create GEDI coverage map and quality-filtered canopy height map.

        Returns:
            (canopy_height, coverage_mask, stats)
            - canopy_height: GEDI canopy top height where available
            - coverage_mask: 1=GEDI coverage, 0=no GEDI data (sparse)
            - stats: summary statistics
        """
        # Apply quality threshold
        valid_quality = gedi_quality >= self.quality_threshold

        # Create coverage mask
        coverage = valid_quality.astype(np.uint8)

        # Quality-filtered canopy height
        canopy_height = np.where(valid_quality, gedi_data, np.nan)

        stats = {
            "gedi_footprint_m": GEDI_FOOTPRINT_M,
            "total_pixels": int(gedi_data.size),
            "covered_pixels": int(np.sum(coverage)),
            "coverage_fraction": float(np.mean(coverage)),
            "mean_canopy_height_m": float(np.nanmean(canopy_height))
            if np.any(coverage) else 0.0,
            "max_canopy_height_m": float(np.nanmax(canopy_height))
            if np.any(coverage) else 0.0,
            "quality_threshold": self.quality_threshold,
        }

        return canopy_height, coverage, stats

    def save_gedi_outputs(
        self,
        canopy_height: np.ndarray,
        coverage: np.ndarray,
        reference_path: Path,
        output_dir: Path,
    ):
        """Save GEDI canopy height and coverage maps."""
        with rasterio.open(reference_path) as src:
            profile = src.profile.copy()
            profile.update(dtype=np.float32, count=1, compress="lzw")

        # Save canopy height (NaN where no coverage)
        height_path = output_dir / "gedi" / "canopy_height.tif"
        with rasterio.open(height_path, "w", **profile) as dst:
            dst.write(canopy_height.astype(np.float32), 1)

        # Save coverage mask
        coverage_path = output_dir / "gedi" / "gedi_coverage.tif"
        cov_profile = profile.copy()
        cov_profile.update(dtype=np.uint8)
        with rasterio.open(coverage_path, "w", **cov_profile) as dst:
            dst.write(coverage, 1)

        return height_path, coverage_path

    def run(self) -> Dict:
        """
        Run GEDI ingestion for AOI.

        NOTE: This is the MVP scaffold. Full implementation requires Earth Engine
        authentication. System MUST remain functional if GEDI is unavailable —
        GEDI is Tier 2 (strongly recommended), not required for MVP.
        """
        print(f"GEDI canopy height ingestion for {self.year}...")
        print(f"  AOI bounds: {self.aoi_bounds}")
        print(f"  IMPORTANT: GEDI is SPARSE (~25m footprint). Missing coverage is expected.")

        # Check if Earth Engine is available
        try:
            import ee
            ee.Initialize()
            ee_available = True
        except Exception as e:
            ee_available = False
            print(f"  Earth Engine not available: {e}")
            print(f"  GEDI data will be marked as UNAVAILABLE.")
            print(f"  System continues with Sentinel-2 + WorldCover only.")

        if not ee_available:
            # Create explicit "unavailable" record
            result = {
                "status": "UNAVAILABLE",
                "reason": "Earth Engine authentication required for GEDI access",
                "data_source": "NASA GEDI L2A",
                "data_source_url": "https://www.earthdata.nasa.gov/data/catalog/lpcloud-gedi02-a-002",
                "aoi": str(self.aoi_path),
                "year": self.year,
                "coverage_fraction": 0.0,
                "note": "GEDI canopy height unavailable — system uses Sentinel-2 NDVI as proxy only. Do NOT treat NDVI as canopy height.",
            }
            summary_path = self.output_dir / "gedi" / "gedi_summary.json"
            with open(summary_path, "w") as f:
                json.dump(result, f, indent=2)
            return result

        # If EE available, would download here
        # URLs = self.get_gedi_asset_urls()
        # For each URL: download, process, create coverage map
        # Skipped in MVP scaffold

        result = {
            "status": "SCAFFOLD",
            "data_source": "NASA GEDI L2A",
            "data_source_url": "https://www.earthdata.nasa.gov/data/catalog/lpcloud-gedi02-a-002",
            "note": "GEDI ingestion scaffold — implement Earth Engine download for production",
        }
        summary_path = self.output_dir / "gedi" / "gedi_summary.json"
        with open(summary_path, "w") as f:
            json.dump(result, f, indent=2)

        return result


def main():
    parser = argparse.ArgumentParser(
        description="NASA GEDI Canopy Height Ingestion (Tier 2)"
    )
    parser.add_argument("--aoi", required=True, help="Path to AOI GeoJSON file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--year", type=int, default=2021, help="Year for GEDI data (default: 2021)")
    parser.add_argument("--target-crs", default="EPSG:4326", help="Target CRS (default: EPSG:4326)")
    parser.add_argument(
        "--quality-threshold", type=int, default=GEDI_QUALITY_THRESHOLD,
        help=f"Minimum GEDI quality flag (default: {GEDI_QUALITY_THRESHOLD})"
    )

    args = parser.parse_args()

    ingestor = GEDIIngestor(
        aoi_geojson=args.aoi,
        output_dir=args.output_dir,
        year=args.year,
        target_crs=args.target_crs,
        quality_threshold=args.quality_threshold,
    )

    result = ingestor.run()
    print(f"\nGEDI ingestion result: {result['status']}")


if __name__ == "__main__":
    main()