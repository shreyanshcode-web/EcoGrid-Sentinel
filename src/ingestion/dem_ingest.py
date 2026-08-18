#!/usr/bin/env python3
"""
Copernicus DEM Ingestion

Official source: https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM

Copernicus DEM provides Digital Surface Model (DSM) products:
- GLO-30: 30m global coverage
- GLO-90: 90m global coverage

IMPORTANT: DEM elevation is NOT canopy height.
Use DEM for:
- Terrain elevation
- Terrain context
- Slope
- Topographic features
- Aspect
- Hillshade

Canopy height comes from GEDI (Tier 2), NOT from DEM.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
import rasterio.warp
from rasterio.enums import Resampling
from scipy import ndimage
from shapely.geometry import box, shape
import geopandas as gpd
from tqdm import tqdm


class DEMIngestor:
    """Download and process Copernicus DEM for AOI."""

    def __init__(
        self,
        aoi_geojson: str,
        output_dir: str,
        resolution: str = "GLO-30",
        target_crs: str = "EPSG:4326",
    ):
        """
        Initialize DEM ingestor.

        Args:
            aoi_geojson: Path to AOI GeoJSON file
            output_dir: Output directory
            resolution: DEM resolution ('GLO-30' or 'GLO-90')
            target_crs: Target CRS (default WGS84)
        """
        self.aoi_path = Path(aoi_geojson)
        self.output_dir = Path(output_dir)
        self.resolution = resolution
        self.target_crs = target_crs

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "dem").mkdir(exist_ok=True)

        # Load AOI
        self.aoi_gdf = gpd.read_file(self.aoi_path)
        if self.aoi_gdf.crs is None:
            raise ValueError("AOI GeoJSON must have a CRS defined")
        self.aoi_gdf = self.aoi_gdf.to_crs(self.target_crs)
        self.aoi_bounds = self.aoi_gdf.total_bounds
        self.aoi_geometry = self.aoi_gdf.geometry.unary_union

        # Copernicus DEM STAC collection
        # Note: Access via Copernicus Data Space STAC or AWS Open Data
        self.stac_url = "https://catalogue.dataspace.copernicus.eu/stac/"
        self.collection_id = f"COP-DEM-{resolution}"

    def get_dem_asset_urls(self) -> List[str]:
        """Get DEM asset URLs for AOI tiles."""
        import pystac_client

        catalog = pystac_client.Client.open(self.stac_url)

        search = catalog.search(
            collections=[self.collection_id],
            intersects=self.aoi_geometry,
        )

        items = list(search.items())
        # For COP-DEM, assets are typically 'data' or 'elevation'
        asset_key = "data" if "data" in items[0].assets else "elevation"
        return [item.assets[asset_key].href for item in items]

    def download_and_clip(self, asset_url: str, output_path: Path) -> Dict:
        """Download DEM tile and clip to AOI."""
        with rasterio.open(asset_url) as src:
            # Read and clip to AOI
            aoi_bounds_src = rasterio.warp.transform_bounds(
                self.target_crs, src.crs, *self.aoi_bounds
            )
            window = rasterio.windows.from_bounds(
                *aoi_bounds_src, transform=src.transform
            )
            window = window.round_offsets().round_shape()

            data = src.read(1, window=window)
            clipped_transform = rasterio.windows.transform(window, src.transform)

            # Reproject to target CRS if needed
            if src.crs.to_string() != self.target_crs:
                transform, width, height = rasterio.warp.calculate_default_transform(
                    src.crs, self.target_crs, data.shape[1], data.shape[0],
                    *rasterio.windows.bounds(window, src.transform)
                )
                out_data = np.empty((height, width), dtype=src.dtypes[0])
                rasterio.warp.reproject(
                    source=data,
                    destination=out_data,
                    src_transform=clipped_transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=self.target_crs,
                    resampling=Resampling.bilinear,
                )
                data = out_data
                final_transform = transform
            else:
                final_transform = clipped_transform

            # Save clipped DEM
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
                "min_elevation": float(np.nanmin(data)),
                "max_elevation": float(np.nanmax(data)),
                "mean_elevation": float(np.nanmean(data)),
            }

    def compute_terrain_derivatives(self, dem_path: Path) -> Dict:
        """Compute terrain derivatives from DEM."""
        with rasterio.open(dem_path) as src:
            dem = src.read(1).astype(np.float32)
            transform = src.transform
            crs = src.crs
            pixel_size = transform[0]  # Assuming square pixels in degrees

        # Convert pixel size to meters (approximate at this latitude)
        # 1 degree latitude ≈ 111,320 meters
        # 1 degree longitude ≈ 111,320 * cos(latitude) meters
        lat = (transform[5] + transform[2] * dem.shape[0] / 2)  # Approximate center lat
        pixel_size_m = pixel_size * 111320 * np.cos(np.radians(lat))

        # Compute slope (degrees)
        # Using Horn's method via scipy
        dzdx = ndimage.sobel(dem, axis=1) / (8 * pixel_size_m)
        dzdy = ndimage.sobel(dem, axis=0) / (8 * pixel_size_m)
        slope = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))

        # Compute aspect (degrees, 0=north, clockwise)
        aspect = np.degrees(np.arctan2(-dzdx, dzdy))
        aspect = np.where(aspect < 0, aspect + 360, aspect)

        # Hillshade (for visualization)
        azimuth = 315  # NW light source
        altitude = 45
        hillshade = 255 * (
            np.cos(np.radians(90 - altitude)) * np.cos(np.radians(slope))
            + np.sin(np.radians(90 - altitude)) * np.sin(np.radians(slope))
            * np.cos(np.radians(azimuth - aspect))
        )
        hillshade = np.clip(hillshade, 0, 255).astype(np.uint8)

        # Save derivatives
        output_dir = self.output_dir / "dem"
        base_name = dem_path.stem

        for name, data, dtype in [
            (f"{base_name}_slope", slope, np.float32),
            (f"{base_name}_aspect", aspect, np.float32),
            (f"{base_name}_hillshade", hillshade, np.uint8),
        ]:
            out_path = output_dir / f"{name}.tif"
            profile = {
                'driver': 'GTiff',
                'height': data.shape[0],
                'width': data.shape[1],
                'count': 1,
                'dtype': dtype,
                'crs': crs,
                'transform': transform,
                'compress': 'lzw',
            }
            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(data, 1)

        stats = {
            "slope_mean": float(np.nanmean(slope)),
            "slope_max": float(np.nanmax(slope)),
            "aspect_mean": float(np.nanmean(aspect)),
        }

        return {
            "slope_path": str(output_dir / f"{base_name}_slope.tif"),
            "aspect_path": str(output_dir / f"{base_name}_aspect.tif"),
            "hillshade_path": str(output_dir / f"{base_name}_hillshade.tif"),
            "stats": stats,
        }

    def run(self) -> Dict:
        """Run DEM ingestion for AOI."""
        print(f"Fetching Copernicus DEM {self.resolution} for AOI...")
        print(f"  AOI bounds: {self.aoi_bounds}")
        print(f"  IMPORTANT: DEM elevation is NOT canopy height.")

        try:
            asset_urls = self.get_dem_asset_urls()
            print(f"Found {len(asset_urls)} DEM tiles")
        except Exception as e:
            print(f"  Could not access DEM STAC catalog: {e}")
            print(f"  Trying AWS Open Data alternative...")
            asset_urls = []

        if not asset_urls:
            # Fallback: SRTM via AWS Open Data or local files
            # For MVP, create explicit "unavailable" record
            result = {
                "status": "UNAVAILABLE",
                "reason": "DEM tiles not accessible via STAC — implement AWS Open Data SRTM fallback",
                "data_source": f"Copernicus DEM {self.resolution}",
                "data_source_url": "https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM",
                "fallback": "SRTM (90m) via AWS Open Data",
                "fallback_url": "https://registry.opendata.aws/srtm/",
                "aoi": str(self.aoi_path),
                "note": "DEM unavailable — system proceeds without terrain derivatives. Add DEM for slope/aspect context.",
            }
            summary_path = self.output_dir / "dem" / "dem_summary.json"
            with open(summary_path, "w") as f:
                json.dump(result, f, indent=2)
            return result

        results = []
        derivatives = []

        for i, url in enumerate(tqdm(asset_urls, desc="Processing DEM tiles")):
            tile_path = self.output_dir / "dem" / f"dem_{self.resolution}_tile_{i}.tif"
            meta = self.download_and_clip(url, tile_path)
            results.append(meta)

            # Compute terrain derivatives
            deriv = self.compute_terrain_derivatives(tile_path)
            derivatives.append(deriv)

            print(f"  Tile {i}: elevation {meta['min_elevation']:.0f}-{meta['max_elevation']:.0f}m")

        summary = {
            "status": "SUCCESS",
            "resolution": self.resolution,
            "data_source": f"Copernicus DEM {self.resolution}",
            "data_source_url": "https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM",
            "tiles_processed": len(results),
            "tile_details": results,
            "derivatives": derivatives,
            "note": "DEM elevation is NOT canopy height. Canopy height from GEDI (Tier 2).",
        }

        summary_path = self.output_dir / "dem" / "dem_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nDEM processing complete.")
        print(f"  Summary saved to: {summary_path}")

        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Copernicus DEM Ingestion (Tier 1/2)"
    )
    parser.add_argument("--aoi", required=True, help="Path to AOI GeoJSON file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--resolution", choices=["GLO-30", "GLO-90"], default="GLO-30",
                        help="DEM resolution (default: GLO-30)")
    parser.add_argument("--target-crs", default="EPSG:4326", help="Target CRS (default: EPSG:4326)")

    args = parser.parse_args()

    ingestor = DEMIngestor(
        aoi_geojson=args.aoi,
        output_dir=args.output_dir,
        resolution=args.resolution,
        target_crs=args.target_crs,
    )

    ingestor.run()


if __name__ == "__main__":
    main()