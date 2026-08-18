#!/usr/bin/env python3
"""
Feature Engineering Script (Stage 5)

Builds a per-corridor-segment feature table combining:
- Vegetation features (fraction, mean NDVI, class distribution)
- Spatial features (distance to line, corridor edge, tower proximity)
- Temporal features (NDVI change, growth rate)
- Land cover features (woody vs non-woody vegetation from WorldCover)
- Canopy height features (from GEDI L2A)
- Terrain features (slope, aspect from Copernicus DEM)
- Weather context features (from NASA POWER)

Outputs as pandas DataFrame → CSV/PostGIS table.

Known Limitations:
1. Features are at corridor-segment-level (10m resolution Sentinel-2 limitation).
2. GEDI is sparse (~25m footprint) — canopy height may be missing for many segments.
3. WorldCover is 10m and static (2020/2021) — may not reflect current conditions.
4. Weather data provides context for vegetation growth analysis.
   Do NOT directly convert rainfall into risk score without validation.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterstats import zonal_stats
from shapely.geometry import Point, shape
from tqdm import tqdm


class FeatureEngineer:
    """Combine vegetation, spatial, temporal, and auxiliary data source features."""

    def __init__(
        self,
        corridor_segments_path: str,
        vegetation_patches_path: str,
        ndvi_early_path: Optional[str] = None,
        ndvi_late_path: Optional[str] = None,
        temporal_stats_path: Optional[str] = None,
        worldcover_path: Optional[str] = None,
        gedi_path: Optional[str] = None,
        dem_path: Optional[str] = None,
        weather_path: Optional[str] = None,
    ):
        """
        Initialize feature engineer.

        Args:
            corridor_segments_path: Path to corridor segments GeoPackage/GeoJSON
            vegetation_patches_path: Path to vegetation patches GeoPackage/GeoJSON
            ndvi_early_path: Path to early NDVI raster (optional)
            ndvi_late_path: Path to late NDVI raster (optional)
            temporal_stats_path: Path to temporal stats JSON (optional)
            worldcover_path: Path to WorldCover tif (woody mask or classification) (optional)
            gedi_path: Path to GEDI canopy height tif (optional)
            dem_path: Path to Copernicus DEM tif (optional)
            weather_path: Path to NASA POWER daily CSV (optional)
        """
        self.segments = gpd.read_file(corridor_segments_path)
        self.patches = gpd.read_file(vegetation_patches_path)
        self.ndvi_early_path = ndvi_early_path
        self.ndvi_late_path = ndvi_late_path
        self.temporal_stats_path = temporal_stats_path
        self.worldcover_path = worldcover_path
        self.gedi_path = gedi_path
        self.dem_path = dem_path
        self.weather_path = weather_path

        # Ensure same CRS
        if self.segments.crs != self.patches.crs:
            self.patches = self.patches.to_crs(self.segments.crs)

        # Load temporal stats if provided
        self.temporal_stats = None
        if temporal_stats_path:
            with open(temporal_stats_path) as f:
                self.temporal_stats = json.load(f)

    def load_auxiliary_rasters(self) -> Dict[str, np.ndarray]:
        """Load auxiliary rasters (WorldCover, GEDI, DEM) into memory."""
        rasters = {}
        if self.worldcover_path:
            try:
                with rasterio.open(self.worldcover_path) as src:
                    rasters["worldcover"] = src.read(1)
                print(f"  Loaded WorldCover: {self.worldcover_path}")
            except Exception as e:
                print(f"  WorldCover load failed: {e}")

        if self.gedi_path:
            try:
                with rasterio.open(self.gedi_path) as src:
                    rasters["gedi"] = src.read(1)
                print(f"  Loaded GEDI: {self.gedi_path}")
            except Exception as e:
                print(f"  GEDI load failed: {e}")

        if self.dem_path:
            try:
                with rasterio.open(self.dem_path) as src:
                    rasters["dem"] = src.read(1)
                print(f"  Loaded DEM: {self.dem_path}")
            except Exception as e:
                print(f"  DEM load failed: {e}")

        return rasters

    def extract_ndvi_zonal_stats(
        self, ndvi_path: str, feature_name: str
    ) -> Dict[int, Dict]:
        """
        Extract NDVI zonal statistics for each corridor segment.

        Args:
            ndvi_path: Path to NDVI raster
            feature_name: Prefix for feature names (e.g., 'early', 'late')

        Returns:
            Dictionary mapping segment index to NDVI stats
        """
        with rasterio.open(ndvi_path) as src:
            ndvi = src.read(1)
            transform = src.transform

        # Prepare segment geometries for zonal stats
        segment_geoms = [geom for geom in self.segments.geometry]

        # Compute zonal statistics
        stats = zonal_stats(
            segment_geoms,
            ndvi,
            affine=transform,
            stats=["mean", "std", "min", "max", "median", "count"],
            nodata=0,
        )

        # Format as dict
        result = {}
        for i, s in enumerate(stats):
            if s is not None:
                result[i] = {
                    f"{feature_name}_ndvi_mean": s.get("mean", 0),
                    f"{feature_name}_ndvi_std": s.get("std", 0),
                    f"{feature_name}_ndvi_min": s.get("min", 0),
                    f"{feature_name}_ndvi_max": s.get("max", 0),
                    f"{feature_name}_ndvi_median": s.get("median", 0),
                    f"{feature_name}_ndvi_count": s.get("count", 0),
                }
            else:
                result[i] = {
                    f"{feature_name}_ndvi_mean": 0,
                    f"{feature_name}_ndvi_std": 0,
                    f"{feature_name}_ndvi_min": 0,
                    f"{feature_name}_ndvi_max": 0,
                    f"{feature_name}_ndvi_median": 0,
                    f"{feature_name}_ndvi_count": 0,
                }

        return result

    def extract_auxiliary_zonal_stats(
        self, raster: np.ndarray, transform, feature_prefix: str
    ) -> Dict[int, Dict]:
        """
        Extract zonal statistics from auxiliary raster (WorldCover, GEDI, DEM) for each segment.

        Args:
            raster: 2D numpy array
            transform: rasterio Affine transform
            feature_prefix: prefix for feature names

        Returns:
            Dictionary mapping segment index to stats dict
        """
        segment_geoms = [geom for geom in self.segments.geometry]
        stats = zonal_stats(
            segment_geoms,
            raster,
            affine=transform,
            stats=["mean", "std", "min", "max", "median", "count"],
            nodata=np.nan,
        )

        result = {}
        for i, s in enumerate(stats):
            if s is not None:
                result[i] = {
                    f"{feature_prefix}_mean": s.get("mean", np.nan),
                    f"{feature_prefix}_std": s.get("std", np.nan),
                    f"{feature_prefix}_min": s.get("min", np.nan),
                    f"{feature_prefix}_max": s.get("max", np.nan),
                    f"{feature_prefix}_median": s.get("median", np.nan),
                    f"{feature_prefix}_count": s.get("count", 0),
                }
            else:
                result[i] = {
                    f"{feature_prefix}_mean": np.nan,
                    f"{feature_prefix}_std": np.nan,
                    f"{feature_prefix}_min": np.nan,
                    f"{feature_prefix}_max": np.nan,
                    f"{feature_prefix}_median": np.nan,
                    f"{feature_prefix}_count": 0,
                }
        return result

    def compute_vegetation_fraction_per_segment(self) -> pd.DataFrame:
        """
        Compute vegetation fraction and class distribution per corridor segment.

        Returns:
            DataFrame with vegetation features per segment
        """
        segment_features = []

        for idx, segment in self.segments.iterrows():
            # Find patches intersecting this segment's corridor buffer
            segment_buffer = segment.geometry.buffer(50)  # 50m buffer (corridor half-width)
            intersecting = self.patches[self.patches.geometry.intersects(segment_buffer)]

            if len(intersecting) > 0:
                total_area = segment.geometry.length * 100  # 50m on each side = 100m wide
                veg_area = intersecting["area_m2"].sum()
                veg_fraction = veg_area / total_area if total_area > 0 else 0

                # Vegetation class distribution (if available)
                class_counts = {}
                if "veg_class" in intersecting.columns:
                    for cls in [1, 2, 3]:
                        class_counts[f"veg_class_{cls}_count"] = len(
                            intersecting[intersecting["veg_class"] == cls]
                        )
                        class_counts[f"veg_class_{cls}_area"] = intersecting[
                            intersecting["veg_class"] == cls
                        ]["area_m2"].sum()
                else:
                    class_counts = {
                        "veg_class_1_count": 0,
                        "veg_class_2_count": 0,
                        "veg_class_3_count": 0,
                        "veg_class_1_area": 0,
                        "veg_class_2_area": 0,
                        "veg_class_3_area": 0,
                    }

                features = {
                    "segment_id": segment.get("segment_id", idx),
                    "vegetation_fraction": veg_fraction,
                    "veg_patch_count": len(intersecting),
                    "total_veg_area_m2": veg_area,
                    "mean_veg_area_m2": intersecting["area_m2"].mean(),
                    "max_veg_area_m2": intersecting["area_m2"].max(),
                    "mean_dist_to_line_m": intersecting["dist_to_line_m"].mean(),
                    "min_dist_to_line_m": intersecting["dist_to_line_m"].min(),
                    "mean_dist_to_corridor_edge_m": intersecting[
                        "dist_to_corridor_edge_m"
                    ].mean()
                    if "dist_to_corridor_edge_m" in intersecting.columns
                    else 0,
                    "pct_inside_corridor": intersecting["inside_corridor"].mean() * 100
                    if "inside_corridor" in intersecting.columns
                    else 0,
                    "pct_near_tower": intersecting["near_tower"].mean() * 100
                    if "near_tower" in intersecting.columns
                    else 0,
                    **class_counts,
                }
            else:
                features = {
                    "segment_id": segment.get("segment_id", idx),
                    "vegetation_fraction": 0,
                    "veg_patch_count": 0,
                    "total_veg_area_m2": 0,
                    "mean_veg_area_m2": 0,
                    "max_veg_area_m2": 0,
                    "mean_dist_to_line_m": float("inf"),
                    "min_dist_to_line_m": float("inf"),
                    "mean_dist_to_corridor_edge_m": 0,
                    "pct_inside_corridor": 0,
                    "pct_near_tower": 0,
                    "veg_class_1_count": 0,
                    "veg_class_2_count": 0,
                    "veg_class_3_count": 0,
                    "veg_class_1_area": 0,
                    "veg_class_2_area": 0,
                    "veg_class_3_area": 0,
                }

            segment_features.append(features)

        return pd.DataFrame(segment_features)

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features (NDVI change, growth rate) if available."""
        if self.temporal_stats is None:
            # Add placeholder columns
            temporal_cols = [
                "ndvi_change_mean",
                "ndvi_change_std",
                "growth_rate_mean",
                "growth_rate_std",
                "pct_significant_increase",
                "pct_significant_decrease",
            ]
            for col in temporal_cols:
                df[col] = 0.0
            return df

        # If we have per-pixel temporal stats, we could add zonal stats
        # For now, add global stats as segment-level features
        change_stats = self.temporal_stats.get("change_stats", {})
        growth_stats = self.temporal_stats.get("growth_stats", {})

        df["ndvi_change_mean"] = change_stats.get("mean_change", 0)
        df["ndvi_change_std"] = change_stats.get("std_change", 0)
        df["growth_rate_mean"] = growth_stats.get("mean_growth_rate_pct_per_day", 0)
        df["growth_rate_std"] = growth_stats.get("std_growth_rate", 0)
        df["pct_significant_increase"] = change_stats.get(
            "pct_significant_increase", 0
        )
        df["pct_significant_decrease"] = change_stats.get(
            "pct_significant_decrease", 0
        )

        return df

    def add_ndvi_zonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add NDVI zonal statistics for early and late dates if available."""
        if self.ndvi_early_path:
            early_stats = self.extract_ndvi_zonal_stats(
                self.ndvi_early_path, "early"
            )
            for idx, stats in early_stats.items():
                if idx < len(df):
                    for k, v in stats.items():
                        df.loc[idx, k] = v

        if self.ndvi_late_path:
            late_stats = self.extract_ndvi_zonal_stats(
                self.ndvi_late_path, "late"
            )
            for idx, stats in late_stats.items():
                if idx < len(df):
                    for k, v in stats.items():
                        df.loc[idx, k] = v

        return df

    def add_auxiliary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features from auxiliary data sources (WorldCover, GEDI, DEM, Weather)."""
        print("  Loading auxiliary rasters...")
        rasters = self.load_auxiliary_rasters()

        # WorldCover features
        if "worldcover" in rasters:
            print("  Adding WorldCover features...")
            # Need to get transform from the raster
            with rasterio.open(self.worldcover_path) as src:
                wc_stats = self.extract_auxiliary_zonal_stats(
                    rasters["worldcover"], src.transform, "worldcover"
                )
            for idx, stats in wc_stats.items():
                if idx < len(df):
                    for k, v in stats.items():
                        df.loc[idx, k] = v

        # GEDI canopy height features
        if "gedi" in rasters:
            print("  Adding GEDI canopy height features...")
            with rasterio.open(self.gedi_path) as src:
                gedi_stats = self.extract_auxiliary_zonal_stats(
                    rasters["gedi"], src.transform, "gedi_canopy_height"
                )
            for idx, stats in gedi_stats.items():
                if idx < len(df):
                    for k, v in stats.items():
                        df.loc[idx, k] = v

        # DEM terrain features
        if "dem" in rasters:
            print("  Adding DEM terrain features...")
            with rasterio.open(self.dem_path) as src:
                dem_stats = self.extract_auxiliary_zonal_stats(
                    rasters["dem"], src.transform, "dem_elevation"
                )
            for idx, stats in dem_stats.items():
                if idx < len(df):
                    for k, v in stats.items():
                        df.loc[idx, k] = v

        # Weather features (load from CSV summary)
        if self.weather_path:
            print("  Adding weather context features...")
            try:
                weather_df = pd.read_csv(self.weather_path)
                if "date" in weather_df.columns:
                    # Compute summary stats
                    df["weather_total_precip_mm"] = weather_df["prectotcorr"].sum() if "prectotcorr" in weather_df.columns else 0
                    df["weather_mean_temp_c"] = weather_df["t2m"].mean() if "t2m" in weather_df.columns else 0
                    df["weather_mean_humidity_pct"] = weather_df["rh2m"].mean() if "rh2m" in weather_df.columns else 0
                    df["weather_mean_wind_ms"] = weather_df["ws2m"].mean() if "ws2m" in weather_df.columns else 0
                    df["weather_mean_solar_radiation"] = weather_df["allsky_sfc_sw_dwn"].mean() if "allsky_sfc_sw_dwn" in weather_df.columns else 0
            except Exception as e:
                print(f"  Weather feature extraction failed: {e}")

        return df

    def build_feature_table(self) -> pd.DataFrame:
        """Build complete feature table."""
        print("Building vegetation features...")
        df = self.compute_vegetation_fraction_per_segment()

        print("Adding temporal features...")
        df = self.add_temporal_features(df)

        print("Adding NDVI zonal features...")
        df = self.add_ndvi_zonal_features(df)

        print("Adding auxiliary data source features...")
        df = self.add_auxiliary_features(df)

        print("Adding spatial segment attributes...")
        # Add segment geometry attributes
        df["segment_length_m"] = self.segments.geometry.length
        df["segment_geometry"] = self.segments.geometry.apply(lambda g: g.wkt if g else None)

        # Fill infinite values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        return df

    def save_feature_table(self, df: pd.DataFrame, output_dir: str, name: str = "features"):
        """Save feature table as CSV and GeoPackage."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save CSV
        csv_path = output_dir / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved CSV: {csv_path}")

        # Save as GeoPackage (with geometry)
        # Create GeoDataFrame with segment geometries
        gdf = self.segments.copy()
        for col in df.columns:
            if col not in gdf.columns:
                gdf[col] = df[col]

        gpkg_path = output_dir / f"{name}.gpkg"
        gdf.to_file(gpkg_path, driver="GPKG")
        print(f"  Saved GeoPackage: {gpkg_path}")

        # Save feature metadata
        meta = {
            "num_segments": len(df),
            "num_features": len(df.columns),
            "feature_names": list(df.columns),
            "crs": str(self.segments.crs),
        }
        meta_path = output_dir / f"{name}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  Saved metadata: {meta_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Feature Engineering (Stage 5): Build per-corridor-segment feature table"
    )
    parser.add_argument(
        "--corridor-segments", required=True, help="Path to corridor segments (GPKG/GeoJSON)"
    )
    parser.add_argument(
        "--vegetation-patches", required=True, help="Path to vegetation patches (GPKG/GeoJSON)"
    )
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--ndvi-early", help="Path to early date NDVI raster (optional)")
    parser.add_argument("--ndvi-late", help="Path to late date NDVI raster (optional)")
    parser.add_argument("--temporal-stats", help="Path to temporal stats JSON (optional)")
    parser.add_argument("--worldcover-path", help="Path to WorldCover raster (optional)")
    parser.add_argument("--gedi-path", help="Path to GEDI canopy height raster (optional)")
    parser.add_argument("--dem-path", help="Path to Copernicus DEM raster (optional)")
    parser.add_argument("--weather-path", help="Path to NASA POWER daily CSV (optional)")

    args = parser.parse_args()

    engineer = FeatureEngineer(
        corridor_segments_path=args.corridor_segments,
        vegetation_patches_path=args.vegetation_patches,
        ndvi_early_path=args.ndvi_early,
        ndvi_late_path=args.ndvi_late,
        temporal_stats_path=args.temporal_stats,
        worldcover_path=args.worldcover_path,
        gedi_path=args.gedi_path,
        dem_path=args.dem_path,
        weather_path=args.weather_path,
    )

    df = engineer.build_feature_table()
    engineer.save_feature_table(df, args.output_dir)

    print(f"\nFeature Engineering Complete:")
    print(f"  Segments: {len(df)}")
    print(f"  Features: {len(df.columns)}")
    print(f"  Key features: vegetation_fraction, mean_dist_to_line, ndvi_change_mean, growth_rate_mean")


if __name__ == "__main__":
    main()