#!/usr/bin/env python3
"""
Spatial Analysis Script (Stage 3)

Computes per-vegetation-patch distance features relative to transmission lines:
- Distance to nearest line segment
- Distance to corridor edge
- Tower proximity (distance to nearest tower)
- Inside/outside corridor buffer (boolean)

Integrates ESA WorldCover woody vegetation mask (optional) to filter
vegetation patches - only woody vegetation (tree cover, shrubland, mangroves)
is retained for risk analysis, excluding crops/grassland.

Uses geopandas + shapely spatial index for performance.

Known Limitations:
1. Sentinel-2 is 10m resolution — cannot resolve individual trees near conductors.
   Risk scores are corridor-segment-level, not tree-level.
2. Tower locations may be approximate (from GeoJSON/Shapefile data).
3. Corridor width is defined as a buffer around lines, not actual surveyed corridor.
4. WorldCover is static (2020/2021) — may not reflect current conditions.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Polygon, shape

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import (
    Point,
    LineString,
    Polygon,
    MultiPolygon,
    shape,
    mapping,
    box,
)
from shapely.ops import nearest_points, transform
from shapely.strtree import STRtree
import pyproj


# Default corridor buffer widths (meters)
DEFAULT_CORRIDOR_BUFFER_M = 50  # Half-width of transmission corridor
DEFAULT_TOWER_INFLUENCE_M = 200  # Buffer around towers for "tower proximity"


class SpatialAnalyzer:
    """Compute spatial features for vegetation patches relative to transmission infrastructure."""

    def __init__(
        self,
        transmission_line_geojson: str,
        tower_geojson: Optional[str] = None,
        corridor_buffer_m: float = DEFAULT_CORRIDOR_BUFFER_M,
        tower_influence_m: float = DEFAULT_TOWER_INFLUENCE_M,
    ):
        """
        Initialize spatial analyzer.

        Args:
            transmission_line_geojson: Path to GeoJSON with transmission line geometry
            tower_geojson: Path to GeoJSON with tower locations (optional)
            corridor_buffer_m: Corridor half-width in meters (default 50m)
            tower_influence_m: Tower influence radius in meters (default 200m)
        """
        self.corridor_buffer_m = corridor_buffer_m
        self.tower_influence_m = tower_influence_m

        # Load transmission lines
        self.lines_gdf = gpd.read_file(transmission_line_geojson)
        # Ensure CRS is in meters for distance calculations
        if self.lines_gdf.crs and self.lines_gdf.crs.is_geographic:
            self.work_crs = "EPSG:3857"  # Web Mercator for metric calculations
            self.lines_gdf = self.lines_gdf.to_crs(self.work_crs)
        else:
            self.work_crs = self.lines_gdf.crs

        # Load towers (optional)
        self.towers_gdf = None
        if tower_geojson:
            self.towers_gdf = gpd.read_file(tower_geojson)
            if self.towers_gdf.crs and self.towers_gdf.crs.is_geographic:
                self.towers_gdf = self.towers_gdf.to_crs(self.work_crs)

        # Create corridor geometry (buffer around lines)
        self.corridor = self._create_corridor()
        self.corridor_buffer = self._create_corridor_buffer()

        # Build spatial indices
        self._build_spatial_indices()

    def _create_corridor(self) -> Polygon:
        """Create corridor geometry by buffering transmission lines."""
        if self.lines_gdf.empty:
            raise ValueError("No transmission lines provided")

        corridor_parts = []
        for geom in self.lines_gdf.geometry:
            if geom is None:
                continue
            if isinstance(geom, LineString):
                corridor_parts.append(geom.buffer(self.corridor_buffer_m, resolution=16))
            elif hasattr(geom, 'geoms'):  # MultiLineString
                for line in geom.geoms:
                    corridor_parts.append(line.buffer(self.corridor_buffer_m, resolution=16))

        if not corridor_parts:
            raise ValueError("No valid LineString geometries found in transmission lines")

        corridor = gpd.GeoSeries(corridor_parts).unary_union
        return corridor

    def _create_corridor_buffer(self) -> Polygon:
        """Create extended corridor buffer (for distance calculations)."""
        return self.corridor.buffer(self.corridor_buffer_m * 2)

    def _build_spatial_indices(self):
        """Build STRtree spatial indices for efficient distance queries."""
        # Line segments index
        line_geoms = []
        for geom in self.lines_gdf.geometry:
            if geom is None:
                continue
            if isinstance(geom, LineString):
                line_geoms.append(geom)
            elif hasattr(geom, 'geoms'):
                line_geoms.extend(geom.geoms)
        self.line_tree = STRtree(line_geoms)
        self.line_geoms = line_geoms

        # Tower index (if available)
        self.tower_tree = None
        self.tower_geoms = []
        if self.towers_gdf is not None:
            tower_points = [
                geom if isinstance(geom, Point) else geom.centroid
                for geom in self.towers_gdf.geometry
                if geom is not None
            ]
            self.tower_geoms = tower_points
            self.tower_tree = STRtree(tower_points)

    def vectorize_vegetation_mask(
        self, mask_path: str, min_patch_area_m2: float = 100.0
    ) -> gpd.GeoDataFrame:
        """
        Convert raster vegetation mask to vector polygons.

        Args:
            mask_path: Path to vegetation mask GeoTIFF (1=vegetation, 0=non-veg)
            min_patch_area_m2: Minimum patch area in square meters to keep

        Returns:
            GeoDataFrame with vegetation patch polygons
        """
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
            transform = src.transform
            crs = src.crs

        # Vectorize mask
        mask_binary = (mask == 1).astype(np.uint8)
        patch_dicts = [
            {"properties": {"veg_id": i + 1}, "geometry": shape(geom)}
            for i, (geom, value) in enumerate(
                shapes(mask_binary, mask=mask_binary, transform=transform)
            )
            if value == 1
        ]

        if not patch_dicts:
            return gpd.GeoDataFrame(columns=["geometry", "veg_id", "area_m2"], crs=crs)

        gdf = gpd.GeoDataFrame.from_features(patch_dicts, crs=crs)

        # Convert to working CRS if needed
        if self.work_crs and gdf.crs and gdf.crs != self.work_crs:
            gdf = gdf.to_crs(self.work_crs)

        # Calculate area and filter small patches
        gdf["area_m2"] = gdf.geometry.area
        gdf = gdf[gdf["area_m2"] >= min_patch_area_m2].reset_index(drop=True)

        print(f"  Vectorized {len(gdf)} vegetation patches (min area: {min_patch_area_m2} m²)")
        return gdf

    def compute_distance_features(self, veg_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Compute spatial features for each vegetation patch.

        Features:
        - dist_to_line_m: Distance to nearest transmission line segment
        - dist_to_corridor_edge_m: Distance to nearest corridor edge
        - dist_to_tower_m: Distance to nearest tower (if tower data available)
        - inside_corridor: Whether patch is inside corridor buffer (boolean)
        - near_tower: Whether patch is within tower influence radius
        """
        print("Computing spatial features...")

        # Distance to nearest line segment
        veg_gdf["dist_to_line_m"] = veg_gdf.geometry.apply(
            lambda geom: self._distance_to_lines(geom)
        )

        # Distance to corridor edge (negative if inside, positive if outside)
        veg_gdf["dist_to_corridor_edge_m"] = veg_gdf.geometry.apply(
            lambda geom: self._distance_to_corridor_edge(geom)
        )

        # Inside corridor buffer
        veg_gdf["inside_corridor"] = veg_gdf.geometry.apply(
            lambda geom: self.corridor.intersects(geom)
        )

        # Tower proximity
        if self.tower_tree is not None:
            veg_gdf["dist_to_tower_m"] = veg_gdf.geometry.apply(
                lambda geom: self._distance_to_nearest_tower(geom)
            )
            veg_gdf["near_tower"] = (
                veg_gdf["dist_to_tower_m"] <= self.tower_influence_m
            )
        else:
            veg_gdf["dist_to_tower_m"] = np.nan
            veg_gdf["near_tower"] = False

        # Additional features
        veg_gdf["centroid_x"] = veg_gdf.geometry.centroid.x
        veg_gdf["centroid_y"] = veg_gdf.geometry.centroid.y

        # Normalize distances (0-1 scale, inversely proportional)
        max_line_dist = veg_gdf["dist_to_line_m"].max()
        if max_line_dist > 0:
            veg_gdf["proximity_score"] = 1 - (veg_gdf["dist_to_line_m"] / max_line_dist)
        else:
            veg_gdf["proximity_score"] = 1.0

        print(f"  Computed features for {len(veg_gdf)} patches")
        return veg_gdf

    def _distance_to_lines(self, geom) -> float:
        """Compute distance from geometry to nearest transmission line segment."""
        min_dist = float("inf")
        for line in self.line_geoms:
            d = geom.distance(line)
            if d < min_dist:
                min_dist = d
        return min_dist

    def _distance_to_corridor_edge(self, geom) -> float:
        """
        Distance to corridor edge.
        Negative if inside corridor, positive if outside.
        """
        centroid = geom.centroid
        if self.corridor.contains(centroid):
            # Inside corridor - distance to boundary
            boundary = self.corridor.boundary
            dist = centroid.distance(boundary)
            return -dist  # Negative means inside
        else:
            # Outside corridor - distance to boundary
            boundary = self.corridor.boundary
            dist = centroid.distance(boundary)
            return dist  # Positive means outside

    def _distance_to_nearest_tower(self, geom) -> float:
        """Compute distance from geometry to nearest tower."""
        centroid = geom.centroid
        nearest = nearest_points(centroid, self.tower_tree.geometries)[1]
        return centroid.distance(nearest)

    def compute_per_line_segment_features(
        self, veg_gdf: gpd.GeoDataFrame, segment_length_m: float = 500.0
    ) -> gpd.GeoDataFrame:
        """
        Compute aggregated features per corridor segment (line segment).

        Args:
            veg_gdf: Vegetation patches with distance features
            segment_length_m: Length of each line segment for aggregation

        Returns:
            GeoDataFrame with per-line-segment aggregated features
        """
        segments = []
        segment_id = 0

        for line_idx, line in enumerate(self.line_geoms):
            length = line.length
            num_segments = int(np.ceil(length / segment_length_m))

            for seg_idx in range(num_segments):
                start_dist = seg_idx * segment_length_m
                end_dist = min((seg_idx + 1) * segment_length_m, length)

                # Extract segment
                segment = line.interpolate(start_dist)
                segment_geom = line.interpolate(end_dist)
                segment_line = LineString([
                    (segment.x, segment.y),
                    (segment_geom.x, segment_geom.y),
                ])

                # Create corridor buffer for this segment
                segment_corridor = segment_line.buffer(self.corridor_buffer_m, resolution=8)

                # Find vegetation patches that intersect this segment corridor
                intersecting = veg_gdf[veg_gdf.geometry.intersects(segment_corridor)]

                if len(intersecting) > 0:
                    # Aggregate features
                    agg_features = {
                        "segment_id": segment_id,
                        "line_idx": line_idx,
                        "segment_idx": seg_idx,
                        "geometry": segment_line,
                        "num_patches": len(intersecting),
                        "total_veg_area_m2": intersecting["area_m2"].sum(),
                        "mean_veg_area_m2": intersecting["area_m2"].mean(),
                        "mean_dist_to_line_m": intersecting["dist_to_line_m"].mean(),
                        "min_dist_to_line_m": intersecting["dist_to_line_m"].min(),
                        "pct_inside_corridor": intersecting["inside_corridor"].mean() * 100,
                        "pct_near_tower": intersecting["near_tower"].mean() * 100
                        if "near_tower" in intersecting.columns
                        else 0,
                        "vegetation_fraction": len(intersecting) / max(len(veg_gdf), 1),
                    }
                else:
                    agg_features = {
                        "segment_id": segment_id,
                        "line_idx": line_idx,
                        "segment_idx": seg_idx,
                        "geometry": segment_line,
                        "num_patches": 0,
                        "total_veg_area_m2": 0,
                        "mean_veg_area_m2": 0,
                        "mean_dist_to_line_m": float("inf"),
                        "min_dist_to_line_m": float("inf"),
                        "pct_inside_corridor": 0,
                        "pct_near_tower": 0,
                        "vegetation_fraction": 0,
                    }

                segments.append(agg_features)
                segment_id += 1

        segments_gdf = gpd.GeoDataFrame(segments, crs=self.work_crs)
        print(f"  Computed features for {len(segments_gdf)} corridor segments")
        return segments_gdf


def main():
    parser = argparse.ArgumentParser(
        description="Spatial Analysis (Stage 3): Compute vegetation proximity to transmission lines"
    )
    parser.add_argument(
        "--transmission-lines", required=True, help="Path to transmission lines GeoJSON"
    )
    parser.add_argument("--tower-locations", help="Path to tower locations GeoJSON (optional)")
    parser.add_argument(
        "--vegetation-mask", required=True, help="Path to vegetation mask GeoTIFF"
    )
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--corridor-buffer-m",
        type=float,
        default=DEFAULT_CORRIDOR_BUFFER_M,
        help=f"Corridor half-width in meters (default: {DEFAULT_CORRIDOR_BUFFER_M})",
    )
    parser.add_argument(
        "--min-patch-area-m2",
        type=float,
        default=100,
        help="Minimum vegetation patch area to include (default: 100 m²)",
    )
    parser.add_argument(
        "--segment-length-m",
        type=float,
        default=500,
        help="Length of each corridor segment for aggregation (default: 500m)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize analyzer
    analyzer = SpatialAnalyzer(
        transmission_line_geojson=args.transmission_lines,
        tower_geojson=args.tower_locations,
        corridor_buffer_m=args.corridor_buffer_m,
    )

    # Vectorize vegetation mask
    veg_gdf = analyzer.vectorize_vegetation_mask(
        args.vegetation_mask, min_patch_area_m2=args.min_patch_area_m2
    )

    # Compute distance features
    veg_gdf = analyzer.compute_distance_features(veg_gdf)

    # Save vegetation patches with features
    patches_path = output_dir / "vegetation_patches.gpkg"
    veg_gdf.to_file(patches_path, driver="GPKG")
    print(f"  Saved vegetation patches to: {patches_path}")

    # Compute per-corridor-segment features
    segments_gdf = analyzer.compute_per_line_segment_features(
        veg_gdf, segment_length_m=args.segment_length_m
    )

    # Save segments
    segments_path = output_dir / "corridor_segments.gpkg"
    segments_gdf.to_file(segments_path, driver="GPKG")
    print(f"  Saved corridor segments to: {segments_path}")

    # Also save as CSV for feature engineering
    csv_path = output_dir / "corridor_segment_features.csv"
    segments_gdf.drop(columns=["geometry"]).to_csv(csv_path, index=False)
    print(f"  Saved features CSV to: {csv_path}")

    # Summary
    print(f"\nSpatial Analysis Summary:")
    print(f"  Vegetation patches: {len(veg_gdf)}")
    print(f"  Corridor segments: {len(segments_gdf)}")
    print(f"  Mean distance to line: {veg_gdf['dist_to_line_m'].mean():.1f} m")
    print(f"  Mean vegetation fraction per segment: {segments_gdf['vegetation_fraction'].mean():.3f}")


if __name__ == "__main__":
    main()