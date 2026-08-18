#!/usr/bin/env python3
"""
Master Pipeline Orchestration Script

Runs all stages in sequence:
1a. Sentinel-2 Ingestion (Tier 1) - Copernicus Data Space STAC
1b. ESA WorldCover Land Cover Ingestion (Tier 1) - woody vegetation mask
1c. NASA GEDI Canopy Height Ingestion (Tier 2) - sparse, ~25m footprint
1d. Copernicus DEM Ingestion (Tier 1) - terrain context, NOT canopy height
1e. NASA POWER Weather/Climate Ingestion (Tier 2) - seasonal growth context
2.  Vegetation Analysis - NDVI/vegetation fraction from Sentinel-2
3.  Spatial Analysis - proximity features relative to transmission lines
4.  Temporal Analysis - NDVI change detection (if multi-date)
5.  Feature Engineering - per-corridor-segment feature table
6.  Risk Scoring - weighted heuristic risk assessment

IMPORTANT: This is a decision-support/inspection-prioritization system.
It does NOT claim Sentinel-2 alone can determine tree-conductor contact.

Usage:
    python pipeline.py --aoi aoi.geojson --start-date 2024-01-01 --end-date 2024-01-31 --output-dir ./data
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional


class Pipeline:
    """Orchestrate the full vegetation risk intelligence pipeline."""

    def __init__(
        self,
        aoi_geojson: str,
        start_date: str,
        end_date: str,
        output_dir: str,
        transmission_lines: Optional[str] = None,
        tower_locations: Optional[str] = None,
        cloud_cover_max: int = 20,
        ndvi_threshold: float = 0.3,
        corridor_buffer_m: float = 50,
        run_temporal: bool = False,
        days_between: Optional[int] = None,
    ):
        self.aoi_geojson = aoi_geojson
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = Path(output_dir)
        self.transmission_lines = transmission_lines
        self.tower_locations = tower_locations
        self.cloud_cover_max = cloud_cover_max
        self.ndvi_threshold = ndvi_threshold
        self.corridor_buffer_m = corridor_buffer_m
        self.run_temporal = run_temporal
        self.days_between = days_between

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)

    def run_command(self, cmd: List[str], description: str, cwd: Optional[Path] = None) -> bool:
        """Run a command and return success status."""
        print(f"\n{'='*60}")
        print(f"STAGE: {description}")
        print(f"COMMAND: {' '.join(cmd)}")
        print(f"{'='*60}")

        log_file = self.output_dir / "logs" / f"{description.lower().replace(' ', '_')}.log"
        with open(log_file, "w") as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Started: {datetime.utcnow().isoformat()}Z\n\n")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=cwd or Path.cwd(),
                timeout=3600,  # 1 hour timeout per stage
            )

            with open(log_file, "a") as f:
                f.write(f"Exit code: {result.returncode}\n")
                f.write(f"STDOUT:\n{result.stdout}\n")
                f.write(f"STDERR:\n{result.stderr}\n")

            if result.returncode == 0:
                print(f"[OK] {description} completed successfully")
                return True
            else:
                print(f"[FAIL] {description} failed with exit code {result.returncode}")
                print(f"  STDERR: {result.stderr[:500]}")
                return False

        except subprocess.TimeoutExpired:
            print(f"[FAIL] {description} timed out after 1 hour")
            return False
        except Exception as e:
            print(f"[FAIL] {description} failed with error: {e}")
            return False

    def stage1_ingestion(self) -> bool:
        """Stage 1: Sentinel-2 Ingestion."""
        cmd = [
            sys.executable,
            "src/ingestion/sentinel2_ingest.py",
            "--aoi", self.aoi_geojson,
            "--start-date", self.start_date,
            "--end-date", self.end_date,
            "--output-dir", str(self.output_dir / "ingestion"),
            "--cloud-cover-max", str(self.cloud_cover_max),
        ]
        return self.run_command(cmd, "Sentinel-2 Ingestion (Stage 1)")

    def stage1b_landcover(self) -> bool:
        """Stage 1b: ESA WorldCover Land Cover Ingestion."""
        cmd = [
            sys.executable,
            "src/ingestion/landcover_ingest.py",
            "--aoi", self.aoi_geojson,
            "--output-dir", str(self.output_dir / "ingestion"),
            "--year", "2021",
            "--target-crs", "EPSG:4326",
        ]
        return self.run_command(cmd, "WorldCover Land Cover Ingestion (Stage 1b)")

    def stage1c_gedi(self) -> bool:
        """Stage 1c: NASA GEDI Canopy Height Ingestion."""
        # Use start_date year as GEDI year
        year = self.start_date[:4]
        cmd = [
            sys.executable,
            "src/ingestion/gedi_ingest.py",
            "--aoi", self.aoi_geojson,
            "--output-dir", str(self.output_dir / "ingestion"),
            "--year", year,
            "--target-crs", "EPSG:4326",
        ]
        return self.run_command(cmd, "GEDI Canopy Height Ingestion (Stage 1c)")

    def stage1d_dem(self) -> bool:
        """Stage 1d: Copernicus DEM Ingestion."""
        cmd = [
            sys.executable,
            "src/ingestion/dem_ingest.py",
            "--aoi", self.aoi_geojson,
            "--output-dir", str(self.output_dir / "ingestion"),
            "--resolution", "GLO-30",
            "--target-crs", "EPSG:4326",
        ]
        return self.run_command(cmd, "Copernicus DEM Ingestion (Stage 1d)")

    def stage1e_weather(self) -> bool:
        """Stage 1e: NASA POWER Weather/Climate Ingestion."""
        cmd = [
            sys.executable,
            "src/ingestion/weather_ingest.py",
            "--aoi", self.aoi_geojson,
            "--output-dir", str(self.output_dir / "ingestion"),
            "--start-date", self.start_date,
            "--end-date", self.end_date,
            "--target-crs", "EPSG:4326",
        ]
        return self.run_command(cmd, "NASA POWER Weather Ingestion (Stage 1e)")

    def stage2_vegetation(self) -> bool:
        """Stage 2: Vegetation Analysis."""
        ingestion_dir = self.output_dir / "ingestion"
        bands_dir = ingestion_dir / "geotiffs"
        output_dir = self.output_dir / "vegetation"

        # Get list of tiles to process
        meta_files = list((ingestion_dir / "metadata").glob("*_meta.json"))
        if not meta_files:
            print("  No metadata files found from ingestion, skipping vegetation analysis")
            return False

        # Find WorldCover woody mask
        worldcover_files = list(ingestion_dir.glob("worldcover/woody_vegetation_combined.tif"))
        if not worldcover_files:
            worldcover_files = list(ingestion_dir.glob("worldcover/woody_vegetation_tile_*.tif"))
        worldcover_path = str(worldcover_files[0]) if worldcover_files else None

        success_count = 0
        for meta_file in meta_files:
            with open(meta_file) as f:
                meta = json.load(f)
            tile_id = meta["tile_id"]

            cmd = [
                sys.executable,
                "src/analysis/vegetation_analysis.py",
                "--tile-id", tile_id,
                "--bands-dir", str(bands_dir),
                "--output-dir", str(output_dir),
                "--ndvi-threshold", str(self.ndvi_threshold),
            ]
            if worldcover_path:
                cmd.extend(["--worldcover-path", worldcover_path])
            if self.run_command(cmd, f"Vegetation Analysis - {tile_id}"):
                success_count += 1

        return success_count > 0

    def stage3_spatial(self) -> bool:
        """Stage 3: Spatial Analysis."""
        if not self.transmission_lines:
            print("  No transmission lines provided, skipping spatial analysis")
            return False

        veg_mask_dir = self.output_dir / "vegetation"
        mask_files = list(veg_mask_dir.glob("*_vegmask.tif"))
        if not mask_files:
            print("  No vegetation masks found, skipping spatial analysis")
            return False

        output_dir = self.output_dir / "spatial"

        # Find WorldCover woody mask
        ingestion_dir = self.output_dir / "ingestion"
        worldcover_files = list(ingestion_dir.glob("worldcover/woody_vegetation_combined.tif"))
        if not worldcover_files:
            worldcover_files = list(ingestion_dir.glob("worldcover/woody_vegetation_tile_*.tif"))
        worldcover_path = str(worldcover_files[0]) if worldcover_files else None

        # Process each mask file
        success_count = 0
        for mask_file in mask_files:
            tile_id = mask_file.stem.replace("_vegmask", "")
            cmd = [
                sys.executable,
                "src/spatial/spatial_analysis.py",
                "--transmission-lines", self.transmission_lines,
                "--vegetation-mask", str(mask_file),
                "--output-dir", str(output_dir / tile_id),
                "--corridor-buffer-m", str(self.corridor_buffer_m),
            ]
            if self.tower_locations:
                cmd.extend(["--tower-locations", self.tower_locations])
            if worldcover_path:
                cmd.extend(["--worldcover-mask", worldcover_path])

            if self.run_command(cmd, f"Spatial Analysis - {tile_id}"):
                success_count += 1

        return success_count > 0

    def stage4_temporal(self) -> bool:
        """Stage 4: Temporal Analysis (if enabled)."""
        if not self.run_temporal or not self.days_between:
            return True  # Not an error, just not running

        ndvi_dir = self.output_dir / "vegetation"
        ndvi_files = list(ndvi_dir.glob("*_ndvi.tif"))

        if len(ndvi_files) < 2:
            print("  Need at least 2 NDVI rasters for temporal analysis, skipping")
            return False

        # For simplicity, compare first two dates
        ndvi_files.sort()
        early = ndvi_files[0]
        late = ndvi_files[1]

        output_dir = self.output_dir / "temporal"
        tile_id = early.stem.replace("_ndvi", "")

        cmd = [
            sys.executable,
            "src/temporal/temporal_analysis.py",
            "--ndvi-early", str(early),
            "--ndvi-late", str(late),
            "--days-between", str(self.days_between),
            "--output-dir", str(output_dir),
            "--tile-id", tile_id,
        ]
        return self.run_command(cmd, "Temporal Analysis (Stage 4)")

    def stage5_features(self) -> bool:
        """Stage 5: Feature Engineering."""
        spatial_dir = self.output_dir / "spatial"
        segment_files = list(spatial_dir.glob("*/corridor_segments.gpkg"))

        if not segment_files:
            print("  No corridor segments found, skipping feature engineering")
            return False

        veg_patch_files = list(spatial_dir.glob("*/vegetation_patches.gpkg"))
        if not veg_patch_files:
            print("  No vegetation patches found, skipping feature engineering")
            return False

        # Use first tile's data
        segments_path = segment_files[0]
        patches_path = veg_patch_files[0]
        output_dir = self.output_dir / "features"

        temporal_files = list(self.output_dir.glob("temporal/*_temporal_stats.json"))

        # Find auxiliary data sources
        ingestion_dir = self.output_dir / "ingestion"

        # WorldCover woody mask
        worldcover_files = list(ingestion_dir.glob("worldcover/woody_vegetation_combined.tif"))
        if not worldcover_files:
            worldcover_files = list(ingestion_dir.glob("worldcover/woody_vegetation_tile_*.tif"))

        # GEDI canopy height
        gedi_files = list(ingestion_dir.glob("gedi/canopy_height.tif"))

        # DEM elevation
        dem_files = list(ingestion_dir.glob("dem/dem_GLO-30_tile_*.tif"))

        # Weather daily CSV
        weather_csv = ingestion_dir / "weather" / "nasa_power_daily.csv"

        cmd = [
            sys.executable,
            "src/features/feature_engineering.py",
            "--corridor-segments", str(segments_path),
            "--vegetation-patches", str(patches_path),
            "--output-dir", str(output_dir),
        ]
        if temporal_files:
            cmd.extend(["--temporal-stats", str(temporal_files[0])])
        if worldcover_files:
            cmd.extend(["--worldcover-path", str(worldcover_files[0])])
        if gedi_files:
            cmd.extend(["--gedi-path", str(gedi_files[0])])
        if dem_files:
            cmd.extend(["--dem-path", str(dem_files[0])])
        if weather_csv.exists():
            cmd.extend(["--weather-path", str(weather_csv)])

        return self.run_command(cmd, "Feature Engineering (Stage 5)")

    def stage6_risk(self) -> bool:
        """Stage 6: Risk Scoring."""
        features_csv = self.output_dir / "features" / "features.csv"
        if not features_csv.exists():
            print("  No features CSV found, skipping risk scoring")
            return False

        output_dir = self.output_dir / "risk"

        cmd = [
            sys.executable,
            "src/risk/risk_scoring.py",
            "--features-csv", str(features_csv),
            "--output-dir", str(output_dir),
        ]
        return self.run_command(cmd, "Risk Scoring (Stage 6)")

    def copy_results(self):
        """Copy final results to data/ directory for API consumption."""
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)

        # Copy risk scores
        risk_json = self.output_dir / "risk" / "risk_scores.json"
        if risk_json.exists():
            import shutil
            shutil.copy(risk_json, data_dir / "risk_scores.json")
            print(f"  Copied risk scores to {data_dir / 'risk_scores.json'}")

        risk_summary = self.output_dir / "risk" / "risk_summary.json"
        if risk_summary.exists():
            shutil.copy(risk_summary, data_dir / "risk_summary.json")

        # Copy corridor segments
        spatial_dir = self.output_dir / "spatial"
        segment_files = list(spatial_dir.glob("*/corridor_segments.gpkg"))
        if segment_files:
            shutil.copy(segment_files[0], data_dir / "corridor_segments.gpkg")
            print(f"  Copied corridor segments to {data_dir / 'corridor_segments.gpkg'}")

    def run(self) -> bool:
        """Run the full pipeline."""
        print(f"\n{'#'*60}")
        print(f"# VEGETATION RISK INTELLIGENCE PIPELINE")
        print(f"# Started: {datetime.utcnow().isoformat()}Z")
        print(f"# AOI: {self.aoi_geojson}")
        print(f"# Date Range: {self.start_date} to {self.end_date}")
        print(f"# Output: {self.output_dir}")
        print(f"{'#'*60}\n")

        stages = [
            ("Stage 1: Sentinel-2 Ingestion", self.stage1_ingestion),
            ("Stage 1b: WorldCover Land Cover", self.stage1b_landcover),
            ("Stage 1c: GEDI Canopy Height", self.stage1c_gedi),
            ("Stage 1d: Copernicus DEM", self.stage1d_dem),
            ("Stage 1e: NASA POWER Weather", self.stage1e_weather),
            ("Stage 2: Vegetation Analysis", self.stage2_vegetation),
            ("Stage 3: Spatial Analysis", self.stage3_spatial),
        ]

        if self.run_temporal:
            stages.append(("Stage 4: Temporal Analysis", self.stage4_temporal))

        stages.extend([
            ("Stage 5: Feature Engineering", self.stage5_features),
            ("Stage 6: Risk Scoring", self.stage6_risk),
        ])

        results = {}
        for name, stage_fn in stages:
            success = stage_fn()
            results[name] = success
            if not success:
                print(f"\n[WARN] {name} failed — continuing with remaining stages")

        # Copy final results
        self.copy_results()

        # Summary
        print(f"\n{'#'*60}")
        print(f"# PIPELINE SUMMARY")
        print(f"{'#'*60}")
        for name, success in results.items():
            status = "[OK] PASS" if success else "[FAIL] FAIL"
            print(f"  {status} - {name}")

        all_passed = all(results.values())
        print(f"\n{'='*60}")
        if all_passed:
            print("[OK] ALL STAGES PASSED")
        else:
            print("[WARN] SOME STAGES FAILED — check logs")
        print(f"{'='*60}")

        return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Vegetation Risk Intelligence Pipeline - Run all stages"
    )
    parser.add_argument("--aoi", required=True, help="Path to AOI GeoJSON")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--transmission-lines", help="Path to transmission lines GeoJSON")
    parser.add_argument("--tower-locations", help="Path to tower locations GeoJSON")
    parser.add_argument(
        "--cloud-cover-max", type=int, default=20, help="Max cloud cover percent (default: 20)"
    )
    parser.add_argument(
        "--ndvi-threshold", type=float, default=0.3, help="NDVI threshold (default: 0.3)"
    )
    parser.add_argument(
        "--corridor-buffer-m", type=float, default=50, help="Corridor buffer (default: 50m)"
    )
    parser.add_argument(
        "--run-temporal", action="store_true", help="Run temporal analysis"
    )
    parser.add_argument(
        "--days-between", type=int, help="Days between observations for temporal analysis"
    )

    args = parser.parse_args()

    pipeline = Pipeline(
        aoi_geojson=args.aoi,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        transmission_lines=args.transmission_lines,
        tower_locations=args.tower_locations,
        cloud_cover_max=args.cloud_cover_max,
        ndvi_threshold=args.ndvi_threshold,
        corridor_buffer_m=args.corridor_buffer_m,
        run_temporal=args.run_temporal,
        days_between=args.days_between,
    )

    success = pipeline.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()