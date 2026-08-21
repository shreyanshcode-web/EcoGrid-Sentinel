#!/usr/bin/env python3
"""On-demand India-wide Sentinel-2 NDVI acquisition.

India-wide imagery is too large to bundle in this repository.  This module
partitions the India bounding box into resumable AOI cells and reuses the
existing Sentinel-2 STAC ingestor for each cell.  It writes a manifest so an
interrupted download can be safely resumed.
"""

import argparse
import json
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List

import geopandas as gpd
from shapely.geometry import box, mapping

try:
    from ingestion.sentinel2_ingest import Sentinel2Ingestor
except ModuleNotFoundError:  # direct execution: python src/ingestion/india_ndvi_ingest.py
    from sentinel2_ingest import Sentinel2Ingestor


INDIA_BOUNDS = (68.0, 6.0, 97.5, 37.5)


def india_grid(cell_size_degrees: float = 2.5) -> List[Dict]:
    """Return regular WGS84 cells covering India's broad bounding box."""
    if cell_size_degrees <= 0:
        raise ValueError("cell_size_degrees must be greater than zero")
    west, south, east, north = INDIA_BOUNDS
    cells = []
    row = 0
    y = south
    while y < north:
        column = 0
        x = west
        while x < east:
            cell = box(x, y, min(x + cell_size_degrees, east), min(y + cell_size_degrees, north))
            cells.append({"cell_id": f"india_{row:02d}_{column:02d}", "geometry": mapping(cell)})
            x += cell_size_degrees
            column += 1
        y += cell_size_degrees
        row += 1
    return cells


def acquire_india_ndvi(
    start_date: str,
    end_date: str,
    output_dir: str,
    cell_size_degrees: float = 2.5,
    cloud_cover_max: int = 20,
) -> Dict:
    """Acquire Sentinel-2 bands for all grid cells and record provenance.

    The existing vegetation stage computes NDVI from the downloaded B04/B08
    bands.  This function deliberately does not hide failed cells: failures
    are recorded in the manifest and successful cells remain reusable.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "india_ndvi_manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    results = previous.get("cells", {})

    for cell in india_grid(cell_size_degrees):
        cell_id = cell["cell_id"]
        if results.get(cell_id, {}).get("status") == "complete":
            continue
        with tempfile.TemporaryDirectory(prefix=f"{cell_id}_") as temp_dir:
            aoi_path = Path(temp_dir) / "aoi.geojson"
            gpd.GeoDataFrame(
                [{"cell_id": cell_id, "geometry": cell["geometry"]}],
                crs="EPSG:4326",
            ).to_file(aoi_path, driver="GeoJSON")
            cell_output = output / "cells" / cell_id
            try:
                metadata = Sentinel2Ingestor(
                    str(aoi_path), start_date, end_date, str(cell_output), cloud_cover_max
                ).run()
                results[cell_id] = {"status": "complete", "items_processed": len(metadata)}
            except Exception as exc:  # keep other cells resumable
                results[cell_id] = {"status": "failed", "error": str(exc)}
        manifest = {
            "coverage": "India bounding box (68E-97.5E, 6N-37.5N)",
            "date_range": f"{start_date}/{end_date}",
            "cell_size_degrees": cell_size_degrees,
            "cloud_cover_max": cloud_cover_max,
            "source": "Copernicus Sentinel-2 L2A via STAC",
            "next_step": "Run vegetation_analysis.py for each B04/B08 pair to produce NDVI GeoTIFFs.",
            "cells": results,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return json.loads(manifest_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire resumable India-wide Sentinel-2 NDVI source tiles")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cell-size-degrees", type=float, default=2.5)
    parser.add_argument("--cloud-cover-max", type=int, default=20)
    args = parser.parse_args()
    acquire_india_ndvi(**vars(args))


if __name__ == "__main__":
    main()