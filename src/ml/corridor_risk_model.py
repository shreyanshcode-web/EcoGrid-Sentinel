#!/usr/bin/env python3
"""Apply the India inspection-priority model to NDVI near transmission lines."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.geometry import LineString, Point, mapping
from shapely.ops import transform
import pyproj

try:
    from ml.train_india_model import FEATURES
except ModuleNotFoundError:  # direct execution: python src/ml/corridor_risk_model.py
    from train_india_model import FEATURES


def _line_segments(lines: gpd.GeoDataFrame, segment_length_m: float) -> gpd.GeoDataFrame:
    work_crs = "EPSG:3857"
    projected = lines.to_crs(work_crs)
    records = []
    segment_id = 0
    for line_index, geom in enumerate(projected.geometry):
        if geom is None or geom.is_empty:
            continue
        parts = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
        for part in parts:
            for start in np.arange(0, part.length, segment_length_m):
                end = min(float(start + segment_length_m), part.length)
                if end <= start:
                    continue
                records.append({"segment_id": segment_id, "line_idx": line_index,
                                "geometry": LineString([part.interpolate(start), part.interpolate(end)])})
                segment_id += 1
    return gpd.GeoDataFrame(records, crs=work_crs)


def score_corridors(
    lines_path: str,
    ndvi_path: str,
    model_path: str,
    output_path: str,
    corridor_buffer_m: float = 50.0,
    segment_length_m: float = 500.0,
    ndvi_threshold: float = 0.3,
) -> Dict:
    """Score line segments using every valid NDVI pixel in its corridor buffer."""
    lines = gpd.read_file(lines_path)
    segments = _line_segments(lines, segment_length_m)
    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_names = artifact.get("features", FEATURES)
    results = []
    with rasterio.open(ndvi_path) as src:
        to_raster = pyproj.Transformer.from_crs("EPSG:3857", src.crs, always_xy=True).transform
        to_work = pyproj.Transformer.from_crs(src.crs, "EPSG:3857", always_xy=True).transform
        to_wgs84 = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform
        for row in segments.itertuples():
            buffered = row.geometry.buffer(corridor_buffer_m)
            raster_geom = transform(to_raster, buffered)
            try:
                values, out_transform = mask(src, [mapping(raster_geom)], crop=True, filled=False)
            except ValueError:
                continue
            ndvi = values[0].compressed().astype(float)
            if ndvi.size == 0:
                continue
            # Pixel centers are measured from the actual line in metres.
            rows, cols = np.where(~values[0].mask)
            xs, ys = rasterio.transform.xy(out_transform, rows, cols)
            distances = np.asarray([row.geometry.distance(transform(to_work, Point(x, y))) for x, y in zip(xs, ys)])
            features = pd.DataFrame({
                "DistanceToLine": distances,
                "NDVI": ndvi,
                "ProximityFactor": np.exp(-distances / max(corridor_buffer_m, 1.0)),
                "Vegetation": (ndvi >= ndvi_threshold).astype(float),
                "VegetationFactor": np.clip(ndvi, 0, 1),
            })[feature_names]
            probabilities = model.predict_proba(features)
            classes = list(model.classes_)
            high_probability = probabilities[:, classes.index(2)] if 2 in classes else np.zeros(len(features))
            risk_score = float(np.mean(high_probability))
            category = "High" if risk_score >= 0.55 else "Medium" if risk_score >= 0.35 else "Low"
            results.append({
                "segment_id": int(row.segment_id), "line_idx": int(row.line_idx),
                "risk_score": risk_score, "risk_category": category,
                "mean_ndvi": float(np.mean(ndvi)), "vegetation_fraction": float(np.mean(ndvi >= ndvi_threshold)),
                "pixels_sampled": int(len(ndvi)), "geometry": mapping(transform(to_wgs84, row.geometry)),
            })
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "id": item["segment_id"], "properties": {k: v for k, v in item.items() if k != "geometry"}, "geometry": item["geometry"]}
        for item in results
    ], "metadata": {"model": Path(model_path).name, "ndvi": Path(ndvi_path).name}}, indent=2), encoding="utf-8")
    return {"segments_scored": len(results), "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Find ML-prioritized vegetation risk near Indian transmission lines")
    parser.add_argument("--transmission-lines", required=True)
    parser.add_argument("--ndvi-raster", required=True)
    parser.add_argument("--model", required=True, help="india_risk_model.joblib")
    parser.add_argument("--output", required=True)
    parser.add_argument("--corridor-buffer-m", type=float, default=50.0)
    parser.add_argument("--segment-length-m", type=float, default=500.0)
    parser.add_argument("--ndvi-threshold", type=float, default=0.3)
    args = parser.parse_args()
    print(json.dumps(score_corridors(args.transmission_lines, args.ndvi_raster, args.model, args.output,
                                      args.corridor_buffer_m, args.segment_length_m, args.ndvi_threshold), indent=2))


if __name__ == "__main__":
    main()