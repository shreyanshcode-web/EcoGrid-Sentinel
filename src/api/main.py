#!/usr/bin/env python3
"""
API Layer (Stage 7)

FastAPI endpoints for serving vegetation risk intelligence:
- GET /hotspots — GeoJSON of risk-scored corridor segments
- GET /hotspots/{id} — Full detail (score breakdown, NDVI images, growth)
- GET /ndvi-layer — Tile/image serving for dashboard

TODO (Security Gap): No auth/RBAC for hackathon demo. In production,
infrastructure data sensitivity requires proper authentication.
"""

import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling


app = FastAPI(
    title="Vegetation Risk Intelligence API",
    description="Transmission corridor vegetation risk intelligence system",
    version="0.1.0",
)

# CORS for dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for hackathon demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path("./data")
RISK_SCORES_PATH = DATA_DIR / "risk_scores.json"
RISK_SUMMARY_PATH = DATA_DIR / "risk_summary.json"
TRAINING_DATASET_PATH = DATA_DIR / "SIH1379_ML_Training_Dataset.csv"
TRANSMISSION_LINES_PATH = DATA_DIR / "GatiShakti_Transmission_Lines_220kV_plus.geojson"
INDIA_MODEL_PATH = DATA_DIR / "models" / "india_risk_model.joblib"
INDIA_MODEL_METRICS_PATH = DATA_DIR / "models" / "india_risk_model_metrics.json"
INDIA_CORRIDOR_RISK_PATH = DATA_DIR / "india_corridor_risk.geojson"
HOTSPOTS_GPKG_PATH = DATA_DIR / "corridor_segments.gpkg"
NDVI_DIR = DATA_DIR / "ndvi"
VEG_MASK_DIR = DATA_DIR / "vegetation_masks"

# Cache for loaded data
_cache = {
    "hotspots": None,
    "hotspots_gdf": None,
    "risk_summary": None,
}


# =============================================================================
# Data Models
# =============================================================================


class RiskComponent(BaseModel):
    score: float
    weight: float
    weighted: float
    contribution_pct: float
    interpretation: str


class RiskBreakdown(BaseModel):
    total_score: float
    risk_category: str
    components: Dict[str, RiskComponent]
    key_factors: List[str]
    recommended_action: str


class Hotspot(BaseModel):
    id: int
    segment_id: int
    risk_score: float
    risk_category: str
    geometry: Dict  # GeoJSON geometry
    vegetation_fraction: float
    mean_dist_to_line_m: float
    breakdown: Optional[RiskBreakdown] = None


class RiskSummary(BaseModel):
    total_segments: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    mean_risk_score: float
    max_risk_score: float
    weights: Dict[str, float]
    thresholds: Dict[str, float]


class IndiaRiskPredictionRequest(BaseModel):
    DistanceToLine: float
    NDVI: float
    ProximityFactor: float
    Vegetation: float
    VegetationFactor: float


# =============================================================================
# Helper Functions
# =============================================================================


def load_hotspots():
    """Load hotspot data from disk."""
    if _cache["hotspots"] is not None:
        return _cache["hotspots"]

    if not RISK_SCORES_PATH.exists():
        # A runnable, data-backed fallback for first-time setup.  The project
        # ships labelled point observations, so the dashboard can be used
        # before the optional Sentinel/GDAL processing pipeline has produced
        # risk_scores.json.  Generated pipeline results always take priority.
        if not TRAINING_DATASET_PATH.exists():
            return []
        hotspots = []
        with open(TRAINING_DATASET_PATH, newline="", encoding="utf-8-sig") as f:
            for index, row in enumerate(csv.DictReader(f), start=1):
                try:
                    score = float(row["RiskScore"])
                    label = int(float(row["RiskLabel"]))
                    longitude = float(row["longitude"])
                    latitude = float(row["latitude"])
                except (KeyError, TypeError, ValueError):
                    continue
                category = "High" if label >= 2 or score >= 0.55 else "Medium" if label >= 1 or score >= 0.35 else "Low"
                ndvi = float(row.get("NDVI") or 0)
                vegetation = float(row.get("Vegetation") or 0)
                hotspots.append({
                    "id": index,
                    "segment_id": index,
                    "risk_score": score,
                    "risk_category": category,
                    "vegetation_fraction": vegetation,
                    "mean_dist_to_line_m": float(row.get("DistanceToLine") or 0),
                    "ndvi": ndvi,
                    "geometry": {"type": "Point", "coordinates": [longitude, latitude]},
                    "breakdown": {
                        "total_score": score,
                        "risk_category": category,
                        "components": {},
                        "key_factors": ["Risk score from SIH1379 labelled training dataset"],
                        "recommended_action": "Review in the field-inspection workflow.",
                    },
                })
        _cache["hotspots"] = hotspots
        return hotspots

    with open(RISK_SCORES_PATH) as f:
        hotspots = json.load(f)

    _cache["hotspots"] = hotspots
    return hotspots


def load_hotspots_gdf():
    """Load hotspot GeoDataFrame with geometry."""
    if _cache["hotspots_gdf"] is not None:
        return _cache["hotspots_gdf"]

    if not HOTSPOTS_GPKG_PATH.exists():
        return None

    gdf = gpd.read_file(HOTSPOTS_GPKG_PATH)
    _cache["hotspots_gdf"] = gdf
    return gdf


def load_risk_summary() -> Dict:
    """Load risk summary statistics."""
    if _cache["risk_summary"] is not None:
        return _cache["risk_summary"]

    if not RISK_SUMMARY_PATH.exists():
        hotspots = load_hotspots()
        scores = [h.get("risk_score", 0) for h in hotspots]
        return {
            "total_segments": len(hotspots),
            "high_risk_count": sum(h.get("risk_category") == "High" for h in hotspots),
            "medium_risk_count": sum(h.get("risk_category") == "Medium" for h in hotspots),
            "low_risk_count": sum(h.get("risk_category") == "Low" for h in hotspots),
            "mean_risk_score": sum(scores) / len(scores) if scores else 0,
            "max_risk_score": max(scores) if scores else 0,
            "weights": {},
            "thresholds": {"high": 0.55, "medium": 0.35},
        }

    with open(RISK_SUMMARY_PATH) as f:
        summary = json.load(f)

    _cache["risk_summary"] = summary
    return summary


def create_geojson_response(hotspots: List[Dict]) -> Dict:
    """Convert hotspots list to GeoJSON FeatureCollection."""
    features = []
    for h in hotspots:
        if "geometry" in h and h["geometry"]:
            feature = {
                "type": "Feature",
                "id": h.get("id", h.get("segment_id")),
                "properties": {
                    "segment_id": h.get("segment_id"),
                    "risk_score": h.get("risk_score"),
                    "risk_category": h.get("risk_category"),
                    "vegetation_fraction": h.get("vegetation_fraction", 0),
                    "mean_dist_to_line_m": h.get("mean_dist_to_line_m", 0),
                },
                "geometry": h["geometry"] if isinstance(h["geometry"], dict) else None,
            }
            features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "count": len(features),
            "summary": load_risk_summary(),
        },
    }


# =============================================================================
# API Endpoints
# =============================================================================


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "Vegetation Risk Intelligence API",
        "version": "0.1.0",
        "endpoints": {
            "/hotspots": "GeoJSON of risk-scored corridor segments",
            "/hotspots/{id}": "Full detail for a specific hotspot",
            "/ndvi-layer": "NDVI imagery tiles",
            "/summary": "Risk summary statistics",
            "/model/status": "India-specific ML model metadata",
            "/model/predict": "India-specific ML risk-label prediction",
            "/india-corridor-risk": "ML-scored NDVI risk along Indian transmission lines",
        },
    }


@app.get("/hotspots")
async def get_hotspots(
    risk_category: Optional[str] = Query(None, description="Filter by risk category (High/Medium/Low)"),
    min_risk_score: float = Query(0.0, description="Minimum risk score (0-1)"),
    limit: int = Query(1000, description="Maximum number of results"),
) -> JSONResponse:
    """
    Get GeoJSON of risk-scored corridor segments.

    Supports filtering by risk category and minimum risk score.
    """
    hotspots = load_hotspots()
    if not hotspots:
        raise HTTPException(status_code=404, detail="No hotspot data available")

    # Apply filters
    filtered = hotspots
    if risk_category:
        filtered = [h for h in filtered if h.get("risk_category") == risk_category]
    if min_risk_score > 0:
        filtered = [h for h in filtered if h.get("risk_score", 0) >= min_risk_score]

    # Sort by risk score (highest first)
    filtered = sorted(filtered, key=lambda x: x.get("risk_score", 0), reverse=True)

    # Apply limit
    filtered = filtered[:limit]

    # Merge with geometry data if available
    gdf = load_hotspots_gdf()
    if gdf is not None:
        for h in filtered:
            seg_id = h.get("segment_id")
            match = gdf[gdf["segment_id"] == seg_id] if "segment_id" in gdf.columns else None
            if match is not None and not match.empty:
                geom = match.geometry.iloc[0]
                if geom is not None:
                    h["geometry"] = json.loads(gpd.GeoSeries([geom], crs=gdf.crs).to_json())["features"][0]["geometry"]

    geojson = create_geojson_response(filtered)
    return JSONResponse(content=geojson)


@app.get("/hotspots/{hotspot_id}")
async def get_hotspot_detail(hotspot_id: int):
    """
    Get full detail for a specific hotspot, including risk breakdown.
    """
    hotspots = load_hotspots()
    hotspot = next((h for h in hotspots if h.get("segment_id") == hotspot_id), None)

    if hotspot is None:
        # Try by index
        if 0 <= hotspot_id < len(hotspots):
            hotspot = hotspots[hotspot_id]
        else:
            raise HTTPException(status_code=404, detail=f"Hotspot {hotspot_id} not found")

    # Get geometry from GeoPackage
    gdf = load_hotspots_gdf()
    if gdf is not None:
        match = gdf[gdf["segment_id"] == hotspot.get("segment_id")] if "segment_id" in gdf.columns else None
        if match is not None and not match.empty:
            geom = match.geometry.iloc[0]
            if geom is not None:
                hotspot["geometry"] = json.loads(
                    gpd.GeoSeries([geom], crs=gdf.crs).to_json()
                )["features"][0]["geometry"]

    # Include full breakdown
    return JSONResponse(content=hotspot)


@app.get("/summary")
async def get_summary():
    """Get risk summary statistics."""
    summary = load_risk_summary()
    return JSONResponse(content=summary)


@app.get("/transmission-lines")
async def get_transmission_lines():
    """Serve the full local PM GatiShakti 220 kV+ transmission-line layer."""
    if not TRANSMISSION_LINES_PATH.exists():
        raise HTTPException(status_code=404, detail="Transmission-line GeoJSON is not available")
    with open(TRANSMISSION_LINES_PATH, encoding="utf-8") as f:
        return JSONResponse(content=json.load(f))


@app.get("/model/status")
async def get_india_model_status():
    """Return metadata for the locally trained SIH1379 India risk model."""
    if not INDIA_MODEL_PATH.exists() or not INDIA_MODEL_METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="India ML model has not been trained yet")
    with open(INDIA_MODEL_METRICS_PATH, encoding="utf-8") as f:
        metrics = json.load(f)
    return JSONResponse(content={"ready": True, "model": INDIA_MODEL_PATH.name, "metrics": metrics})


@app.post("/model/predict")
async def predict_india_risk(request: IndiaRiskPredictionRequest):
    """Predict the training-dataset risk label for one Indian observation."""
    if not INDIA_MODEL_PATH.exists():
        raise HTTPException(status_code=404, detail="India ML model has not been trained yet")
    import joblib
    import pandas as pd
    artifact = joblib.load(INDIA_MODEL_PATH)
    features = artifact["features"]
    model = artifact["model"]
    row = pd.DataFrame([{feature: getattr(request, feature) for feature in features}])
    prediction = int(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]
    return JSONResponse(content={
        "risk_label": prediction,
        "risk_category": {0: "Low", 1: "Medium", 2: "High"}.get(prediction, "Unknown"),
        "probabilities": {str(label): float(probability) for label, probability in zip(model.classes_, probabilities)},
        "model_scope": "SIH1379 Indian training dataset inspection-priority prototype",
    })


@app.get("/india-corridor-risk")
async def get_india_corridor_risk():
    """Serve the optional all-India corridor ML inference GeoJSON."""
    if not INDIA_CORRIDOR_RISK_PATH.exists():
        raise HTTPException(status_code=404, detail="India corridor risk has not been generated yet")
    with open(INDIA_CORRIDOR_RISK_PATH, encoding="utf-8") as f:
        return JSONResponse(content=json.load(f))


@app.get("/ndvi-layer")
async def get_ndvi_layer(
    tile_id: Optional[str] = Query(None, description="Specific tile ID"),
    band: str = Query("ndvi", description="Band to serve (ndvi, vegmask)"),
):
    """
    Serve NDVI imagery or vegetation masks.

    Returns tile metadata and file paths for dashboard rendering.
    """
    ndvi_dir = NDVI_DIR if band == "ndvi" else VEG_MASK_DIR

    if not ndvi_dir.exists():
        raise HTTPException(status_code=404, detail="NDVI data directory not found")

    # Find matching files
    if tile_id:
        files = list(ndvi_dir.glob(f"*{tile_id}*.{band}*.tif"))
    else:
        files = list(ndvi_dir.glob(f"*.{band}*.tif"))

    if not files:
        raise HTTPException(status_code=404, detail="No NDVI tiles found")

    # Return file metadata
    tiles = []
    for f in files[:10]:  # Limit to 10 tiles
        with rasterio.open(f) as src:
            bounds = src.bounds
            transform = src.transform
            tiles.append({
                "filename": f.name,
                "path": str(f),
                "bounds": {
                    "left": bounds.left,
                    "bottom": bounds.bottom,
                    "right": bounds.right,
                    "top": bounds.top,
                },
                "shape": {"height": src.height, "width": src.width},
                "crs": str(src.crs),
            })

    return JSONResponse(content={"tiles": tiles, "band": band})


@app.get("/hotspots/{hotspot_id}/ndvi")
async def get_hotspot_ndvi(hotspot_id: int):
    """
    Get NDVI image for a specific hotspot.
    """
    hotspots = load_hotspots()
    hotspot = next((h for h in hotspots if h.get("segment_id") == hotspot_id), None)

    if hotspot is None:
        raise HTTPException(status_code=404, detail=f"Hotspot {hotspot_id} not found")

    # Look for NDVI files for this segment
    # This is a placeholder — actual implementation would find the relevant NDVI tile
    return JSONResponse(
        content={
            "segment_id": hotspot_id,
            "ndvi_files": [],  # Would contain NDVI tile paths
            "message": "NDVI visualization would be served here",
        }
    )


# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
