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


# =============================================================================
# Helper Functions
# =============================================================================


def load_hotspots():
    """Load hotspot data from disk."""
    if _cache["hotspots"] is not None:
        return _cache["hotspots"]

    if not RISK_SCORES_PATH.exists():
        return []

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
        return {
            "total_segments": 0,
            "high_risk_count": 0,
            "medium_risk_count": 0,
            "low_risk_count": 0,
            "mean_risk_score": 0,
            "max_risk_score": 0,
            "weights": {},
            "thresholds": {},
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