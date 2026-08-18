# SIH1379 — AI-Powered Transmission Corridor Vegetation Risk Intelligence System

A prototype system that ingests Sentinel-2 imagery + transmission infrastructure geometry, computes vegetation risk near power line corridors, and serves it via API + dashboard.

> **Status**: Hackathon-grade prototype (not production-ready). See [Known Limitations](#known-limitations) below.

---

## Architecture Overview

```
Sentinel-2 (Planetary Computer STAC)
    │
    ▼
┌─────────────────────────────────────────┐
│ Stage 1: Data Ingestion                 │
│ - Search S2 L2A by AOI + date range     │
│ - Cloud filtering (scene-level)         │
│ - Download B2,B3,B4,B8,SCL bands        │
│ - Clip to AOI, reproject to EPSG:4326   │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Stage 2: Vegetation Analysis            │
│ - NDVI = (B8-B4)/(B8+B4)                │
│ - Threshold mask (default 0.3)          │
│ - Morphological open/close denoising    │
│ - U-Net stub for v2                     │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Stage 3: Spatial Analysis               │
│ - Load transmission lines + towers      │
│ - Distance to nearest line segment      │
│ - Distance to corridor edge             │
│ - Tower proximity / inside corridor     │
│ - STRtree spatial index for performance │
└─────────────────────────────────────────┘
    │
    ▼ (optional)
┌─────────────────────────────────────────┐
│ Stage 4: Temporal Analysis              │
│ - NDVI change between two dates         │
│ - Linear growth rate (% change / days)  │
│ - Rolling/seasonal trends → v2          │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Stage 5: Feature Engineering            │
│ - Per-corridor-segment feature table    │
│ - Vegetation fraction, mean NDVI        │
│ - Spatial + temporal features           │
│ - Output: CSV + GeoPackage              │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Stage 6: Risk Scoring (Heuristic v1)    │
│ - Weighted sum:                         │
│   risk = w1*proximity + w2*density      │
│        + w3*growth + w4*condition       │
│ - Normalized 0-1, bucketed Low/Med/High │
│ - Explainable breakdown per component   │
│ - Supervised ML extension point         │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Stage 7: API Layer (FastAPI)            │
│ - GET /hotspots → GeoJSON               │
│ - GET /hotspots/{id} → full detail      │
│ - GET /ndvi-layer → tiles               │
│ - No auth (hackathon demo)              │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Stage 8: Dashboard (React + Leaflet)    │
│ - Map with color-coded risk markers     │
│ - Side panel: score breakdown, NDVI     │
│ - Sortable top-priorities table         │
└─────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+ (for dashboard)
- Planetary Computer access (free, no auth for STAC search)

### Installation

```bash
# Python dependencies
pip install -r requirements.txt

# Dashboard dependencies
cd dashboard && npm install && cd ..
```

### Run Pipeline

```bash
# Full pipeline (ingestion → risk scoring)
python src/pipeline.py \
  --aoi data/sample_aoi.geojson \
  --start-date 2024-01-01 \
  --end-date 2024-01-31 \
  --output-dir ./data/output \
  --transmission-lines data/sample_transmission_lines.geojson \
  --tower-locations data/sample_towers.geojson

# With temporal analysis (needs multi-date imagery)
python src/pipeline.py \
  --aoi data/sample_aoi.geojson \
  --start-date 2024-01-01 \
  --end-date 2024-02-28 \
  --output-dir ./data/output \
  --transmission-lines data/sample_transmission_lines.geojson \
  --run-temporal \
  --days-between 30
```

### Run API Server

```bash
# After pipeline completes, results copied to ./data/
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Run Dashboard

```bash
cd dashboard && npm start
# Opens http://localhost:3000
```

---

## Project Structure

```
SIH/
├── src/
│   ├── ingestion/          # Stage 1: Sentinel-2 ingestion
│   │   └── sentinel2_ingest.py
│   ├── analysis/           # Stage 2: NDVI + vegetation mask
│   │   └── vegetation_analysis.py
│   ├── spatial/            # Stage 3: Distance features
│   │   └── spatial_analysis.py
│   ├── temporal/           # Stage 4: NDVI change / growth
│   │   └── temporal_analysis.py
│   ├── features/           # Stage 5: Feature table
│   │   └── feature_engineering.py
│   ├── risk/               # Stage 6: Heuristic risk scorer
│   │   └── risk_scoring.py
│   ├── api/                # Stage 7: FastAPI endpoints
│   │   └── main.py
│   └── pipeline.py         # Master orchestrator
├── dashboard/              # Stage 8: React + Leaflet
│   ├── src/
│   │   ├── App.js
│   │   ├── index.js
│   │   └── index.css
│   └── package.json
├── data/
│   ├── sample_aoi.geojson
│   ├── sample_transmission_lines.geojson
│   └── sample_towers.geojson
├── requirements.txt
└── README.md
```

---

## Known Limitations

These are explicitly documented design constraints for v1 (hackathon prototype):

### 1. **10m Resolution Limit** ⚠️
> Sentinel-2 is 10m resolution — **cannot resolve individual trees near conductors**.
> Risk scores are **corridor-segment-level**, not tree-level.
> *Documented in code comments throughout pipeline.*

### 2. **No Canopy Height Data** ⚠️
> NDVI/vegetation fraction is a **proxy** for risk, not a direct measurement of fall/contact risk.
> *Explicitly stated in risk score explanations shown to users.*

### 3. **No Historical Ground Truth** ⚠️
> Risk Model 2 is built as an **EXPLICIT WEIGHTED HEURISTIC SCORER** (transparent, tunable weights),
> NOT framed as trained supervised ML.
> A clear extension point exists for supervised training when labeled incident data becomes available.
> *See `src/risk/risk_scoring.py` — weights are named constants with rationale comments.*

### 4. **Land Cover Not Guaranteed** ⚠️
> When land cover data is unavailable, the system **flags** when NDVI-high areas might be
> cropland vs. woody vegetation.
> *Vegetation class map distinguishes sparse/moderate/dense but cannot differentiate crop types.*

### 5. **Cloud Filtering (Scene-Level Only)** ⚠️
> Uses scene-level cloud cover metadata (<20% default). No per-pixel cloud masking for v1.
> SCL band available but not fully utilized.

### 6. **Security: No Auth/RBAC** ⚠️
> API has **no authentication** for hackathon demo.
> **TODO comment in `src/api/main.py`** flags this as a gap for any real deployment
> (infrastructure data sensitivity).

---

## Configuration

Key parameters (in `src/pipeline.py` or via CLI):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--cloud-cover-max` | 20% | Max scene cloud cover |
| `--ndvi-threshold` | 0.3 | NDVI threshold for vegetation |
| `--corridor-buffer-m` | 50m | Corridor half-width |
| `--days-between` | N/A | Days between observations (temporal) |

Risk weights (in `src/risk/risk_scoring.py`):
```python
WEIGHTS = {
    "proximity": 0.40,   # Distance to line
    "density": 0.25,     # Vegetation fraction
    "growth": 0.20,      # Growth rate
    "condition": 0.15,   # Vegetation health (NDVI)
}
THRESHOLDS = {"high": 0.7, "medium": 0.4}
```

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | API info |
| `GET /hotspots` | GeoJSON FeatureCollection of risk segments |
| `GET /hotspots/{id}` | Full detail with breakdown |
| `GET /summary` | Risk statistics |
| `GET /ndvi-layer` | NDVI tile metadata |

Example `/hotspots` response:
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "id": 0,
      "properties": {
        "segment_id": 0,
        "risk_score": 0.82,
        "risk_category": "High",
        "vegetation_fraction": 0.45,
        "mean_dist_to_line_m": 12.3
      },
      "geometry": { "type": "LineString", "coordinates": [...] }
    }
  ],
  "metadata": { "generated_at": "...", "count": 42, "summary": {...} }
}
```

---

## Future Work (v2+)

- [ ] **U-Net segmentation** for vegetation (replace threshold)
- [ ] **LiDAR canopy height** integration
- [ ] **Supervised risk model** with outage incident labels
- [ ] **Land cover classification** (ESA WorldCover / Dynamic World)
- [ ] **Rolling temporal stats** & seasonal trend modeling
- [ ] **Auth/RBAC** on API
- [ ] **PostGIS backend** for production-scale queries
- [ ] **LSTM/Transformer** for growth forecasting

---

## License

MIT License — Hackathon prototype for SIH1379.