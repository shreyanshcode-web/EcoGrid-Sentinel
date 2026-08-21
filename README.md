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

### India data prototype (deployment-ready demo)

The checked-in `data/` directory is the prototype's data source:

- `GatiShakti_Transmission_Lines_220kV_plus.geojson` — 379 Indian 220 kV+ line features.
- `SIH1379_ML_Training_Dataset.csv` — 1,659 labelled Indian observations used by the API and ML training.

Train the reproducible India-specific inspection-priority classifier:

```bash
python src/ml/train_india_model.py \
  --dataset data/SIH1379_ML_Training_Dataset.csv \
  --output-dir data/models
```

This writes `data/models/india_risk_model.joblib` and a held-out evaluation
report at `data/models/india_risk_model_metrics.json`. Start the API and build
the dashboard for deployment:

```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
cd dashboard && npm run build
```

Deploy `dashboard/build` as a static site and configure its API proxy/base URL
to the deployed FastAPI URL. The API exposes `/hotspots`, `/summary`,
`/model/status`, and `/model/predict`.

> The model is trained on the supplied SIH1379 labels. Its rare high-risk class
> has only five examples, so its output is appropriate for prototype inspection
> prioritization, not autonomous operational decisions.

### India-wide NDVI and corridor ML scoring

India-wide Sentinel-2 imagery is acquired on demand; it is not bundled in this
repository because the coverage is large. The resumable grid downloader records
successful and failed cells in `india_ndvi_manifest.json`:

```bash
python src/ingestion/india_ndvi_ingest.py \
  --start-date 2025-01-01 --end-date 2025-01-31 \
  --output-dir data/india_ndvi --cell-size-degrees 2.5
```

Run the existing vegetation analysis for each downloaded B04/B08 pair to create
NDVI GeoTIFFs, then score the line corridors with the trained India model:

```bash
python src/ml/corridor_risk_model.py \
  --transmission-lines data/GatiShakti_Transmission_Lines_220kV_plus.geojson \
  --ndvi-raster data/india_ndvi/ndvi/india_latest_ndvi.tif \
  --model data/models/india_risk_model.joblib \
  --output data/india_corridor_risk.geojson
```

The output is segment-level inspection prioritization and is available through
`GET /india-corridor-risk`. It does not identify individual trees or prove
conductor contact. Sentinel-2 NDVI is a vegetation proxy, and the checked-in
classifier is trained on weak inspection labels rather than outage ground truth.

## Detailed Setup and Run Guide

This section describes the recommended workflow from a fresh Windows checkout.
Run all commands from the repository root (`SIH`). Keep the API and dashboard
running in separate terminal windows.

### 1. Check the required software

Install the following before starting:

- Python 3.10 or newer
- Node.js 18 or newer and npm
- Git, if cloning the repository
- Internet access for Sentinel-2/WorldCover STAC downloads and map tiles

Check the installed versions:

```powershell
python --version
node --version
npm --version
```

### 2. Create and activate the Python environment

PowerShell commands:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the Python packages:

```powershell
python -m pip install -r requirements.txt
```

The geospatial packages (`geopandas`, `rasterio`, `fiona`, `pyproj`, and
`shapely`) must install successfully because the API, spatial analysis, and
NDVI workflow use them. If a Python version does not have compatible wheels,
use Python 3.11 or 3.12 and recreate `.venv` rather than mixing packages from
different interpreters.

### 3. Install the dashboard dependencies

Open a second terminal at the repository root and run:

```powershell
Push-Location dashboard
npm install
Pop-Location
```

The dashboard is a React application using Leaflet. It reads API data when the
API is available and falls back to the static files in
`dashboard/public/data/` when it is not.

### 4. Train the included India model

The repository includes the SIH1379 labelled observations and the PM GatiShakti
220 kV+ India transmission-line layer. Train the reproducible inspection-
priority classifier with:

```powershell
python src/ml/train_india_model.py `
  --dataset data/SIH1379_ML_Training_Dataset.csv `
  --output-dir data/models
```

Expected outputs:

- `data/models/india_risk_model.joblib`
- `data/models/india_risk_model_metrics.json`

The classifier predicts an inspection-priority label from distance, NDVI, and
vegetation factors. It is not an outage predictor and should not be used for
automatic switching or maintenance decisions.

### 5. Start the FastAPI backend

From the repository root, with `.venv` activated:

```powershell
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Verify the backend in a browser:

- API information: `http://127.0.0.1:8000/`
- Swagger UI: `http://127.0.0.1:8000/docs`
- Transmission lines: `http://127.0.0.1:8000/transmission-lines`
- Risk summary: `http://127.0.0.1:8000/summary`
- Model status: `http://127.0.0.1:8000/model/status`

The backend expects to be started from the repository root because its default
data paths are relative to `./data`.

### 6. Start the dashboard

In another terminal:

```powershell
Push-Location dashboard
npm start
```

Open `http://localhost:3000`. The dashboard proxy forwards API requests to
`http://localhost:8000` using the proxy setting in `dashboard/package.json`.

If the Create React App development server fails during startup because of a
webpack-dev-server `allowedHosts` error, build and serve the verified static
bundle instead:

```powershell
Push-Location dashboard
npm run build
Pop-Location
python -m http.server 3000 --directory dashboard/build
```

Then open `http://127.0.0.1:3000`. The static server does not provide the
development proxy; use the API URL configuration appropriate for your deployed
environment if live API data is required.

### 7. Run the sample end-to-end pipeline

Use a small AOI first. This downloads external imagery and may take time:

```powershell
python src/pipeline.py `
  --aoi data/sample_aoi.geojson `
  --start-date 2024-01-01 `
  --end-date 2024-01-31 `
  --output-dir data/output `
  --transmission-lines data/sample_transmission_lines.geojson `
  --tower-locations data/sample_towers.geojson
```

Pipeline results are written under `data/output/`. The final risk JSON and
summary are copied to `data/` for API consumption. Check stage logs under
`data/output/logs/` if a stage fails; optional data sources may fail while the
core stages continue.

### 8. Acquire India-wide NDVI on demand

India-wide imagery is not stored in Git because of its size. The downloader
uses manageable geographic cells and writes a resumable manifest:

```powershell
python src/ingestion/india_ndvi_ingest.py `
  --start-date 2025-01-01 `
  --end-date 2025-01-31 `
  --output-dir data/india_ndvi `
  --cell-size-degrees 2.5 `
  --cloud-cover-max 20
```

The command searches Copernicus Sentinel-2 L2A scenes and stores downloaded
bands below `data/india_ndvi/cells/`. Progress is recorded in
`data/india_ndvi/india_ndvi_manifest.json`, so completed cells are skipped on
the next run. This operation can require substantial storage, bandwidth, and
time. It is better to begin with a larger cell size or a smaller date range.

The existing vegetation stage computes NDVI from matching B04 (red) and B08
(near-infrared) bands:

```powershell
python src/analysis/vegetation_analysis.py `
  --tile-id <tile-id> `
  --bands-dir data/india_ndvi/cells/<cell-id>/geotiffs `
  --output-dir data/india_ndvi/vegetation `
  --ndvi-threshold 0.3
```

Repeat this for the downloaded tiles, or automate it after confirming the
manifest and tile naming for the selected date range.

### 9. Score vegetation risk along the India line layer

After creating an NDVI GeoTIFF covering the line network, run:

```powershell
python src/ml/corridor_risk_model.py `
  --transmission-lines data/GatiShakti_Transmission_Lines_220kV_plus.geojson `
  --ndvi-raster <path-to-ndvi-geotiff> `
  --model data/models/india_risk_model.joblib `
  --output data/india_corridor_risk.geojson `
  --corridor-buffer-m 50 `
  --segment-length-m 500 `
  --ndvi-threshold 0.3
```

The output contains GeoJSON line segments with mean NDVI, vegetation fraction,
sampled-pixel count, risk score, and Low/Medium/High category. The API serves
it at `http://127.0.0.1:8000/india-corridor-risk` when the file exists.

### 10. Run tests and basic checks

With the Python environment activated:

```powershell
python -m pytest -q
python -m py_compile src/api/main.py src/pipeline.py
```

Build the frontend before deployment:

```powershell
Push-Location dashboard
npm run build
Pop-Location
```

### Troubleshooting

**`ModuleNotFoundError` for `geopandas`, `rasterio`, or another package**

Confirm the virtual environment is active and install from the same
interpreter:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If Fiona/GDAL cannot build on the selected Python version, recreate the
environment with Python 3.11 or 3.12 and install again.

**`npm start` says `package.json` cannot be found**

Run it inside `dashboard`, not the repository root.

**The dashboard shows fallback/demo data**

Start the API on port 8000, confirm `/hotspots` and `/transmission-lines`
respond, and reload the dashboard. If no generated risk file exists, the API
uses the checked-in training observations as a first-run fallback.

**No Sentinel-2 scenes are found**

Expand the date range, increase `--cloud-cover-max`, verify the AOI CRS, and
check internet/STAC access. Do not interpret missing imagery as zero vegetation.

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

## Data Sources

The project uses the following datasets for vegetation analysis, transmission-line mapping, and validation:

### 1. Sentinel-2 Surface Reflectance
- **Source:** ESA / Copernicus
- **Google Earth Engine Dataset:** `COPERNICUS/S2_SR_HARMONIZED`
- **Purpose:** Satellite-based vegetation monitoring and NDVI calculation.
- **Access:** Google Earth Engine

### 2. GridFinder Power Grid Dataset
- **Source:** GridFinder / World Bank research dataset
- **Purpose:** Initial power-grid and transmission-line spatial analysis.
- **Usage:** Distance-to-power-line analysis and preliminary infrastructure mapping.

### 3. PM GatiShakti 220 kV+ Transmission Lines
- **Source:** Ministry of Power, Government of India, through PM GatiShakti
- **Dataset:** 220 kV+ transmission lines
- **Purpose:** Mapping and validation of high-voltage transmission infrastructure.
- **Local files:**
  - `data/GatiShakti_Transmission_Lines_220kV_plus.geojson`
  - `data/GatiShakti_220kV_Transmission_Shapefile.zip`

### 4. Study Area
- **Location:** Kanpur region, Uttar Pradesh, India
- **Source:** Project-defined geometry
- **Purpose:** Defines the geographical boundary used for vegetation and transmission-corridor analysis.
