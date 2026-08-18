#!/usr/bin/env python3
"""
NASA POWER Weather/Climate Data Ingestion

Official source: https://power.larc.nasa.gov/
NASA POWER provides global meteorological and solar energy parameters.

Parameters of interest for vegetation risk:
- PRECTOTCORR: Precipitation (mm/day)
- T2M: Temperature at 2 meters (°C)
- T2M_MAX: Maximum temperature at 2m (°C)
- T2M_MIN: Minimum temperature at 2m (°C)
- RH2M: Relative humidity at 2m (%)
- WS2M: Wind speed at 2m (m/s)
- ALLSKY_SFC_SW_DWN: All-sky surface shortwave downward irradiance (MJ/m2/day)
- CLRSKY_SFC_SW_DWN: Clear-sky surface shortwave downward irradiance

IMPORTANT: Weather data provides CONTEXT for vegetation growth analysis.
Do NOT directly convert rainfall into risk score without validation.
Use weather to understand:
- Seasonal vegetation growth patterns
- Monsoon periods
- Drought/stress conditions
- Data quality context for satellite observations
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import requests
import numpy as np
import pandas as pd
from tqdm import tqdm


# NASA POWER API endpoint
POWER_API_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Parameters to request
POWER_PARAMETERS = [
    "PRECTOTCORR",   # Precipitation (mm/day)
    "T2M",           # Temperature at 2m (°C)
    "T2M_MAX",       # Max temperature at 2m (°C)
    "T2M_MIN",       # Min temperature at 2m (°C)
    "RH2M",          # Relative humidity at 2m (%)
    "WS2M",          # Wind speed at 2m (m/s)
    "ALLSKY_SFC_SW_DWN",  # All-sky shortwave radiation (MJ/m2/day)
    "CLRSKY_SFC_SW_DWN",  # Clear-sky shortwave radiation (MJ/m2/day)
]


class WeatherIngestor:
    """Download NASA POWER weather data for AOI centroid or transmission line points."""

    def __init__(
        self,
        aoi_geojson: str,
        output_dir: str,
        start_date: str,
        end_date: str,
        target_crs: str = "EPSG:4326",
    ):
        """
        Initialize weather ingestor.

        Args:
            aoi_geojson: Path to AOI GeoJSON file (uses centroid for point query)
            output_dir: Output directory
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            target_crs: Target CRS (default WGS84)
        """
        self.aoi_path = Path(aoi_geojson)
        self.output_dir = Path(output_dir)
        self.start_date = start_date
        self.end_date = end_date
        self.target_crs = target_crs

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "weather").mkdir(exist_ok=True)

        # Load AOI and get centroid for point query
        import geopandas as gpd
        self.aoi_gdf = gpd.read_file(self.aoi_path)
        if self.aoi_gdf.crs is None:
            raise ValueError("AOI GeoJSON must have a CRS defined")
        self.aoi_gdf = self.aoi_gdf.to_crs(self.target_crs)
        self.aoi_centroid = self.aoi_gdf.geometry.unary_union.centroid
        self.lat = self.aoi_centroid.y
        self.lon = self.aoi_centroid.x

    def fetch_power_data(self) -> Dict:
        """Fetch weather data from NASA POWER API."""
        params = {
            "parameters": ",".join(POWER_PARAMETERS),
            "community": "AG",  # Agroclimatology community
            "longitude": self.lon,
            "latitude": self.lat,
            "start": self.start_date,
            "end": self.end_date,
            "format": "JSON",
        }

        print(f"Fetching NASA POWER data for ({self.lat:.4f}, {self.lon:.4f})...")
        print(f"  Date range: {self.start_date} to {self.end_date}")

        response = requests.get(POWER_API_URL, params=params, timeout=60)
        response.raise_for_status()

        data = response.json()

        if "properties" not in data or "parameter" not in data["properties"]:
            raise ValueError("Unexpected POWER API response format")

        return data["properties"]["parameter"]

    def process_weather_data(self, raw_data: Dict) -> pd.DataFrame:
        """Process raw POWER data into daily DataFrame."""
        # POWER returns data as: {parameter: {date_string: value}}
        dates = list(raw_data[POWER_PARAMETERS[0]].keys())

        rows = []
        for date_str in dates:
            row = {"date": date_str}
            for param in POWER_PARAMETERS:
                value = raw_data[param].get(date_str)
                # POWER uses -999 for missing
                if value is not None and value > -900:
                    row[param.lower()] = value
                else:
                    row[param.lower()] = np.nan
            rows.append(row)

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        return df

    def compute_derived_metrics(self, df: pd.DataFrame) -> Dict:
        """Compute derived weather metrics relevant to vegetation."""
        metrics = {}

        # Total precipitation
        if "prectotcorr" in df.columns:
            metrics["total_precipitation_mm"] = float(df["prectotcorr"].sum())
            metrics["mean_daily_precipitation_mm"] = float(df["prectotcorr"].mean())
            metrics["max_daily_precipitation_mm"] = float(df["prectotcorr"].max())
            metrics["rainy_days"] = int((df["prectotcorr"] > 1.0).sum())

        # Temperature metrics
        if "t2m" in df.columns:
            metrics["mean_temperature_c"] = float(df["t2m"].mean())
            metrics["max_temperature_c"] = float(df["t2m_max"].max())
            metrics["min_temperature_c"] = float(df["t2m_min"].min())

        # Humidity
        if "rh2m" in df.columns:
            metrics["mean_humidity_pct"] = float(df["rh2m"].mean())

        # Wind
        if "ws2m" in df.columns:
            metrics["mean_wind_speed_ms"] = float(df["ws2m"].mean())

        # Radiation
        if "allsky_sfc_sw_dwn" in df.columns:
            metrics["mean_solar_radiation_mj_m2"] = float(df["allsky_sfc_sw_dwn"].mean())

        # Growing season indicators
        if "t2m" in df.columns and "prectotcorr" in df.columns:
            # Growing degree days (base 10°C)
            gdd = np.maximum(df["t2m"] - 10, 0)
            metrics["growing_degree_days"] = float(gdd.sum())

            # Water stress index (precipitation / potential evapotranspiration proxy)
            # Simplified: temperature * radiation as PET proxy
            if "allsky_sfc_sw_dwn" in df.columns:
                pet_proxy = df["t2m"] * df["allsky_sfc_sw_dwn"]
                total_pet = pet_proxy.sum()
                total_precip = df["prectotcorr"].sum()
                if total_pet > 0:
                    metrics["water_stress_index"] = float(total_precip / total_pet)
                else:
                    metrics["water_stress_index"] = None

        return metrics

    def run(self) -> Dict:
        """Run weather ingestion."""
        try:
            raw_data = self.fetch_power_data()
            df = self.process_weather_data(raw_data)
            metrics = self.compute_derived_metrics(df)

            # Save daily data
            csv_path = self.output_dir / "weather" / "nasa_power_daily.csv"
            df.to_csv(csv_path, index=False)

            # Save summary
            summary = {
                "status": "SUCCESS",
                "data_source": "NASA POWER",
                "data_source_url": "https://power.larc.nasa.gov/",
                "latitude": self.lat,
                "longitude": self.lon,
                "date_range": f"{self.start_date}/{self.end_date}",
                "parameters": POWER_PARAMETERS,
                "daily_data_path": str(csv_path),
                "derived_metrics": metrics,
                "note": "Weather data provides CONTEXT for vegetation growth analysis. Do NOT directly convert rainfall into risk score without validation.",
            }

            summary_path = self.output_dir / "weather" / "weather_summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)

            print(f"\nWeather data fetched successfully.")
            print(f"  Days: {len(df)}")
            print(f"  Total precipitation: {metrics.get('total_precipitation_mm', 0):.1f} mm")
            print(f"  Mean temperature: {metrics.get('mean_temperature_c', 0):.1f} °C")
            print(f"  Summary saved to: {summary_path}")

            return summary

        except Exception as e:
            print(f"Error fetching weather data: {e}")
            result = {
                "status": "ERROR",
                "reason": str(e),
                "data_source": "NASA POWER",
                "data_source_url": "https://power.larc.nasa.gov/",
            }
            summary_path = self.output_dir / "weather" / "weather_summary.json"
            with open(summary_path, "w") as f:
                json.dump(result, f, indent=2)
            return result


def main():
    parser = argparse.ArgumentParser(
        description="NASA POWER Weather Data Ingestion (Tier 2)"
    )
    parser.add_argument("--aoi", required=True, help="Path to AOI GeoJSON file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--target-crs", default="EPSG:4326", help="Target CRS (default: EPSG:4326)")

    args = parser.parse_args()

    ingestor = WeatherIngestor(
        aoi_geojson=args.aoi,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        target_crs=args.target_crs,
    )

    ingestor.run()


if __name__ == "__main__":
    main()