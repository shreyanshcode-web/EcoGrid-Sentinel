#!/usr/bin/env python3
"""
Test suite for vegetation risk intelligence pipeline components.
Run with: pytest tests/test_pipeline.py -v
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# =============================================================================
# Test Risk Scoring
# =============================================================================

def test_risk_scorer_basic():
    """Test basic risk scoring functionality."""
    from risk.risk_scoring import RiskScorer, normalize_feature

    scorer = RiskScorer()

    # Create mock feature dataframe
    df = pd.DataFrame({
        "mean_dist_to_line_m": [10, 50, 200],
        "vegetation_fraction": [0.5, 0.2, 0.1],
        "veg_patch_count": [10, 5, 1],
        "growth_rate_mean": [0.1, 0.05, 0.0],
        "ndvi_change_mean": [0.05, 0.02, 0.0],
        "early_ndvi_mean": [0.6, 0.4, 0.2],
        "late_ndvi_mean": [0.65, 0.42, 0.2],
    })

    result = scorer.compute_risk_score(df)

    assert "risk_score" in result.columns
    assert "risk_category" in result.columns
    assert "risk_breakdown" in result.columns
    assert len(result) == 3

    # Higher risk for closer, denser vegetation
    assert result.iloc[0]["risk_score"] > result.iloc[2]["risk_score"]


def test_normalize_feature():
    """Test feature normalization."""
    from risk.risk_scoring import normalize_feature

    values = np.array([10, 50, 100, 200])
    norm = normalize_feature(values)

    assert np.min(norm) == 0.0
    assert np.max(norm) == 1.0
    assert norm[0] < norm[-1]  # 10 < 200

    # Test invert
    norm_inv = normalize_feature(values, invert=True)
    assert norm_inv[0] > norm_inv[-1]  # inverted


def test_risk_breakdown_structure():
    """Test that risk breakdown has expected structure."""
    from risk.risk_scoring import RiskScorer

    scorer = RiskScorer()

    df = pd.DataFrame({
        "mean_dist_to_line_m": [10],
        "vegetation_fraction": [0.5],
        "veg_patch_count": [10],
        "growth_rate_mean": [0.1],
        "ndvi_change_mean": [0.05],
        "early_ndvi_mean": [0.6],
        "late_ndvi_mean": [0.65],
    })

    result = scorer.compute_risk_score(df)
    breakdown = result.iloc[0]["risk_breakdown"]

    assert "total_score" in breakdown
    assert "risk_category" in breakdown
    assert "components" in breakdown
    assert "key_factors" in breakdown
    assert "recommended_action" in breakdown

    for comp in ["proximity", "density", "growth", "condition"]:
        assert comp in breakdown["components"]
        c = breakdown["components"][comp]
        assert "score" in c
        assert "weight" in c
        assert "weighted" in c
        assert "contribution_pct" in c
        assert "interpretation" in c


# =============================================================================
# Test Vegetation Analysis
# =============================================================================

def test_ndvi_computation():
    """Test NDVI formula with known values."""
    from analysis.vegetation_analysis import VegetationAnalyzer

    analyzer = VegetationAnalyzer()

    # Mock NIR and Red arrays
    nir = np.array([[1000, 2000], [3000, 4000]], dtype=np.float32)
    red = np.array([[500, 1000], [1500, 2000]], dtype=np.float32)

    # NDVI = (NIR - Red) / (NIR + Red)
    # (1000-500)/(1000+500) = 500/1500 = 0.333
    expected = (nir - red) / (nir + red)

    # We can't easily test compute_ndvi without raster files,
    # but we can test the vegetation class map logic
    ndvi_test = np.array([[0.2, 0.4], [0.6, 0.8]])
    class_map = analyzer.create_vegetation_class_map(ndvi_test)

    assert class_map[0, 0] == 0  # Bare (< 0.3)
    assert class_map[0, 1] == 1  # Sparse (0.3-0.5)
    assert class_map[1, 0] == 2  # Moderate (0.5-0.7)
    assert class_map[1, 1] == 3  # Dense (> 0.7)


# =============================================================================
# Test Spatial Analysis
# =============================================================================

def test_spatial_analyzer_import():
    """Test that spatial analyzer imports without error."""
    from spatial.spatial_analysis import SpatialAnalyzer
    assert SpatialAnalyzer is not None


# =============================================================================
# Test Temporal Analysis
# =============================================================================

def test_growth_rate_calculation():
    """Test growth rate computation."""
    from temporal.temporal_analysis import TemporalAnalyzer

    analyzer = TemporalAnalyzer()

    ndvi_early = np.array([[0.5, 0.6], [0.4, 0.5]])
    ndvi_late = np.array([[0.55, 0.66], [0.42, 0.52]])

    result = analyzer.compute_growth_rate(ndvi_early, ndvi_late, days_between=30)

    assert "growth_rate" in result
    assert "stats" in result
    assert result["stats"]["days_between"] == 30

    # Growth should be positive (increase)
    assert np.mean(result["growth_rate"]) > 0


# =============================================================================
# Test Feature Engineering
# =============================================================================

def test_feature_engineer_import():
    """Test that feature engineer imports without error."""
    from features.feature_engineering import FeatureEngineer
    assert FeatureEngineer is not None


# =============================================================================
# Test API Models
# =============================================================================

def test_api_models():
    """Test Pydantic models for API."""
    from api.main import Hotspot, RiskSummary, RiskComponent, RiskBreakdown

    # Test RiskComponent
    comp = RiskComponent(
        score=0.8, weight=0.4, weighted=0.32, contribution_pct=40.0,
        interpretation="Test interpretation"
    )
    assert comp.score == 0.8

    # Test RiskBreakdown
    breakdown = RiskBreakdown(
        total_score=0.75,
        risk_category="High",
        components={"proximity": comp},
        key_factors=["Test factor"],
        recommended_action="Test action"
    )
    assert breakdown.total_score == 0.75


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])