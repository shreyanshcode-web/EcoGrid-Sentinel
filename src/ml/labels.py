#!/usr/bin/env python3
"""
Label Generation for ML Training

Since historical incident/outage data is not available, this module generates
synthetic/pseudo-labels using rule-based heuristics and the existing risk scorer.
These serve as weak supervision for initial model training.

Label Strategies:
1. Heuristic-based: Use existing weighted risk scorer as pseudo-labels
2. Rule-based: Explicit domain rules for high/medium/low risk
3. Hybrid: Combine both with confidence weighting
4. Future: Load real incident data when available
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json


# Risk category mapping
RISK_CATEGORIES = {"Low": 0, "Medium": 1, "High": 2}
RISK_CATEGORIES_REV = {v: k for k, v in RISK_CATEGORIES.items()}


def generate_heuristic_labels(
    features_df: pd.DataFrame,
    risk_scorer=None,
    score_col: str = "risk_score",
    category_col: str = "risk_category",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate labels using the existing heuristic risk scorer.

    Args:
        features_df: Feature dataframe from feature engineering
        risk_scorer: Optional RiskScorer instance (creates default if None)
        score_col: Column name for continuous risk score
        category_col: Column name for categorical risk

    Returns:
        (y_score, y_category) arrays
    """
    if risk_scorer is None:
        from src.risk.risk_scoring import RiskScorer
        risk_scorer = RiskScorer()

    # Compute heuristic risk scores
    risk_df = risk_scorer.compute_risk_score(features_df.copy())

    y_score = risk_df[score_col].values
    y_category = risk_df[category_col].map(RISK_CATEGORIES).values

    return y_score, y_category


def generate_rule_based_labels(
    features_df: pd.DataFrame,
    proximity_thresh: float = 20.0,
    veg_fraction_thresh: float = 0.3,
    growth_thresh: float = 0.05,
    canopy_height_thresh: float = 15.0,
    slope_thresh: float = 15.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate labels using explicit domain rules.

    High risk if ANY of:
    - Very close to line (< proximity_thresh m) AND significant vegetation
    - High vegetation fraction + positive growth trend
    - Tall vegetation near line (canopy height > thresh)
    - Steep slope + vegetation (fall risk)

    Medium risk if:
    - Moderate proximity + vegetation
    - Some growth trend

    Low risk otherwise.
    """
    n = len(features_df)
    y_score = np.zeros(n)
    y_category = np.zeros(n, dtype=int)  # 0=Low, 1=Medium, 2=High

    # Extract relevant features (handle missing columns gracefully)
    def get_col(name, default=0):
        return features_df[name].values if name in features_df.columns else np.full(n, default)

    # Key features
    dist_to_line = get_col("mean_dist_to_line_m", 1000)
    min_dist_to_line = get_col("min_dist_to_line_m", 1000)
    veg_fraction = get_col("vegetation_fraction", 0)
    veg_patch_count = get_col("veg_patch_count", 0)
    growth_rate = get_col("growth_rate_mean", 0)
    ndvi_change = get_col("ndvi_change_mean", 0)
    canopy_height = get_col("gedi_canopy_height_mean", 0)
    slope = get_col("dem_elevation_std", 0)  # proxy for terrain roughness
    pct_inside = get_col("pct_inside_corridor", 0)
    pct_near_tower = get_col("pct_near_tower", 0)

    # Rule 1: Very close to line with vegetation
    very_close = (min_dist_to_line < proximity_thresh) & (veg_fraction > veg_fraction_thresh)

    # Rule 2: High veg fraction + positive growth
    high_veg_growing = (veg_fraction > veg_fraction_thresh) & (growth_rate > growth_thresh)

    # Rule 3: Tall vegetation near corridor
    tall_veg_near = (canopy_height > canopy_height_thresh) & (min_dist_to_line < 50)

    # Rule 4: Steep terrain + vegetation (fall risk)
    steep_veg = (slope > slope_thresh) & (veg_fraction > 0.1)

    # Rule 5: Inside corridor + near tower
    tower_risk = (pct_inside > 50) & (pct_near_tower > 30)

    # High risk conditions
    high_risk = very_close | high_veg_growing | tall_veg_near | steep_veg | tower_risk

    # Medium risk conditions
    med_proximity = (min_dist_to_line < 50) & (veg_fraction > 0.15)
    med_growth = (growth_rate > 0.02) & (veg_fraction > 0.1)
    med_veg = (veg_fraction > 0.25)
    medium_risk = med_proximity | med_growth | med_veg

    # Assign categories
    y_category[high_risk] = 2  # High
    y_category[~high_risk & medium_risk] = 1  # Medium
    y_category[~high_risk & ~medium_risk] = 0  # Low

    # Continuous score based on rules
    y_score = np.zeros(n)
    y_score[high_risk] = 0.7 + 0.3 * np.random.random(np.sum(high_risk))
    y_score[medium_risk & ~high_risk] = 0.4 + 0.3 * np.random.random(np.sum(medium_risk & ~high_risk))
    y_score[~high_risk & ~medium_risk] = 0.1 * np.random.random(np.sum(~high_risk & ~medium_risk))

    return y_score, y_category


def generate_hybrid_labels(
    features_df: pd.DataFrame,
    heuristic_weight: float = 0.6,
    rule_weight: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine heuristic and rule-based labels with weighted voting.

    Args:
        features_df: Feature dataframe
        heuristic_weight: Weight for heuristic labels
        rule_weight: Weight for rule-based labels

    Returns:
        (y_score, y_category) arrays
    """
    # Get both label sets
    h_score, h_cat = generate_heuristic_labels(features_df)
    r_score, r_cat = generate_rule_based_labels(features_df)

    # Weighted combination for continuous score
    y_score = heuristic_weight * h_score + rule_weight * r_score

    # For category, use weighted voting
    # Convert to one-hot, weight, then argmax
    n = len(features_df)
    h_onehot = np.zeros((n, 3))
    r_onehot = np.zeros((n, 3))
    h_onehot[np.arange(n), h_cat] = 1
    r_onehot[np.arange(n), r_cat] = 1

    combined = heuristic_weight * h_onehot + rule_weight * r_onehot
    y_category = np.argmax(combined, axis=1)

    return y_score, y_category


def generate_labels(
    features_df: pd.DataFrame,
    method: str = "hybrid",
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Main label generation function.

    Args:
        features_df: Feature dataframe from feature engineering
        method: "heuristic", "rule", or "hybrid"
        **kwargs: Additional arguments for specific methods

    Returns:
        (y_score, y_category, metadata) tuple
    """
    print(f"Generating labels using method: {method}")

    if method == "heuristic":
        y_score, y_category = generate_heuristic_labels(features_df, **kwargs)
    elif method == "rule":
        y_score, y_category = generate_rule_based_labels(features_df, **kwargs)
    elif method == "hybrid":
        y_score, y_category = generate_hybrid_labels(features_df, **kwargs)
    else:
        raise ValueError(f"Unknown label method: {method}")

    # Metadata
    metadata = {
        "method": method,
        "n_samples": len(features_df),
        "class_distribution": {
            RISK_CATEGORIES_REV[i]: int(np.sum(y_category == i))
            for i in range(3)
        },
        "score_stats": {
            "mean": float(y_score.mean()),
            "std": float(y_score.std()),
            "min": float(y_score.min()),
            "max": float(y_score.max()),
        },
    }

    print(f"  Class distribution: {metadata['class_distribution']}")
    print(f"  Score stats: mean={metadata['score_stats']['mean']:.3f}, std={metadata['score_stats']['std']:.3f}")

    return y_score, y_category, metadata


def load_incident_labels(
    incident_csv: str,
    features_df: pd.DataFrame,
    segment_id_col: str = "segment_id",
    incident_id_col: str = "segment_id",
    label_col: str = "risk_category",
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load real incident/outage labels from CSV.

    Expected CSV format:
    segment_id, risk_category (Low/Medium/High or 0/1/2), [optional: risk_score]

    Args:
        incident_csv: Path to incident labels CSV
        features_df: Feature dataframe to align labels with
        segment_id_col: Column in features_df with segment IDs
        incident_id_col: Column in incident CSV with segment IDs
        label_col: Column in incident CSV with risk labels

    Returns:
        (y_score, y_category, metadata) tuple
    """
    incidents = pd.read_csv(incident_csv)

    # Align with features
    merged = features_df[[segment_id_col]].merge(
        incidents[[incident_id_col, label_col]],
        left_on=segment_id_col,
        right_on=incident_id_col,
        how="left"
    )

    # Map categories
    y_category = merged[label_col].map(RISK_CATEGORIES).fillna(0).astype(int).values

    # If risk_score column exists, use it; otherwise convert categories to scores
    if "risk_score" in incidents.columns:
        y_score = merged["risk_score"].fillna(0.5).values
    else:
        # Map categories to representative scores
        cat_to_score = {0: 0.15, 1: 0.55, 2: 0.85}
        y_score = np.array([cat_to_score[c] for c in y_category])

    metadata = {
        "method": "incident_data",
        "n_samples": len(features_df),
        "n_labeled": int(merged[label_col].notna().sum()),
        "class_distribution": {
            RISK_CATEGORIES_REV[i]: int(np.sum(y_category == i))
            for i in range(3)
        },
    }

    print(f"Loaded incident labels: {metadata['n_labeled']}/{metadata['n_samples']} segments labeled")
    print(f"  Class distribution: {metadata['class_distribution']}")

    return y_score, y_category, metadata


def save_labels(
    y_score: np.ndarray,
    y_category: np.ndarray,
    features_df: pd.DataFrame,
    output_dir: str,
    segment_id_col: str = "segment_id",
    metadata: Optional[Dict] = None,
):
    """Save generated labels alongside features for training."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create labels dataframe
    labels_df = pd.DataFrame({
        segment_id_col: features_df[segment_id_col].values,
        "risk_score": y_score,
        "risk_category": [RISK_CATEGORIES_REV[c] for c in y_category],
        "risk_category_id": y_category,
    })

    # Save
    labels_path = output_dir / "ml_labels.csv"
    labels_df.to_csv(labels_path, index=False)
    print(f"Saved labels to: {labels_path}")

    # Save metadata
    if metadata:
        meta_path = output_dir / "ml_labels_meta.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to: {meta_path}")

    return labels_path


def load_labels(labels_csv: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load previously saved labels."""
    df = pd.read_csv(labels_csv)
    y_score = df["risk_score"].values
    y_category = df["risk_category_id"].values
    return y_score, y_category, df


if __name__ == "__main__":
    # Test with sample features
    import argparse
    parser = argparse.ArgumentParser(description="Generate ML labels from features")
    parser.add_argument("--features-csv", required=True, help="Path to features.csv from feature engineering")
    parser.add_argument("--output-dir", required=True, help="Output directory for labels")
    parser.add_argument("--method", default="hybrid", choices=["heuristic", "rule", "hybrid"])
    args = parser.parse_args()

    features_df = pd.read_csv(args.features_csv)
    y_score, y_category, meta = generate_labels(features_df, method=args.method)
    save_labels(y_score, y_category, features_df, args.output_dir, metadata=meta)