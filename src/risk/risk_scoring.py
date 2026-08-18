#!/usr/bin/env python3
"""
Risk Scoring Script (Stage 6)

Implements explicit weighted sum heuristic risk scorer with explainability breakdown.
Each weight is a named constant with rationale comment.

risk = w1*proximity_score + w2*veg_density + w3*growth_rate + w4*veg_condition

Normalized to 0-1, bucketed into Low/Medium/High with configurable thresholds.

Known Limitations:
1. This is a transparent, tunable weighted heuristic (NOT trained supervised ML).
   Leave extension point for supervised training when incident data becomes available.
2. No canopy height data — vegetation condition proxy only.
3. Risk scores are corridor-segment-level, not individual tree risk.
4. No historical incident/outage ground truth for calibration.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# =============================================================================
# RISK SCORING WEIGHTS AND THRESHOLDS
# =============================================================================
# These weights are tuned for vegetation risk near transmission corridors.
# Each weight has a one-line rationale comment explaining why it matters.

WEIGHTS = {
    # Proximity to line: Closer vegetation has higher fall/contact risk
    # Weight: 0.40 — Primary factor for fall hazard assessment
    "proximity": 0.40,

    # Vegetation density: Higher density = more fuel for fires, more branches
    # Weight: 0.25 — Dense vegetation increases both fire and contact risk
    "density": 0.25,

    # Growth rate: Fast-growing species may reach conductors quicker
    # Weight: 0.20 — Temporal component for future risk estimation
    "growth": 0.20,

    # Vegetation condition: Stressed/diseased trees are more likely to fail
    # Weight: 0.15 — Condition affects failure probability (fall risk)
    "condition": 0.15,
}

# Risk bucket thresholds (normalized 0-1 score)
THRESHOLDS = {
    "high": 0.7,    # Above this: High risk
    "medium": 0.4,  # Above this: Medium risk
    "low": 0.0,     # Below medium: Low risk
}


def normalize_feature(
    values: np.ndarray,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    invert: bool = False,
) -> np.ndarray:
    """
    Normalize a feature to 0-1 range using min-max normalization.

    Args:
        values: Input values
        min_val: Minimum value (computed from data if None)
        max_val: Maximum value (computed from data if None)
        invert: If True, invert (1 - normalized) for inverse relationships

    Returns:
        Normalized values in [0, 1]
    """
    values = np.array(values, dtype=float)
    if min_val is None:
        min_val = np.nanmin(values)
    if max_val is None:
        max_val = np.nanmax(values)

    if max_val == min_val:
        normalized = np.zeros_like(values)
    else:
        normalized = (values - min_val) / (max_val - min_val)

    normalized = np.clip(normalized, 0, 1)
    if invert:
        normalized = 1 - normalized

    return normalized


class RiskScorer:
    """
    Compute risk scores for corridor segments using weighted heuristic.

    Risk formula:
        risk = w1*proximity_score + w2*veg_density + w3*growth_rate + w4*veg_condition

    Returns explainable breakdown dict for each segment.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize risk scorer.

        Args:
            weights: Custom weights dict (uses defaults if None)
            thresholds: Custom risk bucket thresholds (uses defaults if None)
        """
        self.weights = weights or WEIGHTS.copy()
        self.thresholds = thresholds or THRESHOLDS.copy()

        # Validate weights sum to 1.0
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            print(
                f"Warning: Weights sum to {weight_sum:.3f}, not 1.0. "
                f"Normalizing..."
            )
            self.weights = {k: v / weight_sum for k, v in self.weights.items()}

    def compute_proximity_score(
        self, dist_to_line_m: np.ndarray
    ) -> np.ndarray:
        """
        Compute proximity score (0-1, higher = closer = more risk).

        Distance is inversely proportional to risk.
        Vegetation within 30m gets full score; beyond 200m gets minimal score.
        """
        return normalize_feature(dist_to_line_m, invert=True)

    def compute_density_score(
        self, vegetation_fraction: np.ndarray, veg_patch_count: np.ndarray
    ) -> np.ndarray:
        """
        Compute vegetation density score (0-1).

        Combines vegetation fraction (normalized area coverage) and
        patch count (normalized number of patches).
        """
        frac_score = normalize_feature(vegetation_fraction)
        count_score = normalize_feature(veg_patch_count)

        # Weighted combination: fraction matters more than count
        density_score = 0.7 * frac_score + 0.3 * count_score
        return density_score

    def compute_growth_score(
        self, growth_rate_mean: np.ndarray, ndvi_change_mean: np.ndarray
    ) -> np.ndarray:
        """
        Compute growth rate score (0-1, higher = faster growth = more risk).

        Positive NDVI change (greening) indicates active growth toward conductors.
        Negative change could indicate stress (but may not be immediate risk).
        """
        # Use absolute growth rate
        abs_growth = np.abs(growth_rate_mean)
        growth_score = normalize_feature(abs_growth)

        # Add NDVI change component
        change_score = normalize_feature(np.abs(ndvi_change_mean))

        # Combine
        combined = 0.6 * growth_score + 0.4 * change_score
        return combined

    def compute_condition_score(
        self,
        ndvi_early_mean: np.ndarray,
        ndvi_late_mean: np.ndarray,
        vegetation_fraction: np.ndarray,
    ) -> np.ndarray:
        """
        Compute vegetation condition score (0-1).

        High NDVI (healthy green vegetation) has higher fall risk than
        stressed/dead vegetation (which may shed leaves/branches but
        has different failure modes).
        """
        # Higher NDVI = healthier = denser canopy = more risk
        health_score = normalize_feature(ndvi_late_mean)

        # Consider if vegetation is declining (potential stress)
        ndvi_change = ndvi_late_mean - ndvi_early_mean
        stress_penalty = normalize_feature(ndvi_change) * 0.3

        # Combine: healthier vegetation is more risky for fall hazard
        condition_score = 0.7 * health_score + 0.3 * vegetation_fraction
        return condition_score

    def compute_risk_score(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute risk scores for all corridor segments.

        Args:
            features_df: DataFrame with feature columns

        Returns:
            DataFrame with risk scores and breakdowns
        """
        df = features_df.copy()

        # Compute individual component scores
        df["proximity_score"] = self.compute_proximity_score(
            df["mean_dist_to_line_m"].values
        )

        df["density_score"] = self.compute_density_score(
            df["vegetation_fraction"].values,
            df["veg_patch_count"].values,
        )

        df["growth_score"] = self.compute_growth_score(
            df["growth_rate_mean"].values if "growth_rate_mean" in df.columns else np.zeros(len(df)),
            df["ndvi_change_mean"].values if "ndvi_change_mean" in df.columns else np.zeros(len(df)),
        )

        df["condition_score"] = self.compute_condition_score(
            df["early_ndvi_mean"].values if "early_ndvi_mean" in df.columns else np.zeros(len(df)),
            df["late_ndvi_mean"].values if "late_ndvi_mean" in df.columns else np.zeros(len(df)),
            df["vegetation_fraction"].values,
        )

        # Compute weighted risk score
        df["risk_score_raw"] = (
            self.weights["proximity"] * df["proximity_score"]
            + self.weights["density"] * df["density_score"]
            + self.weights["growth"] * df["growth_score"]
            + self.weights["condition"] * df["condition_score"]
        )

        # Normalize to 0-1
        df["risk_score"] = normalize_feature(df["risk_score_raw"].values)

        # Assign risk category
        df["risk_category"] = pd.cut(
            df["risk_score"],
            bins=[-np.inf, self.thresholds["medium"], self.thresholds["high"], np.inf],
            labels=["Low", "Medium", "High"],
        )

        # Create explainable breakdown for each segment
        df["risk_breakdown"] = df.apply(
            lambda row: self._create_breakdown_dict(row), axis=1
        )

        return df

    def _create_breakdown_dict(self, row) -> Dict:
        """Create explainable risk breakdown dictionary for a segment."""
        return {
            "total_score": float(row["risk_score"]),
            "risk_category": str(row["risk_category"]),
            "components": {
                "proximity": {
                    "score": float(row["proximity_score"]),
                    "weight": self.weights["proximity"],
                    "weighted": float(row["proximity_score"] * self.weights["proximity"]),
                    "contribution_pct": float(
                        row["proximity_score"] * self.weights["proximity"] / max(row["risk_score_raw"], 0.001) * 100
                    ),
                    "interpretation": self._interpret_proximity(row["mean_dist_to_line_m"]),
                },
                "density": {
                    "score": float(row["density_score"]),
                    "weight": self.weights["density"],
                    "weighted": float(row["density_score"] * self.weights["density"]),
                    "contribution_pct": float(
                        row["density_score"] * self.weights["density"] / max(row["risk_score_raw"], 0.001) * 100
                    ),
                    "interpretation": self._interpret_density(row["vegetation_fraction"]),
                },
                "growth": {
                    "score": float(row["growth_score"]),
                    "weight": self.weights["growth"],
                    "weighted": float(row["growth_score"] * self.weights["growth"]),
                    "contribution_pct": float(
                        row["growth_score"] * self.weights["growth"] / max(row["risk_score_raw"], 0.001) * 100
                    ),
                    "interpretation": self._interpret_growth(
                        row.get("growth_rate_mean", 0)
                    ),
                },
                "condition": {
                    "score": float(row["condition_score"]),
                    "weight": self.weights["condition"],
                    "weighted": float(row["condition_score"] * self.weights["condition"]),
                    "contribution_pct": float(
                        row["condition_score"] * self.weights["condition"] / max(row["risk_score_raw"], 0.001) * 100
                    ),
                    "interpretation": self._interpret_condition(
                        row.get("late_ndvi_mean", 0)
                    ),
                },
            },
            "key_factors": self._identify_key_factors(row),
            "recommended_action": self._recommend_action(row),
        }

    def _interpret_proximity(self, dist_m: float) -> str:
        """Human-readable interpretation of proximity score."""
        if dist_m < 10:
            return "Critical: Vegetation within 10m of line — immediate inspection recommended"
        elif dist_m < 30:
            return "High risk: Vegetation within 30m — high probability of contact during growth/storm"
        elif dist_m < 100:
            return "Moderate risk: Vegetation 30-100m from line — monitor growth trends"
        else:
            return "Low risk: Vegetation >100m from line — low direct contact probability"

    def _interpret_density(self, fraction: float) -> str:
        """Human-readable interpretation of density score."""
        if fraction > 0.5:
            return "High density: >50% vegetation coverage — significant fire/conductor contact risk"
        elif fraction > 0.2:
            return "Moderate density: 20-50% coverage — moderate vegetation risk"
        else:
            return "Low density: <20% coverage — minimal vegetation risk"

    def _interpret_growth(self, rate: float) -> str:
        """Human-readable interpretation of growth rate."""
        if rate > 0.5:
            return "Rapid growth: >0.5%/day — vegetation may reach conductors quickly"
        elif rate > 0.1:
            return "Active growth: 0.1-0.5%/day — vegetation is expanding"
        elif rate > -0.1:
            return "Stable: Growth rate near zero — minimal change expected"
        else:
            return "Declining: Negative growth — vegetation stress or seasonal dieback"

    def _interpret_condition(self, ndvi_mean: float) -> str:
        """Human-readable interpretation of vegetation condition."""
        if ndvi_mean > 0.7:
            return "Very healthy (NDVI >0.7): Dense, vigorous vegetation — highest fall risk"
        elif ndvi_mean > 0.5:
            return "Healthy (NDVI 0.5-0.7): Moderate vegetation health"
        elif ndvi_mean > 0.3:
            return "Stressed (NDVI 0.3-0.5): Vegetation may be water-stressed or diseased"
        else:
            return "Very stressed (NDVI <0.3): Likely not woody vegetation or severely stressed"

    def _identify_key_factors(self, row) -> List[str]:
        """Identify the top risk factors for this segment."""
        factors = []

        if row["proximity_score"] > 0.7:
            factors.append("Vegetation very close to transmission line")
        if row["density_score"] > 0.7:
            factors.append("High vegetation density in corridor")
        if row["growth_score"] > 0.7:
            factors.append("Rapid vegetation growth detected")
        if row["condition_score"] > 0.7:
            factors.append("Vegetation in vigorous health (high fall risk)")

        if row["risk_category"] == "High":
            factors.append("Overall HIGH risk level — prioritize for inspection")

        return factors if factors else ["No significant risk factors identified"]

    def _recommend_action(self, row) -> str:
        """Generate recommended action based on risk score."""
        if row["risk_category"] == "High":
            return "Immediate inspection and potential vegetation management required"
        elif row["risk_category"] == "Medium":
            return "Schedule monitoring and preventive vegetation management"
        else:
            return "Routine monitoring sufficient — no immediate action needed"

    def save_risk_scores(
        self,
        df: pd.DataFrame,
        output_dir: str,
        format: str = "both",
    ):
        """
        Save risk scores.

        Args:
            df: DataFrame with risk scores
            output_dir: Output directory
            format: 'csv', 'json', or 'both'
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if format in ("csv", "both"):
            # Save CSV without breakdown dict
            csv_df = df.drop(columns=["risk_breakdown"], errors="ignore")
            csv_path = output_path / "risk_scores.csv"
            csv_df.to_csv(csv_path, index=False)
            print(f"  Saved CSV: {csv_path}")

        if format in ("json", "both"):
            # Save JSON with full breakdown
            records = []
            for idx, row in df.iterrows():
                record = {
                    "segment_id": row.get("segment_id", idx),
                    "risk_score": float(row["risk_score"]),
                    "risk_category": str(row["risk_category"]),
                    "breakdown": row.get("risk_breakdown", {}),
                }
                records.append(record)

            json_path = output_path / "risk_scores.json"
            with open(json_path, "w") as f:
                json.dump(records, f, indent=2)
            print(f"  Saved JSON: {json_path}")

        # Save risk summary
        summary = {
            "total_segments": len(df),
            "high_risk_count": int((df["risk_category"] == "High").sum()),
            "medium_risk_count": int((df["risk_category"] == "Medium").sum()),
            "low_risk_count": int((df["risk_category"] == "Low").sum()),
            "mean_risk_score": float(df["risk_score"].mean()),
            "max_risk_score": float(df["risk_score"].max()),
            "weights": self.weights,
            "thresholds": self.thresholds,
        }
        summary_path = output_path / "risk_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Risk Scoring (Stage 6): Heuristic weighted risk scoring with explainability"
    )
    parser.add_argument(
        "--features-csv", required=True, help="Path to features CSV (from Stage 5)"
    )
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--proximity-weight",
        type=float,
        default=WEIGHTS["proximity"],
        help=f"Proximity weight (default: {WEIGHTS['proximity']})",
    )
    parser.add_argument(
        "--density-weight",
        type=float,
        default=WEIGHTS["density"],
        help=f"Density weight (default: {WEIGHTS['density']})",
    )
    parser.add_argument(
        "--growth-weight",
        type=float,
        default=WEIGHTS["growth"],
        help=f"Growth weight (default: {WEIGHTS['growth']})",
    )
    parser.add_argument(
        "--condition-weight",
        type=float,
        default=WEIGHTS["condition"],
        help=f"Condition weight (default: {WEIGHTS['condition']})",
    )
    parser.add_argument(
        "--high-threshold",
        type=float,
        default=THRESHOLDS["high"],
        help=f"High risk threshold (default: {THRESHOLDS['high']})",
    )
    parser.add_argument(
        "--medium-threshold",
        type=float,
        default=THRESHOLDS["medium"],
        help=f"Medium risk threshold (default: {THRESHOLDS['medium']})",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "json", "both"],
        default="both",
        help="Output format (default: both)",
    )

    args = parser.parse_args()

    # Build custom weights
    weights = {
        "proximity": args.proximity_weight,
        "density": args.density_weight,
        "growth": args.growth_weight,
        "condition": args.condition_weight,
    }

    thresholds = {
        "high": args.high_threshold,
        "medium": args.medium_threshold,
        "low": 0.0,
    }

    # Load features
    df = pd.read_csv(args.features_csv)

    # Compute risk scores
    scorer = RiskScorer(weights=weights, thresholds=thresholds)
    risk_df = scorer.compute_risk_score(df)

    # Save results
    scorer.save_risk_scores(risk_df, args.output_dir, format=args.format)

    # Print summary
    print(f"\nRisk Scoring Complete:")
    print(f"  Total segments: {len(risk_df)}")
    print(f"  High risk: {(risk_df['risk_category'] == 'High').sum()}")
    print(f"  Medium risk: {(risk_df['risk_category'] == 'Medium').sum()}")
    print(f"  Low risk: {(risk_df['risk_category'] == 'Low').sum()}")
    print(f"  Mean risk score: {risk_df['risk_score'].mean():.3f}")
    print(f"  Max risk score: {risk_df['risk_score'].max():.3f}")


if __name__ == "__main__":
    main()