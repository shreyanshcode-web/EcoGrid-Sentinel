#!/usr/bin/env python3
"""
Data Preparation for ML Training

Prepares training data from pipeline outputs:
- Loads features from feature engineering
- Generates/loads labels
- Creates train/val/test splits (spatially aware)
- Handles missing values, feature scaling
- Exports to format suitable for XGBoost/sklearn
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


# Feature columns to use (exclude non-predictive)
EXCLUDE_COLS = {
    "segment_id", "segment_geometry", "segment_idx", "line_idx",
    "geometry", "index", "level_0", "level_1",
}

# Target columns
TARGET_COLS = ["risk_score", "risk_category", "risk_category_id"]


def load_features(features_path: Union[str, Path]) -> pd.DataFrame:
    """Load features CSV from feature engineering stage."""
    df = pd.read_csv(features_path)
    print(f"Loaded features: {len(df)} segments, {len(df.columns)} columns")
    return df


def load_labels(labels_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load labels CSV."""
    df = pd.read_csv(labels_path)
    y_score = df["risk_score"].values
    y_category = df["risk_category_id"].values
    return y_score, y_category, df


def prepare_feature_matrix(
    features_df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
    target_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare clean feature matrix for ML.

    Args:
        features_df: Raw features dataframe
        exclude_cols: Additional columns to exclude
        target_cols: Target columns to exclude

    Returns:
        (X_df, feature_names) tuple
    """
    exclude = set(EXCLUDE_COLS)
    if exclude_cols:
        exclude.update(exclude_cols)
    if target_cols:
        exclude.update(target_cols)
    else:
        exclude.update(TARGET_COLS)

    # Select numeric columns only
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude]

    X = features_df[feature_cols].copy()

    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan)

    # Fill NaN with column median (robust to outliers)
    X = X.fillna(X.median())

    # Final check - fill any remaining with 0
    X = X.fillna(0)

    print(f"Prepared feature matrix: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Features: {feature_cols}")

    return X, feature_cols


def spatial_train_test_split(
    features_df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    spatial_col: str = "segment_id",
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Create spatially aware train/val/test split.

    Since segments from the same line are spatially correlated,
    we split by line_idx to avoid data leakage.

    Args:
        features_df: Features with line_idx column
        test_size: Fraction for test set
        val_size: Fraction for validation set (from remaining)
        spatial_col: Column to use for spatial grouping (line_idx)
        random_state: Random seed

    Returns:
        Dict with 'train', 'val', 'test' index arrays
    """
    if "line_idx" not in features_df.columns:
        # Fallback to random split with warning
        print("Warning: No line_idx column, using random split (may leak spatial info)")
        indices = np.arange(len(features_df))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state
        )
        train_idx, val_idx = train_test_split(
            train_idx, test_size=val_size/(1-test_size), random_state=random_state
        )
        return {"train": train_idx, "val": val_idx, "test": test_idx}

    # Group by line_idx
    lines = features_df["line_idx"].unique()
    n_lines = len(lines)

    # Split lines (not segments)
    n_test_lines = max(1, int(n_lines * test_size))
    n_val_lines = max(1, int(n_lines * val_size))

    np.random.seed(random_state)
    shuffled_lines = np.random.permutation(lines)

    test_lines = shuffled_lines[:n_test_lines]
    val_lines = shuffled_lines[n_test_lines:n_test_lines + n_val_lines]
    train_lines = shuffled_lines[n_test_lines + n_val_lines:]

    # Get segment indices for each split
    train_idx = features_df[features_df["line_idx"].isin(train_lines)].index.values
    val_idx = features_df[features_df["line_idx"].isin(val_lines)].index.values
    test_idx = features_df[features_df["line_idx"].isin(test_lines)].index.values

    print(f"Spatial split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test segments")
    print(f"  Lines: {len(train_lines)} train, {len(val_lines)} val, {len(test_lines)} test")

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def scale_features(
    X_train: pd.DataFrame,
    X_val: Optional[pd.DataFrame] = None,
    X_test: Optional[pd.DataFrame] = None,
    scaler_type: str = "standard",
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], object]:
    """
    Scale features using StandardScaler or RobustScaler.

    Tree-based models (XGBoost, RF) don't strictly need scaling,
    but it helps with neural networks and some linear models.

    Returns scaled DataFrames and fitted scaler.
    """
    from sklearn.preprocessing import StandardScaler, RobustScaler

    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaler: {scaler_type}")

    # Fit on train only
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    X_val_scaled = None
    if X_val is not None:
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )

    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def prepare_training_data(
    features_csv: Union[str, Path],
    labels_csv: Union[str, Path],
    output_dir: Union[str, Path],
    test_size: float = 0.2,
    val_size: float = 0.1,
    spatial_split: bool = True,
    scale: bool = False,
    scaler_type: str = "standard",
    random_state: int = 42,
) -> Dict:
    """
    Full data preparation pipeline.

    Args:
        features_csv: Path to features.csv from feature engineering
        labels_csv: Path to ml_labels.csv from label generation
        output_dir: Directory to save prepared data
        test_size: Test set fraction
        val_size: Validation set fraction
        spatial_split: Use spatial (by line) split
        scale: Whether to scale features
        scaler_type: Scaler type if scaling
        random_state: Random seed

    Returns:
        Dict with paths to saved files and metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    features_df = load_features(features_csv)
    y_score, y_category, labels_df = load_labels(labels_csv)

    # Align features and labels by segment_id
    if "segment_id" in features_df.columns and "segment_id" in labels_df.columns:
        merged = features_df.merge(labels_df[["segment_id", "risk_score", "risk_category_id"]],
                                   on="segment_id", how="inner")
        print(f"Merged {len(merged)} segments with labels")
    else:
        # Assume same order
        merged = features_df.copy()
        merged["risk_score"] = y_score
        merged["risk_category_id"] = y_category
        print(f"Assumed aligned order: {len(merged)} segments")

    # Prepare feature matrix
    X, feature_names = prepare_feature_matrix(merged)

    # Targets
    y_reg = merged["risk_score"].values  # Regression target
    y_clf = merged["risk_category_id"].values  # Classification target

    # Split
    if spatial_split:
        splits = spatial_train_test_split(merged, test_size, val_size, random_state=random_state)
    else:
        indices = np.arange(len(merged))
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)
        train_idx, val_idx = train_test_split(train_idx, test_size=val_size/(1-test_size), random_state=random_state)
        splits = {"train": train_idx, "val": val_idx, "test": test_idx}

    # Split data
    X_train, X_val, X_test = X.iloc[splits["train"]], X.iloc[splits["val"]], X.iloc[splits["test"]]
    y_reg_train, y_reg_val, y_reg_test = y_reg[splits["train"]], y_reg[splits["val"]], y_reg[splits["test"]]
    y_clf_train, y_clf_val, y_clf_test = y_clf[splits["train"]], y_clf[splits["val"]], y_clf[splits["test"]]

    # Scale if requested
    scaler = None
    if scale:
        X_train, X_val, X_test, scaler = scale_features(
            X_train, X_val, X_test, scaler_type
        )
        # Save scaler
        scaler_path = output_dir / "scaler.joblib"
        joblib.dump(scaler, scaler_path)
        print(f"Saved scaler to: {scaler_path}")

    # Save splits
    np.savez(
        output_dir / "train_data.npz",
        X=X_train.values, y_reg=y_reg_train, y_clf=y_clf_train,
        feature_names=np.array(feature_names)
    )
    np.savez(
        output_dir / "val_data.npz",
        X=X_val.values, y_reg=y_reg_val, y_clf=y_clf_val,
        feature_names=np.array(feature_names)
    )
    np.savez(
        output_dir / "test_data.npz",
        X=X_test.values, y_reg=y_reg_test, y_clf=y_clf_test,
        feature_names=np.array(feature_names)
    )

    # Save feature names
    with open(output_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    # Save metadata
    metadata = {
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "target_names": ["risk_score", "risk_category"],
        "class_distribution_train": {str(i): int(np.sum(y_clf_train == i)) for i in range(3)},
        "class_distribution_val": {str(i): int(np.sum(y_clf_val == i)) for i in range(3)},
        "class_distribution_test": {str(i): int(np.sum(y_clf_test == i)) for i in range(3)},
        "spatial_split": spatial_split,
        "scaled": scale,
        "scaler_type": scaler_type if scale else None,
        "random_state": random_state,
    }

    with open(output_dir / "data_meta.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nData preparation complete. Saved to: {output_dir}")
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Class dist (train): {metadata['class_distribution_train']}")

    return metadata


def load_prepared_data(data_dir: Union[str, Path], split: str = "train") -> Tuple:
    """Load prepared data split."""
    data_dir = Path(data_dir)
    data = np.load(data_dir / f"{split}_data.npz")
    X = data["X"]
    y_reg = data["y_reg"]
    y_clf = data["y_clf"]
    feature_names = data["feature_names"].tolist()
    return X, y_reg, y_clf, feature_names


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare ML training data")
    parser.add_argument("--features-csv", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--no-spatial-split", action="store_true")
    parser.add_argument("--scale", action="store_true")
    parser.add_argument("--scaler", default="standard", choices=["standard", "robust"])
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    prepare_training_data(
        features_csv=args.features_csv,
        labels_csv=args.labels_csv,
        output_dir=args.output_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        spatial_split=not args.no_spatial_split,
        scale=args.scale,
        scaler_type=args.scaler,
        random_state=args.random_state,
    )