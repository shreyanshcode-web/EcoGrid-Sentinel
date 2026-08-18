#!/usr/bin/env python3
"""Train and evaluate a risk classifier on SIH1379 Indian training data.

The model uses only the supplied observation features (NDVI, vegetation,
distance and proximity) and writes a versioned artefact plus metrics.  It is
an inspection-priority prototype, not an outage/contact prediction model.
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import train_test_split

FEATURES = ["DistanceToLine", "NDVI", "ProximityFactor", "Vegetation", "VegetationFactor"]

def train(dataset_path: str, output_dir: str, random_state: int = 42):
    df = pd.read_csv(dataset_path)
    required = FEATURES + ["RiskLabel", "latitude", "longitude"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df.dropna(subset=required).copy()
    X, y = df[FEATURES], df["RiskLabel"].astype(int)
    # Stratification keeps the rare high-risk class represented in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=400, class_weight="balanced_subsample", min_samples_leaf=2,
        random_state=random_state, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "dataset": Path(dataset_path).name}, output / "india_risk_model.joblib")
    metrics = {
        "dataset": Path(dataset_path).name, "rows_used": int(len(df)),
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "feature_importance": {name: float(value) for name, value in zip(FEATURES, model.feature_importances_)},
    }
    (output / "india_risk_model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"model": str(output / "india_risk_model.joblib"), "balanced_accuracy": metrics["balanced_accuracy"]}, indent=2))
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/SIH1379_ML_Training_Dataset.csv")
    parser.add_argument("--output-dir", default="data/models")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    train(args.dataset, args.output_dir, args.random_state)
