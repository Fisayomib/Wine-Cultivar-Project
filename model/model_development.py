"""
Wine Cultivar Origin Prediction - Model Development
---------------------------------------------------
Trains a multiclass classifier on the UCI/Sklearn Wine dataset using a
selected subset of numerical chemical features. The resulting pipeline
includes preprocessing (scaling) and is persisted to disk for use by the
web app.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Destination for the trained model
MODEL_PATH = Path("artifacts/wine_cultivar_model.joblib")

# Choose any six features from the allowed list (excluding target)
SELECTED_FEATURES: List[str] = [
    "alcohol",
    "malic_acid",
    "alcalinity_of_ash",
    "flavanoids",
    "color_intensity",
    "proline",
]


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the wine dataset and return features and target."""
    wine = load_wine(as_frame=True)
    df: pd.DataFrame = wine.frame

    # Keep only selected features and target
    features = df[SELECTED_FEATURES].copy()
    target = df["target"]
    return features, target


def preprocess(features: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values (if any) and return cleaned dataframe."""
    # The sklearn wine dataset has no missing values, but guard anyway.
    cleaned = features.copy()
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.dropna()
    return cleaned


def build_pipeline() -> Pipeline:
    """Create the model pipeline with scaling + classifier."""
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", gamma="scale", probability=True)),
        ]
    )
    return pipeline


def evaluate_model(
    model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> dict:
    """Compute evaluation metrics for the trained model."""
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    report = metrics.classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "classification_report": report,
    }


def save_model(model: Pipeline, model_path: Path = MODEL_PATH) -> None:
    """Persist the trained pipeline and feature metadata."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "features": SELECTED_FEATURES},
        model_path,
    )
    print(f"Model saved to {model_path.resolve()}")


def main() -> None:
    # Load and preprocess data
    X_raw, y = load_data()
    X = preprocess(X_raw)

    # Align target with any dropped rows during preprocessing
    y_aligned = y.loc[X.index]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_aligned, test_size=0.2, stratify=y_aligned, random_state=42
    )

    # Build and train model
    model = build_pipeline()
    model.fit(X_train, y_train)

    # Evaluate
    metrics_dict = evaluate_model(model, X_test, y_test)
    print("Model evaluation:")
    print(json.dumps(metrics_dict, indent=2))

    # Save model
    save_model(model)


if __name__ == "__main__":
    main()
