"""Reusable machine learning helpers for leakage-safe benchmarking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import ID_COLUMN, NUMERIC_COLUMNS, ORIGINAL_TARGET_COLUMN, PROCESSED_DATA_FILE, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE
from src.utils import ensure_directory, get_feature_names_from_column_transformer


def load_cleaned_data(path: Path = PROCESSED_DATA_FILE) -> pd.DataFrame:
    """Load the cleaned modeling dataset."""
    return pd.read_csv(path)


def make_feature_target_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create features and target without leakage-prone columns."""
    drop_cols = [col for col in [ID_COLUMN, ORIGINAL_TARGET_COLUMN, TARGET_COLUMN] if col in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def infer_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Infer numeric and categorical feature lists from the modeling frame."""
    numeric_cols = [col for col in NUMERIC_COLUMNS if col in X.columns]
    categorical_cols = [col for col in X.columns if col not in numeric_cols]
    return numeric_cols, categorical_cols


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Build a leakage-safe preprocessing pipeline."""
    numeric_cols, categorical_cols = infer_feature_types(X)
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", _make_one_hot_encoder()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor, numeric_cols, categorical_cols


def _make_one_hot_encoder() -> OneHotEncoder:
    """Create a dense one-hot encoder compatible with multiple sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - older sklearn versions
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """Create a stratified train/test split."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def build_pipeline(model: Any, X_train: pd.DataFrame) -> Pipeline:
    """Create a full preprocessing + estimator pipeline."""
    preprocessor, _, _ = build_preprocessor(X_train)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def predict_scores(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return probability-like scores for ROC-AUC and threshold analysis."""
    estimator = model.named_steps["model"]
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)
        if scores.ndim == 2:
            return scores[:, 1]
        return scores
    if hasattr(estimator, "decision_function"):
        scores = model.decision_function(X)
        return np.asarray(scores)
    predictions = model.predict(X)
    return np.asarray(predictions, dtype=float)


def evaluate_binary_classifier(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Compute standard binary classification metrics."""
    y_pred = model.predict(X_test)
    y_score = predict_scores(model, X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
    }


def classification_report_df(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    """Return a dataframe version of sklearn's classification report."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return pd.DataFrame(report).T.reset_index().rename(columns={"index": "label"})


def save_metrics_row(metrics: dict[str, float], model_name: str) -> pd.DataFrame:
    """Return a one-row dataframe of metrics for aggregation."""
    data = {"model": model_name}
    data.update(metrics)
    return pd.DataFrame([data])


def extract_preprocessed_feature_names(pipeline: Pipeline) -> list[str]:
    """Get feature names from a fitted pipeline."""
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_cols = [col for col in pipeline.feature_names_in_]
    return get_feature_names_from_column_transformer(preprocessor, feature_cols)


def get_positive_class_scores(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return positive class scores, preferring predict_proba."""
    return predict_scores(model, X)


def threshold_metrics(y_true: pd.Series, y_scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Evaluate precision, recall, and F1 at a classification threshold."""
    y_pred = (y_scores >= threshold).astype(int)
    return {
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def plot_metric_comparison(df: pd.DataFrame, metric: str, title: str, path: Path) -> Path:
    """Plot a bar chart for a single metric across models."""
    ensure_directory(path.parent)
    fig, ax = plt.subplots(figsize=(10, 5))
    order = df.sort_values(metric, ascending=False)
    ax.bar(order["model"], order[metric], color="#2a6fdb")
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.tick_params(axis="x", rotation=30)
    for index, value in enumerate(order[metric]):
        ax.text(index, value + 0.005, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_threshold_tradeoff(df: pd.DataFrame, path: Path) -> Path:
    """Plot precision/recall/F1 across thresholds."""
    ensure_directory(path.parent)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["threshold"], df["precision"], marker="o", label="Precision")
    ax.plot(df["threshold"], df["recall"], marker="o", label="Recall")
    ax.plot(df["threshold"], df["f1"], marker="o", label="F1")
    if "predicted_positive_rate" in df.columns:
        ax.plot(df["threshold"], df["predicted_positive_rate"], marker="o", label="Predicted positive rate", linestyle="--")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Precision-Recall Tradeoff")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def roc_pr_curves(y_true: pd.Series, y_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ROC and PR curve coordinates."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return fpr, tpr, precision, recall, y_true.to_numpy(), y_scores
