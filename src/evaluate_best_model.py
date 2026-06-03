"""Final evaluation of the selected churn prediction model."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.config import MODELS_DIR, REPORTS_DIR, VISUALS_DIR
from src.ml_helpers import (
    classification_report_df,
    load_cleaned_data,
    make_feature_target_split,
    make_train_test_split,
    plot_threshold_tradeoff,
    predict_scores,
    threshold_metrics,
)
from src.utils import ensure_directory, print_section_header
from joblib import load


def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, path: Path) -> Path:
    """Save a confusion matrix figure."""
    ensure_directory(path.parent)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Best Model Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roc_curve(y_true: pd.Series, y_scores: np.ndarray, path: Path) -> Path:
    """Save ROC curve figure."""
    ensure_directory(path.parent)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}", color="#2a6fdb")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Best Model ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_precision_recall_curve(y_true: pd.Series, y_scores: np.ndarray, path: Path) -> Path:
    """Save precision-recall curve figure."""
    ensure_directory(path.parent)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#f28e2b")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Best Model Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> pd.DataFrame:
    """Evaluate the best model on the holdout test set."""
    print_section_header("Best Model Evaluation")
    df = load_cleaned_data()
    X, y = make_feature_target_split(df)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)

    best_model = load(MODELS_DIR / "best_model.joblib")
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    y_scores = predict_scores(best_model, X_test)

    metrics = {
        "model": "best_model",
        "accuracy": (y_pred == y_test).mean(),
        "precision": classification_report(y_test, y_pred, output_dict=True, zero_division=0)["1"]["precision"],
        "recall": classification_report(y_test, y_pred, output_dict=True, zero_division=0)["1"]["recall"],
        "f1": classification_report(y_test, y_pred, output_dict=True, zero_division=0)["1"]["f1-score"],
        "roc_auc": roc_auc_score(y_test, y_scores),
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(REPORTS_DIR / "best_model_metrics.csv", index=False)

    report_df = classification_report_df(y_test, y_pred)
    report_df.to_csv(REPORTS_DIR / "best_model_classification_report.csv", index=False)

    plot_confusion_matrix(y_test, y_pred, VISUALS_DIR / "best_model_confusion_matrix.png")
    plot_roc_curve(y_test, y_scores, VISUALS_DIR / "best_model_roc_curve.png")
    plot_precision_recall_curve(y_test, y_scores, VISUALS_DIR / "best_model_precision_recall_curve.png")

    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
    threshold_rows = [threshold_metrics(y_test, y_scores, threshold) for threshold in thresholds]
    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(REPORTS_DIR / "threshold_analysis.csv", index=False)
    plot_threshold_tradeoff(threshold_df, VISUALS_DIR / "threshold_precision_recall_tradeoff.png")

    recommendation_row = threshold_df.sort_values(["recall", "f1"], ascending=False).iloc[0]
    recommendation = (
        f"Recommended threshold for high churn recall: {recommendation_row['threshold']:.2f}. "
        f"This threshold provides recall={recommendation_row['recall']:.3f} and precision={recommendation_row['precision']:.3f}."
    )
    (REPORTS_DIR / "threshold_recommendation.md").write_text(recommendation, encoding="utf-8")

    print(f"Best model ROC-AUC: {metrics['roc_auc']:.4f}")
    print(recommendation)
    return metrics_df


if __name__ == "__main__":
    main()
