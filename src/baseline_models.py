"""Baseline model experiments for churn classification."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_STATE, REPORTS_DIR, TARGET_COLUMN, VISUALS_DIR
from src.ml_helpers import (
    build_pipeline,
    evaluate_binary_classifier,
    load_cleaned_data,
    make_feature_target_split,
    make_train_test_split,
    save_metrics_row,
)
from src.utils import ensure_directory, print_section_header


def baseline_models() -> dict[str, object]:
    """Return the baseline estimators to compare."""
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "knn": KNeighborsClassifier(),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def plot_baseline_comparison(metrics_df: pd.DataFrame, path: Path) -> Path:
    """Create a grouped bar chart for ROC-AUC and F1."""
    ensure_directory(path.parent)
    order = metrics_df.sort_values("roc_auc", ascending=False).reset_index(drop=True)
    x = range(len(order))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar([i - width / 2 for i in x], order["roc_auc"], width=width, label="ROC-AUC", color="#2a6fdb")
    ax.bar([i + width / 2 for i in x], order["f1"], width=width, label="F1", color="#f28e2b")
    ax.set_xticks(list(x))
    ax.set_xticklabels(order["model"], rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Baseline Model Comparison")
    ax.legend()
    for idx, row in order.iterrows():
        ax.text(idx - width / 2, row["roc_auc"] + 0.005, f"{row['roc_auc']:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(idx + width / 2, row["f1"] + 0.005, f"{row['f1']:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> pd.DataFrame:
    """Run the baseline benchmark and persist metrics."""
    print_section_header("Baseline Model Benchmarking")
    df = load_cleaned_data()
    X, y = make_feature_target_split(df)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)

    results = []
    for model_name, estimator in baseline_models().items():
        pipeline = build_pipeline(estimator, X_train)
        pipeline.fit(X_train, y_train)
        metrics = evaluate_binary_classifier(pipeline, X_test, y_test)
        results.append(save_metrics_row(metrics, model_name))
        print(f"Evaluated baseline model: {model_name} | ROC-AUC={metrics['roc_auc']:.4f} | F1={metrics['f1']:.4f}")

    metrics_df = pd.concat(results, ignore_index=True).sort_values("roc_auc", ascending=False)
    ensure_directory(REPORTS_DIR)
    metrics_path = REPORTS_DIR / "baseline_model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    plot_baseline_comparison(metrics_df, VISUALS_DIR / "baseline_model_comparison.png")

    best_row = metrics_df.iloc[0]
    print(f"Best baseline model: {best_row['model']} (ROC-AUC={best_row['roc_auc']:.4f})")
    return metrics_df


if __name__ == "__main__":
    main()
