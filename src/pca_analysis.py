"""PCA experiments and feature-space analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.config import REPORTS_DIR, VISUALS_DIR
from src.ml_helpers import (
    build_preprocessor,
    load_cleaned_data,
    make_feature_target_split,
    make_train_test_split,
)
from src.utils import ensure_directory, print_section_header


def plot_explained_variance(pca_df: pd.DataFrame, path: Path) -> Path:
    """Plot explained variance and cumulative explained variance."""
    ensure_directory(path.parent)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(pca_df["component"], pca_df["explained_variance_ratio"], color="#2a6fdb", alpha=0.75, label="Explained variance")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_title("PCA Explained Variance")
    ax1.tick_params(axis="x", rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(pca_df["component"], pca_df["cumulative_explained_variance"], color="#f28e2b", marker="o", label="Cumulative variance")
    ax2.set_ylabel("Cumulative Explained Variance")
    ax2.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pca_scatter(pca_scores: np.ndarray, y_train: pd.Series, path: Path) -> Path:
    """Plot a 2D PCA scatter colored by churn."""
    ensure_directory(path.parent)
    fig, ax = plt.subplots(figsize=(9, 6))
    for target_value, label_name, color in [(0, "No Churn", "#2a6fdb"), (1, "Churn", "#d62728")]:
        mask = y_train.eq(target_value)
        ax.scatter(pca_scores[mask, 0], pca_scores[mask, 1], s=14, alpha=0.35, c=color, label=label_name, edgecolors="none")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("2D PCA Projection of Telco Churn Data")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> pd.DataFrame:
    """Run PCA on the training split and persist outputs."""
    print_section_header("PCA Analysis")
    df = load_cleaned_data()
    X, y = make_feature_target_split(df)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)

    preprocessor, _, _ = build_preprocessor(X_train)
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    n_components = min(X_train_preprocessed.shape[0], X_train_preprocessed.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    pca_scores_train = pca.fit_transform(X_train_preprocessed)
    pca_scores_test = pca.transform(X_test_preprocessed)

    explained = pd.DataFrame(
        {
            "component": [f"PC{i}" for i in range(1, len(pca.explained_variance_ratio_) + 1)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    explained.to_csv(REPORTS_DIR / "pca_explained_variance.csv", index=False)
    plot_explained_variance(explained.head(min(15, len(explained))), VISUALS_DIR / "pca_explained_variance.png")
    plot_pca_scatter(pca_scores_train[:, :2], y_train, VISUALS_DIR / "pca_churn_scatter.png")

    pc1 = explained.iloc[0]
    pc2 = explained.iloc[1] if len(explained) > 1 else None
    cumulative_two = explained.iloc[1]["cumulative_explained_variance"] if len(explained) > 1 else explained.iloc[0]["cumulative_explained_variance"]

    interpretation = [
        "# PCA Interpretation",
        "",
        f"- PC1 explains {pc1['explained_variance_ratio']:.2%} of the variance.",
        f"- The first two components explain {cumulative_two:.2%} of the variance combined.",
        "- The 2D PCA plot is useful for visual inspection, but PCA alone is not expected to fully separate churn classes.",
        "- For this churn problem, PCA should be treated as an exploratory visualization tool rather than the primary predictive representation.",
    ]
    (REPORTS_DIR / "pca_interpretation.md").write_text("\n".join(interpretation), encoding="utf-8")

    print(f"PC1 explained variance: {pc1['explained_variance_ratio']:.4f}")
    if pc2 is not None:
        print(f"PC2 explained variance: {pc2['explained_variance_ratio']:.4f}")
    print(f"PC1 + PC2 cumulative variance: {cumulative_two:.4f}")
    print("PCA is helpful for visualization, but the churn classes are not expected to be cleanly separable in 2D alone.")
    return explained


if __name__ == "__main__":
    main()
