"""Model interpretation and feature importance reporting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
from sklearn.inspection import permutation_importance

from src.config import MODELS_DIR, REPORTS_DIR, VISUALS_DIR
from src.ml_helpers import (
    extract_preprocessed_feature_names,
    load_cleaned_data,
    make_feature_target_split,
    make_train_test_split,
)
from src.utils import ensure_directory, print_section_header


def _save_bar_plot(df: pd.DataFrame, value_col: str, title: str, path: Path, top_n: int = 20, color: str = "#2a6fdb") -> Path:
    """Save a horizontal bar plot for the top features."""
    ensure_directory(path.parent)
    top_df = df.sort_values(value_col, ascending=False).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    ax.barh(top_df["feature"], top_df[value_col], color=color)
    ax.set_title(title)
    ax.set_xlabel(value_col.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_signed_bar_plot(df: pd.DataFrame, value_col: str, title: str, path: Path, top_n: int = 20) -> Path:
    """Save a signed coefficient plot."""
    ensure_directory(path.parent)
    top_df = df.reindex(df[value_col].abs().sort_values(ascending=False).head(top_n).index).iloc[::-1]
    colors = ["#d62728" if val < 0 else "#2a6fdb" for val in top_df[value_col]]
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    ax.barh(top_df["feature"], top_df[value_col], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Coefficient")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> pd.DataFrame:
    """Explain the selected model using multiple attribution methods."""
    print_section_header("Model Explanation")
    df = load_cleaned_data()
    X, y = make_feature_target_split(df)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)

    best_model = load(MODELS_DIR / "best_model.joblib")
    best_model.fit(X_train, y_train)
    estimator = best_model.named_steps["model"]
    feature_names = extract_preprocessed_feature_names(best_model)

    importance_df = pd.DataFrame()
    if hasattr(estimator, "feature_importances_"):
        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "feature_importance": estimator.feature_importances_,
            }
        ).sort_values("feature_importance", ascending=False)
        importance_df.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
        _save_bar_plot(importance_df, "feature_importance", "Top 20 Feature Importances", VISUALS_DIR / "feature_importance_top20.png")

    if estimator.__class__.__name__ == "LogisticRegression" or hasattr(estimator, "coef_"):
        coefficients = np.ravel(estimator.coef_)
        coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
        coef_df["absolute_coefficient"] = coef_df["coefficient"].abs()
        coef_df.sort_values("absolute_coefficient", ascending=False).to_csv(REPORTS_DIR / "logistic_coefficients.csv", index=False)
        _save_signed_bar_plot(coef_df.sort_values("absolute_coefficient", ascending=False), "coefficient", "Top Logistic Regression Coefficients", VISUALS_DIR / "logistic_coefficients_top20.png")

    # Model-agnostic permutation importance on a manageable test sample.
    sample_size = min(5000, len(X_test))
    sample_idx = X_test.sample(n=sample_size, random_state=42).index
    X_perm = X_test.loc[sample_idx]
    y_perm = y_test.loc[sample_idx]

    perm = permutation_importance(
        best_model,
        X_perm,
        y_perm,
        n_repeats=5,
        random_state=42,
        scoring="roc_auc",
        n_jobs=-1,
    )
    perm_df = pd.DataFrame(
        {
            "feature": X.columns,
            "mean_importance": perm.importances_mean,
            "std_importance": perm.importances_std,
        }
    ).sort_values("mean_importance", ascending=False)
    perm_df.to_csv(REPORTS_DIR / "permutation_importance.csv", index=False)
    _save_bar_plot(perm_df, "mean_importance", "Top 20 Permutation Importances", VISUALS_DIR / "permutation_importance_top20.png")

    shap_note_path = REPORTS_DIR / "shap_note.md"
    try:
        import shap  # type: ignore

        background = best_model.named_steps["preprocessor"].transform(X_train.sample(n=min(200, len(X_train)), random_state=42))
        shap_sample = best_model.named_steps["preprocessor"].transform(X_perm.sample(n=min(500, len(X_perm)), random_state=42))

        explainer = shap.Explainer(best_model.named_steps["model"], background)
        shap_values = explainer(shap_sample)
        shap.summary_plot(shap_values, shap_sample, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(VISUALS_DIR / "shap_summary.png", dpi=300, bbox_inches="tight")
        plt.close()
        shap_note_path.write_text("SHAP executed successfully on the fitted model. The summary plot was saved to visuals/shap_summary.png.", encoding="utf-8")
    except Exception as exc:  # pragma: no cover - environment-specific compatibility
        shap_note_path.write_text(
            "SHAP was not available or did not run reliably in this environment. "
            "Permutation importance was used as the dependable fallback for model explanation.\n\n"
            f"Technical note: {exc}",
            encoding="utf-8",
        )

    top_perm = perm_df.head(10)["feature"].tolist()
    top_lines = [
        "# Model Interpretation",
        "",
        "The most influential churn drivers are highlighted below based on permutation importance, with model-specific attribution used when available.",
        "",
        "## Key Features",
    ]
    for feature in top_perm:
        top_lines.append(f"- {feature}")
    top_lines.extend(
        [
            "",
            "## Plain-English Summary",
            "- Contract type, tenure, monthly charges, and service bundle variables are commonly among the strongest churn signals in telco classification problems.",
            "- Model interpretation should be read alongside benchmark metrics and threshold analysis, because a feature can be predictive without being a causal driver.",
            "- Permutation importance provides the most reliable model-agnostic explanation in this repository, while SHAP is treated as an optional enhancement.",
        ]
    )
    (REPORTS_DIR / "model_interpretation.md").write_text("\n".join(top_lines), encoding="utf-8")

    print("Top permutation importance features:")
    print(perm_df.head(10)[["feature", "mean_importance"]].to_string(index=False))
    print(f"Interpretation report saved to: {REPORTS_DIR / 'model_interpretation.md'}")
    return perm_df


if __name__ == "__main__":
    main()
