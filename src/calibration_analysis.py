"""Calibration analysis for the selected churn model."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from src.config import MODELS_DIR, REPORTS_DIR, VISUALS_DIR
from src.ml_helpers import load_cleaned_data, make_feature_target_split, make_train_test_split, predict_scores
from src.utils import ensure_directory, print_section_header


def plot_calibration_curve(y_true: pd.Series, y_scores: np.ndarray, path: Path) -> tuple[float, float]:
    """Plot calibration curve and return summary calibration diagnostics."""
    ensure_directory(path.parent)
    frac_pos, mean_pred = calibration_curve(y_true, y_scores, n_bins=10, strategy="quantile")
    brier = brier_score_loss(y_true, y_scores)
    calibration_gap = float(np.mean(np.abs(frac_pos - mean_pred))) if len(frac_pos) else float("nan")

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(mean_pred, frac_pos, marker="o", label="Model", color="#2a6fdb")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_title("Calibration Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return brier, calibration_gap


def main() -> pd.DataFrame:
    """Evaluate calibration on the holdout test split."""
    print_section_header("Calibration Analysis")
    df = load_cleaned_data()
    X, y = make_feature_target_split(df)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)

    model = load(MODELS_DIR / "best_model.joblib")
    model.fit(X_train, y_train)
    y_scores = predict_scores(model, X_test)

    brier, calibration_gap = plot_calibration_curve(y_test, y_scores, VISUALS_DIR / "calibration_curve.png")
    calibration_df = pd.DataFrame(
        [
            {
                "model": "best_model",
                "brier_score": brier,
                "mean_absolute_calibration_gap": calibration_gap,
                "mean_predicted_probability": float(np.mean(y_scores)),
                "observed_positive_rate": float(np.mean(y_test)),
            }
        ]
    )
    calibration_df.to_csv(REPORTS_DIR / "calibration_metrics.csv", index=False)

    if calibration_gap <= 0.05 and brier <= 0.20:
        interpretation = (
            "The model probabilities appear reasonably calibrated for a portfolio benchmark. "
            f"The Brier score is {brier:.4f}, and the average calibration gap across quantile bins is {calibration_gap:.4f}."
        )
    else:
        interpretation = (
            "The model probabilities are only moderately calibrated, so probability outputs should be used carefully. "
            f"The Brier score is {brier:.4f}, and the average calibration gap across quantile bins is {calibration_gap:.4f}. "
            "A calibration step such as isotonic regression or Platt scaling could be considered if well-calibrated probabilities are required."
        )

    (REPORTS_DIR / "calibration_interpretation.md").write_text(interpretation, encoding="utf-8")
    print(f"Brier score: {brier:.4f}")
    print(f"Mean calibration gap: {calibration_gap:.4f}")
    print(interpretation)
    return calibration_df


if __name__ == "__main__":
    main()
