"""Hyperparameter tuning for XGBoost churn models."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.model_selection import RandomizedSearchCV

from src.config import MODELS_DIR, RANDOM_STATE, REPORTS_DIR
from src.ml_helpers import (
    build_pipeline,
    evaluate_binary_classifier,
    load_cleaned_data,
    make_feature_target_split,
    make_train_test_split,
)
from src.utils import ensure_directory, print_section_header

try:  # pragma: no cover - import guard for environments without xgboost
    from xgboost import XGBClassifier

    HAVE_XGBOOST = True
except Exception:  # pragma: no cover - import guard for environments without xgboost
    XGBClassifier = None
    HAVE_XGBOOST = False


def main() -> pd.DataFrame | None:
    """Tune XGBoost if available and persist the outputs."""
    print_section_header("XGBoost Tuning")
    if not HAVE_XGBOOST:
        print("XGBoost is not installed in this environment, so tuning was skipped cleanly.")
        return None

    df = load_cleaned_data()
    X, y = make_feature_target_split(df)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)

    estimator = XGBClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss",
        tree_method="hist",
        verbosity=0,
    )
    pipeline = build_pipeline(estimator, X_train)

    param_distributions = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [3, 4, 5],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=10,
        scoring="roc_auc",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    search.fit(X_train, y_train)

    cv_results = pd.DataFrame(search.cv_results_)
    cv_results = cv_results.sort_values("rank_test_score")
    cv_results.to_csv(REPORTS_DIR / "xgboost_tuning_results.csv", index=False)

    best_params = pd.DataFrame([search.best_params_])
    best_params.to_csv(REPORTS_DIR / "best_xgboost_params.csv", index=False)

    best_model = search.best_estimator_
    dump(best_model, MODELS_DIR / "tuned_xgboost.joblib")

    tuned_metrics = evaluate_binary_classifier(best_model, X_test, y_test)
    pd.DataFrame([{"model": "tuned_xgboost", **tuned_metrics}]).to_csv(REPORTS_DIR / "tuned_xgboost_metrics.csv", index=False)

    benchmark_metrics_path = REPORTS_DIR / "model_benchmark_metrics.csv"
    benchmark_best_roc_auc = None
    benchmark_best_model = None
    if benchmark_metrics_path.exists():
        benchmark_df = pd.read_csv(benchmark_metrics_path)
        if not benchmark_df.empty and "roc_auc" in benchmark_df.columns:
            benchmark_best = benchmark_df.sort_values("roc_auc", ascending=False).iloc[0]
            benchmark_best_roc_auc = float(benchmark_best["roc_auc"])
            benchmark_best_model = str(benchmark_best["model"])

    promoted = False
    if benchmark_best_roc_auc is None or tuned_metrics["roc_auc"] > benchmark_best_roc_auc:
        dump(best_model, MODELS_DIR / "best_model.joblib")
        promoted = True
        best_model_note = (
            "Tuned XGBoost became the new best model because it achieved a higher ROC-AUC "
            f"than the benchmark best model."
        )
    else:
        best_model_note = (
            "Tuned XGBoost did not beat the benchmark best model by ROC-AUC, so "
            "models/best_model.joblib was left unchanged."
        )

    (REPORTS_DIR / "xgboost_benchmark_note.md").write_text(
        "\n".join(
            [
                f"Benchmark best model: {benchmark_best_model or 'unknown'}",
                f"Benchmark best ROC-AUC: {benchmark_best_roc_auc:.4f}" if benchmark_best_roc_auc is not None else "Benchmark best ROC-AUC: unavailable",
                f"Tuned XGBoost ROC-AUC: {tuned_metrics['roc_auc']:.4f}",
                best_model_note,
            ]
        ),
        encoding="utf-8",
    )

    print(f"Best XGBoost CV ROC-AUC: {search.best_score_:.4f}")
    print(f"Best XGBoost parameters: {search.best_params_}")
    print(f"Test ROC-AUC after tuning: {tuned_metrics['roc_auc']:.4f}")
    if promoted:
        print("Tuned XGBoost promoted to best_model.joblib.")
    else:
        print("Tuned XGBoost did not replace the benchmark best model.")
    return cv_results


if __name__ == "__main__":
    main()
