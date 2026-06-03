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
        n_iter=15,
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

    print(f"Best XGBoost CV ROC-AUC: {search.best_score_:.4f}")
    print(f"Best XGBoost parameters: {search.best_params_}")
    print(f"Test ROC-AUC after tuning: {tuned_metrics['roc_auc']:.4f}")
    return cv_results


if __name__ == "__main__":
    main()
