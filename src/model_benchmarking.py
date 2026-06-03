"""Model benchmarking workflow for comparing supervised learners."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from src.config import MODELS_DIR, RANDOM_STATE, REPORTS_DIR, VISUALS_DIR
from src.ml_helpers import (
    build_pipeline,
    evaluate_binary_classifier,
    load_cleaned_data,
    make_feature_target_split,
    make_train_test_split,
    plot_metric_comparison,
    save_metrics_row,
)
from src.utils import ensure_directory, print_section_header

try:  # pragma: no cover - import guard for environments without xgboost
    from xgboost import XGBClassifier

    HAVE_XGBOOST = True
except Exception:  # pragma: no cover - import guard for environments without xgboost
    XGBClassifier = None
    HAVE_XGBOOST = False


def benchmark_models() -> dict[str, object]:
    """Return the benchmark estimators to compare."""
    models = {
        "logistic_balanced": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest_300": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "knn": KNeighborsClassifier(),
    }
    if HAVE_XGBOOST:
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss",
            tree_method="hist",
            verbosity=0,
        )
    return models


def main() -> pd.DataFrame:
    """Run the stronger benchmark suite and persist outputs."""
    print_section_header("Model Benchmarking")
    df = load_cleaned_data()
    X, y = make_feature_target_split(df)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)

    ensure_directory(MODELS_DIR)
    results = []
    fitted_pipelines: dict[str, object] = {}

    for model_name, estimator in benchmark_models().items():
        pipeline = build_pipeline(estimator, X_train)
        pipeline.fit(X_train, y_train)
        fitted_pipelines[model_name] = pipeline
        metrics = evaluate_binary_classifier(pipeline, X_test, y_test)
        results.append(save_metrics_row(metrics, model_name))
        model_path = MODELS_DIR / f"{model_name}.joblib"
        dump(pipeline, model_path)
        print(f"Saved fitted pipeline: {model_path.name}")
        print(f"Evaluated {model_name} | ROC-AUC={metrics['roc_auc']:.4f} | F1={metrics['f1']:.4f} | Recall={metrics['recall']:.4f}")

    metrics_df = pd.concat(results, ignore_index=True).sort_values("roc_auc", ascending=False)
    metrics_df.to_csv(REPORTS_DIR / "model_benchmark_metrics.csv", index=False)

    plot_metric_comparison(metrics_df, "roc_auc", "ROC-AUC Comparison", VISUALS_DIR / "model_roc_auc_comparison.png")
    plot_metric_comparison(metrics_df, "f1", "F1 Comparison", VISUALS_DIR / "model_f1_comparison.png")
    plot_metric_comparison(metrics_df, "recall", "Recall Comparison", VISUALS_DIR / "model_recall_comparison.png")

    best_row = metrics_df.iloc[0]
    best_model_name = best_row["model"]
    best_model = fitted_pipelines[best_model_name]
    dump(best_model, MODELS_DIR / "best_model.joblib")

    print(f"Best benchmark model: {best_model_name}")
    print(f"Best benchmark ROC-AUC: {best_row['roc_auc']:.4f}")
    if not HAVE_XGBOOST:
        print("XGBoost was not available in this environment, so it was skipped cleanly.")
    return metrics_df


if __name__ == "__main__":
    main()
