"""Clean and prepare the raw Telco churn dataset for ML benchmarking."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CLEANING_SUMMARY_FILE,
    ORIGINAL_TARGET_COLUMN,
    PROCESSED_DATA_FILE,
    RAW_DATA_FILE,
    TARGET_COLUMN,
)
from src.utils import (
    ensure_directory,
    normalize_key,
    normalize_text_values,
    print_section_header,
)


YES_NO_COLUMNS = [
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaperlessBilling",
]

INTERNET_SERVICE_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def load_raw_data(path: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """Load the raw CSV using string dtype to keep messy values visible."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _map_gender(value: object) -> str:
    key = normalize_key(value)
    if key in {"female", "f"}:
        return "Female"
    if key in {"male", "m", "man"}:
        return "Male"
    return "Missing" if key in {None, "", "unknown", "none", "nan", "?"} else str(normalize_text_values(value))


def _map_senior_citizen(value: object) -> str:
    key = normalize_key(value)
    if key in {"1", "yes", "y", "true", "senior", "senior citizen"}:
        return "Yes"
    if key in {"0", "no", "n", "false", "not senior", "not senior citizen"}:
        return "No"
    return "Missing" if key in {None, "", "unknown", "none", "nan", "?"} else str(normalize_text_values(value))


def _map_yes_no(value: object) -> str:
    key = normalize_key(value)
    if key in {"yes", "y", "true", "1"}:
        return "Yes"
    if key in {
        "no",
        "n",
        "false",
        "0",
        "no internet service",
        "no phone service",
        "none",
        "not available",
    }:
        return "No"
    return "Missing" if key in {None, "", "unknown", "none", "nan", "?"} else str(normalize_text_values(value))


def _map_internet_service(value: object) -> str:
    key = normalize_key(value)
    if key in {"dsl", "dsl service"}:
        return "DSL"
    if key in {"fiber optic", "fiber", "fiberoptic"}:
        return "Fiber optic"
    if key in {"no", "no internet service", "none"}:
        return "No"
    return "Missing" if key in {None, "", "unknown", "none", "nan", "?"} else str(normalize_text_values(value))


def _map_contract(value: object) -> str:
    key = normalize_key(value)
    if key in {"m-m", "month-to-month", "month to month", "monthly"}:
        return "Month-to-month"
    if key in {"one year", "1 year", "1-year"}:
        return "One year"
    if key in {"two year", "2 year", "2-year"}:
        return "Two year"
    return "Missing" if key in {None, "", "unknown", "none", "nan", "?"} else str(normalize_text_values(value))


def _map_payment_method(value: object) -> str:
    key = normalize_key(value)
    if key in {"bank transfer", "bank transfer automatic", "bank transfer (automatic)", "bank transfer auto"}:
        return "Bank transfer (automatic)"
    if key in {"credit card", "credit card automatic", "credit card (automatic)", "credit card auto"}:
        return "Credit card (automatic)"
    if key in {"electronic check", "electronic cheque"}:
        return "Electronic check"
    if key in {"mailed check", "mailed cheque"}:
        return "Mailed check"
    return "Missing" if key in {None, "", "unknown", "none", "nan", "?"} else str(normalize_text_values(value))


def _clean_numeric(series: pd.Series, lower: float, upper: float, lower_inclusive: bool = True, upper_inclusive: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if lower_inclusive:
        valid = values >= lower
    else:
        valid = values > lower
    if upper_inclusive:
        valid &= values <= upper
    else:
        valid &= values < upper
    values = values.where(valid)
    return values


def _map_churn(value: object) -> str | None:
    key = normalize_key(value)
    if key in {"yes", "y", "churned", "churn", "1", "true"}:
        return "Yes"
    if key in {"no", "n", "no churn", "0", "false"}:
        return "No"
    if key in {"", "unknown", "none", "nan", "?"}:
        return None
    return None


def _service_inconsistency_mask(df: pd.DataFrame) -> pd.Series:
    internet_mask = df["InternetService"].eq("No")
    phone_mask = df["PhoneService"].eq("No")

    internet_cols_inconsistent = pd.Series(False, index=df.index)
    for col in INTERNET_SERVICE_COLUMNS:
        internet_cols_inconsistent |= internet_mask & ~df[col].eq("No")

    phone_inconsistent = phone_mask & ~df["MultipleLines"].eq("No")
    return internet_cols_inconsistent | phone_inconsistent


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply standardization, target cleanup, and consistency fixes."""
    df = df.copy()
    df.columns = [normalize_text_values(col) for col in df.columns]

    string_cols = [col for col in df.columns if col not in {"SeniorCitizen"}]
    for col in string_cols:
        df[col] = df[col].map(normalize_text_values)

    if "gender" in df.columns:
        df["gender"] = df["gender"].map(_map_gender)

    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].map(_map_senior_citizen)

    for col in YES_NO_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(_map_yes_no)

    if "InternetService" in df.columns:
        df["InternetService"] = df["InternetService"].map(_map_internet_service)
    if "Contract" in df.columns:
        df["Contract"] = df["Contract"].map(_map_contract)
    if "PaymentMethod" in df.columns:
        df["PaymentMethod"] = df["PaymentMethod"].map(_map_payment_method)

    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace({"": np.nan, "nan": np.nan, "None": np.nan}), errors="coerce")

    if "tenure" in df.columns:
        df["tenure"] = _clean_numeric(df["tenure"], 0, 72, lower_inclusive=True, upper_inclusive=True)
    if "MonthlyCharges" in df.columns:
        df["MonthlyCharges"] = _clean_numeric(df["MonthlyCharges"], 0, 300, lower_inclusive=False, upper_inclusive=True)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = _clean_numeric(df["TotalCharges"], 0, 21600, lower_inclusive=True, upper_inclusive=True)

    if "InternetService" in df.columns:
        no_internet = df["InternetService"].eq("No")
        for col in INTERNET_SERVICE_COLUMNS:
            df.loc[no_internet, col] = "No"
    if "PhoneService" in df.columns:
        no_phone = df["PhoneService"].eq("No")
        if "MultipleLines" in df.columns:
            df.loc[no_phone, "MultipleLines"] = "No"

    if ORIGINAL_TARGET_COLUMN in df.columns:
        churn_clean = df[ORIGINAL_TARGET_COLUMN].map(_map_churn)
        df[TARGET_COLUMN] = churn_clean.map({"Yes": 1, "No": 0})
        df = df.loc[df[TARGET_COLUMN].notna()].copy()
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
        df[ORIGINAL_TARGET_COLUMN] = df[TARGET_COLUMN].map({1: "Yes", 0: "No"})
    else:
        raise KeyError(f"Expected target column '{ORIGINAL_TARGET_COLUMN}' was not found.")

    df = df.drop_duplicates().reset_index(drop=True)
    return df


def build_cleaning_summary(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> pd.DataFrame:
    """Create a recruiter-readable cleaning summary table."""
    summary = {
        "rows_before": len(raw_df),
        "rows_after": len(clean_df),
        "duplicate_rows_remaining": int(clean_df.duplicated().sum()),
        "duplicate_customerID_remaining": int(clean_df["customerID"].duplicated().sum()) if "customerID" in clean_df.columns else np.nan,
        "invalid_tenure_remaining": int(clean_df["tenure"].notna().sum() - clean_df["tenure"].between(0, 72, inclusive="both").sum()) if "tenure" in clean_df.columns else np.nan,
        "invalid_MonthlyCharges_remaining": int(clean_df["MonthlyCharges"].notna().sum() - clean_df["MonthlyCharges"].between(0, 300, inclusive="right").sum()) if "MonthlyCharges" in clean_df.columns else np.nan,
        "invalid_TotalCharges_remaining": int(clean_df["TotalCharges"].notna().sum() - clean_df["TotalCharges"].between(0, 21600, inclusive="both").sum()) if "TotalCharges" in clean_df.columns else np.nan,
        "service_logic_inconsistencies_remaining": int(_service_inconsistency_mask(clean_df).sum()),
    }
    return pd.DataFrame([{"metric": key, "value": value} for key, value in summary.items()])


def save_outputs(clean_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    """Persist cleaned data and summary outputs."""
    ensure_directory(PROCESSED_DATA_FILE.parent)
    ensure_directory(CLEANING_SUMMARY_FILE.parent)
    clean_df.to_csv(PROCESSED_DATA_FILE, index=False)
    summary_df.to_csv(CLEANING_SUMMARY_FILE, index=False)


def main() -> pd.DataFrame:
    """Execute the full cleaning workflow."""
    print_section_header("Cleaning Raw Telco Data")
    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)
    summary_df = build_cleaning_summary(raw_df, clean_df)
    save_outputs(clean_df, summary_df)

    print(f"Raw rows: {len(raw_df):,}")
    print(f"Cleaned rows: {len(clean_df):,}")
    print(f"Saved cleaned data to: {PROCESSED_DATA_FILE}")
    print(f"Saved cleaning summary to: {CLEANING_SUMMARY_FILE}")
    return clean_df


if __name__ == "__main__":
    main()
