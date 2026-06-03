"""Central configuration for paths and experiment settings."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
VISUALS_DIR = PROJECT_ROOT / "visuals"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

RAW_DATA_FILE = RAW_DATA_DIR / "telco_customer_data_v2.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "telco_churn_ml_cleaned.csv"
CLEANING_SUMMARY_FILE = REPORTS_DIR / "cleaning_summary.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.25

TARGET_COLUMN = "ChurnBinary"
ID_COLUMN = "customerID"
ORIGINAL_TARGET_COLUMN = "Churn"

NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
BASE_CATEGORICAL_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

