# Model Interpretation

The most influential churn drivers are highlighted below based on permutation importance, with model-specific attribution used when available.

## Key Features
- Contract
- Partner
- OnlineBackup
- Dependents
- tenure
- SeniorCitizen
- TotalCharges
- StreamingMovies
- MonthlyCharges
- MultipleLines

## Plain-English Summary
- Contract type, tenure, monthly charges, and service bundle variables are commonly among the strongest churn signals in telco classification problems.
- Model interpretation should be read alongside benchmark metrics and threshold analysis, because a feature can be predictive without being a causal driver.
- Permutation importance provides the most reliable model-agnostic explanation in this repository, while SHAP is treated as an optional enhancement.