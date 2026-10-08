# Validation evidence

The full analytical run was validated on 7 October 2026 using macOS and Python
3.14. The focused repository was checked again on 8 October 2026 after separating
automotive acquisition; all 15 delivery tests and artifact checks passed.

| Check | Observed result |
|---|---|
| Original Kaggle archive | ZIP verified; six required source tables extracted |
| Source fingerprinting | SHA-256 recorded for each analytical source table |
| Full narrative execution | All 10 code cells executed; no error outputs |
| Source cohort | 96,470 valid delivered orders; unique order IDs |
| Temporal label boundary | 75,099 training orders; 19,363 later test orders |
| Training-label maturity | Every retained training delivery occurs before test purchases |
| Numerical transformation | Train-fitted imputation, scaling and one-hot encoding |
| Feature selection / PCA | Training-only MI; 10 PCA components reach 90% variance |
| Regression tests | 15 passed, including malformed inputs and held-out invariance |
| Style checks | Ruff lint and formatting checks passed |
| Packaging | Standard Python wheel installed; package import and CLI help verified |
| Figures | 14 generated PNGs visually reviewed; readable titles/scales/legends |
| Interactive output | Self-contained Plotly HTML generated; embedded notebook output present |

The main risk benchmark has later-period ROC AUC 0.7150, average precision 0.0757
and Brier score 0.0331. MI selection's lower average precision remains documented.
Predictions and transformed outcomes do not feed training-time feature selection.

CI covers Python 3.11 and 3.14, including installation, the delivery regression
tests, Ruff lint/format checks and committed artifact consistency.
Published run results are [available in GitHub Actions](https://github.com/Pariamazaheri/commerce-delivery-intelligence/actions/workflows/quality.yml).
Local runtime evidence uses Python 3.14. This project deploys an analytical repository, not a live
prediction service. The Colab guide supports reproducing the analysis from GitHub.

## Independent audit

An independent recalculation from the original order timestamps confirmed 6,534
late deliveries out of 96,470 retained orders. All six original source-file hashes
match the recorded fingerprints. Customer, order-item, product and seller foreign
keys have no orphan records in this source snapshot.

The audit added explicit source schema checks, nonfinite/nonnumeric amount handling,
integer review-score validation and valid binomial denominators. Altering held-out
feature values leaves training MI rankings,
PCA coefficients and encoded training examples unchanged. These changes preserve
the measured Olist results while handling malformed future inputs more clearly.
