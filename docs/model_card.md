# Model card

## Model overview

| Item | Logistic Regression | Random Forest |
|---|---|---|
| Role | Explainable baseline | Main nonlinear comparison |
| Target | `Exited = 1` | `Exited = 1` |
| Class handling | Balanced weights | Balanced weights |
| Output | Churn probability | Churn probability |
| Interpretation | Signed coefficients and local contributions | Holdout permutation and impurity importance |
| Model version | 1.1.0 | 1.1.0 |

## Intended use

- Educational analysis of churn patterns.
- Prioritizing customers for a reviewed retention experiment.
- Comparing model families and probability thresholds.

## Out-of-scope use

- Denying products, changing prices, limiting service, or making adverse decisions.
- Treating a score as proof that a customer will churn.
- Production use without data-owner, privacy, risk, legal, and model-governance approval.

## Features deliberately excluded

- `CustomerId`: identifier with no stable behavioral meaning.
- `Surname`: personal identifier and a leakage/fairness risk.
- Constant `Year`: contains no variance in the supplied snapshot.

## Evaluation

Accuracy is reported but must not be used alone because the target is imbalanced. ROC-AUC measures
ranking over both classes; PR-AUC emphasizes the churn class; recall measures the share of actual
churners found; precision measures how many flagged customers actually churned; F1 balances recall
and precision.

The dashboard evaluates the final models on one stratified holdout set and five stratified folds.
It reports calibration curves, Brier score, mean fold performance, and fold variability. The
Random Forest achieved five-fold ROC-AUC `0.861 ± 0.008` and PR-AUC `0.680 ± 0.019` on the supplied
snapshot. Each run exposes the model version, random seed, row counts, and a short dataset
fingerprint.

A production project should still add temporal validation, calibrated probability correction,
confidence intervals, independent validation data, and a champion/challenger approval process.

## Fairness and risk

Gender and geography are included in the educational comparison because the project explicitly asks
for demographic and regional patterns. The dashboard compares predicted high-risk rate, precision,
recall, false-positive rate, and false-negative rate across gender, geography, and age groups.
Differences are screening signals, not a declaration that the model is fair. Before operational
use, evaluate whether sensitive or proxy variables should be removed and document the legal basis
for each feature.

Feature importance is not causal explanation. A high importance value means the fitted model used a
feature to split records; it does not prove that changing the feature changes churn.

## Data and monitoring limitations

- Single snapshot with no customer event timeline.
- No stated reason for exit or campaign-treatment history.
- No authoritative dataset provenance in the workbook.
- No revenue, margin, fees, or customer-lifetime-value fields.
- No temporal drift, latency, treatment-effect, or outcome monitoring in the demo.
- Calibration is diagnosed but probabilities are not automatically recalibrated.
- ROI outputs depend on user assumptions and are not recognized revenue or causal estimates.
