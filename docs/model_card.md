# Model card

## Model overview

| Item | Logistic Regression | Random Forest |
|---|---|---|
| Role | Explainable baseline | Main nonlinear comparison |
| Target | `Exited = 1` | `Exited = 1` |
| Class handling | Balanced weights | Balanced weights |
| Output | Churn probability | Churn probability |
| Interpretation | Coefficients after preprocessing | Impurity-based feature importance |

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

The dashboard evaluates the final models on one stratified holdout set. A production project should
add cross-validation, probability calibration, temporal validation, confidence intervals, and a
champion/challenger process.

## Fairness and risk

Gender and geography are included in the educational comparison because the project explicitly asks
for demographic and regional patterns. Before operational use, compare performance and error rates
across groups, evaluate whether sensitive or proxy variables should be removed, and document the
legal basis for each feature.

Feature importance is not causal explanation. A high importance value means the fitted model used a
feature to split records; it does not prove that changing the feature changes churn.

## Data and monitoring limitations

- Single snapshot with no customer event timeline.
- No stated reason for exit or campaign-treatment history.
- No authoritative dataset provenance in the workbook.
- No revenue, margin, fees, or customer-lifetime-value fields.
- No drift, calibration, latency, or fairness monitoring in the demo.
