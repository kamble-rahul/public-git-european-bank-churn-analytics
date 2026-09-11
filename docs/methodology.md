# Analytical methodology

## 1. Ingestion and validation

Load the first worksheet with `pandas.read_excel`. Validate the required schema before calculating
any metric. Report quality findings instead of silently removing or correcting questionable values.

## 2. Cleaning and preparation

- Convert numeric fields with invalid text becoming missing values for review.
- Trim categorical strings and label missing categories as `Unknown`.
- Keep `CustomerId` only for drill-down; exclude `Surname` from analysis and modeling.
- Exclude a constant `Year` field from modeling.
- Retain only valid binary target rows for churn calculations.

## 3. Segment definitions

The brief did not define every boundary. These explicit assumptions are mutually exclusive:

| Dimension | Definition |
|---|---|
| Age | `<30`, `30–45`, `46–59`, `60+` |
| Credit score | Low `<580`, Medium `580–669`, High `670+` |
| Tenure | New `0–2`, Mid-term `3–6`, Long-term `7–10` |
| Balance | Zero `=0`; Low `>0` through median positive balance; High above the median |
| High value | Balance at or above the selected percentile; default is 75th |

The source brief's `46–60` and `60+` labels overlap at 60. This implementation uses `46–59` and
`60+` so each person belongs to exactly one age group.

## 4. KPI formulas

- **Overall churn rate** = all churners / all customers.
- **Segment churn rate** = churners in segment / customers in segment.
- **Churn contribution** = churners in segment / all churners in the filtered population.
- **High-value churn ratio** = high-value churners / all high-value customers.
- **Geographic risk index** = geography churn rate / overall churn rate.
- **Engagement risk ratio** = inactive churn rate / active churn rate.
- **Balance at risk** = sum of balances held by churned high-value customers.

Always present a rate with its sample size. Do not average previously calculated percentages.

## 5. Descriptive analysis

Compare churn across geography, age, tenure, products, activity, gender, credit, and balance.
Use a geography × age matrix to identify interactions. Compare average financial profiles for
retained and churned customers. Treat all differences as associations, not causes.

## 6. Modeling

The target is binary, so Logistic Regression is the explainable baseline and Random Forest is the
nonlinear comparison model.

1. Use numeric and categorical features listed in `config.py`.
2. Create one 80/20 stratified split with `random_state=42`.
3. Median-impute and standardize numeric features for Logistic Regression.
4. Most-frequent-impute and one-hot encode categories.
5. Use a single scikit-learn pipeline for preprocessing and each classifier.
6. Apply balanced class weights.
7. Compare ROC-AUC, PR-AUC, recall, precision, F1, accuracy, and confusion matrix.
8. Select a decision threshold from campaign capacity and the relative cost of false negatives and
   false positives—not from accuracy alone.

## 7. Quality assurance

Synthetic tests check schema errors, binary consistency, band boundaries, all-zero balances, KPI
denominators, segment reconciliation, model outputs, and threshold validation. GitHub Actions runs
the checks on Python 3.11 and 3.12.
