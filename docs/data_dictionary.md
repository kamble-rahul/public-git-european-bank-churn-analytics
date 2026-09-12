# Data dictionary

The private source workbook and bundled standardized dataset use the following fields.

| Column | Type | Meaning | Analytical use |
|---|---|---|---|
| `CustomerId` | Identifier | Generated `DEMO-xxxxx` record ID in the public asset | Drill-down only; excluded from ML |
| `Surname` | Text | `Anonymous` placeholder in the public asset | Excluded from analysis and ML |
| `CreditScore` | Numeric | Creditworthiness score | Credit bands and ML feature |
| `Geography` | Category | France, Germany, or Spain | Regional segmentation and ML feature |
| `Gender` | Category | Female or Male in the supplied data | Fairness-sensitive analysis and ML feature |
| `Age` | Numeric | Customer age | Age bands and ML feature |
| `Tenure` | Numeric | Years with the bank | Tenure bands and ML feature |
| `Balance` | Numeric | Account balance | Balance bands, high-value analysis, ML feature |
| `NumOfProducts` | Integer | Number of products held | Engagement analysis and ML feature |
| `HasCrCard` | Binary | 1 if customer has a credit card | ML feature |
| `IsActiveMember` | Binary | 1 if customer is active | Engagement KPI and ML feature |
| `EstimatedSalary` | Numeric | Estimated annual salary | Financial profile and ML feature |
| `Exited` | Binary | 1 if customer churned | KPI numerator and supervised target |

An optional `Year` field may be present. In the supplied workbook it is constant, so it is not a
useful predictive feature.

## Validation rules

- Required columns must exist and the worksheet must contain rows.
- `HasCrCard`, `IsActiveMember`, and `Exited` must contain only 0 or 1.
- Numeric fields must be convertible to numbers.
- `CustomerId` duplicates, missing cells, negative balances, non-positive/non-integer product
  counts, unusual ages, and unexpected categories are reported for review.
- Rows with an invalid `Exited` value cannot be used to calculate churn or train a classifier.

## Data protection

The private Excel source and ordinary CSV exports are excluded by `.gitignore`. The sole exception
is `data/processed/european_bank_dashboard.csv.gz`, built by the project's de-identification script.
Keep authorized source data in `data/raw/` locally and never commit original names, original
identifiers, credentials, or unapproved model artifacts.
