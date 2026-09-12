# Exploratory Data Analysis: European Bank Customer Churn

## Report scope

This report analyzes `European_Bank (5).xlsx` using the reusable functions in the project package.
It addresses the required customer-segmentation, churn-distribution, demographic, engagement,
financial-profile, high-value, and supervised-learning questions. Results describe associations in
this educational dataset; they do not prove why a customer churned.

- **Generated:** 2026-09-12
- **Rows:** 10,000
- **Columns in source:** 14
- **Target:** `Exited` (`1` = churned, `0` = retained)

## 1. Executive findings

- Overall churn is **20.37%** (2,037 of 10,000 customers).
- **Germany** has the highest country churn rate at
  **32.44%** and a geographic risk index of
  **1.59x** the portfolio average.
- The **46-59** age group has the highest age-band churn at
  **51.10%**.
- The highest geography-age intersection is **Germany ×
  46-59** at **67.28%**.
- Inactive customers have **1.88x** the churn rate of active
  customers.
- The high-value threshold is **127,644.24** (75th balance
  percentile). High-value churn is **23.68%**.
- Churned high-value customers hold **88,654,932.44** in
  balances, or **47.77%** of all churned-customer balance. This is an
  exposure measure, not revenue loss.
- Random Forest produces the stronger ranking result with ROC-AUC
  **0.861** versus **0.777** for Logistic
  Regression on the fixed holdout set.

## 2. Data quality and validation

| Check | Result |
| --- | --- |
| Required schema | Present |
| Missing cells in required fields | 0 |
| Duplicate CustomerId values | 0 |
| Retained customers | 7,963 |
| Churned customers | 2,037 |
| Validation messages | Year has one value only, so it is excluded from ML features. |

`Surname` is retained only for source traceability and is excluded from analytics and modeling.
`CustomerId` is used only in drill-down views. A constant `Year` field, when present, is excluded
from modeling because it adds no predictive variation.

## 3. Numeric profile

| Variable | Valid | Missing | Mean | Median | Std. dev. | Minimum | Maximum |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CreditScore | 10,000 | 0 | 650.53 | 652.00 | 96.65 | 350.00 | 850.00 |
| Age | 10,000 | 0 | 38.92 | 37.00 | 10.49 | 18.00 | 92.00 |
| Tenure | 10,000 | 0 | 5.01 | 5.00 | 2.89 | 0.00 | 10.00 |
| Balance | 10,000 | 0 | 76,485.89 | 97,198.54 | 62,397.41 | 0.00 | 250,898.09 |
| NumOfProducts | 10,000 | 0 | 1.53 | 1.00 | 0.58 | 1.00 | 4.00 |
| EstimatedSalary | 10,000 | 0 | 100,090.24 | 100,193.91 | 57,510.49 | 11.58 | 199,992.48 |

## 4. Churn distribution by required segments

Each table reports segment size, churn rate, and churn contribution. Churn contribution answers:
“What share of all churned customers came from this segment?” It must be read together with churn
rate because a small segment can have high risk but low portfolio impact.

### Geography

| Geography | Customers | Churners | Churn rate | Churn contribution | Average balance | Risk index |
| --- | --- | --- | --- | --- | --- | --- |
| France | 5,014 | 810 | 16.15% | 39.76% | 62,092.64 | 0.79x |
| Germany | 2,509 | 814 | 32.44% | 39.96% | 119,730.12 | 1.59x |
| Spain | 2,477 | 413 | 16.67% | 20.27% | 61,818.15 | 0.82x |

### Age group

| Age group | Customers | Churners | Churn rate | Churn contribution | Average balance |
| --- | --- | --- | --- | --- | --- |
| <30 | 1,641 | 124 | 7.56% | 6.09% | 73,698.72 |
| 30-45 | 6,248 | 956 | 15.30% | 46.93% | 75,951.26 |
| 46-59 | 1,585 | 810 | 51.10% | 39.76% | 81,926.98 |
| 60+ | 526 | 147 | 27.95% | 7.22% | 75,136.10 |

### Gender

| Gender | Customers | Churners | Churn rate | Churn contribution | Average balance |
| --- | --- | --- | --- | --- | --- |
| Female | 4,543 | 1,139 | 25.07% | 55.92% | 75,659.37 |
| Male | 5,457 | 898 | 16.46% | 44.08% | 77,173.97 |

### Tenure group

| Tenure group | Customers | Churners | Churn rate | Churn contribution | Average balance |
| --- | --- | --- | --- | --- | --- |
| New (0-2) | 2,496 | 528 | 21.15% | 25.92% | 78,053.98 |
| Mid-term (3-6) | 3,977 | 821 | 20.64% | 40.30% | 75,665.59 |
| Long-term (7-10) | 3,527 | 688 | 19.51% | 33.78% | 76,301.14 |

### Credit-score band

| Credit band | Customers | Churners | Churn rate | Churn contribution | Average balance |
| --- | --- | --- | --- | --- | --- |
| Low (<580) | 2,362 | 520 | 22.02% | 25.53% | 75,900.47 |
| Medium (580-669) | 3,331 | 685 | 20.56% | 33.63% | 76,919.46 |
| High (670+) | 4,307 | 832 | 19.32% | 40.84% | 76,471.62 |

### Balance segment

The median among positive balances is **119,839.69**. Zero
balances remain a separate business segment.

| Balance segment | Customers | Churners | Churn rate | Churn contribution | Average balance |
| --- | --- | --- | --- | --- | --- |
| High balance | 3,191 | 771 | 24.16% | 37.85% | 143,576.95 |
| Low balance | 3,192 | 766 | 24.00% | 37.60% | 96,085.47 |
| Zero balance | 3,617 | 500 | 13.82% | 24.55% | 0.00 |

### Engagement status

| Activity | Customers | Churners | Churn rate | Churn contribution | Average balance |
| --- | --- | --- | --- | --- | --- |
| Active | 5,151 | 735 | 14.27% | 36.08% | 75,875.42 |
| Inactive | 4,849 | 1,302 | 26.85% | 63.92% | 77,134.38 |

### Products held

| Products | Customers | Churners | Churn rate | Churn contribution | Average balance |
| --- | --- | --- | --- | --- | --- |
| 1 | 5,084 | 1,409 | 27.71% | 69.17% | 98,551.87 |
| 2 | 4,590 | 348 | 7.58% | 17.08% | 51,879.15 |
| 3 | 266 | 220 | 82.71% | 10.80% | 75,458.33 |
| 4 | 60 | 60 | 100.00% | 2.95% | 93,733.13 |

### Credit-card ownership

| Has credit card | Customers | Churners | Churn rate | Churn contribution | Average balance |
| --- | --- | --- | --- | --- | --- |
| 0 | 2,945 | 613 | 20.81% | 30.09% | 77,920.79 |
| 1 | 7,055 | 1,424 | 20.18% | 69.91% | 75,886.91 |

## 5. Geography and age interaction

| Geography | <30 | 30-45 | 46-59 | 60+ |
| --- | --- | --- | --- | --- |
| France | 5.20% | 11.87% | 45.47% | 24.82% |
| Germany | 12.63% | 25.30% | 67.28% | 41.46% |
| Spain | 7.94% | 12.52% | 40.83% | 21.71% |

The interaction table is more useful than reading geography and age independently. It helps locate
concentrated risk while preserving the sample-size context available in the dashboard.

## 6. Churned versus retained financial profile

| Status | Credit score | Age | Tenure | Balance | Products | Estimated salary |
| --- | --- | --- | --- | --- | --- | --- |
| Churned | 645.35 | 44.84 | 4.93 | 91,108.54 | 1.48 | 101,465.68 |
| Retained | 651.85 | 37.41 | 5.03 | 72,745.30 | 1.54 | 99,738.39 |

### Estimated-salary quartiles

| Salary quartile | Customers | Churners | Churn rate | Churn contribution | Average balance |
| --- | --- | --- | --- | --- | --- |
| Q1 (lowest) | 2,500 | 500 | 20.00% | 24.55% | 75,631.51 |
| Q2 | 2,500 | 495 | 19.80% | 24.30% | 75,923.89 |
| Q3 | 2,500 | 503 | 20.12% | 24.69% | 77,264.22 |
| Q4 (highest) | 2,500 | 539 | 21.56% | 26.46% | 77,123.94 |

Salary quartiles show whether churn changes consistently across salary levels. The Streamlit
high-value explorer adds an interactive salary-versus-balance scatter plot so the two measures can
be studied together instead of treating salary as customer value.

## 7. High-value customer churn

High value is defined transparently as balance at or above the selected percentile. The dashboard
allows the percentile to change from 50 to 95.

| Metric | Result |
| --- | --- |
| Balance threshold | 127,644.24 |
| High-value customers | 2,500 |
| High-value churners | 592 |
| High-value churn rate | 23.68% |
| Churned high-value balance | 88,654,932.44 |
| Share of all churned balance | 47.77% |

### High-value churn by geography

| Geography | Customers | Churners | Churn rate | Churn contribution | Average balance |
| --- | --- | --- | --- | --- | --- |
| France | 1,056 | 214 | 20.27% | 36.15% | 150,110.82 |
| Germany | 944 | 274 | 29.03% | 46.28% | 146,552.67 |
| Spain | 500 | 104 | 20.80% | 17.57% | 151,528.63 |

## 8. Numeric association with churn

Pearson correlation is a screening statistic for linear association. Small values do not rule out
nonlinear effects or interactions, and none of these coefficients establish causation.

| Feature | Correlation with Exited |
| --- | --- |
| Age | +0.285 |
| IsActiveMember | -0.156 |
| Balance | +0.119 |
| NumOfProducts | -0.048 |
| CreditScore | -0.027 |
| Tenure | -0.014 |
| EstimatedSalary | +0.012 |
| HasCrCard | -0.007 |

## 9. Machine-learning comparison

Both models use the same 80/20 stratified split (`random_state=42`). Numeric fields are imputed;
Logistic Regression also standardizes them. Geography and gender are imputed and one-hot encoded.
Both classifiers use balanced class weights. Identifiers, surname, derived segments, and constant
year are excluded from training.

| Model | ROC-AUC | PR-AUC | Recall | Precision | F1 | Accuracy | Confusion matrix |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.777 | 0.468 | 70.02% | 38.72% | 0.499 | 71.35% | TN=1142, FP=451, FN=122, TP=285 |
| Random Forest | 0.861 | 0.689 | 67.81% | 57.62% | 0.623 | 83.30% | TN=1390, FP=203, FN=131, TP=276 |

Accuracy alone is not sufficient because the majority-class baseline is **79.63%**.
Random Forest is the recommended demonstration model because it captures nonlinear patterns, while
Logistic Regression remains the explainable baseline. Neither model is ready for automated banking
decisions.

### Random Forest feature importance

| Feature | Importance |
| --- | --- |
| Age | 0.3269 |
| NumOfProducts | 0.2022 |
| Balance | 0.1155 |
| EstimatedSalary | 0.0745 |
| CreditScore | 0.0715 |
| IsActiveMember | 0.0557 |
| Geography_Germany | 0.0452 |
| Tenure | 0.0428 |
| Geography_France | 0.0161 |
| Gender_Female | 0.0149 |
| Gender_Male | 0.0143 |
| Geography_Spain | 0.0102 |

Feature importance describes model reliance, not causal influence. A production study should add
cross-validation or temporal validation, probability calibration, fairness analysis, drift checks,
and customer-contact cost assumptions.

## 10. Recommendations

1. Investigate the highest-risk geography and geography-age intersection through service, pricing,
   complaint, and channel data before designing a retention treatment.
2. Test re-engagement for inactive customers with a randomized control group and track incremental
   retention rather than contact volume alone.
3. Review the product journey and definitions for customers with three or four products. Very high
   churn rates in small groups require sample-size and data-quality checks.
4. Prioritize high-value churn cases for human review, but estimate revenue using fees, margin,
   cost-to-serve, and lifetime value rather than balance alone.
5. Select the model threshold from campaign capacity and the cost of false negatives versus false
   positives. Monitor precision, recall, calibration, complaints, and subgroup performance.

## 11. Limitations

- The workbook is a single snapshot, so it cannot measure behavior immediately before churn.
- Churn reason, transaction history, service interactions, marketing treatment, and product revenue
  are not available.
- Segment differences are observational associations and should not be presented as causes.
- Gender and geography require legal, ethical, privacy, and fairness review before operational use.
- Dataset provenance and currency are not documented in the supplied workbook.

## 12. Reproduce this report

From the project root:

```bash
python -m pip install -r requirements.txt
python scripts/generate_eda_report.py "/full/path/to/European_Bank (5).xlsx"
```

Or use the Makefile:

```bash
make eda WORKBOOK="/full/path/to/European_Bank (5).xlsx"
```

The default output is `reports/eda_report.md`. The input workbook remains unchanged and is excluded
from version control by `.gitignore`.

## References

- pandas documentation: <https://pandas.pydata.org/docs/>
- scikit-learn model evaluation: <https://scikit-learn.org/stable/modules/model_evaluation.html>
- scikit-learn preprocessing and pipelines:
  <https://scikit-learn.org/stable/modules/compose.html>
- Streamlit documentation: <https://docs.streamlit.io/>
