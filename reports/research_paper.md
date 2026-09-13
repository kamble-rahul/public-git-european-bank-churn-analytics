# Customer Segmentation and Churn Pattern Analytics in European Banking

## Abstract

Customer churn reduces customer lifetime value, increases replacement costs, and creates revenue
instability. This study analyzes a 10,000-row educational European banking dataset to measure churn,
identify high-risk segments, compare geographic and demographic patterns, and examine high-balance
exposure. The methodology combines rule-based segmentation, descriptive statistics, interactive
visualization, and two supervised classifiers. Overall churn is 20.37%, with materially higher churn
in Germany, inactive customers, and the 46–59 age group. Logistic Regression provides an explainable
baseline and Random Forest captures nonlinear interactions. Results support targeted investigation
and retention experimentation, but the data cannot establish causation or recognized revenue loss.

## 1. Background and problem statement

Retail banks often measure total churn but lack sufficiently granular evidence to decide which
customers to study or contact. A single average can hide differences by geography, age, engagement,
tenure, product holdings, credit score, and balance. Generic retention campaigns may therefore spend
resources on low-risk customers while missing smaller, high-risk or high-value groups.

This project addresses three related problems: identifying high-risk customer segments,
understanding demographic and regional differences, and quantifying the financial profile associated
with exits. Its primary contribution is transparent descriptive analytics. Machine learning is a
secondary risk-ranking extension.

## 2. Research questions

1. What proportion of customers exited?
2. Which business-defined segments have the highest churn rate and largest churn contribution?
3. How do churn patterns vary across France, Germany, Spain, age, gender, and engagement?
4. How much account balance is associated with churn among high-value customers?
5. Does a Random Forest improve churn ranking over an explainable Logistic Regression baseline?

## 3. Dataset

The workbook contains 10,000 customer rows and the fields documented in
`docs/data_dictionary.md`. The target `Exited` is binary. The analyzed file contains 2,037 exits and
7,963 retained customers. Required fields have no missing cells, customer identifiers are unique,
and the binary fields are consistent in the supplied snapshot. An optional `Year` field contains only
2025 and is excluded from modeling because a constant cannot improve prediction.

The workbook does not contain authoritative source documentation. References to European banking
describe its geography fields, not verified institutional ownership or endorsement.

## 4. Methodology

### 4.1 Validation and preparation

The application verifies the required schema, row availability, numeric convertibility, binary
values, identifiers, product counts, balances, age ranges, and expected categories. It reports
problems rather than silently deleting observations. `Surname` is excluded from analysis and model
training; `CustomerId` remains available only for controlled drill-down.

### 4.2 Segmentation

Age is divided into `<30`, `30–45`, `46–59`, and `60+`. This removes the source brief's overlap at
age 60. Credit score uses `<580`, `580–669`, and `670+`. Tenure uses `0–2`, `3–6`, and `7–10` years.
Balance is split into zero, low, and high, with the median positive balance separating low and high.
High value defaults to the 75th portfolio balance percentile.

### 4.3 KPIs

Overall and segment churn rates use customer counts in the numerator and denominator. Churn
contribution divides each segment's exits by all exits in the filtered population. The geographic
risk index divides regional churn by overall churn. Engagement risk divides inactive churn by active
churn. Balance at risk sums balances among churned high-value customers; it is not a revenue estimate.

### 4.4 Predictive modeling

The modeling sample uses an 80/20 stratified holdout split controlled by `random_state=42`.
Identifiers and the constant year are excluded. Numeric values are median-imputed; Logistic
Regression also standardizes them. Categories are most-frequent-imputed and one-hot encoded. All
steps are contained in scikit-learn pipelines to reduce train/test leakage. Both classifiers use
balanced class weights. Evaluation includes ROC-AUC, PR-AUC, recall, precision, F1, accuracy, and a
confusion matrix. The dashboard permits threshold analysis because campaign decisions should reflect
the relative cost of missed churners and unnecessary outreach.

## 5. Exploratory findings

### 5.1 Overall and geographic churn

Overall churn is 20.37%. Germany records 32.44% churn, almost double France's 16.15% and Spain's
16.67%. Germany's geographic risk index is approximately 1.59, meaning its churn rate is 59% above
the portfolio average. This difference warrants investigation into product mix, service channels,
pricing, and data definitions; geography alone is not a causal explanation.

### 5.2 Engagement and age

Inactive customers record 26.85% churn compared with 14.27% for active customers, producing an
inactive/active risk ratio of approximately 1.88. The 46–59 age band records 51.10% churn, the highest
age-band result. The Germany × 46–59 intersection records 67.28%, showing why interaction views can
be more informative than independent averages.

### 5.3 Gender and products

Female customers record 25.07% churn and male customers 16.46%. This difference must be treated as
an association and subjected to fairness analysis before any operational targeting. Customers with
three products record 82.71% churn across 266 records, while four-product customers record 100%
across only 60 records. These extreme, small groups require sample-size context and a product-count
definition review.

### 5.4 High-value customers

The 75th percentile balance threshold is 127,644.24. It identifies 2,500 customers, of whom 592
exited, for a 23.68% churn rate. Their balances total approximately 88.65 million and account for
47.77% of all balance held by churned customers. These balances indicate exposure and potential
retention priority; revenue impact cannot be calculated from balance alone.

## 6. Model results and interpretation

On the fixed holdout split, the initial Random Forest run produced ROC-AUC 0.861, PR-AUC 0.689,
recall 67.8%, precision 57.6%, and F1 0.623 at a 0.50 threshold. These metrics indicate useful
ranking, but not production readiness. Accuracy alone would be misleading because predicting every
customer as retained already yields 79.63% accuracy.

Logistic Regression remains valuable even when Random Forest ranks better: its standardized
coefficients provide a simpler baseline, expose unexpected directional relationships, and help
identify leakage or data-quality errors. Random Forest feature importance shows model reliance, not
causation. A production evaluation should add temporal or independent validation, calibrated
probability correction, confidence intervals, stability monitoring, and comparison with a no-model
campaign.

The enhanced application adds five-fold stratified validation. Random Forest records mean ROC-AUC
of 0.861 ± 0.008 and mean PR-AUC of 0.680 ± 0.019, indicating relatively stable ranking across
folds. On the fixed holdout, its Brier score is 0.129 versus 0.194 for Logistic Regression. A
reliability curve is still required alongside Brier score because probability error combines
calibration and discrimination.

Holdout permutation importance, signed Logistic Regression coefficients, customer-level
contributions, and subgroup error tables improve transparency. The retention ROI simulator connects
campaign capacity to cost and expected value, but every financial input is an explicit scenario
assumption rather than measured revenue or causal impact.

## 7. Recommendations

1. Conduct qualitative and operational research focused on Germany and ages 46–59.
2. Test a respectful re-engagement intervention for inactive customers using randomized treatment
   and control groups.
3. Audit product-count definitions and journeys for customers with three or four products.
4. Route high-value risk flags to trained staff for review; do not automate adverse actions.
5. Choose the prediction threshold from contact capacity, false-negative cost, false-positive cost,
   expected incremental margin, and customer-contact constraints.
6. Monitor retention lift, complaints, fairness, calibration, and drift after deployment.

## 8. Limitations

- One snapshot cannot measure changes before churn or support temporal validation.
- There is no churn reason, transaction history, interaction sequence, treatment history, or channel
  data.
- Balance is not revenue, profit, or customer lifetime value.
- Observational differences do not establish causal effects.
- Gender and geography may raise legal, ethical, or fairness concerns when used as model features.
- Dataset provenance and currency are not documented in the supplied workbook.

## 9. Conclusion

Segmentation reveals actionable heterogeneity hidden by the 20.37% portfolio average. Germany,
inactive customers, the 46–59 group, and selected high-balance customers deserve prioritized
investigation. The dashboard makes definitions and denominators visible, while the model comparison
introduces risk ranking without replacing business analysis. The appropriate next step is a governed
retention experiment with privacy controls, human oversight, and outcome measurement—not automatic
customer treatment based solely on a score.

## References and learning resources

- pandas documentation: <https://pandas.pydata.org/docs/>
- scikit-learn model evaluation: <https://scikit-learn.org/stable/modules/model_evaluation.html>
- scikit-learn mixed-type preprocessing:
  <https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html>
- Streamlit documentation: <https://docs.streamlit.io/>
- Cookiecutter Data Science: <https://github.com/drivendataorg/cookiecutter-data-science>
