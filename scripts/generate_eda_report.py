"""Generate a reproducible Markdown EDA report from a compatible bank workbook."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from european_bank_churn.analytics import (  # noqa: E402
    engagement_risk_ratio,
    geography_age_matrix,
    high_value_summary,
    profile_comparison,
    segment_summary,
)
from european_bank_churn.config import NUMERIC_FEATURES, REQUIRED_COLUMNS  # noqa: E402
from european_bank_churn.data import load_excel, prepare_data, validate_dataset  # noqa: E402
from european_bank_churn.modeling import (  # noqa: E402
    evaluate_probabilities,
    train_model_comparison,
)


def _escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers: list[str], rows: list[list[object]]) -> str:
    header = "| " + " | ".join(_escape(value) for value in headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(_escape(value) for value in row) + " |" for row in rows]
    return "\n".join([header, rule, *body])


def _percent(value: float) -> str:
    return f"{value:.2%}"


def _number(value: float) -> str:
    return f"{value:,.2f}"


def _segment_rows(summary: pd.DataFrame, dimension: str) -> list[list[object]]:
    rows = []
    for _, row in summary.iterrows():
        label = row[dimension]
        if dimension in {"NumOfProducts", "HasCrCard"} and pd.notna(label):
            label = int(label)
        rows.append(
            [
                label,
                f"{int(row['Customers']):,}",
                f"{int(row['Churners']):,}",
                _percent(float(row["ChurnRate"])),
                _percent(float(row["ChurnContribution"])),
                _number(float(row["AverageBalance"])),
            ]
        )
    return rows


def _clean_feature_name(name: str) -> str:
    return name.replace("numeric__", "").replace("categorical__", "")


def _numeric_rows(data: pd.DataFrame) -> list[list[object]]:
    rows = []
    for column in [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "EstimatedSalary",
    ]:
        series = data[column]
        rows.append(
            [
                column,
                f"{int(series.count()):,}",
                f"{int(series.isna().sum()):,}",
                _number(series.mean()),
                _number(series.median()),
                _number(series.std()),
                _number(series.min()),
                _number(series.max()),
            ]
        )
    return rows


def _model_rows(model_result: dict) -> tuple[list[list[object]], dict, dict]:
    logistic_metrics = evaluate_probabilities(
        model_result["y_test"], model_result["logistic_probabilities"]
    )
    forest_metrics = evaluate_probabilities(
        model_result["y_test"], model_result["random_forest_probabilities"]
    )
    rows = []
    for model_name, metrics in [
        ("Logistic Regression", logistic_metrics),
        ("Random Forest", forest_metrics),
    ]:
        tn, fp, fn, tp = metrics["confusion_matrix"].ravel()
        rows.append(
            [
                model_name,
                f"{metrics['roc_auc']:.3f}",
                f"{metrics['pr_auc']:.3f}",
                _percent(metrics["recall"]),
                _percent(metrics["precision"]),
                f"{metrics['f1']:.3f}",
                _percent(metrics["accuracy"]),
                f"TN={tn}, FP={fp}, FN={fn}, TP={tp}",
            ]
        )
    return rows, logistic_metrics, forest_metrics


def build_report(workbook_path: Path) -> str:
    raw = load_excel(workbook_path)
    findings = validate_dataset(raw)
    errors = [finding for finding in findings if finding["level"] == "error"]
    if errors:
        messages = "; ".join(item["message"] for item in errors)
        raise ValueError(f"Dataset validation failed: {messages}")

    data, thresholds = prepare_data(raw)
    overall_rate = float(data["Exited"].mean())
    churners = int(data["Exited"].sum())
    retained = int(len(data) - churners)

    geography = segment_summary(data, "Geography")
    geography["RiskIndex"] = geography["ChurnRate"] / overall_rate
    age = segment_summary(data, "AgeGroup")
    gender = segment_summary(data, "Gender")
    tenure = segment_summary(data, "TenureGroup")
    credit = segment_summary(data, "CreditBand")
    balance = segment_summary(data, "BalanceSegment")
    activity = segment_summary(data, "ActivityLabel")
    products = segment_summary(data, "NumOfProducts")
    card = segment_summary(data, "HasCrCard")

    salary_data = data.copy()
    salary_data["SalaryQuartile"] = pd.qcut(
        salary_data["EstimatedSalary"],
        q=4,
        labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"],
    )
    salary = segment_summary(salary_data, "SalaryQuartile")

    high_value, high_value_kpis = high_value_summary(
        data, thresholds["high_value_threshold"]
    )
    high_value_geo = segment_summary(high_value, "Geography")
    total_churned_balance = float(data.loc[data["Exited"] == 1, "Balance"].sum())
    balance_exposure_share = (
        high_value_kpis["balance_at_risk"] / total_churned_balance
        if total_churned_balance
        else 0.0
    )

    age_geo = geography_age_matrix(data)
    age_geo_rows = [
        [geography_name, *[_percent(float(value)) for value in values]]
        for geography_name, values in age_geo.iterrows()
    ]
    age_geo_stack = age_geo.stack(future_stack=True).dropna()

    profiles = profile_comparison(data)
    profile_rows = [
        [
            row["ChurnLabel"],
            _number(row["CreditScore"]),
            _number(row["Age"]),
            _number(row["Tenure"]),
            _number(row["Balance"]),
            _number(row["NumOfProducts"]),
            _number(row["EstimatedSalary"]),
        ]
        for _, row in profiles.iterrows()
    ]

    correlations = (
        data[list(NUMERIC_FEATURES) + ["Exited"]]
        .corr(numeric_only=True)["Exited"]
        .drop("Exited")
        .sort_values(key=lambda values: values.abs(), ascending=False)
    )
    correlation_rows = [[feature, f"{value:+.3f}"] for feature, value in correlations.items()]

    model_result = train_model_comparison(data)
    model_rows, logistic_metrics, forest_metrics = _model_rows(model_result)
    importance_rows = [
        [_clean_feature_name(row["Feature"]), f"{row['Importance']:.4f}"]
        for _, row in model_result["feature_importances"].head(12).iterrows()
    ]

    geography_rows = []
    for _, row in geography.iterrows():
        geography_rows.append(
            [
                row["Geography"],
                f"{int(row['Customers']):,}",
                f"{int(row['Churners']):,}",
                _percent(float(row["ChurnRate"])),
                _percent(float(row["ChurnContribution"])),
                _number(float(row["AverageBalance"])),
                f"{float(row['RiskIndex']):.2f}x",
            ]
        )

    highest_geo = geography.sort_values("ChurnRate", ascending=False).iloc[0]
    highest_age = age.sort_values("ChurnRate", ascending=False).iloc[0]
    highest_intersection = age_geo_stack.idxmax()
    highest_intersection_rate = float(age_geo_stack.max())

    segment_headers = [
        "Customers",
        "Churners",
        "Churn rate",
        "Churn contribution",
        "Average balance",
    ]
    quality_table = _table(
        ["Check", "Result"],
        [
            ["Required schema", "Present"],
            [
                "Missing cells in required fields",
                f"{int(raw[sorted(REQUIRED_COLUMNS)].isna().sum().sum()):,}",
            ],
            ["Duplicate CustomerId values", f"{int(raw['CustomerId'].duplicated().sum()):,}"],
            ["Retained customers", f"{retained:,}"],
            ["Churned customers", f"{churners:,}"],
            ["Validation messages", "; ".join(item["message"] for item in findings)],
        ],
    )
    numeric_table = _table(
        [
            "Variable",
            "Valid",
            "Missing",
            "Mean",
            "Median",
            "Std. dev.",
            "Minimum",
            "Maximum",
        ],
        _numeric_rows(data),
    )
    geography_table = _table(
        ["Geography", *segment_headers, "Risk index"], geography_rows
    )
    segment_tables = {
        "age": _table(["Age group", *segment_headers], _segment_rows(age, "AgeGroup")),
        "gender": _table(["Gender", *segment_headers], _segment_rows(gender, "Gender")),
        "tenure": _table(
            ["Tenure group", *segment_headers], _segment_rows(tenure, "TenureGroup")
        ),
        "credit": _table(
            ["Credit band", *segment_headers], _segment_rows(credit, "CreditBand")
        ),
        "balance": _table(
            ["Balance segment", *segment_headers],
            _segment_rows(balance, "BalanceSegment"),
        ),
        "activity": _table(
            ["Activity", *segment_headers], _segment_rows(activity, "ActivityLabel")
        ),
        "products": _table(
            ["Products", *segment_headers], _segment_rows(products, "NumOfProducts")
        ),
        "card": _table(
            ["Has credit card", *segment_headers], _segment_rows(card, "HasCrCard")
        ),
        "salary": _table(
            ["Salary quartile", *segment_headers],
            _segment_rows(salary, "SalaryQuartile"),
        ),
        "high_value_geo": _table(
            ["Geography", *segment_headers], _segment_rows(high_value_geo, "Geography")
        ),
    }
    age_geo_table = _table(
        ["Geography", *[str(column) for column in age_geo.columns]], age_geo_rows
    )
    profile_table = _table(
        [
            "Status",
            "Credit score",
            "Age",
            "Tenure",
            "Balance",
            "Products",
            "Estimated salary",
        ],
        profile_rows,
    )
    high_value_table = _table(
        ["Metric", "Result"],
        [
            ["Balance threshold", _number(float(high_value_kpis["threshold"]))],
            ["High-value customers", f"{int(high_value_kpis['customers']):,}"],
            ["High-value churners", f"{int(high_value_kpis['churners']):,}"],
            ["High-value churn rate", _percent(float(high_value_kpis["churn_rate"]))],
            [
                "Churned high-value balance",
                _number(float(high_value_kpis["balance_at_risk"])),
            ],
            ["Share of all churned balance", _percent(balance_exposure_share)],
        ],
    )
    correlation_table = _table(["Feature", "Correlation with Exited"], correlation_rows)
    model_table = _table(
        [
            "Model",
            "ROC-AUC",
            "PR-AUC",
            "Recall",
            "Precision",
            "F1",
            "Accuracy",
            "Confusion matrix",
        ],
        model_rows,
    )
    importance_table = _table(["Feature", "Importance"], importance_rows)
    majority_baseline = _percent(retained / len(data))

    return f"""# Exploratory Data Analysis: European Bank Customer Churn

## Report scope

This report analyzes `{workbook_path.name}` using the reusable functions in the project package.
It addresses the required customer-segmentation, churn-distribution, demographic, engagement,
financial-profile, high-value, and supervised-learning questions. Results describe associations in
this educational dataset; they do not prove why a customer churned.

- **Generated:** {date.today().isoformat()}
- **Rows:** {len(data):,}
- **Columns in source:** {raw.shape[1]:,}
- **Target:** `Exited` (`1` = churned, `0` = retained)

## 1. Executive findings

- Overall churn is **{_percent(overall_rate)}** ({churners:,} of {len(data):,} customers).
- **{highest_geo['Geography']}** has the highest country churn rate at
  **{_percent(float(highest_geo['ChurnRate']))}** and a geographic risk index of
  **{float(highest_geo['RiskIndex']):.2f}x** the portfolio average.
- The **{highest_age['AgeGroup']}** age group has the highest age-band churn at
  **{_percent(float(highest_age['ChurnRate']))}**.
- The highest geography-age intersection is **{highest_intersection[0]} ×
  {highest_intersection[1]}** at **{_percent(highest_intersection_rate)}**.
- Inactive customers have **{engagement_risk_ratio(data):.2f}x** the churn rate of active
  customers.
- The high-value threshold is **{_number(thresholds['high_value_threshold'])}** (75th balance
  percentile). High-value churn is **{_percent(float(high_value_kpis['churn_rate']))}**.
- Churned high-value customers hold **{_number(float(high_value_kpis['balance_at_risk']))}** in
  balances, or **{_percent(balance_exposure_share)}** of all churned-customer balance. This is an
  exposure measure, not revenue loss.
- Random Forest produces the stronger ranking result with ROC-AUC
  **{forest_metrics['roc_auc']:.3f}** versus **{logistic_metrics['roc_auc']:.3f}** for Logistic
  Regression on the fixed holdout set.

## 2. Data quality and validation

{quality_table}

`Surname` is retained only for source traceability and is excluded from analytics and modeling.
`CustomerId` is used only in drill-down views. A constant `Year` field, when present, is excluded
from modeling because it adds no predictive variation.

## 3. Numeric profile

{numeric_table}

## 4. Churn distribution by required segments

Each table reports segment size, churn rate, and churn contribution. Churn contribution answers:
“What share of all churned customers came from this segment?” It must be read together with churn
rate because a small segment can have high risk but low portfolio impact.

### Geography

{geography_table}

### Age group

{segment_tables['age']}

### Gender

{segment_tables['gender']}

### Tenure group

{segment_tables['tenure']}

### Credit-score band

{segment_tables['credit']}

### Balance segment

The median among positive balances is **{_number(thresholds['positive_balance_median'])}**. Zero
balances remain a separate business segment.

{segment_tables['balance']}

### Engagement status

{segment_tables['activity']}

### Products held

{segment_tables['products']}

### Credit-card ownership

{segment_tables['card']}

## 5. Geography and age interaction

{age_geo_table}

The interaction table is more useful than reading geography and age independently. It helps locate
concentrated risk while preserving the sample-size context available in the dashboard.

## 6. Churned versus retained financial profile

{profile_table}

### Estimated-salary quartiles

{segment_tables['salary']}

Salary quartiles show whether churn changes consistently across salary levels. The Streamlit
high-value explorer adds an interactive salary-versus-balance scatter plot so the two measures can
be studied together instead of treating salary as customer value.

## 7. High-value customer churn

High value is defined transparently as balance at or above the selected percentile. The dashboard
allows the percentile to change from 50 to 95.

{high_value_table}

### High-value churn by geography

{segment_tables['high_value_geo']}

## 8. Numeric association with churn

Pearson correlation is a screening statistic for linear association. Small values do not rule out
nonlinear effects or interactions, and none of these coefficients establish causation.

{correlation_table}

## 9. Machine-learning comparison

Both models use the same 80/20 stratified split (`random_state=42`). Numeric fields are imputed;
Logistic Regression also standardizes them. Geography and gender are imputed and one-hot encoded.
Both classifiers use balanced class weights. Identifiers, surname, derived segments, and constant
year are excluded from training.

{model_table}

Accuracy alone is not sufficient because the majority-class baseline is **{majority_baseline}**.
Random Forest is the recommended demonstration model because it captures nonlinear patterns, while
Logistic Regression remains the explainable baseline. Neither model is ready for automated banking
decisions.

### Random Forest feature importance

{importance_table}

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
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path, help="Path to the source .xlsx workbook")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "eda_report.md",
        help="Markdown output path (default: reports/eda_report.md)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.workbook.is_file():
        raise FileNotFoundError(f"Workbook not found: {args.workbook}")
    report = build_report(args.workbook)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"EDA report written to {args.output}")


if __name__ == "__main__":
    main()
