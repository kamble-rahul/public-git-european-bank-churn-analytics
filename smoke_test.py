"""Small command-line check for the data and reusable functions."""

from __future__ import annotations

import argparse

from european_bank_churn import (
    high_value_summary,
    load_excel,
    overall_kpis,
    prepare_data,
    segment_summary,
    validate_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a banking workbook and print headline churn KPIs."
    )
    parser.add_argument("workbook", help="Path to European_Bank (5).xlsx")
    args = parser.parse_args()

    raw = load_excel(args.workbook)
    findings = validate_dataset(raw)
    errors = [item for item in findings if item["level"] == "error"]
    if errors:
        raise SystemExit(errors)

    data, thresholds = prepare_data(raw)
    kpis = overall_kpis(data)
    geography = segment_summary(data, "Geography")
    _, high_value = high_value_summary(data, thresholds["high_value_threshold"])

    assert len(data) > 0
    assert 0 <= kpis["churn_rate"] <= 1
    assert geography["Customers"].sum() == len(data)

    print(f"Rows: {len(data):,}")
    print(f"Churn rate: {kpis['churn_rate']:.2%}")
    print(f"High-value threshold: {thresholds['high_value_threshold']:,.2f}")
    print(f"High-value churn rate: {high_value['churn_rate']:.2%}")
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
