"""Tests for schema validation and rule-based feature engineering."""

from __future__ import annotations

from european_bank_churn.data import (
    prepare_data,
    standardize_for_dashboard,
    validate_dataset,
)


def test_valid_frame_has_no_validation_errors(customer_frame):
    findings = validate_dataset(customer_frame)
    assert not [item for item in findings if item["level"] == "error"]


def test_missing_required_column_is_an_error(customer_frame):
    findings = validate_dataset(customer_frame.drop(columns="Exited"))
    assert findings == [{"level": "error", "message": "Missing required columns: Exited"}]


def test_invalid_binary_value_is_an_error(customer_frame):
    invalid = customer_frame.copy()
    invalid.loc[0, "IsActiveMember"] = 2
    findings = validate_dataset(invalid)
    assert any("IsActiveMember must contain only 0 and 1" in item["message"] for item in findings)


def test_age_credit_and_tenure_boundaries_do_not_overlap(customer_frame):
    prepared, _ = prepare_data(customer_frame)
    assert prepared["AgeGroup"].astype(str).tolist() == [
        "<30",
        "30-45",
        "30-45",
        "46-59",
        "46-59",
        "60+",
        "60+",
        "30-45",
    ]
    assert prepared["CreditBand"].astype(str).tolist()[:6] == [
        "Low (<580)",
        "Low (<580)",
        "Medium (580-669)",
        "Medium (580-669)",
        "High (670+)",
        "High (670+)",
    ]
    assert prepared["TenureGroup"].astype(str).tolist()[:6] == [
        "New (0-2)",
        "New (0-2)",
        "Mid-term (3-6)",
        "Mid-term (3-6)",
        "Long-term (7-10)",
        "Long-term (7-10)",
    ]


def test_all_zero_balances_are_handled(customer_frame):
    zero_balance = customer_frame.assign(Balance=0)
    prepared, thresholds = prepare_data(zero_balance)
    assert thresholds["positive_balance_median"] == 0.0
    assert set(prepared["BalanceSegment"]) == {"Zero balance"}


def test_dashboard_standardization_removes_direct_identifiers(customer_frame):
    standardized = standardize_for_dashboard(customer_frame)

    assert standardized["CustomerId"].tolist() == [
        f"DEMO-{row_number:05d}" for row_number in range(1, len(customer_frame) + 1)
    ]
    assert standardized["CustomerId"].is_unique
    assert set(standardized["Surname"]) == {"Anonymous"}
    assert standardized["Exited"].equals(customer_frame["Exited"])
    assert standardized["Balance"].equals(customer_frame["Balance"])
    assert not set(customer_frame["Surname"]) & set(standardized["Surname"])
