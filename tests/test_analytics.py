"""Tests for KPI denominators and segment reconciliation."""

from __future__ import annotations

import pytest

from european_bank_churn.analytics import (
    engagement_risk_ratio,
    geography_age_matrix,
    high_value_summary,
    overall_kpis,
    profile_comparison,
    segment_summary,
)
from european_bank_churn.data import prepare_data


def test_overall_kpis(customer_frame):
    prepared, _ = prepare_data(customer_frame)
    kpis = overall_kpis(prepared)
    assert kpis["customers"] == 8
    assert kpis["churners"] == 4
    assert kpis["retained"] == 4
    assert kpis["churn_rate"] == pytest.approx(0.5)


def test_segment_counts_and_contribution_reconcile(customer_frame):
    prepared, _ = prepare_data(customer_frame)
    summary = segment_summary(prepared, "Geography")
    assert summary["Customers"].sum() == len(prepared)
    assert summary["Churners"].sum() == prepared["Exited"].sum()
    assert summary["ChurnContribution"].sum() == pytest.approx(1.0)


def test_high_value_and_engagement_metrics(customer_frame):
    prepared, _ = prepare_data(customer_frame)
    premium, kpis = high_value_summary(prepared, threshold=40_000)
    assert len(premium) == 3
    assert kpis["churners"] == 2
    assert kpis["churn_rate"] == pytest.approx(2 / 3)
    assert engagement_risk_ratio(prepared) == pytest.approx(3.0)


def test_comparison_tables_have_expected_shape(customer_frame):
    prepared, _ = prepare_data(customer_frame)
    matrix = geography_age_matrix(prepared)
    profile = profile_comparison(prepared)
    assert matrix.columns.tolist() == ["<30", "30-45", "46-59", "60+"]
    assert set(profile["ChurnLabel"].astype(str)) == {"Retained", "Churned"}
