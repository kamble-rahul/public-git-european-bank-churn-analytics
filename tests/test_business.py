"""Tests for transparent retention campaign scenario calculations."""

from __future__ import annotations

import pandas as pd
import pytest

from european_bank_churn.business import simulate_retention_campaign


def test_retention_campaign_ranks_risk_and_calculates_roi():
    scored = pd.DataFrame(
        {
            "CustomerId": ["A", "B", "C"],
            "RandomForestProbability": [0.10, 0.90, 0.50],
        }
    )
    summary, targets = simulate_retention_campaign(
        scored,
        capacity=2,
        contact_cost=10,
        retention_success_rate=0.50,
        retained_customer_value=100,
    )

    assert targets["CustomerId"].tolist() == ["B", "C"]
    assert summary["expected_churners_reached"] == pytest.approx(1.4)
    assert summary["expected_customers_retained"] == pytest.approx(0.7)
    assert summary["campaign_cost"] == 20
    assert summary["estimated_value_protected"] == pytest.approx(70)
    assert summary["estimated_net_benefit"] == pytest.approx(50)
    assert summary["estimated_roi"] == pytest.approx(2.5)


def test_retention_campaign_rejects_invalid_success_rate():
    scored = pd.DataFrame(
        {"CustomerId": ["A"], "RandomForestProbability": [0.5]}
    )
    with pytest.raises(ValueError, match="between 0 and 1"):
        simulate_retention_campaign(scored, 1, 10, 1.5, 100)
