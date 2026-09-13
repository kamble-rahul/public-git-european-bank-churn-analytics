"""Transparent business-scenario calculations for retention planning."""

from __future__ import annotations

import math

import pandas as pd


def simulate_retention_campaign(
    scored_customers: pd.DataFrame,
    capacity: int,
    contact_cost: float,
    retention_success_rate: float,
    retained_customer_value: float,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Rank customers and estimate campaign economics from user-supplied assumptions."""
    required = {"CustomerId", "RandomForestProbability"}
    missing = required - set(scored_customers.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if not 0 <= capacity <= len(scored_customers):
        raise ValueError("capacity must be between 0 and the number of scored customers")
    if contact_cost < 0 or retained_customer_value < 0:
        raise ValueError("cost and retained customer value must be non-negative")
    if not 0 <= retention_success_rate <= 1:
        raise ValueError("retention_success_rate must be between 0 and 1")

    targets = scored_customers.sort_values(
        "RandomForestProbability", ascending=False
    ).head(capacity).copy()
    expected_churners_reached = float(targets["RandomForestProbability"].sum())
    expected_customers_retained = expected_churners_reached * retention_success_rate
    campaign_cost = float(capacity * contact_cost)
    estimated_value_protected = expected_customers_retained * retained_customer_value
    estimated_net_benefit = estimated_value_protected - campaign_cost
    estimated_roi = (
        estimated_net_benefit / campaign_cost if campaign_cost else math.nan
    )

    summary: dict[str, float | int] = {
        "customers_targeted": int(capacity),
        "expected_churners_reached": expected_churners_reached,
        "expected_customers_retained": expected_customers_retained,
        "campaign_cost": campaign_cost,
        "estimated_value_protected": estimated_value_protected,
        "estimated_net_benefit": estimated_net_benefit,
        "estimated_roi": estimated_roi,
    }
    return summary, targets
