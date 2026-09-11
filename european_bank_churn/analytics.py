"""Business KPI and segmentation calculations."""

from __future__ import annotations

import pandas as pd

from .config import AGE_LABELS


def segment_summary(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Calculate segment size, churn rate, contribution, and financial profile."""
    if dimension not in df.columns:
        raise KeyError(f"Unknown segmentation field: {dimension}")

    total_churners = int(df["Exited"].sum())
    result = (
        df.groupby(dimension, observed=False, dropna=False)
        .agg(
            Customers=("Exited", "size"),
            Churners=("Exited", "sum"),
            ChurnRate=("Exited", "mean"),
            AverageBalance=("Balance", "mean"),
            AverageSalary=("EstimatedSalary", "mean"),
        )
        .reset_index()
    )
    result["ChurnContribution"] = (
        result["Churners"] / total_churners if total_churners else 0.0
    )
    return result


def overall_kpis(df: pd.DataFrame) -> dict[str, float | int | str]:
    """Return headline KPIs for the current filtered population."""
    customer_count = int(len(df))
    churners = int(df["Exited"].sum())
    churn_rate = float(df["Exited"].mean()) if customer_count else 0.0

    geography = segment_summary(df, "Geography") if customer_count else pd.DataFrame()
    if not geography.empty and churn_rate:
        geography["GeographicRiskIndex"] = geography["ChurnRate"] / churn_rate
        riskiest = geography.sort_values("GeographicRiskIndex", ascending=False).iloc[0]
        riskiest_text = f"{riskiest['Geography']} ({riskiest['GeographicRiskIndex']:.2f}x)"
    else:
        riskiest_text = "N/A"

    return {
        "customers": customer_count,
        "churners": churners,
        "retained": customer_count - churners,
        "churn_rate": churn_rate,
        "riskiest_geography": riskiest_text,
    }


def high_value_summary(
    df: pd.DataFrame, threshold: float
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Summarize customers whose balance is at or above a selected threshold."""
    premium = df[df["Balance"] >= threshold].copy()
    churned = premium[premium["Exited"] == 1]
    values = {
        "threshold": float(threshold),
        "customers": int(len(premium)),
        "churners": int(churned.shape[0]),
        "churn_rate": float(premium["Exited"].mean()) if len(premium) else 0.0,
        "balance_at_risk": float(churned["Balance"].sum()),
    }
    return premium, values


def engagement_risk_ratio(df: pd.DataFrame) -> float:
    """Return inactive churn rate divided by active churn rate."""
    inactive_rate = df.loc[df["IsActiveMember"] == 0, "Exited"].mean()
    active_rate = df.loc[df["IsActiveMember"] == 1, "Exited"].mean()
    if pd.isna(inactive_rate) or pd.isna(active_rate) or active_rate == 0:
        return float("nan")
    return float(inactive_rate / active_rate)


def geography_age_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return churn rates for each geography and non-overlapping age band."""
    return (
        df.groupby(["Geography", "AgeGroup"], observed=False)["Exited"]
        .mean()
        .unstack()
        .reindex(columns=AGE_LABELS)
    )


def profile_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Compare the average retained and churned customer profiles."""
    columns = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "EstimatedSalary",
    ]
    return (
        df.groupby("ChurnLabel", observed=False)[columns]
        .mean()
        .round(2)
        .reset_index()
    )
