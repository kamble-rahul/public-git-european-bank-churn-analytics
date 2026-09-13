"""Reusable analytics package for the European banking churn project."""

from .analytics import (
    engagement_risk_ratio,
    geography_age_matrix,
    high_value_summary,
    overall_kpis,
    profile_comparison,
    segment_summary,
)
from .business import simulate_retention_campaign
from .data import (
    load_dashboard_data,
    load_excel,
    prepare_data,
    standardize_for_dashboard,
    validate_dataset,
)
from .modeling import (
    calibration_table,
    cross_validate_models,
    evaluate_probabilities,
    explain_logistic_customer,
    subgroup_performance,
    train_logistic_baseline,
    train_model_comparison,
)

__all__ = [
    "calibration_table",
    "cross_validate_models",
    "engagement_risk_ratio",
    "evaluate_probabilities",
    "explain_logistic_customer",
    "geography_age_matrix",
    "high_value_summary",
    "load_dashboard_data",
    "load_excel",
    "overall_kpis",
    "prepare_data",
    "profile_comparison",
    "segment_summary",
    "simulate_retention_campaign",
    "standardize_for_dashboard",
    "subgroup_performance",
    "train_logistic_baseline",
    "train_model_comparison",
    "validate_dataset",
]
