"""Reusable analytics package for the European banking churn project."""

from .analytics import (
    engagement_risk_ratio,
    geography_age_matrix,
    high_value_summary,
    overall_kpis,
    profile_comparison,
    segment_summary,
)
from .data import (
    load_dashboard_data,
    load_excel,
    prepare_data,
    standardize_for_dashboard,
    validate_dataset,
)
from .modeling import (
    evaluate_probabilities,
    train_logistic_baseline,
    train_model_comparison,
)

__all__ = [
    "engagement_risk_ratio",
    "evaluate_probabilities",
    "geography_age_matrix",
    "high_value_summary",
    "load_dashboard_data",
    "load_excel",
    "overall_kpis",
    "prepare_data",
    "profile_comparison",
    "segment_summary",
    "standardize_for_dashboard",
    "train_logistic_baseline",
    "train_model_comparison",
    "validate_dataset",
]
