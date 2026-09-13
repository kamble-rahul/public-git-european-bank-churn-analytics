"""Tests for model pipelines and threshold-dependent evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from european_bank_churn.data import prepare_data
from european_bank_churn.modeling import (
    calibration_table,
    cross_validate_models,
    evaluate_probabilities,
    explain_logistic_customer,
    subgroup_performance,
    train_model_comparison,
)


def test_evaluate_probabilities_uses_selected_threshold():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.10, 0.40, 0.60, 0.90])
    metrics = evaluate_probabilities(y_true, probabilities, threshold=0.50)
    assert metrics["accuracy"] == 1.0
    assert metrics["brier_score"] == pytest.approx(0.085)
    assert metrics["confusion_matrix"].tolist() == [[2, 0], [0, 2]]


def test_invalid_threshold_is_rejected():
    with pytest.raises(ValueError, match="between 0 and 1"):
        evaluate_probabilities(np.array([0, 1]), np.array([0.2, 0.8]), threshold=1.1)


def test_model_comparison_is_reproducible_and_complete(modeling_frame):
    prepared, _ = prepare_data(modeling_frame)
    result = train_model_comparison(prepared, random_state=42)
    assert result["test_rows"] == 48
    assert len(result["y_test"]) == len(result["random_forest_probabilities"])
    assert len(result["y_test"]) == len(result["logistic_probabilities"])
    assert result["feature_importances"]["Importance"].sum() == pytest.approx(1.0)
    assert set(result["permutation_importances"]["Feature"]) == {
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Geography",
        "Gender",
    }
    assert result["metadata"]["model_version"] == "1.1.0"
    assert len(result["metadata"]["dataset_fingerprint"]) == 12

    customer = result["test_records"].head(1)
    explanation = explain_logistic_customer(result["logistic_pipeline"], customer)
    assert not explanation.empty
    assert {"Feature", "Contribution", "Direction"} <= set(explanation.columns)


def test_calibration_and_subgroup_tables_are_complete():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9, 0.2, 0.8])
    calibration = calibration_table(y_true, probabilities, n_bins=3)
    assert list(calibration.columns) == [
        "MeanPredictedProbability",
        "ObservedChurnRate",
    ]
    assert len(calibration) == 3

    groups = np.array(["A", "A", "A", "B", "B", "B"])
    audit = subgroup_performance(y_true, probabilities, groups, threshold=0.5)
    assert audit["Customers"].sum() == 6
    assert set(audit["Group"]) == {"A", "B"}
    assert audit["Recall"].between(0, 1).all()


def test_cross_validation_reports_mean_and_variability(modeling_frame):
    prepared, _ = prepare_data(modeling_frame)
    validation = cross_validate_models(prepared, folds=3, random_state=42)
    assert set(validation["Model"]) == {"Logistic Regression", "Random Forest"}
    assert set(validation["Folds"]) == {3}
    assert validation["PR-AUCMean"].between(0, 1).all()
    assert validation["ROC-AUCStd"].ge(0).all()
