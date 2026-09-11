"""Tests for model pipelines and threshold-dependent evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from european_bank_churn.data import prepare_data
from european_bank_churn.modeling import evaluate_probabilities, train_model_comparison


def test_evaluate_probabilities_uses_selected_threshold():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.10, 0.40, 0.60, 0.90])
    metrics = evaluate_probabilities(y_true, probabilities, threshold=0.50)
    assert metrics["accuracy"] == 1.0
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
