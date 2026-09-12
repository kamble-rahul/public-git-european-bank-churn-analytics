"""Tests that chart builders return complete Plotly figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from european_bank_churn.data import prepare_data
from european_bank_churn.visualization import (
    churn_bar,
    confusion_matrix_figure,
    feature_importance_figure,
    geography_age_heatmap,
    salary_balance_scatter,
)


def test_churn_bar_contains_segment_labels():
    summary = pd.DataFrame(
        {
            "Geography": ["France", "Germany"],
            "Customers": [100, 80],
            "Churners": [15, 24],
            "ChurnRate": [0.15, 0.30],
            "ChurnContribution": [0.38, 0.62],
        }
    )
    figure = churn_bar(summary, "Geography", "Test churn")
    assert isinstance(figure, go.Figure)
    assert figure.layout.title.text == "Test churn"
    assert list(figure.data[0].x) == ["France", "Germany"]


def test_geography_age_heatmap_contains_expected_axes():
    matrix = pd.DataFrame(
        [[0.1, 0.2], [0.3, 0.4]],
        index=["France", "Germany"],
        columns=["<30", "30-45"],
    )
    figure = geography_age_heatmap(matrix)
    assert isinstance(figure, go.Figure)
    assert figure.layout.xaxis.title.text == "Age group"
    assert list(figure.data[0].y) == ["France", "Germany"]


def test_model_diagnostic_figures_are_labeled():
    matrix_figure = confusion_matrix_figure(np.array([[8, 2], [3, 7]]))
    importance_figure = feature_importance_figure(
        pd.DataFrame(
            {
                "Feature": ["Age", "Balance", "Activity"],
                "Importance": [0.5, 0.3, 0.2],
            }
        )
    )
    assert matrix_figure.layout.title.text == "Confusion matrix"
    assert importance_figure.layout.title.text == "Random Forest feature importance"


def test_salary_balance_scatter_contains_both_churn_classes(customer_frame):
    prepared_data, _ = prepare_data(customer_frame)
    figure = salary_balance_scatter(prepared_data)
    assert figure.layout.title.text == "Salary and balance profile by churn status"
    assert {trace.name.split(",")[0] for trace in figure.data} == {
        "Retained",
        "Churned",
    }
