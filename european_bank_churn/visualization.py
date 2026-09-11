"""Plotly chart builders kept separate from the Streamlit interface."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def churn_bar(summary: pd.DataFrame, dimension: str, title: str) -> go.Figure:
    """Build a segment churn-rate bar chart with customer counts in tooltips."""
    chart_data = summary.copy()
    chart_data[dimension] = chart_data[dimension].astype(str)
    figure = px.bar(
        chart_data,
        x=dimension,
        y="ChurnRate",
        color="ChurnRate",
        color_continuous_scale="Reds",
        text=chart_data["ChurnRate"].map(lambda value: f"{value:.1%}"),
        hover_data={"Customers": ":,", "Churners": ":,", "ChurnContribution": ":.1%"},
        title=title,
    )
    figure.update_traces(textposition="outside")
    figure.update_yaxes(tickformat=".0%", title="Churn rate")
    upper_bound = max(0.05, float(chart_data["ChurnRate"].max()) * 1.18)
    figure.update_layout(coloraxis_showscale=False, yaxis_range=[0, upper_bound])
    return figure


def geography_age_heatmap(matrix: pd.DataFrame) -> go.Figure:
    """Build a geography-by-age churn-rate heatmap."""
    labels = matrix.map(lambda value: "" if pd.isna(value) else f"{value:.1%}")
    figure = go.Figure(
        data=go.Heatmap(
            z=matrix.to_numpy(),
            x=matrix.columns.astype(str),
            y=matrix.index.astype(str),
            text=labels.to_numpy(),
            texttemplate="%{text}",
            colorscale="Reds",
            colorbar={"title": "Churn rate", "tickformat": ".0%"},
            hovertemplate="Geography=%{y}<br>Age=%{x}<br>Churn=%{z:.1%}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Geography × age interaction",
        xaxis_title="Age group",
        yaxis_title="Geography",
    )
    return figure


def confusion_matrix_figure(matrix) -> go.Figure:
    """Build a labeled binary-classification confusion matrix."""
    return px.imshow(
        matrix,
        text_auto=True,
        labels={"x": "Predicted", "y": "Actual", "color": "Customers"},
        x=["Retained", "Churned"],
        y=["Retained", "Churned"],
        title="Confusion matrix",
        color_continuous_scale="Blues",
    )


def feature_importance_figure(importances: pd.DataFrame) -> go.Figure:
    """Build a horizontal Random Forest feature-importance chart."""
    top_features = importances.head(15).sort_values("Importance")
    return px.bar(
        top_features,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Random Forest feature importance",
    )
