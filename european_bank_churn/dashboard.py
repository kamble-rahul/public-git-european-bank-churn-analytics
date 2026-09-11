"""Streamlit interface for the customer segmentation and churn project."""

from __future__ import annotations

from io import BytesIO

import pandas as pd
import streamlit as st

from .analytics import (
    engagement_risk_ratio,
    geography_age_matrix,
    high_value_summary,
    overall_kpis,
    profile_comparison,
    segment_summary,
)
from .config import AGE_LABELS, SEGMENT_DIMENSIONS
from .data import load_excel, prepare_data, validate_dataset
from .modeling import evaluate_probabilities, train_model_comparison
from .visualization import (
    churn_bar,
    confusion_matrix_figure,
    feature_importance_figure,
    geography_age_heatmap,
)

st.set_page_config(
    page_title="European Bank Churn Analytics",
    page_icon="🏦",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def read_uploaded_file(file_bytes: bytes) -> pd.DataFrame:
    """Read an uploaded workbook once per unique file."""
    return load_excel(BytesIO(file_bytes))


@st.cache_resource(show_spinner="Training Logistic Regression and Random Forest...")
def fit_models(data: pd.DataFrame) -> dict:
    """Cache trained models for an unchanged prepared dataset."""
    return train_model_comparison(data)


def _render_overview(
    filtered: pd.DataFrame,
    high_value_threshold: float,
) -> None:
    kpis = overall_kpis(filtered)
    _, high_value_kpis = high_value_summary(filtered, high_value_threshold)
    engagement_ratio = engagement_risk_ratio(filtered)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Customers", f"{kpis['customers']:,}")
    c2.metric("Churners", f"{kpis['churners']:,}")
    c3.metric("Overall churn rate", f"{kpis['churn_rate']:.1%}")
    c4.metric("High-value churn rate", f"{high_value_kpis['churn_rate']:.1%}")
    c5.metric(
        "Inactive / active risk",
        f"{engagement_ratio:.2f}x" if pd.notna(engagement_ratio) else "N/A",
    )

    geography_summary = segment_summary(filtered, "Geography")
    st.plotly_chart(
        churn_bar(geography_summary, "Geography", "Churn rate by geography"),
        width="stretch",
    )
    st.caption(
        "Risk describes association in this dataset. It does not prove that geography, "
        "age, or another field caused churn."
    )


def _render_segments(filtered: pd.DataFrame) -> None:
    selected_label = st.selectbox(
        "Choose a segmentation dimension",
        list(SEGMENT_DIMENSIONS),
    )
    selected_dimension = SEGMENT_DIMENSIONS[selected_label]
    summary = segment_summary(filtered, selected_dimension)
    st.plotly_chart(
        churn_bar(summary, selected_dimension, f"Churn rate by {selected_label.lower()}"),
        width="stretch",
    )
    display = summary.rename(
        columns={
            "ChurnRate": "Churn rate",
            "ChurnContribution": "Share of all churners",
            "AverageBalance": "Average balance",
            "AverageSalary": "Average salary",
        }
    )
    st.dataframe(
        display.style.format(
            {
                "Churn rate": "{:.1%}",
                "Share of all churners": "{:.1%}",
                "Average balance": "{:,.2f}",
                "Average salary": "{:,.2f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "Download this segment summary",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name="segment_summary.csv",
        mime="text/csv",
    )


def _render_demographics(filtered: pd.DataFrame) -> None:
    left, right = st.columns(2)
    with left:
        age_summary = segment_summary(filtered, "AgeGroup")
        st.plotly_chart(churn_bar(age_summary, "AgeGroup", "Age-group churn"), width="stretch")
    with right:
        tenure_summary = segment_summary(filtered, "TenureGroup")
        st.plotly_chart(
            churn_bar(tenure_summary, "TenureGroup", "Tenure-group churn"),
            width="stretch",
        )

    st.plotly_chart(
        geography_age_heatmap(geography_age_matrix(filtered)),
        width="stretch",
    )
    st.subheader("Average profile: churned vs retained")
    st.dataframe(profile_comparison(filtered), width="stretch", hide_index=True)


def _render_high_value(
    data: pd.DataFrame,
    filtered: pd.DataFrame,
) -> None:
    percentile = st.slider(
        "Premium balance percentile",
        min_value=50,
        max_value=95,
        value=75,
        step=5,
    )
    threshold = float(data["Balance"].quantile(percentile / 100))
    premium, premium_kpis = high_value_summary(filtered, threshold)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Balance threshold", f"{threshold:,.2f}")
    c2.metric("High-value customers", f"{premium_kpis['customers']:,}")
    c3.metric("High-value churn rate", f"{premium_kpis['churn_rate']:.1%}")
    c4.metric("Churned balance at risk", f"{premium_kpis['balance_at_risk']:,.2f}")
    st.warning(
        "Balance at risk is an exposure proxy, not revenue loss. Revenue estimation "
        "requires fees, margin, cost-to-serve, and customer lifetime data."
    )

    if premium.empty:
        st.info("No customers meet this threshold under the current filters.")
        return

    premium_geo = segment_summary(premium, "Geography")
    st.plotly_chart(
        churn_bar(premium_geo, "Geography", "High-value churn by geography"),
        width="stretch",
    )
    columns = [
        "CustomerId",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "EstimatedSalary",
        "NumOfProducts",
        "IsActiveMember",
        "Exited",
    ]
    st.dataframe(
        premium.sort_values(["Exited", "Balance"], ascending=[False, False])[columns].head(200),
        width="stretch",
        hide_index=True,
    )


def _render_model_comparison(data: pd.DataFrame) -> None:
    st.subheader("Supervised-learning model comparison")
    st.write(
        "The models predict the probability of Exited = 1. Logistic Regression is the "
        "explainable baseline; Random Forest captures nonlinear churn patterns."
    )
    st.caption(
        "CustomerId, Surname, and the constant Year field are deliberately excluded "
        "from training."
    )
    if st.button("Train and compare models", type="primary"):
        st.session_state["model_result"] = fit_models(data)

    if "model_result" not in st.session_state:
        return

    model_result = st.session_state["model_result"]
    comparison_rows = []
    for model_name, probability_key in [
        ("Logistic Regression", "logistic_probabilities"),
        ("Random Forest", "random_forest_probabilities"),
    ]:
        model_metrics = evaluate_probabilities(
            model_result["y_test"],
            model_result[probability_key],
            0.50,
        )
        comparison_rows.append(
            {
                "Model": model_name,
                "ROC-AUC": model_metrics["roc_auc"],
                "PR-AUC": model_metrics["pr_auc"],
                "Recall": model_metrics["recall"],
                "Precision": model_metrics["precision"],
                "F1": model_metrics["f1"],
                "Accuracy": model_metrics["accuracy"],
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    st.subheader("Model results at the default 0.50 threshold")
    st.dataframe(
        comparison.style.format(
            {
                "ROC-AUC": "{:.3f}",
                "PR-AUC": "{:.3f}",
                "Recall": "{:.1%}",
                "Precision": "{:.1%}",
                "F1": "{:.3f}",
                "Accuracy": "{:.1%}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Random Forest decision threshold")
    decision_threshold = st.slider(
        "Probability threshold for classifying a customer as high risk",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
    )
    metrics = evaluate_probabilities(
        model_result["y_test"],
        model_result["random_forest_probabilities"],
        decision_threshold,
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    c2.metric("PR-AUC", f"{metrics['pr_auc']:.3f}")
    c3.metric("Recall", f"{metrics['recall']:.1%}")
    c4.metric("Precision", f"{metrics['precision']:.1%}")
    c5.metric("F1", f"{metrics['f1']:.3f}")
    st.caption(
        "For retention outreach, recall is often important because it measures how many "
        "actual churners the model finds. Threshold choice must reflect campaign capacity and cost."
    )
    st.plotly_chart(
        confusion_matrix_figure(metrics["confusion_matrix"]),
        width="stretch",
    )
    st.plotly_chart(
        feature_importance_figure(model_result["feature_importances"]),
        width="stretch",
    )
    st.warning(
        "Feature importance shows model reliance, not causation. Before real banking use, "
        "test calibration, drift, privacy, and performance across demographic groups."
    )


def run_dashboard() -> None:
    """Render the complete Streamlit dashboard."""
    st.title("Customer Segmentation & Churn Pattern Analytics")
    st.caption(
        "Upload the European bank workbook. The original file is read-only; "
        "analysis happens in memory."
    )

    uploaded = st.file_uploader("Upload the .xlsx dataset", type=["xlsx"])
    if uploaded is None:
        st.info("Choose European_Bank (5).xlsx to open the dashboard.")
        st.markdown(
            "The dashboard checks the schema, creates segment fields, calculates KPIs, "
            "and optionally compares Logistic Regression with a Random Forest churn model."
        )
        st.stop()

    try:
        raw = read_uploaded_file(uploaded.getvalue())
    except Exception as exc:
        st.error(f"The workbook could not be read: {exc}")
        st.stop()

    findings = validate_dataset(raw)
    errors = [item for item in findings if item["level"] == "error"]
    if errors:
        for item in errors:
            st.error(item["message"])
        st.stop()

    data, default_thresholds = prepare_data(raw)
    with st.expander("Data validation", expanded=False):
        for item in findings:
            if item["level"] == "warning":
                st.warning(item["message"])
            else:
                st.info(item["message"])
        st.write(f"Rows available for analysis: {len(data):,}")

    st.sidebar.header("Filters")
    geography_options = sorted(data["Geography"].dropna().unique())
    gender_options = sorted(data["Gender"].dropna().unique())
    geographies = st.sidebar.multiselect(
        "Geography",
        geography_options,
        default=geography_options,
    )
    genders = st.sidebar.multiselect(
        "Gender",
        gender_options,
        default=gender_options,
    )
    age_groups = st.sidebar.multiselect(
        "Age group",
        list(AGE_LABELS),
        default=list(AGE_LABELS),
    )

    filtered = data[
        data["Geography"].isin(geographies)
        & data["Gender"].isin(genders)
        & data["AgeGroup"].astype(str).isin(age_groups)
    ].copy()
    if filtered.empty:
        st.warning("The current filters return no customers.")
        st.stop()

    overview, segments, demographics, high_value, model = st.tabs(
        [
            "Overview",
            "Segments",
            "Geography & demographics",
            "High-value customers",
            "ML model comparison",
        ]
    )
    with overview:
        _render_overview(filtered, default_thresholds["high_value_threshold"])
    with segments:
        _render_segments(filtered)
    with demographics:
        _render_demographics(filtered)
    with high_value:
        _render_high_value(data, filtered)
    with model:
        _render_model_comparison(data)
