"""Streamlit dashboard for customer segmentation and churn pattern analytics."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from churn_analysis import (
    evaluate_probabilities,
    high_value_summary,
    load_excel,
    overall_kpis,
    prepare_data,
    segment_summary,
    train_model_comparison,
    validate_dataset,
)


st.set_page_config(
    page_title="European Bank Churn Analytics",
    page_icon="🏦",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def read_uploaded_file(file_bytes: bytes) -> pd.DataFrame:
    from io import BytesIO

    return load_excel(BytesIO(file_bytes))


@st.cache_resource(show_spinner="Training Logistic Regression and Random Forest...")
def fit_models(data: pd.DataFrame) -> dict:
    return train_model_comparison(data)


def churn_bar(summary: pd.DataFrame, dimension: str, title: str):
    chart_data = summary.copy()
    chart_data[dimension] = chart_data[dimension].astype(str)
    fig = px.bar(
        chart_data,
        x=dimension,
        y="ChurnRate",
        color="ChurnRate",
        color_continuous_scale="Reds",
        text=chart_data["ChurnRate"].map(lambda value: f"{value:.1%}"),
        hover_data={"Customers": ":,", "Churners": ":,", "ChurnContribution": ":.1%"},
        title=title,
    )
    fig.update_traces(textposition="outside")
    fig.update_yaxes(tickformat=".0%", title="Churn rate")
    fig.update_layout(coloraxis_showscale=False, yaxis_range=[0, max(0.05, chart_data["ChurnRate"].max() * 1.18)])
    return fig


st.title("Customer Segmentation & Churn Pattern Analytics")
st.caption("Upload the European bank workbook. The original file is read-only; analysis happens in memory.")

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
geographies = st.sidebar.multiselect(
    "Geography", sorted(data["Geography"].dropna().unique()), default=sorted(data["Geography"].dropna().unique())
)
genders = st.sidebar.multiselect(
    "Gender", sorted(data["Gender"].dropna().unique()), default=sorted(data["Gender"].dropna().unique())
)
age_groups = st.sidebar.multiselect(
    "Age group",
    ["<30", "30-45", "46-59", "60+"],
    default=["<30", "30-45", "46-59", "60+"],
)

filtered = data[
    data["Geography"].isin(geographies)
    & data["Gender"].isin(genders)
    & data["AgeGroup"].astype(str).isin(age_groups)
].copy()
if filtered.empty:
    st.warning("The current filters return no customers.")
    st.stop()

overview_tab, segments_tab, demographic_tab, value_tab, ml_tab = st.tabs(
    ["Overview", "Segments", "Geography & demographics", "High-value customers", "ML model comparison"]
)

with overview_tab:
    kpis = overall_kpis(filtered)
    high_default, high_default_kpis = high_value_summary(
        filtered, default_thresholds["high_value_threshold"]
    )
    inactive_rate = filtered.loc[filtered["IsActiveMember"] == 0, "Exited"].mean()
    active_rate = filtered.loc[filtered["IsActiveMember"] == 1, "Exited"].mean()
    engagement_ratio = inactive_rate / active_rate if active_rate and pd.notna(active_rate) else float("nan")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Customers", f"{kpis['customers']:,}")
    c2.metric("Churners", f"{kpis['churners']:,}")
    c3.metric("Overall churn rate", f"{kpis['churn_rate']:.1%}")
    c4.metric("High-value churn rate", f"{high_default_kpis['churn_rate']:.1%}")
    c5.metric("Inactive / active risk", f"{engagement_ratio:.2f}x" if pd.notna(engagement_ratio) else "N/A")

    geography_summary = segment_summary(filtered, "Geography")
    st.plotly_chart(
        churn_bar(geography_summary, "Geography", "Churn rate by geography"),
        width="stretch",
    )
    st.caption(
        "Risk describes association in this dataset. It does not prove that geography, age, or another field caused churn."
    )

with segments_tab:
    dimension_labels = {
        "Age group": "AgeGroup",
        "Credit-score band": "CreditBand",
        "Tenure group": "TenureGroup",
        "Balance segment": "BalanceSegment",
        "Products held": "NumOfProducts",
        "Activity": "ActivityLabel",
        "Gender": "Gender",
    }
    selected_label = st.selectbox("Choose a segmentation dimension", list(dimension_labels))
    selected_dimension = dimension_labels[selected_label]
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

with demographic_tab:
    left, right = st.columns(2)
    with left:
        age_summary = segment_summary(filtered, "AgeGroup")
        st.plotly_chart(churn_bar(age_summary, "AgeGroup", "Age-group churn"), width="stretch")
    with right:
        tenure_summary = segment_summary(filtered, "TenureGroup")
        st.plotly_chart(churn_bar(tenure_summary, "TenureGroup", "Tenure-group churn"), width="stretch")

    matrix = (
        filtered.groupby(["Geography", "AgeGroup"], observed=False)["Exited"]
        .mean()
        .unstack()
        .reindex(columns=["<30", "30-45", "46-59", "60+"])
    )
    labels = matrix.map(lambda value: "" if pd.isna(value) else f"{value:.1%}")
    heatmap = go.Figure(
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
    heatmap.update_layout(title="Geography × age interaction", xaxis_title="Age group", yaxis_title="Geography")
    st.plotly_chart(heatmap, width="stretch")

    profile = (
        filtered.groupby("ChurnLabel", observed=False)[
            ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
        ]
        .mean()
        .round(2)
        .reset_index()
    )
    st.subheader("Average profile: churned vs retained")
    st.dataframe(profile, width="stretch", hide_index=True)

with value_tab:
    percentile = st.slider("Premium balance percentile", min_value=50, max_value=95, value=75, step=5)
    threshold = float(data["Balance"].quantile(percentile / 100))
    premium, premium_kpis = high_value_summary(filtered, threshold)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Balance threshold", f"{threshold:,.2f}")
    c2.metric("High-value customers", f"{premium_kpis['customers']:,}")
    c3.metric("High-value churn rate", f"{premium_kpis['churn_rate']:.1%}")
    c4.metric("Churned balance at risk", f"{premium_kpis['balance_at_risk']:,.2f}")
    st.warning(
        "Balance at risk is an exposure proxy, not revenue loss. Revenue estimation requires fees, margin, cost-to-serve, and customer lifetime data."
    )

    if not premium.empty:
        premium_geo = segment_summary(premium, "Geography")
        st.plotly_chart(churn_bar(premium_geo, "Geography", "High-value churn by geography"), width="stretch")
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

with ml_tab:
    st.subheader("Supervised-learning model comparison")
    st.write(
        "The models predict the probability of Exited = 1. Logistic Regression is the explainable baseline; Random Forest is the main model because it captures nonlinear churn patterns."
    )
    st.caption("CustomerId, Surname, and the constant Year field are deliberately excluded from training.")
    if st.button("Train and compare models", type="primary"):
        model_result = fit_models(data)
        st.session_state["model_result"] = model_result

    if "model_result" in st.session_state:
        model_result = st.session_state["model_result"]
        comparison_rows = []
        for model_name, probability_key in [
            ("Logistic Regression", "logistic_probabilities"),
            ("Random Forest", "random_forest_probabilities"),
        ]:
            model_metrics = evaluate_probabilities(
                model_result["y_test"], model_result[probability_key], 0.50
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
            "For retention outreach, recall is often important because it measures how many actual churners the model finds. Threshold choice must reflect campaign capacity and cost."
        )

        cm = metrics["confusion_matrix"]
        cm_fig = px.imshow(
            cm,
            text_auto=True,
            labels={"x": "Predicted", "y": "Actual", "color": "Customers"},
            x=["Retained", "Churned"],
            y=["Retained", "Churned"],
            title="Confusion matrix",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(cm_fig, width="stretch")

        importances = model_result["feature_importances"].head(15).sort_values("Importance")
        importance_fig = px.bar(
            importances,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Random Forest feature importance",
        )
        st.plotly_chart(importance_fig, width="stretch")
        st.warning(
            "Feature importance shows model reliance, not causation. Before real banking use, test calibration, drift, privacy, and performance across demographic groups."
        )
