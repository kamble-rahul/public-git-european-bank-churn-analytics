"""Streamlit interface for the customer segmentation and churn project."""

from __future__ import annotations

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
from .business import simulate_retention_campaign
from .config import AGE_LABELS, SEGMENT_DIMENSIONS
from .data import DASHBOARD_DATA_PATH, load_dashboard_data, prepare_data, validate_dataset
from .modeling import (
    calibration_table,
    cross_validate_models,
    evaluate_probabilities,
    explain_logistic_customer,
    subgroup_performance,
    train_model_comparison,
)
from .visualization import (
    calibration_figure,
    churn_bar,
    confusion_matrix_figure,
    feature_importance_figure,
    geography_age_heatmap,
    logistic_coefficient_figure,
    permutation_importance_figure,
    salary_balance_scatter,
)

st.set_page_config(
    page_title="European Bank Churn Analytics",
    page_icon="🏦",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def read_standardized_data() -> pd.DataFrame:
    """Read the bundled dashboard dataset once per application process."""
    return load_dashboard_data()


@st.cache_resource(show_spinner="Training Logistic Regression and Random Forest...")
def fit_models(data: pd.DataFrame) -> dict:
    """Cache trained models for an unchanged prepared dataset."""
    return train_model_comparison(data)


@st.cache_data(show_spinner="Running five-fold stratified cross-validation...")
def validate_models(data: pd.DataFrame) -> pd.DataFrame:
    """Cache expensive cross-validation results for an unchanged dataset."""
    return cross_validate_models(data)


def _require_model_result(data: pd.DataFrame, button_key: str) -> dict | None:
    """Provide one consistent model-training gate across advanced dashboard tabs."""
    if "model_result" not in st.session_state:
        st.info("Train the models once to unlock validation, explanations, fairness, and ROI.")
        if st.button("Train models and unlock advanced analysis", key=button_key, type="primary"):
            st.session_state["model_result"] = fit_models(data)
    return st.session_state.get("model_result")


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
    geography_summary["GeographicRiskIndex"] = (
        geography_summary["ChurnRate"] / kpis["churn_rate"]
        if kpis["churn_rate"]
        else 0.0
    )
    st.plotly_chart(
        churn_bar(geography_summary, "Geography", "Churn rate by geography"),
        width="stretch",
    )
    st.subheader("Geographic risk index")
    st.dataframe(
        geography_summary[
            ["Geography", "Customers", "Churners", "ChurnRate", "GeographicRiskIndex"]
        ].style.format(
            {"ChurnRate": "{:.1%}", "GeographicRiskIndex": "{:.2f}x"}
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "A value above 1.00x means the geography's churn rate is above the current "
        "filtered portfolio average."
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
    st.plotly_chart(salary_balance_scatter(premium), width="stretch")
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
    st.caption(
        "CustomerId values are generated dashboard record IDs. Original customer names and "
        "identifiers are not included in this public application."
    )
    st.download_button(
        "Download filtered high-value customers",
        data=premium[columns].to_csv(index=False).encode("utf-8"),
        file_name="high_value_customers.csv",
        mime="text/csv",
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
    model_result = _require_model_result(data, "train_models_comparison")
    if model_result is None:
        return

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
                "Brier score": model_metrics["brier_score"],
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
                "Brier score": "{:.3f}",
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
        "Feature importance shows model reliance, not causation. Use the validation and "
        "explainability tabs for stronger diagnostics before interpreting the model."
    )


def _render_model_validation(data: pd.DataFrame) -> None:
    st.subheader("Model robustness and probability calibration")
    st.write(
        "Holdout metrics measure one unseen sample. Five-fold validation checks whether model "
        "quality is stable across multiple class-balanced splits."
    )
    model_result = _require_model_result(data, "train_models_validation")
    if model_result is None:
        return

    y_test = model_result["y_test"]
    curves = {
        "Logistic Regression": calibration_table(
            y_test, model_result["logistic_probabilities"]
        ),
        "Random Forest": calibration_table(
            y_test, model_result["random_forest_probabilities"]
        ),
    }
    logistic_metrics = evaluate_probabilities(
        y_test, model_result["logistic_probabilities"]
    )
    forest_metrics = evaluate_probabilities(
        y_test, model_result["random_forest_probabilities"]
    )
    c1, c2 = st.columns(2)
    c1.metric("Logistic Brier score", f"{logistic_metrics['brier_score']:.3f}")
    c2.metric("Random Forest Brier score", f"{forest_metrics['brier_score']:.3f}")
    st.caption(
        "Brier score measures probability error; lower is better. Always read it together "
        "with the reliability curve and ranking metrics."
    )
    st.plotly_chart(calibration_figure(curves), width="stretch")

    risk_bands = model_result["test_records"].copy()
    risk_bands["RiskBand"] = pd.cut(
        risk_bands["RandomForestProbability"],
        bins=[-float("inf"), 0.30, 0.60, float("inf")],
        labels=["Low (<30%)", "Medium (30-59%)", "High (60%+)"],
        include_lowest=True,
        right=False,
    )
    band_summary = (
        risk_bands.groupby("RiskBand", observed=True)
        .agg(
            Customers=("CustomerId", "size"),
            MeanPredictedProbability=("RandomForestProbability", "mean"),
            ObservedChurnRate=("ActualExited", "mean"),
        )
        .reset_index()
    )
    st.subheader("Random Forest risk bands")
    st.dataframe(
        band_summary.style.format(
            {
                "MeanPredictedProbability": "{:.1%}",
                "ObservedChurnRate": "{:.1%}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    if st.button("Run five-fold cross-validation", key="run_cross_validation"):
        st.session_state["cross_validation"] = validate_models(data)
    if "cross_validation" in st.session_state:
        validation = st.session_state["cross_validation"]
        percentage_columns = [
            column for column in validation.columns if column not in {"Model", "Folds"}
        ]
        st.subheader("Five-fold validation results")
        st.dataframe(
            validation.style.format({column: "{:.3f}" for column in percentage_columns}),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Mean measures average fold performance; standard deviation shows variability. "
            "Temporal validation is not possible because the supplied Year field is constant."
        )

    with st.expander("Model and dataset version", expanded=False):
        metadata = model_result["metadata"]
        st.json(
            {
                "model_version": metadata["model_version"],
                "dataset_fingerprint": metadata["dataset_fingerprint"],
                "random_state": metadata["random_state"],
                "training_rows": metadata["training_rows"],
                "holdout_rows": metadata["test_rows"],
            }
        )


def _render_explainability_and_fairness(data: pd.DataFrame) -> None:
    st.subheader("Model explainability")
    st.write(
        "Permutation importance tests the Random Forest on unseen rows. Logistic coefficients "
        "add direction, showing associations with higher or lower predicted churn."
    )
    model_result = _require_model_result(data, "train_models_explainability")
    if model_result is None:
        return

    st.plotly_chart(
        permutation_importance_figure(model_result["permutation_importances"]),
        width="stretch",
    )
    st.plotly_chart(
        logistic_coefficient_figure(model_result["logistic_coefficients"]),
        width="stretch",
    )
    st.caption(
        "These are predictive associations, not proof that changing a feature will prevent churn."
    )

    st.subheader("Customer-level explanation")
    records = model_result["test_records"]
    customer_id = st.selectbox(
        "Select a generated dashboard record ID",
        records["CustomerId"].astype(str).sort_values().tolist(),
        key="explanation_customer",
    )
    customer = records[records["CustomerId"].astype(str).eq(customer_id)].head(1)
    probability = float(customer["RandomForestProbability"].iloc[0])
    logistic_probability = float(customer["LogisticProbability"].iloc[0])
    risk_label = "High" if probability >= 0.60 else "Medium" if probability >= 0.30 else "Low"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Random Forest probability", f"{probability:.1%}")
    c2.metric("Logistic probability", f"{logistic_probability:.1%}")
    c3.metric("Random Forest risk band", risk_label)
    c4.metric(
        "Observed outcome",
        "Churned" if int(customer["ActualExited"].iloc[0]) else "Retained",
    )
    explanation = explain_logistic_customer(
        model_result["logistic_pipeline"], customer
    ).head(10)
    st.dataframe(
        explanation.style.format({"Contribution": "{:+.3f}"}),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "The signed contribution table explains the Logistic Regression probability. The Random "
        "Forest probability is shown separately because its behavior is nonlinear."
    )

    actions = []
    if int(customer["IsActiveMember"].iloc[0]) == 0:
        actions.append("Offer a reviewed re-engagement conversation; the account is inactive.")
    if int(customer["NumOfProducts"].iloc[0]) == 1:
        actions.append("Review whether a relevant additional product could improve engagement.")
    if float(customer["Balance"].iloc[0]) >= float(data["Balance"].quantile(0.75)):
        actions.append("Prioritize a relationship-manager review because the balance is high.")
    if not actions:
        actions.append("Collect feedback before choosing a retention treatment.")
    st.write("Illustrative reviewed actions:")
    for action in actions:
        st.markdown(f"- {action}")

    st.subheader("Subgroup performance audit")
    group_label = st.selectbox(
        "Audit model outcomes by",
        ["Gender", "Geography", "Age group"],
        key="fairness_group",
    )
    group_column = {
        "Gender": "Gender",
        "Geography": "Geography",
        "Age group": "AgeGroup",
    }[group_label]
    fairness_threshold = st.slider(
        "Fairness-audit decision threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        key="fairness_threshold",
    )
    audit = subgroup_performance(
        model_result["y_test"],
        model_result["random_forest_probabilities"],
        records[group_column],
        fairness_threshold,
    )
    rate_columns = [
        "ActualChurnRate",
        "PredictedHighRiskRate",
        "Precision",
        "Recall",
        "FalsePositiveRate",
        "FalseNegativeRate",
    ]
    st.dataframe(
        audit.style.format({column: "{:.1%}" for column in rate_columns}),
        width="stretch",
        hide_index=True,
    )
    st.warning(
        "Different subgroup metrics are a screening signal, not a legal fairness conclusion. "
        "Review sample sizes, confidence intervals, feature necessity, and applicable policy."
    )


def _render_retention_roi(data: pd.DataFrame) -> None:
    st.subheader("Retention campaign ROI simulator")
    st.write(
        "Rank held-out customers by predicted churn risk and test transparent campaign assumptions."
    )
    model_result = _require_model_result(data, "train_models_roi")
    if model_result is None:
        return

    records = model_result["test_records"]
    capacity = st.slider(
        "Maximum customers the campaign can contact",
        min_value=1,
        max_value=len(records),
        value=min(500, len(records)),
        step=1,
    )
    c1, c2, c3 = st.columns(3)
    contact_cost = c1.number_input(
        "Contact cost per customer",
        min_value=0.0,
        value=10.0,
        step=1.0,
    )
    success_rate_percent = c2.slider(
        "Retention success rate (%)",
        min_value=0,
        max_value=100,
        value=20,
        step=5,
    )
    success_rate = success_rate_percent / 100
    retained_value = c3.number_input(
        "Estimated value per retained customer",
        min_value=0.0,
        value=1_000.0,
        step=100.0,
    )
    scenario, targets = simulate_retention_campaign(
        records,
        capacity,
        contact_cost,
        success_rate,
        retained_value,
    )
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Customers targeted", f"{scenario['customers_targeted']:,}")
    r2.metric("Expected retained", f"{scenario['expected_customers_retained']:.1f}")
    r3.metric("Campaign cost", f"{scenario['campaign_cost']:,.2f}")
    r4.metric("Estimated net benefit", f"{scenario['estimated_net_benefit']:,.2f}")
    r5, r6, r7 = st.columns(3)
    r5.metric(
        "Expected churners reached",
        f"{scenario['expected_churners_reached']:.1f}",
    )
    r6.metric(
        "Estimated value protected",
        f"{scenario['estimated_value_protected']:,.2f}",
    )
    roi = scenario["estimated_roi"]
    r7.metric("Estimated ROI", f"{roi:.1%}" if pd.notna(roi) else "N/A")

    display_columns = [
        "CustomerId",
        "RandomForestProbability",
        "Geography",
        "Age",
        "Balance",
        "NumOfProducts",
        "IsActiveMember",
    ]
    st.subheader("Highest-risk customers selected by campaign capacity")
    st.dataframe(
        targets[display_columns].head(200).style.format(
            {
                "RandomForestProbability": "{:.1%}",
                "Balance": "{:,.2f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "Download campaign target list",
        data=targets[display_columns].to_csv(index=False).encode("utf-8"),
        file_name="retention_campaign_targets.csv",
        mime="text/csv",
    )
    st.warning(
        "This is a scenario, not forecast revenue. It uses uncalibrated holdout probabilities "
        "and user assumptions because the dataset has no campaign cost, margin, or lifetime value."
    )


def run_dashboard() -> None:
    """Render the complete Streamlit dashboard."""
    st.title("Customer Segmentation & Churn Pattern Analytics")
    st.caption(
        "Interactive analysis of the standardized European banking project dataset. "
        "No file upload is required."
    )

    try:
        raw = read_standardized_data()
    except Exception as exc:
        st.error("The standardized dashboard dataset could not be loaded.")
        st.caption(f"Expected application asset: {DASHBOARD_DATA_PATH.name}. Details: {exc}")
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
    tenure_options = [str(value) for value in data["TenureGroup"].cat.categories]
    tenure_groups = st.sidebar.multiselect(
        "Tenure group",
        tenure_options,
        default=tenure_options,
    )
    credit_options = [str(value) for value in data["CreditBand"].cat.categories]
    credit_bands = st.sidebar.multiselect(
        "Credit-score band",
        credit_options,
        default=credit_options,
    )
    balance_options = sorted(data["BalanceSegment"].dropna().unique())
    balance_segments = st.sidebar.multiselect(
        "Balance segment",
        balance_options,
        default=balance_options,
    )
    activity_options = sorted(data["ActivityLabel"].dropna().unique())
    activity_labels = st.sidebar.multiselect(
        "Activity",
        activity_options,
        default=activity_options,
    )
    product_options = sorted(data["NumOfProducts"].dropna().astype(int).unique())
    products = st.sidebar.multiselect(
        "Number of products",
        product_options,
        default=product_options,
    )

    filtered = data[
        data["Geography"].isin(geographies)
        & data["Gender"].isin(genders)
        & data["AgeGroup"].astype(str).isin(age_groups)
        & data["TenureGroup"].astype(str).isin(tenure_groups)
        & data["CreditBand"].astype(str).isin(credit_bands)
        & data["BalanceSegment"].isin(balance_segments)
        & data["ActivityLabel"].isin(activity_labels)
        & data["NumOfProducts"].isin(products)
    ].copy()
    if filtered.empty:
        st.warning("The current filters return no customers.")
        st.stop()

    (
        overview,
        segments,
        demographics,
        high_value,
        model,
        validation,
        explainability,
        roi,
    ) = st.tabs(
        [
            "Overview",
            "Segments",
            "Geography & demographics",
            "High-value customers",
            "ML model comparison",
            "Model validation",
            "Explainability & fairness",
            "Retention ROI",
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
    with validation:
        _render_model_validation(data)
    with explainability:
        _render_explainability_and_fairness(data)
    with roi:
        _render_retention_roi(data)
