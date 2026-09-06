"""Reusable data preparation, KPI, segmentation, and ML functions."""

from __future__ import annotations

from typing import BinaryIO

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "CustomerId",
    "Surname",
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Exited",
}

NUMERIC_COLUMNS = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Exited",
]


def load_excel(source: str | BinaryIO) -> pd.DataFrame:
    """Load the first worksheet of an .xlsx file."""
    return pd.read_excel(source)


def validate_dataset(df: pd.DataFrame) -> list[dict[str, str]]:
    """Return human-readable validation findings without changing the data."""
    findings: list[dict[str, str]] = []
    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        findings.append(
            {
                "level": "error",
                "message": f"Missing required columns: {', '.join(missing_columns)}",
            }
        )
        return findings

    if df.empty:
        findings.append({"level": "error", "message": "The worksheet has no data rows."})
        return findings

    missing_cells = int(df[list(REQUIRED_COLUMNS)].isna().sum().sum())
    if missing_cells:
        findings.append(
            {
                "level": "warning",
                "message": f"There are {missing_cells:,} missing cells in required fields.",
            }
        )

    duplicated_ids = int(df["CustomerId"].duplicated().sum())
    if duplicated_ids:
        findings.append(
            {
                "level": "warning",
                "message": f"There are {duplicated_ids:,} duplicate CustomerId values.",
            }
        )

    for column in ["HasCrCard", "IsActiveMember", "Exited"]:
        values = set(pd.to_numeric(df[column], errors="coerce").dropna().unique())
        unexpected = sorted(values - {0, 1})
        if unexpected:
            findings.append(
                {
                    "level": "error",
                    "message": f"{column} must contain only 0 and 1; found {unexpected}.",
                }
            )

    invalid_balance = int((pd.to_numeric(df["Balance"], errors="coerce") < 0).sum())
    if invalid_balance:
        findings.append(
            {
                "level": "warning",
                "message": f"There are {invalid_balance:,} negative balances; review them before analysis.",
            }
        )

    if "Year" in df.columns and df["Year"].nunique(dropna=True) <= 1:
        findings.append(
            {
                "level": "info",
                "message": "Year has one value only, so it should not be used as an ML feature.",
            }
        )

    if not findings:
        findings.append(
            {"level": "info", "message": "Required columns and binary fields are valid."}
        )
    return findings


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Clean types and create mutually exclusive, documented segment fields."""
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    data = df.copy()
    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    for column in ["Geography", "Gender", "Surname"]:
        data[column] = data[column].fillna("Unknown").astype(str).str.strip()

    # A missing/invalid target cannot contribute to a churn rate.
    data = data[data["Exited"].isin([0, 1])].copy()
    data["Exited"] = data["Exited"].astype(int)

    # right=False makes the bands: [30,46) = ages 30 through 45, for example.
    data["AgeGroup"] = pd.cut(
        data["Age"],
        bins=[-np.inf, 30, 46, 60, np.inf],
        labels=["<30", "30-45", "46-59", "60+"],
        right=False,
    )
    data["CreditBand"] = pd.cut(
        data["CreditScore"],
        bins=[-np.inf, 580, 670, np.inf],
        labels=["Low (<580)", "Medium (580-669)", "High (670+)"],
        right=False,
    )
    data["TenureGroup"] = pd.cut(
        data["Tenure"],
        bins=[-np.inf, 3, 7, np.inf],
        labels=["New (0-2)", "Mid-term (3-6)", "Long-term (7-10)"],
        right=False,
    )

    positive_balances = data.loc[data["Balance"] > 0, "Balance"]
    positive_balance_median = float(positive_balances.median())
    data["BalanceSegment"] = np.select(
        [
            data["Balance"].eq(0),
            data["Balance"].gt(0) & data["Balance"].le(positive_balance_median),
            data["Balance"].gt(positive_balance_median),
        ],
        ["Zero balance", "Low balance", "High balance"],
        default="Unknown",
    )

    high_value_threshold = float(data["Balance"].quantile(0.75))
    data["HighValue"] = data["Balance"] >= high_value_threshold
    data["ActivityLabel"] = data["IsActiveMember"].map({0: "Inactive", 1: "Active"})
    data["ChurnLabel"] = data["Exited"].map({0: "Retained", 1: "Churned"})

    thresholds = {
        "positive_balance_median": positive_balance_median,
        "high_value_threshold": high_value_threshold,
    }
    return data, thresholds


def segment_summary(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Calculate size, churn rate, contribution, and average financial profile."""
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
    """Return the headline KPIs for the current filtered population."""
    customer_count = int(len(df))
    churners = int(df["Exited"].sum())
    churn_rate = float(df["Exited"].mean()) if customer_count else 0.0

    geo = segment_summary(df, "Geography") if customer_count else pd.DataFrame()
    if not geo.empty and churn_rate:
        geo["GeographicRiskIndex"] = geo["ChurnRate"] / churn_rate
        riskiest = geo.sort_values("GeographicRiskIndex", ascending=False).iloc[0]
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
    """Summarize customers whose balance is at or above a chosen threshold."""
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


def train_logistic_baseline(df: pd.DataFrame, random_state: int = 42) -> dict:
    """Train an explainable churn baseline on a stratified holdout set."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric_features = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]
    categorical_features = ["Geography", "Gender"]
    features = numeric_features + categorical_features

    X = df[features]
    y = df["Exited"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced", max_iter=1_000, random_state=random_state
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefficients = pipeline.named_steps["model"].coef_[0]
    coefficient_table = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": coefficients,
            "AbsoluteCoefficient": np.abs(coefficients),
        }
    ).sort_values("AbsoluteCoefficient", ascending=False)

    return {
        "pipeline": pipeline,
        "y_test": y_test.to_numpy(),
        "probabilities": probabilities,
        "coefficients": coefficient_table,
        "test_rows": int(len(y_test)),
    }


def train_model_comparison(df: pd.DataFrame, random_state: int = 42) -> dict:
    """Train the explainable baseline and the main Random Forest model."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    logistic = train_logistic_baseline(df, random_state=random_state)

    numeric_features = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]
    categorical_features = ["Geography", "Gender"]
    features = numeric_features + categorical_features

    X = df[features]
    y = df["Exited"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )

    tree_preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                SimpleImputer(strategy="median"),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    random_forest = Pipeline(
        [
            ("preprocessor", tree_preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=10,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    random_forest.fit(X_train, y_train)
    random_forest_probabilities = random_forest.predict_proba(X_test)[:, 1]

    feature_names = random_forest.named_steps["preprocessor"].get_feature_names_out()
    importances = random_forest.named_steps["model"].feature_importances_
    importance_table = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values("Importance", ascending=False)

    return {
        "y_test": y_test.to_numpy(),
        "test_rows": int(len(y_test)),
        "logistic_pipeline": logistic["pipeline"],
        "logistic_probabilities": logistic["probabilities"],
        "random_forest_pipeline": random_forest,
        "random_forest_probabilities": random_forest_probabilities,
        "feature_importances": importance_table,
    }


def evaluate_probabilities(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.50
) -> dict[str, float | np.ndarray]:
    """Evaluate probability predictions at a business-selected threshold."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, predictions),
    }
