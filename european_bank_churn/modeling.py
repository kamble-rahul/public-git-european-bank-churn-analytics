"""Reproducible supervised-learning pipelines and evaluation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import (
    CATEGORICAL_FEATURES,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    NUMERIC_FEATURES,
)


def _feature_target_split(
    df: pd.DataFrame, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    from sklearn.model_selection import train_test_split

    features = list(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    target = df["Exited"].astype(int)
    if target.nunique() < 2:
        raise ValueError("Model training requires both retained and churned customers.")
    return train_test_split(
        df[features],
        target,
        test_size=DEFAULT_TEST_SIZE,
        random_state=random_state,
        stratify=target,
    )


def _logistic_pipeline(random_state: int):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
            ("numeric", numeric_pipeline, list(NUMERIC_FEATURES)),
            ("categorical", categorical_pipeline, list(CATEGORICAL_FEATURES)),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_logistic_baseline(
    df: pd.DataFrame, random_state: int = DEFAULT_RANDOM_STATE
) -> dict[str, Any]:
    """Train an explainable baseline on a reproducible stratified holdout set."""
    x_train, x_test, y_train, y_test = _feature_target_split(df, random_state)
    pipeline = _logistic_pipeline(random_state)
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_test)[:, 1]

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


def train_model_comparison(
    df: pd.DataFrame, random_state: int = DEFAULT_RANDOM_STATE
) -> dict[str, Any]:
    """Train Logistic Regression and Random Forest on the same holdout split."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    x_train, x_test, y_train, y_test = _feature_target_split(df, random_state)

    logistic = _logistic_pipeline(random_state)
    logistic.fit(x_train, y_train)
    logistic_probabilities = logistic.predict_proba(x_test)[:, 1]

    tree_preprocessor = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), list(NUMERIC_FEATURES)),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                list(CATEGORICAL_FEATURES),
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
    random_forest.fit(x_train, y_train)
    random_forest_probabilities = random_forest.predict_proba(x_test)[:, 1]

    feature_names = random_forest.named_steps["preprocessor"].get_feature_names_out()
    importances = random_forest.named_steps["model"].feature_importances_
    importance_table = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values("Importance", ascending=False)

    return {
        "y_test": y_test.to_numpy(),
        "test_rows": int(len(y_test)),
        "logistic_pipeline": logistic,
        "logistic_probabilities": logistic_probabilities,
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

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
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
