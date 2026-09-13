"""Reproducible supervised-learning pipelines and evaluation metrics."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    CATEGORICAL_FEATURES,
    DEFAULT_CV_FOLDS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    MODEL_VERSION,
    NUMERIC_FEATURES,
)

MODEL_FEATURES = list(NUMERIC_FEATURES + CATEGORICAL_FEATURES)


def _feature_target_split(
    df: pd.DataFrame, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    from sklearn.model_selection import train_test_split

    target = df["Exited"].astype(int)
    if target.nunique() < 2:
        raise ValueError("Model training requires both retained and churned customers.")
    return train_test_split(
        df[MODEL_FEATURES],
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


def _random_forest_pipeline(random_state: int):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    preprocessor = ColumnTransformer(
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
    return Pipeline(
        [
            ("preprocessor", preprocessor),
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


def _coefficient_table(logistic_pipeline) -> pd.DataFrame:
    feature_names = logistic_pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefficients = logistic_pipeline.named_steps["model"].coef_[0]
    table = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": coefficients,
            "AbsoluteCoefficient": np.abs(coefficients),
        }
    )
    table["Direction"] = np.where(
        table["Coefficient"] >= 0,
        "Higher predicted churn",
        "Lower predicted churn",
    )
    return table.sort_values("AbsoluteCoefficient", ascending=False).reset_index(drop=True)


def _dataset_fingerprint(df: pd.DataFrame) -> str:
    """Create a short reproducibility fingerprint without exposing row values."""
    columns = MODEL_FEATURES + ["Exited"]
    row_hashes = pd.util.hash_pandas_object(df[columns], index=True).to_numpy()
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()[:12]


def calibration_table(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Summarize mean predicted and observed churn rates by probability bin."""
    from sklearn.calibration import calibration_curve

    observed, predicted = calibration_curve(
        y_true,
        probabilities,
        n_bins=n_bins,
        strategy="quantile",
    )
    return pd.DataFrame(
        {
            "MeanPredictedProbability": predicted,
            "ObservedChurnRate": observed,
        }
    )


def subgroup_performance(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    groups: pd.Series,
    threshold: float = 0.50,
) -> pd.DataFrame:
    """Compare classification outcomes across a selected customer grouping."""
    from sklearn.metrics import confusion_matrix, precision_score, recall_score

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if len(y_true) != len(probabilities) or len(y_true) != len(groups):
        raise ValueError("y_true, probabilities, and groups must have equal lengths")

    frame = pd.DataFrame(
        {
            "Actual": np.asarray(y_true, dtype=int),
            "Probability": np.asarray(probabilities, dtype=float),
            "Group": pd.Series(groups).astype(str).to_numpy(),
        }
    )
    frame["Predicted"] = (frame["Probability"] >= threshold).astype(int)
    rows = []
    for group_name, group in frame.groupby("Group", observed=True):
        tn, fp, fn, tp = confusion_matrix(
            group["Actual"], group["Predicted"], labels=[0, 1]
        ).ravel()
        rows.append(
            {
                "Group": group_name,
                "Customers": int(len(group)),
                "ActualChurnRate": float(group["Actual"].mean()),
                "PredictedHighRiskRate": float(group["Predicted"].mean()),
                "Precision": float(
                    precision_score(group["Actual"], group["Predicted"], zero_division=0)
                ),
                "Recall": float(
                    recall_score(group["Actual"], group["Predicted"], zero_division=0)
                ),
                "FalsePositiveRate": float(fp / (fp + tn)) if fp + tn else np.nan,
                "FalseNegativeRate": float(fn / (fn + tp)) if fn + tp else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("Group").reset_index(drop=True)


def explain_logistic_customer(pipeline, customer: pd.DataFrame) -> pd.DataFrame:
    """Return signed feature contributions to one customer's logistic log-odds."""
    if len(customer) != 1:
        raise ValueError("customer must contain exactly one row")
    transformed = pipeline.named_steps["preprocessor"].transform(customer[MODEL_FEATURES])
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefficients = pipeline.named_steps["model"].coef_[0]
    contributions = np.asarray(transformed)[0] * coefficients
    explanation = pd.DataFrame(
        {
            "Feature": feature_names,
            "Contribution": contributions,
        }
    )
    explanation["Direction"] = np.where(
        explanation["Contribution"] >= 0,
        "Increases predicted churn",
        "Reduces predicted churn",
    )
    order = explanation["Contribution"].abs().sort_values(ascending=False).index
    return explanation.reindex(order).reset_index(drop=True)


def cross_validate_models(
    df: pd.DataFrame,
    folds: int = DEFAULT_CV_FOLDS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Evaluate both pipelines with stratified folds and report mean plus variability."""
    from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
    from sklearn.model_selection import StratifiedKFold, cross_validate

    target = df["Exited"].astype(int)
    if folds < 2:
        raise ValueError("folds must be at least 2")
    if target.value_counts().min() < folds:
        raise ValueError("each target class must contain at least one row per fold")

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    scoring = {
        "ROC-AUC": "roc_auc",
        "PR-AUC": "average_precision",
        "Recall": make_scorer(recall_score, zero_division=0),
        "Precision": make_scorer(precision_score, zero_division=0),
        "F1": make_scorer(f1_score, zero_division=0),
    }
    models = {
        "Logistic Regression": _logistic_pipeline(random_state),
        "Random Forest": _random_forest_pipeline(random_state).set_params(model__n_jobs=1),
    }
    rows = []
    for model_name, pipeline in models.items():
        scores = cross_validate(
            pipeline,
            df[MODEL_FEATURES],
            target,
            cv=splitter,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )
        row: dict[str, float | int | str] = {"Model": model_name, "Folds": folds}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}Mean"] = float(values.mean())
            row[f"{metric}Std"] = float(values.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def train_logistic_baseline(
    df: pd.DataFrame, random_state: int = DEFAULT_RANDOM_STATE
) -> dict[str, Any]:
    """Train an explainable baseline on a reproducible stratified holdout set."""
    x_train, x_test, y_train, y_test = _feature_target_split(df, random_state)
    pipeline = _logistic_pipeline(random_state)
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    return {
        "pipeline": pipeline,
        "y_test": y_test.to_numpy(),
        "probabilities": probabilities,
        "coefficients": _coefficient_table(pipeline),
        "test_rows": int(len(y_test)),
    }


def train_model_comparison(
    df: pd.DataFrame, random_state: int = DEFAULT_RANDOM_STATE
) -> dict[str, Any]:
    """Train two models and return holdout diagnostics, explanations, and metadata."""
    from sklearn.inspection import permutation_importance

    x_train, x_test, y_train, y_test = _feature_target_split(df, random_state)

    logistic = _logistic_pipeline(random_state)
    logistic.fit(x_train, y_train)
    logistic_probabilities = logistic.predict_proba(x_test)[:, 1]

    random_forest = _random_forest_pipeline(random_state)
    random_forest.fit(x_train, y_train)
    random_forest_probabilities = random_forest.predict_proba(x_test)[:, 1]

    feature_names = random_forest.named_steps["preprocessor"].get_feature_names_out()
    importances = random_forest.named_steps["model"].feature_importances_
    importance_table = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values("Importance", ascending=False)

    permutation = permutation_importance(
        random_forest,
        x_test,
        y_test,
        scoring="average_precision",
        n_repeats=5,
        random_state=random_state,
        n_jobs=-1,
    )
    permutation_table = pd.DataFrame(
        {
            "Feature": x_test.columns,
            "ImportanceMean": permutation.importances_mean,
            "ImportanceStd": permutation.importances_std,
        }
    ).sort_values("ImportanceMean", ascending=False)

    test_records = df.loc[x_test.index].copy()
    test_records["ActualExited"] = y_test.astype(int)
    test_records["LogisticProbability"] = logistic_probabilities
    test_records["RandomForestProbability"] = random_forest_probabilities
    test_records = test_records.reset_index(drop=True)

    return {
        "y_test": y_test.to_numpy(),
        "test_rows": int(len(y_test)),
        "training_rows": int(len(y_train)),
        "logistic_pipeline": logistic,
        "logistic_probabilities": logistic_probabilities,
        "logistic_coefficients": _coefficient_table(logistic),
        "random_forest_pipeline": random_forest,
        "random_forest_probabilities": random_forest_probabilities,
        "feature_importances": importance_table.reset_index(drop=True),
        "permutation_importances": permutation_table.reset_index(drop=True),
        "test_records": test_records,
        "metadata": {
            "model_version": MODEL_VERSION,
            "dataset_fingerprint": _dataset_fingerprint(df),
            "random_state": random_state,
            "training_rows": int(len(y_train)),
            "test_rows": int(len(y_test)),
        },
    }


def evaluate_probabilities(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.50
) -> dict[str, float | np.ndarray]:
    """Evaluate probability predictions at a business-selected threshold."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
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
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, predictions),
    }
