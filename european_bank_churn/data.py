"""Dataset ingestion, validation, cleaning, and feature engineering."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd

from .config import (
    AGE_LABELS,
    BINARY_COLUMNS,
    CREDIT_LABELS,
    DEFAULT_HIGH_VALUE_QUANTILE,
    EXPECTED_GENDERS,
    EXPECTED_GEOGRAPHIES,
    NUMERIC_COLUMNS,
    REQUIRED_COLUMNS,
    TENURE_LABELS,
)

DASHBOARD_DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "european_bank_dashboard.csv.gz"
)


def load_excel(source: str | Path | BinaryIO) -> pd.DataFrame:
    """Load the first worksheet of an Excel workbook without modifying the source."""
    return pd.read_excel(source)


def load_dashboard_data(path: str | Path = DASHBOARD_DATA_PATH) -> pd.DataFrame:
    """Load the bundled, standardized dataset used by the public dashboard."""
    return pd.read_csv(path)


def standardize_for_dashboard(df: pd.DataFrame) -> pd.DataFrame:
    """Replace direct identifiers while preserving analytical values and row order."""
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    standardized = df.copy()
    standardized["CustomerId"] = [
        f"DEMO-{row_number:05d}" for row_number in range(1, len(standardized) + 1)
    ]
    standardized["Surname"] = "Anonymous"
    return standardized


def _finding(level: str, message: str) -> dict[str, str]:
    return {"level": level, "message": message}


def validate_dataset(df: pd.DataFrame) -> list[dict[str, str]]:
    """Return human-readable data-quality findings without changing the data."""
    findings: list[dict[str, str]] = []
    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        return [_finding("error", f"Missing required columns: {', '.join(missing_columns)}")]

    if df.empty:
        return [_finding("error", "The worksheet has no data rows.")]

    required = sorted(REQUIRED_COLUMNS)
    missing_cells = int(df[required].isna().sum().sum())
    if missing_cells:
        findings.append(
            _finding("warning", f"There are {missing_cells:,} missing cells in required fields.")
        )

    duplicated_ids = int(df["CustomerId"].duplicated().sum())
    if duplicated_ids:
        findings.append(
            _finding("warning", f"There are {duplicated_ids:,} duplicate CustomerId values.")
        )

    for column in NUMERIC_COLUMNS:
        coerced = pd.to_numeric(df[column], errors="coerce")
        invalid_count = int(coerced.isna().sum() - df[column].isna().sum())
        if invalid_count:
            findings.append(
                _finding(
                    "error",
                    f"{column} contains {invalid_count:,} value(s) that are not numeric.",
                )
            )

    for column in BINARY_COLUMNS:
        values = set(pd.to_numeric(df[column], errors="coerce").dropna().unique())
        unexpected = sorted(values - {0, 1})
        if unexpected:
            findings.append(
                _finding("error", f"{column} must contain only 0 and 1; found {unexpected}.")
            )

    numeric_products = pd.to_numeric(df["NumOfProducts"], errors="coerce")
    invalid_products = numeric_products.le(0) | numeric_products.mod(1).ne(0)
    if int(invalid_products.fillna(False).sum()):
        findings.append(
            _finding("warning", "NumOfProducts should contain positive whole numbers.")
        )

    invalid_balance = int((pd.to_numeric(df["Balance"], errors="coerce") < 0).sum())
    if invalid_balance:
        findings.append(
            _finding(
                "warning",
                f"There are {invalid_balance:,} negative balances; review them before analysis.",
            )
        )

    numeric_age = pd.to_numeric(df["Age"], errors="coerce")
    unusual_ages = int(((numeric_age < 18) | (numeric_age > 120)).sum())
    if unusual_ages:
        findings.append(
            _finding("warning", f"There are {unusual_ages:,} ages outside the 18–120 range.")
        )

    geographies = set(df["Geography"].dropna().astype(str).str.strip())
    unexpected_geographies = sorted(geographies - EXPECTED_GEOGRAPHIES)
    if unexpected_geographies:
        findings.append(
            _finding(
                "warning",
                f"Unexpected Geography values: {', '.join(unexpected_geographies)}.",
            )
        )

    genders = set(df["Gender"].dropna().astype(str).str.strip())
    unexpected_genders = sorted(genders - EXPECTED_GENDERS)
    if unexpected_genders:
        findings.append(
            _finding("warning", f"Unexpected Gender values: {', '.join(unexpected_genders)}.")
        )

    if "Year" in df.columns and df["Year"].nunique(dropna=True) <= 1:
        findings.append(
            _finding("info", "Year has one value only, so it is excluded from ML features.")
        )

    if not findings:
        findings.append(_finding("info", "Required columns and data-quality rules are valid."))
    return findings


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Clean data types and create mutually exclusive, documented segment fields."""
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    data = df.copy()
    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    for column in ("Geography", "Gender", "Surname"):
        data[column] = data[column].fillna("Unknown").astype(str).str.strip()

    # Rows with an invalid target cannot contribute to a churn rate or train a classifier.
    data = data[data["Exited"].isin([0, 1])].copy()
    data["Exited"] = data["Exited"].astype(int)

    # right=False creates [30, 46), meaning ages 30 through 45 are one group.
    data["AgeGroup"] = pd.cut(
        data["Age"],
        bins=[-np.inf, 30, 46, 60, np.inf],
        labels=AGE_LABELS,
        right=False,
    )
    data["CreditBand"] = pd.cut(
        data["CreditScore"],
        bins=[-np.inf, 580, 670, np.inf],
        labels=CREDIT_LABELS,
        right=False,
    )
    data["TenureGroup"] = pd.cut(
        data["Tenure"],
        bins=[-np.inf, 3, 7, np.inf],
        labels=TENURE_LABELS,
        right=False,
    )

    positive_balances = data.loc[data["Balance"] > 0, "Balance"]
    positive_balance_median = (
        float(positive_balances.median()) if not positive_balances.empty else 0.0
    )
    data["BalanceSegment"] = np.select(
        [
            data["Balance"].eq(0),
            data["Balance"].gt(0) & data["Balance"].le(positive_balance_median),
            data["Balance"].gt(positive_balance_median),
        ],
        ["Zero balance", "Low balance", "High balance"],
        default="Unknown",
    )

    high_value_threshold = float(data["Balance"].quantile(DEFAULT_HIGH_VALUE_QUANTILE))
    data["HighValue"] = data["Balance"] >= high_value_threshold
    data["ActivityLabel"] = data["IsActiveMember"].map({0: "Inactive", 1: "Active"})
    data["ChurnLabel"] = data["Exited"].map({0: "Retained", 1: "Churned"})

    thresholds = {
        "positive_balance_median": positive_balance_median,
        "high_value_threshold": high_value_threshold,
    }
    return data, thresholds
