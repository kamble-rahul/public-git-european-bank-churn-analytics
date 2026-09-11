"""Shared synthetic fixtures; no customer data is required for automated tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def customer_frame() -> pd.DataFrame:
    """Small deterministic frame covering every documented age boundary."""
    return pd.DataFrame(
        {
            "CustomerId": range(1, 9),
            "Surname": [f"Customer{i}" for i in range(1, 9)],
            "CreditScore": [500, 579, 580, 669, 670, 720, 800, 640],
            "Geography": ["France", "Germany", "Spain", "France"] * 2,
            "Gender": ["Female", "Male"] * 4,
            "Age": [29, 30, 45, 46, 59, 60, 75, 38],
            "Tenure": [0, 2, 3, 6, 7, 10, 5, 1],
            "Balance": [0, 0, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000],
            "NumOfProducts": [1, 2, 1, 2, 3, 1, 2, 4],
            "HasCrCard": [0, 1, 1, 0, 1, 0, 1, 1],
            "IsActiveMember": [1, 0, 1, 0, 1, 0, 0, 1],
            "EstimatedSalary": [50_000, 55_000, 60_000, 65_000, 70_000, 75_000, 80_000, 85_000],
            "Exited": [0, 1, 0, 1, 0, 1, 0, 1],
            "Year": [2025] * 8,
        }
    )


@pytest.fixture
def modeling_frame() -> pd.DataFrame:
    """Larger synthetic frame suitable for a stratified model test."""
    rng = np.random.default_rng(42)
    rows = 240
    geography = rng.choice(["France", "Germany", "Spain"], size=rows)
    gender = rng.choice(["Female", "Male"], size=rows)
    age = rng.integers(18, 85, size=rows)
    active = rng.integers(0, 2, size=rows)
    probability = (
        0.08
        + 0.18 * (geography == "Germany")
        + 0.20 * (age >= 50)
        + 0.18 * (active == 0)
    ).clip(0, 0.90)
    exited = rng.binomial(1, probability)
    return pd.DataFrame(
        {
            "CustomerId": np.arange(1, rows + 1),
            "Surname": [f"Synthetic{i}" for i in range(rows)],
            "CreditScore": rng.integers(350, 851, size=rows),
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": rng.integers(0, 11, size=rows),
            "Balance": rng.uniform(0, 250_000, size=rows),
            "NumOfProducts": rng.integers(1, 5, size=rows),
            "HasCrCard": rng.integers(0, 2, size=rows),
            "IsActiveMember": active,
            "EstimatedSalary": rng.uniform(10_000, 200_000, size=rows),
            "Exited": exited,
        }
    )
