"""Project-wide schema, segment, and model configuration."""

from __future__ import annotations

REQUIRED_COLUMNS = frozenset(
    {
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
)

NUMERIC_COLUMNS = (
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Exited",
)

NUMERIC_FEATURES = (
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
)

CATEGORICAL_FEATURES = ("Geography", "Gender")
BINARY_COLUMNS = ("HasCrCard", "IsActiveMember", "Exited")

EXPECTED_GEOGRAPHIES = frozenset({"France", "Germany", "Spain"})
EXPECTED_GENDERS = frozenset({"Female", "Male"})

AGE_LABELS = ("<30", "30-45", "46-59", "60+")
CREDIT_LABELS = ("Low (<580)", "Medium (580-669)", "High (670+)")
TENURE_LABELS = ("New (0-2)", "Mid-term (3-6)", "Long-term (7-10)")

DEFAULT_HIGH_VALUE_QUANTILE = 0.75
DEFAULT_TEST_SIZE = 0.20
DEFAULT_RANDOM_STATE = 42
DEFAULT_CV_FOLDS = 5
MODEL_VERSION = "1.1.0"

SEGMENT_DIMENSIONS = {
    "Age group": "AgeGroup",
    "Credit-score band": "CreditBand",
    "Tenure group": "TenureGroup",
    "Balance segment": "BalanceSegment",
    "Products held": "NumOfProducts",
    "Activity": "ActivityLabel",
    "Gender": "Gender",
}
