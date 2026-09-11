# Contributing

## Development setup

1. Create and activate a Python 3.11 or 3.12 virtual environment.
2. Run `python -m pip install -r requirements-dev.txt`.
3. Create a focused branch from `main`.
4. Make the smallest change that solves the issue.
5. Run `ruff check .` and `pytest --cov=european_bank_churn`.
6. Test `streamlit run app.py` with synthetic or authorized local data.

## Analytical change checklist

Any change to a KPI, segment, feature, model, or threshold must document:

- the business definition and denominator;
- why the previous behavior is insufficient;
- the effect on existing results;
- tests for boundaries and empty/small groups;
- fairness, privacy, and interpretation risks.

## Data policy

Never commit customer-level workbooks, exported rows, identifiers, credentials, secrets, serialized
models trained on restricted data, or screenshots containing customer information. Use synthetic or
fully approved anonymized examples in issues and tests.

## Pull requests

Keep pull requests focused. Complete the repository template, link the related issue, explain how the
change was validated, and call out any result that changes the documented executive summary.
