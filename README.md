# European Bank Customer Segmentation & Churn Analytics

[![Python quality checks](https://github.com/kamble-rahul/public-git-european-bank-churn-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/kamble-rahul/public-git-european-bank-churn-analytics/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)

An end-to-end portfolio project for validating customer data, creating business-defined
segments, measuring churn patterns, comparing classification models, and presenting the
results in an interactive Streamlit dashboard.

> The public application automatically loads a standardized, de-identified copy of the supplied
> project dataset. Original customer IDs and surnames are not published. The source workbook does
> not contain authoritative provenance proving that it came from the European Central Bank, so
> this project describes it as an educational European banking dataset.

## Business questions

1. What is the overall customer churn rate?
2. Which geographic, demographic, tenure, engagement, credit, and balance segments have the
   highest churn rate and churn contribution?
3. Is churn concentrated among high-balance customers?
4. How do an explainable Logistic Regression baseline and a nonlinear Random Forest compare?

## Product capabilities

- Validates schema, missing values, duplicates, numeric types, binary fields, and unusual ranges.
- Creates non-overlapping age, credit-score, tenure, balance, activity, and churn segments.
- Updates KPIs and charts from geography, gender, and age filters.
- Provides segment drill-down tables and CSV downloads.
- Explores geography × age interactions and high-value balance exposure.
- Filters by geography, gender, age, tenure, credit, balance, activity, and product count.
- Compares estimated salary and balance visually for high-value churn analysis.
- Compares Logistic Regression and Random Forest using ROC-AUC, PR-AUC, recall, precision,
  F1, accuracy, and a confusion matrix.
- Allows the Random Forest decision threshold to be adjusted for campaign capacity.
- Opens the complete dashboard immediately; visitors do not need to upload a file.

## Verified analytical highlights

Using the supplied 10,000-row workbook and the documented segment rules:

| Measure | Result |
|---|---:|
| Overall churn | 20.37% |
| Germany churn | 32.44% |
| France churn | 16.15% |
| Spain churn | 16.67% |
| Inactive/active churn-risk ratio | 1.88× |
| High-value threshold (75th percentile) | 127,644.24 |
| High-value churn | 23.68% |

These values are descriptive associations, not causal effects. `Balance at risk` is an exposure
proxy, not recognized revenue loss.

## Repository structure

```text
.
├── .github/                 # CI workflow, issue and pull-request templates
├── data/                    # Private raw folders plus the de-identified dashboard asset
├── docs/                    # Architecture, dictionary, methodology, and model card
├── european_bank_churn/     # Reusable production Python package
│   ├── analytics.py         # KPI and segment calculations
│   ├── config.py            # Schema, band definitions, and model settings
│   ├── dashboard.py         # Streamlit page composition
│   ├── data.py              # Ingestion, validation, and feature engineering
│   ├── modeling.py          # Logistic Regression and Random Forest pipelines
│   └── visualization.py     # Plotly chart builders
├── models/                  # Local model artifacts; contents are gitignored
├── notebooks/               # Numbered exploration notebooks and guidance
├── reports/                 # Research paper, executive summary, and figures
├── scripts/                 # Reproducible report-generation commands
├── tests/                   # Synthetic unit and integration tests
├── app.py                   # Thin Streamlit Cloud entrypoint
├── Makefile                 # Common local commands
├── pyproject.toml           # Package and quality-tool configuration
├── requirements.txt         # Streamlit/runtime dependencies
└── smoke_test.py            # End-to-end check against a local workbook
```

This structure is intentionally based on the separation encouraged by
[Cookiecutter Data Science](https://github.com/drivendataorg/cookiecutter-data-science), while
remaining small enough for a beginner to understand.

## Quick start

### 1. Clone and create an environment

```bash
git clone https://github.com/kamble-rahul/public-git-european-bank-churn-analytics.git
cd public-git-european-bank-churn-analytics
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Run the application

```bash
streamlit run app.py
```

The application opens directly with the bundled standardized dataset—no upload is required.

### 3. Run quality checks

```bash
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m pytest --cov=european_bank_churn
```

To validate the real workbook locally without publishing it:

```bash
python smoke_test.py "/full/path/to/European_Bank (5).xlsx"
```

To rebuild the public, de-identified dashboard asset from an authorized source workbook:

```bash
python scripts/build_dashboard_dataset.py "/full/path/to/European_Bank (5).xlsx"
```

The build replaces every original `CustomerId` with a generated `DEMO-xxxxx` record ID and every
surname with `Anonymous`, then verifies the row count and churn target before writing the compressed
asset. Never commit the private source workbook.

To regenerate the full Markdown EDA report from the workbook:

```bash
python scripts/generate_eda_report.py "/full/path/to/European_Bank (5).xlsx"
```

## Method summary

```mermaid
flowchart LR
    A[Private source workbook] --> B[Validate and de-identify offline]
    B --> C[Bundled standardized data]
    C --> D[Type cleaning and segment features]
    D --> E[KPIs and EDA]
    D --> F[Stratified train/test split]
    F --> G[Logistic Regression]
    F --> H[Random Forest]
    E --> I[Direct-load Streamlit dashboard]
    G --> I
    H --> I
```

The required project is primarily segmentation and descriptive analytics. ML is a secondary
component used to rank churn risk, not a replacement for business KPI analysis.

## Documentation and deliverables

- [Architecture](docs/architecture.md)
- [Beginner implementation guide](docs/implementation_guide.md)
- [Data dictionary](docs/data_dictionary.md)
- [Analytical methodology and KPI formulas](docs/methodology.md)
- [Model card and responsible-use limits](docs/model_card.md)
- [Research paper](reports/research_paper.md)
- [Exploratory data analysis report](reports/eda_report.md)
- [Executive summary for public-sector stakeholders](reports/executive_summary.md)
- [Notebook workflow](notebooks/README.md)
- [Contribution guide](CONTRIBUTING.md)
- [Security and data-reporting policy](SECURITY.md)

## Reproducibility and automation

- Segment thresholds and model settings are centralized in `config.py`.
- The public dashboard data can be rebuilt deterministically with
  `scripts/build_dashboard_dataset.py`.
- The split is stratified and controlled by `random_state=42`.
- Preprocessing and classifiers use scikit-learn `Pipeline` objects to reduce leakage risk.
- Tests use synthetic records and verify KPI denominators, boundaries, edge cases, and model output.
- GitHub Actions runs linting, tests, coverage, and compilation on Python 3.11 and 3.12.

## Responsible use

This is an educational decision-support project, not a production banking decision system.
Do not use its scores to deny services or make adverse decisions. Before operational use, obtain
data-owner approval and complete privacy, fairness, calibration, drift, security, and causal-impact
reviews. Gender and geography analysis must be interpreted as association and audited for fairness.

## License

No open-source license has been selected. Public visibility permits viewing and forking under
GitHub's terms, but it does not automatically grant broader reuse rights. The repository owner
should choose a license explicitly before inviting reuse or external contributions.
