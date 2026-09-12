# Changelog

## 0.4.0 — 2026-09-12

- Removed the visitor-facing workbook upload step so the dashboard opens immediately.
- Added a bundled, compressed dataset generated from the supplied project workbook.
- Replaced original customer IDs and surnames before publication and documented the data boundary.
- Added a reproducible dataset-build command and automated de-identification tests.

## 0.3.0 — 2026-09-12

- Added a reproducible Markdown EDA report generated from the supplied workbook.
- Added a beginner implementation guide and requirement traceability matrix.
- Added tenure, credit, balance, activity, and product filters to the dashboard.
- Added the geographic risk-index table and salary-versus-balance high-value analysis.
- Added a high-value customer CSV download and expanded visualization tests.
- Standardized local test execution across PyCharm, terminal, and CI environments.

## 0.2.0 — 2026-09-11

- Restructured the project into reusable data, analytics, modeling, visualization, and dashboard
  modules.
- Added synthetic automated tests and Python 3.11/3.12 GitHub Actions checks.
- Added architecture, data dictionary, methodology, model card, research paper, executive summary,
  contribution guidance, security guidance, and citation metadata.
- Added professional data, model, notebook, report, and figure directories with safe ignore rules.
- Kept `app.py` as the stable Streamlit Community Cloud entrypoint.

## 0.1.0 — 2026-09-06

- Initial Streamlit dashboard, KPI analysis, segmentation, Logistic Regression baseline, Random
  Forest comparison, smoke test, and public deployment.
