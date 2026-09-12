.PHONY: install install-dev run test lint quality smoke eda dataset

install:
	python -m pip install -r requirements.txt

install-dev:
	python -m pip install -r requirements-dev.txt

run:
	python -m streamlit run app.py

test:
	python -m pytest

lint:
	python -m ruff check .

quality: lint test

smoke:
	python smoke_test.py "$(WORKBOOK)"

eda:
	python scripts/generate_eda_report.py "$(WORKBOOK)"

dataset:
	python scripts/build_dashboard_dataset.py "$(WORKBOOK)"
