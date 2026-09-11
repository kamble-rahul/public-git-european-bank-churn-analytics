.PHONY: install install-dev run test lint quality smoke

install:
	python -m pip install -r requirements.txt

install-dev:
	python -m pip install -r requirements-dev.txt

run:
	streamlit run app.py

test:
	pytest

lint:
	ruff check .

quality: lint test

smoke:
	python smoke_test.py "$(WORKBOOK)"
