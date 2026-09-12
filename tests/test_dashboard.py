"""Streamlit integration test for the submission-ready dashboard."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_dashboard_opens_with_bundled_data_and_no_upload_control():
    app = AppTest.from_file(APP_PATH, default_timeout=30).run(timeout=30)

    assert not app.exception
    assert len(app.file_uploader) == 0
    assert app.title[0].value == "Customer Segmentation & Churn Pattern Analytics"
    assert [(metric.label, metric.value) for metric in app.metric[:3]] == [
        ("Customers", "10,000"),
        ("Churners", "2,037"),
        ("Overall churn rate", "20.4%"),
    ]
    assert [tab.label for tab in app.tabs] == [
        "Overview",
        "Segments",
        "Geography & demographics",
        "High-value customers",
        "ML model comparison",
    ]
