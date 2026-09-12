"""Build the privacy-safe dataset bundled with the public Streamlit dashboard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from european_bank_churn.data import (  # noqa: E402
    DASHBOARD_DATA_PATH,
    load_excel,
    standardize_for_dashboard,
    validate_dataset,
)


def build_dashboard_dataset(source: Path, output: Path = DASHBOARD_DATA_PATH) -> None:
    """Validate, de-identify, and write a deterministic compressed CSV asset."""
    raw = load_excel(source)
    errors = [item for item in validate_dataset(raw) if item["level"] == "error"]
    if errors:
        details = "; ".join(item["message"] for item in errors)
        raise ValueError(f"Source validation failed: {details}")

    standardized = standardize_for_dashboard(raw)
    output.parent.mkdir(parents=True, exist_ok=True)
    standardized.to_csv(
        output,
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )

    reloaded = pd.read_csv(output)
    if len(reloaded) != len(raw):
        raise RuntimeError("Row-count verification failed after writing the dashboard asset.")
    if set(reloaded["Surname"]) != {"Anonymous"}:
        raise RuntimeError("Surname de-identification verification failed.")
    if not reloaded["CustomerId"].astype(str).str.fullmatch(r"DEMO-\d{5}").all():
        raise RuntimeError("Generated CustomerId verification failed.")
    if not reloaded["Exited"].equals(raw["Exited"].reset_index(drop=True)):
        raise RuntimeError("Target values changed while building the dashboard asset.")

    print(f"Wrote {len(reloaded):,} de-identified rows to {output}")
    print(f"Overall churn rate: {reloaded['Exited'].mean():.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the de-identified dataset bundled with the Streamlit dashboard."
    )
    parser.add_argument("source", type=Path, help="Path to the private source .xlsx workbook")
    parser.add_argument(
        "--output",
        type=Path,
        default=DASHBOARD_DATA_PATH,
        help="Destination .csv.gz path",
    )
    args = parser.parse_args()
    build_dashboard_dataset(args.source, args.output)


if __name__ == "__main__":
    main()
