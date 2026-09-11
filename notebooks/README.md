# Notebook workflow

Use notebooks for exploration and communication, not as the only location of reusable logic.
Import validation, segmentation, KPI, and modeling functions from `european_bank_churn/`.

Recommended naming convention:

1. `01_data_quality_and_eda.ipynb`
2. `02_segmentation_and_kpis.ipynb`
3. `03_model_comparison.ipynb`

Keep notebook outputs small, restart the kernel before committing, and verify that no customer-level
rows, absolute local paths, credentials, or hidden metadata are published. Promote stable code from
notebooks into the package and cover it with tests.

Example first cell:

```python
from pathlib import Path

from european_bank_churn import load_excel, prepare_data, validate_dataset

workbook = Path("../data/raw/European_Bank.xlsx")
raw = load_excel(workbook)
validate_dataset(raw)
data, thresholds = prepare_data(raw)
```
