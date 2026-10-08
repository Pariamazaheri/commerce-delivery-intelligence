"""Validate committed analytical evidence without fetching source datasets."""

import json
from pathlib import Path

import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
notebook = nbformat.read(ROOT / "notebooks/delivery_intelligence.ipynb", as_version=4)
nbformat.validate(notebook)
code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
assert all(cell.execution_count is not None for cell in code_cells)
assert not any(
    output.output_type == "error" for cell in code_cells for output in cell.outputs
)
summary = json.loads((ROOT / "reports/summary.json").read_text())
metrics = json.loads((ROOT / "reports/metrics.json").read_text())
audit = json.loads((ROOT / "reports/data_quality.json").read_text())
assert summary["orders"] == audit["cohort_rows"]
assert abs(sum(metrics["pca_explained_variance"])) >= 0.9
monthly = pd.read_csv(ROOT / "reports/tables/monthly_cohorts.csv")
assert monthly.orders.sum() == summary["orders"]
assert (
    abs(
        (monthly.late_rate * monthly.orders).sum() / monthly.orders.sum()
        - summary["late_rate"]
    )
    < 1e-10
)
assert len(list((ROOT / "reports/figures").glob("*.png"))) == 14
print(f"Verified {len(code_cells)} executed cells and 14 figures.")
