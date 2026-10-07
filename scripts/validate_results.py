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
snapshot = pd.read_csv(ROOT / "reports/market_snapshot/samand_listings.csv")
assert len(snapshot) == 50 and snapshot.source_url.is_unique
assert snapshot.production_year_sh.gt(1385).all()
excel = pd.read_excel(ROOT / "reports/market_snapshot/samand_listings.xlsx").rename(
    columns={
        "Source URL": "source_url",
        "Price (Toman)": "price_toman",
        "Mileage (km)": "mileage_km",
        "Year (SH)": "production_year_sh",
    }
)
assert len(excel) == 50 and excel.source_url.is_unique
for column in ["price_toman", "mileage_km", "production_year_sh"]:
    pd.testing.assert_series_equal(
        snapshot.set_index("source_url")[column].sort_index(),
        excel.set_index("source_url")[column].sort_index(),
        check_dtype=False,
    )
print(f"Verified {len(code_cells)} executed cells, 14 figures and 50 real listings.")
