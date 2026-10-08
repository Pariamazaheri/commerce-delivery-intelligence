# Analytical architecture

| Layer | Responsibility | Evidence |
|---|---|---|
| Source integrity | Schema, keys, invalid values, missingness and source fingerprints | `data.py`, `reports/data_quality.json` |
| Order-level integration | Aggregate items before entity joins; resolve repeated reviews | `data.py`, data quality audit |
| Service diagnostics | Delivery trends, routing economics, rating associations and uncertainty | `analysis.py`, figures and executive summary |
| Purchase-time representation | Basket ratios, routing geography and calendar features | Feature dictionary and encoded sample |
| Temporal evaluation | Observable training labels; training-only transforms and selection | `modeling.py`, model metrics and leakage tests |
| Dimensional diagnostics | Standardized PCA and component coefficients | PCA figures and loadings table |
| Reproducible communication | Executed notebook, static figures and interactive route explorer | `notebooks/`, `reports/` |

The analysis links operational decisions to auditable evidence. Reported rates
describe the retained delivered-order population; they do not estimate causal
intervention effects or financial savings.
