# Analysis capability map

| Capability | Implementation and evidence |
|---|---|
| EDA, missingness, duplicates, invalid values | `data.py`, notebook sections 1-3, `reports/data_quality.json` |
| Numerical conversion and standardization | Train-fitted ColumnTransformer in `modeling.py`; encoded sample table |
| Pie and box charts | Figures 01-02 |
| Line and multi-line charts | Figures 03-04 |
| Bar, grouped and stacked bars | Figures 05-07 |
| Scatter and bubble charts | Figures 08-09 |
| Uncertainty/error bars | Figure 10, Wilson state intervals, bootstrap review gap |
| Interactive visualization | Plotly explorer in notebook and generated standalone HTML |
| Ratios, bins, mathematics, combinations | freight/value, fixed value bands, log value, interstate share |
| Date/time transformations and aggregation | Purchase calendar features; item/seller counts and totals |
| Mutual information selection | Notebook section 5, figure 11, top-12 temporal comparison |
| PCA reduction and explanation | Notebook section 6, figures 12/14, loadings table |
| Feature engineering reflection | Notebook section 7 |
| Iranian public market scraper | `scraping.py`; validated 50-row CSV/JSON/Excel snapshot |

Every static figure has a meaningful title and relevant labels. Cartesian charts
show scale/range through labeled ticks; pie charts use percentage labels and an
outcome legend because axes are not applicable. Single-series bar/line charts have
legends; categorical boxes use their labeled groups. Full notebook outputs are saved.

The public repository follows a business question from validated source data to
decision-oriented findings. The independent acquisition extension documents local
market-data collection without mixing unrelated car listings into ecommerce models.
