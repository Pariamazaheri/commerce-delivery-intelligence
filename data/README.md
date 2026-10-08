# Data acquisition and provenance

Source: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce),
published by Olist on Kaggle. License: CC BY-NC-SA 4.0, verified in Kaggle's public
dataset-list metadata on 7 October 2026.

`commerce-insights download` extracts only the six required CSVs from the original
public archive: orders, items, customers, sellers, products and reviews. Payments,
translations and geolocation are not used in the delivery analysis. The pipeline
records source-table SHA-256 checksums, row counts, duplicate counts and missingness
in `reports/data_quality.json`. Keep those fingerprints to detect upstream changes.

`raw/` is created by the downloader and contains unmodified local input. `processed/orders.csv` is created by the analysis pipeline as analytical
output with engineered features. Both are ignored by Git. Acquisition is networked;
Analysis has no network dependency. Published source code omits raw datasets;
download them locally before execution.

Amounts are historical nominal BRL and timestamps follow the source. The complete
delivered analytical cohort spans September 2016-August 2018.
