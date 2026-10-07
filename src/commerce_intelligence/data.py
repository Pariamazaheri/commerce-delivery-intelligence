"""Download original data and construct a validated, one-row-per-order table."""

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SOURCE = "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce"
DOWNLOAD = "https://www.kaggle.com/api/v1/datasets/download/olistbr/brazilian-ecommerce"
TABLES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
}
REQUIRED_COLUMNS = {
    "orders": {
        "order_id",
        "customer_id",
        "order_status",
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    },
    "items": {
        "order_id",
        "order_item_id",
        "seller_id",
        "product_id",
        "price",
        "freight_value",
    },
    "customers": {"customer_id", "customer_state"},
    "sellers": {"seller_id", "seller_state"},
    "products": {"product_id", "product_weight_g"},
    "reviews": {"order_id", "review_id", "review_answer_timestamp", "review_score"},
}


def extract_archive(archive: Path, raw: Path) -> None:
    """Extract only expected flat CSV files; reject corrupt or unexpected archives."""
    with zipfile.ZipFile(archive) as bundle:
        missing = set(TABLES.values()) - set(bundle.namelist())
        if missing:
            raise ValueError(f"Archive is missing required files: {sorted(missing)}")
        for name in TABLES.values():
            destination = raw / name
            destination.write_bytes(bundle.read(name))


def download_data(raw: Path) -> None:
    """Public Kaggle endpoint; never read or store account credentials."""
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "olist.zip"
    temporary = archive.with_suffix(".part")
    # Bounded retries recover transient partial transfers; each attempt restarts.
    for attempt in range(3):
        try:
            with requests.get(DOWNLOAD, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        handle.write(chunk)
            break
        except requests.RequestException:
            if attempt == 2:
                raise
    if not zipfile.is_zipfile(temporary):
        raise ValueError(
            "Kaggle returned no ZIP; download manually from the data card."
        )
    temporary.replace(archive)
    extract_archive(archive, raw)


def build_orders(raw: Path) -> tuple[pd.DataFrame, dict]:
    """Audit source tables and aggregate children before validated joins."""
    tables = {}
    audit = {"source": SOURCE, "tables": {}}
    for key, filename in TABLES.items():
        path = raw / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run commerce-insights download.")
        frame = pd.read_csv(path)
        missing_columns = REQUIRED_COLUMNS[key] - set(frame.columns)
        if missing_columns:
            raise ValueError(f"{filename}: missing columns {sorted(missing_columns)}")
        audit["tables"][key] = {
            "rows": len(frame),
            "exact_duplicates": int(frame.duplicated().sum()),
            "missing": frame.isna().sum().to_dict(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        tables[key] = frame.drop_duplicates().copy()

    orders = tables["orders"]
    for key, column in [
        ("orders", "order_id"),
        ("customers", "customer_id"),
        ("sellers", "seller_id"),
        ("products", "product_id"),
    ]:
        if tables[key][column].isna().any() or tables[key][column].duplicated().any():
            raise ValueError(f"Invalid primary key {key}.{column}")
    item_keys = tables["items"][["order_id", "order_item_id"]]
    if item_keys.isna().any().any() or item_keys.duplicated().any():
        raise ValueError("Conflicting item primary keys")
    for column in [c for c in orders if c.endswith("timestamp") or c.endswith("date")]:
        orders[column] = pd.to_datetime(orders[column], errors="coerce", format="mixed")
    items = tables["items"]
    audit["numeric_parse_failures"] = {}
    for column in ["price", "freight_value"]:
        converted = pd.to_numeric(items[column], errors="coerce")
        audit["numeric_parse_failures"][column] = int(
            (items[column].notna() & ~np.isfinite(converted)).sum()
        )
        items[column] = converted
        items.loc[~np.isfinite(items[column]), column] = np.nan
    invalid = (items["price"] <= 0) | (items["freight_value"] < 0)
    audit["invalid_price_items"] = int(invalid.sum())
    items.loc[invalid, ["price", "freight_value"]] = np.nan
    products = tables["products"]
    converted_weight = pd.to_numeric(products["product_weight_g"], errors="coerce")
    audit["numeric_parse_failures"]["product_weight_g"] = int(
        (products.product_weight_g.notna() & ~np.isfinite(converted_weight)).sum()
    )
    products["product_weight_g"] = converted_weight
    products.loc[~np.isfinite(products["product_weight_g"]), "product_weight_g"] = (
        np.nan
    )
    products.loc[products["product_weight_g"] <= 0, "product_weight_g"] = np.nan
    items = items.merge(
        products[["product_id", "product_weight_g"]],
        on="product_id",
        how="left",
        validate="many_to_one",
    )
    items = items.merge(
        tables["sellers"][["seller_id", "seller_state"]],
        on="seller_id",
        how="left",
        validate="many_to_one",
    )
    order_customer = orders[["order_id", "customer_id"]].merge(
        tables["customers"][["customer_id", "customer_state"]],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )
    items = items.merge(
        order_customer[["order_id", "customer_state"]],
        on="order_id",
        how="left",
        validate="many_to_one",
    )
    items["interstate"] = (items.seller_state != items.customer_state).astype(float)
    items.loc[items.seller_state.isna() | items.customer_state.isna(), "interstate"] = (
        np.nan
    )
    grouped = items.groupby("order_id")
    totals = grouped.agg(
        item_count=("order_item_id", "size"),
        seller_count=("seller_id", "nunique"),
        interstate_share=("interstate", "mean"),
    )
    # min_count preserves all-missing amounts rather than turning them into zero.
    for source, target in [
        ("price", "item_value"),
        ("freight_value", "freight"),
        ("product_weight_g", "weight_g"),
    ]:
        totals[target] = grouped[source].sum(min_count=1)
        incomplete = grouped[source].apply(lambda values: values.isna().any())
        totals.loc[incomplete, target] = np.nan
    totals["amount_incomplete"] = grouped["price"].apply(lambda s: s.isna().any())
    totals.loc[totals.amount_incomplete, ["item_value", "freight"]] = np.nan
    orders = orders.merge(
        tables["customers"][["customer_id", "customer_state"]],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )
    orders = orders.merge(totals, on="order_id", how="left", validate="one_to_one")

    reviews = tables["reviews"].copy()
    reviews["review_answer_timestamp"] = pd.to_datetime(
        reviews.review_answer_timestamp, errors="coerce", format="mixed"
    )
    audit["extra_review_rows"] = int(reviews.duplicated("order_id").sum())
    # Latest answered review wins; review_id breaks equal-time ties deterministically.
    reviews = reviews.sort_values(
        ["review_answer_timestamp", "review_id"], na_position="first"
    ).drop_duplicates("order_id", keep="last")
    reviews["review_score"] = pd.to_numeric(reviews.review_score, errors="coerce")
    audit["invalid_review_scores"] = int(
        (
            reviews.review_score.notna() & ~reviews.review_score.isin([1, 2, 3, 4, 5])
        ).sum()
    )
    reviews.loc[~reviews.review_score.isin([1, 2, 3, 4, 5]), "review_score"] = np.nan
    orders = orders.merge(
        reviews[["order_id", "review_score"]],
        on="order_id",
        how="left",
        validate="one_to_one",
    )
    audit["status_counts"] = orders.order_status.value_counts().to_dict()
    valid = (
        (orders.order_status == "delivered")
        & orders.order_purchase_timestamp.notna()
        & orders.order_delivered_customer_date.notna()
        & orders.order_estimated_delivery_date.notna()
        & (orders.order_delivered_customer_date >= orders.order_purchase_timestamp)
        & (orders.order_estimated_delivery_date >= orders.order_purchase_timestamp)
    )
    audit["excluded_from_delivered_cohort"] = int((~valid).sum())
    frame = orders.loc[valid].copy().sort_values("order_purchase_timestamp")
    frame["delivery_days"] = (
        frame.order_delivered_customer_date - frame.order_purchase_timestamp
    ).dt.total_seconds() / 86400
    frame["promised_days"] = (
        frame.order_estimated_delivery_date - frame.order_purchase_timestamp
    ).dt.total_seconds() / 86400
    # Estimate is a calendar date; delivery on that date counts as on time.
    frame["delay_days"] = (
        frame.order_delivered_customer_date.dt.normalize()
        - frame.order_estimated_delivery_date.dt.normalize()
    ).dt.days
    frame["late"] = frame.delay_days.gt(0).astype(int)
    frame["bad_review"] = frame.review_score.le(2).where(frame.review_score.notna())
    frame["freight_ratio"] = frame.freight / frame.item_value.replace(0, np.nan)
    frame["log_value"] = np.log1p(frame.item_value)
    frame["value_per_item"] = frame.item_value / frame.item_count
    frame["purchase_hour"] = frame.order_purchase_timestamp.dt.hour
    frame["weekday"] = frame.order_purchase_timestamp.dt.dayofweek
    frame["month_sin"] = np.sin(
        2 * np.pi * frame.order_purchase_timestamp.dt.month / 12
    )
    frame["month_cos"] = np.cos(
        2 * np.pi * frame.order_purchase_timestamp.dt.month / 12
    )
    frame["weekend"] = frame.weekday.ge(5).astype(int)
    frame["value_band"] = pd.cut(
        frame.item_value,
        [0, 50, 150, 500, np.inf],
        labels=["budget", "standard", "premium", "high_value"],
    )
    audit["cohort_rows"] = len(frame)
    audit["cohort_missing"] = frame.isna().sum().to_dict()
    if frame.empty:
        raise ValueError("No valid delivered orders")
    return frame.reset_index(drop=True), audit


def save_json(value: dict, path: Path) -> None:
    """Serialize small reproducibility artifacts with explicit native types."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
