"""Small adversarial fixtures verify grain, exclusions and leakage boundaries."""

import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from commerce_intelligence.analysis import wilson_interval
from commerce_intelligence.data import TABLES, build_orders, extract_archive
from commerce_intelligence.modeling import FEATURES, chronological_split
from commerce_intelligence.scraping import parse_detail, parse_listing


def test_order_grain_calendar_boundary_and_missing_amounts(raw):
    orders, audit = build_orders(raw)
    indexed = orders.set_index("order_id")
    assert len(orders) == 2
    assert indexed.loc["o1", "item_value"] == 100
    assert indexed.loc["o1", "item_count"] == 2
    assert indexed.loc["o1", "late"] == 0  # 23:00 on promised day is on time.
    assert indexed.loc["o1", "review_score"] == 5  # Latest review wins.
    assert indexed.loc["o2", "late"] == 1
    assert pd.isna(indexed.loc["o2", "item_value"])
    assert pd.isna(indexed.loc["o2", "freight_ratio"])
    assert audit["excluded_from_delivered_cohort"] == 1


def test_conflicting_entity_keys_fail(raw):
    path = raw / TABLES["customers"]
    frame = pd.read_csv(path)
    frame.loc[len(frame)] = ["c1", "MG"]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="primary key"):
        build_orders(raw)


def test_exact_duplicates_do_not_change_grain(raw):
    path = raw / TABLES["items"]
    frame = pd.read_csv(path)
    pd.concat([frame, frame.iloc[[0]]]).to_csv(path, index=False)
    orders, audit = build_orders(raw)
    assert orders.set_index("order_id").loc["o1", "item_count"] == 2
    assert audit["tables"]["items"]["exact_duplicates"] == 1


def test_future_training_labels_are_purged():
    dates = pd.date_range("2018-01-01", periods=100)
    frame = pd.DataFrame(
        {
            "order_purchase_timestamp": dates,
            "order_delivered_customer_date": dates + pd.Timedelta(days=5),
            "late": np.arange(100) % 2,
        }
    )
    train, test = chronological_split(frame)
    assert (
        train.order_delivered_customer_date.max() < test.order_purchase_timestamp.min()
    )
    assert len(train) == 75
    assert len(test) == 20


def test_feature_allowlist_excludes_outcomes_and_identifiers():
    assert not set(FEATURES) & {
        "order_id",
        "customer_id",
        "late",
        "delay_days",
        "review_score",
        "delivery_days",
        "order_delivered_customer_date",
    }


def test_wilson_handles_extreme_observations():
    lower, upper = wilson_interval([0, 100], [100, 100])
    assert lower[0] == pytest.approx(0)
    assert upper[1] == pytest.approx(1)
    assert upper[0] < 0.05
    assert lower[1] > 0.95


def test_archive_rejects_unexpected_payload(tmp_path):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../escape.csv", "bad")
    with pytest.raises(ValueError, match="missing required"):
        extract_archive(path, tmp_path)
    assert not (tmp_path.parent / "escape.csv").exists()


def test_listing_links_are_deduplicated():
    link = "/car/detail-abc-samand-lx-1390"
    html = f'<a href="{link}">car</a><a href="{link}">same</a>'
    assert parse_listing(html) == [link]


def test_scraper_fails_closed_on_schema_change():
    html = '<script id="__NUXT_DATA__">' + json.dumps([{}]) + "</script>"
    with pytest.raises(ValueError, match="schema changed"):
        parse_detail(html, "https://bama.ir/car/detail-example")


def vehicle_payload(year, price):
    """Construct an explicitly synthetic reference pool for boundary tests."""
    pool = []

    def encode(value):
        encoded = (
            {key: encode(item) for key, item in value.items()}
            if isinstance(value, dict)
            else value
        )
        pool.append(encoded)
        return len(pool) - 1

    encode(
        {
            "vehicle": {
                "brand": {"value": "samand"},
                "year": {"value": year},
                "mileage": {"value": 0},
                "color": {"body": {"value": "سفید"}},
                "transmission": {"value": "دنده ای"},
            },
            "content": {"description": "Synthetic vehicle observation"},
            "price": {"type": "lumpsum", "fixed": {"value": price}},
        }
    )
    return '<script id="__NUXT_DATA__">' + json.dumps(pool) + "</script>"


def test_scraper_year_1385_is_excluded():
    with pytest.raises(ValueError, match="strictly after"):
        parse_detail(vehicle_payload(1385, 100), "https://bama.ir/car/detail-fixture")


def test_scraper_missing_price_and_zero_mileage_are_distinct():
    row = parse_detail(
        vehicle_payload(1386, None), "https://bama.ir/car/detail-fixture"
    )
    assert row["price_toman"] is None
    assert row["mileage_km"] == 0
    assert row["production_year_sh"] == 1386
    assert row["transmission"] == "manual"
