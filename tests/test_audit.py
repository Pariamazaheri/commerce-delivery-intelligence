"""Regression checks for malformed inputs and train-only feature transforms."""

import numpy as np
import pandas as pd
import pytest

from commerce_intelligence.analysis import wilson_interval
from commerce_intelligence.data import TABLES, build_orders
from commerce_intelligence.modeling import DISCRETE, NUMERIC, benchmark


def test_missing_source_columns_have_actionable_error(raw):
    path = raw / TABLES["items"]
    pd.read_csv(path).drop(columns="price").to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns.*price"):
        build_orders(raw)


@pytest.mark.parametrize("invalid", ["not-a-price", "inf"])
def test_nonnumeric_and_infinite_amounts_are_not_model_inputs(raw, invalid):
    path = raw / TABLES["items"]
    items = pd.read_csv(path).astype({"price": "str"})
    items.loc[0, "price"] = invalid
    items.to_csv(path, index=False)
    frame, _ = build_orders(raw)
    assert pd.isna(frame.set_index("order_id").loc["o1", "item_value"])


def test_fractional_review_score_is_unavailable(raw):
    path = raw / TABLES["reviews"]
    reviews = pd.read_csv(path).astype({"review_score": float})
    reviews.loc[1, "review_score"] = 4.5
    reviews.to_csv(path, index=False)
    frame, _ = build_orders(raw)
    assert pd.isna(frame.set_index("order_id").loc["o1", "review_score"])


@pytest.mark.parametrize("successes,count", [(0, 0), (3, 2), (-1, 2)])
def test_invalid_uncertainty_denominators_fail(successes, count):
    with pytest.raises(ValueError, match="positive count"):
        wilson_interval(successes, count)


def test_heldout_feature_changes_do_not_refit_training_transforms():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({column: rng.uniform(1, 10, 150) for column in NUMERIC})
    for column in DISCRETE:
        frame[column] = rng.integers(0, 3, 150)
    dates = pd.date_range("2018-01-01", periods=len(frame))
    frame["order_purchase_timestamp"] = dates
    frame["order_delivered_customer_date"] = dates + pd.Timedelta(days=2)
    frame["late"] = (np.arange(len(frame)) % 5 == 0).astype(int)
    frame["customer_state"] = np.where(np.arange(len(frame)) % 2, "SP", "RJ")
    frame["value_band"] = "standard"
    original = benchmark(frame)
    mutated = frame.copy()
    mutated.loc[120:, "item_value"] += 1000
    mutated.loc[120:, "customer_state"] = "new-state"
    changed = benchmark(mutated)
    pd.testing.assert_frame_equal(original["ranking"], changed["ranking"])
    pd.testing.assert_frame_equal(original["loadings"], changed["loadings"])
    pd.testing.assert_frame_equal(original["encoded_sample"], changed["encoded_sample"])
    assert sum(original["metrics"]["pca_explained_variance"]) >= 0.9
