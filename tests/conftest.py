"""Synthetic relational source data shared by offline regression tests."""

import pandas as pd
import pytest

from commerce_intelligence.data import TABLES


@pytest.fixture
def raw(tmp_path):
    tables = {
        "orders": pd.DataFrame(
            {
                "order_id": ["o1", "o2", "o3"],
                "customer_id": ["c1", "c2", "c3"],
                "order_status": ["delivered", "delivered", "canceled"],
                "order_purchase_timestamp": ["2018-01-01 10:00:00"] * 3,
                "order_delivered_customer_date": [
                    "2018-01-04 23:00:00",
                    "2018-01-06",
                    None,
                ],
                "order_estimated_delivery_date": ["2018-01-04"] * 3,
            }
        ),
        "customers": pd.DataFrame(
            {"customer_id": ["c1", "c2", "c3"], "customer_state": ["SP", "RJ", "SP"]}
        ),
        "items": pd.DataFrame(
            {
                "order_id": ["o1", "o1", "o2"],
                "order_item_id": [1, 2, 1],
                "seller_id": ["s1"] * 3,
                "product_id": ["p1"] * 3,
                "price": [50.0, 50.0, -1.0],
                "freight_value": [10.0, 10.0, 5.0],
            }
        ),
        "sellers": pd.DataFrame({"seller_id": ["s1"], "seller_state": ["SP"]}),
        "products": pd.DataFrame({"product_id": ["p1"], "product_weight_g": [100.0]}),
        "reviews": pd.DataFrame(
            {
                "order_id": ["o1", "o1", "o2"],
                "review_id": ["r1", "r2", "r3"],
                "review_answer_timestamp": ["2018-01-05", "2018-01-07", "2018-01-07"],
                "review_score": [1, 5, 2],
            }
        ),
    }
    for key, frame in tables.items():
        frame.to_csv(tmp_path / TABLES[key], index=False)
    return tmp_path
