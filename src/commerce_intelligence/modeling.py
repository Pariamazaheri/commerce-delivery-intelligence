"""Train-only feature selection and PCA with an out-of-time risk benchmark."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = [
    "item_value",
    "freight",
    "item_count",
    "seller_count",
    "weight_g",
    "interstate_share",
    "promised_days",
    "freight_ratio",
    "log_value",
    "value_per_item",
    "purchase_hour",
    "weekday",
    "month_sin",
    "month_cos",
    "weekend",
]
CATEGORICAL = ["customer_state", "value_band"]
FEATURES = NUMERIC + CATEGORICAL
DISCRETE = ["item_count", "seller_count", "purchase_hour", "weekday", "weekend"]


def chronological_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split entire calendar dates to avoid same-day boundary contamination."""
    ordered = frame.sort_values("order_purchase_timestamp")
    cutoff = ordered.order_purchase_timestamp.iloc[int(len(ordered) * 0.8)].normalize()
    # Training labels must have matured before the first test purchase day.
    train = ordered.loc[
        (ordered.order_purchase_timestamp < cutoff)
        & (ordered.order_delivered_customer_date < cutoff)
    ]
    test = ordered.loc[ordered.order_purchase_timestamp >= cutoff]
    if train.empty or test.empty or min(train.late.nunique(), test.late.nunique()) < 2:
        raise ValueError("Insufficient temporal/class diversity for evaluation")
    return train, test


def benchmark(frame: pd.DataFrame) -> dict:
    """Select numerical encodings on training data, then evaluate once on later data."""
    train, test = chronological_split(frame)
    categorical = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    )
    numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    transformer = ColumnTransformer(
        [
            ("num", numeric, NUMERIC),
            ("cat", categorical, CATEGORICAL),
        ]
    )
    x_train = transformer.fit_transform(train[FEATURES])
    x_test = transformer.transform(test[FEATURES])
    names = transformer.get_feature_names_out()
    # Count/calendar fields and one-hot indicators are discrete for MI.
    discrete = np.array(
        [name.startswith("cat__") or name[5:] in DISCRETE for name in names]
    )
    rng = np.random.default_rng(42)
    sample = rng.choice(len(train), min(15000, len(train)), replace=False)
    mi_input = x_train[sample].copy()
    for index, name in enumerate(names):
        if name.startswith("num__") and name[5:] in DISCRETE:
            mi_input[:, index] = train.iloc[sample][name[5:]].fillna(
                train[name[5:]].median()
            )
    mi = mutual_info_classif(
        mi_input,
        train.late.to_numpy()[sample],
        discrete_features=discrete,
        random_state=42,
    )
    ranking = pd.DataFrame({"feature": names, "mutual_information": mi})
    ranking = ranking.sort_values("mutual_information", ascending=False)
    selected = np.argsort(-mi, kind="stable")[:12]
    results = {}
    predictions = {}
    for label, estimator, cols in [
        (
            "prevalence_baseline",
            DummyClassifier(strategy="prior"),
            np.arange(len(names)),
        ),
        (
            "logistic_all",
            LogisticRegression(max_iter=1500, random_state=42),
            np.arange(len(names)),
        ),
        ("logistic_mi12", LogisticRegression(max_iter=1500, random_state=42), selected),
    ]:
        estimator.fit(x_train[:, cols], train.late)
        probability = estimator.predict_proba(x_test[:, cols])[:, 1]
        predictions[label] = probability
        results[label] = {
            "roc_auc": float(roc_auc_score(test.late, probability)),
            "average_precision": float(average_precision_score(test.late, probability)),
            "brier_score": float(brier_score_loss(test.late, probability)),
        }
    # PCA is for continuous standardized order geometry, not target prediction.
    numeric_train = transformer.named_transformers_["num"].transform(train[NUMERIC])
    numeric_test = transformer.named_transformers_["num"].transform(test[NUMERIC])
    pca = PCA(n_components=0.9, svd_solver="full")
    pca.fit(numeric_train)
    scores = pca.transform(numeric_test)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=NUMERIC,
        columns=[f"PC{i + 1}" for i in range(pca.n_components_)],
    )
    metrics = {
        "train_rows": len(train),
        "test_rows": len(test),
        "purged_unobserved_labels": len(frame) - len(train) - len(test),
        "train_end": str(train.order_purchase_timestamp.max()),
        "test_start": str(test.order_purchase_timestamp.min()),
        "train_late_rate": float(train.late.mean()),
        "test_late_rate": float(test.late.mean()),
        "models": results,
        "pca_components_90pct": int(pca.n_components_),
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
        "selected_features": names[selected].tolist(),
        "encoded_feature_count": len(names),
        "seed": 42,
    }
    return {
        "metrics": metrics,
        "ranking": ranking,
        "loadings": loadings,
        "scores": scores,
        "test": test,
        "predictions": predictions,
        "encoded_sample": pd.DataFrame(x_train[:8], columns=names),
    }
