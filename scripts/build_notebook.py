"""Build the narrative notebook; execution is a separate validation step."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def markdown(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


markdown("""
# Commerce Delivery Intelligence
### Delivery reliability, customer experience and purchase-time risk

**Decision:** Where should a marketplace investigate delivery reliability, and
what information available at checkout helps identify future delays?

This analysis uses Olist's original public Kaggle data. It connects order,
customer, item, seller, product and review tables at the **order** grain.
The executed results, charts and feature diagnostics below are reproducible
from the modular Python package. Monetary amounts are nominal Brazilian reais
(BRL). Dates are source-local timestamps; no unsupported timezone is imposed.

Read the executive findings first, then follow the evidence and limitations.
Dataset: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
""")
code("""
from pathlib import Path
import sys

# Works from the repository root or its notebooks/ directory.
ROOT = Path.cwd().resolve()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
if not (ROOT / "src").exists():
    raise RuntimeError("Open this notebook inside the cloned repository.")
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from IPython.display import Image, Markdown, display
from commerce_intelligence.cli import run_project
from commerce_intelligence.modeling import FEATURES, NUMERIC, chronological_split

# Download separately with `commerce-insights download` if raw files are absent.
# This analytical run is offline and does not scrape live listings.
result = run_project(ROOT)
orders = result["orders"]
audit = result["audit"]
model = result["model"]
display(Markdown((ROOT / "reports/executive_summary.md").read_text()))
""")
markdown("""
## 1. Source quality and analysis population

Olist provides real multi-table ecommerce transactions with operational dates,
amounts and customer ratings. This supports routing economics and reliability
analysis more directly than a flat synthetic sales table. No Iranian Kaggle
dataset was selected: the core question benefits from Olist's documented event
sequence and relational structure. The independent Samand extension adds a
local market acquisition example without mixing unrelated data into the model.

We remove exact duplicate rows and fail on conflicting entity/item primary keys.
Missing or invalid dates are parsed to missing values. Nonpositive item prices,
negative freight and nonpositive product weights become missing. Monetary totals
with invalid items stay missing. Reviews can repeat per order: the latest
answered review wins, with review ID as a deterministic tie-breaker.

Item counts and seller counts are aggregated **before** one-to-one order joins.
Customers, sellers and products use validated many-to-one joins. The cohort
requires delivered status and plausible purchase, delivery and estimated dates.
Missing reviews remain unavailable and do not become good ratings. Numerical
features use train-only median imputation; categories use the training mode.
No outliers are removed merely because they are large.
""")
code("""
quality = pd.DataFrame({
    name: {"source_rows": value["rows"],
           "exact_duplicates": value["exact_duplicates"],
           "missing_cells": sum(value["missing"].values())}
    for name, value in audit["tables"].items()
}).T
display(quality)
display(pd.Series(audit["status_counts"], name="source_order_status"))
display(pd.Series({
    "excluded_orders": audit["excluded_from_delivered_cohort"],
    "retained_orders": audit["cohort_rows"],
    "invalid_price_items": audit["invalid_price_items"],
    "extra_review_rows": audit["extra_review_rows"],
}, name="cleaning_audit"))
display(orders[FEATURES + ["review_score"]].isna().mean()
        .sort_values(ascending=False).rename("missing_share").to_frame())
display(orders[NUMERIC + ["delivery_days", "delay_days"]]
        .describe(percentiles=[.01, .5, .95, .99]).T.round(2))
assert orders.order_id.is_unique
""")
markdown("""
## 2. Delivery patterns and customer experience

**Definition:** an order is late if its actual delivery calendar date is after
the estimated delivery date. A delivery at 23:00 on the promised date is on time.
`delivery_days` uses elapsed time; `delay_days` uses calendar dates.

The chart sequence separates frequency, distribution and time. The donut pie
chart gives the outcome share. The box plot hides fliers to reveal central spread
but retains every valid row in estimates. Volume and multi-line duration charts
use purchase cohorts, avoiding the misleading comparison of sales and delivery
months. First and last months can be incomplete.
""")
code("""
def show_chart(name):
    display(Image(filename=str(ROOT / "reports/figures" / name)))

for name in ["01_delivery_share.png", "02_delivery_box.png",
             "03_monthly_volume.png", "04_delivery_trends.png"]:
    show_chart(name)
""")
markdown("""
The destination bar chart establishes business scale. Grouped rating counts show
volume; stacked proportions make rating mixes comparable despite unequal cohort
sizes. Reviews are selected responses and can precede delivery, so the observed
rating gap does not establish that lateness caused dissatisfaction.
""")
code("""
for name in ["05_state_volume.png", "06_grouped_reviews.png",
             "07_stacked_reviews.png"]:
    show_chart(name)
display(result["summary"]["reviews"].rename(index={0: "On time", 1: "Late"}))
""")
markdown("""
## 3. Routing economics and uncertainty

The seeded scatter sample relates promise duration to realized delivery. Visible
axes stop at the full cohort's 99.5th percentile; extreme values remain in the
statistics. The equality line is a duration reference, while the official target
uses calendar dates. State bubbles combine freight, late rate, order volume and
median delivery time. Bubble **area** scales with order count.

State error bars are 95% Wilson intervals, restricted to destinations with at
least 500 delivered orders. Ranking states is descriptive, without correction
for multiple comparisons. The review risk difference uses a seeded 10,000-draw
order-level bootstrap (empirical Bernoulli resampling). Repeated customers and
common carrier shocks can make these intervals too narrow. A production study
should use customer/carrier clusters and check review nonresponse.
""")
code("""
for name in ["08_promise_scatter.png", "09_state_bubbles.png",
             "10_state_uncertainty.png"]:
    show_chart(name)
display(result["summary"]["states"].round(4))
""")
markdown("""
### Interactive route explorer

Hover to inspect a state's volume and economics, zoom into overlapping points,
and compare destination service levels. A self-contained offline HTML export is
saved under `reports/interactive/route_explorer.html`.
""")
code("""
# GitHub may not render Plotly outputs; the preceding PNGs remain readable there.
display(result["interactive"])
""")
markdown("""
## 4. Feature engineering and numerical preprocessing

| Family | Features | Analytical purpose |
|---|---|---|
| Ratios | freight/value, value/item | Shipping burden and basket economics |
| Fixed bins | 50, 150, 500 BRL cutoffs | Spending segments |
| Mathematical | log(1 + value) | Compress a heavy monetary tail |
| Combinations | interstate item share | Combine customer and seller routing geography |
| Date/time | hour, weekday, weekend, cyclical month | Calendar context |
| Aggregates | item/seller counts, value, freight, weight sums | Order grain |

Fixed bins are defined before observing the target. No future customer history or
target-encoded seller averages are used. IDs are join keys, and event dates are
converted into meaningful numerical inputs. We exclude raw IDs, free text and
post-purchase outcomes from the design matrix. All **model inputs** become numeric:
one-hot state/value-band encoding, median/mode imputation and numeric z-scores.
PCA operates on standardized numeric inputs so grams do not dominate BRL or counts.
""")
code("""
display(orders[["item_value", "freight_ratio", "value_per_item", "value_band",
                "interstate_share", "item_count", "seller_count", "month_sin"]].head())
display(model["encoded_sample"].round(3))
print("Encoded numerical inputs:", model["metrics"]["encoded_feature_count"])
""")
markdown("""
## 5. Mutual information and temporal generalization

Split at the purchase date around the 80th percentile and hold all later dates
out. Training orders must also have been delivered before the cutoff, so labels
are known at training time. Preprocessing, MI selection and PCA fit **only** on
this training population. The later cohort is evaluated once.

MI captures nonlinear marginal dependence, not causal effects or conditional
incremental usefulness. Use at most 15,000 seeded training rows, mark count and
calendar variables as discrete, and keep one-hot indicators discrete. Select
the top 12 encoded inputs under a fixed rule. Compare that benchmark with a
logistic model using all inputs and a constant training-prevalence baseline.

ROC AUC measures ranking; average precision is informative with rare late orders;
Brier score evaluates probability error (lower is better). Calibration curves
show whether probabilities transport across time. A random split would obscure
the changes in delay prevalence and seasonality visible in the date cohorts.
""")
code("""
train, test = chronological_split(orders)
assert train.order_delivered_customer_date.max() < test.order_purchase_timestamp.min()
display(pd.DataFrame(model["metrics"]["models"]).T.round(4))
display(pd.Series({k: model["metrics"][k] for k in [
    "train_rows", "test_rows", "train_end", "test_start",
    "train_late_rate", "test_late_rate"]}))
display(model["ranking"].head(15).round(5))
show_chart("11_mutual_information.png")
show_chart("13_model_validation.png")
""")
code("""
all_metrics = model["metrics"]["models"]["logistic_all"]
mi_metrics = model["metrics"]["models"]["logistic_mi12"]
delta = mi_metrics["average_precision"] - all_metrics["average_precision"]
display(Markdown(
    f"**Selection diagnostic:** the MI subset changes later-period average "
    f"precision by {delta:+.4f} compared with all features. "
    "Marginal relevance need not transfer across time; correlated features and "
    "seasonal relationships can dominate MI. Keep the observed result and use "
    "rolling validation in future development rather than choosing a better "
    "subset on this test period."
))
""")
markdown("""
## 6. PCA and interpretation

Retain enough principal components to explain at least 90% of training variance.
PCA is unsupervised: it preserves geometry rather than optimizing delay detection.
The held-out projection colors later orders by their actual outcome for diagnosis;
outcomes do not fit the axes. Inspect loadings to see which basket, routing and
calendar measurements drive each direction. Sign is arbitrary, and correlated
engineered versions of value can overweight that concept. This redundancy is
disclosed so the component directions can be interpreted through their coefficients.
""")
code("""
show_chart("12_pca_variance.png")
show_chart("14_pca_projection.png")
display(model["loadings"].round(3))
for component in model["loadings"].columns[:2]:
    print(component, "largest absolute loadings:")
    display(model["loadings"][component].abs().nlargest(5))
""")
markdown("""
## 7. When is feature engineering optional, and when is it essential?

Feature engineering is a nice-to-have when the source already expresses the
decision in useful variables, such as a tidy table of validated numerical
measurements, or a representation-learning model has sufficient data to learn
those patterns. Extra handcrafted inputs still need out-of-time validation.

It is essential when the raw representation cannot express the business unit or
mechanism. Here an order may have several item rows and sellers: order-level
counts and totals are necessary to avoid double counting. Freight/value exposes
shipping burden that two separate currency columns obscure. Cyclical month
encoding expresses calendar adjacency. Transformations also become essential
for scale-sensitive methods such as PCA. Feature timing is as important as feature
construction: a highly predictive review or delivery timestamp would invalidate
a purchase-time model.

## 8. Operational recommendations and next steps

Prioritize high-volume destinations with elevated late rates for route and
promise audits, then pilot communication or capacity changes with randomized
evaluation. Monitor delivery completion, review response, prevalence drift,
calibration and intervention cost. Use rolling temporal validation and carrier
information before choosing a deployment model. No cost savings or causal lift
are claimed from this observational dataset.

The optional Samand acquisition module is independent of this ecommerce analysis.
It collects 50 unique public cars manufactured strictly after Solar Hijri 1385,
records price/mileage/color/year/transmission/description, and retains source URLs
and acquisition timestamps. Negotiable and installment-only prices remain missing.
The snapshot is a convenience sample, not a representative Iranian price index.
See `docs/market_snapshot.md` for execution and coverage.
""")

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
notebook.metadata["language_info"] = {"name": "python", "version": "3.11"}
(ROOT / "notebooks").mkdir(exist_ok=True)
nbf.write(notebook, ROOT / "notebooks/delivery_intelligence.ipynb")
print("Narrative notebook generated.")
