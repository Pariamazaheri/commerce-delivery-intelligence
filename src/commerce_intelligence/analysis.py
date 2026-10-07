"""Transparent uncertainty summaries and decision-oriented reporting."""

from pathlib import Path

import numpy as np
import pandas as pd

from commerce_intelligence.modeling import NUMERIC


def wilson_interval(successes, count):
    """95% Wilson binomial interval, robust near zero and one."""
    count = np.asarray(count, dtype=float)
    successes = np.asarray(successes, dtype=float)
    if (
        np.any(~np.isfinite(count))
        or np.any(~np.isfinite(successes))
        or np.any(count <= 0)
        or np.any(successes < 0)
        or np.any(successes > count)
    ):
        raise ValueError("Wilson intervals require 0 <= successes <= positive count")
    p = np.divide(successes, count, out=np.zeros_like(count), where=count > 0)
    denominator = 1 + 1.96**2 / count
    center = (p + 1.96**2 / (2 * count)) / denominator
    radius = (
        1.96 * np.sqrt(p * (1 - p) / count + 1.96**2 / (4 * count**2)) / denominator
    )
    return center - radius, center + radius


def summarize(frame: pd.DataFrame) -> dict:
    """Order-level denominators; reviewed orders only for satisfaction estimates."""
    states = frame.groupby("customer_state").late.agg(["sum", "count", "mean"])
    states["lower"], states["upper"] = wilson_interval(states["sum"], states["count"])
    states = states.loc[states["count"] >= 500].sort_values("mean", ascending=False)
    reviewed = frame.loc[frame.review_score.notna()]
    review = reviewed.groupby("late").bad_review.agg(["sum", "count", "mean"])
    review["lower"], review["upper"] = wilson_interval(review["sum"], review["count"])
    # Bootstrap the risk difference using empirical Bernoulli sufficient statistics.
    rng = np.random.default_rng(42)
    on_time, late = review.loc[0], review.loc[1]
    differences = (
        rng.binomial(int(late["count"]), float(late["mean"]), 10000) / late["count"]
        - rng.binomial(int(on_time["count"]), float(on_time["mean"]), 10000)
        / on_time["count"]
    )
    stats = {
        "orders": len(frame),
        "late_rate": float(frame.late.mean()),
        "median_delivery_days": float(frame.delivery_days.median()),
        "review_coverage": len(reviewed) / len(frame),
        "poor_review_rate_late": float(late["mean"]),
        "poor_review_rate_on_time": float(on_time["mean"]),
        "poor_review_risk_difference": float(late["mean"] - on_time["mean"]),
        "risk_difference_95ci": np.quantile(differences, [0.025, 0.975]).tolist(),
        "start": str(frame.order_purchase_timestamp.min()),
        "end": str(frame.order_purchase_timestamp.max()),
    }
    return {"stats": stats, "states": states, "reviews": review}


def write_report(summary: dict, metrics: dict, path: Path) -> None:
    """Generate measured findings so documentation cannot drift from outputs."""
    s = summary["stats"]
    lower, upper = s["risk_difference_95ci"]
    state = summary["states"].iloc[0]
    state_name = summary["states"].index[0]
    model = metrics["models"]["logistic_all"]
    path.write_text(
        f"""# Delivery reliability: executive findings

## Evidence

- The valid delivered cohort contains **{s["orders"]:,} orders**, purchased from
  {s["start"][:10]} through {s["end"][:10]}. **{s["late_rate"]:.1%}** arrived after
  their promised calendar date. Median delivery time was
  **{s["median_delivery_days"]:.1f} days**.
- Reviews cover **{s["review_coverage"]:.1%}** of the cohort. Among reviewed orders,
  ratings of 1-2 occur in **{s["poor_review_rate_late"]:.1%}** of late deliveries
  versus **{s["poor_review_rate_on_time"]:.1%}** of on-time deliveries.
  The difference is **{s["poor_review_risk_difference"] * 100:.1f} percentage points**
  (95% order-level bootstrap interval: **{lower * 100:.1f}-{upper * 100:.1f} points**).
- Among states with at least 500 orders, **{state_name}** has the highest observed
  late rate, **{state["mean"]:.1%}**, on **{int(state["count"]):,}** orders.
  Inspect the Wilson intervals before prioritizing a small state over a large one.
- A purchase-time logistic benchmark achieves out-of-time ROC AUC
  **{model["roc_auc"]:.3f}**, average precision **{model["average_precision"]:.3f}**,
  and Brier score **{model["brier_score"]:.4f}**. Test late prevalence is
  **{metrics["test_late_rate"]:.1%}**. See metrics.json for the baseline
  and MI comparison.
- PCA retains **{metrics["pca_components_90pct"]}** components to explain at least
  90% of training variance across {len(NUMERIC)} standardized numeric inputs.

## Decisions supported

1. Audit routes in high-volume states with elevated late rates; measure carrier
   capacity and promise accuracy before changing operations.
2. Pilot proactive communication for at-risk deliveries. Evaluate intervention
   effects with a randomized experiment and track dissatisfaction, cost and retention.
3. Monitor monthly rate drift and calibration before using the benchmark as a
   decision service. Set thresholds using intervention costs, not arbitrary accuracy.

## Interpretation boundaries

This is historical observational evidence, not a causal estimate. Review response
and delivery completion select the population; canceled and undelivered orders are
excluded. Order-level intervals assume independence and do not account for repeated
customers or shared carrier shocks. Month-edge cohorts can be incomplete. State is
a routing proxy, not a cause. The dataset is historical Brazilian marketplace data,
so findings should not be generalized to current operations or Iran.

The model is an analytical benchmark, not a deployed prediction service. Seller
assignment and shipping quote are assumed known at checkout; validate this timing
in a real platform. Never feed delivery timestamps, delay, reviews, or identifiers
into purchase-time prediction. Test outcomes do not tune feature selection or PCA.
""",
        encoding="utf-8",
    )
