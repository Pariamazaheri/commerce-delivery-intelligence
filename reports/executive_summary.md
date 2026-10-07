# Delivery reliability: executive findings

## Evidence

- The valid delivered cohort contains **96,470 orders**, purchased from
  2016-09-15 through 2018-08-29. **6.8%** arrived after
  their promised calendar date. Median delivery time was
  **10.2 days**.
- Reviews cover **99.3%** of the cohort. Among reviewed orders,
  ratings of 1-2 occur in **62.4%** of late deliveries
  versus **9.3%** of on-time deliveries.
  The difference is **53.2 percentage points**
  (95% order-level bootstrap interval: **51.9-54.4 points**).
- Among states with at least 500 orders, **MA** has the highest observed
  late rate, **17.4%**, on **717** orders.
  Inspect the Wilson intervals before prioritizing a small state over a large one.
- A purchase-time logistic benchmark achieves out-of-time ROC AUC
  **0.715**, average precision **0.076**,
  and Brier score **0.0331**. Test late prevalence is
  **3.5%**. See metrics.json for the baseline
  and MI comparison.
- PCA retains **10** components to explain at least
  90% of training variance across 15 standardized numeric inputs.

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
