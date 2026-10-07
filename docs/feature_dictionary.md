# Feature dictionary and availability

The analytical unit is one delivered order. IDs join source tables and are excluded
from numerical modeling. Source-local purchase timestamps are transformed rather
than coerced into arbitrary integer dates. All selected model inputs are numerical
after imputation/encoding; raw source tables retain their original meaning.

| Feature | Definition | Unit / handling |
|---|---|---|
| item_value | Sum of item prices | BRL; missing if any item price is invalid/missing |
| freight | Sum of per-item freight | BRL; missing if incomplete or item amounts invalid |
| weight_g | Sum of item product weights | Grams; missing if any item weight is unavailable |
| item_count | Number of item rows | Count; includes repeated products |
| seller_count | Distinct sellers within order | Count |
| interstate_share | Fraction of items whose seller/customer states differ | 0-1; unknown state pairs ignored |
| promised_days | Estimated delivery minus purchase time | Elapsed days; assumed known at checkout |
| freight_ratio | freight / item_value | Missing for unavailable/zero denominator |
| log_value | log(1 + item_value) | Natural logarithm |
| value_per_item | item_value / item_count | BRL per item |
| purchase_hour | Source-local purchase hour | 0-23; discrete for MI |
| weekday | Purchase weekday | Monday=0 to Sunday=6; discrete for MI |
| weekend | weekday >= 5 | Binary |
| month_sin / month_cos | sin/cos(2*pi*month/12) | Cyclical calendar position |
| customer_state | Destination state | Train-fitted one-hot; unknown test levels ignored |
| value_band | (0,50], (50,150], (150,500], (500,infinity) | Fixed BRL bins, one-hot encoded |

Numeric inputs receive training-median imputation and training-mean/standard-deviation
scaling. Category missing values receive the training mode. Missingness is disclosed
in the audit; a future study could test missingness indicators on temporal validation.
MI treats counts/calendar integers and categorical indicators as discrete. MI uses
unscaled original count labels to preserve discrete semantics.

**Retrospective measurements:** `delivery_days`, `delay_days`, `late`, `review_score`
and `bad_review`. They are useful for analysis and targets, but cannot become model
inputs at purchase time. The PCA projection's outcome colors are for evaluation only.

The current feature matrix intentionally includes correlated raw and engineered
monetary inputs. Logistic regularization helps numerical stability; PCA interpretation
must account for repeated representations of basket value. Training labels must have
matured before test purchases. Seller/product attributes are assumed static in this
public extract; a live system would require historical as-of feature snapshots.

The maturity filter removes 2,008 unresolved training outcomes in this snapshot.
It prevents future-label access, but selects orders that completed by the cutoff
and can underrepresent slow deliveries. Rolling validation with a fixed observation
window should assess that selection effect before operational use.

`pca_loadings.csv` stores `components_.T`: the unit coefficients defining each PCA
direction, also called the rotation matrix. They are not feature-component
correlations or variance-scaled statistical loadings.
