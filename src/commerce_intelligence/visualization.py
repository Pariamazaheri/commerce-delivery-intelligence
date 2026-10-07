"""Publication-friendly charts and a self-contained interactive route explorer."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import PrecisionRecallDisplay

PALETTE = ["#167d8d", "#e36b44", "#344665", "#ac8bc2"]


def create_charts(frame: pd.DataFrame, summary: dict, model: dict, reports: Path):
    """Every plotted cohort and denominator is explicit in titles or captions."""
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.05)
    plt.rcParams.update(
        {"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False}
    )

    def canvas():
        return plt.subplots(figsize=(10, 5.5), layout="constrained")

    def save(fig, name):
        fig.savefig(figures / f"{name}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    fig, ax = canvas()
    counts = frame.late.value_counts().reindex([0, 1])
    wedges, _, _ = ax.pie(
        counts,
        autopct="%.1f%%",
        colors=PALETTE[:2],
        startangle=90,
        wedgeprops={"width": 0.45},
    )
    ax.legend(wedges, ["On time", "Late"], title="Delivery outcome", loc="upper right")
    ax.set_title(f"Delivery promise performance | {len(frame):,} delivered orders")
    save(fig, "01_delivery_share")

    fig, ax = canvas()
    chart = frame.assign(outcome=frame.late.map({0: "On time", 1: "Late"}))
    sns.boxplot(
        data=chart,
        x="outcome",
        y="delivery_days",
        hue="outcome",
        order=["On time", "Late"],
        hue_order=["On time", "Late"],
        legend=False,
        showfliers=False,
        ax=ax,
    )
    ax.set(
        title="Delivery time distribution | outliers hidden, rows retained",
        xlabel="Promise outcome",
        ylabel="Purchase to delivery (days)",
        ylim=(0, frame.delivery_days.quantile(0.995)),
    )
    save(fig, "02_delivery_box")

    monthly = frame.groupby(frame.order_purchase_timestamp.dt.to_period("M")).agg(
        orders=("order_id", "size"),
        late_rate=("late", "mean"),
        delivery_days=("delivery_days", "median"),
        promised_days=("promised_days", "median"),
    )
    monthly.index = monthly.index.to_timestamp()
    fig, ax = canvas()
    ax.plot(monthly.index, monthly.orders, marker="o", label="Delivered orders")
    ax.set(
        title="Purchase cohorts over time | edge months may be incomplete",
        xlabel="Purchase month",
        ylabel="Delivered order count",
        ylim=(0, None),
    )
    ax.legend()
    save(fig, "03_monthly_volume")

    fig, ax = canvas()
    ax.plot(monthly.index, monthly.delivery_days, marker="o", label="Actual median")
    ax.plot(monthly.index, monthly.promised_days, marker="o", label="Promised median")
    ax.set(
        title="Delivery time versus promise | monthly purchase cohorts",
        xlabel="Purchase month",
        ylabel="Days from purchase",
        ylim=(0, None),
    )
    ax.legend(title="Timeline")
    save(fig, "04_delivery_trends")

    states = summary["states"]
    top_volume = frame.customer_state.value_counts().head(10)
    fig, ax = canvas()
    top_volume.plot.bar(ax=ax, color=PALETTE[0], label="Delivered orders")
    ax.set(
        title="Top 10 destination states by delivered order volume",
        xlabel="Customer state",
        ylabel="Orders",
        ylim=(0, None),
    )
    ax.tick_params(axis="x", rotation=0)
    ax.legend()
    save(fig, "05_state_volume")

    reviewed = frame.loc[frame.review_score.notna()]
    table = pd.crosstab(
        reviewed.late.map({0: "On time", 1: "Late"}),
        reviewed.review_score.map(lambda x: f"Rating {int(x)}"),
    )
    fig, ax = canvas()
    table.plot.bar(ax=ax, color=sns.color_palette("viridis", 5))
    ax.set(
        title="Rating counts by delivery outcome | reviewed orders only",
        xlabel="Promise outcome",
        ylabel="Reviewed orders",
        ylim=(0, None),
    )
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Review score")
    save(fig, "06_grouped_reviews")

    fig, ax = canvas()
    proportions = table.div(table.sum(axis=1), axis=0) * 100
    proportions.plot.bar(stacked=True, ax=ax, color=sns.color_palette("viridis", 5))
    ax.set(
        title="Rating mix by delivery outcome | within-group percentages",
        xlabel="Promise outcome",
        ylabel="Reviewed orders (%)",
        ylim=(0, 100),
    )
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Review score", bbox_to_anchor=(1.02, 1), loc="upper left")
    save(fig, "07_stacked_reviews")

    fig, ax = canvas()
    sample = frame.sample(min(2500, len(frame)), random_state=42)
    ax.scatter(
        sample.promised_days,
        sample.delivery_days,
        s=14,
        alpha=0.3,
        label="Random sample of orders",
        color=PALETTE[0],
    )
    limit = float(
        max(frame.promised_days.quantile(0.995), frame.delivery_days.quantile(0.995))
    )
    ax.plot(
        [0, limit], [0, limit], linestyle="--", color=PALETTE[1], label="Equal duration"
    )
    ax.set(
        title="Promise duration and realized delivery | seeded sample",
        xlabel="Promised duration (days)",
        ylabel="Actual duration (days)",
        xlim=(0, limit),
        ylim=(0, limit),
    )
    ax.legend()
    save(fig, "08_promise_scatter")

    routes = frame.groupby("customer_state").agg(
        orders=("order_id", "size"),
        late_rate=("late", "mean"),
        median_freight=("freight", "median"),
        median_delivery=("delivery_days", "median"),
    )
    routes = routes.loc[routes.orders >= 500].reset_index()
    fig, ax = canvas()
    points = ax.scatter(
        routes.median_freight,
        routes.late_rate * 100,
        s=routes.orders / 35,
        c=routes.median_delivery,
        cmap="viridis",
        alpha=0.75,
        edgecolors="white",
    )
    offsets = {
        "SP": (-18, -15),
        "MG": (10, -12),
        "PR": (-20, -8),
        "RS": (8, -10),
        "DF": (-22, 4),
        "GO": (-18, 10),
        "SC": (8, 8),
        "MS": (-22, 4),
        "MT": (6, -6),
    }
    for row in routes.itertuples():
        ax.annotate(
            row.customer_state,
            (row.median_freight, row.late_rate * 100),
            xytext=offsets.get(row.customer_state, (4, 4)),
            textcoords="offset points",
            fontsize=9,
        )
    plt.colorbar(points, ax=ax, label="Median delivery duration (days)")
    handles, labels = points.legend_elements(prop="sizes", num=3, func=lambda x: x * 35)
    ax.legend(handles, labels, title="Order count", loc="upper left")
    ax.set(
        title="State service economics | destinations with ≥500 orders",
        xlabel="Median order freight (BRL)",
        ylabel="Late deliveries (%)",
        xlim=(0, None),
        ylim=(0, max(routes.late_rate * 100) * 1.2),
    )
    save(fig, "09_state_bubbles")

    fig, ax = canvas()
    error = (
        np.vstack([states["mean"] - states.lower, states.upper - states["mean"]]) * 100
    )
    ax.errorbar(
        states.index,
        states["mean"] * 100,
        yerr=error,
        fmt="o",
        capsize=4,
        color=PALETTE[0],
        label="95% Wilson interval",
    )
    ax.axhline(
        frame.late.mean() * 100,
        linestyle="--",
        color=PALETTE[1],
        label="All delivered orders",
    )
    ax.set(
        title="Late delivery rates with uncertainty | states with ≥500 orders",
        xlabel="Destination state",
        ylabel="Late deliveries (%)",
        ylim=(0, None),
    )
    ax.legend()
    save(fig, "10_state_uncertainty")

    fig, ax = canvas()
    ranking = model["ranking"].head(12).iloc[::-1]
    ax.barh(
        ranking.feature.str.replace("num__", "").str.replace("cat__", ""),
        ranking.mutual_information,
        color=PALETTE[0],
        label="Training MI",
    )
    ax.set(
        title="Nonlinear association with delivery risk | train sample only",
        xlabel="Mutual information (nats)",
        ylabel="Encoded feature",
        xlim=(0, None),
    )
    ax.legend()
    save(fig, "11_mutual_information")

    fig, ax = canvas()
    variance = model["metrics"]["pca_explained_variance"]
    ax.plot(
        range(1, len(variance) + 1),
        np.cumsum(variance) * 100,
        marker="o",
        label="Cumulative training variance",
    )
    ax.axhline(90, linestyle="--", color=PALETTE[1], label="90% target")
    ax.set(
        title="PCA compression of standardized numeric features",
        xlabel="Components retained",
        ylabel="Explained variance (%)",
        ylim=(0, 100),
    )
    ax.set_xticks(range(1, len(variance) + 1))
    ax.legend()
    save(fig, "12_pca_variance")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    y = model["test"].late
    reliability_limit = 0.1
    for label, prediction in model["predictions"].items():
        PrecisionRecallDisplay.from_predictions(y, prediction, name=label, ax=axes[0])
        observed, predicted = calibration_curve(
            y, prediction, n_bins=8, strategy="quantile"
        )
        axes[1].plot(predicted, observed, marker="o", label=label)
        reliability_limit = max(reliability_limit, max(predicted), max(observed))
    axes[0].axhline(y.mean(), linestyle="--", color="gray", label="Test prevalence")
    axes[0].set(title="Out-of-time precision-recall", xlim=(0, 1), ylim=(0, 1))
    axes[0].legend(fontsize=8)
    axes[1].plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    axes[1].set(
        title="Out-of-time reliability",
        xlabel="Mean predicted late probability",
        ylabel="Observed late share",
        xlim=(0, min(1, reliability_limit * 1.2)),
        ylim=(0, min(1, reliability_limit * 1.2)),
    )
    axes[1].legend(fontsize=8)
    save(fig, "13_model_validation")

    fig, ax = canvas()
    scores = model["scores"]
    points = ax.scatter(
        scores[:, 0],
        scores[:, 1],
        c=y.to_numpy(),
        cmap="coolwarm",
        alpha=0.25,
        s=7,
        label="Later-period orders",
    )
    colorbar = plt.colorbar(points, ax=ax, ticks=[0, 1])
    colorbar.ax.set_yticklabels(["On time", "Late"])
    colorbar.set_label("Actual delivery outcome")
    ax.set(
        title="PCA projection | later-period orders, color indicates lateness",
        xlabel="Principal component 1",
        ylabel="Principal component 2",
    )
    ax.legend()
    save(fig, "14_pca_projection")

    interactive = px.scatter(
        routes,
        x="median_freight",
        y="late_rate",
        size="orders",
        color="median_delivery",
        hover_name="customer_state",
        hover_data={"orders": ":,", "late_rate": ":.1%", "median_freight": ":.2f"},
        labels={
            "median_freight": "Median order freight (BRL)",
            "late_rate": "Late delivery share",
            "median_delivery": "Delivery days",
            "orders": "Delivered orders",
        },
        title="Delivery reliability explorer | states with at least 500 orders",
        template="plotly_white",
        size_max=65,
    )
    interactive.update_xaxes(range=[0, routes.median_freight.max() * 1.15])
    interactive.update_yaxes(tickformat=".0%", range=[0, routes.late_rate.max() * 1.2])
    folder = reports / "interactive"
    folder.mkdir(parents=True, exist_ok=True)
    interactive.write_html(folder / "route_explorer.html", include_plotlyjs=True)
    monthly.to_csv(reports / "tables" / "monthly_cohorts.csv")
    return interactive
