"""Single entry point for reproducible acquisition and analysis."""

import argparse
from pathlib import Path

from commerce_intelligence.analysis import summarize, write_report
from commerce_intelligence.data import build_orders, download_data, save_json
from commerce_intelligence.modeling import benchmark


def run_project(root: Path) -> dict:
    """Rebuild every result from source data without network access."""
    from commerce_intelligence.visualization import create_charts

    reports = root / "reports"
    (reports / "tables").mkdir(parents=True, exist_ok=True)
    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    frame, audit = build_orders(root / "data" / "raw")
    frame.to_csv(root / "data" / "processed" / "orders.csv", index=False)
    summary = summarize(frame)
    model = benchmark(frame)
    save_json(audit, reports / "data_quality.json")
    save_json(summary["stats"], reports / "summary.json")
    save_json(model["metrics"], reports / "metrics.json")
    summary["states"].to_csv(reports / "tables" / "state_rates.csv")
    summary["reviews"].to_csv(reports / "tables" / "review_rates.csv")
    model["ranking"].to_csv(reports / "tables" / "mutual_information.csv", index=False)
    model["loadings"].to_csv(reports / "tables" / "pca_loadings.csv")
    model["encoded_sample"].to_csv(
        reports / "tables" / "encoded_features_sample.csv", index=False
    )
    interactive = create_charts(frame, summary, model, reports)
    write_report(summary, model["metrics"], reports / "executive_summary.md")
    return {
        "orders": frame,
        "audit": audit,
        "summary": summary,
        "model": model,
        "interactive": interactive,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "run"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "download":
        download_data(root / "data" / "raw")
        print("Original Kaggle tables downloaded.")
    else:
        result = run_project(root)
        print(
            f"Analyzed {len(result['orders']):,} delivered orders. "
            f"See {root / 'reports'}."
        )


if __name__ == "__main__":
    main()
