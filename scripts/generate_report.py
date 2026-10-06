"""Generate report artifacts from a transaction file."""

from __future__ import annotations

import argparse
from pathlib import Path

from ecommerce_clustering.report import generate_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV or Parquet transaction file")
    parser.add_argument("--output", type=Path, default=Path("reports/results"))
    args = parser.parse_args()
    metrics = generate_report(args.input, args.output)
    print(f"Wrote report artifacts to {args.output}")
    print(f"Silhouette: {metrics['silhouette']:.4f}")
    print(f"Davies-Bouldin: {metrics['davies_bouldin']:.4f}")
    if metrics["hybrid_silhouette"] is not None:
        print(f"Hybrid silhouette: {metrics['hybrid_silhouette']:.4f}")
        print(f"Hybrid DBI: {metrics['hybrid_davies_bouldin']:.4f}")
        print(f"Hybrid outliers: {int(metrics['hybrid_outliers'])}")
    if metrics.get("selected_silhouette") is not None:
        print(f"Selected retention-first silhouette: {metrics['selected_silhouette']:.4f}")
        print(f"Selected retention: {metrics['selected_retention_pct']:.1f}%")


if __name__ == "__main__":
    main()
