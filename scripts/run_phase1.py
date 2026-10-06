"""Run the Phase 1 baseline against a transaction CSV or Parquet file."""

from __future__ import annotations

import argparse
from pathlib import Path

from ecommerce_clustering.phase1 import build_clustering_features, load_transactions, run_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV or Parquet transaction file")
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("data/processed/rfm_clusters.csv"))
    args = parser.parse_args()

    transactions = load_transactions(args.input)
    features = build_clustering_features(transactions)
    result = run_baseline(features, n_clusters=args.clusters)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.customers.to_csv(args.output, index=False)
    print(f"Saved {len(result.customers)} customer segments to {args.output}")
    for name, value in result.metrics.items():
        print(f"{name}: {value:.6f}")


if __name__ == "__main__":
    main()
