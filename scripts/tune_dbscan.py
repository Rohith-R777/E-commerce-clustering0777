"""Run DBSCAN parameter sensitivity analysis on a transaction file."""

from __future__ import annotations

import argparse
from pathlib import Path

from ecommerce_clustering.hybrid import select_business_configuration, tune_dbscan
from ecommerce_clustering.phase1 import build_rfm_features, load_transactions, run_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    rfm = build_rfm_features(load_transactions(args.input))
    baseline = run_baseline(rfm)
    values = baseline.scaler.transform(rfm[["recency", "frequency", "monetary"]])
    for row in tune_dbscan(values):
        print(row)
    selected, result = select_business_configuration(values)
    print("selected:", selected)
    print("selected_metrics:", result.metrics)


if __name__ == "__main__":
    main()