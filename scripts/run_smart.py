"""Run the automatic clustering selector on a transaction CSV or Parquet file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecommerce_clustering.phase1 import build_clustering_features, load_transactions, run_baseline
from ecommerce_clustering.smart_selector import smart_cluster


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/smart_report.json"))
    args = parser.parse_args()
    features = build_clustering_features(load_transactions(args.input))
    baseline = run_baseline(features)
    feature_names = list(features.columns[1:])
    values = baseline.scaler.transform(features[feature_names])
    output = smart_cluster(values)
    serializable = {
        "profile": output["profile"],
        "decision": output["decision"],
        "metrics": output["result"]["metrics"],
        "retention": output["result"]["retention"],
        "outlier_count": int(output["result"]["outlier_flags"].sum()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    main()
