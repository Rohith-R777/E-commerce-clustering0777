"""Generate reproducible metrics and segment artifacts for the project report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .phase1 import build_rfm_features, load_transactions, run_baseline
from .phase6 import build_segment_profiles
from .hybrid import run_dbscan_kmeans, select_business_configuration, tune_dbscan
from .smart_selector import smart_cluster


def generate_report(input_path: str | Path, output_dir: str | Path) -> dict:
    """Run the baseline and write metrics, customer labels, profiles, and Markdown."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    transactions = load_transactions(input_path)
    rfm = build_rfm_features(transactions)
    result = run_baseline(rfm)
    profiles = build_segment_profiles(result.customers)
    feature_names = ["recency", "frequency", "monetary"]
    scaled_rfm = result.scaler.transform(rfm[feature_names])
    smart_output = smart_cluster(scaled_rfm)
    smart_result = smart_output["result"]
    smart_summary = {
        "smart_algorithm": smart_output["decision"]["algorithm"],
        "smart_reason": smart_output["decision"]["reason"],
        "smart_retention": smart_result["retention"],
        "smart_outlier_count": int(smart_result["outlier_flags"].sum()),
        "smart_silhouette": smart_result["metrics"].get("silhouette"),
        "smart_davies_bouldin": smart_result["metrics"].get("davies_bouldin"),
    }
    try:
        hybrid = run_dbscan_kmeans(scaled_rfm)
        hybrid_metrics = {f"hybrid_{key}": value for key, value in hybrid.metrics.items()}
    except ValueError:
        hybrid_metrics = {
            "hybrid_silhouette": None,
            "hybrid_davies_bouldin": None,
            "hybrid_retained_customers": 0,
            "hybrid_outliers": len(rfm),
        }
    sensitivity = tune_dbscan(scaled_rfm)
    try:
        selected_config, selected_hybrid = select_business_configuration(scaled_rfm)
        selected_metrics = {
            "selected_eps": selected_config["eps"],
            "selected_min_samples": selected_config["min_samples"],
            "selected_silhouette": selected_config["silhouette"],
            "selected_davies_bouldin": selected_config["davies_bouldin"],
            "selected_retention_pct": selected_config["retention_pct"],
            "selected_silhouette_change_pct": (
                (selected_config["silhouette"] - result.metrics["silhouette"])
                / abs(result.metrics["silhouette"])
                * 100
            ),
            "selected_dbi_change_pct": (
                (selected_config["davies_bouldin"] - result.metrics["davies_bouldin"])
                / result.metrics["davies_bouldin"]
                * 100
            ),
        }
    except ValueError:
        selected_config = None
        selected_hybrid = None
        selected_metrics = {}
    metrics = {
        "transaction_rows": int(len(transactions)),
        "customer_count": int(len(rfm)),
        "cluster_count": int(result.customers["cluster"].nunique()),
        **result.metrics,
        **hybrid_metrics,
        **selected_metrics,
        **smart_summary,
    }
    if metrics["hybrid_silhouette"] is not None:
        metrics["hybrid_silhouette_improvement_pct"] = (
            (metrics["hybrid_silhouette"] - metrics["silhouette"])
            / abs(metrics["silhouette"])
            * 100
        )
        metrics["hybrid_dbi_reduction_pct"] = (
            (metrics["davies_bouldin"] - metrics["hybrid_davies_bouldin"])
            / metrics["davies_bouldin"]
            * 100
        )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    result.customers.to_csv(output / "customer_segments.csv", index=False)
    profiles.to_csv(output / "segment_profiles.csv", index=False)
    markdown = "# Customer Clustering Results\n\n"
    markdown += "## Baseline vs. Hybrid\n\n"
    markdown += "The baseline uses standardized RFM features with K-Means++. "
    markdown += "The hybrid method filters DBSCAN noise before fitting K-Means. "
    markdown += "Hybrid metrics describe retained customers only.\n\n"
    markdown += "\n".join(f"- **{key}**: {value}" for key, value in metrics.items())
    markdown += "\n\n## Segment Profiles\n\n" + profiles.to_markdown(index=False) + "\n"
    markdown += "\n## Automated Algorithm Selection\n\n"
    markdown += "The selector profiles the standardized RFM data before choosing a pipeline.\n\n"
    markdown += pd.DataFrame([smart_output["profile"]]).to_markdown(index=False) + "\n\n"
    markdown += (
        f"**Recommended algorithm:** `{smart_output['decision']['algorithm']}`. "
        f"**Reason:** {smart_output['decision']['reason']}.\n\n"
    )
    markdown += (
        f"The selected pipeline retains {smart_result['retention']:.1%} of customers and "
        f"flags {int(smart_result['outlier_flags'].sum())} potential outliers without excluding "
        "them from K-Means assignment.\n"
    )
    markdown += "\n## DBSCAN Parameter Sensitivity Analysis\n\n"
    markdown += "The sweep evaluates k-distance-derived epsilon values. "
    markdown += "The selected configuration requires at least 80% customer retention.\n\n"
    sensitivity_frame = pd.DataFrame(sensitivity)
    markdown += sensitivity_frame.to_markdown(index=False) + "\n"
    if selected_config is not None:
        markdown += (
            f"\n**Selected configuration:** eps={selected_config['eps']}, "
            f"min_samples={selected_config['min_samples']}, "
            f"retention={selected_config['retention_pct']:.1f}%, "
            f"Silhouette={selected_config['silhouette']:.4f}, "
            f"DBI={selected_config['davies_bouldin']:.4f}.\n"
        )
        markdown += (
            f"Relative to the baseline, this retention-first setting changes "
            f"Silhouette by {selected_metrics['selected_silhouette_change_pct']:.2f}% "
            f"and DBI by {selected_metrics['selected_dbi_change_pct']:.2f}%.\n"
        )
    markdown += "\n## Business Viability\n\n"
    markdown += "The untuned hybrid result is useful for analysis but excluded too many customers. "
    markdown += "The selected configuration prioritizes broad customer coverage before optimizing quality metrics. "
    markdown += "On this small dataset, that trade-off retains customers but lowers cluster quality, so it should "
    markdown += "be treated as a coverage comparison rather than a replacement for the baseline.\n"
    markdown += "\n## SDG Alignment\n\n"
    markdown += "- **SDG 8:** Prioritize high-value and loyal customer segments for efficient growth.\n"
    markdown += "- **SDG 9:** Apply reproducible analytics and hybrid machine-learning methods.\n"
    markdown += "- **SDG 12:** Reduce marketing waste through targeted segment campaigns.\n"
    (output / "report.md").write_text(markdown, encoding="utf-8")
    return metrics
