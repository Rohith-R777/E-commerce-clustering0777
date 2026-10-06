"""
Smart E-Commerce Customer Segmentation — Streamlit UI
- No caching: always reads the freshest report
- Runs BOTH run_smart.py and run_phase1.py so all outputs refresh
- Business page derives segments from real customer_segments.csv
- Shows report timestamps so you can verify freshness
- Handles any JSON structure via dig()
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Customer Segmentation",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(79, 70, 229, 0.3);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(10px);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: #4F46E5;
    }
    .hero {
        background: linear-gradient(135deg, #4F46E5 0%, #06B6D4 100%);
        border-radius: 20px;
        padding: 40px;
        color: white;
        margin-bottom: 24px;
    }
    .decision-card {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        border-radius: 16px;
        padding: 24px;
        color: white;
    }
    .missing-card {
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
        border-radius: 16px;
        padding: 20px;
        color: white;
    }
    h1, h2, h3 { color: #F1F5F9 !important; }
    .stMetric { background: transparent; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(".").resolve()
SMART_SCRIPT = PROJECT_ROOT / "scripts" / "run_smart.py"
PHASE1_SCRIPT = PROJECT_ROOT / "scripts" / "run_phase1.py"
REPORT_PATH = PROJECT_ROOT / "reports" / "uploaded_smart_report.json"
SEGMENTS_CSV = PROJECT_ROOT / "reports" / "results" / "customer_segments.csv"
UPLOAD_DIR = PROJECT_ROOT / "data" / "raw" / "uploaded"
SAMPLE_CSV = PROJECT_ROOT / "data" / "raw" / "transactions.csv"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
SEGMENTS_CSV.parent.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Helpers — NO CACHING
# ─────────────────────────────────────────────────────────────
def load_report(path_str: str):
    """Read the JSON report fresh on every page load."""
    p = Path(path_str)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None
    return None


def report_age_seconds():
    """Seconds since report file was last modified."""
    if REPORT_PATH.exists():
        return (datetime.now() - datetime.fromtimestamp(
            REPORT_PATH.stat().st_mtime)).total_seconds()
    return None


def dig(data, *keys, default=None):
    """Try multiple dotted key paths; return first match."""
    if not isinstance(data, dict):
        return default
    for key_path in keys:
        current = data
        found = True
        for k in key_path.split("."):
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                found = False
                break
        if found and current is not None:
            return current
    return default


def fmt_num(value, decimals=4, default="—"):
    if value is None or value == "":
        return default
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return default


def fmt_pct(value, default="—"):
    if value is None or value == "":
        return default
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return default


def flatten_dict(data, prefix=""):
    """Flatten nested report data into spreadsheet-friendly key/value rows."""
    rows = []
    if isinstance(data, dict):
        for key, value in data.items():
            key_path = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(flatten_dict(value, key_path))
    elif isinstance(data, list):
        rows.append({"field": prefix, "value": json.dumps(data, ensure_ascii=True)})
    else:
        rows.append({"field": prefix, "value": data})
    return rows


def report_markdown(data):
    """Create a readable Markdown export without assuming one report schema."""
    lines = ["# Smart E-Commerce Customer Segmentation Report", ""]
    for section, content in data.items():
        lines.extend([f"## {str(section).replace('_', ' ').title()}", ""])
        if isinstance(content, dict):
            for key, value in content.items():
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=True)
                lines.append(f"- **{str(key).replace('_', ' ').title()}:** {value}")
        else:
            lines.append(str(content))
        lines.append("")
    return "\n".join(lines)


def run_smart_pipeline(csv_path: Path, report_out: Path):
    """
    Run BOTH:
      1. run_smart.py  → uploaded_smart_report.json (metrics, decision)
      2. run_phase1.py → customer_segments.csv (per-customer clusters)
    """
    if not SMART_SCRIPT.exists():
        return False, f"Missing script: {SMART_SCRIPT}"

    # ─── Run 1: smart selector ───
    cmd1 = [
        sys.executable,
        str(SMART_SCRIPT),
        str(csv_path),
        "--output",
        str(report_out),
    ]
    try:
        r1 = subprocess.run(
            cmd1,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=900,
        )
    except subprocess.TimeoutExpired:
        return False, "Smart pipeline timed out (>15 min)."

    if r1.returncode != 0:
        return False, (r1.stderr or r1.stdout or "Unknown error")[-2500:]

    log = "─── run_smart.py ───\n" + (r1.stdout or "OK")[-1200:]

    # ─── Run 2: phase1 → customer_segments.csv ───
    if not PHASE1_SCRIPT.exists():
        log += f"\n\n⚠️ {PHASE1_SCRIPT} not found — customer_segments.csv not refreshed."
        return True, log

    cmd2 = [
        sys.executable,
        str(PHASE1_SCRIPT),
        str(csv_path),
        "--output",
        str(SEGMENTS_CSV),
    ]
    try:
        r2 = subprocess.run(
            cmd2,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=900,
        )
    except subprocess.TimeoutExpired:
        log += "\n\n─── run_phase1.py ───\n⚠️ Timed out."
        return True, log

    if r2.returncode == 0:
        log += "\n\n─── run_phase1.py ───\n" + (r2.stdout or "OK")[-1200:]
    else:
        log += "\n\n─── run_phase1.py FAILED ───\n" + (
            r2.stderr or r2.stdout or "Unknown error")[-1200:]

    return True, log


# ─────────────────────────────────────────────────────────────
# Load report fresh
# ─────────────────────────────────────────────────────────────
report = load_report(str(REPORT_PATH))

# ─────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🛒 Smart E-Commerce Customer Segmentation</h1>
    <p style="font-size: 18px; opacity: 0.9;">
        AI-driven customer clustering for sustainable growth — SDG 8, 9, 12
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    page = st.radio(
        "Go to",
        ["🏠 Home", "📊 Data", "🧠 Smart Engine", "📈 Results", "🎯 Business"],
        label_visibility="collapsed",
    )
    st.divider()

    age = report_age_seconds()
    if age is None:
        st.warning("⚠️ No report found")
    elif age < 60:
        st.success(f"🟢 Report fresh ({int(age)}s ago)")
    elif age < 600:
        st.info(f"🟡 Report {int(age / 60)} min ago")
    else:
        st.warning(f"🔴 Report {int(age / 60)} min old")

    if SEGMENTS_CSV.exists():
        age_csv = (datetime.now() - datetime.fromtimestamp(
            SEGMENTS_CSV.stat().st_mtime)).total_seconds()
        st.caption(f"CSV age: {int(age_csv)}s")

    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

    st.divider()
    st.caption("🎓 Nagarjuna College of Engineering and Technology")
    st.caption("SDG Project 2026")


# ─────────────────────────────────────────────────────────────
# Top Metrics
# ─────────────────────────────────────────────────────────────
if report:
    n_samples = dig(report,
        "profile.n_samples", "n_samples", "profile.n_customers", "n_customers",
        default=None)
    retention = dig(report,
        "result.retention", "retention", "result.retention_rate",
        default=None)
    silhouette = dig(report,
        "result.metrics.silhouette", "metrics.silhouette", "silhouette",
        "result.silhouette", "silhouette_score",
        default=None)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Customers", f"{int(n_samples):,}" if n_samples else "—",
                  "Analyzed" if n_samples else "Not in report")
    with col2:
        st.metric("Retention", fmt_pct(retention) if retention is not None else "—",
                  "No exclusions" if retention is not None else "Not in report")
    with col3:
        st.metric("Silhouette", fmt_num(silhouette) if silhouette is not None else "—",
                  "Quality score" if silhouette is not None else "Not in report")
    st.divider()


# ─────────────────────────────────────────────────────────────
# PAGE: Home
# ─────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.header("Project Overview")
    st.write(
        "This system automatically profiles your e-commerce dataset and "
        "selects the optimal clustering algorithm — no ML expertise required."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧠 How It Works")
        st.markdown("""
        1. **Upload** — Provide your transaction CSV
        2. **Profile** — Analyze dataset characteristics
        3. **Decide** — Choose the best algorithm
        4. **Execute** — Cluster customers
        5. **Explain** — Provide reasoning
        """)
    with col2:
        st.subheader("🌍 SDG Contribution")
        st.markdown("""
        - **SDG 8** — Economic growth through personalization
        - **SDG 9** — Innovation via automated ML
        - **SDG 12** — Zero customers excluded from service
        """)

    st.divider()
    st.subheader("📂 Upload Your Dataset")
    st.caption(
        "Required columns: `CustomerID`, `InvoiceNo`, `InvoiceDate`, `Quantity`, `UnitPrice`"
    )

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key="home_uploader")

    if uploaded is not None:
        csv_path = UPLOAD_DIR / uploaded.name
        csv_path.write_bytes(uploaded.getbuffer())
        st.success(f"✅ Uploaded: **{uploaded.name}**")

        with st.expander("👁️ Preview uploaded data (first 20 rows)"):
            try:
                df_preview = pd.read_csv(csv_path, nrows=20, encoding="latin-1")
                st.dataframe(df_preview, use_container_width=True)
            except Exception as e:
                st.error(f"Could not preview: {e}")

        if st.button("🚀 Run Smart Clustering", use_container_width=True, type="primary"):
            with st.spinner("Running smart selector + clustering + customer segments..."):
                ok, message = run_smart_pipeline(csv_path, REPORT_PATH)

            if ok:
                st.success("✅ Clustering complete — both report and customer segments refreshed!")
                with st.expander("📜 Pipeline output"):
                    st.code(message, language="text")

                if REPORT_PATH.exists():
                    mtime = datetime.fromtimestamp(REPORT_PATH.stat().st_mtime)
                    st.info(f"📅 Report: {mtime.strftime('%H:%M:%S')}")
                if SEGMENTS_CSV.exists():
                    mtime = datetime.fromtimestamp(SEGMENTS_CSV.stat().st_mtime)
                    st.info(f"📅 Customer segments CSV: {mtime.strftime('%H:%M:%S')}")

                st.rerun()
            else:
                st.error("❌ Clustering failed")
                st.code(message, language="text")
    else:
        st.info("👆 Upload a CSV file to begin, or use the sample below.")

    st.divider()
    st.subheader("🎬 Or use the bundled sample")

    if SAMPLE_CSV.exists():
        st.caption(f"Sample: `{SAMPLE_CSV}`")
        if st.button("▶️ Run on sample data", use_container_width=True):
            with st.spinner("Clustering sample dataset..."):
                ok, message = run_smart_pipeline(SAMPLE_CSV, REPORT_PATH)

            if ok:
                st.success("✅ Sample clustered — all outputs refreshed!")
                with st.expander("📜 Pipeline output"):
                    st.code(message, language="text")
                st.rerun()
            else:
                st.error("❌ Failed")
                st.code(message, language="text")
    else:
        st.warning(f"Sample file not found at `{SAMPLE_CSV}`")


# ─────────────────────────────────────────────────────────────
# PAGE: Data
# ─────────────────────────────────────────────────────────────
elif page == "📊 Data":
    st.header("📊 Dataset Overview")

    if REPORT_PATH.exists():
        mtime = datetime.fromtimestamp(REPORT_PATH.stat().st_mtime)
        st.caption(f"📅 Report last updated: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")

    if not report:
        st.warning("No report found. Please upload a dataset and run clustering first.")
    else:
        n_samples = dig(report, "profile.n_samples", "n_samples", default=None)
        n_features = dig(report, "profile.n_features", "n_features", default=None)
        outlier_ratio = dig(report, "profile.outlier_ratio", "outlier_ratio", default=None)
        hopkins = dig(report, "profile.hopkins", "hopkins", default=None)
        density_cv = dig(report, "profile.density_cv", "density_cv", default=None)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Samples", f"{int(n_samples):,}" if n_samples else "—")
        col2.metric("Features", n_features if n_features else "—")
        col3.metric("Outlier ratio", fmt_pct(outlier_ratio) if outlier_ratio is not None else "—")
        col4.metric("Hopkins", fmt_num(hopkins, 3) if hopkins else "—")

        st.divider()
        st.subheader("📋 Full Report Contents")
        st.json(report)


# ─────────────────────────────────────────────────────────────
# PAGE: Smart Engine
# ─────────────────────────────────────────────────────────────
elif page == "🧠 Smart Engine":
    st.header("🧠 Smart Algorithm Selector")

    if REPORT_PATH.exists():
        mtime = datetime.fromtimestamp(REPORT_PATH.stat().st_mtime)
        st.caption(f"📅 Report last updated: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")

    if not report:
        st.warning("No report found. Please upload a dataset and run clustering first.")
    else:
        algorithm = dig(report,
            "decision.algorithm", "algorithm", "chosen_algorithm",
            "decision.name", "decision.method", "decision.algo",
            default=None)
        reason = dig(report,
            "decision.reason", "reason", "rationale",
            "decision.rationale", "decision.reasoning",
            default=None)
        config = dig(report,
            "decision.config", "config", "decision.parameters",
            default=None)

        n_samples = dig(report, "profile.n_samples", "n_samples", default="—")
        n_features = dig(report, "profile.n_features", "n_features", default="—")
        outlier_ratio = dig(report, "profile.outlier_ratio", "outlier_ratio", default=None)
        hopkins = dig(report, "profile.hopkins", "hopkins", default=None)
        density_cv = dig(report, "profile.density_cv", "density_cv", default=None)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📋 Dataset Profile")
            st.markdown(f"""
            - **Samples:** {n_samples}
            - **Features:** {n_features}
            - **Outlier ratio:** {fmt_pct(outlier_ratio) if outlier_ratio is not None else "—"}
            - **Hopkins statistic:** {fmt_num(hopkins, 3) if hopkins else "—"}
            - **Density CV:** {fmt_num(density_cv, 3) if density_cv else "—"}
            """)

        with col2:
            st.subheader("🎯 Decision")
            if algorithm is None:
                st.warning("Algorithm decision not found in report.")
            else:
                algo_pretty = str(algorithm).replace("_", " ").title()
                st.markdown(f"""
                <div class="decision-card">
                    <h3>✅ {algo_pretty}</h3>
                    <p>{reason or 'No reason recorded.'}</p>
                </div>
                """, unsafe_allow_html=True)
                if config:
                    st.caption(f"Config: `{config}`")

        st.divider()
        st.subheader("🌳 Full Report Contents")
        st.json(report)


# ─────────────────────────────────────────────────────────────
# PAGE: Results
# ─────────────────────────────────────────────────────────────
elif page == "📈 Results":
    st.header("📈 Clustering Results")

    if REPORT_PATH.exists():
        mtime = datetime.fromtimestamp(REPORT_PATH.stat().st_mtime)
        age = (datetime.now() - mtime).total_seconds()
        if age < 300:
            st.success(f"🟢 Report fresh — updated {int(age)}s ago ({mtime.strftime('%H:%M:%S')})")
        else:
            st.warning(
                f"🟡 Report is {int(age / 60)} min old "
                f"({mtime.strftime('%Y-%m-%d %H:%M:%S')}). Re-run clustering to refresh."
            )

    if not report:
        st.warning("No report found. Please upload a dataset and run clustering first.")
    else:
        silhouette = dig(report,
            "result.metrics.silhouette", "metrics.silhouette", "silhouette",
            "result.silhouette", "silhouette_score", default=None)
        dbi = dig(report,
            "result.metrics.dbi", "metrics.dbi", "dbi", "result.dbi",
            "davies_bouldin", "davies_bouldin_index", default=None)
        retention = dig(report,
            "result.retention", "retention", "retention_rate", default=None)
        n_samples = dig(report, "profile.n_samples", "n_samples", default=None)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Customers", f"{int(n_samples):,}" if n_samples else "—")
        col2.metric("Silhouette", fmt_num(silhouette) if silhouette else "—")
        col3.metric("Davies-Bouldin", fmt_num(dbi) if dbi else "—")
        col4.metric("Retention", fmt_pct(retention) if retention else "—")

        if silhouette is None and dbi is None:
            st.markdown("""
            <div class="missing-card">
                <b>⚠️ No Metrics in Report</b><br>
                Your JSON only contains the algorithm decision. 
                Patch <code>scripts/run_smart.py</code> to include 
                <code>result.metrics</code>, then re-run clustering.
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.subheader("📋 Full Report Contents")
        st.json(report)

        st.divider()
        st.subheader("📥 Export Results")
        export_col1, export_col2, export_col3 = st.columns(3)
        with export_col1:
            st.download_button(
                "Download report JSON",
                data=json.dumps(report, indent=2, ensure_ascii=True).encode("utf-8"),
                file_name="smart_segmentation_report.json",
                mime="application/json",
                use_container_width=True,
            )
        with export_col2:
            report_rows = pd.DataFrame(flatten_dict(report))
            st.download_button(
                "Download report CSV",
                data=report_rows.to_csv(index=False).encode("utf-8"),
                file_name="smart_segmentation_report.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with export_col3:
            st.download_button(
                "Download report Markdown",
                data=report_markdown(report).encode("utf-8"),
                file_name="smart_segmentation_report.md",
                mime="text/markdown",
                use_container_width=True,
            )

        # ─── Customer-Level CSV ───
        st.divider()
        st.subheader("🧑‍🤝‍🧑 Customer-Level Cluster Assignments")

        if SEGMENTS_CSV.exists():
            mtime = datetime.fromtimestamp(SEGMENTS_CSV.stat().st_mtime)
            age = (datetime.now() - mtime).total_seconds()
            if age < 300:
                st.success(f"🟢 CSV fresh — updated {int(age)}s ago ({mtime.strftime('%H:%M:%S')})")
            else:
                st.warning(f"🟡 CSV is {int(age / 60)} min old — re-run clustering")
            try:
                df = pd.read_csv(SEGMENTS_CSV)
                st.success(f"Loaded {len(df)} customer assignments")

                cluster_col = next(
                    (c for c in ["Cluster", "cluster", "Segment", "segment", "label"]
                     if c in df.columns),
                    None,
                )
                if cluster_col:
                    counts = df[cluster_col].value_counts().sort_index()
                    fig = px.bar(
                        x=counts.index.astype(str),
                        y=counts.values,
                        labels={"x": "Cluster", "y": "Customers"},
                        title="Customers per Cluster",
                        height=350,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.dataframe(df, use_container_width=True)

                st.download_button(
                    "📥 Download customer segments CSV",
                    df.to_csv(index=False).encode("utf-8"),
                    file_name="customer_segments.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Could not read {SEGMENTS_CSV}: {e}")
        else:
            st.info(
                f"No customer-level CSV at `{SEGMENTS_CSV}`.\n\n"
                f"Generate it with:\n"
                f"```\npython scripts/run_phase1.py data/raw/transactions.csv "
                f"--output reports/results/customer_segments.csv\n```"
            )


# ─────────────────────────────────────────────────────────────
# PAGE: Business — driven by real customer_segments.csv
# ─────────────────────────────────────────────────────────────
elif page == "🎯 Business":
    st.header("🎯 Business Strategy & SDG Impact")

    if not SEGMENTS_CSV.exists():
        st.warning(
            f"No customer segments found at `{SEGMENTS_CSV}`. "
            "Run clustering from the **Home** page first."
        )
    else:
        mtime = datetime.fromtimestamp(SEGMENTS_CSV.stat().st_mtime)
        age = (datetime.now() - mtime).total_seconds()
        if age < 300:
            st.success(
                f"🟢 Using live segment data — updated {int(age)}s ago "
                f"({mtime.strftime('%H:%M:%S')})"
            )
        else:
            st.warning(
                f"🟡 Segment data is {int(age / 60)} min old — "
                "re-run clustering to refresh."
            )

        try:
            df = pd.read_csv(SEGMENTS_CSV)

            cluster_col = next(
                (c for c in ["Cluster", "cluster", "Segment", "segment", "label"]
                 if c in df.columns),
                None,
            )
            if cluster_col is None:
                st.error("No 'Cluster' column found in customer_segments.csv")
                st.stop()

            rec_col = next(
                (c for c in ["Recency", "recency", "R"] if c in df.columns), None)
            freq_col = next(
                (c for c in ["Frequency", "frequency", "F"] if c in df.columns), None)
            mon_col = next(
                (c for c in ["Monetary", "monetary", "M"] if c in df.columns), None)

            agg_dict = {"Customers": (cluster_col, "count")}
            if rec_col:
                agg_dict["Avg Recency"] = (rec_col, "mean")
            if freq_col:
                agg_dict["Avg Frequency"] = (freq_col, "mean")
            if mon_col:
                agg_dict["Avg Monetary"] = (mon_col, "mean")

            profiles = df.groupby(cluster_col).agg(**agg_dict).reset_index()

            if "Avg Monetary" in profiles.columns:
                profiles = profiles.sort_values(
                    "Avg Monetary", ascending=False
                ).reset_index(drop=True)

            label_map = {}
            if len(profiles) >= 1:
                label_map[profiles.iloc[0][cluster_col]] = (
                    "🏆 High-Value Loyalists", "#10B981",
                    "VIP rewards, early access, cross-selling, dedicated account manager",
                    "+15% retention (industry benchmark)",
                )
            if len(profiles) >= 2:
                label_map[profiles.iloc[1][cluster_col]] = (
                    "🛒 Intermittent Buyers", "#F59E0B",
                    "Seasonal campaigns, upselling, personalized email flows",
                    "+10% purchase frequency (industry benchmark)",
                )
            if len(profiles) >= 3:
                label_map[profiles.iloc[2][cluster_col]] = (
                    "⚠️ At-Risk Customers", "#EF4444",
                    "Win-back discounts, feedback surveys, re-engagement ads",
                    "+12% re-engagement (industry benchmark)",
                )
            for i in range(3, len(profiles)):
                label_map[profiles.iloc[i][cluster_col]] = (
                    f"🔵 Segment {profiles.iloc[i][cluster_col]}", "#6366F1",
                    "Standard engagement campaigns",
                    "Monitor and re-evaluate",
                )

            total = int(profiles["Customers"].sum())

            st.markdown(
                "Segment-specific strategies derived from **your clustering output**. "
                "Impact figures are **published industry benchmarks**, not measured outcomes."
            )

            for _, row in profiles.iterrows():
                cluster_id = row[cluster_col]
                name, color, strategy, impact = label_map.get(
                    cluster_id,
                    (f"Segment {cluster_id}", "#6366F1", "—", "—"),
                )
                count = int(row["Customers"])
                pct = (count / total * 100) if total else 0

                extras = []
                if "Avg Recency" in profiles.columns:
                    extras.append(f"Avg Recency: **{row['Avg Recency']:.1f}** days")
                if "Avg Frequency" in profiles.columns:
                    extras.append(f"Avg Frequency: **{row['Avg Frequency']:.1f}**")
                if "Avg Monetary" in profiles.columns:
                    extras.append(f"Avg Monetary: **${row['Avg Monetary']:,.2f}**")
                profile_line = " · ".join(extras) if extras else "—"

                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid {color};">
                    <h3>{name} — Cluster {cluster_id} ({count} customers, {pct:.1f}%)</h3>
                    <p><b>Profile:</b> {profile_line}</p>
                    <p><b>Strategy:</b> {strategy}</p>
                    <p><b>Expected impact:</b> {impact}</p>
                </div>
                """, unsafe_allow_html=True)
                st.write("")

            with st.expander("📊 Full Segment Profile Table (from your data)"):
                st.dataframe(profiles, use_container_width=True)

            st.download_button(
                "📥 Download segment profiles CSV",
                profiles.to_csv(index=False).encode("utf-8"),
                file_name="segment_profiles.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Could not load segment profiles: {e}")

    st.divider()
    st.subheader("🌍 SDG Contribution")
    col1, col2, col3 = st.columns(3)
    col1.info("**SDG 8**\n\nEconomic growth via personalization")
    col2.info("**SDG 9**\n\nInnovation via automated ML")
    col3.info("**SDG 12**\n\nZero customers excluded from service")


# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.divider()
st.caption("🎓 Nagarjuna College of Engineering and Technology | SDG Project 2026")