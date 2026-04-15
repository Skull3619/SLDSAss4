from __future__ import annotations

import io
from datetime import datetime

import pandas as pd
import streamlit as st

from app_utils import dataset_status_caption, get_active_dataset_bundle, require_active_dataset

st.set_page_config(page_title="Q5 Overall Report", page_icon="📝", layout="wide")
st.title("📝 Q5. Overall report and submission summary")


def detect_feature_family(col: str) -> str:
    c = col.lower()
    if c.startswith(("occ_",)):
        return "occupancy"
    if c.startswith(("radial_", "d2_", "proj_", "hist_")):
        return "distribution_histograms"
    if c.startswith(("nn_", "hull_", "compactness", "density_", "bbox_", "extent_", "diag")):
        return "density_hull_spacing"
    if c.startswith(("eig", "linearity", "planarity", "sphericity", "anisotropy", "curvature", "eigentropy")):
        return "shape_local_geometry"
    if c.startswith(("centroid_", "min_", "max_", "x_q", "y_q", "z_q", "std_", "mean_", "var_")):
        return "global_geometry"
    return "other"


def class_breakdown(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    vc = df[target_col].value_counts(dropna=False).rename_axis("class").reset_index(name="count")
    vc["percent"] = 100 * vc["count"] / max(len(df), 1)
    return vc


def build_dataset_summary(df: pd.DataFrame, numeric_cols: list[str], target_col: str) -> dict:
    vc = df[target_col].value_counts()
    majority = int(vc.max()) if len(vc) else 0
    minority = int(vc.min()) if len(vc) else 0
    imbalance_ratio = majority / max(minority, 1)
    fam = pd.Series([detect_feature_family(c) for c in numeric_cols], name="family").value_counts().reset_index()
    fam.columns = ["family", "count"]
    return {
        "n_samples": int(len(df)),
        "n_features": int(len(numeric_cols)),
        "target_col": target_col,
        "n_classes": int(df[target_col].nunique(dropna=False)),
        "majority_count": majority,
        "minority_count": minority,
        "imbalance_ratio": float(imbalance_ratio),
        "family_table": fam,
        "class_table": class_breakdown(df, target_col),
    }


def top_ranked_features_from_state() -> pd.DataFrame | None:
    for key in ["q3_rank_table", "rank_table", "feature_rank_table", "q1_rank_table"]:
        obj = st.session_state.get(key)
        if isinstance(obj, pd.DataFrame) and not obj.empty:
            return obj.copy()
    return None


def q2_results_from_state() -> pd.DataFrame | None:
    for key in ["q2_results", "q2_sampling_results", "sampling_results"]:
        obj = st.session_state.get(key)
        if isinstance(obj, pd.DataFrame) and not obj.empty:
            return obj.copy()
    return None


def q4_results_from_state() -> pd.DataFrame | None:
    obj = st.session_state.get("q4_results")
    if isinstance(obj, pd.DataFrame) and not obj.empty:
        return obj.copy()
    return None


def make_markdown_report(dataset_summary: dict, rank_df: pd.DataFrame | None, q2_df: pd.DataFrame | None, q4_df: pd.DataFrame | None) -> str:
    lines = []
    lines.append("# ISE 5334 Homework 4: Overall Computational Report")
    lines.append("")
    lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Dataset Summary")
    lines.append(f"- Number of samples: **{dataset_summary['n_samples']}**")
    lines.append(f"- Number of numeric features: **{dataset_summary['n_features']}**")
    lines.append(f"- Target column: **{dataset_summary['target_col']}**")
    lines.append(f"- Number of classes: **{dataset_summary['n_classes']}**")
    lines.append(f"- Majority class count: **{dataset_summary['majority_count']}**")
    lines.append(f"- Minority class count: **{dataset_summary['minority_count']}**")
    lines.append(f"- Imbalance ratio (majority/minority): **{dataset_summary['imbalance_ratio']:.3f}**")
    lines.append("")
    lines.append("## Q1 to Q4 Summary")
    if q2_df is not None and not q2_df.empty and "macro_f1" in q2_df.columns:
        best_q2 = q2_df.sort_values("macro_f1", ascending=False).head(1)
        lines.append(f"- Best Q2 sampling strategy by macro F1: **{best_q2.iloc[0].get('strategy', 'N/A')}**")
    if rank_df is not None and not rank_df.empty:
        lines.append(f"- Top ranked feature from Q3: **{rank_df.iloc[0]['feature']}**")
    if q4_df is not None and not q4_df.empty:
        sort_col = "macro_f1" if "macro_f1" in q4_df.columns else q4_df.columns[0]
        best_q4 = q4_df.sort_values(sort_col, ascending=False).head(1)
        if "pipeline" in best_q4.columns:
            lines.append(f"- Best Q4 pipeline: **{best_q4.iloc[0]['pipeline']}**")
    lines.append("")
    return "\n".join(lines)


bundle = require_active_dataset()
st.caption(dataset_status_caption())

with st.sidebar:
    include_rank_table = st.checkbox("Include ranked feature table", value=True)
    include_q2 = st.checkbox("Include Q2 sampling summary", value=True)
    include_q4 = st.checkbox("Include Q4 benchmark summary", value=True)
    run_q5 = st.button("Generate overall report", type="primary", use_container_width=True)

if run_q5:
    dataset_summary = build_dataset_summary(bundle.df, bundle.numeric_cols, bundle.target_col)
    st.session_state["q5_payload"] = {
        "dataset_summary": dataset_summary,
        "rank_df": top_ranked_features_from_state(),
        "q2_df": q2_results_from_state(),
        "q4_df": q4_results_from_state(),
    }

payload = st.session_state.get("q5_payload")
if payload is None:
    st.info("Click **Generate overall report** to build the consolidated summary.")
    st.stop()

dataset_summary = payload["dataset_summary"]
rank_df = payload["rank_df"]
q2_df = payload["q2_df"]
q4_df = payload["q4_df"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Samples", dataset_summary["n_samples"])
c2.metric("Numeric features", dataset_summary["n_features"])
c3.metric("Classes", dataset_summary["n_classes"])
c4.metric("Imbalance ratio", f"{dataset_summary['imbalance_ratio']:.2f}")

st.subheader("Dataset summary")
left, right = st.columns([1, 1])
with left:
    st.dataframe(dataset_summary["class_table"], use_container_width=True)
with right:
    st.dataframe(dataset_summary["family_table"], use_container_width=True)

st.subheader("Q2 summary")
if include_q2 and q2_df is not None and not q2_df.empty:
    st.dataframe(q2_df, use_container_width=True)

st.subheader("Q3 summary")
if include_rank_table and rank_df is not None and not rank_df.empty:
    st.dataframe(rank_df.head(50), use_container_width=True)

st.subheader("Q4 summary")
if include_q4 and q4_df is not None and not q4_df.empty:
    st.dataframe(q4_df, use_container_width=True)

markdown_report = make_markdown_report(
    dataset_summary=dataset_summary,
    rank_df=rank_df if include_rank_table else None,
    q2_df=q2_df if include_q2 else None,
    q4_df=q4_df if include_q4 else None,
)

st.download_button(
    "Download markdown report",
    data=markdown_report.encode("utf-8"),
    file_name="q5_overall_report.md",
    mime="text/markdown",
)

workbook = io.BytesIO()
with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
    dataset_summary["class_table"].to_excel(writer, index=False, sheet_name="class_summary")
    dataset_summary["family_table"].to_excel(writer, index=False, sheet_name="feature_families")
    if include_rank_table and rank_df is not None and not rank_df.empty:
        rank_df.to_excel(writer, index=False, sheet_name="ranked_features")
    if include_q2 and q2_df is not None and not q2_df.empty:
        q2_df.to_excel(writer, index=False, sheet_name="q2_sampling")
    if include_q4 and q4_df is not None and not q4_df.empty:
        q4_df.to_excel(writer, index=False, sheet_name="q4_benchmark")

st.download_button(
    "Download report workbook",
    data=workbook.getvalue(),
    file_name="q5_overall_report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
