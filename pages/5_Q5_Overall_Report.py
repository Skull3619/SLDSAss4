from __future__ import annotations

import io
from datetime import datetime

import pandas as pd
import streamlit as st

from app_utils import dataset_status_caption, require_active_dataset

st.set_page_config(page_title="Q5 Overall Report", page_icon="📝", layout="wide")
st.title("📝 Q5. Overall report and submission summary")

try:
    bundle = require_active_dataset()
except RuntimeError as e:
    st.info(str(e))
    st.stop()

st.caption(dataset_status_caption())


def class_breakdown(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    vc = df[target_col].value_counts(dropna=False).rename_axis("class").reset_index(name="count")
    vc["percent"] = 100 * vc["count"] / max(len(df), 1)
    return vc


def make_markdown_report(dataset_summary: dict, q2_df: pd.DataFrame | None, q4_df: pd.DataFrame | None) -> str:
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
    lines.append(f"- Imbalance ratio (majority/minority): **{dataset_summary['imbalance_ratio']:.3f}**")
    lines.append("")
    if q2_df is not None and not q2_df.empty:
        lines.append("## Q2 Summary")
        lines.append(q2_df.head(10).to_markdown(index=False))
        lines.append("")
    if q4_df is not None and not q4_df.empty:
        lines.append("## Q4 Summary")
        lines.append(q4_df.head(15).to_markdown(index=False))
        lines.append("")
    return "
".join(lines)

with st.sidebar:
    include_q2 = st.checkbox("Include Q2 sampling summary", value=True)
    include_q4 = st.checkbox("Include Q4 benchmark summary", value=True)
    run_q5 = st.button("Generate overall report", type="primary", use_container_width=True)

if run_q5:
    vc = bundle.df[bundle.target_col].value_counts()
    dataset_summary = {
        "n_samples": int(len(bundle.df)),
        "n_features": int(len(bundle.numeric_cols)),
        "target_col": bundle.target_col,
        "n_classes": int(bundle.df[bundle.target_col].nunique(dropna=False)),
        "imbalance_ratio": float(int(vc.max()) / max(int(vc.min()), 1)),
        "class_table": class_breakdown(bundle.df, bundle.target_col),
    }
    st.session_state["q5_summary"] = dataset_summary

summary = st.session_state.get("q5_summary")
if summary is None:
    st.info("Click **Generate overall report** to build the summary.")
    st.stop()

q2_df = st.session_state.get("q2_results") if include_q2 else None
q4_df = st.session_state.get("q4_results") if include_q4 else None

c1, c2, c3, c4 = st.columns(4)
c1.metric("Samples", summary["n_samples"])
c2.metric("Numeric features", summary["n_features"])
c3.metric("Classes", summary["n_classes"])
c4.metric("Imbalance ratio", f"{summary['imbalance_ratio']:.2f}")

st.subheader("Class summary")
st.dataframe(summary["class_table"], use_container_width=True)
if q2_df is not None:
    st.subheader("Q2 summary")
    st.dataframe(q2_df, use_container_width=True)
if q4_df is not None:
    st.subheader("Q4 summary")
    st.dataframe(q4_df, use_container_width=True)

markdown_report = make_markdown_report(summary, q2_df, q4_df)
st.download_button("Download markdown report", data=markdown_report.encode("utf-8"), file_name="q5_overall_report.md", mime="text/markdown")

workbook = io.BytesIO()
with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
    summary["class_table"].to_excel(writer, index=False, sheet_name="class_summary")
    if q2_df is not None:
        q2_df.to_excel(writer, index=False, sheet_name="q2_sampling")
    if q4_df is not None:
        q4_df.to_excel(writer, index=False, sheet_name="q4_benchmark")
st.download_button("Download report workbook", data=workbook.getvalue(), file_name="q5_overall_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
