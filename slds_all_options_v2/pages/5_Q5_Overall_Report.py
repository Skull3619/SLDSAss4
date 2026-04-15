from __future__ import annotations

import io
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

from app_utils import dataset_status_caption, get_run_history_df, require_active_dataset

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


def make_markdown_report(dataset_summary: dict, q1_payload, q2_df: pd.DataFrame | None, q3_payload, q4_df: pd.DataFrame | None) -> str:
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
    if q1_payload:
        lines.append("## Q1 Summary")
        lines.append(f"- {q1_payload.get('caption','')}")
        lines.append("")
    if q2_df is not None and not q2_df.empty:
        lines.append("## Q2 Summary")
        lines.append(q2_df.head(10).to_markdown(index=False))
        lines.append("")
    if q3_payload:
        lines.append("## Q3 Summary")
        lines.append(f"- {q3_payload.get('caption','')}")
        lines.append("")
    if q4_df is not None and not q4_df.empty:
        lines.append("## Q4 Summary")
        lines.append(q4_df.head(15).to_markdown(index=False))
        lines.append("")
    return "\n".join(lines)

with st.sidebar:
    include_q2 = st.checkbox("Include Q2 sampling summary", value=True)
    include_q4 = st.checkbox("Include Q4 benchmark summary", value=True)
    include_visuals = st.checkbox("Include visuals from Q1 to Q4", value=True)
    include_history = st.checkbox("Include run history", value=True)
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

q1_payload = st.session_state.get("q1_payload")
q2_df = st.session_state.get("q2_results") if include_q2 else None
q3_payload = st.session_state.get("q3_payload")
q4_df = st.session_state.get("q4_results") if include_q4 else None
q4_detail = st.session_state.get("q4_detail")
run_history = get_run_history_df() if include_history else pd.DataFrame()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Samples", summary["n_samples"])
c2.metric("Numeric features", summary["n_features"])
c3.metric("Classes", summary["n_classes"])
c4.metric("Imbalance ratio", f"{summary['imbalance_ratio']:.2f}")

st.subheader("Class summary")
st.dataframe(summary["class_table"], use_container_width=True)

if include_visuals and q1_payload:
    st.subheader("Q1 visuals")
    fig = px.scatter(q1_payload["plot_df"], x="emb_1", y="emb_2", color=bundle.label_col, title="Q1 embedding")
    st.plotly_chart(fig, use_container_width=True)
    fig2 = px.bar(q1_payload["rank_df"].head(20), x="feature", y="rank_score", color="family", title="Q1 top ranked features")
    fig2.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(fig2, use_container_width=True)
    if not q1_payload.get("splom_df", pd.DataFrame()).empty:
        dims = [c for c in q1_payload["splom_df"].columns if c != bundle.label_col]
        fig3 = px.scatter_matrix(q1_payload["splom_df"], dimensions=dims, color=bundle.label_col)
        fig3.update_traces(diagonal_visible=False, showupperhalf=False)
        st.plotly_chart(fig3, use_container_width=True)
    st.caption(q1_payload.get("caption", ""))

if q2_df is not None:
    st.subheader("Q2 summary")
    st.dataframe(q2_df, use_container_width=True)
    fig = px.bar(q2_df, x="strategy", y=["macro_f1", "f1_infeasible", "accuracy"], color="oversample_train" if "oversample_train" in q2_df.columns else None, barmode="group")
    st.plotly_chart(fig, use_container_width=True)

if include_visuals and q3_payload:
    st.subheader("Q3 visuals")
    fig = px.bar(q3_payload["rank_df"].head(20), x="feature", y="rank_score", color="family", title="Q3 top ranked features")
    fig.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)
    if not q3_payload.get("splom_df", pd.DataFrame()).empty:
        dims = [c for c in q3_payload["splom_df"].columns if c != bundle.label_col]
        fig_s = px.scatter_matrix(q3_payload["splom_df"], dimensions=dims, color=bundle.label_col)
        fig_s.update_traces(diagonal_visible=False, showupperhalf=False)
        st.plotly_chart(fig_s, use_container_width=True)
    st.caption(q3_payload.get("caption", ""))

if q4_df is not None:
    st.subheader("Q4 summary")
    st.dataframe(q4_df, use_container_width=True)
    fig = px.bar(q4_df.head(20), x="pipeline", y=["macro_f1", "f1_infeasible", "accuracy"], barmode="group")
    fig.update_layout(xaxis_tickangle=-40)
    st.plotly_chart(fig, use_container_width=True)
if include_visuals and q4_detail is not None:
    cm = q4_detail["confusion_matrix"]
    z_text = [[str(v) for v in row] for row in cm]
    cm_fig = ff.create_annotated_heatmap(z=cm, x=["Pred infeasible", "Pred feasible"], y=["True infeasible", "True feasible"], annotation_text=z_text, colorscale="Blues", showscale=True)
    st.plotly_chart(cm_fig, use_container_width=True)
    if q4_detail.get("feature_importance") is not None:
        fig2 = px.bar(q4_detail["feature_importance"].head(20), x="feature", y="importance", title="Q4 top feature importances")
        fig2.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig2, use_container_width=True)

if include_history and not run_history.empty:
    st.subheader("Run history")
    st.dataframe(run_history, use_container_width=True)

markdown_report = make_markdown_report(summary, q1_payload, q2_df, q3_payload, q4_df)
st.download_button("Download markdown report", data=markdown_report.encode("utf-8"), file_name="q5_overall_report.md", mime="text/markdown")

workbook = io.BytesIO()
with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
    summary["class_table"].to_excel(writer, index=False, sheet_name="class_summary")
    if q1_payload:
        q1_payload["rank_df"].to_excel(writer, index=False, sheet_name="q1_rankings")
    if q2_df is not None:
        q2_df.to_excel(writer, index=False, sheet_name="q2_sampling")
    if q3_payload:
        q3_payload["rank_df"].to_excel(writer, index=False, sheet_name="q3_rankings")
    if q4_df is not None:
        q4_df.to_excel(writer, index=False, sheet_name="q4_benchmark")
    if include_history and not run_history.empty:
        run_history.to_excel(writer, index=False, sheet_name="run_history")
st.download_button("Download report workbook", data=workbook.getvalue(), file_name="q5_overall_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
