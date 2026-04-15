from __future__ import annotations

import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from app_utils import infer_dataset_schema, load_feature_table

st.set_page_config(page_title="Q5 Overall Report", page_icon="📝", layout="wide")
st.title("📝 Q5. Overall report and submission summary")
st.caption("Central summary for Q1 to Q4, with downloadable report artifacts.")

# -----------------------------
# Helpers
# -----------------------------
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
    # Try common session-state keys without breaking if absent
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


def make_markdown_report(
    dataset_summary: dict,
    rank_df: pd.DataFrame | None,
    q2_df: pd.DataFrame | None,
    q4_df: pd.DataFrame | None,
) -> str:
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

    lines.append("## Q1. Visualization, Inclusion/Exclusion of Points, Dimension Reduction, and Clustering")
    lines.append("- The app supports 3D point-cloud inspection for a few designs, plus feature-based PCA/t-SNE and clustering.")
    lines.append("- Include points representing the main body and meaningful geometric structure of the part.")
    lines.append("- Exclude duplicated points, isolated extreme outliers, and obviously corrupted sparse fragments.")
    lines.append("- For final interpretation, PCA/t-SNE and clustering should be discussed using shape-centered features rather than absolute placement features whenever alignment is uncertain.")
    lines.append("")

    lines.append("## Q2. Smart Data Selection and Class Imbalance")
    if q2_df is not None and not q2_df.empty:
        lines.append(f"- Number of sampling strategy rows captured: **{len(q2_df)}**")
        if "macro_f1" in q2_df.columns:
            best_q2 = q2_df.sort_values("macro_f1", ascending=False).head(1)
            lines.append(f"- Best sampling strategy by macro F1: **{best_q2.iloc[0].get('strategy', 'N/A')}**")
            lines.append(f"- Best macro F1: **{float(best_q2.iloc[0]['macro_f1']):.4f}**")
    else:
        lines.append("- Sampling experiment results were not found in session state.")
    lines.append("- Class imbalance can bias pipelines toward the feasible class; therefore macro F1, minority-class F1, and recall for infeasible parts should be emphasized rather than accuracy alone.")
    lines.append("")

    lines.append("## Q3. Feature Engineering and Unsupervised Learning")
    lines.append("- Feature engineering includes geometric, hull/density, occupancy, radial/projection histogram, and local-shape descriptors.")
    lines.append("- Unsupervised learning can support feature engineering through PCA latent features, clustering-derived signals, and anomaly scores.")
    if rank_df is not None and not rank_df.empty:
        lines.append(f"- Top ranked feature: **{rank_df.iloc[0]['feature']}**")
    else:
        lines.append("- Ranked feature table was not found in session state.")
    lines.append("")

    lines.append("## Q4. Pipeline System, Benchmarking, and Misclassification Diagnostics")
    if q4_df is not None and not q4_df.empty:
        lines.append(f"- Number of benchmarked pipelines captured: **{len(q4_df)}**")
        sort_col = "macro_f1" if "macro_f1" in q4_df.columns else q4_df.columns[0]
        best_q4 = q4_df.sort_values(sort_col, ascending=False).head(1)
        best_name = best_q4.iloc[0]["pipeline"] if "pipeline" in best_q4.columns else "N/A"
        lines.append(f"- Best pipeline: **{best_name}**")
        if "macro_f1" in q4_df.columns:
            lines.append(f"- Best macro F1: **{float(best_q4.iloc[0]['macro_f1']):.4f}**")
        if "f1_infeasible" in q4_df.columns:
            lines.append(f"- Best infeasible-class F1: **{float(best_q4.iloc[0]['f1_infeasible']):.4f}**")
        if "accuracy" in q4_df.columns:
            lines.append(f"- Best accuracy: **{float(best_q4.iloc[0]['accuracy']):.4f}**")
    else:
        lines.append("- Q4 benchmark results were not found in session state.")
    lines.append("- Misclassification analysis is essential because high feasible-class performance can still hide poor infeasible-class recall.")
    lines.append("")

    lines.append("## Q5. Deployment Guidance")
    lines.append("- Raw 3D data should remain offline for feature extraction.")
    lines.append("- The deployed Streamlit app should operate on the extracted feature table only.")
    lines.append("- This reduces upload size, avoids memory issues, and keeps the online system practical.")
    lines.append("")

    lines.append("## Centralized Notes and Interpretation Tips")
    lines.append("- Use shape-centered or translation-robust features when alignment across designs is uncertain.")
    lines.append("- If centroid or raw coordinate features rank highly, verify that this reflects a common CAD coordinate system rather than arbitrary placement.")
    lines.append("- Prefer macro F1 and infeasible-class F1 over plain accuracy.")
    lines.append("- If confusion matrices show many infeasible parts predicted as feasible, investigate imbalance handling, threshold choice, and additional discriminative features.")
    lines.append("- For report writing, connect top-ranked features back to geometric intuition: compactness, hull structure, occupancy, and radial/projection distributions are often more meaningful than raw bounding-box size alone.")
    lines.append("")

    return "\n".join(lines)


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    uploaded = st.file_uploader("Upload feature dataset", type=["csv", "xlsx", "parquet"], key="q5_report_upload")
    include_notes = st.checkbox("Include centralized tips / notes section", value=True)
    include_rank_table = st.checkbox("Include ranked feature table", value=True)
    include_q2 = st.checkbox("Include Q2 sampling summary", value=True)
    include_q4 = st.checkbox("Include Q4 benchmark summary", value=True)

if uploaded is None:
    st.info("Upload your extracted feature dataset to generate the overall report.")
    st.stop()

df = load_feature_table(uploaded)
bundle = infer_dataset_schema(df)
dataset_summary = build_dataset_summary(bundle.df, bundle.numeric_cols, bundle.target_col)

rank_df = top_ranked_features_from_state()
q2_df = q2_results_from_state()
q4_df = q4_results_from_state()

# -----------------------------
# Main layout
# -----------------------------
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

st.divider()

st.subheader("Q1 summary")
st.markdown(
    """
**What the analysis should include**
- visualize a few 3D designs
- keep main-body points and meaningful structural points
- exclude duplicated points, isolated extreme outliers, and corrupted sparse fragments
- use PCA / t-SNE and clustering to understand dataset structure

**Interpretation notes**
- if PCA / clustering changes sharply when absolute-position features are removed, the original analysis may have been influenced by placement rather than shape
- if the parts are known to share a common CAD coordinate system, some coordinate-based features may still be acceptable
"""
)

st.subheader("Q2 summary")
if include_q2 and q2_df is not None and not q2_df.empty:
    st.dataframe(q2_df, use_container_width=True)
else:
    st.info("No Q2 sampling results were found in session state yet.")

st.markdown(
    """
**Class imbalance note**
- class imbalance can cause pipelines to over-predict the majority class
- therefore macro F1, minority-class recall, and infeasible-class F1 should be used in addition to accuracy
"""
)

st.subheader("Q3 summary")
st.markdown(
    """
**Feature engineering note**
- stronger features usually come from hull structure, occupancy patterns, radial/projection distributions, and shape descriptors
- unsupervised learning can help by adding PCA latent features, cluster distances, and anomaly scores
"""
)

if include_rank_table:
    if rank_df is not None and not rank_df.empty:
        st.dataframe(rank_df.head(50), use_container_width=True)
    else:
        st.info("No ranked feature table was found in session state.")

st.subheader("Q4 summary")
if include_q4 and q4_df is not None and not q4_df.empty:
    st.dataframe(q4_df, use_container_width=True)
    display_cols = [c for c in ["pipeline", "macro_f1", "f1_infeasible", "accuracy", "precision_infeasible", "recall_infeasible"] if c in q4_df.columns]
    if display_cols:
        best_sort = "macro_f1" if "macro_f1" in q4_df.columns else display_cols[-1]
        best_q4 = q4_df.sort_values(best_sort, ascending=False).head(10)
        st.bar_chart(best_q4.set_index("pipeline")[ [c for c in display_cols if c != "pipeline"] ])
else:
    st.info("No Q4 benchmark results were found in session state.")

st.markdown(
    """
**Diagnostics note**
- a model may have moderate accuracy but still perform poorly on infeasible designs
- confusion matrices and misclassified-sample analysis are essential for understanding this failure mode
"""
)

st.subheader("Q5 deployment summary")
st.markdown(
    """
- Feature extraction should remain **offline**
- The online Streamlit app should operate on the **extracted feature table**
- This design is appropriate for large raw datasets and avoids upload-memory failures
"""
)

if include_notes:
    st.subheader("Centralized tips / interpretation notes")
    st.markdown(
        """
- Prefer shape-centered or translation-robust features when design alignment is uncertain.
- Treat high importance of centroid / raw coordinate features with caution unless a common CAD coordinate system is guaranteed.
- For model selection, do not rely on accuracy alone; use macro F1 and infeasible-class F1.
- If infeasible recall is poor, examine imbalance handling, thresholding, and richer geometric features.
- When writing the report, connect important features back to geometry: compactness, hull structure, occupancy, and radial/projection shape signatures.
"""
    )

st.divider()

# -----------------------------
# Downloads
# -----------------------------
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
