from __future__ import annotations

import streamlit as st

from app_utils import (
    active_dataset_available,
    clear_analysis_results,
    dataset_status_caption,
    get_active_dataset_bundle,
    get_active_dataset_name,
    load_feature_table,
    set_active_dataset,
)

st.set_page_config(page_title="Manufacturing Feasibility App", page_icon="🧩", layout="wide")
st.title("🧩 Manufacturing Feasibility Analysis App")
st.write(
    """
This app is organized to match the assignment workflow:

- **Q1 Visualization**
- **Q2 Smart Data Selection**
- **Q3 Feature Engineering**
- **Q4 Pipelines and Diagnostics**
- **Q5 Overall Report**
"""
)

with st.sidebar:
    uploaded = st.file_uploader("Upload feature dataset once", type=["csv", "xlsx", "parquet"], key="home_shared_upload")
    if st.button("Load dataset", type="primary", use_container_width=True, disabled=uploaded is None):
        df = load_feature_table(uploaded)
        set_active_dataset(df, uploaded.name)
        st.success(f"Loaded {uploaded.name}")
    if active_dataset_available() and st.button("Clear loaded dataset", use_container_width=True):
        for key in ["active_dataset_df", "active_dataset_bundle", "active_dataset_name", "active_dataset_token"]:
            st.session_state.pop(key, None)
        clear_analysis_results()
        st.success("Cleared shared dataset")

st.subheader("Shared dataset status")
st.write(dataset_status_caption())

if active_dataset_available():
    bundle = get_active_dataset_bundle()
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(bundle.df))
    c2.metric("Numeric features", len(bundle.numeric_cols))
    c3.metric("Classes", bundle.df[bundle.label_col].nunique())
    st.write(f"Active dataset: **{get_active_dataset_name()}**")
    preview_cols = [c for c in ["file_name", bundle.label_col, bundle.target_col] if c in bundle.df.columns]
    if preview_cols:
        st.dataframe(bundle.df[preview_cols].head(10), use_container_width=True)
else:
    st.info("Upload your extracted feature dataset here once. The same dataset will then be used across all pages.")
