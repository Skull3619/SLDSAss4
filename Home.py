from __future__ import annotations

import streamlit as st

from app_utils import dataset_status_caption, load_active_dataset, clear_active_dataset

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

st.subheader("Load feature dataset once")
with st.sidebar:
    uploaded = st.file_uploader("Upload feature dataset", type=["csv", "xlsx", "parquet"], key="home_upload")
    load_btn = st.button("Load dataset", type="primary", use_container_width=True)
    clear_btn = st.button("Clear loaded dataset", use_container_width=True)

if clear_btn:
    clear_active_dataset()
    st.success("Loaded dataset cleared.")

if load_btn:
    if uploaded is None:
        st.warning("Please choose a dataset file first.")
    else:
        load_active_dataset(uploaded)
        st.success("Dataset loaded. You can now use all pages without uploading again.")

st.caption(dataset_status_caption())

if st.session_state.get("active_dataset_name"):
    st.info("Use the sidebar pages to run Q1 to Q5. Each page now has its own Run button so the analysis does not start immediately.")
else:
    st.info("Upload the extracted feature table in the sidebar and click **Load dataset**.")
