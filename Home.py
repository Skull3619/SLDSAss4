from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Manufacturing Feasibility Analytics Suite", page_icon="🏭", layout="wide")

st.title("🏭 Manufacturing Feasibility Analytics Suite")
st.caption("Streamlit app for Q1 to Q4 using a precomputed feature table. Raw point-cloud feature extraction stays completely offline.")

st.markdown(
    """
This project is split into two layers:

**Offline**
- Read raw 3D point clouds from your local `feasible/` and `infeasible/` folders
- Clean duplicates / obvious outliers
- Extract a rich feature table
- Save `CSV`, `XLSX`, or `Parquet`

**Online / Deployable Streamlit**
- Upload the extracted feature table
- Visualize and cluster the dataset
- Simulate smart subset selection
- Add unsupervised augmentations
- Benchmark many pipelines
- Diagnose misclassified samples
"""
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.subheader("Q1")
    st.write("Visualization, PCA / t-SNE, clustering, outlier review, and data-quality summary.")
with col2:
    st.subheader("Q2")
    st.write("Subset selection and class-imbalance simulations with F1 comparison.")
with col3:
    st.subheader("Q3")
    st.write("Feature families, unsupervised augmentations, and feature-selection analysis.")
with col4:
    st.subheader("Q4")
    st.write("50+ pipelines, benchmarking, confusion matrices, and misclassification diagnostics.")

st.markdown("### Included offline extractor")
st.code(
    "python offline_feature_extractor.py --root_dir /path/to/dataset --output features.parquet --workers 4",
    language="bash",
)

st.markdown("### Expected feature-table columns")
st.write(
    "The app auto-detects a label column and numeric feature columns. "
    "It works best when the table includes `file_name`, `label`, `target`, and the extracted numeric descriptors."
)

st.info(
    "Use the sidebar pages to run each assignment section. For deployment, keep `Home.py`, `pages/`, and `requirements.txt` in the repo."
)
