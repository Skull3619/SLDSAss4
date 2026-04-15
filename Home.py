from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Manufacturing Feasibility App", page_icon="🧩", layout="wide")

st.title("🧩 Manufacturing Feasibility Analysis App")
st.write(
    """
This app is organized to match the assignment workflow:

- **Q1 Visualization**: feature-space visualization, dimension reduction, clustering
- **Q2 Smart Data Selection**: compare subset strategies and imbalance handling
- **Q3 Feature Engineering**: rank features and create augmented feature sets
- **Q4 Pipelines and Diagnostics**: benchmark many pipelines and inspect failures
- **Q5 Overall Report**: consolidated summary and downloadable report artifacts

Use the sidebar to move across pages.
"""
)

st.subheader("Recommended workflow")
st.write(
    """
1. Extract features **offline** from the raw 3D point clouds.  
2. Upload the extracted feature dataset to Pages 1 to 5.  
3. Use Pages 1 to 4 to generate results.  
4. Use Page 5 to download the overall report.
"""
)
