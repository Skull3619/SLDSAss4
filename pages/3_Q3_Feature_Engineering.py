from __future__ import annotations

import io

import pandas as pd
import plotly.express as px
import streamlit as st

from app_utils import add_unsupervised_features, build_feature_family_table, feature_rankings, infer_dataset_schema, load_feature_table

st.set_page_config(page_title="Q3 Feature Engineering", page_icon="🧩", layout="wide")
st.title("🧩 Q3. Feature engineering + unsupervised augmentation")
st.caption("Review feature families, add unsupervised descriptors, and export an augmented feature table.")

with st.sidebar:
    uploaded = st.file_uploader("Upload feature dataset", type=["csv", "xlsx", "parquet"])
    add_unsup = st.checkbox("Add unsupervised features", value=True)
    n_pca = st.slider("Unsupervised PCA components", 2, 6, 3)
    n_clusters = st.slider("Cluster features", 2, 8, 4)

if uploaded is None:
    st.info("Upload your extracted feature table.")
    st.stop()

df = load_feature_table(uploaded)
bundle = infer_dataset_schema(df)

st.subheader("Current feature families")
family_df = build_feature_family_table(bundle.numeric_cols)
st.dataframe(family_df["family"].value_counts().rename_axis("family").reset_index(name="count"), use_container_width=True)

rank_df = feature_rankings(bundle.df, bundle.numeric_cols, bundle.target_col)
st.subheader("Top candidate features")
st.dataframe(rank_df.head(30), use_container_width=True)

fig = px.bar(rank_df.head(20), x="feature", y="mutual_info", color="family", title="Top 20 by mutual information")
fig.update_layout(xaxis_tickangle=-35)
st.plotly_chart(fig, use_container_width=True)

aug_df = bundle.df.copy()
if add_unsup:
    aug_df = add_unsupervised_features(aug_df, bundle.numeric_cols, n_pca=n_pca, n_clusters=n_clusters)
    aug_bundle = infer_dataset_schema(aug_df)
    st.subheader("Added unsupervised features")
    unsup_cols = [c for c in aug_bundle.numeric_cols if c.startswith("unsup_")]
    st.write(unsup_cols)
    emb_cols = [c for c in unsup_cols if c.startswith("unsup_pca_")]
    if len(emb_cols) >= 2:
        emb = aug_df[emb_cols[:2]].copy()
        emb.columns = ["pc1", "pc2"]
        tmp = emb.copy()
        tmp[bundle.label_col] = aug_df[bundle.label_col].values
        fig2 = px.scatter(tmp, x="pc1", y="pc2", color=bundle.label_col, title="Unsupervised PCA augmentation view")
        st.plotly_chart(fig2, use_container_width=True)

st.subheader("Why unsupervised learning helps here")
st.markdown(
    """
Unsupervised learning is useful because it can:
- compress correlated geometric features into a smaller latent space
- expose natural groups or shape families
- produce cluster-distance or anomaly-score features
- act as preprocessing for downstream supervised models
"""
)

csv_bytes = aug_df.to_csv(index=False).encode("utf-8")
st.download_button("Download augmented CSV", data=csv_bytes, file_name="q3_augmented_features.csv", mime="text/csv")

xlsx_buffer = io.BytesIO()
with pd.ExcelWriter(xlsx_buffer, engine="openpyxl") as writer:
    aug_df.to_excel(writer, index=False, sheet_name="features")
st.download_button(
    "Download augmented XLSX",
    data=xlsx_buffer.getvalue(),
    file_name="q3_augmented_features.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
