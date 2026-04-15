from __future__ import annotations

import plotly.express as px
import streamlit as st

from app_utils import build_feature_family_table, cluster_features, compute_embedding, feature_rankings, infer_dataset_schema, load_feature_table, missingness_table

st.set_page_config(page_title="Q1 Visualization", page_icon="📊", layout="wide")
st.title("📊 Q1. Visualization, clustering, and data-quality view")
st.caption("Upload the offline-extracted feature table. This page answers the dataset summary and feature-space visualization part.")

with st.sidebar:
    uploaded = st.file_uploader("Upload feature dataset", type=["csv", "xlsx", "parquet"])
    emb_method = st.selectbox("2D embedding", ["PCA", "t-SNE"])
    tsne_perplexity = st.slider("t-SNE perplexity", 5, 40, 20)
    cluster_method = st.selectbox("Clustering", ["KMeans", "DBSCAN"])
    n_clusters = st.slider("KMeans clusters", 2, 8, 4)
    eps = st.slider("DBSCAN eps", 0.2, 2.0, 0.9, 0.05)

if uploaded is None:
    st.info("Upload your extracted CSV / XLSX / Parquet table.")
    st.stop()

df = load_feature_table(uploaded)
bundle = infer_dataset_schema(df)

st.subheader("Dataset snapshot")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", len(bundle.df))
c2.metric("Numeric features", len(bundle.numeric_cols))
c3.metric("Classes", bundle.df[bundle.label_col].nunique())

preview_cols = [c for c in ["file_name", bundle.label_col, bundle.target_col] if c in bundle.df.columns]
st.write(bundle.df[preview_cols].head(10))
st.bar_chart(bundle.df[bundle.label_col].value_counts())

st.subheader("Missingness and feature families")
colA, colB = st.columns(2)
with colA:
    st.dataframe(missingness_table(bundle.df).head(25), use_container_width=True)
with colB:
    family_df = build_feature_family_table(bundle.numeric_cols)
    st.dataframe(family_df["family"].value_counts().rename_axis("family").reset_index(name="count"), use_container_width=True)

st.subheader("Embedding + clustering")
emb = compute_embedding(bundle.df, bundle.numeric_cols, method=emb_method, perplexity=tsne_perplexity)
clusters = cluster_features(bundle.df, bundle.numeric_cols, method=cluster_method.lower(), n_clusters=n_clusters, eps=eps)
plot_df = emb.copy()
plot_df[bundle.label_col] = bundle.df[bundle.label_col].values
plot_df["cluster"] = clusters.astype(str)
color_by = st.radio("Color by", ["label", "cluster"], horizontal=True)
color_col = bundle.label_col if color_by == "label" else "cluster"
fig = px.scatter(plot_df, x="emb_1", y="emb_2", color=color_col, title=f"{emb_method} embedding of extracted features")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Top discriminating features")
rank_df = feature_rankings(bundle.df, bundle.numeric_cols, bundle.target_col)
st.dataframe(rank_df.head(25), use_container_width=True)
fig2 = px.bar(rank_df.head(20), x="feature", y="rank_score", color="family", title="Top 20 ranked features")
fig2.update_layout(xaxis_tickangle=-35)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Recommended point inclusion / exclusion policy")
st.markdown(
    """
**Usually keep**
- points belonging to the main geometric body of the part
- points on stable surfaces, edges, corners, and cavities that define manufacturability
- repeated dense regions that consistently represent true geometry

**Usually remove or downweight**
- exact duplicate points
- isolated floating points far from the main body
- corrupted fragments or scan artifacts
- extreme sparse outliers that distort hull, density, and spacing features

For a feature-table workflow, the practical version is: deduplicate, optionally clip very far outliers, then extract features from the cleaned cloud.
"""
)

st.download_button("Download ranked-feature table", data=rank_df.to_csv(index=False).encode("utf-8"), file_name="q1_feature_rankings.csv", mime="text/csv")
