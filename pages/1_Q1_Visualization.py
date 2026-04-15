from __future__ import annotations

import plotly.express as px
import streamlit as st

from app_utils import (
    build_feature_family_table,
    cluster_features,
    compute_embedding,
    dataset_status_caption,
    feature_rankings,
    missingness_table,
    require_active_dataset,
)

st.set_page_config(page_title="Q1 Visualization", page_icon="📊", layout="wide")
st.title("📊 Q1. Visualization, clustering, and data-quality view")

bundle = require_active_dataset()
st.caption(dataset_status_caption())

with st.sidebar:
    emb_method = st.selectbox("2D embedding", ["PCA", "t-SNE"])
    tsne_perplexity = st.slider("t-SNE perplexity", 5, 40, 20)
    cluster_method = st.selectbox("Clustering", ["KMeans", "DBSCAN"])
    n_clusters = st.slider("KMeans clusters", 2, 8, 4)
    eps = st.slider("DBSCAN eps", 0.2, 2.0, 0.9, 0.05)
    run_q1 = st.button("Run Q1 analysis", type="primary", use_container_width=True)

if run_q1:
    emb = compute_embedding(bundle.df, bundle.numeric_cols, method=emb_method, perplexity=tsne_perplexity)
    clusters = cluster_features(bundle.df, bundle.numeric_cols, method=cluster_method.lower(), n_clusters=n_clusters, eps=eps)
    plot_df = emb.copy()
    plot_df[bundle.label_col] = bundle.df[bundle.label_col].values
    plot_df["cluster"] = clusters.astype(str)
    rank_df = feature_rankings(bundle.df, bundle.numeric_cols, bundle.target_col)
    st.session_state["q1_payload"] = {
        "plot_df": plot_df,
        "rank_df": rank_df,
        "emb_method": emb_method,
        "family_counts": build_feature_family_table(bundle.numeric_cols)["family"].value_counts().rename_axis("family").reset_index(name="count"),
    }

payload = st.session_state.get("q1_payload")
if payload is None:
    st.info("Adjust the sidebar settings and click **Run Q1 analysis**.")
    st.stop()

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
    st.dataframe(payload["family_counts"], use_container_width=True)

st.subheader("Embedding + clustering")
color_by = st.radio("Color by", ["label", "cluster"], horizontal=True)
color_col = bundle.label_col if color_by == "label" else "cluster"
fig = px.scatter(payload["plot_df"], x="emb_1", y="emb_2", color=color_col, title=f"{payload['emb_method']} embedding of extracted features")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Top discriminating features")
st.dataframe(payload["rank_df"].head(25), use_container_width=True)
fig2 = px.bar(payload["rank_df"].head(20), x="feature", y="rank_score", color="family", title="Top 20 ranked features")
fig2.update_layout(xaxis_tickangle=-35)
st.plotly_chart(fig2, use_container_width=True)

st.download_button("Download ranked-feature table", data=payload["rank_df"].to_csv(index=False).encode("utf-8"), file_name="q1_feature_rankings.csv", mime="text/csv")
