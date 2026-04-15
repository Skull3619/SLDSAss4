from __future__ import annotations

import plotly.express as px
import streamlit as st

from app_utils import (
    available_feature_families,
    build_feature_family_table,
    cluster_features,
    compute_embedding,
    dataset_status_caption,
    feature_rankings,
    filter_feature_columns,
    log_run,
    missingness_table,
    require_active_dataset,
    top_feature_scatter_matrix,
)

st.set_page_config(page_title="Q1 Visualization", page_icon="📊", layout="wide")
st.title("📊 Q1. Visualization, clustering, and data-quality view")

try:
    bundle = require_active_dataset()
except RuntimeError as e:
    st.info(str(e))
    st.stop()

st.caption(dataset_status_caption())
all_families = available_feature_families(bundle.numeric_cols)

with st.sidebar:
    include_families = st.multiselect("Feature families", all_families, default=all_families)
    exclude_absolute = st.checkbox("Exclude absolute-position features", value=False)
    emb_method = st.selectbox("2D embedding", ["PCA", "t-SNE"])
    tsne_perplexity = st.slider("t-SNE perplexity", 5, 40, 20)
    cluster_method = st.selectbox("Clustering", ["KMeans", "DBSCAN"])
    n_clusters = st.slider("KMeans clusters", 2, 8, 4)
    dbscan_min_samples = st.slider("DBSCAN min_samples", 3, 20, 5)
    eps = st.slider("DBSCAN eps", 0.2, 2.0, 0.9, 0.05)
    splom_n = st.slider("Scatter plot matrix top features", 3, 8, 5)
    run_q1 = st.button("Run Q1 analysis", type="primary", use_container_width=True)

if run_q1:
    selected_cols = filter_feature_columns(bundle.numeric_cols, include_families, exclude_absolute)
    emb = compute_embedding(bundle.df, selected_cols, method=emb_method, perplexity=tsne_perplexity)
    clusters = cluster_features(
        bundle.df,
        selected_cols,
        method=cluster_method.lower(),
        n_clusters=n_clusters,
        eps=eps,
        min_samples=dbscan_min_samples,
    )
    plot_df = emb.copy()
    plot_df[bundle.label_col] = bundle.df[bundle.label_col].values
    plot_df["cluster"] = clusters.astype(str)
    rank_df = feature_rankings(bundle.df, selected_cols, bundle.target_col)
    family_counts = build_feature_family_table(selected_cols).groupby("family", as_index=False).size().rename(columns={"size": "count"})
    splom_df = top_feature_scatter_matrix(bundle.df, rank_df, bundle.label_col, top_n=splom_n)
    caption = f"{emb_method} with {cluster_method} used {len(selected_cols)} selected features."
    st.session_state["q1_payload"] = {
        "plot_df": plot_df,
        "rank_df": rank_df,
        "emb_method": emb_method,
        "cluster_method": cluster_method,
        "family_counts": family_counts,
        "splom_df": splom_df,
        "caption": caption,
        "selected_cols": selected_cols,
    }
    log_run("Q1", {
        "families": ",".join(include_families),
        "exclude_absolute": exclude_absolute,
        "embedding": emb_method,
        "cluster": cluster_method,
        "kmeans_k": n_clusters,
        "dbscan_eps": eps,
        "dbscan_min_samples": dbscan_min_samples,
    }, {"n_selected_features": len(selected_cols)})

payload = st.session_state.get("q1_payload")
if payload is None:
    st.info("Adjust the sidebar settings and click **Run Q1 analysis**.")
    st.stop()

st.subheader("Dataset snapshot")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", len(bundle.df))
c2.metric("Numeric features used", len(payload["selected_cols"]))
c3.metric("Classes", bundle.df[bundle.label_col].nunique())

preview_cols = [c for c in ["file_name", bundle.label_col, bundle.target_col] if c in bundle.df.columns]
if preview_cols:
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
st.caption(payload["caption"])

st.subheader("Top discriminating features")
st.dataframe(payload["rank_df"].head(25), use_container_width=True)
fig2 = px.bar(payload["rank_df"].head(20), x="feature", y="rank_score", color="family", title="Top 20 ranked features")
fig2.update_layout(xaxis_tickangle=-35)
st.plotly_chart(fig2, use_container_width=True)

if not payload["splom_df"].empty:
    st.subheader("Scatter plot matrix of most significant features")
    dims = [c for c in payload["splom_df"].columns if c != bundle.label_col]
    fig3 = px.scatter_matrix(payload["splom_df"], dimensions=dims, color=bundle.label_col)
    fig3.update_traces(diagonal_visible=False, showupperhalf=False)
    st.plotly_chart(fig3, use_container_width=True)

st.download_button("Download ranked-feature table", data=payload["rank_df"].to_csv(index=False).encode("utf-8"), file_name="q1_feature_rankings.csv", mime="text/csv")
