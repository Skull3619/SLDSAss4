from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
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


def pick_diverse_top_features(rank_df, n=5):
    chosen = []
    used_families = set()
    if "family" not in rank_df.columns:
        return rank_df["feature"].head(n).tolist()

    for _, row in rank_df.iterrows():
        fam = row["family"]
        feat = row["feature"]
        if fam not in used_families:
            chosen.append(feat)
            used_families.add(fam)
        if len(chosen) >= n:
            break

    if len(chosen) < n:
        for feat in rank_df["feature"].tolist():
            if feat not in chosen:
                chosen.append(feat)
            if len(chosen) >= n:
                break
    return chosen[:n]


with st.sidebar:
    emb_method = st.selectbox("2D embedding", ["PCA", "t-SNE"])
    tsne_perplexity = st.slider("t-SNE perplexity", 5, 40, 20)
    cluster_method = st.selectbox("Clustering", ["KMeans", "DBSCAN"])
    n_clusters = st.slider("KMeans clusters", 2, 8, 4)
    dbscan_min_samples = st.slider("DBSCAN min_samples", 3, 20, 5)
    eps = st.slider("DBSCAN eps", 0.2, 2.0, 0.9, 0.05)
    run_q1 = st.button("Run Q1 analysis", type="primary", use_container_width=True)

if run_q1:
    emb = compute_embedding(
        bundle.df,
        bundle.numeric_cols,
        method=emb_method,
        perplexity=tsne_perplexity,
    )

    clusters = cluster_features(
        bundle.df,
        bundle.numeric_cols,
        method=cluster_method.lower(),
        n_clusters=n_clusters,
        eps=eps,
        min_samples=dbscan_min_samples,
    )

    plot_df = emb.copy()
    plot_df[bundle.label_col] = bundle.df[bundle.label_col].values
    plot_df["cluster"] = clusters.astype(str)

    rank_df = feature_rankings(bundle.df, bundle.numeric_cols, bundle.target_col)

    family_df = build_feature_family_table(bundle.numeric_cols)
    if "family" in family_df.columns:
        family_counts = (
            family_df.groupby("family", as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
    else:
        family_counts = family_df.copy()

    diverse_feats = pick_diverse_top_features(rank_df, n=5)

    st.session_state["q1_payload"] = {
        "plot_df": plot_df,
        "rank_df": rank_df,
        "emb_method": emb_method,
        "cluster_method": cluster_method,
        "family_counts": family_counts,
        "diverse_feats": diverse_feats,
    }

payload = st.session_state.get("q1_payload")
if payload is None:
    st.info("Adjust the sidebar settings and click **Run Q1 analysis**.")
    st.stop()

rank_df = payload["rank_df"]
diverse_feats = payload["diverse_feats"]

st.subheader("Dataset snapshot")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", len(bundle.df))
c2.metric("Numeric features", len(bundle.numeric_cols))
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

fig = px.scatter(
    payload["plot_df"],
    x="emb_1",
    y="emb_2",
    color=color_col,
    title=f"{payload['emb_method']} embedding of extracted features",
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Top discriminating features")
st.dataframe(rank_df.head(25), use_container_width=True)

fig2 = px.bar(
    rank_df.head(20),
    x="feature",
    y="rank_score",
    color="family",
    title="Top 20 ranked features",
)
fig2.update_layout(xaxis_tickangle=-35)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Class-wise distributions of top features")
n_box = st.slider("Number of top features for class-wise plots", 3, 8, 5, key="q1_n_box")
plot_type = st.radio("Distribution plot type", ["Box", "Violin"], horizontal=True, key="q1_dist_plot")
top_feats = rank_df["feature"].head(n_box).tolist()

for feat in top_feats:
    if plot_type == "Box":
        f = px.box(
            bundle.df,
            x=bundle.label_col,
            y=feat,
            color=bundle.label_col,
            points="outliers",
            title=f"{feat} by class",
        )
    else:
        f = px.violin(
            bundle.df,
            x=bundle.label_col,
            y=feat,
            color=bundle.label_col,
            box=True,
            points="outliers",
            title=f"{feat} by class",
        )
    st.plotly_chart(f, use_container_width=True)

st.subheader("Correlation heatmap of top features")
n_corr = st.slider("Number of top features in heatmap", 5, 15, 10, key="q1_n_corr")
corr_feats = [f for f in rank_df["feature"].head(n_corr).tolist() if f in bundle.df.columns]
corr_df = bundle.df[corr_feats].corr(numeric_only=True)
fig_corr = px.imshow(
    corr_df,
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    title="Correlation heatmap of top ranked features",
)
fig_corr.update_layout(height=700)
st.plotly_chart(fig_corr, use_container_width=True)

st.subheader("Selected feature-pair relationship")
pair_feats = rank_df["feature"].head(12).tolist()
col1, col2 = st.columns(2)
with col1:
    x_feat = st.selectbox("X feature", pair_feats, index=0, key="q1_pair_x")
with col2:
    y_feat = st.selectbox("Y feature", pair_feats, index=min(1, len(pair_feats)-1), key="q1_pair_y")
fig_pair = px.scatter(
    bundle.df,
    x=x_feat,
    y=y_feat,
    color=bundle.label_col,
    opacity=0.7,
    title=f"{x_feat} vs {y_feat}",
)
st.plotly_chart(fig_pair, use_container_width=True)

st.subheader("Parallel coordinates of diverse top features")
par_feats = [f for f in diverse_feats if f in bundle.df.columns][:5]
if len(par_feats) >= 2:
    par_df = bundle.df[par_feats + [bundle.target_col]].copy()
    color_series = bundle.df[bundle.target_col].astype("category").cat.codes
    fig_par = px.parallel_coordinates(
        par_df,
        dimensions=par_feats,
        color=color_series,
        title="Parallel coordinates for top diverse features",
    )
    st.plotly_chart(fig_par, use_container_width=True)

st.subheader("Scatter plot matrix of significant features")
show_scatter_matrix = st.checkbox("Show scatter plot matrix", value=False, key="q1_show_spm")
if show_scatter_matrix:
    spm_feats = [f for f in diverse_feats if f in bundle.df.columns][:4]
    if len(spm_feats) >= 2:
        fig_spm = px.scatter_matrix(
            bundle.df[spm_feats + [bundle.label_col]],
            dimensions=spm_feats,
            color=bundle.label_col,
            title="Scatter plot matrix of top diverse features",
        )
        fig_spm.update_traces(diagonal_visible=False, marker=dict(size=5, opacity=0.65))
        fig_spm.update_layout(height=950, width=1100)
        st.plotly_chart(fig_spm, use_container_width=True)

st.download_button(
    "Download ranked-feature table",
    data=rank_df.to_csv(index=False).encode("utf-8"),
    file_name="q1_feature_rankings.csv",
    mime="text/csv",
)
