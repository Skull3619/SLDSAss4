from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from app_utils import infer_dataset_schema, load_feature_table

st.set_page_config(page_title="Q1 Visualization", page_icon="📊", layout="wide")
st.title("📊 Q1. Visualization")

with st.sidebar:
    uploaded = st.file_uploader("Upload feature dataset", type=["csv", "xlsx", "parquet"], key="q1_upload")
    reducer = st.selectbox("Dimension reduction", ["PCA", "t-SNE"])
    n_components = st.selectbox("Components", [2, 3], index=0)
    clustering = st.selectbox("Clustering", ["None", "KMeans", "DBSCAN"])
    kmeans_k = st.slider("KMeans clusters", 2, 10, 3)
    dbscan_eps = st.slider("DBSCAN eps", 0.2, 5.0, 1.2, 0.1)
    dbscan_min_samples = st.slider("DBSCAN min_samples", 3, 20, 5)
    max_features = st.slider("Max numeric features used", 5, 200, 60, 5)

if uploaded is None:
    st.info("Upload your extracted feature dataset.")
    st.stop()

df = load_feature_table(uploaded)
bundle = infer_dataset_schema(df)

X = bundle.df[bundle.numeric_cols].copy()
X = X.loc[:, X.nunique(dropna=False) > 1]
X = X.iloc[:, : min(max_features, X.shape[1])]
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True))

y = bundle.df[bundle.target_col].astype(str)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

if reducer == "PCA":
    embed = PCA(n_components=n_components, random_state=42).fit_transform(X_scaled)
else:
    perplexity = min(30, max(5, len(X_scaled) // 10))
    embed = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=42,
        init="pca",
        learning_rate="auto",
    ).fit_transform(X_scaled)

embed_cols = [f"dim_{i+1}" for i in range(n_components)]
vis_df = pd.DataFrame(embed, columns=embed_cols)
vis_df["target"] = y.values

if clustering == "KMeans":
    vis_df["cluster"] = KMeans(n_clusters=kmeans_k, random_state=42, n_init=10).fit_predict(X_scaled).astype(str)
elif clustering == "DBSCAN":
    vis_df["cluster"] = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit_predict(X_scaled).astype(str)
else:
    vis_df["cluster"] = "NA"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Samples", len(bundle.df))
c2.metric("Numeric features used", X.shape[1])
c3.metric("Classes", bundle.df[bundle.target_col].nunique(dropna=False))
c4.metric("Reducer", reducer)

st.subheader("Class summary")
class_df = bundle.df[bundle.target_col].astype(str).value_counts().rename_axis("class").reset_index(name="count")
class_df["percent"] = 100 * class_df["count"] / max(len(bundle.df), 1)
st.dataframe(class_df, use_container_width=True)

st.subheader("Reduced-dimension visualization")
if n_components == 2:
    fig = px.scatter(
        vis_df,
        x="dim_1",
        y="dim_2",
        color="target",
        symbol="cluster" if clustering != "None" else None,
        title=f"{reducer} embedding",
    )
else:
    fig = px.scatter_3d(
        vis_df,
        x="dim_1",
        y="dim_2",
        z="dim_3",
        color="target",
        symbol="cluster" if clustering != "None" else None,
        title=f"{reducer} embedding",
    )
st.plotly_chart(fig, use_container_width=True)

if clustering != "None":
    st.subheader("Cluster counts")
    cluster_counts = vis_df.groupby(["cluster", "target"]).size().reset_index(name="count")
    st.dataframe(cluster_counts, use_container_width=True)

st.subheader("Feature summary preview")
st.dataframe(bundle.df[[bundle.target_col] + X.columns[: min(20, len(X.columns))].tolist()].head(30), use_container_width=True)

st.session_state["q1_embedding"] = vis_df
