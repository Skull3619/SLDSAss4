from __future__ import annotations

import io

import pandas as pd
import plotly.express as px
import streamlit as st

from app_utils import (
    add_unsupervised_features,
    available_feature_families,
    build_feature_family_table,
    dataset_status_caption,
    feature_rankings,
    filter_feature_columns,
    infer_dataset_schema,
    log_run,
    require_active_dataset,
    top_feature_scatter_matrix,
)

st.set_page_config(page_title="Q3 Feature Engineering", page_icon="🧩", layout="wide")
st.title("🧩 Q3. Feature engineering + unsupervised augmentation")

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
    add_unsup = st.checkbox("Add unsupervised features", value=True)
    n_pca = st.slider("Unsupervised PCA components", 2, 6, 3)
    n_clusters = st.slider("Cluster features", 2, 8, 4)
    splom_n = st.slider("Scatter plot matrix top features", 3, 8, 5)
    run_q3 = st.button("Run Q3 feature engineering", type="primary", use_container_width=True)

if run_q3:
    selected_cols = filter_feature_columns(bundle.numeric_cols, include_families, exclude_absolute)
    family_df = build_feature_family_table(selected_cols)
    rank_df = feature_rankings(bundle.df, selected_cols, bundle.target_col)
    aug_df = bundle.df.copy()
    if add_unsup:
        aug_df = add_unsupervised_features(aug_df, selected_cols, n_pca=n_pca, n_clusters=n_clusters)
    splom_df = top_feature_scatter_matrix(bundle.df, rank_df, bundle.label_col, top_n=splom_n)
    st.session_state["q3_payload"] = {
        "family_df": family_df,
        "rank_df": rank_df,
        "aug_df": aug_df,
        "add_unsup": add_unsup,
        "splom_df": splom_df,
        "caption": f"Used {len(selected_cols)} features. Unsupervised augmentation={'ON' if add_unsup else 'OFF'}."
    }
    st.session_state["q3_rank_table"] = rank_df
    st.session_state["q3_augmented_df"] = aug_df
    log_run("Q3", {
        "families": ",".join(include_families),
        "exclude_absolute": exclude_absolute,
        "add_unsup": add_unsup,
        "n_pca": n_pca,
        "n_clusters": n_clusters,
    }, {"n_selected_features": len(selected_cols)})

payload = st.session_state.get("q3_payload")
if payload is None:
    st.info("Adjust the sidebar settings and click **Run Q3 feature engineering**.")
    st.stop()

st.subheader("Current feature families")
st.dataframe(payload["family_df"]["family"].value_counts().rename_axis("family").reset_index(name="count"), use_container_width=True)

st.subheader("Top candidate features")
st.dataframe(payload["rank_df"].head(30), use_container_width=True)
fig = px.bar(payload["rank_df"].head(20), x="feature", y="rank_score", color="family", title="Top 20 ranked features")
fig.update_layout(xaxis_tickangle=-35)
st.plotly_chart(fig, use_container_width=True)
st.caption(payload["caption"])

if not payload["splom_df"].empty:
    st.subheader("Scatter plot matrix of most significant features")
    dims = [c for c in payload["splom_df"].columns if c != bundle.label_col]
    fig_s = px.scatter_matrix(payload["splom_df"], dimensions=dims, color=bundle.label_col)
    fig_s.update_traces(diagonal_visible=False, showupperhalf=False)
    st.plotly_chart(fig_s, use_container_width=True)

if payload["add_unsup"]:
    aug_bundle = infer_dataset_schema(payload["aug_df"])
    st.subheader("Added unsupervised features")
    unsup_cols = [c for c in aug_bundle.numeric_cols if c.startswith("unsup_")]
    st.write(unsup_cols)
    emb_cols = [c for c in unsup_cols if c.startswith("unsup_pca_")]
    if len(emb_cols) >= 2:
        emb = payload["aug_df"][emb_cols[:2]].copy()
        emb.columns = ["pc1", "pc2"]
        tmp = emb.copy()
        tmp[bundle.label_col] = payload["aug_df"][bundle.label_col].values
        fig2 = px.scatter(tmp, x="pc1", y="pc2", color=bundle.label_col, title="Unsupervised PCA augmentation view")
        st.plotly_chart(fig2, use_container_width=True)

csv_bytes = payload["aug_df"].to_csv(index=False).encode("utf-8")
st.download_button("Download augmented CSV", data=csv_bytes, file_name="q3_augmented_features.csv", mime="text/csv")

xlsx_buffer = io.BytesIO()
with pd.ExcelWriter(xlsx_buffer, engine="openpyxl") as writer:
    payload["aug_df"].to_excel(writer, index=False, sheet_name="features")
st.download_button("Download augmented XLSX", data=xlsx_buffer.getvalue(), file_name="q3_augmented_features.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
