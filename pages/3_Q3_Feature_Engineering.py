from __future__ import annotations

import io

import pandas as pd
import plotly.express as px
import streamlit as st

from app_utils import add_unsupervised_features, build_feature_family_table, dataset_status_caption, feature_rankings, infer_dataset_schema, require_active_dataset

st.set_page_config(page_title="Q3 Feature Engineering", page_icon="🧩", layout="wide")
st.title("🧩 Q3. Feature engineering + unsupervised augmentation")

bundle = require_active_dataset()
st.caption(dataset_status_caption())

with st.sidebar:
    add_unsup = st.checkbox("Add unsupervised features", value=True)
    n_pca = st.slider("Unsupervised PCA components", 2, 6, 3)
    n_clusters = st.slider("Cluster features", 2, 8, 4)
    run_q3 = st.button("Run Q3 feature engineering", type="primary", use_container_width=True)

if run_q3:
    family_df = build_feature_family_table(bundle.numeric_cols)
    rank_df = feature_rankings(bundle.df, bundle.numeric_cols, bundle.target_col)
    aug_df = bundle.df.copy()
    unsup_cols = []
    if add_unsup:
        aug_df = add_unsupervised_features(aug_df, bundle.numeric_cols, n_pca=n_pca, n_clusters=n_clusters)
        aug_bundle = infer_dataset_schema(aug_df)
        unsup_cols = [c for c in aug_bundle.numeric_cols if c.startswith("unsup_")]
    st.session_state["q3_payload"] = {
        "family_counts": family_df["family"].value_counts().rename_axis("family").reset_index(name="count"),
        "rank_df": rank_df,
        "aug_df": aug_df,
        "unsup_cols": unsup_cols,
        "label_col": bundle.label_col,
    }
    st.session_state["q3_rank_table"] = rank_df
    st.session_state["q3_augmented_df"] = aug_df

payload = st.session_state.get("q3_payload")
if payload is None:
    st.info("Choose the Q3 settings in the sidebar and click **Run Q3 feature engineering**.")
    st.stop()

st.subheader("Current feature families")
st.dataframe(payload["family_counts"], use_container_width=True)

st.subheader("Top candidate features")
st.dataframe(payload["rank_df"].head(30), use_container_width=True)
fig = px.bar(payload["rank_df"].head(20), x="feature", y="mutual_info", color="family", title="Top 20 by mutual information")
fig.update_layout(xaxis_tickangle=-35)
st.plotly_chart(fig, use_container_width=True)

if payload["unsup_cols"]:
    st.subheader("Added unsupervised features")
    st.write(payload["unsup_cols"])
    emb_cols = [c for c in payload["unsup_cols"] if c.startswith("unsup_pca_")]
    if len(emb_cols) >= 2:
        emb = payload["aug_df"][emb_cols[:2]].copy()
        emb.columns = ["pc1", "pc2"]
        tmp = emb.copy()
        tmp[payload["label_col"]] = payload["aug_df"][payload["label_col"]].values
        fig2 = px.scatter(tmp, x="pc1", y="pc2", color=payload["label_col"], title="Unsupervised PCA augmentation view")
        st.plotly_chart(fig2, use_container_width=True)

csv_bytes = payload["aug_df"].to_csv(index=False).encode("utf-8")
st.download_button("Download augmented CSV", data=csv_bytes, file_name="q3_augmented_features.csv", mime="text/csv")

xlsx_buffer = io.BytesIO()
with pd.ExcelWriter(xlsx_buffer, engine="openpyxl") as writer:
    payload["aug_df"].to_excel(writer, index=False, sheet_name="features")
st.download_button(
    "Download augmented XLSX",
    data=xlsx_buffer.getvalue(),
    file_name="q3_augmented_features.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
