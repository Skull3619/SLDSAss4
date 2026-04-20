from __future__ import annotations

import io

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import SVC

from app_utils import (
    available_feature_families,
    build_feature_family_table,
    dataset_status_caption,
    feature_rankings,
    filter_feature_columns,
    fit_unsupervised_augmenter,
    infer_dataset_schema,
    log_run,
    require_active_dataset,
    shared_train_test_split,
    top_feature_scatter_matrix,
    transform_with_unsupervised_augmenter,
)

st.set_page_config(page_title="Q3 Feature Engineering", page_icon="🧩", layout="wide")
st.title("🧩 Q3. Feature engineering + unsupervised augmentation")


def build_eval_model(name: str, random_state: int = 42):
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, random_state=random_state, class_weight="balanced")
    if name == "logreg":
        return LogisticRegression(max_iter=1500, random_state=random_state, class_weight="balanced")
    if name == "svm_rbf":
        return SVC(kernel="rbf", probability=True, random_state=random_state, class_weight="balanced")
    if name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=300, random_state=random_state, class_weight="balanced")
    raise ValueError(name)


def metric_row(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


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
    clustering_mode = st.selectbox("Clustering mode", ["KMeans", "Off"], index=0)
    n_pca = st.slider("Unsupervised PCA components", 0, 6, 3)
    n_clusters = st.slider("Cluster features", 0, 8, 4)
    eval_model = st.selectbox("Evaluation model", ["random_forest", "extra_trees", "logreg", "svm_rbf"], index=0)
    test_size = st.slider("Test fraction", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random seed", 0, 9999, 42, 1)
    splom_n = st.slider("Scatter plot matrix top features", 3, 8, 5)
    run_q3 = st.button("Run Q3 feature engineering", type="primary", use_container_width=True)

if run_q3:
    selected_cols = filter_feature_columns(bundle.numeric_cols, include_families, exclude_absolute)
    family_df = build_feature_family_table(selected_cols)
    rank_df = feature_rankings(bundle.df, selected_cols, bundle.target_col)

    # For export / visualization of the full dataset, keep backward-compatible augmentation.
    if add_unsup:
        full_augmenter = fit_unsupervised_augmenter(
            bundle.df,
            selected_cols,
            n_pca=n_pca,
            n_clusters=(0 if clustering_mode == "Off" else n_clusters),
            random_state=int(random_state),
        )
        aug_df = transform_with_unsupervised_augmenter(bundle.df, selected_cols, full_augmenter)
        if clustering_mode == "Off":
            aug_df["unsup_cluster"] = 0
            aug_df["unsup_cluster_dist"] = 0.0
    else:
        aug_df = bundle.df.copy()
        full_augmenter = None

    # Leakage-safe evaluation: fit augmentation on train only, then transform train/test.
    train_df, test_df, train_idx, test_idx = shared_train_test_split(
        bundle.df,
        bundle.target_col,
        test_size=float(test_size),
        random_state=int(random_state),
        key="q3",
    )
    X_train_base = train_df[selected_cols].replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
    X_test_base = test_df[selected_cols].replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
    y_train = train_df[bundle.target_col].astype(int)
    y_test = test_df[bundle.target_col].astype(int)

    model_base = build_eval_model(eval_model, random_state=int(random_state))
    model_base.fit(X_train_base, y_train)
    pred_base = model_base.predict(X_test_base)

    compare_rows = [{"dataset": "base_features", **metric_row(y_test, pred_base)}]
    pca_var_df = pd.DataFrame(columns=["component", "explained_variance_ratio"])
    cluster_sizes_df = pd.DataFrame(columns=["cluster", "count"])
    anomaly_by_class_df = pd.DataFrame(columns=[bundle.label_col, "mean", "std", "median"])

    if add_unsup:
        train_augmenter = fit_unsupervised_augmenter(
            train_df,
            selected_cols,
            n_pca=n_pca,
            n_clusters=(0 if clustering_mode == "Off" else n_clusters),
            random_state=int(random_state),
        )
        train_aug = transform_with_unsupervised_augmenter(train_df, selected_cols, train_augmenter)
        test_aug = transform_with_unsupervised_augmenter(test_df, selected_cols, train_augmenter)
        if clustering_mode == "Off":
            train_aug["unsup_cluster"] = 0
            train_aug["unsup_cluster_dist"] = 0.0
            test_aug["unsup_cluster"] = 0
            test_aug["unsup_cluster_dist"] = 0.0

        train_feat_cols = [
            c for c in train_aug.columns
            if c not in {"file_name", "file_path", "label", "target"} and pd.api.types.is_numeric_dtype(train_aug[c])
        ]
        X_train_aug = train_aug[train_feat_cols].replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
        X_test_aug = test_aug[train_feat_cols].replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)

        model_aug = build_eval_model(eval_model, random_state=int(random_state))
        model_aug.fit(X_train_aug, y_train)
        pred_aug = model_aug.predict(X_test_aug)
        compare_rows.append({"dataset": "augmented_features", **metric_row(y_test, pred_aug)})

        pca_var_df = pd.DataFrame({
            "component": [f"PC{i+1}" for i in range(len(train_augmenter["explained_variance_ratio"]))],
            "explained_variance_ratio": train_augmenter["explained_variance_ratio"],
        })
        cluster_sizes_df = pd.DataFrame({
            "cluster": list(train_augmenter["cluster_sizes"].keys()),
            "count": list(train_augmenter["cluster_sizes"].values()),
        }).sort_values("cluster")
        anomaly_tmp = train_aug[[bundle.label_col, "unsup_anomaly_score"]].copy()
        anomaly_by_class_df = anomaly_tmp.groupby(bundle.label_col)["unsup_anomaly_score"].agg(["mean", "std", "median"]).reset_index()

    compare_df = pd.DataFrame(compare_rows)
    splom_df = top_feature_scatter_matrix(bundle.df, rank_df, bundle.label_col, top_n=splom_n)

    st.session_state["q3_payload"] = {
        "family_df": family_df,
        "rank_df": rank_df,
        "aug_df": aug_df,
        "add_unsup": add_unsup,
        "splom_df": splom_df,
        "compare_df": compare_df,
        "pca_var_df": pca_var_df,
        "cluster_sizes_df": cluster_sizes_df,
        "anomaly_by_class_df": anomaly_by_class_df,
        "caption": f"Used {len(selected_cols)} features. Unsupervised augmentation={'ON' if add_unsup else 'OFF'} (fit on training only for evaluation).",
    }
    st.session_state["q3_rank_table"] = rank_df
    st.session_state["q3_augmented_df"] = aug_df
    st.session_state["q3_compare_df"] = compare_df
    log_run("Q3", {
        "families": ",".join(include_families),
        "exclude_absolute": exclude_absolute,
        "add_unsup": add_unsup,
        "n_pca": n_pca,
        "n_clusters": n_clusters,
        "clustering_mode": clustering_mode,
        "test_size": test_size,
        "eval_model": eval_model,
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

st.subheader("Leakage-safe augmentation benchmark")
st.dataframe(payload["compare_df"], use_container_width=True)
fig_cmp = px.bar(
    payload["compare_df"],
    x="dataset",
    y=["macro_f1", "weighted_f1", "balanced_accuracy", "accuracy"],
    barmode="group",
    title="Base vs augmented features",
)
st.plotly_chart(fig_cmp, use_container_width=True)

if not payload["pca_var_df"].empty:
    st.subheader("PCA variance explained (training fit)")
    st.dataframe(payload["pca_var_df"], use_container_width=True)
    fig_p = px.bar(payload["pca_var_df"], x="component", y="explained_variance_ratio", title="Explained variance ratio")
    st.plotly_chart(fig_p, use_container_width=True)

if not payload["cluster_sizes_df"].empty:
    st.subheader("Cluster sizes (training fit)")
    st.dataframe(payload["cluster_sizes_df"], use_container_width=True)

if not payload["anomaly_by_class_df"].empty:
    st.subheader("Anomaly score by class (training fit)")
    st.dataframe(payload["anomaly_by_class_df"], use_container_width=True)

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
    payload["compare_df"].to_excel(writer, index=False, sheet_name="benchmark")
    payload["pca_var_df"].to_excel(writer, index=False, sheet_name="pca_variance")
    payload["cluster_sizes_df"].to_excel(writer, index=False, sheet_name="cluster_sizes")
    payload["anomaly_by_class_df"].to_excel(writer, index=False, sheet_name="anomaly_by_class")
st.download_button("Download augmented XLSX", data=xlsx_buffer.getvalue(), file_name="q3_augmented_features.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
