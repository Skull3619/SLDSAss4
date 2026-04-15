from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app_utils import infer_dataset_schema, load_feature_table, require_active_dataset

st.set_page_config(page_title="Q3 Feature Engineering", page_icon="🧱", layout="wide")
st.title("🧱 Q3. Feature Engineering")


def detect_feature_family(col: str) -> str:
    c = col.lower()
    if c.startswith(("occ_",)):
        return "occupancy"
    if c.startswith(("radial_", "d2_", "proj_", "hist_")):
        return "distribution_histograms"
    if c.startswith(("nn_", "hull_", "compactness", "density_", "bbox_", "extent_", "diag")):
        return "density_hull_spacing"
    if c.startswith(("eig", "linearity", "planarity", "sphericity", "anisotropy", "curvature", "eigentropy")):
        return "shape_local_geometry"
    if c.startswith(("centroid_", "min_", "max_", "x_q", "y_q", "z_q", "std_", "mean_", "var_")):
        return "global_geometry"
    return "other"


def build_rank_table(df: pd.DataFrame, numeric_cols: list[str], target_col: str, random_state: int) -> pd.DataFrame:
    X = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    y = df[target_col].astype(str)

    mi = mutual_info_classif(X, y, random_state=random_state)
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf.fit(X, y)
    imp = rf.feature_importances_

    classes = y.value_counts().index.tolist()
    c0, c1 = classes[0], classes[1]

    rows = []
    for i, col in enumerate(X.columns):
        x0 = X.loc[y == c0, col].values
        x1 = X.loc[y == c1, col].values
        try:
            mw_p = mannwhitneyu(x0, x1, alternative="two-sided").pvalue
        except Exception:
            mw_p = 1.0

        rows.append(
            {
                "feature": col,
                "family": detect_feature_family(col),
                "mutual_info": float(mi[i]),
                "rf_importance": float(imp[i]),
                "mw_pvalue": float(mw_p),
                "class1_mean": float(np.mean(x1)),
                "class0_mean": float(np.mean(x0)),
                "abs_mean_gap": float(abs(np.mean(x1) - np.mean(x0))),
            }
        )

    out = pd.DataFrame(rows)
    out["mw_signal"] = -np.log10(out["mw_pvalue"].clip(lower=1e-300).fillna(1.0))
    out["rank_score"] = (
        0.35 * out["mutual_info"].rank(pct=True)
        + 0.35 * out["rf_importance"].rank(pct=True)
        + 0.15 * out["abs_mean_gap"].rank(pct=True)
        + 0.15 * out["mw_signal"].rank(pct=True)
    )
    return out.sort_values("rank_score", ascending=False).reset_index(drop=True)


def pick_diverse_top_features(rank_df, n=5):
    chosen = []
    used_families = set()
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


bundle = require_active_dataset()

with st.sidebar:
    n_pca = st.slider("Number of PCA latent features", 0, 10, 3, 1)
    n_clusters = st.slider("KMeans clusters for augmentation", 2, 12, 4, 1)
    add_anomaly_score = st.checkbox("Add anomaly score", value=True)
    test_size = st.slider("Test fraction", 0.1, 0.4, 0.2, 0.05, key="q3_test_fraction")
    random_state = st.number_input("Random seed", 0, 9999, 42, 1, key="q3_seed")
    run_q3 = st.button("Run Q3 analysis", type="primary", use_container_width=True)

if run_q3:
    base = bundle.df.copy()
    X = base[bundle.numeric_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    y = base[bundle.target_col].astype(str)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rank_df = build_rank_table(base, bundle.numeric_cols, bundle.target_col, int(random_state))
    aug_df = base.copy()

    if n_pca > 0:
        pca = PCA(n_components=n_pca, random_state=int(random_state))
        Z = pca.fit_transform(X_scaled)
        for i in range(n_pca):
            aug_df[f"pca_{i+1}"] = Z[:, i]

    km = KMeans(n_clusters=n_clusters, random_state=int(random_state), n_init=10)
    cluster_id = km.fit_predict(X_scaled)
    aug_df["cluster_id"] = cluster_id.astype(int)
    aug_df["cluster_center_dist"] = np.linalg.norm(X_scaled - km.cluster_centers_[cluster_id], axis=1)

    if add_anomaly_score:
        iso = IsolationForest(random_state=int(random_state), contamination="auto")
        iso.fit(X_scaled)
        aug_df["anomaly_score"] = -iso.score_samples(X_scaled)

    X_before = X.copy()
    X_after = aug_df.drop(columns=[bundle.target_col], errors="ignore").select_dtypes(include=[np.number]).copy()

    X_tr_b, X_te_b, y_tr, y_te = train_test_split(
        X_before, y, test_size=test_size, random_state=int(random_state), stratify=y
    )
    X_tr_a, X_te_a, _, _ = train_test_split(
        X_after, y, test_size=test_size, random_state=int(random_state), stratify=y
    )

    clf_b = RandomForestClassifier(n_estimators=300, random_state=int(random_state), class_weight="balanced", n_jobs=-1)
    clf_b.fit(X_tr_b, y_tr)
    pred_b = clf_b.predict(X_te_b)

    clf_a = RandomForestClassifier(n_estimators=300, random_state=int(random_state), class_weight="balanced", n_jobs=-1)
    clf_a.fit(X_tr_a, y_tr)
    pred_a = clf_a.predict(X_te_a)

    compare_df = pd.DataFrame(
        [
            {"dataset": "base_features", "accuracy": accuracy_score(y_te, pred_b), "macro_f1": f1_score(y_te, pred_b, average="macro")},
            {"dataset": "augmented_features", "accuracy": accuracy_score(y_te, pred_a), "macro_f1": f1_score(y_te, pred_a, average="macro")},
        ]
    )

    st.session_state["q3_rank_table"] = rank_df
    st.session_state["q3_augmented_df"] = aug_df
    st.session_state["q3_compare_df"] = compare_df

rank_df = st.session_state.get("q3_rank_table")
aug_df = st.session_state.get("q3_augmented_df")
compare_df = st.session_state.get("q3_compare_df")

if rank_df is None:
    st.info("Adjust the sidebar settings and click **Run Q3 analysis**.")
    st.stop()

st.subheader("Top ranked features")
fig = px.bar(rank_df.head(20), x="feature", y="rank_score", color="family", title="Top 20 ranked features")
fig.update_layout(xaxis_tickangle=-35)
st.plotly_chart(fig, use_container_width=True)

st.dataframe(rank_df.head(50), use_container_width=True)

st.subheader("Augmentation benchmark")
st.dataframe(compare_df, use_container_width=True)
st.bar_chart(compare_df.set_index("dataset")[["macro_f1", "accuracy"]])

st.subheader("Class-wise distributions of top features")
dist_plot = st.radio("Distribution plot type", ["Box", "Violin"], horizontal=True, key="q3_dist")
top_feats = rank_df["feature"].head(5).tolist()
for feat in top_feats:
    if dist_plot == "Box":
        f = px.box(bundle.df, x=bundle.label_col, y=feat, color=bundle.label_col, points="outliers", title=f"{feat} by class")
    else:
        f = px.violin(bundle.df, x=bundle.label_col, y=feat, color=bundle.label_col, box=True, points="outliers", title=f"{feat} by class")
    st.plotly_chart(f, use_container_width=True)

st.subheader("Correlation heatmap of top features")
corr_feats = rank_df["feature"].head(10).tolist()
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

st.subheader("Parallel coordinates of diverse top features")
diverse_feats = pick_diverse_top_features(rank_df, n=5)
par_df = bundle.df[diverse_feats + [bundle.target_col]].copy()
color_series = bundle.df[bundle.target_col].astype("category").cat.codes
fig_par = px.parallel_coordinates(
    par_df,
    dimensions=diverse_feats,
    color=color_series,
    title="Parallel coordinates of diverse top features",
)
st.plotly_chart(fig_par, use_container_width=True)

st.subheader("Selected feature-pair relationship")
pair_feats = rank_df["feature"].head(12).tolist()
col1, col2 = st.columns(2)
with col1:
    x_feat = st.selectbox("X feature", pair_feats, index=0, key="q3_pair_x")
with col2:
    y_feat = st.selectbox("Y feature", pair_feats, index=min(1, len(pair_feats)-1), key="q3_pair_y")
fig_pair = px.scatter(
    bundle.df,
    x=x_feat,
    y=y_feat,
    color=bundle.label_col,
    opacity=0.7,
    title=f"{x_feat} vs {y_feat}",
)
st.plotly_chart(fig_pair, use_container_width=True)

st.subheader("Scatter plot matrix of significant features")
show_scatter_matrix = st.checkbox("Show scatter plot matrix", value=False, key="q3_show_spm")
if show_scatter_matrix:
    spm_feats = diverse_feats[:4]
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
    "Download augmented dataset CSV",
    data=aug_df.to_csv(index=False).encode("utf-8"),
    file_name="q3_augmented_features.csv",
    mime="text/csv",
)
