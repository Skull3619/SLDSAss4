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

from app_utils import infer_dataset_schema, load_feature_table

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


def _minmax(s: pd.Series) -> pd.Series:
    s = s.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


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
    if len(classes) != 2:
        raise ValueError("Feature ranking page currently expects a binary target.")

    c0, c1 = classes[0], classes[1]
    rows = []
    for i, col in enumerate(X.columns):
        x0 = X.loc[y == c0, col].values
        x1 = X.loc[y == c1, col].values
        try:
            mw_p = mannwhitneyu(x0, x1, alternative="two-sided").pvalue
        except Exception:
            mw_p = 1.0

        row = {
            "feature": col,
            "family": detect_feature_family(col),
            "mutual_info": float(mi[i]),
            "rf_importance": float(imp[i]),
            "mw_pvalue": float(mw_p),
            "class1_mean": float(np.mean(x1)),
            "class0_mean": float(np.mean(x0)),
            "abs_mean_gap": float(abs(np.mean(x1) - np.mean(x0))),
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    out["mw_signal"] = -np.log10(out["mw_pvalue"].clip(lower=1e-300).fillna(1.0))
    out["rank_score"] = (
        0.35 * out["mutual_info"].rank(pct=True)
        + 0.35 * out["rf_importance"].rank(pct=True)
        + 0.15 * out["abs_mean_gap"].rank(pct=True)
        + 0.15 * out["mw_signal"].rank(pct=True)
    )
    out = out.sort_values("rank_score", ascending=False).reset_index(drop=True)
    return out


with st.sidebar:
    uploaded = st.file_uploader("Upload feature dataset", type=["csv", "xlsx", "parquet"], key="q3_upload")
    n_pca = st.slider("Number of PCA latent features", 0, 10, 3, 1)
    n_clusters = st.slider("KMeans clusters for augmentation", 2, 12, 4, 1)
    add_anomaly_score = st.checkbox("Add anomaly score", value=True)
    test_size = st.slider("Test fraction", 0.1, 0.4, 0.2, 0.05, key="q3_test_fraction")
    random_state = st.number_input("Random seed", 0, 9999, 42, 1, key="q3_seed")

if uploaded is None:
    st.info("Upload your extracted feature dataset.")
    st.stop()

df = load_feature_table(uploaded)
bundle = infer_dataset_schema(df)

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
center_dists = np.linalg.norm(X_scaled - km.cluster_centers_[cluster_id], axis=1)
aug_df["cluster_center_dist"] = center_dists

if add_anomaly_score:
    iso = IsolationForest(random_state=int(random_state), contamination="auto")
    iso.fit(X_scaled)
    aug_df["anomaly_score"] = -iso.score_samples(X_scaled)

# Quick before/after benchmark
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
        {
            "dataset": "base_features",
            "accuracy": accuracy_score(y_te, pred_b),
            "macro_f1": f1_score(y_te, pred_b, average="macro"),
        },
        {
            "dataset": "augmented_features",
            "accuracy": accuracy_score(y_te, pred_a),
            "macro_f1": f1_score(y_te, pred_a, average="macro"),
        },
    ]
)

st.subheader("Top ranked features")
fig = px.bar(
    rank_df.head(20),
    x="feature",
    y="rank_score",
    color="family",
    title="Top 20 ranked features",
)
fig.update_layout(xaxis_tickangle=-35)
st.plotly_chart(fig, use_container_width=True)

st.dataframe(rank_df.head(50), use_container_width=True)

st.subheader("Augmentation benchmark")
st.dataframe(compare_df, use_container_width=True)
st.bar_chart(compare_df.set_index("dataset")[["macro_f1", "accuracy"]])

st.subheader("Augmented dataset preview")
st.dataframe(aug_df.head(30), use_container_width=True)

st.download_button(
    "Download augmented dataset CSV",
    data=aug_df.to_csv(index=False).encode("utf-8"),
    file_name="q3_augmented_features.csv",
    mime="text/csv",
)

workbook = io.BytesIO()
with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
    rank_df.to_excel(writer, index=False, sheet_name="ranked_features")
    compare_df.to_excel(writer, index=False, sheet_name="benchmark")
st.download_button(
    "Download Q3 summary workbook",
    data=workbook.getvalue(),
    file_name="q3_feature_engineering_summary.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.session_state["q3_rank_table"] = rank_df
st.session_state["q3_augmented_df"] = aug_df
