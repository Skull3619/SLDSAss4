from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

LABEL_TO_TARGET = {"infeasible": 0, "feasible": 1}
ID_COLS = {"file_name", "file_path", "label", "target", "pred_target", "pred_label", "correct", "pipeline"}


@dataclass
class DatasetBundle:
    df: pd.DataFrame
    label_col: str
    target_col: str
    numeric_cols: list[str]


def load_feature_table(file_or_path) -> pd.DataFrame:
    if isinstance(file_or_path, (str, Path)):
        path = Path(file_or_path)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        if suffix == ".parquet":
            return pd.read_parquet(path)
        raise ValueError(f"Unsupported file format: {suffix}")
    name = getattr(file_or_path, "name", "uploaded")
    suffix = Path(name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_or_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_or_path)
    if suffix == ".parquet":
        return pd.read_parquet(file_or_path)
    raise ValueError(f"Unsupported upload format: {suffix}")


def infer_dataset_schema(df: pd.DataFrame) -> DatasetBundle:
    label_col = "label" if "label" in df.columns else None
    target_col = "target" if "target" in df.columns else None

    if target_col is None and label_col is not None:
        tmp = df.copy()
        tmp["target"] = tmp[label_col].map(LABEL_TO_TARGET)
        df = tmp
        target_col = "target"

    if label_col is None and target_col is not None:
        tmp = df.copy()
        tmp["label"] = tmp[target_col].map({0: "infeasible", 1: "feasible"})
        df = tmp
        label_col = "label"

    if label_col is None or target_col is None:
        candidates = [c for c in df.columns if df[c].nunique(dropna=True) == 2]
        if not candidates:
            raise ValueError("Could not infer label/target columns. Please include `label` and/or `target`.")
        c = candidates[0]
        if pd.api.types.is_numeric_dtype(df[c]):
            target_col = c
            if label_col is None:
                tmp = df.copy()
                tmp["label"] = tmp[target_col].map({0: "infeasible", 1: "feasible"})
                df = tmp
                label_col = "label"
        else:
            label_col = c
            if target_col is None:
                tmp = df.copy()
                le = LabelEncoder()
                tmp["target"] = le.fit_transform(tmp[label_col].astype(str))
                df = tmp
                target_col = "target"

    numeric_cols = [
        c for c in df.columns
        if c not in ID_COLS and pd.api.types.is_numeric_dtype(df[c]) and c != target_col
    ]
    if not numeric_cols:
        raise ValueError("No numeric feature columns were found.")
    return DatasetBundle(df=df, label_col=label_col, target_col=target_col, numeric_cols=numeric_cols)


def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum().values,
    })
    out["missing_pct"] = out["missing_count"] / max(len(df), 1)
    return out.sort_values(["missing_pct", "missing_count"], ascending=False).reset_index(drop=True)


def feature_family(col: str) -> str:
    c = col.lower()
    if c.startswith(("min_", "max_", "extent_", "bbox_", "centroid_", "std_", "x_q", "y_q", "z_q", "aspect_")):
        return "global_geometry"
    if c.startswith(("eig", "linearity", "planarity", "sphericity", "anisotropy", "curvature", "local_", "normal_")):
        return "shape_local_geometry"
    if c.startswith(("radial_", "d2_", "proj_", "slice_")):
        return "distribution_histograms"
    if "occupancy" in c or c.startswith("occ"):
        return "occupancy"
    if c.startswith(("nn_", "density_", "hull_", "compactness")):
        return "density_hull_spacing"
    if c.startswith("unsup_"):
        return "unsupervised"
    return "other"


def build_feature_family_table(cols: Iterable[str]) -> pd.DataFrame:
    return pd.DataFrame([{"feature": c, "family": feature_family(c)} for c in cols])


def standardize_frame(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    X = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    return StandardScaler().fit_transform(X)


def compute_embedding(df: pd.DataFrame, cols: list[str], method: str = "PCA", random_state: int = 42, perplexity: int = 20) -> pd.DataFrame:
    Xs = standardize_frame(df, cols)
    if len(df) < 2:
        return pd.DataFrame({"emb_1": [0.0] * len(df), "emb_2": [0.0] * len(df)})
    method = method.lower()
    if method == "pca":
        emb = PCA(n_components=2, random_state=random_state).fit_transform(Xs)
    else:
        perpl = min(max(5, perplexity), max(5, len(df) - 1))
        emb = TSNE(n_components=2, random_state=random_state, perplexity=perpl, init="pca", learning_rate="auto").fit_transform(Xs)
    return pd.DataFrame({"emb_1": emb[:, 0], "emb_2": emb[:, 1]})


def cluster_features(df: pd.DataFrame, cols: list[str], method: str = "kmeans", n_clusters: int = 4, eps: float = 0.9, min_samples: int = 5) -> np.ndarray:
    Xs = standardize_frame(df, cols)
    if method.lower() == "kmeans":
        n_clusters = min(max(2, n_clusters), len(df))
        return KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(Xs)
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Xs)


def add_unsupervised_features(df: pd.DataFrame, cols: list[str], n_pca: int = 3, n_clusters: int = 4, random_state: int = 42) -> pd.DataFrame:
    out = df.copy()
    Xs = standardize_frame(out, cols)
    if len(out) < 2:
        for i in range(n_pca):
            out[f"unsup_pca_{i+1}"] = 0.0
        out["unsup_cluster"] = 0
        out["unsup_cluster_dist"] = 0.0
        out["unsup_anomaly_score"] = 0.0
        return out

    k = min(max(1, n_pca), min(Xs.shape[0] - 1, Xs.shape[1]))
    pca = PCA(n_components=k, random_state=random_state)
    comps = pca.fit_transform(Xs)
    for i in range(comps.shape[1]):
        out[f"unsup_pca_{i+1}"] = comps[:, i]

    kk = min(max(2, n_clusters), len(out))
    km = KMeans(n_clusters=kk, random_state=random_state, n_init=10)
    labels = km.fit_predict(Xs)
    dists = np.linalg.norm(Xs - km.cluster_centers_[labels], axis=1)
    out["unsup_cluster"] = labels
    out["unsup_cluster_dist"] = dists

    if len(out) >= 10:
        iso = IsolationForest(random_state=random_state, contamination="auto")
        iso.fit(Xs)
        out["unsup_anomaly_score"] = -iso.score_samples(Xs)
    else:
        out["unsup_anomaly_score"] = 0.0
    return out


def feature_rankings(df: pd.DataFrame, cols: list[str], target_col: str, random_state: int = 42) -> pd.DataFrame:
    X = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df[target_col].astype(int)
    mi = mutual_info_classif(X, y, random_state=random_state)
    rf = RandomForestClassifier(n_estimators=300, random_state=random_state, class_weight="balanced")
    rf.fit(X, y)
    pos = df[y == 1]
    neg = df[y == 0]
    rows = []
    for i, c in enumerate(cols):
        try:
            _, p = mannwhitneyu(pos[c].dropna(), neg[c].dropna(), alternative="two-sided")
        except Exception:
            p = np.nan
        rows.append({
            "feature": c,
            "family": feature_family(c),
            "mutual_info": float(mi[i]),
            "rf_importance": float(rf.feature_importances_[i]),
            "mw_pvalue": p,
            "class1_mean": float(pos[c].mean()),
            "class0_mean": float(neg[c].mean()),
            "abs_mean_gap": abs(float(pos[c].mean()) - float(neg[c].mean())),
        })
    out = pd.DataFrame(rows)
    out["rank_score"] = out["mutual_info"] + out["rf_importance"] + out["abs_mean_gap"] / max(out["abs_mean_gap"].max(), 1e-12)
    return out.sort_values("rank_score", ascending=False).reset_index(drop=True)


def select_feature_mode(df: pd.DataFrame, cols: list[str], mode: str = "all") -> list[str]:
    groups = build_feature_family_table(cols)
    mode = mode.lower()
    if mode == "all":
        return cols
    if mode == "base":
        fams = {"global_geometry", "shape_local_geometry", "density_hull_spacing"}
    elif mode == "hist":
        fams = {"distribution_histograms", "occupancy"}
    elif mode == "rich":
        fams = {"global_geometry", "shape_local_geometry", "density_hull_spacing", "distribution_histograms", "occupancy"}
    elif mode == "unsup":
        fams = {"unsupervised"}
    else:
        fams = {mode}
    selected = groups.loc[groups["family"].isin(fams), "feature"].tolist()
    return selected or cols


def split_train_test(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(df, test_size=test_size, stratify=df[target_col], random_state=random_state)
