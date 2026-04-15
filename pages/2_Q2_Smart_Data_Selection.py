from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from app_utils import infer_dataset_schema, load_feature_table

st.set_page_config(page_title="Q2 Smart Data Selection", page_icon="🧠", layout="wide")
st.title("🧠 Q2. Smart Data Selection")

def oversample_minority(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    vc = y.value_counts()
    max_n = int(vc.max())
    parts_X = []
    parts_y = []
    rng = np.random.default_rng(random_state)

    for cls, count in vc.items():
        idx = np.where(y.values == cls)[0]
        if len(idx) == 0:
            continue
        if count < max_n:
            extra = rng.choice(idx, size=max_n - count, replace=True)
            full = np.concatenate([idx, extra])
        else:
            full = idx
        parts_X.append(X.iloc[full])
        parts_y.append(y.iloc[full])

    X_new = pd.concat(parts_X, axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)
    y_new = pd.concat(parts_y, axis=0).loc[X_new.index if isinstance(X_new.index, pd.Index) else :]

    # Rebuild y cleanly to match X_new order
    y_new = pd.Series(np.concatenate([p.values for p in parts_y]), name=y.name)
    shuffled = np.arange(len(y_new))
    rng.shuffle(shuffled)
    return X_new.iloc[shuffled].reset_index(drop=True), y_new.iloc[shuffled].reset_index(drop=True)


def evaluate_subset(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    train_idx: np.ndarray,
    oversample: bool,
    random_state: int,
    minority_label: str,
) -> dict:
    X_sub = X_train.iloc[train_idx].reset_index(drop=True)
    y_sub = y_train.iloc[train_idx].reset_index(drop=True)

    if oversample:
        X_sub, y_sub = oversample_minority(X_sub, y_sub, random_state=random_state)

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_sub, y_sub)
    pred = clf.predict(X_test)

    labels = sorted(pd.Series(y_test).astype(str).unique().tolist())
    y_test_s = y_test.astype(str)
    pred_s = pd.Series(pred).astype(str)

    return {
        "train_subset_size": int(len(train_idx)),
        "accuracy": float(accuracy_score(y_test_s, pred_s)),
        "macro_f1": float(f1_score(y_test_s, pred_s, average="macro")),
        "precision_macro": float(precision_score(y_test_s, pred_s, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test_s, pred_s, average="macro", zero_division=0)),
        "f1_infeasible": float(f1_score(y_test_s, pred_s, pos_label=minority_label, average="binary")),
        "precision_infeasible": float(precision_score(y_test_s, pred_s, pos_label=minority_label, average="binary", zero_division=0)),
        "recall_infeasible": float(recall_score(y_test_s, pred_s, pos_label=minority_label, average="binary", zero_division=0)),
    }


def random_subset(n: int, k: int, random_state: int) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(np.arange(n), size=k, replace=False))


def stratified_subset(y: pd.Series, k: int, random_state: int) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    idx_all = np.arange(len(y))
    classes = y.astype(str).unique().tolist()
    chosen = []
    for cls in classes:
        idx = idx_all[y.astype(str).values == cls]
        take = max(1, int(round(k * len(idx) / len(y))))
        take = min(take, len(idx))
        chosen.extend(rng.choice(idx, size=take, replace=False).tolist())
    chosen = np.array(sorted(set(chosen)))
    if len(chosen) < k:
        remaining = np.setdiff1d(idx_all, chosen)
        extra = rng.choice(remaining, size=k - len(chosen), replace=False)
        chosen = np.sort(np.concatenate([chosen, extra]))
    return chosen[:k]


def balanced_random_subset(y: pd.Series, k: int, random_state: int) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    idx_all = np.arange(len(y))
    vc = y.astype(str).value_counts()
    classes = vc.index.tolist()
    per_class = max(1, k // len(classes))
    chosen = []
    for cls in classes:
        idx = idx_all[y.astype(str).values == cls]
        take = min(per_class, len(idx))
        chosen.extend(rng.choice(idx, size=take, replace=False).tolist())
    chosen = np.array(sorted(set(chosen)))
    if len(chosen) < k:
        remaining = np.setdiff1d(idx_all, chosen)
        extra = rng.choice(remaining, size=min(k - len(chosen), len(remaining)), replace=False)
        chosen = np.sort(np.concatenate([chosen, extra]))
    return chosen[:k]


def diversity_subset(X: np.ndarray, k: int, random_state: int) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    first = int(rng.integers(0, n))
    chosen = [first]
    dists = np.linalg.norm(X - X[first], axis=1)
    for _ in range(1, k):
        nxt = int(np.argmax(dists))
        chosen.append(nxt)
        dists = np.minimum(dists, np.linalg.norm(X - X[nxt], axis=1))
    return np.array(sorted(set(chosen)))[:k]


def balanced_diversity_subset(X: np.ndarray, y: pd.Series, k: int, random_state: int) -> np.ndarray:
    classes = y.astype(str).unique().tolist()
    per_class = max(1, k // len(classes))
    idx_all = np.arange(len(y))
    chosen = []
    for cls in classes:
        idx = idx_all[y.astype(str).values == cls]
        if len(idx) == 0:
            continue
        sub = diversity_subset(X[idx], min(per_class, len(idx)), random_state=random_state)
        chosen.extend(idx[sub].tolist())
    chosen = np.array(sorted(set(chosen)))
    if len(chosen) < k:
        remaining = np.setdiff1d(idx_all, chosen)
        extra = remaining[: min(k - len(chosen), len(remaining))]
        chosen = np.sort(np.concatenate([chosen, extra]))
    return chosen[:k]


def uncertainty_subset(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    k: int,
    random_state: int,
) -> np.ndarray:
    idx_seed = stratified_subset(y_train, min(max(20, k // 4), len(y_train)), random_state=random_state)
    clf = RandomForestClassifier(
        n_estimators=250,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train.iloc[idx_seed], y_train.iloc[idx_seed])
    proba = clf.predict_proba(X_train)
    if proba.shape[1] == 2:
        uncertainty = 1.0 - np.abs(proba[:, 1] - 0.5) * 2.0
    else:
        uncertainty = -np.max(proba, axis=1)
    order = np.argsort(-uncertainty)
    return np.sort(order[:k])


def hybrid_subset(
    X_scaled: np.ndarray,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    k: int,
    random_state: int,
) -> np.ndarray:
    unc = uncertainty_subset(X_train, y_train, k, random_state=random_state)
    div = diversity_subset(X_scaled, k, random_state=random_state)
    merged = np.unique(np.concatenate([unc[: max(1, k // 2)], div[: max(1, k // 2)]]))
    if len(merged) < k:
        remaining = np.setdiff1d(np.arange(len(y_train)), merged)
        extra = remaining[: min(k - len(merged), len(remaining))]
        merged = np.concatenate([merged, extra])
    return np.sort(merged[:k])


with st.sidebar:
    uploaded = st.file_uploader("Upload feature dataset", type=["csv", "xlsx", "parquet"], key="q2_upload")
    test_size = st.slider("Test fraction", 0.1, 0.4, 0.2, 0.05)
    subset_fraction = st.slider("Training subset fraction", 0.1, 1.0, 0.4, 0.05)
    oversample = st.checkbox("Oversample minority class in training", value=False)
    random_state = st.number_input("Random seed", 0, 9999, 42, 1)

if uploaded is None:
    st.info("Upload your extracted feature dataset.")
    st.stop()

df = load_feature_table(uploaded)
bundle = infer_dataset_schema(df)

X = bundle.df[bundle.numeric_cols].copy()
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True))
y = bundle.df[bundle.target_col].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=int(random_state),
    stratify=y,
)

minority_label = y_train.value_counts().idxmin()
k = max(10, int(round(len(X_train) * subset_fraction)))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

strategies = {
    "random": random_subset(len(X_train), k, int(random_state)),
    "stratified": stratified_subset(y_train, k, int(random_state)),
    "balanced_random": balanced_random_subset(y_train, k, int(random_state)),
    "diversity": diversity_subset(X_train_scaled, k, int(random_state)),
    "balanced_diversity": balanced_diversity_subset(X_train_scaled, y_train, k, int(random_state)),
    "uncertainty": uncertainty_subset(X_train, y_train, k, int(random_state)),
    "hybrid": hybrid_subset(X_train_scaled, X_train, y_train, k, int(random_state)),
}

rows = []
for name, idx in strategies.items():
    metrics = evaluate_subset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        train_idx=idx,
        oversample=oversample,
        random_state=int(random_state),
        minority_label=minority_label,
    )
    metrics["strategy"] = name
    rows.append(metrics)

res = pd.DataFrame(rows).sort_values(["macro_f1", "f1_infeasible"], ascending=False).reset_index(drop=True)

st.subheader("Sampling benchmark")
st.dataframe(res, use_container_width=True)

st.subheader("Class summary")
class_df = y.value_counts().rename_axis("class").reset_index(name="count")
class_df["percent"] = 100 * class_df["count"] / max(len(y), 1)
st.dataframe(class_df, use_container_width=True)

chart_cols = [c for c in ["macro_f1", "f1_infeasible", "accuracy"] if c in res.columns]
st.bar_chart(res.set_index("strategy")[chart_cols])

st.download_button(
    "Download Q2 sampling results CSV",
    data=res.to_csv(index=False).encode("utf-8"),
    file_name="q2_sampling_results.csv",
    mime="text/csv",
)

st.session_state["q2_results"] = res
