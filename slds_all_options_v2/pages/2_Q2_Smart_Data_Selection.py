from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app_utils import infer_dataset_schema, load_feature_table, require_active_dataset

st.set_page_config(page_title="Q2 Smart Data Selection", page_icon="🧠", layout="wide")
st.title("🧠 Q2. Smart Data Selection")


def oversample_minority(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
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
    y_new = pd.concat(parts_y, axis=0).reset_index(drop=True)
    return X_new, y_new


def evaluate_subset(X_train, y_train, X_test, y_test, train_idx, oversample, random_state, minority_label):
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


def random_subset(n, k, random_state):
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(np.arange(n), size=k, replace=False))


def stratified_subset(y, k, random_state):
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


bundle = require_active_dataset()

with st.sidebar:
    test_size = st.slider("Test fraction", 0.1, 0.4, 0.2, 0.05)
    subset_fraction = st.slider("Training subset fraction", 0.1, 1.0, 0.4, 0.05)
    oversample = st.checkbox("Oversample minority class in training", value=False)
    random_state = st.number_input("Random seed", 0, 9999, 42, 1)
    run_q2 = st.button("Run Q2 analysis", type="primary", use_container_width=True)

if run_q2:
    X = bundle.df[bundle.numeric_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    y = bundle.df[bundle.target_col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=int(random_state), stratify=y
    )

    minority_label = y_train.value_counts().idxmin()
    k = max(10, int(round(len(X_train) * subset_fraction)))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    strategies = {
        "random": random_subset(len(X_train), k, int(random_state)),
        "stratified": stratified_subset(y_train, k, int(random_state)),
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
    st.session_state["q2_results"] = res

res = st.session_state.get("q2_results")
if res is None:
    st.info("Adjust the sidebar settings and click **Run Q2 analysis**.")
    st.stop()

st.subheader("Sampling benchmark")
st.dataframe(res, use_container_width=True)

metric_to_plot = st.selectbox(
    "Metric to plot",
    ["macro_f1", "f1_infeasible", "accuracy", "recall_infeasible", "precision_infeasible"],
    index=0,
)

fig = px.bar(
    res,
    x="strategy",
    y=metric_to_plot,
    color="strategy",
    title=f"{metric_to_plot} by subset strategy",
)
fig.update_layout(showlegend=False, xaxis_tickangle=-20)
st.plotly_chart(fig, use_container_width=True)

st.download_button(
    "Download Q2 sampling results CSV",
    data=res.to_csv(index=False).encode("utf-8"),
    file_name="q2_sampling_results.csv",
    mime="text/csv",
)
