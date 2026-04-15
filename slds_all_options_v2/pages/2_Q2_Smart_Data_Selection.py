from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.spatial.distance import cdist
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from app_utils import (
    available_feature_families,
    dataset_status_caption,
    filter_feature_columns,
    log_run,
    require_active_dataset,
)

st.set_page_config(page_title="Q2 Smart Data Selection", page_icon="🎯", layout="wide")
st.title("🎯 Q2. Smart data selection and class-imbalance simulation")


def build_model(model_name: str, class_weight_mode: bool, random_state: int):
    class_weight = "balanced" if class_weight_mode else None

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=-1,
        )
    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=300,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=-1,
        )
    if model_name == "logreg":
        return LogisticRegression(
            max_iter=2000,
            random_state=random_state,
            class_weight=class_weight,
        )
    if model_name == "svm_rbf":
        return SVC(
            kernel="rbf",
            probability=True,
            random_state=random_state,
            class_weight=class_weight,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def minority_label_from_y(y: pd.Series) -> str:
    return y.astype(str).value_counts().idxmin()


def apply_random_oversampling(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    vc = y.value_counts()
    max_n = int(vc.max())
    rng = np.random.default_rng(random_state)

    X_parts = []
    y_parts = []
    for cls, cnt in vc.items():
        idx = np.where(y.values == cls)[0]
        if cnt < max_n:
            extra = rng.choice(idx, size=max_n - cnt, replace=True)
            full_idx = np.concatenate([idx, extra])
        else:
            full_idx = idx
        X_parts.append(X.iloc[full_idx])
        y_parts.append(y.iloc[full_idx])

    X_new = pd.concat(X_parts, axis=0).reset_index(drop=True)
    y_new = pd.concat(y_parts, axis=0).reset_index(drop=True)
    order = rng.permutation(len(X_new))
    return X_new.iloc[order].reset_index(drop=True), y_new.iloc[order].reset_index(drop=True)


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, minority_label: str) -> dict:
    y_true = y_true.astype(str)
    y_pred = pd.Series(y_pred).astype(str)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="binary", pos_label=minority_label, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary", pos_label=minority_label, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "minority_recall": float(recall_score(y_true, y_pred, average="binary", pos_label=minority_label, zero_division=0)),
    }


def metric_key_from_ui(name: str) -> str:
    return {
        "Macro F1": "macro_f1",
        "Weighted F1": "weighted_f1",
        "Minority recall": "minority_recall",
        "Balanced accuracy": "balanced_accuracy",
    }[name]


def distance_matrix(X: np.ndarray, metric: str) -> np.ndarray:
    if metric == "Euclidean":
        return euclidean_distances(X, X)
    if metric == "Cosine":
        return cosine_distances(X, X)
    if metric == "Mahalanobis":
        cov = np.cov(X, rowvar=False)
        cov += np.eye(cov.shape[0]) * 1e-6
        VI = np.linalg.pinv(cov)
        return cdist(X, X, metric="mahalanobis", VI=VI)
    raise ValueError(metric)


def select_seed_indices(X_train, y_train, seed_size, strategy, random_state, distance_metric):
    rng = np.random.default_rng(random_state)
    n = len(X_train)
    all_idx = np.arange(n)

    if strategy == "Random":
        return np.sort(rng.choice(all_idx, size=seed_size, replace=False))

    if strategy == "Stratified":
        chosen = []
        for cls in y_train.astype(str).unique():
            cls_idx = all_idx[y_train.astype(str).values == cls]
            take = max(1, int(round(seed_size * len(cls_idx) / n)))
            take = min(take, len(cls_idx))
            chosen.extend(rng.choice(cls_idx, size=take, replace=False).tolist())
        chosen = np.array(sorted(set(chosen)))
        if len(chosen) < seed_size:
            remaining = np.setdiff1d(all_idx, chosen)
            extra = rng.choice(remaining, size=seed_size - len(chosen), replace=False)
            chosen = np.sort(np.concatenate([chosen, extra]))
        return chosen[:seed_size]

    if strategy == "Diverse seed":
        Xn = StandardScaler().fit_transform(X_train)
        D = distance_matrix(Xn, distance_metric)
        first = int(rng.integers(0, n))
        chosen = [first]
        min_d = D[first].copy()
        for _ in range(1, seed_size):
            nxt = int(np.argmax(min_d))
            chosen.append(nxt)
            min_d = np.minimum(min_d, D[nxt])
        return np.array(sorted(set(chosen)))[:seed_size]

    raise ValueError(strategy)


def choose_next_indices(
    X_lab, y_lab, X_pool, y_pool, pool_idx, batch_size,
    strategy, model_name, class_weight_mode, random_state,
    distance_metric, hybrid_weight, minority_protection,
):
    rng = np.random.default_rng(random_state)
    pool_local = np.arange(len(X_pool))

    if strategy == "Random":
        chosen_local = rng.choice(pool_local, size=min(batch_size, len(pool_local)), replace=False)
        return pool_idx[np.sort(chosen_local)]

    if strategy == "Stratified random":
        chosen_local = []
        n_pool = len(X_pool)
        for cls in y_pool.astype(str).unique():
            cls_local = pool_local[y_pool.astype(str).values == cls]
            take = max(1, int(round(batch_size * len(cls_local) / n_pool)))
            take = min(take, len(cls_local))
            chosen_local.extend(rng.choice(cls_local, size=take, replace=False).tolist())
        chosen_local = np.array(sorted(set(chosen_local)))
        if len(chosen_local) < batch_size:
            remaining = np.setdiff1d(pool_local, chosen_local)
            extra = rng.choice(remaining, size=min(batch_size - len(chosen_local), len(remaining)), replace=False)
            chosen_local = np.sort(np.concatenate([chosen_local, extra]))
        return pool_idx[chosen_local[:batch_size]]

    scaler = StandardScaler()
    X_lab_n = scaler.fit_transform(X_lab)
    X_pool_n = scaler.transform(X_pool)

    if strategy == "Diversity sampling":
        X_all = np.vstack([X_lab_n, X_pool_n])
        D = distance_matrix(X_all, distance_metric)
        lab_idx = np.arange(len(X_lab_n))
        pool_start = len(X_lab_n)
        min_d = np.min(D[pool_start:, :][:, lab_idx], axis=1)
        order = np.argsort(-min_d)
        chosen_local = order[: min(batch_size, len(order))]
        return pool_idx[chosen_local]

    provisional = build_model(model_name, class_weight_mode=class_weight_mode, random_state=random_state)
    provisional.fit(X_lab, y_lab)

    if hasattr(provisional, "predict_proba"):
        prob = provisional.predict_proba(X_pool)
        if prob.shape[1] == 2:
            uncertainty = 1.0 - np.abs(prob[:, 1] - 0.5) * 2.0
        else:
            uncertainty = 1.0 - np.max(prob, axis=1)
    else:
        pred = provisional.decision_function(X_pool)
        uncertainty = -np.abs(pred)

    if strategy == "Uncertainty sampling":
        base_order = np.argsort(-uncertainty)
    elif strategy == "Hybrid: uncertainty + diversity":
        X_all = np.vstack([X_lab_n, X_pool_n])
        D = distance_matrix(X_all, distance_metric)
        lab_idx = np.arange(len(X_lab_n))
        pool_start = len(X_lab_n)
        div_score = np.min(D[pool_start:, :][:, lab_idx], axis=1)

        def norm01(v):
            v = np.asarray(v, dtype=float)
            lo, hi = np.min(v), np.max(v)
            return np.zeros_like(v) if hi - lo < 1e-12 else (v - lo) / (hi - lo)

        combo = hybrid_weight * norm01(uncertainty) + (1.0 - hybrid_weight) * norm01(div_score)
        base_order = np.argsort(-combo)
    else:
        raise ValueError(strategy)

    if minority_protection == "Off":
        return pool_idx[base_order[: min(batch_size, len(base_order))]]

    pool_labels = y_pool.astype(str).values
    minority = minority_label_from_y(pd.Series(pool_labels))

    if minority_protection == "Maintain class ratio":
        target_vc = y_lab.astype(str).value_counts(normalize=True)
        chosen = []
        remaining = list(base_order)
        for cls, frac in target_vc.items():
            cls_needed = max(1, int(round(batch_size * frac)))
            cls_candidates = [i for i in remaining if pool_labels[i] == cls]
            chosen.extend(cls_candidates[:cls_needed])
        chosen = list(dict.fromkeys(chosen))
        if len(chosen) < batch_size:
            for i in base_order:
                if i not in chosen:
                    chosen.append(i)
                if len(chosen) >= batch_size:
                    break
        return pool_idx[np.array(chosen[:batch_size])]

    if minority_protection == "Minimum minority quota":
        chosen = []
        minority_candidates = [i for i in base_order if pool_labels[i] == minority]
        quota = max(1, batch_size // 3)
        chosen.extend(minority_candidates[:quota])
        for i in base_order:
            if i not in chosen:
                chosen.append(i)
            if len(chosen) >= batch_size:
                break
        return pool_idx[np.array(chosen[:batch_size])]

    if minority_protection == "Minority-priority acquisition":
        minority_order = [i for i in base_order if pool_labels[i] == minority]
        other_order = [i for i in base_order if pool_labels[i] != minority]
        return pool_idx[np.array((minority_order + other_order)[:batch_size])]

    return pool_idx[base_order[: min(batch_size, len(base_order))]]


def run_single_sequential_experiment(
    df, feature_cols, target_col, test_fraction, training_subset_fraction,
    selection_strategy, acquisition_mode, batch_size, budget_mode,
    minority_handling, minority_protection, initial_seed_strategy,
    initial_seed_fraction, distance_metric, hybrid_weight,
    evaluation_metric_ui, baseline_model, random_state,
):
    X = df[feature_cols].copy().replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    y = df[target_col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=random_state, stratify=y
    )

    n_train = len(X_train)
    minority = minority_label_from_y(y_train)
    budget = max(5, int(round(n_train * training_subset_fraction))) if budget_mode == "Fraction of training set" else max(5, min(st.session_state.get("q2_num_samples_budget", 40), n_train))
    seed_size = min(max(2, int(round(n_train * initial_seed_fraction))), budget)

    seed_idx = select_seed_indices(X_train, y_train, seed_size, initial_seed_strategy, random_state, distance_metric)
    all_idx = np.arange(n_train)
    labeled_idx = np.array(seed_idx, dtype=int)
    step = 1 if acquisition_mode == "Single-sample acquisition" else batch_size

    class_weight_mode = minority_handling in {"Class-weighted training", "Oversampling + class-weighted training"}
    oversample_mode = minority_handling in {"Random oversampling", "Oversampling + class-weighted training"}
    eval_metric_key = metric_key_from_ui(evaluation_metric_ui)
    learning_rows = []

    while len(labeled_idx) < budget:
        pool_idx = np.setdiff1d(all_idx, labeled_idx)
        if len(pool_idx) == 0:
            break

        X_lab = X_train.iloc[labeled_idx].reset_index(drop=True)
        y_lab = y_train.iloc[labeled_idx].reset_index(drop=True)
        X_pool = X_train.iloc[pool_idx].reset_index(drop=True)
        y_pool = y_train.iloc[pool_idx].reset_index(drop=True)
        acquire_now = min(step, budget - len(labeled_idx), len(pool_idx))

        next_idx = choose_next_indices(
            X_lab, y_lab, X_pool, y_pool, pool_idx, acquire_now,
            selection_strategy, baseline_model, class_weight_mode,
            random_state, distance_metric, hybrid_weight, minority_protection,
        )
        labeled_idx = np.unique(np.concatenate([labeled_idx, next_idx]))

        X_cur = X_train.iloc[labeled_idx].reset_index(drop=True)
        y_cur = y_train.iloc[labeled_idx].reset_index(drop=True)
        X_fit, y_fit = apply_random_oversampling(X_cur, y_cur, random_state) if oversample_mode else (X_cur, y_cur)

        model = build_model(baseline_model, class_weight_mode, random_state)
        model.fit(X_fit, y_fit)
        pred = model.predict(X_test)
        metrics = compute_metrics(y_test, pred, minority)
        learning_rows.append({
            "acquired_size": len(labeled_idx),
            "strategy": selection_strategy,
            "metric": eval_metric_key,
            "metric_value": metrics[eval_metric_key],
            **metrics,
        })

    X_final = X_train.iloc[labeled_idx].reset_index(drop=True)
    y_final = y_train.iloc[labeled_idx].reset_index(drop=True)
    X_fit, y_fit = apply_random_oversampling(X_final, y_final, random_state) if oversample_mode else (X_final, y_final)
    model = build_model(baseline_model, class_weight_mode, random_state)
    model.fit(X_fit, y_fit)
    pred = model.predict(X_test)
    final_metrics = compute_metrics(y_test, pred, minority)

    subset_class_dist = y_final.value_counts().rename_axis("class").reset_index(name="count")
    subset_class_dist["percent"] = 100 * subset_class_dist["count"] / max(len(y_final), 1)

    summary = {
        "train_subset_size": int(len(y_final)),
        "budget": int(budget),
        "test_fraction": float(test_fraction),
        "training_subset_fraction": float(training_subset_fraction),
        "selection_strategy": selection_strategy,
        "baseline_model": baseline_model,
        "minority_handling": minority_handling,
        "minority_protection": minority_protection,
        "initial_seed_strategy": initial_seed_strategy,
        "initial_seed_fraction": float(initial_seed_fraction),
        "subset_class_distribution": subset_class_dist,
        **final_metrics,
    }
    return summary, pd.DataFrame(learning_rows)


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
    test_fraction = st.select_slider("Test fraction", options=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40], value=0.20)
    training_subset_fraction = st.select_slider("Training subset fraction", options=[0.20, 0.40, 0.60, 0.80, 1.00], value=0.40)
    baseline_model = st.selectbox("Baseline model", ["random_forest", "extra_trees", "logreg", "svm_rbf"], index=0)
    selection_strategy = st.selectbox("Selection strategy", ["Random", "Stratified random", "Diversity sampling", "Uncertainty sampling", "Hybrid: uncertainty + diversity"], index=1)
    sequential_mode = st.selectbox("Sequential acquisition mode", ["Single-sample acquisition", "Batch acquisition"], index=1)
    batch_size = st.select_slider("Batch size", options=[1, 5, 10, 20], value=10) if sequential_mode == "Batch acquisition" else 1
    budget_mode = st.selectbox("Acquisition budget mode", ["Fraction of training set", "Number of samples"], index=0)
    if budget_mode == "Number of samples":
        st.session_state["q2_num_samples_budget"] = st.number_input("Number of acquired training samples", min_value=5, max_value=10000, value=40, step=5)
    minority_handling = st.radio("Minority-class handling", ["None", "Random oversampling", "Class-weighted training", "Oversampling + class-weighted training"], index=2)
    minority_protection = st.selectbox("Minority protection in acquisition", ["Off", "Maintain class ratio", "Minimum minority quota", "Minority-priority acquisition"], index=1)
    initial_seed_strategy = st.selectbox("Initial seed set strategy", ["Random", "Stratified", "Diverse seed"], index=1)
    initial_seed_percent = st.select_slider("Initial seed size", options=[5, 10, 15, 20], value=10)
    initial_seed_fraction = initial_seed_percent / 100.0
    if selection_strategy in {"Diversity sampling", "Hybrid: uncertainty + diversity"} or initial_seed_strategy == "Diverse seed":
        distance_metric = st.selectbox("Distance metric", ["Euclidean", "Cosine", "Mahalanobis"], index=0)
    else:
        distance_metric = "Euclidean"
    hybrid_weight = st.slider("Hybrid weight", 0.0, 1.0, 0.5, 0.05, help="0 = pure diversity, 1 = pure uncertainty") if selection_strategy == "Hybrid: uncertainty + diversity" else 0.5
    evaluation_metric = st.selectbox("Evaluation metric", ["Macro F1", "Weighted F1", "Minority recall", "Balanced accuracy"], index=0)
    repeat_runs = st.select_slider("Repeat runs", options=[1, 3, 5, 10], value=3)
    output_mode = st.selectbox("Output mode", ["Learning curve", "Final metric table", "Both"], index=2)
    random_state = st.number_input("Random seed", 0, 9999, 42, 1)
    run_q2 = st.button("Run Q2 analysis", type="primary", use_container_width=True)

if run_q2:
    selected_cols = filter_feature_columns(bundle.numeric_cols, include_families, exclude_absolute)
    final_rows = []
    curve_rows = []

    for run_i in range(repeat_runs):
        seed_i = int(random_state) + run_i
        summary, learning_df = run_single_sequential_experiment(
            df=bundle.df,
            feature_cols=selected_cols,
            target_col=bundle.target_col,
            test_fraction=float(test_fraction),
            training_subset_fraction=float(training_subset_fraction),
            selection_strategy=selection_strategy,
            acquisition_mode=sequential_mode,
            batch_size=int(batch_size),
            budget_mode=budget_mode,
            minority_handling=minority_handling,
            minority_protection=minority_protection,
            initial_seed_strategy=initial_seed_strategy,
            initial_seed_fraction=float(initial_seed_fraction),
            distance_metric=distance_metric,
            hybrid_weight=float(hybrid_weight),
            evaluation_metric_ui=evaluation_metric,
            baseline_model=baseline_model,
            random_state=seed_i,
        )
        summary["run_seed"] = seed_i
        final_rows.append(summary)
        if not learning_df.empty:
            learning_df["run_seed"] = seed_i
            curve_rows.append(learning_df)

    final_df = pd.DataFrame([{k: v for k, v in row.items() if k != "subset_class_distribution"} for row in final_rows])
    metric_cols = ["accuracy", "precision", "recall", "macro_f1", "weighted_f1", "balanced_accuracy", "minority_recall"]
    agg_rows = {f"{c}_mean": final_df[c].mean() for c in metric_cols} | {f"{c}_std": final_df[c].std(ddof=0) for c in metric_cols}
    best_metric_key = metric_key_from_ui(evaluation_metric)
    subset_dist = final_rows[0]["subset_class_distribution"].copy() if final_rows else pd.DataFrame()

    curve_df = pd.concat(curve_rows, axis=0).reset_index(drop=True) if curve_rows else pd.DataFrame()
    curve_agg = (
        curve_df.groupby("acquired_size", as_index=False)["metric_value"].agg(["mean", "std"]).reset_index().rename(columns={"mean": "metric_mean", "std": "metric_std"})
        if not curve_df.empty else pd.DataFrame()
    )

    st.session_state["q2_results"] = final_df
    st.session_state["q2_curve_agg"] = curve_agg
    st.session_state["q2_summary"] = pd.DataFrame([agg_rows])
    st.session_state["q2_subset_dist"] = subset_dist
    st.session_state["q2_payload"] = {
        "caption": f"{selection_strategy} using {len(selected_cols)} selected features across {repeat_runs} run(s).",
        "best_metric_key": best_metric_key,
        "best_metric_mean": agg_rows[f"{best_metric_key}_mean"],
        "settings": {
            "selection_strategy": selection_strategy,
            "repeat_runs": repeat_runs,
            "output_mode": output_mode,
        },
    }
    log_run("Q2", {
        "families": ",".join(include_families),
        "exclude_absolute": exclude_absolute,
        "test_fraction": test_fraction,
        "training_subset_fraction": training_subset_fraction,
        "baseline_model": baseline_model,
        "selection_strategy": selection_strategy,
        "minority_handling": minority_handling,
        "minority_protection": minority_protection,
        "repeat_runs": repeat_runs,
    }, {
        "optimized_metric_mean": float(agg_rows[f"{best_metric_key}_mean"]),
        "n_runs": int(repeat_runs),
    })

final_df = st.session_state.get("q2_results")
if final_df is None:
    st.info("Adjust the sidebar settings and click **Run Q2 analysis**.")
    st.stop()

summary_df = st.session_state.get("q2_summary")
curve_agg = st.session_state.get("q2_curve_agg")
subset_dist = st.session_state.get("q2_subset_dist")
payload = st.session_state.get("q2_payload", {})

if payload.get("settings", {}).get("output_mode", output_mode) in {"Final metric table", "Both"}:
    st.subheader("Final metrics across runs")
    st.dataframe(final_df, use_container_width=True)
    if summary_df is not None:
        st.subheader("Aggregated summary (mean ± std)")
        show_cols = []
        for base in ["accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "minority_recall"]:
            show_cols.extend([f"{base}_mean", f"{base}_std"])
        st.dataframe(summary_df[show_cols], use_container_width=True)

if payload.get("settings", {}).get("output_mode", output_mode) in {"Learning curve", "Both"} and curve_agg is not None and not curve_agg.empty:
    st.subheader("Learning curve")
    fig_curve = px.line(curve_agg, x="acquired_size", y="metric_mean", markers=True, title=f"Performance vs acquired subset size ({payload.get('best_metric_key', 'metric')})")
    st.plotly_chart(fig_curve, use_container_width=True)

st.subheader("Selected subset class distribution")
if subset_dist is not None and not subset_dist.empty:
    st.dataframe(subset_dist, use_container_width=True)

if payload.get("caption"):
    st.caption(payload["caption"])

if payload.get("best_metric_mean") is not None:
    st.success(
        f"Best-performing current setup: {payload['settings']['selection_strategy']} with mean {payload['best_metric_key']} = {payload['best_metric_mean']:.4f} over {payload['settings']['repeat_runs']} run(s)."
    )

st.download_button("Download Q2 final metrics CSV", data=final_df.to_csv(index=False).encode("utf-8"), file_name="q2_final_metrics.csv", mime="text/csv")
if curve_agg is not None and not curve_agg.empty:
    st.download_button("Download Q2 learning curve CSV", data=curve_agg.to_csv(index=False).encode("utf-8"), file_name="q2_learning_curve.csv", mime="text/csv")
