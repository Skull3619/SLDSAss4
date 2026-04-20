from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from app_utils import LABEL_TO_TARGET, fit_unsupervised_augmenter, shared_train_test_split, transform_with_unsupervised_augmenter, select_feature_mode


@dataclass
class PipelineSpec:
    name: str
    feature_mode: str = "rich"
    scale: str = "standard"
    add_unsup: bool = False
    reducer: str = "none"
    model_name: str = "random_forest"
    use_balanced_weights: bool = False


MODEL_NAMES = [
    "logreg",
    "lda",
    "gaussian_nb",
    "knn",
    "svm_rbf",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "hist_gb",
    "adaboost",
    "mlp",
]
FEATURE_MODES = ["base", "hist", "rich"]
SCALES = ["none", "standard", "robust"]
REDUCERS = ["none", "variance", "pca95", "kbest20", "kbest30"]


def _make_classifier(name: str, balanced: bool = False, random_state: int = 42):
    if name == "logreg":
        return LogisticRegression(max_iter=1000, class_weight="balanced" if balanced else None, random_state=random_state)
    if name == "lda":
        return LinearDiscriminantAnalysis()
    if name == "gaussian_nb":
        return GaussianNB()
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=7)
    if name == "svm_rbf":
        return SVC(C=2.0, gamma="scale", probability=True, class_weight="balanced" if balanced else None, random_state=random_state)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=400, random_state=random_state, class_weight="balanced" if balanced else None)
    if name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=400, random_state=random_state, class_weight="balanced" if balanced else None)
    if name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=random_state)
    if name == "hist_gb":
        return HistGradientBoostingClassifier(random_state=random_state)
    if name == "adaboost":
        return AdaBoostClassifier(random_state=random_state)
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=random_state)
    raise ValueError(name)


def build_estimator(spec: PipelineSpec, random_state: int = 42) -> Pipeline:
    steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if spec.scale == "standard":
        steps.append(("scaler", StandardScaler()))
    elif spec.scale == "robust":
        steps.append(("scaler", RobustScaler()))

    if spec.reducer == "variance":
        steps.append(("reduce", VarianceThreshold(0.0)))
    elif spec.reducer == "pca95":
        steps.append(("reduce", PCA(n_components=0.95, random_state=random_state)))
    elif spec.reducer == "kbest20":
        steps.append(("reduce", SelectKBest(mutual_info_classif, k=20)))
    elif spec.reducer == "kbest30":
        steps.append(("reduce", SelectKBest(mutual_info_classif, k=30)))

    steps.append(("model", _make_classifier(spec.model_name, spec.use_balanced_weights, random_state=random_state)))
    return Pipeline(steps)


def generate_pipeline_specs() -> list[PipelineSpec]:
    specs: list[PipelineSpec] = []
    for feature_mode, scale, reducer, model_name, balanced in product(
        FEATURE_MODES, SCALES, REDUCERS, MODEL_NAMES, [False, True]
    ):
        if model_name in {"gaussian_nb", "lda"} and balanced:
            continue
        name = f"{feature_mode}|{scale}|{reducer}|{model_name}|bw={balanced}"
        specs.append(
            PipelineSpec(
                name=name,
                feature_mode=feature_mode,
                scale=scale,
                add_unsup=False,
                reducer=reducer,
                model_name=model_name,
                use_balanced_weights=balanced,
            )
        )
    for model_name in ["logreg", "svm_rbf", "random_forest", "extra_trees", "hist_gb", "mlp"]:
        for reducer in ["none", "pca95"]:
            specs.append(
                PipelineSpec(
                    name=f"rich+unsup|standard|{reducer}|{model_name}|bw=True",
                    feature_mode="rich",
                    scale="standard",
                    add_unsup=True,
                    reducer=reducer,
                    model_name=model_name,
                    use_balanced_weights=True,
                )
            )
    uniq = {}
    for s in specs:
        uniq[s.name] = s
    return list(uniq.values())


DEFAULT_PIPELINES = generate_pipeline_specs()


def _oversample_minority(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    counts = y.value_counts()
    if len(counts) < 2 or counts.iloc[0] == counts.iloc[-1]:
        return X, y
    minority_label = counts.idxmin()
    n_add = counts.max() - counts.min()
    idx = np.where(y.to_numpy() == minority_label)[0]
    rng = np.random.default_rng(random_state)
    sampled = rng.choice(idx, size=n_add, replace=True)
    X_aug = pd.concat([X, X.iloc[sampled]], ignore_index=True)
    y_aug = pd.concat([y, y.iloc[sampled]], ignore_index=True)
    return X_aug, y_aug


def _prepare_xy(df: pd.DataFrame, numeric_cols: list[str], target_col: str, spec: PipelineSpec, random_state: int = 42, augmenter: dict | None = None):
    work = df.copy()
    cols = list(numeric_cols)
    if spec.add_unsup:
        if augmenter is None:
            augmenter = fit_unsupervised_augmenter(work, cols, random_state=random_state)
        work = transform_with_unsupervised_augmenter(work, cols, augmenter)
        cols = [c for c in work.columns if c not in {"file_name", "file_path", "label", "target"} and pd.api.types.is_numeric_dtype(work[c])]
    feat_cols = select_feature_mode(work, cols, spec.feature_mode)
    feat_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(work[c])]
    X = work[feat_cols].replace([np.inf, -np.inf], np.nan)
    y = work[target_col].astype(int)
    meta = work[[c for c in ["file_name", "file_path", "label", target_col] if c in work.columns]].copy()
    return work, X, y, meta, feat_cols


def benchmark_pipelines(df: pd.DataFrame, numeric_cols: list[str], target_col: str, specs: Iterable[PipelineSpec], test_size: float = 0.2, oversample_train: bool = False, random_state: int = 42):
    train_df, test_df, train_idx, test_idx = shared_train_test_split(df, target_col, test_size=test_size, random_state=random_state, key="pipeline")
    rows = []
    for spec in specs:
        augmenter = fit_unsupervised_augmenter(train_df, numeric_cols, random_state=random_state) if spec.add_unsup else None
        _, X_train, y_train, _, feat_cols = _prepare_xy(train_df, numeric_cols, target_col, spec, random_state=random_state, augmenter=augmenter)
        _, X_test, y_test, _, _ = _prepare_xy(test_df, numeric_cols, target_col, spec, random_state=random_state, augmenter=augmenter)
        if oversample_train:
            X_train, y_train = _oversample_minority(X_train, y_train, random_state=random_state)
        est = build_estimator(spec, random_state=random_state)
        est.fit(X_train, y_train)
        pred = est.predict(X_test)
        rows.append({
            "pipeline": spec.name,
            "feature_mode": spec.feature_mode,
            "unsup": spec.add_unsup,
            "scale": spec.scale,
            "reducer": spec.reducer,
            "model": spec.model_name,
            "balanced_weights": spec.use_balanced_weights,
            "n_features": len(feat_cols),
            "macro_f1": float(f1_score(y_test, pred, average="macro")),
            "f1_infeasible": float(f1_score(y_test, pred, pos_label=LABEL_TO_TARGET["infeasible"])),
            "accuracy": float(accuracy_score(y_test, pred)),
            "weighted_f1": float(f1_score(y_test, pred, average="weighted")),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "precision_macro": float(precision_score(y_test, pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, pred, average="macro", zero_division=0)),
            "precision_infeasible": float(precision_score(y_test, pred, pos_label=LABEL_TO_TARGET["infeasible"], zero_division=0)),
            "recall_infeasible": float(recall_score(y_test, pred, pos_label=LABEL_TO_TARGET["infeasible"], zero_division=0)),
        })
    return pd.DataFrame(rows).sort_values(["macro_f1", "f1_infeasible", "accuracy"], ascending=False).reset_index(drop=True)


def fit_pipeline_with_diagnostics(df: pd.DataFrame, numeric_cols: list[str], target_col: str, spec: PipelineSpec, test_size: float = 0.2, oversample_train: bool = False, random_state: int = 42, threshold: float = 0.5):
    train_df, test_df, train_idx, test_idx = shared_train_test_split(df, target_col, test_size=test_size, random_state=random_state, key="pipeline")
    augmenter = fit_unsupervised_augmenter(train_df, numeric_cols, random_state=random_state) if spec.add_unsup else None
    _, X_train, y_train, train_meta, feat_cols = _prepare_xy(train_df, numeric_cols, target_col, spec, random_state=random_state, augmenter=augmenter)
    _, X_test, y_test, test_meta, _ = _prepare_xy(test_df, numeric_cols, target_col, spec, random_state=random_state, augmenter=augmenter)
    if oversample_train:
        X_train, y_train = _oversample_minority(X_train, y_train, random_state=random_state)
    est = build_estimator(spec, random_state=random_state)
    est.fit(X_train, y_train)
    pred = est.predict(X_test)

    prob_infeasible = np.full(len(X_test), np.nan)
    if hasattr(est, "predict_proba"):
        probs = est.predict_proba(X_test)
        classes = list(est.named_steps["model"].classes_)
        if LABEL_TO_TARGET["infeasible"] in classes:
            idx = classes.index(LABEL_TO_TARGET["infeasible"])
            prob_infeasible = probs[:, idx]
            pred = (prob_infeasible < threshold).astype(int) if idx == 1 else (prob_infeasible >= threshold).astype(int)
            # prob_infeasible is probability of class 0 when classes are [0,1]; use threshold on that directly
            pred = np.where(prob_infeasible >= threshold, LABEL_TO_TARGET["infeasible"], 1 - LABEL_TO_TARGET["infeasible"])
    elif hasattr(est, "decision_function"):
        scores = est.decision_function(X_test)
        prob_infeasible = 1 / (1 + np.exp(scores))
        pred = np.where(prob_infeasible >= threshold, LABEL_TO_TARGET["infeasible"], 1 - LABEL_TO_TARGET["infeasible"])

    cm = confusion_matrix(y_test, pred)
    report = pd.DataFrame(classification_report(y_test, pred, output_dict=True, zero_division=0)).T

    mis = test_meta.copy()
    mis["true_target"] = y_test.to_numpy()
    mis["pred_target"] = pred
    mis["true_label"] = mis["true_target"].map({0: "infeasible", 1: "feasible"})
    mis["pred_label"] = mis["pred_target"].map({0: "infeasible", 1: "feasible"})
    mis["correct"] = mis["true_target"] == mis["pred_target"]
    mis["prob_infeasible"] = prob_infeasible
    mis["confidence"] = np.where(mis["pred_target"].to_numpy() == 0, prob_infeasible, 1 - prob_infeasible)

    nbrs = NearestNeighbors(n_neighbors=min(5, len(X_train))).fit(X_train.fillna(0.0).to_numpy())
    distances, indices = nbrs.kneighbors(X_test.fillna(0.0).to_numpy())
    neighbor_rows = []
    train_meta_reset = train_meta.reset_index(drop=True)
    for test_i, (dvec, ivec) in enumerate(zip(distances, indices)):
        if mis.iloc[test_i]["correct"]:
            continue
        row = {
            "file_name": mis.iloc[test_i].get("file_name", f"test_{test_i}"),
            "true_label": mis.iloc[test_i]["true_label"],
            "pred_label": mis.iloc[test_i]["pred_label"],
            "confidence": mis.iloc[test_i]["confidence"],
        }
        for j, (dist, idx) in enumerate(zip(dvec, ivec), start=1):
            idx = int(idx)
            if 0 <= idx < len(train_meta_reset):
                meta_row = train_meta_reset.iloc[idx]
                row[f"nn{j}_file"] = meta_row.get("file_name", f"train_{idx}")
                row[f"nn{j}_label"] = meta_row.get("label", "")
            else:
                row[f"nn{j}_file"] = f"train_{idx}"
                row[f"nn{j}_label"] = "index_out_of_bounds"
            row[f"nn{j}_distance"] = dist
        neighbor_rows.append(row)
    neighbor_df = pd.DataFrame(neighbor_rows)

    importances = None
    model = est.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importances = pd.DataFrame({"feature": feat_cols, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    elif hasattr(model, "coef_"):
        coef = np.abs(model.coef_[0]) if np.ndim(model.coef_) > 1 else np.abs(model.coef_)
        importances = pd.DataFrame({"feature": feat_cols, "importance": coef}).sort_values("importance", ascending=False)

    metrics = {
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "f1_infeasible": float(f1_score(y_test, pred, pos_label=LABEL_TO_TARGET["infeasible"])),
        "accuracy": float(accuracy_score(y_test, pred)),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted")),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "precision_macro": float(precision_score(y_test, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, pred, average="macro", zero_division=0)),
        "precision_infeasible": float(precision_score(y_test, pred, pos_label=LABEL_TO_TARGET["infeasible"], zero_division=0)),
        "recall_infeasible": float(recall_score(y_test, pred, pos_label=LABEL_TO_TARGET["infeasible"], zero_division=0)),
    }

    return {
        "estimator": est,
        "metrics": metrics,
        "confusion_matrix": cm,
        "classification_report": report,
        "misclassified": mis.loc[~mis["correct"]].sort_values("confidence"),
        "neighbor_context": neighbor_df,
        "feature_importance": importances,
        "feature_columns": feat_cols,
    }


def _oversample_for_selection(X: np.ndarray, selected: list[int]) -> np.ndarray:
    if not selected:
        return np.ones(len(X))
    selected_X = X[selected]
    d = np.min(((X[:, None, :] - selected_X[None, :, :]) ** 2).sum(axis=2), axis=1)
    return np.sqrt(d)


def _distance_to_selected(X_pool: np.ndarray, X_selected: np.ndarray) -> np.ndarray:
    if len(X_selected) == 0:
        return np.ones(len(X_pool))
    d = np.min(((X_pool[:, None, :] - X_selected[None, :, :]) ** 2).sum(axis=2), axis=1)
    return np.sqrt(d)


def select_subset_indices(df: pd.DataFrame, numeric_cols: list[str], target_col: str, budget: int, strategy: str = "stratified", random_state: int = 42, seed_size: int = 20, model_name: str = "logreg") -> np.ndarray:
    budget = int(min(max(1, budget), len(df)))
    rng = np.random.default_rng(random_state)
    if budget >= len(df):
        return np.arange(len(df))
    strategy = strategy.lower()

    if strategy == "random":
        return np.sort(rng.choice(len(df), size=budget, replace=False))

    y = df[target_col].astype(int)
    if strategy == "stratified":
        sss = StratifiedShuffleSplit(n_splits=1, train_size=budget, random_state=random_state)
        idx = np.arange(len(df))
        subset, _ = next(sss.split(idx.reshape(-1, 1), y))
        return np.sort(subset)

    if strategy == "balanced_random":
        selected = []
        classes = sorted(y.unique())
        per_class = max(1, budget // len(classes))
        for c in classes:
            cls_idx = np.where(y.to_numpy() == c)[0]
            take = min(len(cls_idx), per_class)
            selected.extend(rng.choice(cls_idx, size=take, replace=False).tolist())
        remaining = np.setdiff1d(np.arange(len(df)), np.array(selected, dtype=int))
        if len(selected) < budget:
            extra = rng.choice(remaining, size=budget - len(selected), replace=False)
            selected.extend(extra.tolist())
        return np.sort(np.array(selected, dtype=int))

    X = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    X = (X - X.mean(axis=0)) / np.where(X.std(axis=0) == 0, 1.0, X.std(axis=0))

    if strategy == "diversity":
        start = int(rng.integers(len(df)))
        selected = [start]
        d = _oversample_for_selection(X, selected)
        while len(selected) < budget:
            nxt = int(np.argmax(d))
            if nxt in selected:
                remaining = np.setdiff1d(np.arange(len(df)), np.array(selected))
                nxt = int(rng.choice(remaining))
            selected.append(nxt)
            d = _oversample_for_selection(X, selected)
        return np.sort(np.array(selected))

    if strategy == "balanced_diversity":
        selected = []
        classes = sorted(y.unique())
        per_class = max(1, budget // len(classes))
        for c in classes:
            cls_idx = np.where(y.to_numpy() == c)[0]
            Xc = X[cls_idx]
            start = int(rng.choice(np.arange(len(cls_idx))))
            sel_local = [start]
            d = _oversample_for_selection(Xc, sel_local)
            while len(sel_local) < min(per_class, len(cls_idx)):
                nxt = int(np.argmax(d))
                if nxt in sel_local:
                    break
                sel_local.append(nxt)
                d = _oversample_for_selection(Xc, sel_local)
            selected.extend(cls_idx[sel_local].tolist())
        remaining = np.setdiff1d(np.arange(len(df)), np.array(selected, dtype=int))
        if len(selected) < budget:
            extra = rng.choice(remaining, size=budget - len(selected), replace=False)
            selected.extend(extra.tolist())
        return np.sort(np.array(selected, dtype=int))

    if strategy in {"uncertainty", "hybrid"}:
        seed_idx = select_subset_indices(df, numeric_cols, target_col, budget=min(seed_size, budget), strategy="balanced_random", random_state=random_state)
        selected = list(seed_idx)
        remaining = list(np.setdiff1d(np.arange(len(df)), np.array(selected)))
        while len(selected) < budget and remaining:
            current_df = df.iloc[selected].copy()
            spec = PipelineSpec(name="selector", feature_mode="rich", scale="standard", add_unsup=False, reducer="none", model_name=model_name, use_balanced_weights=True)
            est = build_estimator(spec, random_state=random_state)
            est.fit(current_df[numeric_cols].fillna(0.0), current_df[target_col].astype(int))
            X_rem = df.iloc[remaining][numeric_cols].fillna(0.0)
            if hasattr(est, "predict_proba"):
                p = est.predict_proba(X_rem)
                cls = list(est.named_steps["model"].classes_)
                idx = cls.index(1) if 1 in cls else 0
                uncertainty = 1 - np.abs(p[:, idx] - 0.5) * 2
            else:
                scores = est.decision_function(X_rem)
                probs = 1 / (1 + np.exp(-scores))
                uncertainty = 1 - np.abs(probs - 0.5) * 2
            if strategy == "hybrid":
                div = _distance_to_selected(X[remaining], X[selected])
                score = 0.6 * uncertainty + 0.4 * (div / max(div.max(), 1e-12))
            else:
                score = uncertainty
            pick_local = int(np.argmax(score))
            selected.append(remaining[pick_local])
            remaining.pop(pick_local)
        return np.sort(np.array(selected, dtype=int))

    return np.sort(rng.choice(len(df), size=budget, replace=False))


def evaluate_subset_strategy(df: pd.DataFrame, numeric_cols: list[str], target_col: str, budget: int, strategy: str, baseline_model: str = "random_forest", oversample_train: bool = False, random_state: int = 42):
    train_df, test_df = train_test_split(df, test_size=0.25, stratify=df[target_col], random_state=random_state)
    subset_idx = select_subset_indices(train_df.reset_index(drop=True), numeric_cols, target_col, budget, strategy, random_state=random_state, model_name=baseline_model)
    subset_df = train_df.reset_index(drop=True).iloc[subset_idx].copy()

    spec = PipelineSpec(name="baseline", feature_mode="rich", scale="standard", add_unsup=True, reducer="none", model_name=baseline_model, use_balanced_weights=True)
    _, X_train, y_train, _, _ = _prepare_xy(subset_df, numeric_cols, target_col, spec, random_state=random_state)
    _, X_test, y_test, _, _ = _prepare_xy(test_df, numeric_cols, target_col, spec, random_state=random_state)

    if oversample_train:
        X_train, y_train = _oversample_minority(X_train, y_train, random_state=random_state)

    est = build_estimator(spec, random_state=random_state)
    est.fit(X_train, y_train)
    pred = est.predict(X_test)
    return {
        "strategy": strategy,
        "budget": int(budget),
        "selected": int(len(subset_df)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "f1_infeasible": float(f1_score(y_test, pred, pos_label=LABEL_TO_TARGET["infeasible"])),
        "accuracy": float(accuracy_score(y_test, pred)),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted")),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "precision_macro": float(precision_score(y_test, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, pred, average="macro", zero_division=0)),
        "precision_infeasible": float(precision_score(y_test, pred, pos_label=LABEL_TO_TARGET["infeasible"], zero_division=0)),
        "recall_infeasible": float(recall_score(y_test, pred, pos_label=LABEL_TO_TARGET["infeasible"], zero_division=0)),
    }
