from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app_utils import available_feature_families, dataset_status_caption, filter_feature_columns, log_run, require_active_dataset
from pipeline_utils import evaluate_subset_strategy

st.set_page_config(page_title="Q2 Smart Data Selection", page_icon="🎯", layout="wide")
st.title("🎯 Q2. Smart data selection and class-imbalance simulation")

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
    budget = st.slider("Subset budget", 20, 300, 120, 10)
    baseline_model = st.selectbox("Baseline model", ["random_forest", "extra_trees", "logreg", "svm_rbf"])
    compare_oversample = st.checkbox("Compare oversampling OFF vs ON", value=True)
    oversample = st.checkbox("Oversample minority class after selection", value=False)
    random_state = st.number_input("Random seed", 0, 9999, 42, 1)
    strategies = st.multiselect(
        "Strategies",
        ["random", "stratified", "balanced_random", "diversity", "balanced_diversity", "uncertainty", "hybrid"],
        default=["random", "stratified", "balanced_random", "diversity", "balanced_diversity", "hybrid"],
    )
    run_q2 = st.button("Run strategy comparison", type="primary", use_container_width=True)

if run_q2:
    selected_cols = filter_feature_columns(bundle.numeric_cols, include_families, exclude_absolute)
    rows = []
    oversample_modes = [False, True] if compare_oversample else [oversample]
    for os_flag in oversample_modes:
        for s in strategies:
            row = evaluate_subset_strategy(
                bundle.df,
                selected_cols,
                bundle.target_col,
                budget=budget,
                strategy=s,
                baseline_model=baseline_model,
                oversample_train=os_flag,
                random_state=int(random_state),
            )
            row["oversample_train"] = os_flag
            rows.append(row)
    res = pd.DataFrame(rows).sort_values(["macro_f1", "f1_infeasible"], ascending=False).reset_index(drop=True)
    st.session_state["q2_results"] = res
    st.session_state["q2_payload"] = {
        "caption": f"Compared {len(strategies)} strategies using {len(selected_cols)} selected features.",
        "selected_cols": selected_cols,
    }
    log_run("Q2", {
        "families": ",".join(include_families),
        "exclude_absolute": exclude_absolute,
        "budget": budget,
        "baseline_model": baseline_model,
        "compare_oversample": compare_oversample,
    }, {"n_rows": len(res)})

res = st.session_state.get("q2_results")
if res is None:
    st.info("Adjust the sidebar settings and click **Run strategy comparison**.")
    st.stop()

st.dataframe(res, use_container_width=True)
fig = px.bar(res, x="strategy", y=["macro_f1", "f1_infeasible", "accuracy"], color="oversample_train", barmode="group", title="Subset strategy comparison")
st.plotly_chart(fig, use_container_width=True)
payload = st.session_state.get("q2_payload", {})
if payload.get("caption"):
    st.caption(payload["caption"])

st.download_button("Download Q2 comparison CSV", data=res.to_csv(index=False).encode("utf-8"), file_name="q2_subset_strategy_comparison.csv", mime="text/csv")
