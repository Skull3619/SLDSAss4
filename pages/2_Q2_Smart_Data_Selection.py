from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app_utils import dataset_status_caption, require_active_dataset
from pipeline_utils import evaluate_subset_strategy

st.set_page_config(page_title="Q2 Smart Data Selection", page_icon="🎯", layout="wide")
st.title("🎯 Q2. Smart data selection and class-imbalance simulation")

bundle = require_active_dataset()
st.caption(dataset_status_caption())

with st.sidebar:
    budget = st.slider("Subset budget", 20, 300, 120, 10)
    baseline_model = st.selectbox("Baseline model", ["random_forest", "extra_trees", "logreg", "svm_rbf"])
    oversample = st.checkbox("Oversample minority class after selection", value=False)
    random_state = st.number_input("Random seed", 0, 9999, 42, 1)
    strategies = st.multiselect(
        "Strategies",
        ["random", "stratified", "balanced_random", "diversity", "balanced_diversity", "uncertainty", "hybrid"],
        default=["random", "stratified", "balanced_random", "diversity", "balanced_diversity", "hybrid"],
    )
    run_q2 = st.button("Run strategy comparison", type="primary", use_container_width=True)

if run_q2:
    rows = []
    for s in strategies:
        rows.append(
            evaluate_subset_strategy(
                bundle.df,
                bundle.numeric_cols,
                bundle.target_col,
                budget=budget,
                strategy=s,
                baseline_model=baseline_model,
                oversample_train=oversample,
                random_state=int(random_state),
            )
        )
    res = pd.DataFrame(rows).sort_values(["macro_f1", "f1_infeasible"], ascending=False).reset_index(drop=True)
    st.session_state["q2_results"] = res

res = st.session_state.get("q2_results")
if res is None:
    st.info("Choose the Q2 settings in the sidebar and click **Run strategy comparison**.")
    st.stop()

st.dataframe(res, use_container_width=True)
fig = px.bar(res, x="strategy", y=["macro_f1", "f1_infeasible", "accuracy"], barmode="group", title="Subset strategy comparison")
st.plotly_chart(fig, use_container_width=True)

st.download_button("Download Q2 comparison CSV", data=res.to_csv(index=False).encode("utf-8"), file_name="q2_subset_strategy_comparison.csv", mime="text/csv")
