from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app_utils import infer_dataset_schema, load_feature_table
from pipeline_utils import evaluate_subset_strategy

st.set_page_config(page_title="Q2 Smart Data Selection", page_icon="🎯", layout="wide")
st.title("🎯 Q2. Smart data selection and class-imbalance simulation")
st.caption("Compare subset-selection strategies and quantify how they affect F1 on a held-out test set.")

with st.sidebar:
    uploaded = st.file_uploader("Upload feature dataset", type=["csv", "xlsx", "parquet"])
    budget = st.slider("Subset budget", 20, 300, 120, 10)
    baseline_model = st.selectbox("Baseline model", ["random_forest", "extra_trees", "logreg", "svm_rbf"])
    oversample = st.checkbox("Oversample minority class after selection", value=False)
    random_state = st.number_input("Random seed", 0, 9999, 42, 1)

if uploaded is None:
    st.info("Upload your extracted feature table.")
    st.stop()

df = load_feature_table(uploaded)
bundle = infer_dataset_schema(df)

strategies = st.multiselect(
    "Strategies",
    ["random", "stratified", "balanced_random", "diversity", "balanced_diversity", "uncertainty", "hybrid"],
    default=["random", "stratified", "balanced_random", "diversity", "balanced_diversity", "hybrid"],
)

if st.button("Run strategy comparison", type="primary"):
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
if res is not None:
    st.dataframe(res, use_container_width=True)
    fig = px.bar(res, x="strategy", y=["macro_f1", "f1_infeasible", "accuracy"], barmode="group", title="Subset strategy comparison")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("How class imbalance affects modeling")
st.markdown(
    """
The full dataset is about **300 feasible vs 200 infeasible**, so it is not extremely imbalanced, but it is imbalanced enough to hurt the minority class if you optimize only accuracy.

The most useful metrics here are:
- **macro F1** for overall balance
- **infeasible-class F1**
- **precision / recall for infeasible**
- confusion matrix, not accuracy alone

The stronger subset strategies are usually the ones that preserve both **class balance** and **shape diversity**.
"""
)

if res is not None:
    st.download_button("Download Q2 comparison CSV", data=res.to_csv(index=False).encode("utf-8"), file_name="q2_subset_strategy_comparison.csv", mime="text/csv")
