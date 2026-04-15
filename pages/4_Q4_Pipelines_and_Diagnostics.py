from __future__ import annotations

import io

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

from app_utils import infer_dataset_schema, load_feature_table
from pipeline_utils import DEFAULT_PIPELINES, benchmark_pipelines, fit_pipeline_with_diagnostics

st.set_page_config(page_title="Q4 Pipelines and Diagnostics", page_icon="🧪", layout="wide")
st.title("🧪 Q4. Many pipelines, benchmark table, and diagnostics")
st.caption("Generate 50+ pipelines, run at least 10, compare F1 scores, and inspect misclassified samples.")

with st.sidebar:
    uploaded = st.file_uploader("Upload feature dataset", type=["csv", "xlsx", "parquet"])
    test_size = st.slider("Test fraction", 0.1, 0.4, 0.2, 0.05)
    oversample = st.checkbox("Oversample minority class in training", value=False)
    random_state = st.number_input("Random seed", 0, 9999, 42, 1)
    max_pipelines = st.slider("Number of pipelines to run", 10, min(120, len(DEFAULT_PIPELINES)), 20, 5)

if uploaded is None:
    st.info("Upload your extracted feature table.")
    st.stop()

df = load_feature_table(uploaded)
bundle = infer_dataset_schema(df)

catalog = pd.DataFrame([s.__dict__ for s in DEFAULT_PIPELINES])
st.subheader("Pipeline catalog")
st.write(f"Total available pipeline combinations: **{len(catalog)}**")
st.dataframe(catalog.head(60), use_container_width=True)

selected_models = st.multiselect(
    "Restrict to models",
    sorted(catalog["model_name"].unique().tolist()),
    default=["logreg", "svm_rbf", "random_forest", "extra_trees", "hist_gb", "lda", "gaussian_nb"],
)

filtered_specs = [s for s in DEFAULT_PIPELINES if s.model_name in selected_models][:max_pipelines]

if st.button("Run benchmark", type="primary"):
    res = benchmark_pipelines(
        bundle.df,
        bundle.numeric_cols,
        bundle.target_col,
        specs=filtered_specs,
        test_size=test_size,
        oversample_train=oversample,
        random_state=int(random_state),
    )
    st.session_state["q4_results"] = res

res = st.session_state.get("q4_results")
if res is not None:
    st.subheader("Benchmark results")
    st.dataframe(res, use_container_width=True)
    fig = px.bar(res.head(20), x="pipeline", y=["macro_f1", "f1_infeasible", "accuracy"], barmode="group", title="Top pipelines")
    fig.update_layout(xaxis_tickangle=-40)
    st.plotly_chart(fig, use_container_width=True)

    best_name = st.selectbox("Detailed diagnostics for pipeline", res["pipeline"].tolist())
    spec_lookup = {s.name: s for s in filtered_specs}
    detail = fit_pipeline_with_diagnostics(
        bundle.df,
        bundle.numeric_cols,
        bundle.target_col,
        spec_lookup[best_name],
        test_size=test_size,
        oversample_train=oversample,
        random_state=int(random_state),
    )

    st.subheader("Confusion matrix")
    cm = detail["confusion_matrix"]
    z_text = [[str(v) for v in row] for row in cm]
    cm_fig = ff.create_annotated_heatmap(
        z=cm,
        x=["Pred infeasible", "Pred feasible"],
        y=["True infeasible", "True feasible"],
        annotation_text=z_text,
        colorscale="Blues",
        showscale=True,
    )
    st.plotly_chart(cm_fig, use_container_width=True)

    st.subheader("Metrics")
    st.write(detail["metrics"])
    st.dataframe(detail["classification_report"], use_container_width=True)

    st.subheader("Misclassified samples")
    st.dataframe(detail["misclassified"].head(100), use_container_width=True)

    st.subheader("Nearest-neighbor context for misclassified samples")
    if detail["neighbor_context"] is not None and not detail["neighbor_context"].empty:
        st.dataframe(detail["neighbor_context"].head(50), use_container_width=True)
    else:
        st.info("No nearest-neighbor context was generated.")

    st.subheader("Top feature importances / coefficients")
    if detail["feature_importance"] is not None:
        st.dataframe(detail["feature_importance"].head(30), use_container_width=True)
        fig2 = px.bar(detail["feature_importance"].head(20), x="feature", y="importance", title="Top 20 important features")
        fig2.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("This pipeline does not expose direct feature importances.")

    st.download_button("Download benchmark CSV", data=res.to_csv(index=False).encode("utf-8"), file_name="q4_pipeline_benchmark.csv", mime="text/csv")

    workbook = io.BytesIO()
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        res.to_excel(writer, index=False, sheet_name="benchmark")
        detail["classification_report"].to_excel(writer, sheet_name="report")
        detail["misclassified"].to_excel(writer, index=False, sheet_name="misclassified")
        if detail["neighbor_context"] is not None and not detail["neighbor_context"].empty:
            detail["neighbor_context"].to_excel(writer, index=False, sheet_name="nn_context")
        if detail["feature_importance"] is not None:
            detail["feature_importance"].to_excel(writer, index=False, sheet_name="feature_importance")
    st.download_button(
        "Download benchmark workbook",
        data=workbook.getvalue(),
        file_name="q4_pipeline_benchmark.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.subheader("Q5 deployment note")
st.markdown(
    """
Use this Streamlit app with a **feature table**, not the full 9 GB raw dataset.
That keeps deployment practical and avoids upload-memory failures.
"""
)
