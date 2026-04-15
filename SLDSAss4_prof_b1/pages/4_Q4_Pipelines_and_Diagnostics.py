from __future__ import annotations

import io

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

from app_utils import available_feature_families, dataset_status_caption, filter_feature_columns, get_run_history_df, log_run, require_active_dataset
from pipeline_utils import DEFAULT_PIPELINES, benchmark_pipelines, fit_pipeline_with_diagnostics

st.set_page_config(page_title="Q4 Pipelines and Diagnostics", page_icon="🧪", layout="wide")
st.title("🧪 Q4. Many pipelines, benchmark table, and diagnostics")

try:
    bundle = require_active_dataset()
except RuntimeError as e:
    st.info(str(e))
    st.stop()

st.caption(dataset_status_caption())
all_families = available_feature_families(bundle.numeric_cols)

catalog = pd.DataFrame([s.__dict__ for s in DEFAULT_PIPELINES])
with st.sidebar:
    include_families = st.multiselect("Feature families", all_families, default=all_families)
    exclude_absolute = st.checkbox("Exclude absolute-position features", value=False)
    test_size = st.slider("Test fraction", 0.1, 0.4, 0.2, 0.05)
    oversample = st.checkbox("Oversample minority class in training", value=False)
    random_state = st.number_input("Random seed", 0, 9999, 42, 1)
    objective_metric = st.selectbox("Sort benchmark by", ["macro_f1", "f1_infeasible", "recall_infeasible", "accuracy"])
    selected_models = st.multiselect("Restrict to models", sorted(catalog["model_name"].unique().tolist()), default=["logreg", "svm_rbf", "random_forest", "extra_trees", "hist_gb", "lda", "gaussian_nb"])
    filtered_catalog = catalog[catalog["model_name"].isin(selected_models)].copy()
    max_pipelines = st.slider("Number of pipelines to run", 1, max(1, len(filtered_catalog)), min(20, max(1, len(filtered_catalog))), 1)
    threshold = st.slider("Infeasible decision threshold", 0.1, 0.9, 0.5, 0.05)
    run_q4 = st.button("Run benchmark", type="primary", use_container_width=True)

st.subheader("Pipeline catalog")
st.write(f"Total available pipeline combinations in full catalog: **{len(catalog)}**")
st.write(f"Pipeline combinations matching selected models: **{len(filtered_catalog)}**")
st.write(f"Pipelines that will actually run with current setting: **{min(len(filtered_catalog), max_pipelines)}**")
st.dataframe(filtered_catalog.head(200), use_container_width=True)

selected_cols = filter_feature_columns(bundle.numeric_cols, include_families, exclude_absolute)
filtered_specs = [s for s in DEFAULT_PIPELINES if s.model_name in selected_models][:max_pipelines]
if run_q4:
    res = benchmark_pipelines(bundle.df, selected_cols, bundle.target_col, specs=filtered_specs, test_size=test_size, oversample_train=oversample, random_state=int(random_state))
    res = res.sort_values(objective_metric, ascending=False).reset_index(drop=True)
    st.session_state["q4_results"] = res
    st.session_state.pop("q4_detail", None)
    st.session_state["q4_payload"] = {"objective_metric": objective_metric, "selected_cols": selected_cols, "caption": f"Benchmark sorted by {objective_metric} using {len(selected_cols)} features."}
    log_run("Q4", {
        "families": ",".join(include_families),
        "exclude_absolute": exclude_absolute,
        "objective_metric": objective_metric,
        "selected_models": ",".join(selected_models),
        "max_pipelines": max_pipelines,
        "threshold": threshold,
    }, {"n_selected_features": len(selected_cols), "n_pipelines": len(filtered_specs)})

res = st.session_state.get("q4_results")
if res is None:
    st.info("Adjust the sidebar settings and click **Run benchmark**.")
    st.stop()

st.subheader("Benchmark results")
st.dataframe(res, use_container_width=True)
fig = px.bar(res.head(20), x="pipeline", y=["macro_f1", "f1_infeasible", "accuracy"], barmode="group", title="Top pipelines")
fig.update_layout(xaxis_tickangle=-40)
st.plotly_chart(fig, use_container_width=True)
q4_payload = st.session_state.get("q4_payload", {})
if q4_payload.get("caption"):
    st.caption(q4_payload["caption"])

available_names = [s.name for s in filtered_specs if s.name in res["pipeline"].tolist()]
best_name = st.selectbox("Detailed diagnostics for pipeline", available_names)
if st.button("Run selected pipeline diagnostics", use_container_width=True):
    spec_lookup = {s.name: s for s in filtered_specs}
    detail = fit_pipeline_with_diagnostics(bundle.df, selected_cols, bundle.target_col, spec_lookup[best_name], test_size=test_size, oversample_train=oversample, random_state=int(random_state), threshold=float(threshold))
    st.session_state["q4_detail"] = detail

detail = st.session_state.get("q4_detail")
if detail is None:
    st.info("Choose a pipeline and click **Run selected pipeline diagnostics**.")
    st.stop()

st.subheader("Confusion matrix")
cm = detail["confusion_matrix"]
z_text = [[str(v) for v in row] for row in cm]
cm_fig = ff.create_annotated_heatmap(z=cm, x=["Pred infeasible", "Pred feasible"], y=["True infeasible", "True feasible"], annotation_text=z_text, colorscale="Blues", showscale=True)
st.plotly_chart(cm_fig, use_container_width=True)

st.subheader("Metrics")
st.write(detail["metrics"])
st.dataframe(detail["classification_report"], use_container_width=True)

st.subheader("Misclassified samples")
st.dataframe(detail["misclassified"].head(100), use_container_width=True)

st.subheader("Nearest-neighbor context for misclassified samples")
if detail["neighbor_context"] is not None and not detail["neighbor_context"].empty:
    st.dataframe(detail["neighbor_context"].head(50), use_container_width=True)

st.subheader("Top feature importances / coefficients")
if detail["feature_importance"] is not None:
    st.dataframe(detail["feature_importance"].head(30), use_container_width=True)
    fig2 = px.bar(detail["feature_importance"].head(20), x="feature", y="importance", title="Top 20 important features")
    fig2.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(fig2, use_container_width=True)

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
st.download_button("Download benchmark workbook", data=workbook.getvalue(), file_name="q4_pipeline_benchmark.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
