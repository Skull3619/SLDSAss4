# --- Q1 visuals in report ---
q1_payload = st.session_state.get("q1_payload")
if q1_payload is not None:
    st.subheader("Q1 report visuals")

    fig_q1_rank = px.bar(
        q1_payload["rank_df"].head(20),
        x="feature",
        y="rank_score",
        color="family",
        title="Q1 Top 20 ranked features",
    )
    fig_q1_rank.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(fig_q1_rank, use_container_width=True)

# --- Q3 visuals in report ---
q3_rank = st.session_state.get("q3_rank_table")
q3_compare = st.session_state.get("q3_compare_df")

if q3_rank is not None:
    st.subheader("Q3 report visuals")

    fig_q3_rank = px.bar(
        q3_rank.head(20),
        x="feature",
        y="rank_score",
        color="family",
        title="Q3 Top 20 ranked features",
    )
    fig_q3_rank.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(fig_q3_rank, use_container_width=True)

    corr_feats = q3_rank["feature"].head(10).tolist()
    corr_df = bundle.df[corr_feats].corr(numeric_only=True)
    fig_q3_corr = px.imshow(
        corr_df,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Q3 Correlation heatmap of top features",
    )
    fig_q3_corr.update_layout(height=700)
    st.plotly_chart(fig_q3_corr, use_container_width=True)

if q3_compare is not None:
    st.dataframe(q3_compare, use_container_width=True)
