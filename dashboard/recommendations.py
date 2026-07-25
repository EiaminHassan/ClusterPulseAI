"""
ClusterPulse AI - Recommendations Page

The decision-oriented output the concept note centers on: plain-language,
prioritized actions rather than raw charts. This is the page an operator
should be able to read top-to-bottom and know exactly what to do next.
"""

import streamlit as st

PRIORITY_LABEL = {
    0: ("🔴", "Act now"),
    1: ("🟡", "Monitor / plan"),
    2: ("🔵", "Optimize"),
    3: ("🟢", "No action needed"),
}


def render(recommendations_df):
    st.subheader("Recommendations")
    st.caption(
        "Combines failure-risk and idle-resource findings into one prioritized, plain-language "
        "action list -- decision-oriented output, not just charts."
    )

    priority_filter = st.radio(
        "Show",
        options=["All", "Needs attention only"],
        horizontal=True,
        label_visibility="collapsed",
    )

    df = recommendations_df.copy()
    if priority_filter == "Needs attention only":
        df = df[df["priority"] <= 2]

    if len(df) == 0:
        st.success("No nodes currently need attention -- cluster is healthy and well-utilized.")
        return

    for _, row in df.iterrows():
        icon, label = PRIORITY_LABEL[row["priority"]]
        with st.container(border=True):
            header_cols = st.columns([3, 1])
            header_cols[0].markdown(f"### {icon} {row['node_id']}")
            header_cols[1].markdown(
                f"<div style='text-align:right; opacity:0.75; padding-top:0.6rem;'>{label}</div>",
                unsafe_allow_html=True,
            )
            if str(row.get("health_recommendation", "")).strip() and not row[
                "health_recommendation"
            ].startswith("OK"):
                st.markdown(f"🩺 {row['health_recommendation']}")
            if str(row.get("cost_recommendation", "")).strip() and not row[
                "cost_recommendation"
            ].startswith("OK"):
                st.markdown(f"💰 {row['cost_recommendation']}")
            if row["health_recommendation"].startswith("OK") and row[
                "cost_recommendation"
            ].startswith("OK"):
                st.markdown("✅ Operating normally -- no action needed.")
