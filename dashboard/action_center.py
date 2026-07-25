"""
ClusterPulse AI - Action Center Page

Displays the output of the Action Ranking & Conflict Resolver: one unified,
ranked action list where every Health-vs-Cost disagreement has already been
resolved by the explainable "Safety First" rule. This is the page that
guarantees the operator never receives two contradictory instructions for
the same node.
"""

import streamlit as st
import pandas as pd

STATUS_BADGE = {
    "Healthy": "🟢 Healthy",
    "Watch": "🟡 Watch",
    "At-Risk": "🔴 At-Risk",
}


def render(action_df, summary):
    st.subheader("Action Center — Ranked, Conflict-Resolved Actions")
    st.caption(
        "The Health module and Cost module score every node independently and can "
        "disagree. This layer collects both recommendations, scores each node with a "
        "transparent Value Score, flags disagreements, resolves them with one explainable "
        "rule (\"Safety First\": risk wins above a threshold, cost savings win below it), "
        "and produces a single list an operator can act on without contradiction."
    )

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Total Nodes Ranked", summary["total_actions"])
    kpi_cols[1].metric("Conflicts Found", summary["total_conflicts_found"])
    kpi_cols[2].metric("Conflicts Resolved", summary["total_conflicts_resolved"])
    kpi_cols[3].metric(
        "Highest Priority",
        summary["highest_priority_node"] or "—",
        help=f"Final action: {summary['highest_priority_action']}" if summary["highest_priority_node"] else None,
    )

    st.divider()

    show_conflicts_only = st.checkbox("Show conflicts only", value=False)
    df = action_df.copy()
    if show_conflicts_only:
        df = df[df["Conflict"]]

    if len(df) == 0:
        st.success("No conflicts to resolve — Health and Cost modules agree on every node.")
        return

    display_df = df.copy()
    display_df["Health Status"] = display_df["Health Status"].map(STATUS_BADGE)
    display_df["Conflict"] = display_df["Conflict"].map({True: "⚠️ Conflict", False: "—"})

    st.markdown("**Ranked action list**")
    st.dataframe(
        display_df[
            [
                "Rank", "Node", "Health Status", "Risk Score", "Idle Status",
                "Idle % of Window", "Estimated Saving", "Value Score",
                "Conflict", "Final Action", "Reason",
            ]
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "Value Score": st.column_config.ProgressColumn(
                "Value Score", min_value=0, max_value=1, format="%.2f"
            ),
            "Risk Score": st.column_config.ProgressColumn(
                "Risk Score", min_value=0, max_value=1, format="%.2f"
            ),
            "Estimated Saving": st.column_config.NumberColumn("Estimated Saving", format="$%.2f"),
        },
    )

    conflicts = df[df["Conflict"]]
    if len(conflicts):
        st.divider()
        st.markdown("**Conflict detail — how each disagreement was resolved**")
        for _, row in conflicts.iterrows():
            with st.container(border=True):
                st.markdown(f"#### ⚠️ {row['Node']}")
                col_a, col_b, col_c = st.columns(3)
                col_a.markdown(f"**Health module said:**  \n{row['Health Action']}")
                col_b.markdown(f"**Cost module said:**  \n{row['Cost Action']}")
                col_c.markdown(f"**Final decision:**  \n✅ {row['Final Action']}")
                st.caption(f"Reason: {row['Reason']}")
