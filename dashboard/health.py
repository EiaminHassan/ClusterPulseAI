"""
ClusterPulse AI - Node Health & Failure-Risk Page

Per-node drill-down: risk label, trend, and raw telemetry over time,
so an operator can see *why* a node was flagged, not just that it was.
"""

import streamlit as st
import pandas as pd

from . import charts

RISK_BADGE = {
    "Healthy": "🟢 Healthy",
    "Watch": "🟡 Watch",
    "At-Risk": "🔴 At-Risk",
}


def render(scored_df, health_summary):
    st.subheader("Node Health & Failure-Risk Detection")
    st.caption(
        "Isolation Forest anomaly detection over synthetic telemetry -- unsupervised, "
        "so no labeled failure data is required. Nodes are scored on thermal and power "
        "stability, not raw utilization, so idle nodes aren't mistaken for at-risk ones."
    )

    filter_cols = st.columns([1, 3])
    with filter_cols[0]:
        risk_filter = st.multiselect(
            "Filter by risk level",
            options=["Healthy", "Watch", "At-Risk"],
            default=["Watch", "At-Risk"],
        )

    display_df = health_summary.copy()
    if risk_filter:
        display_df = display_df[display_df["risk_label"].isin(risk_filter)]
    else:
        display_df = health_summary

    display_df = display_df.assign(
        Status=display_df["risk_label"].map(RISK_BADGE)
    )[["node_id", "Status", "risk_score", "risk_trend_delta", "temperature", "power_draw", "last_reading"]]
    display_df.columns = [
        "Node", "Status", "Risk Score", "Trend Δ", "Temp (°C)", "Power (W)", "Last Reading",
    ]

    st.dataframe(
        display_df,
        width='stretch',
        hide_index=True,
        column_config={
            "Risk Score": st.column_config.ProgressColumn(
                "Risk Score", min_value=0, max_value=1, format="%.2f"
            ),
        },
    )

    st.divider()

    node_ids = sorted(scored_df["node_id"].unique())
    default_node = (
        health_summary.sort_values("risk_score", ascending=False)["node_id"].iloc[0]
        if len(health_summary)
        else node_ids[0]
    )
    selected_node = st.selectbox(
        "Inspect a node's raw telemetry", node_ids, index=node_ids.index(default_node)
    )

    node_row = health_summary[health_summary["node_id"] == selected_node].iloc[0]
    badge_cols = st.columns(4)
    badge_cols[0].metric("Status", RISK_BADGE[node_row["risk_label"]])
    badge_cols[1].metric("Risk Score", f"{node_row['risk_score']:.2f}")
    badge_cols[2].metric("Temperature", f"{node_row['temperature']:.1f} °C")
    badge_cols[3].metric("Power Draw", f"{node_row['power_draw']:.0f} W")

    st.plotly_chart(
        charts.node_timeseries(scored_df, selected_node),
        width='stretch',
        config={"displayModeBar": False},
    )
