"""
ClusterPulse AI - Idle Resource & Cost Optimization Page

Rule-based + statistical idle detection surfaced as a plain-language cost
report: which nodes are wasting spend, how much, and what to do about it.
"""

import streamlit as st
import pandas as pd

from . import charts
from src import config as cfg


def render(idle_summary, gpu_hour_cost):
    st.subheader("Idle Resource & Cost Optimization")
    st.caption(
        f"A GPU is flagged idle when utilization stays below {cfg.IDLE_UTIL_THRESHOLD}% for at "
        f"least {cfg.IDLE_SUSTAINED_HOURS}+ hours, confirmed by a 1-hour moving average to avoid "
        f"false positives from brief lulls. Savings = idle hours × ${gpu_hour_cost:.2f}/GPU-hour."
    )

    total_wasted = idle_summary["estimated_wasted_usd"].sum()
    total_idle_hours = idle_summary["idle_hours"].sum()
    n_idle_now = int(idle_summary["currently_idle"].sum())

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Idle Right Now", n_idle_now)
    kpi_cols[1].metric("Total Idle GPU-Hours", f"{total_idle_hours:.1f}")
    kpi_cols[2].metric("Estimated Wasted Spend", f"${total_wasted:,.2f}")
    kpi_cols[3].metric(
        "Recoverable (est.)",
        f"${total_wasted * cfg.CONSOLIDATION_SAVINGS_FACTOR:,.2f}",
    )

    st.divider()

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.markdown("**Idle GPU-Hours by Node**")
        st.plotly_chart(
            charts.idle_hours_bar(idle_summary),
            width='stretch',
            config={"displayModeBar": False},
        )
    with chart_cols[1]:
        st.markdown("**Estimated Wasted Spend by Node**")
        st.plotly_chart(
            charts.cost_waste_bar(idle_summary),
            width='stretch',
            config={"displayModeBar": False},
        )

    st.divider()
    st.markdown("**Full Idle Report**")

    display_df = idle_summary.copy()
    display_df["currently_idle"] = display_df["currently_idle"].map(
        {True: "🟡 Idle now", False: "🟢 Active"}
    )
    display_df.columns = [
        "Node", "Status", "Latest GPU %", "Avg GPU %", "Idle Hours",
        "Window Hours", "Idle % of Window", "Est. Wasted ($)",
    ]
    st.dataframe(
        display_df,
        width='stretch',
        hide_index=True,
        column_config={
            "Idle % of Window": st.column_config.ProgressColumn(
                "Idle % of Window", min_value=0, max_value=100, format="%.0f%%"
            ),
        },
    )
