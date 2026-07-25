"""
ClusterPulse AI - Overview Page

Cluster-wide health + cost KPIs, at a glance. This is the "single
intelligent view" the concept note promises: what's wrong and what to
do about it, in one screen.
"""

import streamlit as st
import pandas as pd

from . import charts


def render(health_summary, idle_summary, cluster_kpis):
    st.subheader("Cluster Status at a Glance")

    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Total Nodes", cluster_kpis["total_nodes"])
    kpi_cols[1].metric("At-Risk", cluster_kpis["at_risk_nodes"], delta=None)
    kpi_cols[2].metric("Watch", cluster_kpis["watch_nodes"])
    kpi_cols[3].metric("Idle Right Now", cluster_kpis["nodes_idle_now"])
    kpi_cols[4].metric(
        "Est. Wasted Spend", f"${cluster_kpis['total_wasted_usd']:,.0f}"
    )

    st.divider()

    col_left, col_right = st.columns([1, 1.4])

    with col_left:
        st.markdown("**Node Health Distribution**")
        st.plotly_chart(
            charts.risk_distribution_donut(health_summary),
            width='stretch',
            config={"displayModeBar": False},
        )

    with col_right:
        st.markdown("**Cost Optimization Opportunity**")
        recoverable = cluster_kpis["recoverable_usd"]
        wasted = cluster_kpis["total_wasted_usd"]
        st.markdown(
            f"""
            <div style="padding: 1.1rem 1.3rem; border-radius: 10px; background: rgba(124,108,240,0.10);
                        border: 1px solid rgba(124,108,240,0.35);">
                <div style="font-size: 0.85rem; opacity: 0.75;">Idle GPU-hours observed this window</div>
                <div style="font-size: 1.6rem; font-weight: 700;">{cluster_kpis['total_idle_hours']:.1f} hrs</div>
                <div style="font-size: 0.85rem; opacity: 0.75; margin-top: 0.6rem;">Estimated wasted spend</div>
                <div style="font-size: 1.6rem; font-weight: 700; color:#F5B14C;">${wasted:,.2f}</div>
                <div style="font-size: 0.85rem; opacity: 0.75; margin-top: 0.6rem;">Recoverable via consolidation / shutdown</div>
                <div style="font-size: 1.6rem; font-weight: 700; color:#3DDC97;">~${recoverable:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("**Node Risk Ranking**")
    st.plotly_chart(
        charts.node_risk_bar(health_summary),
        width='stretch',
        config={"displayModeBar": False},
    )

    st.caption(
        "Risk score is derived from an Isolation Forest trained on thermal and power-draw "
        "stability signals -- higher means the node's recent behavior looks more like the "
        "early telemetry of a node that later failed, in the synthetic training data."
    )
