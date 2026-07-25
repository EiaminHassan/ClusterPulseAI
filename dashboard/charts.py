"""
ClusterPulse AI - Dashboard Chart Builders

Reusable Plotly figure constructors, kept separate from page layout so the
same charts can be reused across Overview / Health / Cost pages.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Palette -- deliberately not the default plotly qualitative set
COLOR_HEALTHY = "#3DDC97"
COLOR_WATCH = "#F5B14C"
COLOR_AT_RISK = "#E8544B"
COLOR_IDLE = "#6C8EBF"
COLOR_ACCENT = "#7C6CF0"
COLOR_BG = "rgba(0,0,0,0)"

RISK_COLOR_MAP = {"Healthy": COLOR_HEALTHY, "Watch": COLOR_WATCH, "At-Risk": COLOR_AT_RISK}


def _base_layout(fig, height=340, title=None):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        font=dict(family="Inter, sans-serif", size=13, color="#E5E5E5"),
        title=dict(text=title, font=dict(size=15)) if title else None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def risk_distribution_donut(health_summary):
    counts = health_summary["risk_label"].value_counts().reindex(
        ["Healthy", "Watch", "At-Risk"], fill_value=0
    )
    fig = go.Figure(
        go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.62,
            marker=dict(colors=[RISK_COLOR_MAP[l] for l in counts.index]),
            textinfo="value",
            hovertemplate="%{label}: %{value} nodes<extra></extra>",
        )
    )
    fig.add_annotation(
        text=f"<b>{counts.sum()}</b><br>nodes", x=0.5, y=0.5, showarrow=False, font=dict(size=16)
    )
    return _base_layout(fig, height=280)


def node_risk_bar(health_summary):
    df = health_summary.sort_values("risk_score", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=df["risk_score"],
            y=df["node_id"],
            orientation="h",
            marker=dict(color=[RISK_COLOR_MAP[l] for l in df["risk_label"]]),
            hovertemplate="%{y}<br>risk score: %{x:.2f}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Risk Score", yaxis_title=None)
    fig.update_xaxes(range=[0, 1])
    return _base_layout(fig, height=max(280, 22 * len(df)))


def node_timeseries(scored_df, node_id, metrics=("temperature", "power_draw", "gpu_util")):
    node_df = scored_df[scored_df["node_id"] == node_id].sort_values("timestamp")
    fig = go.Figure()
    colors = [COLOR_AT_RISK, COLOR_ACCENT, COLOR_HEALTHY]
    for metric, color in zip(metrics, colors):
        fig.add_trace(
            go.Scatter(
                x=node_df["timestamp"],
                y=node_df[metric],
                mode="lines",
                name=metric.replace("_", " ").title(),
                line=dict(color=color, width=1.8),
            )
        )
    # shade anomalous points
    anomalies = node_df[node_df["is_anomaly"]]
    if len(anomalies):
        fig.add_trace(
            go.Scatter(
                x=anomalies["timestamp"],
                y=anomalies[metrics[0]],
                mode="markers",
                name="Flagged reading",
                marker=dict(color=COLOR_AT_RISK, size=7, symbol="x"),
            )
        )
    fig.update_layout(xaxis_title=None, yaxis_title=None)
    return _base_layout(fig, height=320, title=f"{node_id} — telemetry over time")


def idle_hours_bar(idle_summary, top_n=15):
    df = idle_summary.sort_values("idle_hours", ascending=True).tail(top_n)
    fig = go.Figure(
        go.Bar(
            x=df["idle_hours"],
            y=df["node_id"],
            orientation="h",
            marker=dict(color=COLOR_IDLE),
            hovertemplate="%{y}<br>%{x:.1f} idle hours<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Idle GPU-Hours", yaxis_title=None)
    return _base_layout(fig, height=max(280, 24 * len(df)))


def cost_waste_bar(idle_summary, top_n=15):
    df = idle_summary.sort_values("estimated_wasted_usd", ascending=True).tail(top_n)
    fig = go.Figure(
        go.Bar(
            x=df["estimated_wasted_usd"],
            y=df["node_id"],
            orientation="h",
            marker=dict(
                color=df["estimated_wasted_usd"],
                colorscale=[[0, "#3A3F58"], [1, COLOR_ACCENT]],
            ),
            hovertemplate="%{y}<br>$%{x:.2f} wasted<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Estimated Wasted Spend (USD)", yaxis_title=None)
    return _base_layout(fig, height=max(280, 24 * len(df)))


def utilization_heatmap(scored_df, metric="gpu_util"):
    pivot = scored_df.pivot_table(index="node_id", columns="timestamp", values=metric)
    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale=[[0, "#1C2033"], [0.5, "#3A3F58"], [1, COLOR_ACCENT]],
            colorbar=dict(title=metric.replace("_", " ").title()),
            hovertemplate="%{y}<br>%{x}<br>%{z:.1f}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title=None, yaxis_title=None)
    return _base_layout(fig, height=max(320, 20 * pivot.shape[0]))


def cluster_gauge(value, title, max_value=100, color=COLOR_ACCENT):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title=dict(text=title, font=dict(size=13)),
            gauge=dict(
                axis=dict(range=[0, max_value]),
                bar=dict(color=color),
                bgcolor=COLOR_BG,
                borderwidth=0,
            ),
        )
    )
    return _base_layout(fig, height=220)
