"""
ClusterPulse AI - Main Streamlit App

AI-Powered Predictive Monitoring & Cost Optimization for Compute Clusters.
Ties together the five-layer pipeline (generate -> preprocess -> anomaly
detection -> idle detection -> recommendation) and renders it as an
interactive dashboard.

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd

from src import config as cfg
from src import generator, preprocessing, anomaly_detection, idle_detection, recommendation, decision_engine
from dashboard import overview, health, cost, recommendations as recs_page, action_center

st.set_page_config(
    page_title=cfg.APP_TITLE,
    page_icon=cfg.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Minimal theming polish on top of Streamlit's defaults
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 3rem; }
        [data-testid="stMetricValue"] { font-size: 1.6rem; }
        h1, h2, h3 { letter-spacing: -0.01em; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data / model pipeline (cached so the dashboard stays snappy)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_or_generate_telemetry(seed, num_nodes, days, interval_minutes):
    try:
        telemetry_df = pd.read_csv(cfg.TELEMETRY_FILE, parse_dates=["timestamp"])
        node_meta_df = pd.read_csv(cfg.NODE_META_FILE)
    except FileNotFoundError:
        telemetry_df, node_meta_df = generator.generate_and_save(
            seed=seed, num_nodes=num_nodes, days=days, interval_minutes=interval_minutes
        )
    return telemetry_df, node_meta_df


@st.cache_data(show_spinner=False)
def regenerate_telemetry(seed, num_nodes, days, interval_minutes):
    return generator.generate_and_save(
        seed=seed, num_nodes=num_nodes, days=days, interval_minutes=interval_minutes
    )


@st.cache_data(show_spinner=False)
def run_pipeline(telemetry_df, idle_threshold, idle_hours, gpu_hour_cost):
    features_df = preprocessing.build_feature_dataset(telemetry_df)
    scored_df, _ = anomaly_detection.score_anomalies(features_df)
    health_summary = anomaly_detection.get_node_health_summary(scored_df)

    sustained_points = int(idle_hours * cfg.POINTS_PER_HOUR)
    flagged_df = idle_detection.flag_idle_periods(
        features_df, util_threshold=idle_threshold, sustained_points=sustained_points
    )
    idle_summary = idle_detection.get_idle_summary(flagged_df, gpu_hour_cost=gpu_hour_cost)

    recs_df = recommendation.generate_recommendations(health_summary, idle_summary)
    kpis = recommendation.cluster_wide_summary(health_summary, idle_summary)

    action_df = decision_engine.build_action_center(health_summary, idle_summary)
    action_summary = decision_engine.action_center_summary(action_df)

    return scored_df, health_summary, idle_summary, recs_df, kpis, action_df, action_summary


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"## {cfg.APP_ICON} {cfg.APP_TITLE}")
    st.caption("Predictive monitoring & cost optimization for compute clusters")
    st.divider()

    st.markdown("**Simulation**")
    num_nodes = st.slider("Number of nodes", 5, 50, cfg.NUM_NODES)
    sim_days = st.slider("Simulated window (days)", 1, 14, cfg.SIMULATION_DAYS)
    regenerate = st.button("🔁 Regenerate telemetry", help=cfg.REFRESH_HELP_TEXT, width='stretch')

    st.divider()
    st.markdown("**Idle Detection**")
    idle_threshold = st.slider("Idle utilization threshold (%)", 1, 30, cfg.IDLE_UTIL_THRESHOLD)
    idle_hours = st.slider("Sustained idle window (hours)", 1, 8, cfg.IDLE_SUSTAINED_HOURS)

    st.divider()
    st.markdown("**Cost Model**")
    gpu_hour_cost = st.number_input(
        "GPU-hour cost (USD)", min_value=0.10, max_value=20.0,
        value=cfg.GPU_HOURLY_COST_USD, step=0.10,
    )

    st.divider()
    st.caption("AI for Cluster Intelligence Hackathon — ClusterPulse AI")


# ---------------------------------------------------------------------------
# Load data + run pipeline
# ---------------------------------------------------------------------------
if regenerate:
    st.cache_data.clear()
    telemetry_df, node_meta_df = regenerate_telemetry(
        cfg.RANDOM_SEED, num_nodes, sim_days, cfg.INTERVAL_MINUTES
    )
else:
    telemetry_df, node_meta_df = load_or_generate_telemetry(
        cfg.RANDOM_SEED, num_nodes, sim_days, cfg.INTERVAL_MINUTES
    )

with st.spinner("Running anomaly detection + idle analysis..."):
    scored_df, health_summary, idle_summary, recs_df, kpis, action_df, action_summary = run_pipeline(
        telemetry_df, idle_threshold, idle_hours, gpu_hour_cost
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title(f"{cfg.APP_ICON} {cfg.APP_TITLE}")
st.markdown(
    "AI-powered monitoring layer that interprets cluster telemetry -- flagging nodes at risk "
    "of failure and identifying wasted GPU capacity -- and translates findings into clear, "
    "actionable recommendations."
)

tab_overview, tab_health, tab_cost, tab_recs, tab_actions = st.tabs(
    ["📊 Overview", "🩺 Node Health", "💰 Cost Optimization", "✅ Recommendations", "🎯 Action Center"]
)

with tab_overview:
    overview.render(health_summary, idle_summary, kpis)

with tab_health:
    health.render(scored_df, health_summary)

with tab_cost:
    cost.render(idle_summary, gpu_hour_cost)

with tab_recs:
    recs_page.render(recs_df)

with tab_actions:
    action_center.render(action_df, action_summary)
