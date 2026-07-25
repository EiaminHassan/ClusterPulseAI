"""
ClusterPulse AI - Recommendation Engine

Converts raw model outputs (health scores + idle summaries) into
plain-language, decision-oriented recommendations for operators, plus a
transparent cost-savings estimate. Outputs are recommendations and risk
scores, not just charts.
"""

import pandas as pd

from . import config as cfg


def _health_recommendation(row):
    if row["risk_label"] == "At-Risk":
        return (
            f"URGENT: {row['node_id']} shows strong signs of thermal/power instability "
            f"(risk score {row['risk_score']:.2f}). Schedule inspection or migrate running "
            f"jobs off this node before it fails."
        )
    if row["risk_label"] == "Watch":
        trend = "rising" if row.get("risk_trend_delta", 0) > 0.05 else "stable"
        return (
            f"WATCH: {row['node_id']} is showing early signs of abnormal behavior "
            f"(risk score {row['risk_score']:.2f}, trend {trend}). No action required yet -- "
            f"monitor closely over the next few hours."
        )
    return f"OK: {row['node_id']} is operating within normal parameters."


def _cost_recommendation(row, savings_factor=cfg.CONSOLIDATION_SAVINGS_FACTOR):
    if not row["currently_idle"] and row["idle_pct_of_window"] < 20:
        return f"OK: {row['node_id']} shows healthy utilization -- no action needed."

    recoverable = round(row["estimated_wasted_usd"] * savings_factor, 2)
    if row["idle_pct_of_window"] >= 60:
        action = "consider shutting down or reallocating this node entirely"
    elif row["idle_pct_of_window"] >= 20:
        action = "consolidate its workloads onto fewer nodes"
    else:
        action = "monitor for a recurring idle pattern"

    return (
        f"COST: {row['node_id']} was idle {row['idle_pct_of_window']:.0f}% of the observed "
        f"window (~{row['idle_hours']:.1f} idle GPU-hours, ~${row['estimated_wasted_usd']:.2f} "
        f"wasted). Recommend: {action}. Estimated recoverable savings: ~${recoverable:.2f}."
    )


def generate_recommendations(health_summary, idle_summary):
    """
    Merge health + idle summaries into a single operator-facing recommendation
    table, one row per node, each with a plain-language health note and a
    plain-language cost note.
    """
    merged = health_summary.merge(idle_summary, on="node_id", how="outer")

    merged["health_recommendation"] = merged.apply(_health_recommendation, axis=1)
    merged["cost_recommendation"] = merged.apply(_cost_recommendation, axis=1)

    def _priority(row):
        if row["risk_label"] == "At-Risk":
            return 0
        if row["risk_label"] == "Watch" or row["idle_pct_of_window"] >= 60:
            return 1
        if row["idle_pct_of_window"] >= 20:
            return 2
        return 3

    merged["priority"] = merged.apply(_priority, axis=1)
    merged = merged.sort_values(["priority", "risk_score"], ascending=[True, False]).reset_index(
        drop=True
    )
    return merged


def cluster_wide_summary(health_summary, idle_summary):
    """Top-line KPIs for the dashboard overview page."""
    n_nodes = health_summary["node_id"].nunique()
    n_at_risk = (health_summary["risk_label"] == "At-Risk").sum()
    n_watch = (health_summary["risk_label"] == "Watch").sum()
    n_healthy = n_nodes - n_at_risk - n_watch

    n_idle_now = idle_summary["currently_idle"].sum()
    total_idle_hours = idle_summary["idle_hours"].sum()
    total_wasted_usd = idle_summary["estimated_wasted_usd"].sum()
    recoverable_usd = total_wasted_usd * cfg.CONSOLIDATION_SAVINGS_FACTOR

    return {
        "total_nodes": n_nodes,
        "healthy_nodes": n_healthy,
        "watch_nodes": n_watch,
        "at_risk_nodes": n_at_risk,
        "nodes_idle_now": int(n_idle_now),
        "total_idle_hours": round(total_idle_hours, 1),
        "total_wasted_usd": round(total_wasted_usd, 2),
        "recoverable_usd": round(recoverable_usd, 2),
    }


if __name__ == "__main__":
    from . import preprocessing as pp
    from . import anomaly_detection as ad
    from . import idle_detection as idd

    features_df = pp.build_feature_dataset()
    scored_df, _ = ad.score_anomalies(features_df)
    health_summary = ad.get_node_health_summary(scored_df)

    flagged_df = idd.flag_idle_periods(features_df)
    idle_summary = idd.get_idle_summary(flagged_df)

    recs = generate_recommendations(health_summary, idle_summary)
    pd.set_option("display.width", 160)
    print(recs[["node_id", "priority", "health_recommendation"]])
    print()
    print(recs[["node_id", "priority", "cost_recommendation"]])
    print()
    print(cluster_wide_summary(health_summary, idle_summary))
