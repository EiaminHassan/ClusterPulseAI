"""
ClusterPulse AI - AI Analytics Engine: Idle / Underutilization Detection

Rule-based approach chosen deliberately over ML: a GPU is flagged idle if
utilization stays below a threshold for a sustained window, layered with a
moving-average trend check to avoid false positives from brief lulls.
Clarity and correctness matter more here than model complexity.
"""

import numpy as np
import pandas as pd

from . import config as cfg


def flag_idle_periods(
    features_df,
    util_threshold=cfg.IDLE_UTIL_THRESHOLD,
    sustained_points=cfg.IDLE_SUSTAINED_POINTS,
):
    """
    Adds two columns:
      - below_threshold : instantaneous reading is below the idle threshold
      - is_idle          : the node has been continuously below threshold for
                            at least `sustained_points` consecutive readings
                            AND the smoothed moving average also confirms it
                            (guards against brief lulls / noise).
    """
    df = features_df.copy()
    df["below_threshold"] = df["gpu_util"] < util_threshold

    def _sustained_flag(group):
        group = group.sort_values("timestamp")
        below = group["below_threshold"].values
        # run-length of consecutive True values ending at each point
        run_length = np.zeros(len(below), dtype=int)
        streak = 0
        for i, val in enumerate(below):
            streak = streak + 1 if val else 0
            run_length[i] = streak
        sustained = run_length >= sustained_points
        # trend confirmation: smoothed moving average also below threshold
        trend_confirmed = group["gpu_util_moving_avg"].values < (util_threshold * 1.5)
        is_idle = sustained & trend_confirmed
        return pd.Series(is_idle, index=group.index)

    df["is_idle"] = df.groupby("node_id", group_keys=False).apply(
        _sustained_flag, include_groups=False
    )
    return df


def get_idle_summary(
    idle_flagged_df,
    interval_minutes=cfg.INTERVAL_MINUTES,
    gpu_hour_cost=cfg.GPU_HOURLY_COST_USD,
):
    """
    Aggregate to one row per node:
      - idle_hours          : total time flagged idle over the observed window
      - idle_pct_of_window   : share of the observed window spent idle
      - currently_idle       : is the node idle right now (latest reading)
      - estimated_wasted_usd : idle_hours * gpu_hour_cost
    """
    points_per_hour = 60 / interval_minutes

    agg = (
        idle_flagged_df.groupby("node_id")
        .agg(
            total_points=("is_idle", "size"),
            idle_points=("is_idle", "sum"),
            avg_gpu_util=("gpu_util", "mean"),
        )
        .reset_index()
    )
    agg["idle_hours"] = agg["idle_points"] / points_per_hour
    agg["total_hours"] = agg["total_points"] / points_per_hour
    agg["idle_pct_of_window"] = (agg["idle_points"] / agg["total_points"] * 100).round(1)
    agg["estimated_wasted_usd"] = (agg["idle_hours"] * gpu_hour_cost).round(2)

    latest = (
        idle_flagged_df.sort_values("timestamp")
        .groupby("node_id")
        .tail(1)[["node_id", "is_idle", "gpu_util"]]
        .rename(columns={"is_idle": "currently_idle", "gpu_util": "latest_gpu_util"})
    )

    summary = agg.merge(latest, on="node_id")
    summary["avg_gpu_util"] = summary["avg_gpu_util"].round(1)
    summary["idle_hours"] = summary["idle_hours"].round(1)
    summary = summary.sort_values("estimated_wasted_usd", ascending=False).reset_index(drop=True)

    cols = [
        "node_id", "currently_idle", "latest_gpu_util", "avg_gpu_util",
        "idle_hours", "total_hours", "idle_pct_of_window", "estimated_wasted_usd",
    ]
    return summary[cols]


if __name__ == "__main__":
    from . import preprocessing as pp

    features_df = pp.build_feature_dataset()
    flagged_df = flag_idle_periods(features_df)
    summary = get_idle_summary(flagged_df)
    print(summary)
    print(f"\nTotal estimated wasted spend: ${summary['estimated_wasted_usd'].sum():,.2f}")
