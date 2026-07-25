"""
ClusterPulse AI - Data Processing & Feature Engineering Layer

Cleans raw telemetry and derives the rolling-window features (rolling
averages, rate-of-change, sustained-high-utilization flags) that feed
both the anomaly detector and the idle detector.
"""

import numpy as np
import pandas as pd

from . import config as cfg


def load_telemetry(path=cfg.TELEMETRY_FILE):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return clean_telemetry(df)


def clean_telemetry(df):
    """Basic cleaning: drop duplicates, sort, clip out-of-range values, ffill small gaps."""
    df = df.drop_duplicates(subset=["node_id", "timestamp"])
    df = df.sort_values(["node_id", "timestamp"]).reset_index(drop=True)

    df["gpu_util"] = df["gpu_util"].clip(0, 100)
    df["cpu_util"] = df["cpu_util"].clip(0, 100)
    df["memory_util"] = df["memory_util"].clip(0, 100)
    df["temperature"] = df["temperature"].clip(0, 120)
    df["power_draw"] = df["power_draw"].clip(0, 600)

    # Forward-fill small gaps per node (simulates handling brief telemetry dropouts)
    numeric_cols = ["gpu_util", "cpu_util", "memory_util", "temperature", "power_draw"]
    df[numeric_cols] = df.groupby("node_id")[numeric_cols].transform(
        lambda s: s.ffill().bfill()
    )
    return df


def engineer_features(df, roll_window=cfg.ROLLING_WINDOW_POINTS):
    """
    Add rolling averages, rate-of-change, and sustained-high-utilization
    flags, computed independently per node so no information leaks across
    node boundaries.
    """
    df = df.copy()
    grouped = df.groupby("node_id", group_keys=False)

    df["temp_roll_avg"] = grouped["temperature"].transform(
        lambda s: s.rolling(roll_window, min_periods=1).mean()
    )
    df["power_roll_avg"] = grouped["power_draw"].transform(
        lambda s: s.rolling(roll_window, min_periods=1).mean()
    )
    df["gpu_util_roll_avg"] = grouped["gpu_util"].transform(
        lambda s: s.rolling(roll_window, min_periods=1).mean()
    )

    df["temp_rate_of_change"] = grouped["temperature"].transform(
        lambda s: s.diff().fillna(0)
    )
    df["power_rate_of_change"] = grouped["power_draw"].transform(
        lambda s: s.diff().fillna(0)
    )

    # Rolling volatility: a pre-failure node's power/temperature swings become
    # erratic before it degrades, whereas a healthy *or* idle node stays stable
    # (just at different absolute levels). This is the strongest failure-risk
    # signal, and importantly does NOT conflate "idle" with "at risk".
    df["temp_roll_std"] = grouped["temperature"].transform(
        lambda s: s.rolling(roll_window, min_periods=1).std().fillna(0)
    )
    df["power_roll_std"] = grouped["power_draw"].transform(
        lambda s: s.rolling(roll_window, min_periods=1).std().fillna(0)
    )

    # Sustained-high-utilization flag: GPU util above 90% for the whole rolling window
    df["sustained_high_util"] = grouped["gpu_util"].transform(
        lambda s: (s.rolling(roll_window, min_periods=1).min() > 90).astype(int)
    )

    # Idle-window moving average, used by idle_detection for trend smoothing
    df["gpu_util_moving_avg"] = grouped["gpu_util"].transform(
        lambda s: s.rolling(cfg.IDLE_MOVING_AVG_WINDOW, min_periods=1).mean()
    )

    return df


def build_feature_dataset(raw_df=None):
    """Convenience wrapper: load (if needed), clean, and engineer features in one call."""
    if raw_df is None:
        raw_df = load_telemetry()
    else:
        raw_df = clean_telemetry(raw_df)
    return engineer_features(raw_df)


if __name__ == "__main__":
    features_df = build_feature_dataset()
    print(features_df.shape)
    print(features_df[cfg.ANOMALY_FEATURES].describe())
