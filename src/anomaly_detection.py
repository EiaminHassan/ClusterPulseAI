"""
ClusterPulse AI - AI Analytics Engine: Node Health & Failure-Risk Detection

Uses an unsupervised Isolation Forest over engineered telemetry features to
flag nodes that behave differently from normal cluster behavior. Chosen
over a black-box deep model because it requires no labeled failure data,
is computationally lightweight, and its output is easy to explain to
non-technical stakeholders: "this node is being isolated because its
recent readings are few and different from the rest of the cluster."
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

from . import config as cfg


def _risk_label(risk_score):
    if risk_score >= cfg.RISK_AT_RISK_THRESHOLD:
        return "At-Risk"
    if risk_score >= cfg.RISK_WATCH_THRESHOLD:
        return "Watch"
    return "Healthy"


def train_anomaly_model(features_df, feature_cols=cfg.ANOMALY_FEATURES, seed=cfg.RANDOM_SEED):
    """Fit an Isolation Forest across the whole cluster's telemetry history."""
    X = features_df[feature_cols].values
    model = IsolationForest(
        n_estimators=cfg.ISOLATION_FOREST_ESTIMATORS,
        contamination=cfg.ISOLATION_FOREST_CONTAMINATION,
        random_state=seed,
    )
    model.fit(X)
    return model


def score_anomalies(features_df, model=None, feature_cols=cfg.ANOMALY_FEATURES):
    """
    Score every telemetry row. Adds:
      - anomaly_score_raw : sklearn's decision_function (higher = more normal)
      - risk_score        : normalized [0, 1], higher = more at-risk
      - risk_label         : Healthy / Watch / At-Risk
    """
    features_df = features_df.copy()
    if model is None:
        model = train_anomaly_model(features_df, feature_cols)

    X = features_df[feature_cols].values
    raw_scores = model.decision_function(X)          # higher = more normal
    features_df["anomaly_score_raw"] = raw_scores

    # Invert + normalize so higher risk_score = more anomalous, scaled to [0, 1]
    inverted = -raw_scores
    scaler = MinMaxScaler()
    features_df["risk_score"] = scaler.fit_transform(inverted.reshape(-1, 1)).flatten()
    features_df["risk_label"] = features_df["risk_score"].apply(_risk_label)
    features_df["is_anomaly"] = (model.predict(X) == -1)

    return features_df, model


def get_node_health_summary(scored_df):
    """
    Collapse the scored time-series down to one row per node representing
    current status: latest reading + recent trend + overall risk label.
    """
    latest = (
        scored_df.sort_values("timestamp")
        .groupby("node_id")
        .tail(1)
        .set_index("node_id")
    )

    # Recent trend: average risk_score over the last 8 readings (2 hours) vs. the
    # first 8 readings, to show whether a node is drifting toward failure.
    def _trend(group):
        group = group.sort_values("timestamp")
        n = len(group)
        window = min(8, n)
        recent = group["risk_score"].tail(window).mean()
        earliest = group["risk_score"].head(window).mean()
        return pd.Series({"risk_trend_delta": recent - earliest})

    trend = scored_df.groupby("node_id").apply(_trend, include_groups=False)

    summary = latest[
        [
            "timestamp", "gpu_util", "cpu_util", "memory_util", "temperature",
            "power_draw", "risk_score", "risk_label", "is_anomaly",
        ]
    ].join(trend)

    summary = summary.rename(columns={"timestamp": "last_reading"})
    summary["risk_score"] = summary["risk_score"].round(3)
    summary["risk_trend_delta"] = summary["risk_trend_delta"].round(3)
    summary = summary.sort_values("risk_score", ascending=False).reset_index()
    return summary


if __name__ == "__main__":
    from . import preprocessing as pp

    features_df = pp.build_feature_dataset()
    scored_df, model = score_anomalies(features_df)
    summary = get_node_health_summary(scored_df)
    print(summary[["node_id", "risk_label", "risk_score", "risk_trend_delta"]])
