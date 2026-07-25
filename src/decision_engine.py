"""
ClusterPulse AI - Action Ranking & Conflict Resolver

Sits AFTER the Health Module (anomaly_detection.py) and Cost Module
(idle_detection.py), and BEFORE the dashboard. The two upstream modules
score every node independently and can recommend different, sometimes
contradictory, actions for the same node (e.g. Health says "inspect this
node", Cost says "shut it down"). This module:

  1. collects both modules' outputs into one unified action format,
  2. scores every action with a transparent, explainable Value Score,
  3. detects when the two modules disagree on the same node,
  4. resolves any disagreement with one explainable business rule
     ("Safety First": risk wins above a threshold, cost wins below it),
  5. produces a single ranked action list, sorted by Value Score, with a
     one-sentence plain-language reason for every row.

Existing module APIs (anomaly_detection.py, idle_detection.py,
recommendation.py) are untouched -- this module only consumes their
outputs, it never modifies them.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pandas as pd

from . import config as cfg


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Action:
    """
    A single module's recommendation for a single node, in a unified format.
    Exactly two Action objects exist per node: one with source="health",
    one with source="cost". Wraps the existing modules' outputs without
    modifying them.
    """

    node_id: str
    source: str                 # "health" or "cost"
    recommendation: str         # categorical action label, e.g. "Inspect Immediately"
    risk_score: float           # node's risk_score (same value on both Actions for a node)
    idle_status: bool           # node's currently_idle flag (same value on both Actions)
    estimated_saving: float     # node's estimated_wasted_usd (same value on both Actions)
    health_label: str           # node's risk_label: Healthy / Watch / At-Risk


@dataclass
class NodeDecision:
    """The fully resolved, ranked decision for one node."""

    node_id: str
    health_label: str
    risk_score: float
    health_action: str
    cost_action: str
    idle_status: bool
    idle_pct_of_window: float
    idle_hours: float
    estimated_saving: float
    value_score: float
    conflict: bool
    conflict_reason: Optional[str]
    final_action: str
    reason: str
    rank: int = field(default=0)


# ---------------------------------------------------------------------------
# 1. Collect actions
# ---------------------------------------------------------------------------
def categorize_health_action(risk_label: str) -> str:
    """Map the Health Module's risk_label to a categorical action."""
    if risk_label == "At-Risk":
        return cfg.HEALTH_ACTION_INSPECT
    if risk_label == "Watch":
        return cfg.HEALTH_ACTION_MONITOR
    return cfg.HEALTH_ACTION_NONE


def categorize_cost_action(idle_pct_of_window: float) -> str:
    """Map the Cost Module's idle percentage to a categorical action."""
    if idle_pct_of_window >= cfg.COST_ACTION_SHUTDOWN_IDLE_PCT:
        return cfg.COST_ACTION_SHUTDOWN
    if idle_pct_of_window >= cfg.COST_ACTION_CONSOLIDATE_IDLE_PCT:
        return cfg.COST_ACTION_CONSOLIDATE
    return cfg.COST_ACTION_NONE


def collect_actions(health_summary: pd.DataFrame, idle_summary: pd.DataFrame) -> List[Action]:
    """
    Receive outputs from both the Health Module and the Cost Module and wrap
    them into a unified list of Action objects (two per node). Does not
    modify either input DataFrame.
    """
    merged = health_summary.merge(idle_summary, on="node_id", how="outer")

    actions: List[Action] = []
    for _, row in merged.iterrows():
        health_action = categorize_health_action(row["risk_label"])
        cost_action = categorize_cost_action(row["idle_pct_of_window"])

        shared = dict(
            node_id=row["node_id"],
            risk_score=float(row["risk_score"]),
            idle_status=bool(row["currently_idle"]),
            estimated_saving=float(row["estimated_wasted_usd"]),
            health_label=row["risk_label"],
        )
        actions.append(Action(source="health", recommendation=health_action, **shared))
        actions.append(Action(source="cost", recommendation=cost_action, **shared))

    return actions


# ---------------------------------------------------------------------------
# 2. Value score
# ---------------------------------------------------------------------------
def calculate_value_score(risk_score: float, estimated_saving: float) -> float:
    """
    Value Score = (risk_score * VALUE_SCORE_RISK_WEIGHT)
                + (normalized_saving * VALUE_SCORE_COST_WEIGHT)

    where normalized_saving = min(estimated_saving / VALUE_SCORE_COST_NORMALIZATION_CAP, 1.0)

    risk_score is already on a 0-1 scale. estimated_saving (USD) is capped
    and normalized to the same 0-1 scale so neither term dominates just
    because dollars are numerically larger than a 0-1 probability. Weights
    are configurable in config.py and sum to 1.0, so the result is itself a
    0-1 "how much does this node matter" score.
    """
    normalized_saving = min(estimated_saving / cfg.VALUE_SCORE_COST_NORMALIZATION_CAP, 1.0)
    score = (
        risk_score * cfg.VALUE_SCORE_RISK_WEIGHT
        + normalized_saving * cfg.VALUE_SCORE_COST_WEIGHT
    )
    return round(score, 4)


# ---------------------------------------------------------------------------
# 3. Conflict detection
# ---------------------------------------------------------------------------
def detect_conflicts(health_action: str, cost_action: str) -> Tuple[bool, Optional[str]]:
    """
    A conflict exists when BOTH modules want to take action on the same node,
    but the actions differ (e.g. Inspect vs Shutdown, Monitor vs Consolidate).
    If either module has nothing to recommend ("No Action"), there's nothing
    to disagree about, so it's not a conflict -- just one module speaking.
    """
    both_active = health_action != cfg.HEALTH_ACTION_NONE and cost_action != cfg.COST_ACTION_NONE
    if not both_active:
        return False, None

    reason = (
        f"Health module recommends '{health_action}' while Cost module "
        f"recommends '{cost_action}' for the same node."
    )
    return True, reason


# ---------------------------------------------------------------------------
# 4. Conflict resolution ("Safety First" rule) + plain-language reasons
# ---------------------------------------------------------------------------
def generate_reason(
    final_action: str,
    conflict: bool,
    risk_score: float,
    idle_hours: float,
) -> str:
    """Rule-based, one-sentence plain-language explanation. No LLM required."""
    if conflict and final_action in (cfg.HEALTH_ACTION_INSPECT, cfg.HEALTH_ACTION_MONITOR):
        return "High failure risk outweighs possible cost savings."
    if conflict:
        return (
            "Node shows low operational risk, so the potential cost savings "
            "take priority over the health flag (Safety First rule)."
        )

    if final_action == cfg.HEALTH_ACTION_INSPECT:
        return "Immediate inspection is recommended due to abnormal thermal behavior."
    if final_action == cfg.HEALTH_ACTION_MONITOR:
        return "Node shows early signs of abnormal behavior and should be monitored closely."
    if final_action in (cfg.COST_ACTION_SHUTDOWN, cfg.COST_ACTION_CONSOLIDATE):
        return f"Node has remained idle for over {idle_hours:.1f} hours with low operational risk."
    return "Node is operating within normal parameters and utilization."


def resolve_conflicts(
    health_action: str,
    cost_action: str,
    conflict: bool,
    risk_score: float,
    idle_hours: float,
    threshold: float = cfg.SAFETY_FIRST_RISK_THRESHOLD,
) -> Tuple[str, str]:
    """
    Apply the "Safety First" rule:
        IF risk_score >= threshold  -> health recommendation wins
        ELSE                        -> cost recommendation wins
    When there's no conflict, whichever module actually recommended an
    action (i.e. isn't "No Action") wins; if neither did, the final action
    is "No Action".
    """
    if conflict:
        final_action = health_action if risk_score >= threshold else cost_action
    elif health_action != cfg.HEALTH_ACTION_NONE:
        final_action = health_action
    elif cost_action != cfg.COST_ACTION_NONE:
        final_action = cost_action
    else:
        final_action = cfg.HEALTH_ACTION_NONE  # "No Action" (same constant value as cost's)

    reason = generate_reason(final_action, conflict, risk_score, idle_hours)
    return final_action, reason


# ---------------------------------------------------------------------------
# 5. Ranking
# ---------------------------------------------------------------------------
def rank_actions(decisions: List[NodeDecision]) -> List[NodeDecision]:
    """Sort by Value Score, highest first, and assign rank 1..N."""
    decisions = sorted(decisions, key=lambda d: d.value_score, reverse=True)
    for i, decision in enumerate(decisions, start=1):
        decision.rank = i
    return decisions


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def build_node_decisions(
    health_summary: pd.DataFrame, idle_summary: pd.DataFrame
) -> List[NodeDecision]:
    """Run the full collect -> score -> detect -> resolve -> rank pipeline."""
    actions = collect_actions(health_summary, idle_summary)

    # idle_hours and idle_pct_of_window aren't on the Action dataclass (they're
    # not part of the spec's unified schema), so pull them from idle_summary
    # directly when assembling the final per-node decision.
    idle_lookup = idle_summary.set_index("node_id")

    decisions: List[NodeDecision] = []
    by_node = {}
    for action in actions:
        by_node.setdefault(action.node_id, {})[action.source] = action

    for node_id, pair in by_node.items():
        health_a, cost_a = pair["health"], pair["cost"]
        idle_row = idle_lookup.loc[node_id]

        conflict, conflict_reason = detect_conflicts(
            health_a.recommendation, cost_a.recommendation
        )
        final_action, reason = resolve_conflicts(
            health_a.recommendation,
            cost_a.recommendation,
            conflict,
            health_a.risk_score,
            float(idle_row["idle_hours"]),
        )
        value_score = calculate_value_score(health_a.risk_score, health_a.estimated_saving)

        decisions.append(
            NodeDecision(
                node_id=node_id,
                health_label=health_a.health_label,
                risk_score=health_a.risk_score,
                health_action=health_a.recommendation,
                cost_action=cost_a.recommendation,
                idle_status=health_a.idle_status,
                idle_pct_of_window=float(idle_row["idle_pct_of_window"]),
                idle_hours=float(idle_row["idle_hours"]),
                estimated_saving=health_a.estimated_saving,
                value_score=value_score,
                conflict=conflict,
                conflict_reason=conflict_reason,
                final_action=final_action,
                reason=reason,
            )
        )

    return rank_actions(decisions)


def build_action_center(health_summary: pd.DataFrame, idle_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Top-level entry point for the dashboard: runs the full decision pipeline
    and returns one ranked DataFrame ready to display.
    """
    decisions = build_node_decisions(health_summary, idle_summary)
    rows = [
        {
            "Rank": d.rank,
            "Node": d.node_id,
            "Health Status": d.health_label,
            "Risk Score": d.risk_score,
            "Idle Status": "Idle" if d.idle_status else "Active",
            "Idle % of Window": d.idle_pct_of_window,
            "Estimated Saving": d.estimated_saving,
            "Value Score": d.value_score,
            "Health Action": d.health_action,
            "Cost Action": d.cost_action,
            "Conflict": d.conflict,
            "Conflict Reason": d.conflict_reason,
            "Final Action": d.final_action,
            "Reason": d.reason,
        }
        for d in decisions
    ]
    return pd.DataFrame(rows)


def action_center_summary(action_df: pd.DataFrame) -> dict:
    """KPIs for the Action Center dashboard header."""
    total_conflicts = int(action_df["Conflict"].sum())
    top_row = action_df.iloc[0] if len(action_df) else None
    return {
        "total_actions": len(action_df),
        "total_conflicts_found": total_conflicts,
        "total_conflicts_resolved": total_conflicts,  # every detected conflict is resolved by the rule
        "highest_priority_node": top_row["Node"] if top_row is not None else None,
        "highest_priority_action": top_row["Final Action"] if top_row is not None else None,
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

    action_df = build_action_center(health_summary, idle_summary)
    pd.set_option("display.width", 200)
    print(action_df[["Rank", "Node", "Health Action", "Cost Action", "Conflict", "Final Action", "Reason"]])
    print()
    print(action_center_summary(action_df))
