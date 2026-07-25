"""
ClusterPulse AI - Central Configuration
All tunable parameters for data generation, models, and cost estimation live here
so the rest of the codebase never hardcodes magic numbers.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TELEMETRY_FILE = os.path.join(DATA_DIR, "telemetry.csv")
NODE_META_FILE = os.path.join(DATA_DIR, "node_metadata.csv")

os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Simulation / Data Generation
# ---------------------------------------------------------------------------
NUM_NODES = 20
SIMULATION_DAYS = 7
INTERVAL_MINUTES = 15                       # telemetry sampling interval
POINTS_PER_HOUR = 60 // INTERVAL_MINUTES
RANDOM_SEED = 42

# Fraction of nodes that get an injected "pre-failure" pattern
FAILURE_NODE_FRACTION = 0.15
# Fraction of nodes that get an injected "chronically idle" pattern
IDLE_NODE_FRACTION = 0.25
# Guaranteed number of "critical idle" demo nodes: idle on utilization (so the
# Cost module wants to shut it down) but thermally/electrically unstable (so
# the Health module wants it inspected) -- i.e. a node engineered to make the
# two modules disagree, so the Action Ranking & Conflict Resolver always has
# at least one real conflict to resolve during a demo.
CONFLICT_DEMO_NODE_COUNT = 1

# "Healthy" baseline operating bands
BASELINE_GPU_UTIL = (55, 85)                # percent
BASELINE_CPU_UTIL = (30, 70)                # percent
BASELINE_MEMORY_UTIL = (40, 80)             # percent
BASELINE_TEMPERATURE = (55, 75)             # Celsius
BASELINE_POWER_DRAW = (180, 280)            # Watts (per-GPU draw)

# ---------------------------------------------------------------------------
# Anomaly / Failure-Risk Detection (Isolation Forest)
# ---------------------------------------------------------------------------
# Note: raw utilization is deliberately excluded here. Utilization is naturally
# low for idle nodes, and including it would cause the anomaly model to flag
# idle (cost-optimization) nodes as failure-risks. Failure risk is driven by
# thermal stress and power instability, so we focus on temperature/power level,
# their trend (rolling average), their volatility (rolling std), and their
# rate of change.
ANOMALY_FEATURES = [
    "temperature", "power_draw",
    "temp_roll_avg", "power_roll_avg",
    "temp_roll_std", "power_roll_std",
    "temp_rate_of_change", "power_rate_of_change",
]
ISOLATION_FOREST_CONTAMINATION = 0.12
ISOLATION_FOREST_ESTIMATORS = 200
ROLLING_WINDOW_POINTS = 4                   # 1 hour at 15-min interval

# Anomaly score thresholds -> risk label
# (Isolation Forest outputs higher score = more normal; we invert to "risk score" in [0,1])
RISK_WATCH_THRESHOLD = 0.55
RISK_AT_RISK_THRESHOLD = 0.75

# ---------------------------------------------------------------------------
# Idle / Underutilization Detection
# ---------------------------------------------------------------------------
IDLE_UTIL_THRESHOLD = 10                    # percent GPU utilization
IDLE_SUSTAINED_HOURS = 2                    # must stay idle this long to count
IDLE_SUSTAINED_POINTS = int(IDLE_SUSTAINED_HOURS * POINTS_PER_HOUR)
IDLE_MOVING_AVG_WINDOW = POINTS_PER_HOUR    # 1-hour moving average, smooths brief lulls

# ---------------------------------------------------------------------------
# Cost Optimization
# ---------------------------------------------------------------------------
GPU_HOURLY_COST_USD = 2.50                  # configurable on-demand A100-class rate
CONSOLIDATION_SAVINGS_FACTOR = 0.6          # est. % of idle spend recoverable via consolidation

# ---------------------------------------------------------------------------
# Action Ranking & Conflict Resolver
# ---------------------------------------------------------------------------
# Categorical action labels produced by each module (kept as named constants
# so the Health/Cost categorization and the conflict resolver always agree
# on the exact strings being compared).
HEALTH_ACTION_INSPECT = "Inspect Immediately"     # risk_label == At-Risk
HEALTH_ACTION_MONITOR = "Monitor Closely"         # risk_label == Watch
HEALTH_ACTION_NONE = "No Action"                  # risk_label == Healthy

COST_ACTION_SHUTDOWN = "Shutdown"                 # idle_pct_of_window >= SHUTDOWN threshold
COST_ACTION_CONSOLIDATE = "Consolidate"           # idle_pct_of_window >= CONSOLIDATE threshold
COST_ACTION_NONE = "No Action"

# Idle-percentage thresholds that decide which cost action a node gets.
# (Same cutoffs the plain-language recommendation.py module uses, centralized
# here so both modules stay consistent.)
COST_ACTION_SHUTDOWN_IDLE_PCT = 60.0
COST_ACTION_CONSOLIDATE_IDLE_PCT = 20.0

# Value Score = (risk_score * VALUE_SCORE_RISK_WEIGHT)
#             + (min(estimated_saving / VALUE_SCORE_COST_NORMALIZATION_CAP, 1) * VALUE_SCORE_COST_WEIGHT)
# risk_score is already 0-1. estimated_saving (USD) is normalized against a
# configurable cap so both terms live on a comparable 0-1 scale before being
# weighted. Weights should sum to 1.0 for an interpretable 0-1 final score.
VALUE_SCORE_RISK_WEIGHT = 0.6
VALUE_SCORE_COST_WEIGHT = 0.4
VALUE_SCORE_COST_NORMALIZATION_CAP = 100.0        # USD wasted spend that maps to a full 1.0 cost score

# "Safety First" conflict-resolution rule: when the Health and Cost modules
# disagree on the same node, the health recommendation wins if risk_score is
# at or above this threshold; otherwise the cost recommendation wins.
SAFETY_FIRST_RISK_THRESHOLD = 0.60

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
APP_TITLE = "ClusterPulse AI"
APP_ICON = "🖥️"
REFRESH_HELP_TEXT = "Regenerate synthetic telemetry to simulate a new monitoring window."
