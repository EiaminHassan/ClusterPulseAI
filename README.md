# 🖥️ ClusterPulse AI

**AI-Powered Predictive Monitoring & Cost Optimization for Compute Clusters**
Built for the AI for Cluster Intelligence Hackathon.

ClusterPulse AI sits on top of cluster telemetry and interprets it — flagging
nodes at risk of failure and identifying wasted GPU capacity — then
translates findings into clear, actionable recommendations instead of raw
charts.

## What it does

- **Node Health & Failure-Risk Detection** — an unsupervised Isolation
  Forest trained on thermal/power stability features (not raw utilization,
  so idle nodes aren't confused with at-risk ones) flags nodes drifting
  toward failure before they cause downtime.
- **Idle Resource & Cost Optimization** — a transparent rule-based +
  moving-average check flags GPUs sitting idle for sustained windows and
  estimates the wasted spend and recoverable savings.
- **Interactive Dashboard** — Streamlit UI with cluster-wide overview,
  per-node health drill-down, cost report, and a prioritized, plain-language
  recommendations feed.

Telemetry is synthetically generated (with a schema matching Prometheus/DCGM
exporters) so the whole pipeline runs end-to-end with zero external
dependencies — no real cluster required for the demo.

## Project structure

```
ClusterPulseAI/
├── app.py                      # Streamlit entry point
├── requirements.txt
├── data/                       # generated telemetry (created on first run)
├── src/
│   ├── config.py                # all tunable parameters live here
│   ├── generator.py             # synthetic telemetry generator
│   ├── preprocessing.py         # cleaning + feature engineering
│   ├── anomaly_detection.py     # Isolation Forest failure-risk model (Health Module)
│   ├── idle_detection.py        # rule-based idle/underutilization detection (Cost Module)
│   ├── recommendation.py        # turns model output into plain-language actions
│   └── decision_engine.py       # Action Ranking & Conflict Resolver (sits after Health/Cost, before dashboard)
└── dashboard/
    ├── charts.py                 # reusable Plotly chart builders
    ├── overview.py                # cluster-wide KPI page
    ├── health.py                  # node health drill-down page
    ├── cost.py                    # idle / cost optimization page
    ├── recommendations.py         # prioritized action list page
    └── action_center.py           # ranked, conflict-resolved action list page
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. On first run it generates a
synthetic 7-day telemetry window for 20 nodes automatically (saved to
`data/`). Use the sidebar to adjust cluster size, simulated window length,
idle-detection thresholds, and GPU-hour cost, or click **Regenerate
telemetry** to simulate a fresh monitoring window.

## Running the pipeline without the UI

Each module in `src/` is runnable standalone for quick inspection:

```bash
python -m src.generator          # generate + save synthetic telemetry
python -m src.preprocessing      # inspect engineered features
python -m src.anomaly_detection  # print per-node health summary
python -m src.idle_detection     # print per-node idle/cost summary
python -m src.recommendation     # print full recommendation feed + cluster KPIs
python -m src.decision_engine    # print ranked, conflict-resolved action list
```

## How the models work

**Failure-risk detection.** An Isolation Forest (scikit-learn) is trained
on temperature, power draw, their rolling averages, their rolling
volatility (std), and their rate of change. It isolates points that are
"few and different" from normal cluster behavior. Raw GPU/CPU utilization
is deliberately excluded from this feature set — utilization is naturally
low for idle nodes, and including it would cause idle (cost) nodes to be
misclassified as failure risks. The resulting anomaly score is normalized
to a 0–1 `risk_score` and bucketed into **Healthy / Watch / At-Risk**.

**Idle detection.** A GPU is flagged idle if utilization stays below a
configurable threshold (default 10%) for a sustained window (default 2+
hours), confirmed by a 1-hour moving average to filter out brief lulls.
This is deliberately rule-based rather than ML-driven: idle detection
doesn't need model sophistication to be convincing, and a transparent rule
is easier to defend when asked "how did you calculate that?"

**Cost estimation.** `estimated_wasted_usd = idle_hours × gpu_hour_cost`,
a fully transparent formula using a configurable on-demand GPU-hour rate.

**Action Ranking & Conflict Resolver.** The Health module and Cost module
score every node independently, and can recommend different actions for the
same node (e.g. Health says "inspect this node," Cost says "shut it down").
`src/decision_engine.py` sits between those two modules and the dashboard:
it wraps both modules' outputs into a unified format, computes a transparent
`Value Score = risk_score × 0.6 + normalized(estimated_saving) × 0.4` per
node, flags any node where both modules recommend a real (non-"No Action")
but different action, and resolves every conflict with one explainable
**Safety First** rule: if `risk_score >= 0.60`, the health recommendation
wins; otherwise the cost recommendation wins. Every resolved node gets a
one-sentence, rule-based plain-language reason — no LLM required. The
synthetic generator includes one guaranteed "conflict demo" node per run
(idle on utilization the whole window, but developing a thermal/power fault
over time) so there's always at least one real conflict to show in a demo.

## Roadmap (beyond hackathon scope)

- Natural-language assistant for ad-hoc questions ("Why is node-3 unhealthy?")
- Live ingestion from Prometheus / NVIDIA DCGM exporters (schema is already
  designed to match)
- Multi-cluster support and periodic model retraining on live data
- Closed-loop actions (autoscaling APIs) instead of advisory-only alerts

## Tech stack

Python · pandas · NumPy · scikit-learn (Isolation Forest) · Streamlit · Plotly
