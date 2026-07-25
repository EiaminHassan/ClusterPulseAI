"""
ClusterPulse AI - Synthetic Telemetry Generator

Produces per-node time-series telemetry (GPU/CPU utilization, memory,
temperature, power draw) with the same schema Prometheus/DCGM exporters
use, so a real data source can later be substituted with minimal changes.

Three node archetypes are generated:
  - healthy    : normal fluctuation around typical operating bands
  - pre_failure: gradually rising temperature + unstable power draw over time
  - idle       : held at low utilization for extended windows
"""

import numpy as np
import pandas as pd

from . import config as cfg


def _node_archetypes(num_nodes, rng):
    """Assign each node an archetype: healthy, pre_failure, idle, or critical_idle."""
    n_conflict_demo = min(cfg.CONFLICT_DEMO_NODE_COUNT, num_nodes)
    remaining = num_nodes - n_conflict_demo
    n_failure = max(1, int(remaining * cfg.FAILURE_NODE_FRACTION))
    n_idle = max(1, int(remaining * cfg.IDLE_NODE_FRACTION))
    n_healthy = remaining - n_failure - n_idle
    if n_healthy < 0:
        raise ValueError("FAILURE_NODE_FRACTION + IDLE_NODE_FRACTION exceed 1.0")

    archetypes = (
        ["healthy"] * n_healthy
        + ["pre_failure"] * n_failure
        + ["idle"] * n_idle
        + ["critical_idle"] * n_conflict_demo
    )
    rng.shuffle(archetypes)
    return archetypes


def _generate_healthy_series(n_points, rng):
    gpu_util = rng.uniform(*cfg.BASELINE_GPU_UTIL, n_points)
    cpu_util = rng.uniform(*cfg.BASELINE_CPU_UTIL, n_points)
    mem_util = rng.uniform(*cfg.BASELINE_MEMORY_UTIL, n_points)
    temperature = rng.uniform(*cfg.BASELINE_TEMPERATURE, n_points)
    power_draw = rng.uniform(*cfg.BASELINE_POWER_DRAW, n_points)

    # smooth with a light random walk so consecutive points aren't pure noise
    for arr, jitter in [
        (gpu_util, 3), (cpu_util, 3), (mem_util, 2), (temperature, 1.5), (power_draw, 8),
    ]:
        walk = np.cumsum(rng.normal(0, jitter, n_points))
        arr += walk - walk.mean()

    return gpu_util, cpu_util, mem_util, temperature, power_draw


def _generate_pre_failure_series(n_points, rng):
    """Gradually rising temperature and unstable (increasingly erratic) power draw."""
    gpu_util, cpu_util, mem_util, temperature, power_draw = _generate_healthy_series(
        n_points, rng
    )

    # The degradation ramps up over the back half of the window, mimicking
    # a node that starts healthy and drifts toward failure.
    onset = int(n_points * rng.uniform(0.3, 0.5))
    ramp = np.zeros(n_points)
    ramp[onset:] = np.linspace(0, 1, n_points - onset) ** 1.5

    temperature += ramp * rng.uniform(15, 25)          # up to +15-25 C by the end
    instability = rng.normal(0, 1, n_points) * ramp * rng.uniform(30, 60)
    power_draw += instability + ramp * rng.uniform(20, 40)
    gpu_util += ramp * rng.uniform(-10, 5)              # utilization often dips/erratic near failure

    temperature = np.clip(temperature, 20, 110)
    power_draw = np.clip(power_draw, 50, 500)
    return gpu_util, cpu_util, mem_util, temperature, power_draw


def _generate_idle_series(n_points, rng):
    """Chronically idle GPU: low utilization for extended synthetic windows."""
    gpu_util, cpu_util, mem_util, temperature, power_draw = _generate_healthy_series(
        n_points, rng
    )

    # Choose 1-3 idle windows covering a large majority of the timeline
    n_windows = rng.integers(1, 4)
    idle_mask = np.zeros(n_points, dtype=bool)
    remaining = n_points
    for _ in range(n_windows):
        if remaining <= 0:
            break
        start = rng.integers(0, n_points)
        length = rng.integers(int(n_points * 0.15), int(n_points * 0.4) + 1)
        end = min(n_points, start + length)
        idle_mask[start:end] = True

    # Force overall idle coverage to be high (this archetype = "chronically idle")
    if idle_mask.mean() < 0.5:
        idle_mask[: int(n_points * 0.6)] = True

    gpu_util[idle_mask] = rng.uniform(0, 8, idle_mask.sum())
    cpu_util[idle_mask] = rng.uniform(2, 15, idle_mask.sum())
    power_draw[idle_mask] = rng.uniform(40, 90, idle_mask.sum())   # idle draw much lower
    temperature[idle_mask] = rng.uniform(30, 45, idle_mask.sum())  # cooler when idle

    return gpu_util, cpu_util, mem_util, temperature, power_draw


def _generate_critical_idle_series(n_points, rng):
    """
    'Conflict demo' archetype: a node that is idle on utilization for the
    ENTIRE window (so the Cost module confidently recommends Shutdown) while
    ALSO developing thermal/power instability over time (so the Health module
    confidently recommends Inspect Immediately). Physically this reads as a
    real scenario: a GPU sitting idle with a stuck fan or failing cooling
    loop -- nobody is running jobs on it, but it's quietly overheating.
    This guarantees the Action Ranking & Conflict Resolver always has at
    least one real Health-vs-Cost disagreement to resolve in a demo.
    """
    gpu_util, cpu_util, mem_util, temperature, power_draw = _generate_healthy_series(
        n_points, rng
    )

    # Force near-zero utilization for the whole window (unlike the regular
    # "idle" archetype, which only idles in windows -- this one never runs a job).
    gpu_util = rng.uniform(0, 6, n_points)
    cpu_util = rng.uniform(2, 12, n_points)

    # Idle baseline power draw, then inject a rising, erratic thermal/power
    # fault starting early in the window so it has room to become severe by
    # the end -- this archetype exists purely to guarantee a demo conflict,
    # so the fault is deliberately pronounced rather than subtle.
    onset = int(n_points * rng.uniform(0.2, 0.3))
    ramp = np.zeros(n_points)
    ramp[onset:] = np.linspace(0, 1, n_points - onset) ** 1.2

    temperature = rng.uniform(30, 45, n_points) + ramp * rng.uniform(55, 70)
    power_draw = rng.uniform(40, 90, n_points)
    power_draw += rng.normal(0, 1, n_points) * ramp * rng.uniform(70, 100)
    power_draw += ramp * rng.uniform(40, 60)

    temperature = np.clip(temperature, 20, 110)
    power_draw = np.clip(power_draw, 20, 500)
    return gpu_util, cpu_util, mem_util, temperature, power_draw


_GENERATORS = {
    "healthy": _generate_healthy_series,
    "pre_failure": _generate_pre_failure_series,
    "idle": _generate_idle_series,
    "critical_idle": _generate_critical_idle_series,
}


def generate_cluster_telemetry(
    num_nodes=cfg.NUM_NODES,
    days=cfg.SIMULATION_DAYS,
    interval_minutes=cfg.INTERVAL_MINUTES,
    seed=cfg.RANDOM_SEED,
):
    """
    Generate synthetic cluster telemetry.

    Returns
    -------
    telemetry_df : DataFrame with columns
        [timestamp, node_id, gpu_util, cpu_util, memory_util, temperature, power_draw]
    node_meta_df : DataFrame with columns [node_id, archetype]
        (ground-truth labels, kept separately -- NOT used by the models,
        only for optional evaluation / demo narration)
    """
    rng = np.random.default_rng(seed)

    n_points = int(days * 24 * 60 / interval_minutes)
    timestamps = pd.date_range(
        end=pd.Timestamp.now().floor("min"), periods=n_points, freq=f"{interval_minutes}min"
    )

    node_ids = [f"node-{i:02d}" for i in range(1, num_nodes + 1)]
    archetypes = _node_archetypes(num_nodes, rng)

    rows = []
    for node_id, archetype in zip(node_ids, archetypes):
        gen_fn = _GENERATORS[archetype]
        gpu_util, cpu_util, mem_util, temperature, power_draw = gen_fn(n_points, rng)

        node_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "node_id": node_id,
                "gpu_util": np.clip(gpu_util, 0, 100),
                "cpu_util": np.clip(cpu_util, 0, 100),
                "memory_util": np.clip(mem_util, 0, 100),
                "temperature": np.clip(temperature, 20, 110),
                "power_draw": np.clip(power_draw, 20, 500),
            }
        )
        rows.append(node_df)

    telemetry_df = pd.concat(rows, ignore_index=True)
    telemetry_df = telemetry_df.sort_values(["node_id", "timestamp"]).reset_index(drop=True)

    node_meta_df = pd.DataFrame({"node_id": node_ids, "archetype": archetypes})

    return telemetry_df, node_meta_df


def generate_and_save(**kwargs):
    """Generate telemetry and persist to CSV under DATA_DIR. Returns both DataFrames."""
    telemetry_df, node_meta_df = generate_cluster_telemetry(**kwargs)
    telemetry_df.to_csv(cfg.TELEMETRY_FILE, index=False)
    node_meta_df.to_csv(cfg.NODE_META_FILE, index=False)
    return telemetry_df, node_meta_df


if __name__ == "__main__":
    tdf, mdf = generate_and_save()
    print(f"Generated {len(tdf)} telemetry rows across {mdf.shape[0]} nodes.")
    print(f"Saved to: {cfg.TELEMETRY_FILE}")
    print(mdf["archetype"].value_counts())
