"""
UAV Failure Resilience Experiment
==================================
This experiment directly validates the core MCP thesis:

  "Decentralized MCP coordination provides resilience against individual
   UAV failures — something centralized and context-free approaches cannot."

Experiment Design
-----------------
Two swarm types are compared:
  1. MCP-Coordinated  – UAVs share a live coverage map via MCP context.
     When a peer fails its coverage gap is visible to survivors who then
     redirect toward uncovered areas.
  2. Independent (Baseline) – UAVs act without shared context.  They do
     not detect the gap left by a failed peer and continue their existing
     exploration patterns, leaving zones permanently uncovered.

Three failure scenarios per swarm type:
  A. No failures       (control)
  B. 1 UAV fails early (step 50  of 200)
  C. 2 UAVs fail       (steps 50 and 120 of 200)

Metrics tracked per step:
  - Total coverage percentage
  - Priority-weighted coverage (high-severity zones count more)
  - Fraction of active UAVs
  - Coverage recovery rate after each failure event

Outputs (saved to results/failure_resilience/):
  - failure_resilience_results.json  — raw numbers
  - coverage_over_time.png           — coverage curves with failure markers
  - recovery_comparison.png          — delta-coverage after each failure
  - priority_coverage.png            — weighted coverage comparison
"""

import asyncio
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats as scipy_stats
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.simulation_config import SimulationConfig
from simulation.environment import SwarmEnvironment
from experiments.exploration_agents import (
    make_baseline_explorers, make_mcp_explorers,
    get_actions_baseline, get_actions_mcp,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# 1000 steps × 10m-cell grid × sensor 50m radius → ~35-55% coverage per episode.
# Failure steps placed at 20% and 60% of the episode to give recovery time.
STEPS_PER_EPISODE = 10_000
FAILURE_STEP_1 = 2_000   # First failure  (20% into episode)
FAILURE_STEP_2 = 6_000   # Second failure (60% into episode)
NUM_EPISODES   = 20      # Averages per scenario — 20 paired trials for statistical significance
SWARM_SIZE     = 5
BASE_SEED      = 100     # Fixed base seed — MCP and Baseline share identical starting conditions
# Realistic scout UAV parameters for this experiment
_UAV_MAX_SPEED    = 15.0    # m/s (scout drone cruise speed)
_UAV_BATTERY_DRAIN = 0.0001 # %/step drain: 0.0001 × 0.1 dt × 10,000 steps = 10% total
_UAV_SENSOR_RANGE  = 50.0   # m — wider footprint for aerial survey


# ---------------------------------------------------------------------------
# Core: run one scenario
# ---------------------------------------------------------------------------
async def _run_scenario(
    scenario_name: str,
    failure_schedule: Dict[int, List[int]],
    num_episodes: int,
    num_uavs: int,
    use_mcp_context: bool,
    gps_denied: bool = False,
) -> Dict[str, Any]:
    """Run a failure scenario and return step-by-step metrics (averaged over episodes)."""

    config = SimulationConfig()
    config.num_uavs = num_uavs
    config.render = False
    config.rl_config.max_steps_per_episode = STEPS_PER_EPISODE
    config.rl_config.max_episode_length = STEPS_PER_EPISODE  # truncation uses this field
    # Override UAV parameters for realistic scout-drone behaviour
    config.uav_config.max_speed = _UAV_MAX_SPEED
    config.uav_config.battery_drain_rate = _UAV_BATTERY_DRAIN
    config.uav_config.sensor_range = _UAV_SENSOR_RANGE

    step_coverage: List[List[float]] = []
    step_pw_coverage: List[List[float]] = []
    active_uav_fractions: List[List[float]] = []
    failure_steps = sorted(failure_schedule.keys())

    for ep in range(num_episodes):
        env = SwarmEnvironment(config, mcp_server_url=None)
        obs, _ = env.reset(seed=BASE_SEED + ep)  # fixed seed: paired comparison
        # Disable mission-complete early termination: we measure full-episode coverage,
        # not just the tiny default target areas.  Truncation (step limit) still applies.
        env._check_termination = lambda: False

        if use_mcp_context:
            explorers = make_mcp_explorers(env, gps_denied=gps_denied)
        else:
            explorers = make_baseline_explorers(env, gps_denied=gps_denied)

        ep_coverage, ep_pw_coverage, ep_active = [], [], []

        for step in range(STEPS_PER_EPISODE):
            # Inject failures at the scheduled steps
            if step in failure_schedule:
                for uav_idx in failure_schedule[step]:
                    env.inject_failure(uav_idx)

            if use_mcp_context:
                actions = get_actions_mcp(explorers, env)
            else:
                actions = get_actions_baseline(explorers, env)

            obs, _reward, done, truncated, info = env.step(actions)

            summary = info.get("scenario_summary", {})
            ep_coverage.append(summary.get("coverage_percentage", 0.0))
            ep_pw_coverage.append(summary.get("priority_weighted_coverage", 0.0))
            active = (num_uavs - len(env.failed_uav_ids)) / num_uavs
            ep_active.append(active)

            if truncated:  # step-limit hit — episode complete
                break
            if done:  # should not happen (termination disabled), but guard anyway
                break

        env.close()
        step_coverage.append(ep_coverage[:STEPS_PER_EPISODE])
        step_pw_coverage.append(ep_pw_coverage[:STEPS_PER_EPISODE])
        active_uav_fractions.append(ep_active[:STEPS_PER_EPISODE])

        logger.info(f"  [{scenario_name}] ep {ep+1}/{num_episodes} "
                    f"cov={ep_coverage[-1]:.2f}% pw={ep_pw_coverage[-1]:.2f}%")

    avg_coverage    = np.mean(step_coverage, axis=0).tolist()
    avg_pw_coverage = np.mean(step_pw_coverage, axis=0).tolist()
    avg_active      = np.mean(active_uav_fractions, axis=0).tolist()
    std_coverage    = np.std(step_coverage, axis=0).tolist()

    # Per-episode final values — needed for paired t-tests
    per_ep_final_cov = [ep[-1] for ep in step_coverage]
    per_ep_final_pw  = [ep[-1] for ep in step_pw_coverage]

    recovery = {}
    for fs in failure_steps:
        if fs + 100 < STEPS_PER_EPISODE:
            recovery[fs] = avg_pw_coverage[fs + 100] - avg_pw_coverage[fs]

    return {
        "scenario":              scenario_name,
        "failure_schedule":      {str(k): v for k, v in failure_schedule.items()},
        "avg_coverage":          avg_coverage,
        "avg_pw_coverage":       avg_pw_coverage,
        "std_coverage":          std_coverage,
        "avg_active_fraction":   avg_active,
        "final_coverage":        float(np.mean(per_ep_final_cov)),
        "final_pw_coverage":     float(np.mean(per_ep_final_pw)),
        "per_ep_final_coverage": per_ep_final_cov,
        "per_ep_final_pw":       per_ep_final_pw,
        "recovery_delta":        recovery,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
async def run_failure_resilience_experiment(
    num_episodes: int = NUM_EPISODES,
    num_uavs: int = SWARM_SIZE,
    output_dir: str = "results/failure_resilience",
    gps_denied: bool = False,
) -> Dict[str, Any]:
    od = Path(output_dir)
    od.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    mode_label = " [GPS-DENIED / SLAM MODE]" if gps_denied else ""
    logger.info(f"UAV FAILURE RESILIENCE EXPERIMENT{mode_label}")
    logger.info("=" * 70)

    scenarios = [
        ("MCP – No Failure",       {},                         True),
        ("MCP – 1 Failure",        {FAILURE_STEP_1: [0]},      True),
        ("MCP – 2 Failures",       {FAILURE_STEP_1: [0],
                                    FAILURE_STEP_2: [1]},      True),
        ("Baseline – No Failure",  {},                         False),
        ("Baseline – 1 Failure",   {FAILURE_STEP_1: [0]},      False),
        ("Baseline – 2 Failures",  {FAILURE_STEP_1: [0],
                                    FAILURE_STEP_2: [1]},      False),
    ]

    all_results = {}
    for name, sched, use_mcp in scenarios:
        logger.info(f"\nRunning: {name}")
        result = await _run_scenario(name, sched, num_episodes, num_uavs, use_mcp,
                                     gps_denied=gps_denied)
        all_results[name] = result

    # ------------------------------------------------------------------
    # Persist raw results
    # ------------------------------------------------------------------
    results_file = od / "failure_resilience_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.success(f"Results saved → {results_file}")

    # ------------------------------------------------------------------
    # Plot 1: Coverage over time with failure markers
    # ------------------------------------------------------------------
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    palette_mcp = ["#1a6faf", "#2196f3", "#90caf9"]
    palette_base = ["#b71c1c", "#f44336", "#ef9a9a"]
    steps = np.arange(STEPS_PER_EPISODE)

    for ax, (group, palette, label) in zip(
        axes,
        [
            (["MCP – No Failure", "MCP – 1 Failure", "MCP – 2 Failures"], palette_mcp, "MCP-Coordinated"),
            (["Baseline – No Failure", "Baseline – 1 Failure", "Baseline – 2 Failures"], palette_base, "Independent (Baseline)"),
        ]
    ):
        for scenario_name, color in zip(group, palette):
            res = all_results[scenario_name]
            mu = np.array(res["avg_coverage"])
            sd = np.array(res["std_coverage"])
            short = scenario_name.split("–")[-1].strip()
            ax.plot(steps, mu, color=color, linewidth=2, label=short)
            ax.fill_between(steps, mu - sd, mu + sd, alpha=0.15, color=color)

        # Failure markers
        for fs, color, lbl in [
            (FAILURE_STEP_1, "#ff6f00", "Failure 1 (step 50)"),
            (FAILURE_STEP_2, "#6a1b9a", "Failure 2 (step 120)")
        ]:
            ax.axvline(x=fs, linestyle="--", color=color, linewidth=1.5, alpha=0.8, label=lbl)

        ax.set_title(f"{label}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Simulation Step", fontsize=11)
        ax.set_ylabel("Coverage (%)", fontsize=11)
        ax.legend(fontsize=9)

    plt.suptitle("UAV Failure Resilience: Coverage Over Time\n"
                 "MCP-Coordinated vs Independent Baseline",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(od / "coverage_over_time.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.success("Saved coverage_over_time.png")

    # ------------------------------------------------------------------
    # Plot 2: Coverage recovery after each failure (bar chart)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_labels = ["After Failure 1\n(30 steps)", "After Failure 2\n(30 steps)"]
    failure_keys = [FAILURE_STEP_1, FAILURE_STEP_2]

    width = 0.35
    x = np.arange(len(bar_labels))

    mcp_recovery = [
        all_results["MCP – 2 Failures"]["recovery_delta"].get(str(k), 0)
        for k in failure_keys
    ]
    base_recovery = [
        all_results["Baseline – 2 Failures"]["recovery_delta"].get(str(k), 0)
        for k in failure_keys
    ]

    bars_mcp = ax.bar(x - width / 2, mcp_recovery, width, label="MCP-Coordinated",
                      color="#2196f3", edgecolor="black", linewidth=0.7)
    bars_base = ax.bar(x + width / 2, base_recovery, width, label="Independent Baseline",
                       color="#f44336", edgecolor="black", linewidth=0.7)

    for bar in bars_mcp:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05,
                f"{h:+.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars_base:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05,
                f"{h:+.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels)
    ax.set_ylabel("Coverage Change (%) in 30 Steps Post-Failure", fontsize=11)
    ax.set_title("Coverage Recovery Rate After UAV Failure\n"
                 "Positive values = swarm adapts and fills gap",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(od / "recovery_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.success("Saved recovery_comparison.png")

    # ------------------------------------------------------------------
    # Plot 3: Priority-weighted coverage (bar chart – final values)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    scenario_labels = ["No Failure", "1 Failure", "2 Failures"]
    mcp_pw = [all_results[f"MCP – {s}"]["final_pw_coverage"] for s in scenario_labels]
    base_pw = [all_results[f"Baseline – {s}"]["final_pw_coverage"] for s in scenario_labels]

    x = np.arange(len(scenario_labels))
    b_mcp = ax.bar(x - width / 2, mcp_pw, width, label="MCP-Coordinated",
                   color="#1a6faf", edgecolor="black", linewidth=0.7)
    b_base = ax.bar(x + width / 2, base_pw, width, label="Independent Baseline",
                    color="#b71c1c", edgecolor="black", linewidth=0.7)

    for bar in list(b_mcp) + list(b_base):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                f"{h:.2f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels)
    ax.set_ylabel("Priority-Weighted Coverage (%)", fontsize=11)
    ax.set_title("Priority-Weighted Coverage: High-Severity Zones First\n"
                 "Shows whether swarm prioritises critical disaster areas",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(od / "priority_coverage.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.success("Saved priority_coverage.png")

    # ------------------------------------------------------------------
    # Summary table + Statistical Significance (paired t-tests)
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("FAILURE RESILIENCE SUMMARY  [PRIMARY: Priority-Weighted Coverage]")
    logger.info("=" * 70)
    header = f"{'Scenario':<30} {'PW Cov %':>10} {'Raw Cov %':>10} {'95% CI':>14}"
    logger.info(header)
    logger.info("-" * 70)
    for name, res in all_results.items():
        pw_vals = res["per_ep_final_pw"]
        n = len(pw_vals)
        se = scipy_stats.sem(pw_vals)
        ci = se * scipy_stats.t.ppf(0.975, df=n - 1)
        logger.info(f"{name:<30} {res['final_pw_coverage']:>9.2f}% {res['final_coverage']:>9.2f}%"
                    f"  ±{ci:.2f}%")

    logger.info("\n--- PAIRED t-TESTS (MCP vs Baseline, same failure scenario) ---")
    for scenario in ["No Failure", "1 Failure", "2 Failures"]:
        mcp_pw   = all_results[f"MCP – {scenario}"]["per_ep_final_pw"]
        base_pw  = all_results[f"Baseline – {scenario}"]["per_ep_final_pw"]
        diff     = [m - b for m, b in zip(mcp_pw, base_pw)]
        t_stat, p_val = scipy_stats.ttest_rel(mcp_pw, base_pw)
        n    = len(diff)
        se   = scipy_stats.sem(diff)
        ci   = se * scipy_stats.t.ppf(0.975, df=n - 1)
        mean_diff = float(np.mean(diff))
        sig  = "**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns")
        logger.info(f"  {scenario:<15}: Δ={mean_diff:+.2f}% ±{ci:.2f}%  "
                    f"t={t_stat:.3f}  p={p_val:.4f}  {sig}")
    logger.info("  (* p<0.05  ** p<0.01  ns = not significant)")
    logger.info("=" * 70)
    return all_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="UAV Failure Resilience Experiment")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES,
                        help="Episodes per scenario (default: %(default)s)")
    parser.add_argument("--uavs", type=int, default=SWARM_SIZE,
                        help="Number of UAVs in swarm (default: %(default)s)")
    parser.add_argument("--output", type=str, default="results/failure_resilience",
                        help="Output directory (default: %(default)s)")
    args = parser.parse_args()

    asyncio.run(run_failure_resilience_experiment(
        num_episodes=args.episodes,
        num_uavs=args.uavs,
        output_dir=args.output,
    ))
