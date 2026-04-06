#!/usr/bin/env python3
"""
Final Review Master Demo
=========================
Runs all four final-review experiments in sequence, collects results,
and generates a combined summary report with all figures.

Usage
-----
Quick (fast, fewer episodes — for live demo):
    python run_final_review_demo.py --mode quick

Full (publication-quality runs — run overnight):
    python run_final_review_demo.py --mode full

Custom:
    python run_final_review_demo.py --failure-episodes 20 \
        --comm-episodes 15 --scale-episodes 12 --uavs 5

Experiments run:
    1. Failure Resilience   → results/final_review/failure_resilience/
    2. Communication Degradation → results/final_review/communication_degradation/
    3. Scalability Study    → results/final_review/scalability/
    4. Combined Summary     → results/final_review/summary_report.json
                           → results/final_review/combined_figure.png
"""

import asyncio
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

from loguru import logger
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from experiments.failure_resilience import run_failure_resilience_experiment
from experiments.communication_degradation import run_communication_degradation_experiment
from experiments.scalability_study import run_scalability_study

# ---------------------------------------------------------------------------
OUTPUT_BASE = Path("results/final_review")

QUICK_CONFIG = dict(failure_episodes=5, comm_episodes=5, scale_episodes=5, num_uavs=4)
FULL_CONFIG  = dict(failure_episodes=20, comm_episodes=15, scale_episodes=12, num_uavs=5)


# ---------------------------------------------------------------------------
def _print_banner(text: str):
    bar = "=" * 72
    logger.info(bar)
    logger.info(f"  {text}")
    logger.info(bar)


# ---------------------------------------------------------------------------
async def run_all(
    failure_episodes: int,
    comm_episodes: int,
    scale_episodes: int,
    num_uavs: int,
    gps_denied: bool = False,
):
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "failure_episodes": failure_episodes,
            "comm_episodes":    comm_episodes,
            "scale_episodes":   scale_episodes,
            "num_uavs":         num_uavs,
            "gps_denied":       gps_denied,
        },
        "experiments": {}
    }

    wall_start = time.time()

    # ------------------------------------------------------------------ 1
    _print_banner("EXPERIMENT 1 / 3 — UAV Failure Resilience")
    t0 = time.time()
    fr_dir = str(OUTPUT_BASE / "failure_resilience")
    fr_results = await run_failure_resilience_experiment(
        num_episodes=failure_episodes,
        num_uavs=num_uavs,
        output_dir=fr_dir,
        gps_denied=gps_denied,
    )
    summary["experiments"]["failure_resilience"] = {
        "runtime_s": round(time.time() - t0, 1),
        "key_metrics": {
            name: {
                "final_coverage": res["final_coverage"],
                "final_pw_coverage": res["final_pw_coverage"],
                "recovery_delta": res["recovery_delta"],
            }
            for name, res in fr_results.items()
        }
    }
    logger.success(f"Failure Resilience done in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------ 2
    _print_banner("EXPERIMENT 2 / 3 — Communication Degradation")
    t0 = time.time()
    cd_dir = str(OUTPUT_BASE / "communication_degradation")
    cd_results = await run_communication_degradation_experiment(
        num_episodes=comm_episodes,
        num_uavs=num_uavs,
        output_dir=cd_dir,
    )
    summary["experiments"]["communication_degradation"] = {
        "runtime_s": round(time.time() - t0, 1),
        "key_metrics": {
            "reliability_levels": [r["reliability"] for r in cd_results["mcp"]],
            "mcp_coverage": [r["avg_coverage"] for r in cd_results["mcp"]],
            "baseline_coverage": [r["avg_coverage"] for r in cd_results["baseline"]],
        }
    }
    logger.success(f"Communication Degradation done in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------ 3
    _print_banner("EXPERIMENT 3 / 3 — Scalability Study")
    t0 = time.time()
    sc_dir = str(OUTPUT_BASE / "scalability")
    sc_results = await run_scalability_study(
        num_episodes=scale_episodes,
        output_dir=sc_dir,
    )
    summary["experiments"]["scalability"] = {
        "runtime_s": round(time.time() - t0, 1),
        "key_metrics": {
            "swarm_sizes": [r["num_uavs"] for r in sc_results["mcp"]],
            "mcp_coverage": [r["avg_coverage"] for r in sc_results["mcp"]],
            "baseline_coverage": [r["avg_coverage"] for r in sc_results["baseline"]],
            "agg_times_ms": [r["context_agg_time_ms"] for r in sc_results["mcp"]],
        }
    }
    logger.success(f"Scalability Study done in {time.time()-t0:.1f}s")

    summary["total_runtime_s"] = round(time.time() - wall_start, 1)

    # Save summary JSON
    with open(OUTPUT_BASE / "summary_report.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.success(f"Summary report → {OUTPUT_BASE / 'summary_report.json'}")

    # ------------------------------------------------------------------ combined figure
    _build_combined_figure(fr_results, cd_results, sc_results)

    # ------------------------------------------------------------------ final console summary
    _print_banner("FINAL REVIEW SUMMARY")
    _summarise(fr_results, cd_results, sc_results)

    return summary


# ---------------------------------------------------------------------------
def _build_combined_figure(fr_results, cd_results, sc_results):
    """Six-panel combined figure suitable for the presentation slide."""
    import seaborn as sns
    sns.set_style("whitegrid")

    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

    steps = np.arange(200)

    # ---- Panel A: Failure resilience — MCP coverage curves -------------
    ax_a = fig.add_subplot(gs[0, 0])
    palette = {"MCP – No Failure": "#1a6faf",
               "MCP – 1 Failure": "#2196f3",
               "MCP – 2 Failures": "#90caf9"}
    for name, color in palette.items():
        if name in fr_results:
            mu = np.array(fr_results[name]["avg_coverage"])
            # Downsample or use actual length for x-axis
            x_steps = np.linspace(0, len(mu) - 1, len(mu)).astype(int)
            ax_a.plot(x_steps, mu, color=color, lw=1.8,
                      label=name.split("–")[-1].strip())
    ax_a.axvline(400, color="#ff6f00", ls="--", lw=1.2, alpha=0.8, label="Failure events")
    ax_a.axvline(1200, color="#6a1b9a", ls="--", lw=1.2, alpha=0.8)
    ax_a.set_title("A: Failure Resilience (MCP)", fontsize=11, fontweight="bold")
    ax_a.set_xlabel("Step"); ax_a.set_ylabel("Coverage (%)")
    ax_a.legend(fontsize=7)

    # ---- Panel B: Failure resilience — Baseline coverage curves ---------
    ax_b = fig.add_subplot(gs[0, 1])
    palette_b = {"Baseline – No Failure": "#b71c1c",
                 "Baseline – 1 Failure": "#f44336",
                 "Baseline – 2 Failures": "#ef9a9a"}
    for name, color in palette_b.items():
        if name in fr_results:
            mu = np.array(fr_results[name]["avg_coverage"])
            x_steps = np.linspace(0, len(mu) - 1, len(mu)).astype(int)
            ax_b.plot(x_steps, mu, color=color, lw=1.8,
                      label=name.split("–")[-1].strip())
    ax_b.axvline(400, color="#ff6f00", ls="--", lw=1.2, alpha=0.8)
    ax_b.axvline(1200, color="#6a1b9a", ls="--", lw=1.2, alpha=0.8)
    ax_b.set_title("B: Failure Resilience (Baseline)", fontsize=11, fontweight="bold")
    ax_b.set_xlabel("Step"); ax_b.set_ylabel("Coverage (%)")
    ax_b.legend(fontsize=7)

    # ---- Panel C: Recovery delta bar chart ------------------------------
    ax_c = fig.add_subplot(gs[0, 2])
    scenarios = ["No Failure", "1 Failure", "2 Failures"]
    mcp_final = [fr_results.get(f"MCP – {s}", {}).get("final_coverage", 0) for s in scenarios]
    base_final = [fr_results.get(f"Baseline – {s}", {}).get("final_coverage", 0) for s in scenarios]
    x = np.arange(len(scenarios))
    w = 0.35
    ax_c.bar(x - w/2, mcp_final, w, color="#1a6faf", label="MCP", edgecolor="k", lw=0.7)
    ax_c.bar(x + w/2, base_final, w, color="#b71c1c", label="Baseline", edgecolor="k", lw=0.7)
    ax_c.set_xticks(x); ax_c.set_xticklabels(scenarios, fontsize=8)
    ax_c.set_ylabel("Final Coverage (%)")
    ax_c.set_title("C: Coverage Under Failures", fontsize=11, fontweight="bold")
    ax_c.legend(fontsize=8)

    # ---- Panel D: Comm degradation — coverage vs reliability -----------
    ax_d = fig.add_subplot(gs[1, 0])
    rel_pct = [r["reliability"] * 100 for r in cd_results["mcp"]]
    mcp_cov_cd = [r["avg_coverage"] for r in cd_results["mcp"]]
    base_cov_cd = [r["avg_coverage"] for r in cd_results["baseline"]]
    ax_d.plot(rel_pct, mcp_cov_cd, "o-", color="#1a6faf", lw=2, ms=7,
              label="MCP-Coordinated")
    ax_d.plot(rel_pct, base_cov_cd, "s--", color="#b71c1c", lw=2, ms=7,
              label="Independent Baseline")
    ax_d.invert_xaxis()
    ax_d.set_xlabel("Communication Reliability (%)")
    ax_d.set_ylabel("Coverage (%)")
    ax_d.set_title("D: Graceful Degradation\nUnder Packet Loss", fontsize=11, fontweight="bold")
    ax_d.legend(fontsize=8)

    # ---- Panel E: Scalability — coverage vs swarm size -----------------
    ax_e = fig.add_subplot(gs[1, 1])
    swarm_sizes = [r["num_uavs"] for r in sc_results["mcp"]]
    mcp_cov_sc = [r["avg_coverage"] for r in sc_results["mcp"]]
    base_cov_sc = [r["avg_coverage"] for r in sc_results["baseline"]]
    ax_e.plot(swarm_sizes, mcp_cov_sc, "o-", color="#1a6faf", lw=2, ms=7,
              label="MCP-Coordinated")
    ax_e.plot(swarm_sizes, base_cov_sc, "s--", color="#b71c1c", lw=2, ms=7,
              label="Independent Baseline")
    ax_e.set_xticks(swarm_sizes)
    ax_e.set_xlabel("Swarm Size (# UAVs)")
    ax_e.set_ylabel("Final Coverage (%)")
    ax_e.set_title("E: Scalability — Coverage Growth", fontsize=11, fontweight="bold")
    ax_e.legend(fontsize=8)

    # ---- Panel F: O(n) timing proof --------------------------------
    ax_f = fig.add_subplot(gs[1, 2])
    agg_times = [r["context_agg_time_ms"] for r in sc_results["mcp"]]
    mcp_msgs = [r["mcp_messages_per_step"] for r in sc_results["mcp"]]
    p2p_msgs = [r["p2p_messages_per_step"] for r in sc_results["mcp"]]
    ax_f_twin = ax_f.twinx()
    ax_f.plot(swarm_sizes, agg_times, "D-", color="#5e35b1", lw=2, ms=7,
              label="Agg. time (ms)")
    ax_f_twin.plot(swarm_sizes, mcp_msgs, "o--", color="#2e7d32", lw=1.5, ms=6,
                   label="MCP O(n)")
    ax_f_twin.plot(swarm_sizes, p2p_msgs, "^:", color="#e65100", lw=1.5, ms=6,
                   label="P2P O(n²)")
    ax_f.set_xticks(swarm_sizes)
    ax_f.set_xlabel("Swarm Size (# UAVs)")
    ax_f.set_ylabel("Aggregation Time (ms)", color="#5e35b1")
    ax_f_twin.set_ylabel("Messages / Step")
    ax_f.set_title("F: O(n) Scaling Validation", fontsize=11, fontweight="bold")
    lines1, labels1 = ax_f.get_legend_handles_labels()
    lines2, labels2 = ax_f_twin.get_legend_handles_labels()
    ax_f.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")

    fig.suptitle(
        "MCP-Coordinated Swarm Intelligence — Final Review Results\n"
        "Failure Resilience  |  Communication Degradation  |  Scalability",
        fontsize=14, fontweight="bold", y=1.01
    )

    plt.savefig(OUTPUT_BASE / "combined_figure.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.success(f"Combined figure → {OUTPUT_BASE / 'combined_figure.png'}")


# ---------------------------------------------------------------------------
def _summarise(fr_results, cd_results, sc_results):
    """Print a concise human-readable review summary."""
    logger.info("")

    # Failure resilience
    mcp_no_fail = fr_results.get("MCP – No Failure", {}).get("final_coverage", 0)
    mcp_2_fail  = fr_results.get("MCP – 2 Failures", {}).get("final_coverage", 0)
    base_no_fail = fr_results.get("Baseline – No Failure", {}).get("final_coverage", 0)
    base_2_fail  = fr_results.get("Baseline – 2 Failures", {}).get("final_coverage", 0)
    mcp_drop  = mcp_no_fail  - mcp_2_fail
    base_drop = base_no_fail - base_2_fail

    logger.info("FAILURE RESILIENCE")
    logger.info(f"  MCP:      healthy={mcp_no_fail:.2f}%  → 2 failures={mcp_2_fail:.2f}%  Δ={-mcp_drop:+.2f}%")
    logger.info(f"  Baseline: healthy={base_no_fail:.2f}% → 2 failures={base_2_fail:.2f}% Δ={-base_drop:+.2f}%")
    resilience_advantage = base_drop - mcp_drop
    logger.info(f"  MCP is {resilience_advantage:+.2f}% more resilient to dual-UAV failure")

    # Comm degradation
    logger.info("")
    logger.info("COMMUNICATION DEGRADATION")
    mcp_100 = next((r["avg_coverage"] for r in cd_results["mcp"] if r["reliability"]==1.0), None)
    mcp_020 = next((r["avg_coverage"] for r in cd_results["mcp"] if r["reliability"]==0.2), None)
    base_100 = next((r["avg_coverage"] for r in cd_results["baseline"] if r["reliability"]==1.0), None)
    if mcp_100 and mcp_020:
        logger.info(f"  MCP coverage: 100% comms={mcp_100:.2f}%  →  20% comms={mcp_020:.2f}%  "
                    f"(only {mcp_100-mcp_020:.2f}% drop)")
    if base_100:
        logger.info(f"  Baseline (no context sharing) stays flat at ≈{base_100:.2f}% regardless of comms")
    logger.info(f"  MCP advantage at 100% comms: {((mcp_100 or 0)-(base_100 or 0)):+.2f}%")

    # Scalability
    logger.info("")
    logger.info("SCALABILITY")
    for r_mcp, r_base in zip(sc_results["mcp"], sc_results["baseline"]):
        gap = r_mcp["avg_coverage"] - r_base["avg_coverage"]
        logger.info(f"  {r_mcp['num_uavs']} UAVs: MCP={r_mcp['avg_coverage']:.2f}%  "
                    f"Base={r_base['avg_coverage']:.2f}%  gap={gap:+.2f}%  "
                    f"agg={r_mcp['context_agg_time_ms']:.2f}ms")

    logger.info("")
    logger.info("All results, plots, and JSON data saved under results/final_review/")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Final Review Master Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  quick  — fast demo (5 episodes each, ~5-8 min total)
  full   — publication-quality (20/15/12 episodes, ~30-40 min total)

Examples:
  python run_final_review_demo.py --mode quick
  python run_final_review_demo.py --mode full
  python run_final_review_demo.py --mode full --gps-denied
  python run_final_review_demo.py --failure-episodes 20 --comm-episodes 15 \\
      --scale-episodes 12 --uavs 5
"""
    )
    parser.add_argument("--mode", choices=["quick", "full"], default=None,
                        help="Preset mode (overrides individual episode counts)")
    parser.add_argument("--failure-episodes", type=int, default=None)
    parser.add_argument("--comm-episodes",    type=int, default=None)
    parser.add_argument("--scale-episodes",   type=int, default=None)
    parser.add_argument("--uavs",             type=int, default=None)
    parser.add_argument("--gps-denied", action="store_true",
                        help="Simulate GPS-denied SLAM positioning (adds decaying "
                             "Gaussian noise to UAV position estimates)")
    args = parser.parse_args()

    if args.mode == "quick":
        cfg = QUICK_CONFIG.copy()
    elif args.mode == "full":
        cfg = FULL_CONFIG.copy()
    else:
        cfg = FULL_CONFIG.copy()

    if args.failure_episodes is not None: cfg["failure_episodes"] = args.failure_episodes
    if args.comm_episodes     is not None: cfg["comm_episodes"]     = args.comm_episodes
    if args.scale_episodes    is not None: cfg["scale_episodes"]    = args.scale_episodes
    if args.uavs              is not None: cfg["num_uavs"]          = args.uavs
    cfg["gps_denied"] = args.gps_denied

    logger.info(f"Configuration: {cfg}")
    asyncio.run(run_all(**cfg))
