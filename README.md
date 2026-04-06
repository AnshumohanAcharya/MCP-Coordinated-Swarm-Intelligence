# MCP-Coordinated Swarm Intelligence
## Adaptive UAV Path Planning for Dynamic Disaster Response

> B.Tech Final Review Project — IIIT Kottayam  
> Demonstrates that **Model Context Protocol (MCP) shared-context coordination**
> statistically outperforms independent (baseline) UAV swarms under failure,
> communication degradation, and GPS-denied SLAM conditions.

---

## Table of Contents

1. [What We Built](#1-what-we-built)
2. [Why Rule-Based Agents? (Not Trained RL)](#2-why-rule-based-agents-not-trained-rl)
3. [Architecture](#3-architecture)
4. [Quick Start](#4-quick-start)
5. [Experiments](#5-experiments)
6. [Key Results](#6-key-results)
7. [GPS-Denied SLAM Mode](#7-gps-denied-slam-mode)
8. [Real Dataset Integration](#8-real-dataset-integration)
9. [Project Structure](#9-project-structure)
10. [Dependencies](#10-dependencies)

---

## 1. What We Built

A simulation platform where **5 UAVs** must coordinate to survey a 1 km × 1 km
wildfire-affected region modelled after real NASA FIRMS fire-event geometry.

**Three controlled experiments** isolate the coordination benefit of MCP:

| Experiment | Question answered |
|---|---|
| UAV Failure Resilience | Does swarm-level context sharing preserve coverage when UAVs fail? |
| Communication Degradation | How does performance degrade as packet loss increases (1.0 → 0.2 reliability)? |
| Scalability Study | Does MCP message complexity stay O(n) as the swarm grows (3–10 UAVs)? |

Each experiment runs **10 000-step missions** with **fixed seeds** for paired
statistical comparison. A **paired t-test** with 95 % confidence interval is
reported for every metric.

---

## 2. Why Rule-Based Agents? (Not Trained RL)

**Short answer:** rule-based agents isolate the *coordination variable* cleanly.

| Approach | Problem |
|---|---|
| Untrained PPO | Random walk → 1–3 % coverage; noise drowns coordination signal entirely |
| Fully-trained PPO | Agents learn *using* MCP context → difference includes policy quality, not just coordination |
| **Rule-based (ours)** | Identical greedy algorithm for both arms; only variable is whether MCP context is shared |

We *can* train RL agents — `make train-agents` does a 200-episode PPO warm-up
to demonstrate the pipeline. However, rule-based comparison is the methodologically
correct choice for the thesis claim: *MCP coordination alone improves coverage*.

The real RL contribution is in Review III: five algorithms (PPO, SAC, TD3, A2C,
DDPG) were compared, with SAC achieving the best asymptotic coverage.

---

## 3. Architecture

```
┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐
│   UAV Agent 1     │  │   UAV Agent 2     │  │   UAV Agent N     │
│  MCPExplorer      │  │  MCPExplorer      │  │  MCPExplorer      │
│  (sector-based)   │  │  (sector-based)   │  │  (sector-based)   │
└────────┬──────────┘  └────────┬──────────┘  └────────┬──────────┘
         │                      │                      │
         └──────────────────────┼──────────────────────┘
                                │  MCP shared context
                   ┌────────────▼────────────┐
                   │      MCP Server         │
                   │  context_manager.py     │
                   │  message_protocol.py    │
                   └────────────┬────────────┘
                                │
                   ┌────────────▼────────────┐
                   │  SwarmEnvironment       │
                   │  (Gymnasium-compatible) │
                   │  100×100 grid, 10 m/cell│
                   │  Priority-weighted zones│
                   └─────────────────────────┘
```

**Baseline arm**: `BaselineExplorer` — same greedy coverage algorithm, but each
UAV acts on its own sensor readings only (no MCP context).

**MCP arm**: `MCPExplorer` — same algorithm, but agents receive shared coverage
maps and sector assignments, eliminating redundant area searches.

---

## 4. Quick Start

```bash
# 1. Create environment and install
make install

# 2. Activate venv (required)
source venv/bin/activate

# 3. Run quick demo (~1-2 min, 5 episodes each)
make final-review-quick

# 4. Run full publication-quality demo
make final-review

# 5. GPS-denied SLAM validation
make gps-denied-test
```

Results are saved to `results/final_review/`.

### Individual experiment commands

```bash
python run_final_review_demo.py --mode quick
python run_final_review_demo.py --mode full
python run_final_review_demo.py --mode full --gps-denied

# Custom configuration
python run_final_review_demo.py \
    --failure-episodes 20 \
    --comm-episodes 15 \
    --scale-episodes 12 \
    --uavs 5
```

---

## 5. Experiments

### Experiment 1 — UAV Failure Resilience

Injects UAV failures at fixed steps (step 2 000 and step 6 000 out of 10 000).

Scenarios: No failure / 1 failure / 2 failures — for both MCP and Baseline arms.

Primary metric: **priority-weighted coverage** at mission end.
Statistical test: paired t-test across episodes for each failure level.

### Experiment 2 — Communication Degradation

Simulates packet loss by setting message delivery probability to:
1.0 → 0.8 → 0.6 → 0.4 → 0.2

Shows how gracefully MCP degrades vs independent baseline.

### Experiment 3 — Scalability Study

Swarm sizes: 3, 5, 7, 10 UAVs.

Proves O(n) message complexity: each UAV sends one broadcast per step, so
total messages = n × steps. Context aggregation time is measured in milliseconds.

---

## 6. Key Results

| Metric | Result |
|---|---|
| Message complexity | **O(n)** — linear, confirmed empirically |
| Failure resilience (net Δ pw-coverage) | MCP retains significantly more coverage under failures |
| Comm degradation P-value | Significant advantage maintained to ~0.6 reliability |
| GPS-denied coverage vs normal | < 3 % degradation (SLAM noise decays from 8 m → 1 m) |

All results include 95 % CI from paired t-tests.  See `results/final_review/`
for the full JSON and combined figure after running `make final-review`.

---

## 7. GPS-Denied SLAM Mode

Run with `--gps-denied` to simulate GPS-unavailable conditions.

**Model:** Gaussian noise added to each UAV's perceived position:

$$\sigma(t) = \sigma_{\text{final}} + (\sigma_{\text{init}} - \sigma_{\text{final}}) \cdot \max\!\left(0,\, 1 - \frac{t}{5000}\right)$$

- $\sigma_{\text{init}} = 8$ m (heavy uncertainty at mission start)
- $\sigma_{\text{final}} = 1$ m (SLAM converges after ~5 000 steps)

**Why MCP helps under GPS-denied conditions:** Sector boundaries are anchored to
episode-start positions (which SLAM knows precisely). The shared coverage map
compensates for individual drift because the aggregate grid is built from actual
sensor footprints (physical coverage), not estimated positions.

---

## 8. Real Dataset Integration

The environment's disaster zones are derived from real-world data:

- **Tidal / flood**: `Visakhapatnam_UTide_full2024_hourly_IST.csv` —
  measured tidal heights drive inundation area simulations.
- **Wildfire geometry**: parameterised after NASA FIRMS detection density grids
  (clustered high-severity zones alongside scattered low-severity spread).

The `simulation/data_loader.py` module loads the CSV at startup; the
`simulation/disaster_scenario.py` module maps measured values to grid severity
and builds the priority-weighted coverage target.

---

## 9. Project Structure

```
.
├── run_final_review_demo.py        # Master experiment runner
├── Makefile                        # All make targets
├── requirements.txt
│
├── experiments/
│   ├── exploration_agents.py       # MCPExplorer + BaselineExplorer (GPS-denied)
│   ├── failure_resilience.py       # Experiment 1
│   ├── communication_degradation.py# Experiment 2
│   └── scalability_study.py        # Experiment 3
│
├── simulation/
│   ├── environment.py              # SwarmEnvironment (Gymnasium)
│   ├── disaster_scenario.py        # Priority zones + dataset integration
│   ├── data_loader.py              # Loads real CSV data
│   ├── uav.py                      # UAV physics
│   └── visualization.py
│
├── mcp_server/
│   ├── server.py                   # Async MCP server
│   ├── context_manager.py          # Shared context aggregation
│   └── message_protocol.py        # Protocol definition
│
├── rl_agents/
│   ├── ppo_agent.py                # PPO implementation
│   ├── advanced_agents.py          # SAC, TD3, A2C, DDPG
│   └── train.py                    # Training entry point
│
├── Docs/                           # Architecture and phase documentation
├── results/final_review/           # Generated experiment results
└── tests/                          # Unit tests
```

---

## 10. Dependencies

```
gymnasium>=0.29
torch>=2.0
numpy
scipy           # paired t-tests and confidence intervals
loguru
matplotlib
seaborn
pandas
```

Install everything: `make install`

---

## Citation

If you use this code, please cite:

```bibtex
@misc{mcp-swarm-2025,
  title  = {MCP-Coordinated Swarm Intelligence: Adaptive UAV Path Planning
             for Dynamic Disaster Response},
  year   = {2025},
  school = {IIIT Kottayam},
  note   = {B.Tech Final Review Project}
}
```
