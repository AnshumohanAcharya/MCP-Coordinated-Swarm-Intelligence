# System Architecture

## MCP-Coordinated Swarm Intelligence

---

## High-Level Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Simulation Layer                              │
│                                                                      │
│   ┌───────────────┐  ┌───────────────┐  ┌───────────────────────┐   │
│   │  UAV Agent 1  │  │  UAV Agent 2  │  │     UAV Agent N       │   │
│   │ MCPExplorer   │  │ MCPExplorer   │  │    MCPExplorer        │   │
│   │ (sector i)    │  │ (sector i+1)  │  │    (sector N)        │   │
│   └──────┬────────┘  └──────┬────────┘  └──────────┬────────────┘   │
│          │                  │                       │                │
│          └──────────────────┼───────────────────────┘                │
│                             │  MCP broadcast (one msg/step/UAV)      │
│                ┌────────────▼────────────┐                           │
│                │      MCP Server         │                           │
│                │  context_manager.py     │      O(n) complexity      │
│                │  message_protocol.py   │                           │
│                └────────────┬────────────┘                           │
│                             │                                        │
│                ┌────────────▼────────────────────────────────────┐   │
│                │          SwarmEnvironment (Gymnasium)           │   │
│                │                                                  │   │
│                │  • 100×100 grid, 10 m/cell, 1 km × 1 km        │   │
│                │  • Priority-weighted disaster zones             │   │
│                │  • UAV physics (speed=15 m/s, sensor=50 m)     │   │
│                │  • Failure injection, comm degradation          │   │
│                └──────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Component Descriptions

### 1. SwarmEnvironment (`simulation/environment.py`)

Gymnasium-compatible environment managing:

- **Grid**: 100×100 cells, each 10 m × 10 m
- **UAV registration**: creates N `UAV` objects from `SimulationConfig`
- **Step loop**: calls each UAV's physics update, then scores the updated coverage grid
- **Coverage scoring**: `coverage_percentage` (raw) + `priority_weighted_coverage` (weighted by zone severity)
- **Failure injection**: `env.inject_failure(uav_idx)` marks a UAV as dead
- **Seed support**: `env.reset(seed=N)` for reproducible paired comparisons

### 2. DisasterScenario (`simulation/disaster_scenario.py`)

Builds the priority map from real and parameterised data:

- **Wildfire zones**: 3–5 clustered high-severity zones + scattered spread cells
- **Flood zones**: derived from `Visakhapatnam_UTide_full2024_hourly_IST.csv` tidal heights
- **Severity grid**: values in [0, 1]; `priority_weighted_coverage = Σ(coverage_ij × severity_ij) / Σ severity_ij`
- **Caching**: `_DATASET_CACHE` singleton prevents regenerating the grid every episode

### 3. MCPExplorer / BaselineExplorer (`experiments/exploration_agents.py`)

Rule-based agents for controlled coordination experiments.

**BaselineExplorer** — greedy independent:
- Finds the nearest uncovered cell in its sensor range
- Acts on local observation only (no shared context)

**MCPExplorer** — sector-partitioned with shared context:
- UAVs are ranked south-to-north; each gets a horizontal sector `[y_lo, y_hi)`
- Within its sector, follows same greedy logic as baseline
- When a sector UAV fails, neighbouring MCPExplorers expand to fill the gap via MCP context
- **GPS-denied mode** (`gps_denied=True`): position used for targeting is perturbed by decaying Gaussian noise (8 m → 1 m over 5 000 steps)

### 4. MCP Server (`mcp_server/`)

Lightweight async WebSocket server:

- `server.py` — accepts connections, routes messages
- `context_manager.py` — aggregates UAV sensor readings into a shared coverage grid;
  broadcasts the merged grid to all connected agents
- `message_protocol.py` — defines `ContextMessage` dataclass (type, sender, data payload)

### 5. RL Agents (`rl_agents/`)

Used in Reviews I–III. Not used in the final rule-based comparison (see reasoning
in `README.md §2`), but available for training experiments.

- `ppo_agent.py` — custom PPO with combined advantage (coverage + coordination)
- `advanced_agents.py` — SAC, TD3, A2C, DDPG via `stable-baselines3`
- `train.py` — entry point; `make train-agents` for a 200-episode warm-up

---

## Data Flow (one simulation step)

```
for each UAV agent:
    1. Agent reads local observation from env
    2. (GPS-denied) Position is perturbed by SLAM noise sigma(t)
    3. Agent computes target cell and action vector
    4. MCPExplorer: broadcasts coverage update to MCP Server
    5. MCP Server: aggregates, rebroadcasts shared grid to all agents

env.step(all_actions):
    6. UAV physics update (velocity, battery drain, sensor sweep)
    7. Coverage grid update: cells within sensor_range marked covered
    8. Compute reward = raw_coverage * severity_weight - overlap_penalty
    9. Return obs, reward, done, info (includes priority_weighted_coverage)
```

---

## Configuration

All physics and experiment parameters live in `config/simulation_config.py`:

| Parameter | Value | Notes |
|---|---|---|
| Grid size | 100×100 | 10 m/cell |
| UAV max speed | 15.0 m/s | |
| Sensor range | 50.0 m | 5-cell radius |
| Battery drain rate | 0.0001/s | ~10% over 10 000-step mission |
| Steps per episode | 10 000 | |
| Coverage threshold | 0.5 | fraction to mark a cell surveyed |
| Coverage amount | 1.0 | per-step coverage increment |

---

## Key Design Decisions

1. **Rule-based comparison agents** — isolate the MCP coordination variable exactly;
   any coverage difference is attributable only to context sharing, not policy quality.

2. **Paired trials with fixed seeds** — each episode uses `seed = BASE_SEED + ep`;
   both MCP and Baseline arms run the same environment realisations, enabling a
   paired t-test with maximum statistical power.

3. **O(n) message complexity** — each UAV emits exactly one broadcast per step;
   the MCP server aggregates in a single pass. Experiment 3 verifies this empirically
   by measuring context aggregation time across swarm sizes 3–10.

4. **Priority-weighted coverage** — raw coverage rewards covering low-value open
   land equally with high-severity fire zones. The weighted metric directly measures
   mission effectiveness, not just area surveyed.
