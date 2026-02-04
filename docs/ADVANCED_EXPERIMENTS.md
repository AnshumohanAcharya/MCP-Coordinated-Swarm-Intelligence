# Advanced Experiments Guide

This document describes datasets, agents, experiment design, metrics, and reproduction steps for the MCP-Coordinated Swarm Intelligence project.

## Datasets

### Vizag (Visakhapatnam Tidal)

- **Location**: `Visakhapatnam_UTide_full2024_hourly_IST.csv` (project root)
- **Format**: `Time(IST)`, `prs(m)` (tidal pressure in meters)
- **Use**: Wind modification factor derived from tidal pressure; higher pressure → stronger wind factor (0.5–1.5)
- **Loader**: `simulation.external_data.VizagLoader`

### NOAA (San Diego Water Level)

- **Location**: `data/noaa/sandiego/2024_*.csv` (monthly files)
- **Format**: `Date Time`, ` Water Level`, Sigma, etc.
- **Use**: Water level time series normalized to wind factor (0.5–1.5)
- **Loader**: `simulation.external_data.NOAALoader`

### AMOVFLY (Wind)

- **Location**: `data/amovfly/wind/*.csv`
- **Format**: `time`, `num`, `w_s` (wind speed), `w_a` (wind angle)
- **Use**: Wind speed/direction normalized to wind factor
- **Loader**: `simulation.external_data.AMOVFLYLoader`

### PSMSL (Optional)

- **Location**: `data/psmsl/rlr_monthly.zip`
- **Use**: Monthly sea level data for comparison in notebooks

---

## Agents

| Agent | Description | Action Space |
|-------|-------------|--------------|
| **PPO** | Proximal Policy Optimization, baseline | Continuous (ax, ay, az) |
| **MCP-PPO** | Context-aware PPO with MCP shared context | Continuous |
| **DQN** | Deep Q-Network with discretized acceleration (3³=27 actions) | Discrete → continuous mapping |
| **MAPPO** | Multi-Agent PPO with centralized critic, decentralized actors | Continuous |

### DQN

- **File**: `rl_agents/experimental/dqn_agent.py`
- **Discretization**: 3 levels per axis (-1, 0, 1) → 27 discrete actions
- **Replay buffer**: Experience replay for off-policy learning
- **Target network**: Updated every `target_update_frequency` steps

### MAPPO

- **File**: `rl_agents/experimental/mappo_agent.py`
- **Actor**: Local observation → action (decentralized)
- **Critic**: Full swarm observation → value (centralized)
- **Training**: PPO-style updates with GAE

---

## Experiment Design

### Extended Benchmark

Runs PPO, MCP-PPO, DQN, and MAPPO on NOAA and AMOVFLY datasets.

- **Episodes per run**: 20 (configurable)
- **Datasets**: `noaa`, `amovfly`
- **Metrics**: reward, coverage (%), battery efficiency, communication reliability, collisions, mission success

### Data Source Configuration

Set `data_source` in environment config to use a specific loader:

```yaml
environment:
  data_source: "noaa"   # or "vizag", "amovfly"
```

Or in code:

```python
from config.simulation_config import SimulationConfig, EnvironmentConfig

env_config = EnvironmentConfig(data_source="noaa")
config = SimulationConfig(environment_config=env_config)
```

---

## Metrics

| Metric | Description |
|--------|-------------|
| **reward** | Per-step reward sum over episode |
| **coverage** | Percentage of area covered by UAV sensors |
| **battery_efficiency** | Mean battery level during episode |
| **communication_reliability** | Mean connectivity between UAVs |
| **collision_count** | Number of obstacle collisions |
| **mission_success** | Whether all targets reached required coverage |

---

## Reproduction Steps

### 1. Environment Setup

```bash
source ~/Desktop/btp/bin/activate  # or your venv
cd MCP-Coordinated-Swarm-Intelligence
pip install -r requirements.txt
```

### 2. Run Extended Benchmark

```bash
# Optional: start MCP server for MCP-PPO (in another terminal)
python -m mcp_server.server

# Run benchmark (PPO, DQN, MAPPO work without MCP; MCP-PPO needs server)
python experiments/extended_benchmark.py
```

Outputs:

- `results/extended_benchmark.csv` — per-episode metrics
- `results/extended_benchmark_plots.png` — matplotlib plots

### 3. Run Notebooks

```bash
cd notebooks
jupyter notebook
```

- **01_compare_datasets.ipynb**: Load Vizag, NOAA, PSMSL and plot normalized time series
- **02_environment_effect.ipynb**: Wind factor over time for each loader
- **03_agent_benchmark.ipynb**: Quick PPO benchmark

### 4. Use External Data in Simulation

```python
from config.simulation_config import SimulationConfig, EnvironmentConfig

env_config = EnvironmentConfig(data_source="noaa")
config = SimulationConfig(environment_config=env_config)
config.render = False

from simulation.environment import SwarmEnvironment
env = SwarmEnvironment(config)
# Wind in DisasterScenario will use NOAALoader.get_wind_factor(t)
```

---

## File Reference

| File | Purpose |
|------|---------|
| `simulation/external_data.py` | VizagLoader, NOAALoader, AMOVFLYLoader, `get_data_loader()` |
| `simulation/disaster_scenario.py` | Uses `data_loader.get_wind_factor()` in `_update_environmental_conditions` |
| `rl_agents/experimental/dqn_agent.py` | DQN agent |
| `rl_agents/experimental/mappo_agent.py` | MAPPO agent |
| `experiments/extended_benchmark.py` | Full benchmark script |
| `notebooks/01_compare_datasets.ipynb` | Dataset comparison |
| `notebooks/02_environment_effect.ipynb` | Wind factor visualization |
| `notebooks/03_agent_benchmark.ipynb` | Agent benchmark |
