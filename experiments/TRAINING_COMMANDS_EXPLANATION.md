# Training Commands Explanation

## Overview

These commands train two different types of agents and then compare their performance:

1. **Baseline agents** (without MCP context)
2. **MCP-coordinated agents** (with MCP context)
3. **Comparison** (evaluates both)

## Command Breakdown

### 1. Train Baseline Agents

```bash
python3 -m rl_agents.train --episodes 200 --no-context
```

**What it does:**
- Trains **PPO agents** (standard reinforcement learning)
- **No MCP connection** - agents work independently
- Each agent only sees its own local observations
- Saves models to `saved_models/agent_*_episode_*.pth`

**What happens:**
- Creates `SwarmEnvironment` without MCP
- Creates `PPOAgent` instances (not `ContextAwareAgent`)
- Trains for 200 episodes
- Agents learn to coordinate without shared context

**Output:**
- Model files: `saved_models/agent_0_episode_100.pth`, etc.
- Training logs: `logs/training.log`
- Metrics: reward, coverage, battery efficiency

### 2. Train MCP-Coordinated Agents

```bash
python3 -m rl_agents.train --episodes 200
```

**What it does:**
- Trains **ContextAwareAgent** (PPO + MCP context)
- **Connects to MCP server** - agents share context
- Each agent sees local observations + aggregated MCP context
- Saves models to `saved_models/agent_*_episode_*.pth`

**What happens:**
- Creates `SwarmEnvironment` with MCP connection
- Creates `ContextAwareAgent` instances
- Connects to MCP server (must be running!)
- Trains for 200 episodes
- Agents learn to use shared context for coordination

**Output:**
- Model files: `saved_models/agent_0_episode_100.pth`, etc.
- Training logs: `logs/training.log`
- Metrics: reward, coverage, battery efficiency
- **Note:** Models will overwrite baseline models (same filenames)

### 3. Compare Both Approaches

```bash
python3 experiments/baseline_comparison.py --episodes 50
```

**What it does:**
- Runs **both** baseline and MCP experiments
- Uses **untrained agents** (random weights) by default
- Compares performance metrics
- Generates plots and reports

**What happens:**
1. Runs baseline experiment (no MCP, PPO agents)
2. Runs MCP experiment (with MCP, ContextAware agents)
3. Calculates improvements/differences
4. Generates:
   - `results/baseline_comparison.png` (plots)
   - `results/comparison_report.txt` (summary)

**Output:**
- Comparison metrics (reward, coverage, battery, etc.)
- Statistical significance tests
- Visual plots showing differences
- Text report with analysis

## Important Notes

### Model Saving

⚠️ **Both training commands save to the same directory!**

- Baseline: `saved_models/agent_0_episode_100.pth`
- MCP: `saved_models/agent_0_episode_100.pth` (overwrites!)

**Solution:** Save to different directories or use different episode numbers.

### MCP Server Requirement

✅ **MCP training requires MCP server running:**

```bash
# Terminal 1: Start MCP server
python3 -m mcp_server.server

# Terminal 2: Train MCP agents
python3 -m rl_agents.train --episodes 200
```

❌ **Baseline training does NOT need MCP server** (uses `--no-context`)

### Comparison Uses Untrained Agents

⚠️ **The comparison script creates fresh agents** - it doesn't load your trained models!

To use trained models in comparison, you'd need to modify `baseline_comparison.py` to:
1. Load baseline models for baseline experiment
2. Load MCP models for MCP experiment

## Typical Workflow

### For Review 2 (Quick Demo)

```bash
# 1. Start MCP server
python3 -m mcp_server.server

# 2. Run comparison (untrained agents - fast)
python3 experiments/baseline_comparison.py --episodes 50

# 3. Show results
# - results/baseline_comparison.png
# - results/comparison_report.txt
```

### For Full Training + Comparison

```bash
# 1. Train baseline (no MCP needed)
python3 -m rl_agents.train --episodes 200 --no-context

# 2. Start MCP server
python3 -m mcp_server.server

# 3. Train MCP agents
python3 -m rl_agents.train --episodes 200

# 4. Compare (modify comparison.py to load trained models)
python3 experiments/baseline_comparison.py --episodes 50
```

## Command Options

### Training Options

```bash
python3 -m rl_agents.train [OPTIONS]

Options:
  --episodes N          Number of training episodes (default: 1000)
  --no-context          Train without MCP context (baseline)
  --config PATH         Path to config file
  --load-episode N      Load models from episode N
  --save-freq N         Save models every N episodes (default: 100)
```

### Comparison Options

```bash
python3 experiments/baseline_comparison.py [OPTIONS]

Options:
  --episodes N          Episodes per experiment (default: 50)
  --runs N              Number of runs for statistics (default: 1)
  --num-uavs N          Number of UAVs (default: 3)
  --sim-time N          Simulation time per episode (default: 60)
```

## Expected Results

### After Training Baseline

- Models saved: `saved_models/agent_*_episode_*.pth`
- Training metrics improve over episodes
- Agents learn basic coordination

### After Training MCP

- Models saved: `saved_models/agent_*_episode_*.pth` (overwrites baseline!)
- Training metrics improve over episodes
- Agents learn to use MCP context for better coordination

### After Comparison

- **Plots:** Episode rewards, coverage, battery efficiency
- **Report:** Statistical comparison, improvements
- **Note:** Results show untrained agents unless you modify comparison.py

## Troubleshooting

### "MCP connection failed" during MCP training

**Solution:** Start MCP server first:
```bash
python3 -m mcp_server.server
```

### Models overwritten

**Solution:** Use different save directories or episode numbers:
- Baseline: Save at episodes 100, 200
- MCP: Save at episodes 300, 400

### Comparison shows poor performance

**Expected!** Comparison uses untrained agents. For better results:
1. Train both agent types first
2. Modify `baseline_comparison.py` to load trained models
3. Then run comparison

