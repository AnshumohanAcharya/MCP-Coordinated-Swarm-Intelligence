# Project Phases — MCP-Coordinated Swarm Intelligence

## IIIT Kottayam B.Tech Project · 2024–2025

---

## Phase I — Review I: MCP + PPO Baseline

**Goal**: Prove that sharing context via MCP improves a single RL algorithm (PPO)
compared to PPO agents operating independently.

**What was built**:
- `SwarmEnvironment` (Gymnasium-compatible), 100×100 grid, N UAVs
- Async MCP Server with `ContextManager` and `MessageProtocol`
- Custom PPO agent in `rl_agents/ppo_agent.py` with:
  - Observation: local sensor grid + MCP shared context vector
  - Reward: incremental coverage + coordination bonus
- Web dashboard (`web_dashboard/`) for real-time visualisation

**Outcome**: Early training showed promising coverage trends; however, training
required a very large number of episodes to converge. Coverage improvement over
the fully-independent baseline was directionally positive but noisy due to the
high variance of on-policy PPO with a sparse reward.

**Key files**: `rl_agents/ppo_agent.py`, `mcp_server/`, `simulation/`, `web_dashboard/`

---

## Phase II — Review II: Multi-Algorithm RL Comparison

**Goal**: Identify the best RL algorithm for the coverage task from five candidates.

**Algorithms compared**:
| Algorithm | Type | Library |
|---|---|---|
| PPO | On-policy AC | custom + SB3 |
| SAC | Off-policy AC (entropy-reg) | stable-baselines3 |
| TD3 | Off-policy deterministic | stable-baselines3 |
| A2C | On-policy AC | stable-baselines3 |
| DDPG | Off-policy deterministic | stable-baselines3 |

**Metrics**: Final coverage %, convergence speed (episodes to 60 % coverage),
sample efficiency (coverage per 1 000 training steps).

**Outcome**: SAC achieved the highest asymptotic coverage and best sample
efficiency due to its entropy-regularisation encouraging exploration. TD3 was
runner-up. PPO was slowest to converge.

**Key files**: `experiments/rl_comparison.py`, `rl_agents/advanced_agents.py`,
`results/review_iii/rl_comparison/`

---

## Phase III — Review III: SLAM Integration + Attention

**Goal**: Add SLAM-based loop closure to correct UAV position drift, and
introduce an attention mechanism in the policy network to weight shared
context selectively.

**What was added**:
- `slam/slam_module.py` — occupancy grid SLAM with loop-closure correction
- Attention layer in the PPO/SAC policy head weighting MCP context tokens
- `experiments/slam_comparison.py` — head-to-head with SLAM vs without

**Outcome**: SLAM loop closure reduced positional drift by ~40 % in long
missions. The attention mechanism improved selective use of MCP context tokens
when only some neighbours were relevant.

**Key files**: `slam/slam_module.py`, `experiments/slam_comparison.py`,
`run_review_iii_demo.py`, `results/review_iii/slam_demo/`

---

## Final Review — Rigorous Experimental Validation

**Goal**: Prove the MCP coordination benefit cleanly, with statistical significance,
under realistic adverse conditions. Use rule-based agents to isolate the
coordination variable.

### Why Rule-Based (not trained RL)?

Trained RL agents confound two variables: (1) policy quality and (2) coordination.
Rule-based agents completely remove variable 1: both arms use the same greedy
coverage algorithm — the only difference is whether MCP context is shared.

### Experiment 1 — UAV Failure Resilience

- 6 scenarios: {MCP, Baseline} × {0, 1, 2 failures}
- Failures injected at steps 2 000 and 6 000 of a 10 000-step mission
- **Metric**: priority-weighted coverage at mission end
- **Stats**: paired t-test + 95 % CI across episodes

**Claim**: MCP context allows surviving UAVs to redistribute across failed
sectors, preserving coverage. Baseline UAVs have no knowledge of which zones
are unattended.

### Experiment 2 — Communication Degradation

- 5 reliability levels: 1.0, 0.8, 0.6, 0.4, 0.2
- At each level, MCP messages are randomly dropped with probability `1 - reliability`
- **Metric**: priority-weighted coverage vs reliability curve
- **Stats**: paired t-test per level

**Claim**: MCP maintains meaningful coordination advantage down to ~0.6
reliability. Below 0.4, the advantage narrows as context becomes too stale.

### Experiment 3 — Scalability Study

- Swarm sizes: 3, 5, 7, 10 UAVs
- Long 10 000-step missions; context aggregation time measured per step
- **Metric**: context_agg_time_ms (should scale linearly with n)
- **Coverage metric**: priority-weighted coverage per UAV vs swarm size
- **Stats**: paired t-test per swarm size

**Claim**: O(n) message complexity confirmed. Coverage per UAV increases
with swarm size up to the environment's diminishing-returns point.

### GPS-Denied SLAM Validation

- `--gps-denied` flag adds decaying Gaussian noise to perceived UAV positions
- Noise model: σ(t) = 1.0 + 7.0 · max(0, 1 − t/5000) metres
- Tests MCP robustness when individual positioning is unreliable
- **Key insight**: MCP sector boundaries are anchored to episode-start positions
  (which SLAM knows precisely); shared coverage maps compensate for local drift

### Key Results

All numerical results in `results/final_review/` after running `make final-review`.

Summary of expected findings:
1. **Scalability**: O(n) message complexity — agg time grows linearly, verified
2. **Failure resilience**: MCP shows statistically significant advantage (p < 0.05)
   over baseline under ≥1 UAV failure
3. **Comm degradation**: MCP advantage maintained (p < 0.05) at reliability ≥ 0.6
4. **GPS-denied**: < 3 % coverage drop vs GPS mode (SLAM noise decays rapidly)

---

## Summary Timeline

| Phase | Period | Main Output |
|---|---|---|
| Review I | Sem 7, Q3 | MCP + PPO system, web dashboard |
| Review II | Sem 7, Q4 | 5-algorithm comparison report |
| Review III | Sem 8, Q1 | SLAM integration, attention mechanism |
| Final Review | Sem 8, Q2 | Statistical validation, GPS-denied mode |
