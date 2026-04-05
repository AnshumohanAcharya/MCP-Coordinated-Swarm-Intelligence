# Benchmark Comparison Report

## Executive Summary

This report compares four RL agents (PPO, MCP-PPO, DQN, MAPPO) across NOAA and AMOVFLY datasets. **MCP-PPO trades maximum area coverage for efficiency and robustness**: it achieves the highest battery efficiency and more stable behavior, while DQN and MAPPO achieve higher raw coverage. This trade-off is valuable for long-duration missions where resource conservation matters.

---

## Best Agent per Metric

| Metric | Best Agent |
|--------|------------|
| Reward | dqn |
| Coverage | mappo |
| Battery Efficiency | **mcp_ppo** |
| Communication Reliability | **mcp_ppo** |
| Collision Count (lower is better) | dqn |

---

## MCP-PPO Highlights

- **Highest battery efficiency**: MCP-PPO conserves energy better than baseline PPO, DQN, and MAPPO.
- **Highest communication reliability**: Shared context enables more consistent coordination.
- **Lower coverage than DQN/MAPPO**: MCP-PPO prioritizes efficient coverage over aggressive area maximization.
- **More stable behavior**: Fewer catastrophic reward crashes; lower variance across episodes.

---

## Key Finding

> **MCP trades maximum area for efficiency and robustness.**

For disaster response missions where battery life and coordination matter, MCP-PPO offers a compelling alternative to coverage-maximizing agents.

---

*Generated from `extended_benchmark.csv`*
