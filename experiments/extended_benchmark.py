"""Extended benchmark: PPO, MCP-PPO, DQN, MAPPO on NOAA and AMOVFLY datasets."""

import asyncio
import sys
import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from config.simulation_config import SimulationConfig, EnvironmentConfig
from simulation.environment import SwarmEnvironment
from rl_agents.ppo_agent import PPOAgent
from rl_agents.context_aware_agent import ContextAwareAgent

try:
    from rl_agents.experimental.dqn_agent import DQNAgent
except ImportError:
    DQNAgent = None
try:
    from rl_agents.experimental.mappo_agent import MAPPOAgent
except ImportError:
    MAPPOAgent = None


@dataclass
class BenchmarkConfig:
    episodes: int = 20
    datasets: List[str] = None  # ["noaa", "amovfly"]
    agents: List[str] = None   # ["ppo", "mcp_ppo", "dqn", "mappo"]

    def __post_init__(self):
        if self.datasets is None:
            self.datasets = ["noaa", "amovfly"]
        if self.agents is None:
            self.agents = ["ppo", "mcp_ppo", "dqn", "mappo"]


def make_config(data_source: Optional[str] = None, reward_mode: str = "default") -> SimulationConfig:
    """Create SimulationConfig with optional data_source and reward_mode."""
    config = SimulationConfig()
    config.render = False
    config.reward_mode = reward_mode
    if data_source:
        config.environment_config = EnvironmentConfig(
            width=config.environment_config.width,
            height=config.environment_config.height,
            disaster_zones=config.environment_config.disaster_zones,
            obstacles=config.environment_config.obstacles,
            target_areas=config.environment_config.target_areas,
            wind_conditions=config.environment_config.wind_conditions,
            data_source=data_source,
        )
    return config


def create_agents(agent_type: str, env: SwarmEnvironment, config: SimulationConfig) -> List:
    """Create agent list for given type."""
    state_dim = env.observation_space.shape[0] // config.num_uavs
    action_dim = env.action_space.shape[0] // config.num_uavs
    agent_config = {
        "learning_rate": config.rl_config.learning_rate,
        "gamma": config.rl_config.gamma,
        "batch_size": config.rl_config.batch_size,
        "buffer_size": min(10000, config.rl_config.buffer_size),
        "action_scale": config.uav_config.max_acceleration,
    }

    if agent_type == "ppo":
        return [PPOAgent(f"ppo_{i}", state_dim, action_dim, agent_config)
                for i in range(config.num_uavs)]
    elif agent_type == "mcp_ppo":
        return [ContextAwareAgent(f"mcp_{i}", state_dim, action_dim, 20, agent_config)
                for i in range(config.num_uavs)]
    elif agent_type == "dqn" and DQNAgent:
        return [DQNAgent(f"dqn_{i}", state_dim, action_dim, agent_config)
                for i in range(config.num_uavs)]
    elif agent_type == "mappo" and MAPPOAgent:
        full_dim = env.observation_space.shape[0]
        return [MAPPOAgent(f"mappo_{i}", state_dim, action_dim, agent_config, full_obs_dim=full_dim)
                for i in range(config.num_uavs)]
    return []


def get_actions(agents: List, obs: np.ndarray, config: SimulationConfig) -> np.ndarray:
    """Get actions from all agents."""
    actions = []
    state_dim = len(obs) // len(agents)
    for i, agent in enumerate(agents):
        agent_obs = obs[i * state_dim:(i + 1) * state_dim]
        action = agent.select_action(agent_obs)
        actions.extend(action)
    return np.array(actions)


def _is_valid_result(row: Dict[str, Any]) -> bool:
    """Check if a result row is valid (not weird/failed from MCP issues)."""
    r = row.get("reward")
    l = row.get("length", 0)
    if r is None or (isinstance(r, float) and (np.isnan(r) or np.isinf(r))):
        return False
    if l is None or (isinstance(l, (int, float)) and l <= 0):
        return False
    return True


async def run_experiment(
    agent_type: str,
    data_source: str,
    num_episodes: int,
    config: SimulationConfig,
) -> List[Dict[str, Any]]:
    """Run experiment for one agent type on one dataset."""
    use_mcp = agent_type == "mcp_ppo"
    env = SwarmEnvironment(config, mcp_server_url="ws://localhost:8765" if use_mcp else None)
    if not use_mcp:
        env.mcp_connected = False
        env.mcp_websocket = None
    elif use_mcp:
        try:
            await env.start_mcp_connection()
        except Exception as e:
            logger.warning(f"MCP connection failed, skipping mcp_ppo: {e}")
            env.close()
            return []

    agents = create_agents(agent_type, env, config)
    if not agents:
        env.close()
        return []

    if use_mcp:
        for agent in agents:
            if hasattr(agent, "start_mcp_connection"):
                await agent.start_mcp_connection()

    results = []
    for ep in range(num_episodes):
        obs, info = env.reset()
        ep_reward = 0
        ep_len = 0
        prev_obs = obs
        while True:
            actions = get_actions(agents, obs, config)
            obs, reward, terminated, truncated, info = env.step(actions)
            ep_reward += reward
            ep_len += 1

            # Store transitions for DQN
            if agent_type == "dqn" and DQNAgent:
                state_dim = len(prev_obs) // len(agents)
                for i, agent in enumerate(agents):
                    if hasattr(agent, "store_transition"):
                        s = prev_obs[i * state_dim:(i + 1) * state_dim]
                        ns = obs[i * state_dim:(i + 1) * state_dim]
                        a = actions[i * 3:(i + 1) * 3]
                        agent.store_transition(s, a, reward, ns, terminated or truncated)
            prev_obs = obs

            if use_mcp:
                for agent in agents:
                    if hasattr(agent, "update_context_async"):
                        await agent.update_context_async()

            if terminated or truncated:
                break

        cov = info["performance_metrics"]["coverage_percentage"][-1] if info["performance_metrics"]["coverage_percentage"] else 0
        bat = np.mean(info["performance_metrics"]["battery_efficiency"]) if info["performance_metrics"]["battery_efficiency"] else 0
        com = np.mean(info["performance_metrics"]["communication_reliability"]) if info["performance_metrics"]["communication_reliability"] else 0
        results.append({
            "episode": ep,
            "reward": ep_reward,
            "length": ep_len,
            "coverage": cov,
            "battery_efficiency": bat,
            "communication_reliability": com,
            "collision_count": info["performance_metrics"]["collision_count"],
            "mission_success": info["performance_metrics"]["mission_success"],
        })
        if ep % 5 == 0:
            logger.info(f"{agent_type}/{data_source} ep {ep}: reward={ep_reward:.1f} coverage={cov:.1f}%")

    env.close()
    if use_mcp:
        for agent in agents:
            if hasattr(agent, "close"):
                agent.close()
    return results


def save_csv(rows: List[Dict], path: str):
    """Save results to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def plot_benchmark(all_results: Dict[str, Dict[str, List[Dict]]], save_path: str):
    """Plot benchmark results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ["reward", "coverage", "battery_efficiency", "communication_reliability"]
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        for (agent, dataset), results in all_results.items():
            if results:
                vals = [r[metric] for r in results]
                ax.plot(vals, label=f"{agent} ({dataset})", alpha=0.8)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Episode")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved to {save_path}")


async def main(
    reward_mode: str = "default",
    retries: int = 0,
    drop_failed: bool = False,
):
    bench = BenchmarkConfig(episodes=20)
    all_results = {}
    csv_rows = []

    logger.info(f"Reward mode: {reward_mode} (collision_penalty={'0.1' if reward_mode == 'balanced' else '1.0'})")
    if retries:
        logger.info(f"Retries: {retries}, drop_failed: {drop_failed}")

    for data_source in bench.datasets:
        config = make_config(data_source, reward_mode=reward_mode)
        for agent_type in bench.agents:
            if agent_type == "dqn" and not DQNAgent:
                continue
            if agent_type == "mappo" and not MAPPOAgent:
                continue
            key = (agent_type, data_source)
            results = []
            for attempt in range(max(1, retries + 1)):
                logger.info(f"Running {agent_type} on {data_source}" + (f" (attempt {attempt + 1})" if retries else ""))
                results = await run_experiment(agent_type, data_source, bench.episodes, config)
                # Check if results are valid (not empty, no weird rows)
                valid = results and all(_is_valid_result(r) for r in results)
                if valid:
                    break
                if attempt < retries:
                    logger.warning(f"{agent_type}/{data_source} produced invalid/empty results, retrying...")
            if not results and drop_failed:
                logger.warning(f"Dropping failed run: {agent_type}/{data_source}")
                continue
            if drop_failed and results:
                results = [r for r in results if _is_valid_result(r)]
                if not results:
                    logger.warning(f"Dropping {agent_type}/{data_source}: all rows invalid")
                    continue
            all_results[key] = results
            for r in results:
                r["agent"] = agent_type
                r["dataset"] = data_source
                csv_rows.append(r)

    # Output filenames include reward_mode for comparison
    suffix = f"_{reward_mode}" if reward_mode != "default" else ""

    # Save CSV
    csv_path = PROJECT_ROOT / "results" / f"extended_benchmark{suffix}.csv"
    save_csv(csv_rows, str(csv_path))

    # Save JSON (for comparison with other reward modes)
    json_path = PROJECT_ROOT / "results" / f"extended_benchmark{suffix}.json"
    os.makedirs(PROJECT_ROOT / "results", exist_ok=True)
    json_data = {
        "reward_mode": reward_mode,
        "episodes_per_run": bench.episodes,
        "datasets": bench.datasets,
        "agents": bench.agents,
        "results_by_agent_dataset": {
            f"{agent}_{ds}": rows for (agent, ds), rows in all_results.items()
        },
        "csv_rows": csv_rows,
        "summary": {
            f"{agent}_{ds}": {
                "mean_reward": float(np.mean([r["reward"] for r in rows])),
                "mean_coverage": float(np.mean([r["coverage"] for r in rows])),
                "mean_battery": float(np.mean([r["battery_efficiency"] for r in rows])),
                "mean_collisions": float(np.mean([r["collision_count"] for r in rows])),
            }
            for (agent, ds), rows in all_results.items()
        },
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    logger.info(f"JSON saved to {json_path}")

    # Plot
    plot_path = PROJECT_ROOT / "results" / f"extended_benchmark{suffix}_plots.png"
    plot_benchmark(all_results, str(plot_path))

    logger.info(f"Extended benchmark complete. Results in results/ (suffix={suffix or 'default'})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extended benchmark for MCP-Coordinated Swarm")
    parser.add_argument("--reward", choices=["default", "balanced"], default="default",
                        help="Reward mode: default (collision=-1.0) or balanced (collision=-0.1)")
    parser.add_argument("--retries", type=int, default=0,
                        help="Number of retries for failed runs (e.g. MCP connection issues)")
    parser.add_argument("--drop_failed", action="store_true",
                        help="Drop failed runs from results instead of including invalid rows")
    args = parser.parse_args()
    asyncio.run(main(
        reward_mode=args.reward,
        retries=args.retries,
        drop_failed=args.drop_failed,
    ))
