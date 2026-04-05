"""MAPPO: Multi-Agent PPO with centralized critic and decentralized actors."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Dict, Any, List, Tuple
from collections import deque

from ..base_agent import BaseAgent


class ActorNetwork(nn.Module):
    """Decentralized actor: local observation -> action."""

    def __init__(self, local_obs_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(local_obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor_mean = nn.Linear(hidden, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        mean = torch.tanh(self.actor_mean(h))
        std = torch.exp(self.actor_logstd).expand_as(mean)
        return mean, std


class CentralizedCritic(nn.Module):
    """Centralized critic: full swarm observation -> value."""

    def __init__(self, full_obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(full_obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MAPPOAgent(BaseAgent):
    """MAPPO agent: decentralized actor, uses centralized critic during training."""

    def __init__(self, agent_id: str, state_dim: int, action_dim: int, config: Dict[str, Any],
                 full_obs_dim: int = None):
        super().__init__(agent_id, state_dim, action_dim, config)
        self.full_obs_dim = full_obs_dim or state_dim  # For single-agent, same as local
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_scale = config.get("action_scale", 2.0)
        self.gamma = config.get("gamma", 0.99)
        self.lambda_gae = config.get("lambda_gae", 0.95)
        self.epsilon_clip = config.get("epsilon_clip", 0.2)
        self.lr = config.get("learning_rate", 3e-4)
        self.batch_size = config.get("batch_size", 64)
        self.buffer_size = config.get("buffer_size", 2048)
        self.ppo_epochs = config.get("ppo_epochs", 4)

        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.critic = CentralizedCritic(self.full_obs_dim).to(self.device)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=self.lr
        )

        self._buffer = {
            "states": [], "actions": [], "rewards": [], "dones": [],
            "log_probs": [], "values": [], "full_states": [],
        }
        self.training = True

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action from local observation."""
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            mean, std = self.actor(s)
            if deterministic:
                action = mean
            else:
                dist = Normal(mean, std)
                action = dist.sample()
            action = action.squeeze(0).cpu().numpy()
        return np.clip(action, -1, 1) * self.action_scale

    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float,
                         next_state: np.ndarray, done: bool, full_state: np.ndarray,
                         log_prob: float, value: float):
        """Store transition for PPO update."""
        if len(self._buffer["states"]) >= self.buffer_size:
            return
        self._buffer["states"].append(state)
        self._buffer["actions"].append(action)
        self._buffer["rewards"].append(reward)
        self._buffer["dones"].append(done)
        self._buffer["log_probs"].append(log_prob)
        self._buffer["values"].append(value)
        self._buffer["full_states"].append(full_state)

    def update(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
               next_states: np.ndarray, dones: np.ndarray) -> Dict[str, float]:
        """Update actor and critic. Expects batch from buffer."""
        if len(self._buffer["states"]) < self.batch_size:
            return {}
        n = len(self._buffer["states"])
        idx = np.random.permutation(n)[:self.batch_size]
        s = torch.FloatTensor(np.array(self._buffer["states"])[idx]).to(self.device)
        a = torch.FloatTensor(np.array(self._buffer["actions"])[idx]).to(self.device)
        r = np.array(self._buffer["rewards"])[idx]
        d = np.array(self._buffer["dones"])[idx]
        old_lp = torch.FloatTensor(np.array(self._buffer["log_probs"])[idx]).to(self.device)
        old_v = torch.FloatTensor(np.array(self._buffer["values"])[idx]).to(self.device)
        full_s = torch.FloatTensor(np.array(self._buffer["full_states"])[idx]).to(self.device)

        # Compute returns and advantages (simplified)
        returns = np.zeros_like(r)
        last_r = 0
        for t in reversed(range(len(r))):
            returns[t] = r[t] + self.gamma * last_r * (1 - d[t])
            last_r = returns[t]
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = returns - old_v

        for _ in range(self.ppo_epochs):
            mean, std = self.actor(s)
            dist = Normal(mean, std.expand_as(mean))
            new_lp = dist.log_prob(a).sum(-1)
            new_v = self.critic(full_s).squeeze(-1)
            ratio = torch.exp(new_lp - old_lp)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(new_v, returns)
            loss = policy_loss + 0.5 * value_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        for k in self._buffer:
            self._buffer[k] = []
        return {"policy_loss": policy_loss.item(), "value_loss": value_loss.item()}

    def save(self, filepath: str) -> None:
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
        }, filepath)

    def load(self, filepath: str) -> None:
        ckpt = torch.load(filepath, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
