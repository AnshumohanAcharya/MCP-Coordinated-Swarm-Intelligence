"""DQN multi-UAV agent with discretized acceleration actions."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Dict, Any, List

from ..base_agent import BaseAgent


# Discretization: 3 levels per axis (-1, 0, 1) -> 27 actions
N_DISCRETE = 3
ACTION_LEVELS = np.linspace(-1, 1, N_DISCRETE)
N_ACTIONS = N_DISCRETE ** 3  # 27


def discrete_to_continuous(action_idx: int, action_scale: float = 1.0) -> np.ndarray:
    """Map discrete action index (0..26) to continuous (ax, ay, az)."""
    ax_idx = action_idx % N_DISCRETE
    ay_idx = (action_idx // N_DISCRETE) % N_DISCRETE
    az_idx = action_idx // (N_DISCRETE * N_DISCRETE)
    return np.array([
        ACTION_LEVELS[ax_idx],
        ACTION_LEVELS[ay_idx],
        ACTION_LEVELS[az_idx]
    ], dtype=np.float32) * action_scale


def continuous_to_discrete(acc: np.ndarray) -> int:
    """Map continuous (ax, ay, az) to nearest discrete index."""
    a = np.clip(acc, -1, 1)
    ax_idx = int(np.round((a[0] + 1) / 2 * (N_DISCRETE - 1)))
    ay_idx = int(np.round((a[1] + 1) / 2 * (N_DISCRETE - 1)))
    az_idx = int(np.round((a[2] + 1) / 2 * (N_DISCRETE - 1)))
    ax_idx = max(0, min(N_DISCRETE - 1, ax_idx))
    ay_idx = max(0, min(N_DISCRETE - 1, ay_idx))
    az_idx = max(0, min(N_DISCRETE - 1, az_idx))
    return ax_idx + ay_idx * N_DISCRETE + az_idx * N_DISCRETE * N_DISCRETE


class DQNNetwork(nn.Module):
    """Q-network for DQN."""

    def __init__(self, state_dim: int, n_actions: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        layers = []
        prev = state_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class ReplayBuffer:
    """Experience replay buffer."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states = torch.FloatTensor(np.array([b[0] for b in batch]))
        actions = torch.LongTensor([b[1] for b in batch])
        rewards = torch.FloatTensor([b[2] for b in batch])
        next_states = torch.FloatTensor(np.array([b[3] for b in batch]))
        dones = torch.FloatTensor([b[4] for b in batch])
        return {"states": states, "actions": actions, "rewards": rewards,
                "next_states": next_states, "dones": dones}

    def __len__(self):
        return len(self.buffer)


class DQNAgent(BaseAgent):
    """DQN agent with discretized acceleration. Reuses SwarmEnvironment."""

    def __init__(self, agent_id: str, state_dim: int, action_dim: int, config: Dict[str, Any]):
        super().__init__(agent_id, state_dim, action_dim, config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_scale = config.get("action_scale", 2.0)
        self.gamma = config.get("gamma", 0.99)
        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_end = config.get("epsilon_end", 0.01)
        self.epsilon_decay = config.get("epsilon_decay", 5000)
        self.batch_size = config.get("batch_size", 64)
        self.buffer_size = config.get("buffer_size", 10000)
        self.lr = config.get("learning_rate", 1e-3)
        self.target_update_freq = config.get("target_update_frequency", 100)

        self.q_net = DQNNetwork(state_dim, N_ACTIONS).to(self.device)
        self.target_net = DQNNetwork(state_dim, N_ACTIONS).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.replay = ReplayBuffer(self.buffer_size)
        self.update_count = 0

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action; returns continuous (ax, ay, az) for env compatibility."""
        if not deterministic and random.random() < self.epsilon:
            action_idx = random.randint(0, N_ACTIONS - 1)
        else:
            with torch.no_grad():
                x = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q = self.q_net(x)
                action_idx = int(q.argmax(dim=1).item())
        return discrete_to_continuous(action_idx, self.action_scale)

    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float,
                         next_state: np.ndarray, done: bool):
        """Store (state, discrete_action, reward, next_state, done)."""
        action_idx = continuous_to_discrete(action / (self.action_scale + 1e-9))
        self.replay.push(state, action_idx, reward, next_state, done)

    def update(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
               next_states: np.ndarray, dones: np.ndarray) -> Dict[str, float]:
        """Update Q-network from replay buffer."""
        if len(self.replay) < self.batch_size:
            return {}
        batch = self.replay.sample(self.batch_size)
        s = batch["states"].to(self.device)
        a = batch["actions"].to(self.device)
        r = batch["rewards"].to(self.device)
        ns = batch["next_states"].to(self.device)
        d = batch["dones"].to(self.device)

        q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(ns).max(1)[0]
            target = r + self.gamma * next_q * (1 - d)
        loss = nn.MSELoss()(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.epsilon = max(self.epsilon_end,
                          self.epsilon - (1.0 - self.epsilon_end) / self.epsilon_decay)
        self.performance_history["exploration_rate"].append(self.epsilon)
        return {"loss": loss.item()}

    def save(self, filepath: str) -> None:
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "config": self.config,
        }, filepath)

    def load(self, filepath: str) -> None:
        ckpt = torch.load(filepath, map_location=self.device, weights_only=False)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt.get("optimizer", self.optimizer.state_dict()))
        self.epsilon = ckpt.get("epsilon", self.epsilon)
