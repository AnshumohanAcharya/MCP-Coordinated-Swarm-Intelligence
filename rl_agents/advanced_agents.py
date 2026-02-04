"""Advanced RL agents (SAC, TD3, A2C, DQN) for UAV swarm coordination comparison."""

import random
from collections import deque
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from .base_agent import BaseAgent


# ============================================
# SAC (Soft Actor-Critic) Agent
# ============================================

class SoftActorNetwork(nn.Module):
    """Soft Actor Network for SAC with stochastic policy."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()

        # Build network
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim

        self.feature_net = nn.Sequential(*layers)

        # Mean and log_std heads
        self.mean_head = nn.Linear(input_dim, action_dim)
        self.log_std_head = nn.Linear(input_dim, action_dim)

        # Log std bounds
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, state):
        """Forward pass to get action distribution."""
        features = self.feature_net(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        """Sample action from policy."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        # Reparameterization trick
        x_t = normal.rsample()
        action = torch.tanh(x_t)

        # Compute log probability
        log_prob = normal.log_prob(x_t)
        # Enforce action bounds
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, mean


class SoftCriticNetwork(nn.Module):
    """Soft Critic Network (Q-function) for SAC."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()

        # Build network
        layers = []
        input_dim = state_dim + action_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, state, action):
        """Forward pass to get Q-value."""
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class SACAgent(BaseAgent):
    """Soft Actor-Critic (SAC) agent for continuous control."""

    def __init__(self, agent_id: str, state_dim: int, action_dim: int, config: Dict[str, Any]):
        super().__init__(agent_id, state_dim, action_dim, config)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters
        self.learning_rate = config.get("learning_rate", 3e-4)
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)  # Soft update coefficient
        self.alpha = config.get("alpha", 0.2)  # Entropy coefficient
        self.auto_entropy_tuning = config.get("auto_entropy_tuning", True)

        # Networks
        self.actor = SoftActorNetwork(state_dim, action_dim).to(self.device)
        self.critic1 = SoftCriticNetwork(state_dim, action_dim).to(self.device)
        self.critic2 = SoftCriticNetwork(state_dim, action_dim).to(self.device)

        # Target networks
        self.critic1_target = SoftCriticNetwork(state_dim, action_dim).to(self.device)
        self.critic2_target = SoftCriticNetwork(state_dim, action_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.learning_rate)

        # Automatic entropy tuning
        if self.auto_entropy_tuning:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.learning_rate)

        # Replay buffer
        self.buffer_size = config.get("buffer_size", 100000)
        self.batch_size = config.get("batch_size", 256)
        self.memory = deque(maxlen=self.buffer_size)

        # Action scaling
        self.action_scale = config.get("action_scale", 1.0)

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action using SAC policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor(state_tensor)
                action = torch.tanh(mean)
            else:
                action, _, _ = self.actor.sample(state_tensor)

        action = action.cpu().numpy()[0] * self.action_scale
        return action

    def store_experience(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def update(self, states: np.ndarray = None, actions: np.ndarray = None,
               rewards: np.ndarray = None, next_states: np.ndarray = None,
               dones: np.ndarray = None) -> Dict[str, float]:
        """Update SAC agent."""
        if len(self.memory) < self.batch_size:
            return {}

        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)

        # Update critics
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)

        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Update actor
        new_actions, log_probs, _ = self.actor.sample(states)
        q1 = self.critic1(states, new_actions)
        q2 = self.critic2(states, new_actions)
        q = torch.min(q1, q2)

        actor_loss = (self.alpha * log_probs - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update alpha (if auto-tuning)
        alpha_loss = 0.0
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # Soft update target networks
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)

        return {
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha,
            "alpha_loss": alpha_loss if isinstance(alpha_loss, float) else alpha_loss.item()
        }

    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, filepath: str):
        """Save agent model."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'alpha': self.alpha,
            'log_alpha': self.log_alpha if self.auto_entropy_tuning else None
        }, filepath)

    def load(self, filepath: str):
        """Load agent model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target'])
        self.alpha = checkpoint['alpha']
        if self.auto_entropy_tuning and checkpoint['log_alpha'] is not None:
            self.log_alpha = checkpoint['log_alpha']


# ============================================
# TD3 (Twin Delayed DDPG) Agent
# ============================================

class TD3ActorNetwork(nn.Module):
    """Deterministic Actor for TD3."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()

        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Tanh())

        self.network = nn.Sequential(*layers)

    def forward(self, state):
        """Forward pass to get deterministic action."""
        return self.network(state)


class TD3CriticNetwork(nn.Module):
    """Twin Q-networks for TD3."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()

        # Q1 network
        layers1 = []
        input_dim = state_dim + action_dim
        for hidden_dim in hidden_dims:
            layers1.append(nn.Linear(input_dim, hidden_dim))
            layers1.append(nn.ReLU())
            layers1.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim
        layers1.append(nn.Linear(input_dim, 1))
        self.q1 = nn.Sequential(*layers1)

        # Q2 network
        layers2 = []
        input_dim = state_dim + action_dim
        for hidden_dim in hidden_dims:
            layers2.append(nn.Linear(input_dim, hidden_dim))
            layers2.append(nn.ReLU())
            layers2.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim
        layers2.append(nn.Linear(input_dim, 1))
        self.q2 = nn.Sequential(*layers2)

    def forward(self, state, action):
        """Forward pass to get both Q-values."""
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_forward(self, state, action):
        """Get Q1 value only."""
        x = torch.cat([state, action], dim=-1)
        return self.q1(x)


class TD3Agent(BaseAgent):
    """Twin Delayed DDPG (TD3) agent for continuous control."""

    def __init__(self, agent_id: str, state_dim: int, action_dim: int, config: Dict[str, Any]):
        super().__init__(agent_id, state_dim, action_dim, config)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters
        self.learning_rate = config.get("learning_rate", 3e-4)
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)
        self.policy_noise = config.get("policy_noise", 0.2)
        self.noise_clip = config.get("noise_clip", 0.5)
        self.policy_freq = config.get("policy_freq", 2)  # Delayed policy updates

        # Networks
        self.actor = TD3ActorNetwork(state_dim, action_dim).to(self.device)
        self.actor_target = TD3ActorNetwork(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = TD3CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic_target = TD3CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        # Replay buffer
        self.buffer_size = config.get("buffer_size", 100000)
        self.batch_size = config.get("batch_size", 256)
        self.memory = deque(maxlen=self.buffer_size)

        # Training state
        self.update_count = 0
        self.action_scale = config.get("action_scale", 1.0)
        self.exploration_noise = config.get("exploration_noise", 0.1)

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action using TD3 policy with exploration noise."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]

        if not deterministic:
            noise = np.random.normal(0, self.exploration_noise, size=action.shape)
            action = np.clip(action + noise, -1, 1)

        action = action * self.action_scale
        return action

    def store_experience(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def update(self, states: np.ndarray = None, actions: np.ndarray = None,
               rewards: np.ndarray = None, next_states: np.ndarray = None,
               dones: np.ndarray = None) -> Dict[str, float]:
        """Update TD3 agent."""
        if len(self.memory) < self.batch_size:
            return {}

        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)

        # Update critic
        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-1, 1)

            # Compute target Q-value
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        actor_loss = torch.tensor(0.0)
        if self.update_count % self.policy_freq == 0:
            # Update actor
            actor_loss = -self.critic.q1_forward(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update target networks
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)

        self.update_count += 1

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item()
        }

    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, filepath: str):
        """Save agent model."""
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict()
        }, filepath)

    def load(self, filepath: str):
        """Load agent model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])


# ============================================
# A2C (Advantage Actor-Critic) Agent
# ============================================

class A2CAgent(BaseAgent):
    """Advantage Actor-Critic (A2C) agent for continuous control."""

    def __init__(self, agent_id: str, state_dim: int, action_dim: int, config: Dict[str, Any]):
        super().__init__(agent_id, state_dim, action_dim, config)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters
        self.learning_rate = config.get("learning_rate", 3e-4)
        self.gamma = config.get("gamma", 0.99)
        self.value_loss_coef = config.get("value_loss_coef", 0.5)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)

        # Networks (shared features)
        from .base_agent import ActorCriticNetwork
        self.actor_critic = ActorCriticNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)

        # Trajectory buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

        # Action scaling
        self.action_scale = config.get("action_scale", 1.0)

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action using A2C policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if deterministic:
                # Get mean action directly
                action, value = self.actor_critic(state_tensor)
            else:
                # Sample action using get_action_and_value
                action, log_prob, value = self.actor_critic.get_action_and_value(state_tensor)

        action = action.cpu().numpy()[0] * self.action_scale
        return action

    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float,
                        value: float, log_prob: float, done: bool):
        """Store transition for on-policy update."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def update(self, states: np.ndarray = None, actions: np.ndarray = None,
               rewards: np.ndarray = None, next_states: np.ndarray = None,
               dones: np.ndarray = None) -> Dict[str, float]:
        """Update A2C agent using collected trajectory."""
        if len(self.states) == 0:
            return {}

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(self.rewards)).to(self.device)
        old_values = torch.FloatTensor(np.array(self.values)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        dones = torch.FloatTensor(np.array(self.dones)).to(self.device)

        # Compute returns
        returns = torch.zeros_like(rewards)
        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return * (1 - dones[t])
            returns[t] = running_return

        # Compute advantages
        advantages = returns - old_values

        # Forward pass - use get_action_and_value
        _, log_probs, values = self.actor_critic.get_action_and_value(states, actions)

        # Compute entropy (approximate with small constant for now)
        entropy = 0.01  # Simplified entropy term

        # Actor loss
        actor_loss = -(log_probs * advantages.detach()).mean()

        # Critic loss
        value_loss = F.mse_loss(values.squeeze(), returns)

        # Total loss
        loss = actor_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Clear trajectory buffer
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

        return {
            "actor_loss": actor_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy if isinstance(entropy, float) else entropy.item(),
            "total_loss": loss.item()
        }

    def save(self, filepath: str):
        """Save agent model."""
        torch.save(self.actor_critic.state_dict(), filepath)

    def load(self, filepath: str):
        """Load agent model."""
        self.actor_critic.load_state_dict(torch.load(filepath, map_location=self.device))


# ============================================
# DQN (Deep Q-Network) Agent - Discretized Actions
# ============================================

class DQNetwork(nn.Module):
    """Deep Q-Network for DQN agent."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()

        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, state):
        """Forward pass to get Q-values for all actions."""
        return self.network(state)


class DQNAgent(BaseAgent):
    """Deep Q-Network (DQN) agent with discretized action space."""

    def __init__(self, agent_id: str, state_dim: int, action_dim: int, config: Dict[str, Any]):
        super().__init__(agent_id, state_dim, action_dim, config)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Discretize continuous action space
        self.num_discrete_actions = config.get("num_discrete_actions", 27)  # 3^3 for 3D control
        self.continuous_action_dim = action_dim
        self._create_action_space()

        # Hyperparameters
        self.learning_rate = config.get("learning_rate", 1e-3)
        self.gamma = config.get("gamma", 0.99)
        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_min = config.get("epsilon_min", 0.01)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)
        self.tau = config.get("tau", 0.005)

        # Networks
        self.q_network = DQNetwork(state_dim, self.num_discrete_actions).to(self.device)
        self.q_target = DQNetwork(state_dim, self.num_discrete_actions).to(self.device)
        self.q_target.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        # Replay buffer
        self.buffer_size = config.get("buffer_size", 50000)
        self.batch_size = config.get("batch_size", 64)
        self.memory = deque(maxlen=self.buffer_size)

        # Action scaling
        self.action_scale = config.get("action_scale", 1.0)

    def _create_action_space(self):
        """Create discrete action space from continuous actions."""
        # For 3D action space, discretize each dimension
        actions_per_dim = int(np.ceil(self.num_discrete_actions ** (1.0 / self.continuous_action_dim)))

        # Create grid of actions
        action_ranges = [np.linspace(-1, 1, actions_per_dim) for _ in range(self.continuous_action_dim)]
        action_grid = np.meshgrid(*action_ranges)

        # Flatten and combine
        self.discrete_actions = np.stack([grid.flatten() for grid in action_grid], axis=1)
        self.num_discrete_actions = self.discrete_actions.shape[0]

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action using epsilon-greedy policy."""
        if not deterministic and np.random.random() < self.epsilon:
            # Random action
            action_idx = np.random.randint(0, self.num_discrete_actions)
        else:
            # Greedy action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            action_idx = q_values.argmax(dim=-1).item()

        # Convert discrete action to continuous
        action = self.discrete_actions[action_idx] * self.action_scale
        return action

    def store_experience(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        # Convert continuous action back to discrete index
        action_idx = np.argmin(np.linalg.norm(self.discrete_actions - action / self.action_scale, axis=1))
        self.memory.append((state, action_idx, reward, next_state, done))

    def update(self, states: np.ndarray = None, actions: np.ndarray = None,
               rewards: np.ndarray = None, next_states: np.ndarray = None,
               dones: np.ndarray = None) -> Dict[str, float]:
        """Update DQN agent."""
        if len(self.memory) < self.batch_size:
            return {}

        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q-values (Double DQN)
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1)
            next_q = self.q_target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Loss
        loss = F.mse_loss(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target network
        self._soft_update()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return {
            "loss": loss.item(),
            "epsilon": self.epsilon,
            "avg_q": current_q.mean().item()
        }

    def _soft_update(self):
        """Soft update target network."""
        for target_param, param in zip(self.q_target.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, filepath: str):
        """Save agent model."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'q_target': self.q_target.state_dict(),
            'epsilon': self.epsilon
        }, filepath)

    def load(self, filepath: str):
        """Load agent model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.q_target.load_state_dict(checkpoint['q_target'])
        self.epsilon = checkpoint['epsilon']
