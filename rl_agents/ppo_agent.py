"""PPO (Proximal Policy Optimization) agent for UAV swarm coordination."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Dict, Any, List, Tuple
import random
from collections import deque

from .base_agent import BaseAgent, ActorCriticNetwork


class PPOBuffer:
    """Buffer for storing PPO experience."""
    
    def __init__(self, buffer_size: int, state_dim: int, action_dim: int, device: str = "cpu"):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Buffers
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool_)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float, 
            value: float, log_prob: float, done: bool):
        """Add experience to buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def compute_advantages_and_returns(self, next_value: float, gamma: float = 0.99, 
                                     lambda_gae: float = 0.95):
        """Compute advantages and returns using GAE."""
        advantages = np.zeros_like(self.rewards)
        last_advantage = 0
        
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value_t = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value_t = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * next_value_t * next_non_terminal - self.values[t]
            advantages[t] = last_advantage = delta + gamma * lambda_gae * next_non_terminal * last_advantage
        
        self.advantages = advantages
        self.returns = advantages + self.values
    
    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Get a batch of experiences."""
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return {
            "states": torch.FloatTensor(self.states[indices]).to(self.device),
            "actions": torch.FloatTensor(self.actions[indices]).to(self.device),
            "rewards": torch.FloatTensor(self.rewards[indices]).to(self.device),
            "values": torch.FloatTensor(self.values[indices]).to(self.device),
            "log_probs": torch.FloatTensor(self.log_probs[indices]).to(self.device),
            "dones": torch.BoolTensor(self.dones[indices]).to(self.device),
            "advantages": torch.FloatTensor(self.advantages[indices]).to(self.device),
            "returns": torch.FloatTensor(self.returns[indices]).to(self.device)
        }
    
    def clear(self):
        """Clear the buffer."""
        self.ptr = 0
        self.size = 0


class PPOAgent(BaseAgent):
    """PPO agent for UAV swarm coordination."""
    
    def __init__(self, agent_id: str, state_dim: int, action_dim: int, config: Dict[str, Any]):
        super().__init__(agent_id, state_dim, action_dim, config)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters
        self.learning_rate = config.get("learning_rate", 3e-4)
        self.gamma = config.get("gamma", 0.99)
        self.lambda_gae = config.get("lambda_gae", 0.95)
        self.epsilon_clip = config.get("epsilon_clip", 0.2)
        self.value_loss_coef = config.get("value_loss_coef", 0.5)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.ppo_epochs = config.get("ppo_epochs", 4)
        self.batch_size = config.get("batch_size", 64)
        self.buffer_size = config.get("buffer_size", 2048)
        
        # Networks
        self.actor_critic = ActorCriticNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)
        
        # Buffer
        self.buffer = PPOBuffer(self.buffer_size, state_dim, action_dim, str(self.device))
        
        # Training state
        self.training = True
        self.update_count = 0
        
        # Action scaling
        self.action_scale = config.get("action_scale", 1.0)
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select an action using the current policy."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if deterministic:
                action, _, _ = self.actor_critic.get_action_and_value(state_tensor)
                action = action.squeeze(0).cpu().numpy()
            else:
                action, log_prob, value = self.actor_critic.get_action_and_value(state_tensor)
                action = action.squeeze(0).cpu().numpy()
                
                # Store for training
                if self.training:
                    self.buffer.add(
                        state, action, 0.0,  # reward will be updated later
                        value.item(), log_prob.item(), False  # done will be updated later
                    )
        
        # Scale action
        action = action * self.action_scale
        
        return action
    
    def update(self, states: np.ndarray, actions: np.ndarray, 
               rewards: np.ndarray, next_states: np.ndarray, 
               dones: np.ndarray) -> Dict[str, float]:
        """Update the agent with new experience."""
        if not self.training:
            return {}
        
        # Update buffer with final rewards and dones
        for i in range(len(states)):
            if i < len(self.buffer.states):
                self.buffer.rewards[i] = rewards[i]
                self.buffer.dones[i] = dones[i]
        
        # Update if buffer is full
        if self.buffer.size >= self.buffer_size:
            return self._update_networks()
        
        return {}
    
    def _update_networks(self) -> Dict[str, float]:
        """Update actor and critic networks using PPO."""
        # Compute advantages and returns
        with torch.no_grad():
            next_state = torch.FloatTensor(self.buffer.states[-1]).unsqueeze(0).to(self.device)
            _, _, next_value = self.actor_critic.get_action_and_value(next_state)
            next_value = next_value.item()
        
        self.buffer.compute_advantages_and_returns(next_value, self.gamma, self.lambda_gae)
        
        # Normalize advantages
        advantages = self.buffer.advantages[:self.buffer.size]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.buffer.advantages[:self.buffer.size] = advantages
        
        # PPO updates
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        
        for _ in range(self.ppo_epochs):
            # Get batch
            batch = self.buffer.get_batch(self.batch_size)
            
            # Get current policy outputs
            _, new_log_probs, new_values = self.actor_critic.get_action_and_value(
                batch["states"], batch["actions"]
            )
            
            # Compute ratios
            ratio = torch.exp(new_log_probs - batch["log_probs"])
            
            # Compute surrogate losses
            surr1 = ratio * batch["advantages"]
            surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * batch["advantages"]
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(new_values.squeeze(), batch["returns"])
            
            # Entropy loss
            entropy_loss = -new_log_probs.mean()
            
            # Total loss
            total_loss = (policy_loss + 
                         self.value_loss_coef * value_loss + 
                         self.entropy_coef * entropy_loss)
            
            # Optimize
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
        
        # Clear buffer
        self.buffer.clear()
        
        # Update performance history
        self.performance_history["losses"].append({
            "policy_loss": total_policy_loss / self.ppo_epochs,
            "value_loss": total_value_loss / self.ppo_epochs,
            "entropy_loss": total_entropy_loss / self.ppo_epochs
        })
        
        self.update_count += 1
        
        return {
            "policy_loss": total_policy_loss / self.ppo_epochs,
            "value_loss": total_value_loss / self.ppo_epochs,
            "entropy_loss": total_entropy_loss / self.ppo_epochs,
            "update_count": self.update_count
        }
    
    def save(self, filepath: str) -> None:
        """Save the agent's model."""
        torch.save({
            "actor_critic_state_dict": self.actor_critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "performance_history": self.performance_history
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """Load the agent's model."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.performance_history = checkpoint.get("performance_history", self.performance_history)
    
    def set_training_mode(self, training: bool):
        """Set training mode for the agent."""
        self.training = training
        if training:
            self.actor_critic.train()
        else:
            self.actor_critic.eval()
    
    def get_exploration_rate(self) -> float:
        """Get current exploration rate (for PPO, this is implicit in the policy)."""
        return 1.0  # PPO doesn't have explicit exploration rate
    
    def decay_exploration(self, decay_rate: float = 0.995):
        """Decay exploration (not applicable for PPO)."""
        pass  # PPO doesn't use explicit exploration decay
