"""Context-aware RL agent that integrates MCP context information."""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional
import asyncio
import websockets
import json
import time
from loguru import logger

from .ppo_agent import PPOAgent
from config.mcp_config import ContextMessage


class ContextAwareNetwork(nn.Module):
    """Neural network that processes both local state and global context with LSTM for temporal dependencies."""
    
    def __init__(self, state_dim: int, context_dim: int, action_dim: int, 
                 hidden_dims: list = [256, 256]):
        super().__init__()
        
        self.state_dim = state_dim
        self.context_dim = context_dim
        self.action_dim = action_dim
        
        # Local state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Context encoder (LSTM for predictive modeling)
        self.context_lstm = nn.LSTM(
            input_size=context_dim,
            hidden_size=hidden_dims[0],
            num_layers=1,
            batch_first=True
        )
        
        # Attention mechanism for context integration
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[0],
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Fusion network
        fusion_input_dim = hidden_dims[0] * 2  # state + context
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[1], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dims[1], action_dim),
            nn.Tanh()
        )
        
        # Critic head
        self.critic = nn.Linear(hidden_dims[1], 1)
    
    def forward(self, state: torch.Tensor, context_seq: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """Forward pass with state and context sequence."""
        # Encode state
        state_features = self.state_encoder(state)
        
        # Process context sequence through LSTM
        # context_seq shape: (batch, seq_len, context_dim)
        lstm_out, new_hidden = self.context_lstm(context_seq, hidden)
        
        # We take the last output or use attention over the whole sequence
        context_features = lstm_out
        
        # Apply attention to context
        # Query: state_features, Key/Value: context_features
        # Ensure state_features has sequence dim for attention
        query = state_features.unsqueeze(1) 
        context_attended, _ = self.attention(
            query,
            context_features,
            context_features
        )
        context_attended = context_attended.squeeze(1)
        
        # Fuse state and context
        fused_features = torch.cat([state_features, context_attended], dim=-1)
        fused_features = self.fusion_network(fused_features)
        
        # Get action and value
        action = self.actor(fused_features)
        value = self.critic(fused_features)
        
        return action, value, new_hidden


class ContextAwareAgent(PPOAgent):
    """Context-aware PPO agent that integrates MCP context information."""
    
    def __init__(self, agent_id: str, state_dim: int, action_dim: int, 
                 context_dim: int, config: Dict[str, Any], mcp_server_url: str = "ws://localhost:8765"):
        super().__init__(agent_id, state_dim, action_dim, config)
        
        self.context_dim = context_dim
        self.mcp_server_url = mcp_server_url
        
        # Replace the actor-critic network with context-aware version
        self.actor_critic = ContextAwareNetwork(state_dim, context_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)
        
        # Adjust context_dim if needed to match feature extraction
        # Current extraction yields ~22 features. Let's ensure context_dim is enough.
        if self.context_dim < 32:
             logger.warning(f"Context dim {self.context_dim} might be too small for all features, increasing to 32")
             self.context_dim = 32
             self.actor_critic = ContextAwareNetwork(state_dim, 32, action_dim).to(self.device)
             self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)
        
        # MCP connection
        self.mcp_websocket = None
        self.mcp_connected = False
        self.context_data = {}
        self.last_context_update = 0.0
        
        # Context processing
        self.context_update_frequency = config.get("context_update_frequency", 1.0)  # Hz
        self.context_timeout = config.get("context_timeout", 5.0)  # seconds
        
        # Performance tracking
        self.context_usage_stats = {
            "context_updates_received": 0,
            "context_updates_used": 0,
            "context_latency": [],
            "context_quality": []
        }
    
    async def _connect_mcp(self):
        """Connect to MCP server."""
        try:
            self.mcp_websocket = await websockets.connect(self.mcp_server_url)
            self.mcp_connected = True
            print(f"Agent {self.agent_id} connected to MCP server")
            
            # Register with MCP server
            await self._register_with_mcp()
            
        except Exception as e:
            print(f"Failed to connect to MCP server: {e}")
            self.mcp_connected = False
    
    async def _register_with_mcp(self):
        """Register agent with MCP server."""
        if not self.mcp_connected:
            return
        
        registration_message = ContextMessage(
            message_type="register",
            sender_id=self.agent_id,
            timestamp=time.time(),
            context_type="agent_registration",
            data={
                "agent_type": "context_aware_ppo",
                "capabilities": ["position", "battery", "sensor_data"],
                "context_requirements": ["coverage_map", "battery_status", "communication_network"]
            }
        )
        
        await self.mcp_websocket.send(json.dumps(registration_message.to_dict()))
    
    async def _update_context(self):
        """Update context information from MCP server."""
        if not self.mcp_connected:
            return
        
        try:
            # Send context query
            query_message = ContextMessage(
                message_type="query",
                sender_id=self.agent_id,
                timestamp=time.time(),
                context_type="swarm_context",
                data={"query_type": "full_context"}
            )
            
            await self.mcp_websocket.send(json.dumps(query_message.to_dict()))
            
            # Wait for response
            response = await asyncio.wait_for(
                self.mcp_websocket.recv(), 
                timeout=self.context_timeout
            )
            
            response_data = json.loads(response)
            if response_data.get("message_type") == "query_response":
                self.context_data = response_data.get("data", {}).get("context", {})
                self.last_context_update = time.time()
                self.context_usage_stats["context_updates_received"] += 1
                
        except asyncio.TimeoutError:
            print(f"Agent {self.agent_id}: Context query timeout")
        except Exception as e:
            print(f"Agent {self.agent_id}: Error updating context: {e}")
    
    def _extract_context_features(self) -> np.ndarray:
        """Extract context features for the neural network."""
        if not self.context_data:
            return np.zeros(self.context_dim, dtype=np.float32)
        
        features = []
        
        # Coverage map features
        if "coverage_map" in self.context_data:
            coverage_map = np.array(self.context_data["coverage_map"])
            if coverage_map.size > 0:
                # Extract local coverage statistics
                local_coverage = self._get_local_coverage_features(coverage_map)
                features.extend(local_coverage)
            else:
                features.extend([0.0] * 5)  # Default features
        else:
            features.extend([0.0] * 5)
        
        # Battery status features
        if "battery_status" in self.context_data:
            battery_values = list(self.context_data["battery_status"].values())
            if battery_values:
                features.extend([
                    np.mean(battery_values) / 100.0,  # Average battery
                    np.min(battery_values) / 100.0,   # Minimum battery
                    np.std(battery_values) / 100.0,   # Battery variance
                    len(battery_values) / 10.0        # Number of UAVs
                ])
            else:
                features.extend([0.0] * 4)
        else:
            features.extend([0.0] * 4)
        
        # Communication network features
        if "communication_network" in self.context_data:
            network = self.context_data["communication_network"]
            if self.agent_id in network:
                connected_uavs = len(network[self.agent_id])
                features.append(connected_uavs / 10.0)  # Normalized
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # Target priorities
        if "target_priorities" in self.context_data:
            priorities = list(self.context_data["target_priorities"].values())
            if priorities:
                features.append(np.mean(priorities) / 3.0)  # Normalized
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # Environmental conditions
        if "environmental_conditions" in self.context_data:
            env_cond = self.context_data["environmental_conditions"]
            features.extend([
                env_cond.get("temperature", 25.0) / 50.0,  # Normalized
                env_cond.get("humidity", 60.0) / 100.0,    # Normalized
                env_cond.get("visibility", 10.0) / 20.0    # Normalized
            ])
        else:
            features.extend([0.0] * 3)
        
        # Wind conditions
        if "wind_conditions" in self.context_data:
            wind = self.context_data["wind_conditions"]
            features.extend([
                wind.get("speed", 0.0) / 10.0,        # Normalized
                wind.get("direction", 0.0) / 360.0     # Normalized
            ])
        else:
            features.extend([0.0] * 2)
        
        # Emergency events
        if "emergency_events" in self.context_data:
            events = self.context_data["emergency_events"]
            features.append(len(events) / 10.0)  # Normalized
        else:
            features.append(0.0)
            
        # Occupancy Grid features (SLAM-inspired)
        if "occupancy_grid" in self.context_data:
            occ_data = self.context_data["occupancy_grid"]
            occ_grid = np.array(occ_data.get("grid", []))
            if occ_grid.size > 0:
                features.extend([
                    np.mean(occ_grid == -1), # Obstacle density
                    np.mean(occ_grid == 1),  # Explored free space ratio
                    np.mean(occ_grid == 0)   # Unknown space ratio
                ])
            else:
                features.extend([0.0] * 3)
        else:
            features.extend([0.0] * 3)
        
        # Pad or truncate to context_dim
        while len(features) < self.context_dim:
            features.append(0.0)
        
        if len(features) > self.context_dim:
            features = features[:self.context_dim]
        
        return np.array(features, dtype=np.float32)
    
    def _get_local_coverage_features(self, coverage_map: np.ndarray) -> List[float]:
        """Extract local coverage features around agent position."""
        # This is a simplified version - in practice, you'd extract features
        # around the agent's current position
        return [
            np.mean(coverage_map),      # Average coverage
            np.std(coverage_map),       # Coverage variance
            np.sum(coverage_map > 0.5) / coverage_map.size,  # High coverage ratio
            np.sum(coverage_map > 0.8) / coverage_map.size,  # Very high coverage ratio
            np.sum(coverage_map == 0) / coverage_map.size    # Uncovered ratio
        ]
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select an action using state and context information."""
        # Extract context features
        context_features = self._extract_context_features()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            # Ensure temporal dimension for LSTM (batch, seq_len, context_dim)
            context_tensor = torch.FloatTensor(context_features).unsqueeze(0).unsqueeze(0).to(self.device)
            
            if deterministic:
                action, _, _ = self.actor_critic(state_tensor, context_tensor)
                action = action.squeeze(0).cpu().numpy()
            else:
                action, value, _ = self.actor_critic(state_tensor, context_tensor)
                action = action.squeeze(0).cpu().numpy()
                
                # Store for training
                if self.training:
                    # Note: This is simplified - in practice, you'd need to store
                    # context features and modify the buffer accordingly
                    self.buffer.add(
                        state, action, 0.0,  # reward will be updated later
                        value.item(), 0.0, False  # log_prob and done will be updated later
                    )
        
        # Scale action
        action = action * self.action_scale
        
        return action
    
    async def start_mcp_connection(self):
        """Start MCP connection."""
        await self._connect_mcp()
    
    async def update_context_async(self):
        """Update context asynchronously."""
        if self.mcp_connected:
            await self._update_context()
    
    def get_context_usage_stats(self) -> Dict[str, Any]:
        """Get context usage statistics."""
        return {
            "mcp_connected": self.mcp_connected,
            "last_context_update": self.last_context_update,
            "context_age": time.time() - self.last_context_update if self.last_context_update > 0 else float('inf'),
            "context_updates_received": self.context_usage_stats["context_updates_received"],
            "context_updates_used": self.context_usage_stats["context_updates_used"],
            "context_quality": np.mean(self.context_usage_stats["context_quality"]) if self.context_usage_stats["context_quality"] else 0.0
        }
    
    def close(self):
        """Close MCP connection."""
        if self.mcp_websocket:
            asyncio.create_task(self.mcp_websocket.close())
