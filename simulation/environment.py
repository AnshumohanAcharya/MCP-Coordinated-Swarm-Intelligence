"""Main simulation environment for UAV swarm coordination."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import asyncio
import websockets
import json
import time
from typing import List, Dict, Any, Tuple, Optional
from loguru import logger

from .uav import UAV
from .disaster_scenario import DisasterScenario
from .visualization import SwarmVisualizer
from config.simulation_config import SimulationConfig
from config.mcp_config import ContextMessage


class SwarmEnvironment(gym.Env):
    """Gymnasium environment for UAV swarm coordination with MCP integration."""
    
    def __init__(self, config: SimulationConfig, mcp_server_url: str = "ws://localhost:8765"):
        super().__init__()
        
        self.config = config
        self.mcp_server_url = mcp_server_url
        
        # Initialize UAVs
        self.uavs = []
        self._initialize_uavs()
        
        # Initialize disaster scenario
        self.scenario = DisasterScenario(config.environment_config)
        
        # Initialize visualization
        self.visualizer = SwarmVisualizer(config) if config.render else None
        
        # MCP connection
        self.mcp_websocket = None
        self.mcp_connected = False
        # Prevent concurrent recv() on the same websocket
        self._mcp_receive_lock = asyncio.Lock()
        # Queue for pending MCP updates (collected synchronously, sent asynchronously)
        self._pending_mcp_updates = []
        self._mcp_update_task = None
        
        # Environment state
        self.current_step = 0
        self.episode_rewards = []
        self.episode_length = 0
        
        # Define action and observation spaces
        self._setup_spaces()
        
        # Performance tracking
        self.performance_metrics = {
            "coverage_percentage": [],
            "battery_efficiency": [],
            "communication_reliability": [],
            "collision_count": 0,
            "mission_success": False
        }
    
    def _initialize_uavs(self):
        """Initialize UAVs with random starting positions."""
        for i in range(self.config.num_uavs):
            # Random starting position
            x = np.random.uniform(50, self.config.environment_config.width - 50)
            y = np.random.uniform(50, self.config.environment_config.height - 50)
            z = np.random.uniform(10, 50)
            
            uav = UAV(f"uav_{i}", self.config.uav_config, (x, y, z))
            self.uavs.append(uav)
    
    def _setup_spaces(self):
        """Setup action and observation spaces."""
        # Action space: acceleration in x, y, z for each UAV
        # Each UAV has 3 actions (ax, ay, az) bounded by max_acceleration
        action_dim = self.config.num_uavs * 3
        self.action_space = spaces.Box(
            low=-self.config.uav_config.max_acceleration,
            high=self.config.uav_config.max_acceleration,
            shape=(action_dim,),
            dtype=np.float32
        )
        
        # Observation space: state + context for each UAV
        # Basic state: 8 features per UAV (position, velocity, battery, payload)
        # Context: variable number of features from MCP
        basic_obs_dim = 8  # Basic UAV state
        context_obs_dim = 10  # Context information (estimated)
        obs_dim_per_uav = basic_obs_dim + context_obs_dim
        total_obs_dim = self.config.num_uavs * obs_dim_per_uav
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_obs_dim,),
            dtype=np.float32
        )
    
    async def _connect_mcp(self):
        """Connect to MCP server."""
        try:
            self.mcp_websocket = await websockets.connect(self.mcp_server_url)
            self.mcp_connected = True
            logger.info("Connected to MCP server")
            
            # Register with MCP server
            for uav in self.uavs:
                await self._register_uav_with_mcp(uav)
                
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            self.mcp_connected = False
    
    async def _register_uav_with_mcp(self, uav: UAV):
        """Register UAV with MCP server."""
        if not self.mcp_connected:
            return
        
        registration_message = ContextMessage(
            message_type="register",
            sender_id=uav.uav_id,
            timestamp=time.time(),
            context_type="registration",
            data={
                "client_info": {
                    "uav_id": uav.uav_id,
                    "capabilities": ["position", "battery", "sensor_data"],
                    "communication_range": uav.communication_range,
                    "sensor_range": uav.sensor_range
                }
            }
        )
        
        await self.mcp_websocket.send(json.dumps(registration_message.to_dict()))
    
    async def _mcp_update_processor(self):
        """Background task that processes queued MCP updates."""
        logger.info("MCP update processor started")
        iteration = 0
        while self.mcp_connected:
            try:
                iteration += 1
                # Process pending updates
                if self._pending_mcp_updates:
                    update_data = self._pending_mcp_updates.pop(0)
                    await self._send_mcp_updates(update_data)
                
                # Log every 20 iterations (every ~1 second) to verify it's running
                if iteration % 20 == 0:
                    logger.debug(f"MCP update processor running (iteration {iteration}, queue size: {len(self._pending_mcp_updates)})")
                
                # Small delay to avoid busy-waiting
                await asyncio.sleep(0.05)  # 50ms
            except asyncio.CancelledError:
                logger.info("MCP update processor cancelled")
                break
            except Exception as e:
                logger.error(f"Error in MCP update processor: {e}", exc_info=True)
                await asyncio.sleep(0.1)
    
    async def _send_mcp_updates(self, update_data: List[Dict]):
        """Send MCP updates for UAVs. This is called asynchronously."""
        # Double-check connection state
        if not self.mcp_connected or self.mcp_websocket is None:
            logger.warning(f"MCP not connected when trying to send updates (step {self.current_step})")
            return
        
        try:
            update_count = 0
            for uav_data in update_data:
                # Re-check connection before each send
                if not self.mcp_connected or self.mcp_websocket is None:
                    logger.warning(f"Connection lost during update loop at step {self.current_step}")
                    break
                
                uav_id = uav_data["uav_id"]
                
                # Send position update
                position_message = ContextMessage(
                    message_type="update",
                    sender_id=uav_id,
                    timestamp=time.time(),
                    context_type="position",
                    data=uav_data["position"]
                )
                
                try:
                    await self.mcp_websocket.send(json.dumps(position_message.to_dict()))
                    update_count += 1
                except websockets.exceptions.ConnectionClosed:
                    logger.warning(f"MCP connection closed while sending position update for {uav_id}")
                    self.mcp_connected = False
                    self.mcp_websocket = None
                    break
                except Exception as e:
                    logger.error(f"Error sending position update for {uav_id}: {e}", exc_info=True)
                    break
                
                # Send battery update
                battery_message = ContextMessage(
                    message_type="update",
                    sender_id=uav_id,
                    timestamp=time.time(),
                    context_type="battery",
                    data=uav_data["battery"]
                )
                
                try:
                    await self.mcp_websocket.send(json.dumps(battery_message.to_dict()))
                    update_count += 1
                except websockets.exceptions.ConnectionClosed:
                    logger.warning(f"MCP connection closed while sending battery update for {uav_id}")
                    self.mcp_connected = False
                    self.mcp_websocket = None
                    break
                except Exception as e:
                    logger.error(f"Error sending battery update for {uav_id}: {e}", exc_info=True)
                    break
            
            # Log successful updates (only log every 50 steps to reduce noise)
            if self.current_step % 50 == 0:
                if update_count > 0:
                    logger.info(f"✓ Sent {update_count} MCP updates for {len(update_data)} UAVs (step {self.current_step})")
                else:
                    logger.warning(f"No MCP updates sent at step {self.current_step} (update_count=0)")
        
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"MCP connection closed during send (step {self.current_step}): {e}")
            self.mcp_connected = False
            self.mcp_websocket = None
        except Exception as e:
            logger.error(f"Unexpected error sending MCP updates (step {self.current_step}): {e}", exc_info=True)
            self.mcp_connected = False
            self.mcp_websocket = None
    
    async def _receive_mcp_context(self):
        """Receive context updates from MCP server."""
        if not self.mcp_connected or self.mcp_websocket is None:
            return
        
        # Ensure only one recv is active at a time
        async with self._mcp_receive_lock:
            try:
                # Check for messages (non-blocking)
                message = await asyncio.wait_for(self.mcp_websocket.recv(), timeout=0.01)
                context_data = json.loads(message)
                
                # Update UAVs with context
                if context_data.get("message_type") == "context_broadcast":
                    context = context_data.get("data", {})
                    for uav in self.uavs:
                        uav.update_context(context)
                        
            except asyncio.TimeoutError:
                pass  # No message available
            except websockets.exceptions.ConnectionClosed:
                logger.debug("MCP connection closed during receive. Disabling MCP.")
                self.mcp_connected = False
                self.mcp_websocket = None
            except Exception as e:
                logger.debug(f"Error receiving MCP context: {e}")
                # Don't disable on transient errors
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Convert actions to per-UAV actions
        actions = actions.reshape(self.config.num_uavs, 3)
        
        # Apply actions to UAVs
        for i, uav in enumerate(self.uavs):
            uav.set_action(actions[i])
        
        # Update environment
        self._update_environment()
        
        # Get observations
        observations = self._get_observations()
        
        # Calculate rewards
        reward = self._calculate_reward()
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self._check_truncation()
        
        # Update performance metrics
        self._update_performance_metrics()
        
        # Update step counter
        self.current_step += 1
        self.episode_length += 1
        
        # Store reward
        self.episode_rewards.append(reward)
        
        # Info dictionary
        info = {
            "performance_metrics": self.performance_metrics.copy(),
            "uav_states": [uav.get_state_dict() for uav in self.uavs],
            "scenario_summary": self.scenario.get_scenario_summary()
        }
        
        return observations, reward, terminated, truncated, info
    
    def _update_environment(self):
        """Update environment state."""
        dt = self.config.time_step
        
        # Update UAVs
        for uav in self.uavs:
            uav.update_physics(dt)
            uav.update_battery(dt)
            
            # Add coverage to scenario
            if uav.state.status == "active":
                self.scenario.add_coverage(
                    uav.state.x, uav.state.y, 
                    uav.sensor_range, 0.1
                )
        
        # Update disaster scenario
        self.scenario.update(dt)
        
        # Collect MCP update data synchronously and queue it
        # A background task will send these updates
        if self.mcp_connected and self.mcp_websocket is not None:
            # Collect update data synchronously
            update_data = []
            for uav in self.uavs:
                update_data.append({
                    "uav_id": uav.uav_id,
                    "position": {"x": uav.state.x, "y": uav.state.y, "z": uav.state.z,
                                "sensor_range": uav.sensor_range, "communication_range": uav.communication_range},
                    "battery": {"battery_level": uav.state.battery, "status": uav.state.status}
                })
            
            # Add to queue (will be processed by background task)
            self._pending_mcp_updates.append(update_data)
            # Keep queue size reasonable
            if len(self._pending_mcp_updates) > 10:
                self._pending_mcp_updates.pop(0)
            
            # Log queue status occasionally
            if self.current_step % 50 == 0:
                logger.debug(f"Queued MCP update at step {self.current_step} (queue size: {len(self._pending_mcp_updates)})")
    
    def _handle_mcp_task_error(self, task):
        """Handle errors from MCP async tasks."""
        try:
            if task.done():
                task.result()  # This will raise any exception that occurred
            else:
                logger.warning(f"MCP task not done when error handler called at step {self.current_step}")
        except websockets.exceptions.ConnectionClosed:
            # Connection closed is expected if server stops
            if self.mcp_connected:
                logger.warning(f"MCP connection closed at step {self.current_step}")
                self.mcp_connected = False
                self.mcp_websocket = None
        except asyncio.CancelledError:
            logger.warning(f"MCP task cancelled at step {self.current_step}")
        except Exception as e:
            # Log other errors but don't crash
            logger.error(f"MCP task error at step {self.current_step}: {e}", exc_info=True)
    
    def _get_observations(self) -> np.ndarray:
        """Get observations for all UAVs."""
        observations = []
        
        for uav in self.uavs:
            obs = uav.get_observation()
            observations.extend(obs)
        
        # Pad observations if necessary
        expected_dim = self.observation_space.shape[0]
        current_dim = len(observations)
        
        if current_dim < expected_dim:
            observations.extend([0.0] * (expected_dim - current_dim))
        elif current_dim > expected_dim:
            observations = observations[:expected_dim]
        
        return np.array(observations, dtype=np.float32)
    
    def _calculate_reward(self) -> float:
        """Calculate reward for the current step."""
        reward = 0.0
        
        # Coverage reward
        coverage_reward = self.scenario.get_coverage_percentage() * 0.01
        reward += coverage_reward
        
        # Battery efficiency reward
        avg_battery = np.mean([uav.state.battery for uav in self.uavs])
        battery_reward = (avg_battery / 100.0) * 0.1
        reward += battery_reward
        
        # Collision penalty
        collision_penalty = 0.0
        for uav in self.uavs:
            if self.scenario.check_collision(uav.state.x, uav.state.y, uav.state.z):
                collision_penalty -= 1.0
                self.performance_metrics["collision_count"] += 1
        reward += collision_penalty
        
        # Communication reward
        communication_reward = 0.0
        for i, uav1 in enumerate(self.uavs):
            for uav2 in self.uavs[i+1:]:
                if uav1.can_communicate_with(uav2):
                    communication_reward += 0.01
        reward += communication_reward
        
        # Target area reward
        target_reward = 0.0
        for target in self.scenario.target_areas:
            if target.current_coverage >= target.coverage_required:
                target_reward += target.priority * 0.1
        reward += target_reward
        
        return reward
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        # Check if all UAVs are in emergency state
        if all(uav.state.status == "emergency" for uav in self.uavs):
            return True
        
        # Check if mission is complete (all targets covered)
        if all(target.current_coverage >= target.coverage_required 
               for target in self.scenario.target_areas):
            self.performance_metrics["mission_success"] = True
            return True
        
        return False
    
    def _check_truncation(self) -> bool:
        """Check if episode should be truncated."""
        return self.episode_length >= self.config.rl_config.max_episode_length
    
    def _update_performance_metrics(self):
        """Update performance metrics."""
        self.performance_metrics["coverage_percentage"].append(
            self.scenario.get_coverage_percentage()
        )
        
        avg_battery = np.mean([uav.state.battery for uav in self.uavs])
        self.performance_metrics["battery_efficiency"].append(avg_battery)
        
        # Communication reliability (percentage of UAVs that can communicate)
        communication_links = 0
        total_possible_links = len(self.uavs) * (len(self.uavs) - 1) // 2
        
        for i, uav1 in enumerate(self.uavs):
            for uav2 in self.uavs[i+1:]:
                if uav1.can_communicate_with(uav2):
                    communication_links += 1
        
        reliability = communication_links / total_possible_links if total_possible_links > 0 else 0
        self.performance_metrics["communication_reliability"].append(reliability)
    
    def render(self):
        """Render the environment."""
        if self.visualizer:
            fps = 30  # Placeholder
            return self.visualizer.render(
                self.uavs, self.scenario, 
                self.current_step * self.config.time_step, fps
            )
        return True
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Reset UAVs
        for i, uav in enumerate(self.uavs):
            x = np.random.uniform(50, self.config.environment_config.width - 50)
            y = np.random.uniform(50, self.config.environment_config.height - 50)
            z = np.random.uniform(10, 50)
            uav.reset((x, y, z))
        
        # Reset scenario
        self.scenario = DisasterScenario(self.config.environment_config)
        
        # Reset counters
        self.current_step = 0
        self.episode_length = 0
        self.episode_rewards = []
        
        # Reset performance metrics
        self.performance_metrics = {
            "coverage_percentage": [],
            "battery_efficiency": [],
            "communication_reliability": [],
            "collision_count": 0,
            "mission_success": False
        }
        
        # Get initial observations
        observations = self._get_observations()
        
        info = {
            "performance_metrics": self.performance_metrics.copy(),
            "uav_states": [uav.get_state_dict() for uav in self.uavs],
            "scenario_summary": self.scenario.get_scenario_summary()
        }
        
        return observations, info
    
    def close(self):
        """Close the environment and cleanup resources."""
        if self.visualizer:
            self.visualizer.cleanup()
        
        # Cancel background update task
        if self._mcp_update_task:
            self._mcp_update_task.cancel()
        
        if self.mcp_websocket:
            asyncio.create_task(self.mcp_websocket.close())
        
        logger.info("Environment closed")
    
    async def start_mcp_connection(self):
        """Start MCP connection."""
        await self._connect_mcp()
        # Start background task to process MCP updates
        if self.mcp_connected:
            self._mcp_update_task = asyncio.create_task(self._mcp_update_processor())
            logger.info("Started MCP update processor background task")
        else:
            logger.warning("MCP not connected, cannot start update processor")
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get information about the environment."""
        return {
            "num_uavs": len(self.uavs),
            "environment_size": (self.config.environment_config.width, 
                               self.config.environment_config.height),
            "mcp_connected": self.mcp_connected,
            "performance_metrics": self.performance_metrics,
            "scenario_summary": self.scenario.get_scenario_summary()
        }
