"""UAV (Unmanned Aerial Vehicle) implementation for swarm simulation."""

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config.simulation_config import UAVConfig


@dataclass
class UAVState:
    """State representation for a UAV."""
    x: float
    y: float
    z: float
    vx: float  # velocity x
    vy: float  # velocity y
    vz: float  # velocity z
    battery: float
    payload: float
    status: str  # "active", "charging", "maintenance", "emergency"
    last_communication: float
    sensor_data: Dict[str, Any]


class UAV:
    """Individual UAV implementation with physics, sensors, and communication."""
    
    def __init__(self, uav_id: str, config: UAVConfig, initial_position: Tuple[float, float, float] = (0, 0, 10)):
        self.uav_id = uav_id
        self.config = config
        
        # Initialize state
        self.state = UAVState(
            x=initial_position[0],
            y=initial_position[1],
            z=initial_position[2],
            vx=0.0,
            vy=0.0,
            vz=0.0,
            battery=config.battery_capacity,
            payload=0.0,
            status="active",
            last_communication=time.time(),
            sensor_data={}
        )
        
        # Physics parameters
        self.max_speed = config.max_speed
        self.max_acceleration = config.max_acceleration
        self.communication_range = config.communication_range
        self.sensor_range = config.sensor_range
        
        # Control inputs
        self.target_position = initial_position
        self.current_action = np.zeros(3)  # [ax, ay, az] acceleration
        
        # History for trajectory tracking
        self.trajectory = [initial_position]
        self.covered_areas = []
        
        # MCP context integration
        self.context_data = {}
        self.last_context_update = 0.0
        self.episode_rewards = []
    
    def update_physics(self, dt: float):
        """Update UAV physics based on current acceleration."""
        # Apply acceleration
        self.state.vx += self.current_action[0] * dt
        self.state.vy += self.current_action[1] * dt
        self.state.vz += self.current_action[2] * dt
        
        # Limit velocity
        speed = math.sqrt(self.state.vx**2 + self.state.vy**2 + self.state.vz**2)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            self.state.vx *= scale
            self.state.vy *= scale
            self.state.vz *= scale
        
        # Update position
        self.state.x += self.state.vx * dt
        self.state.y += self.state.vy * dt
        self.state.z += self.state.vz * dt
        
        # Keep UAV above ground
        self.state.z = max(self.state.z, 5.0)
        
        # Update trajectory
        self.trajectory.append((self.state.x, self.state.y, self.state.z))
        if len(self.trajectory) > 1000:  # Limit trajectory history
            self.trajectory.pop(0)
    
    def update_battery(self, dt: float):
        """Update battery level based on usage."""
        # Base battery drain
        base_drain = self.config.battery_drain_rate * dt
        
        # Additional drain based on speed and acceleration
        speed = math.sqrt(self.state.vx**2 + self.state.vy**2 + self.state.vz**2)
        speed_drain = speed * 0.01 * dt  # More speed = more battery drain
        
        accel_magnitude = math.sqrt(self.current_action[0]**2 + self.current_action[1]**2 + self.current_action[2]**2)
        accel_drain = accel_magnitude * 0.05 * dt  # Acceleration uses more battery
        
        total_drain = base_drain + speed_drain + accel_drain
        self.state.battery = max(0.0, self.state.battery - total_drain)
        
        # Check if battery is critically low
        if self.state.battery < 10.0 and self.state.status == "active":
            self.state.status = "low_battery"
        elif self.state.battery <= 0.0:
            self.state.status = "emergency"
    
    def set_action(self, action: np.ndarray):
        """Set UAV action (acceleration in x, y, z directions)."""
        # Clamp acceleration to maximum
        self.current_action = np.clip(action, -self.max_acceleration, self.max_acceleration)
    
    def get_observation(self) -> np.ndarray:
        """Get current observation for RL agent."""
        # Basic state information
        obs = np.array([
            self.state.x / 1000.0,  # Normalize position
            self.state.y / 1000.0,
            self.state.z / 100.0,
            self.state.vx / self.max_speed,
            self.state.vy / self.max_speed,
            self.state.vz / self.max_speed,
            self.state.battery / 100.0,
            self.state.payload / self.config.payload_capacity
        ])
        
        # Add context information if available
        if self.context_data:
            context_obs = self._extract_context_observation()
            obs = np.concatenate([obs, context_obs])
        
        return obs
    
    def _extract_context_observation(self) -> np.ndarray:
        """Extract relevant context information for observation."""
        context_obs = []
        
        # Coverage information
        if "coverage_map" in self.context_data:
            # Get local coverage information around UAV
            local_coverage = self._get_local_coverage()
            context_obs.extend(local_coverage)
        
        # Battery status of other UAVs
        if "battery_status" in self.context_data:
            other_batteries = list(self.context_data["battery_status"].values())
            if other_batteries:
                context_obs.append(np.mean(other_batteries) / 100.0)
                context_obs.append(np.min(other_batteries) / 100.0)
            else:
                context_obs.extend([0.0, 0.0])
        
        # Communication network
        if "communication_network" in self.context_data:
            connected_uavs = len(self.context_data["communication_network"].get(self.uav_id, []))
            context_obs.append(connected_uavs / 10.0)  # Normalize
        
        # Target priorities
        if "target_priorities" in self.context_data:
            priorities = list(self.context_data["target_priorities"].values())
            if priorities:
                context_obs.append(np.mean(priorities) / 3.0)  # Normalize
            else:
                context_obs.append(0.0)
        
        return np.array(context_obs)
    
    def _get_local_coverage(self) -> List[float]:
        """Get local coverage information around UAV position."""
        # This is a simplified version - in practice, you'd query the coverage grid
        # around the UAV's position
        return [0.0, 0.0, 0.0]  # Placeholder
    
    def get_sensor_data(self) -> Dict[str, Any]:
        """Get sensor data from UAV's sensors."""
        # Simulate sensor readings
        sensor_data = {
            "position": (self.state.x, self.state.y, self.state.z),
            "velocity": (self.state.vx, self.state.vy, self.state.vz),
            "battery": self.state.battery,
            "payload": self.state.payload,
            "status": self.state.status,
            "sensor_range": self.sensor_range,
            "communication_range": self.communication_range,
            "timestamp": time.time()
        }
        
        self.state.sensor_data = sensor_data
        return sensor_data
    
    def update_context(self, context_data: Dict[str, Any]):
        """Update context information from MCP server."""
        self.context_data = context_data
        self.last_context_update = time.time()
    
    def can_communicate_with(self, other_uav: 'UAV') -> bool:
        """Check if this UAV can communicate with another UAV."""
        distance = math.sqrt(
            (self.state.x - other_uav.state.x)**2 + 
            (self.state.y - other_uav.state.y)**2 + 
            (self.state.z - other_uav.state.z)**2
        )
        return distance <= self.communication_range
    
    def get_covered_area(self) -> List[Tuple[float, float]]:
        """Get area covered by this UAV's sensors."""
        # Return circular area around UAV
        center = (self.state.x, self.state.y)
        radius = self.sensor_range
        
        # Generate points around the circle
        points = []
        for angle in np.linspace(0, 2 * math.pi, 16):
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            points.append((x, y))
        
        return points
    
    def is_in_target_area(self, target_areas: List[Tuple[float, float, float, float]]) -> bool:
        """Check if UAV is in any target area."""
        for x, y, width, height in target_areas:
            if (x <= self.state.x <= x + width and 
                y <= self.state.y <= y + height):
                return True
        return False
    
    def get_distance_to_target(self, target_position: Tuple[float, float, float]) -> float:
        """Get distance to a target position."""
        return math.sqrt(
            (self.state.x - target_position[0])**2 + 
            (self.state.y - target_position[1])**2 + 
            (self.state.z - target_position[2])**2
        )
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get complete state as dictionary."""
        return {
            "uav_id": self.uav_id,
            "position": (self.state.x, self.state.y, self.state.z),
            "velocity": (self.state.vx, self.state.vy, self.state.vz),
            "battery": self.state.battery,
            "payload": self.state.payload,
            "status": self.state.status,
            "trajectory": self.trajectory[-10:],  # Last 10 positions
            "context_data": self.context_data,
            "last_communication": self.state.last_communication
        }
    
    def add_reward(self, reward: float):
        """Add reward to UAV (placeholder for metrics)."""
        self.episode_rewards.append(reward)

    def reset(self, initial_position: Tuple[float, float, float] = None):
        """Reset UAV to initial state."""
        if initial_position is None:
            initial_position = (0, 0, 10)
        
        self.state = UAVState(
            x=initial_position[0],
            y=initial_position[1],
            z=initial_position[2],
            vx=0.0,
            vy=0.0,
            vz=0.0,
            battery=self.config.battery_capacity,
            payload=0.0,
            status="active",
            last_communication=time.time(),
            sensor_data={}
        )
        
        self.trajectory = [initial_position]
        self.covered_areas = []
        self.context_data = {}
        self.current_action = np.zeros(3)
        self.episode_rewards = []
