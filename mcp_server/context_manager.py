"""Context manager for aggregating and managing swarm context information."""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass

from config.mcp_config import SwarmContext, ContextMessage


@dataclass
class OccupancyGrid:
    """Grid-based occupancy tracking for SLAM-inspired mapping."""
    width: int
    height: int
    resolution: float
    grid: np.ndarray  # 0 = unknown, 1 = free, -1 = occupied
    last_updated: float
    
    def update_occupancy(self, x: float, y: float, sensor_data: Dict[str, Any]):
        """Update occupancy grid with sensor data."""
        # Convert world coordinates to grid coordinates
        grid_x = int(x / self.resolution)
        grid_y = int(y / self.resolution)
        
        # Simple laser scan-like update if sensor_data contains obstacles
        if "obstacles" in sensor_data:
            for obs in sensor_data["obstacles"]:
                obs_x, obs_y = obs["x"], obs["y"]
                g_obs_x = int(obs_x / self.resolution)
                g_obs_y = int(obs_y / self.resolution)
                
                if 0 <= g_obs_x < self.width and 0 <= g_obs_y < self.height:
                    self.grid[g_obs_y, g_obs_x] = -1
        
        # Mark area around UAV as free
        radius = int(5.0 / self.resolution)
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                cell_x, cell_y = grid_x + dx, grid_y + dy
                if (0 <= cell_x < self.width and 
                    0 <= cell_y < self.height and
                    dx*dx + dy*dy <= radius*radius):
                    if self.grid[cell_y, cell_x] == 0:
                        self.grid[cell_y, cell_x] = 1
        
        self.last_updated = time.time()

@dataclass
class CoverageGrid:
    """Grid-based coverage tracking."""
    width: int
    height: int
    resolution: float
    grid: np.ndarray  # 0 = uncovered, 1 = covered
    last_updated: float
    
    def update_coverage(self, x: float, y: float, radius: float):
        """Update coverage grid with new coverage area."""
        # Convert world coordinates to grid coordinates
        grid_x = int(x / self.resolution)
        grid_y = int(y / self.resolution)
        grid_radius = int(radius / self.resolution)
        
        # Update grid cells within radius
        for dx in range(-grid_radius, grid_radius + 1):
            for dy in range(-grid_radius, grid_radius + 1):
                cell_x = grid_x + dx
                cell_y = grid_y + dy
                
                if (0 <= cell_x < self.width and 
                    0 <= cell_y < self.height and
                    dx*dx + dy*dy <= grid_radius*grid_radius):
                    self.grid[cell_y, cell_x] = 1
        
        self.last_updated = time.time()
    
    def get_coverage_percentage(self) -> float:
        """Get percentage of area covered."""
        total_cells = self.width * self.height
        covered_cells = np.sum(self.grid)
        return (covered_cells / total_cells) * 100.0
    
    def get_uncovered_areas(self) -> List[Tuple[int, int]]:
        """Get list of uncovered grid cells."""
        uncovered = np.where(self.grid == 0)
        return list(zip(uncovered[1], uncovered[0]))  # (x, y) format


class ContextManager:
    """Manages and aggregates context information for the swarm."""
    
    def __init__(self, config):
        self.config = config
        self.context_history = deque(maxlen=100)  # Keep last 100 context updates
        self.client_data = {}  # Store individual client data
        self.coverage_grid = None
        self.occupancy_grid = None
        self.last_context_update = 0.0
        
        # Initialize grids
        self._initialize_grids()
    
    def _initialize_grids(self):
        """Initialize tracking grids."""
        # Assuming environment is 1000x1000 meters
        grid_resolution = max(self.config.coverage_grid_resolution, 20)
        grid_width = 1000 // grid_resolution
        grid_height = 1000 // grid_resolution
        
        self.coverage_grid = CoverageGrid(
            width=grid_width,
            height=grid_height,
            resolution=grid_resolution,
            grid=np.zeros((grid_height, grid_width), dtype=np.int8),
            last_updated=time.time()
        )
        
        self.occupancy_grid = OccupancyGrid(
            width=grid_width,
            height=grid_height,
            resolution=grid_resolution,
            grid=np.zeros((grid_height, grid_width), dtype=np.int8),
            last_updated=time.time()
        )
    
    def update_client_data(self, client_id: str, message: ContextMessage):
        """Update data from a specific client."""
        self.client_data[client_id] = {
            "data": message.data,
            "timestamp": message.timestamp,
            "context_type": message.context_type
        }
        
        # Update coverage and occupancy if this is a position update
        if message.context_type == "position" and "x" in message.data and "y" in message.data:
            x = message.data["x"]
            y = message.data["y"]
            sensor_range = message.data.get("sensor_range", 30.0)
            self.coverage_grid.update_coverage(x, y, sensor_range)
            
            # Update occupancy with sensor data
            sensor_data = message.data.get("sensor_data", {})
            self.occupancy_grid.update_occupancy(x, y, sensor_data)
    
    def has_critical_events(self) -> bool:
        """Check if any critical events exist that require higher protocol frequency."""
        # Check battery levels
        for client_id, data in self.client_data.items():
             if data.get("context_type") == "position": # Position updates often contain battery in 'data'
                  battery = data.get("data", {}).get("battery", 100.0)
                  if battery < 20.0:
                       return True
             elif data.get("context_type") == "battery":
                  battery = data.get("data", {}).get("battery_level", 100.0)
                  if battery < 20.0:
                       return True
        
        # Check for emergency events
        if any(event.get("severity", 0) > 0.8 for event in self._get_emergency_events()):
             return True
             
        # Check for high disaster zone severity
        for zone in getattr(self.config, "disaster_zones", []):
             if len(zone) >= 4 and zone[3] > 0.8: # Assuming (x,y,w,h,severity) or similar
                  return True
                  
        return False

    def aggregate_context(self) -> SwarmContext:
        """Aggregate context from all clients into a unified view."""
        current_time = time.time()
        
        # Aggregate battery status
        battery_status = {}
        for client_id, data in self.client_data.items():
            if data["context_type"] == "battery":
                battery_status[client_id] = data["data"].get("battery_level", 100.0)
        
        # Aggregate communication network
        communication_network = self._build_communication_network()
        
        # Get environmental conditions
        environmental_conditions = self._get_environmental_conditions()
        
        # Get target priorities
        target_priorities = self._get_target_priorities()
        
        # Get obstacle map
        obstacle_map = self._get_obstacle_map()
        
        # Get wind conditions
        wind_conditions = self._get_wind_conditions()
        
        # Get occupancy grid (SLAM feature)
        occupancy_grid_data = {
            "grid": self.occupancy_grid.grid[::10, ::10].tolist(), # Downsample
            "resolution": self.occupancy_grid.resolution * 10,
            "width": self.occupancy_grid.width // 10,
            "height": self.occupancy_grid.height // 10
        }
        
        # Get emergency events
        emergency_events = self._get_emergency_events()
        
        # Create aggregated context (reduce coverage map size for performance)
        coverage_map = self.coverage_grid.grid[::10, ::10].tolist()  # Downsample by 10x
        
        context = SwarmContext(
            coverage_map=coverage_map,
            occupancy_grid=occupancy_grid_data,
            battery_status=battery_status,
            communication_network=communication_network,
            environmental_conditions=environmental_conditions,
            target_priorities=target_priorities,
            obstacle_map=obstacle_map,
            wind_conditions=wind_conditions,
            emergency_events=emergency_events,
            last_updated=current_time
        )
        
        # Store in history
        self.context_history.append(context)
        self.last_context_update = current_time
        
        return context
    
    def _build_communication_network(self) -> Dict[str, List[str]]:
        """Build communication network based on client positions and ranges."""
        network = defaultdict(list)
        positions = {}
        
        # Collect positions
        for client_id, data in self.client_data.items():
            if data["context_type"] == "position":
                positions[client_id] = {
                    "x": data["data"].get("x", 0),
                    "y": data["data"].get("y", 0),
                    "range": data["data"].get("communication_range", 50.0)
                }
        
        # Build network connections
        for client_id, pos in positions.items():
            for other_id, other_pos in positions.items():
                if client_id != other_id:
                    distance = np.sqrt(
                        (pos["x"] - other_pos["x"])**2 + 
                        (pos["y"] - other_pos["y"])**2
                    )
                    if distance <= pos["range"]:
                        network[client_id].append(other_id)
        
        return dict(network)
    
    def _get_environmental_conditions(self) -> Dict[str, Any]:
        """Get current environmental conditions."""
        # This would typically come from sensors or external data
        # For now, return simulated data
        return {
            "temperature": 25.0,  # Celsius
            "humidity": 60.0,  # Percentage
            "visibility": 10.0,  # km
            "weather": "clear"
        }
    
    def _get_target_priorities(self) -> Dict[str, int]:
        """Get target priorities based on current context."""
        # This would be determined by mission objectives
        # For now, return simulated priorities
        return {
            "target_1": 3,  # High priority
            "target_2": 2,  # Medium priority
            "target_3": 1   # Low priority
        }
    
    def _get_obstacle_map(self) -> Dict[str, Any]:
        """Get current obstacle map."""
        # This would come from sensor data or pre-mapped obstacles
        return {
            "obstacles": [
                {"x": 300, "y": 300, "width": 50, "height": 50},
                {"x": 700, "y": 600, "width": 80, "height": 60}
            ],
            "last_updated": time.time()
        }
    
    def _get_wind_conditions(self) -> Dict[str, float]:
        """Get current wind conditions."""
        # This would come from weather data or sensors
        return {
            "speed": 2.0,  # m/s
            "direction": 45.0,  # degrees
            "gust_speed": 3.5  # m/s
        }
    
    def _get_emergency_events(self) -> List[Dict[str, Any]]:
        """Get current emergency events."""
        # This would come from emergency detection systems
        return []
    
    def get_context_for_client(self, client_id: str) -> Optional[SwarmContext]:
        """Get context information relevant to a specific client."""
        if not self.client_data:
            return None
        
        # Get aggregated context
        context = self.aggregate_context()
        
        # Filter context based on client's communication range
        client_pos = None
        client_range = 50.0
        
        if client_id in self.client_data:
            client_data = self.client_data[client_id]
            if client_data["context_type"] == "position":
                client_pos = {
                    "x": client_data["data"].get("x", 0),
                    "y": client_data["data"].get("y", 0)
                }
                client_range = client_data["data"].get("communication_range", 50.0)
        
        # For now, return full context
        # In a more sophisticated implementation, we would filter based on relevance
        return context
    
    def get_coverage_statistics(self) -> Dict[str, Any]:
        """Get coverage statistics."""
        if not self.coverage_grid:
            return {}
        
        return {
            "coverage_percentage": self.coverage_grid.get_coverage_percentage(),
            "uncovered_areas": self.coverage_grid.get_uncovered_areas(),
            "last_updated": self.coverage_grid.last_updated
        }
    
    def cleanup_old_data(self, max_age: float = 60.0):
        """Clean up old client data."""
        current_time = time.time()
        clients_to_remove = []
        
        for client_id, data in self.client_data.items():
            if current_time - data["timestamp"] > max_age:
                clients_to_remove.append(client_id)
        
        for client_id in clients_to_remove:
            del self.client_data[client_id]
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of current context state."""
        return {
            "num_clients": len(self.client_data),
            "coverage_stats": self.get_coverage_statistics(),
            "last_update": self.last_context_update,
            "context_history_size": len(self.context_history)
        }
