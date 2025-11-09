"""Context manager for aggregating and managing swarm context information."""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass

from config.mcp_config import SwarmContext, ContextMessage


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
        self.last_context_update = 0.0
        
        # Initialize coverage grid
        self._initialize_coverage_grid()
    
    def _initialize_coverage_grid(self):
        """Initialize the coverage tracking grid."""
        # Assuming environment is 1000x1000 meters, use larger resolution for performance
        grid_resolution = max(self.config.coverage_grid_resolution, 50)  # Minimum 50m resolution
        grid_width = 1000 // grid_resolution
        grid_height = 1000 // grid_resolution
        
        self.coverage_grid = CoverageGrid(
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
        
        # Update coverage if this is a position update
        if message.context_type == "position" and "x" in message.data and "y" in message.data:
            x = message.data["x"]
            y = message.data["y"]
            sensor_range = message.data.get("sensor_range", 30.0)
            self.coverage_grid.update_coverage(x, y, sensor_range)
    
    def aggregate_context(self) -> SwarmContext:
        """Aggregate context from all clients into a unified view."""
        current_time = time.time()
        
        # Aggregate battery status and positions
        # First pass: collect all data by client_id
        client_info = {}
        for client_id, data in self.client_data.items():
            if client_id not in client_info:
                client_info[client_id] = {"battery": None, "position": None}
            
            if data["context_type"] == "battery":
                client_info[client_id]["battery"] = data["data"].get("battery_level", 100.0)
            elif data["context_type"] == "position":
                client_info[client_id]["position"] = {
                    "x": data["data"].get("x", 0.0),
                    "y": data["data"].get("y", 0.0),
                    "z": data["data"].get("z", 10.0)
                }
        
        # Second pass: build final dictionaries
        battery_status = {}
        uav_positions = {}
        for client_id, info in client_info.items():
            if info["battery"] is not None:
                battery_status[client_id] = info["battery"]
            if info["position"] is not None:
                uav_positions[client_id] = info["position"]
                # Ensure battery exists for UAVs with positions (for dashboard)
                if client_id not in battery_status:
                    battery_status[client_id] = 100.0  # Default
        
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
        
        # Get emergency events
        emergency_events = self._get_emergency_events()
        
        # Create aggregated context (reduce coverage map size for performance)
        coverage_map = self.coverage_grid.grid[::10, ::10].tolist()  # Downsample by 10x
        
        context = SwarmContext(
            coverage_map=coverage_map,
            battery_status=battery_status,
            uav_positions=uav_positions,
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
        # For now, return simulated data with tidal information if available
        conditions = {
            "temperature": 25.0,  # Celsius
            "humidity": 60.0,  # Percentage
            "visibility": 10.0,  # km
            "weather": "clear"
        }
        
        # Try to get tidal data from simulation if available
        # This is a placeholder - in practice, this would come from the simulation environment
        try:
            from simulation.tidal_data import get_tidal_loader
            import time
            tidal_loader = get_tidal_loader()
            if tidal_loader is not None:
                simulation_time = time.time() - (getattr(self, 'simulation_start_time', time.time()))
                tidal_conditions = tidal_loader.get_environmental_conditions(simulation_time)
                conditions.update({
                    "tidal_pressure": tidal_conditions.get('tidal_pressure', 0.0),
                    "tidal_phase": tidal_conditions.get('tidal_phase', 'unknown'),
                    "wind_modification_factor": tidal_conditions.get('wind_modification_factor', 1.0)
                })
        except Exception:
            # Tidal data not available, use defaults
            pass
        
        return conditions
    
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
