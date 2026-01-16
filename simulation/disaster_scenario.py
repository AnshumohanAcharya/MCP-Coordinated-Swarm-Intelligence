"""Disaster scenario implementation for UAV swarm simulation."""

import numpy as np
import random
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import time

from config.simulation_config import EnvironmentConfig
from .data_loader import DisasterDatasetLoader


@dataclass
class DisasterZone:
    """Represents a disaster zone in the environment."""
    x: float
    y: float
    width: float
    height: float
    severity: float  # 0.0 to 1.0
    priority: int  # 1-4 (low to critical)
    discovered: bool = False
    last_updated: float = 0.0
    radius: float = 50.0 # Added for circular zones support


@dataclass
class Obstacle:
    """Represents an obstacle in the environment."""
    x: float
    y: float
    width: float
    height: float
    height_3d: float  # Height of obstacle
    obstacle_type: str  # "building", "tree", "mountain", "no_fly_zone"


@dataclass
class TargetArea:
    """Represents a target area for UAVs to cover."""
    x: float
    y: float
    width: float
    height: float
    priority: int  # 1-4 (low to critical)
    coverage_required: float  # 0.0 to 1.0
    current_coverage: float = 0.0
    last_updated: float = 0.0


@dataclass
class EmergencyEvent:
    """Represents an emergency event in the environment."""
    event_id: str
    x: float
    y: float
    event_type: str  # "fire", "flood", "earthquake", "medical_emergency"
    severity: float  # 0.0 to 1.0
    priority: int  # 1-4 (low to critical)
    discovered_time: float
    resolved: bool = False


class DisasterScenario:
    """Manages disaster scenario elements and dynamics."""
    
    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.width = config.width
        self.height = config.height
        
        # Initialize Dataset Loader (Vast Dataset Integration)
        self.dataset_loader = DisasterDatasetLoader(dataset_type="wildfire")
        self.timestep_counter = 0
        
        # Initialize disaster zones
        self.disaster_zones = []
        for x, y, w, h in config.disaster_zones:
            zone = DisasterZone(
                x=x, y=y, width=w, height=h,
                severity=random.uniform(0.5, 1.0),
                priority=random.randint(2, 4),
                radius=w/2 # Default radius from config
            )
            self.disaster_zones.append(zone)
        
        # Initialize obstacles
        self.obstacles = []
        for x, y, w, h in config.obstacles:
            obstacle = Obstacle(
                x=x, y=y, width=w, height=h,
                height_3d=random.uniform(20, 100),
                obstacle_type=random.choice(["building", "tree", "mountain"])
            )
            self.obstacles.append(obstacle)
        
        # Initialize target areas
        self.target_areas = []
        for x, y, w, h in config.target_areas:
            target = TargetArea(
                x=x, y=y, width=w, height=h,
                priority=random.randint(1, 3),
                coverage_required=random.uniform(0.7, 1.0)
            )
            self.target_areas.append(target)
        
        # Emergency events
        self.emergency_events = []
        self.event_counter = 0
        
        # Environmental conditions
        self.wind_speed = config.wind_conditions["speed"]
        self.wind_direction = config.wind_conditions["direction"]
        self.weather_conditions = "clear"
        
        # Coverage tracking
        grid_h = int(self.height // 10)
        grid_w = int(self.width // 10)
        self.coverage_grid = np.zeros((grid_h, grid_w), dtype=np.float32)
        self.coverage_threshold = 0.8
    
    def update(self, dt: float):
        """Update scenario dynamics."""
        current_time = time.time()
        
        # Update disaster zones
        self._update_disaster_zones(dt)
        
        # Update emergency events
        self._update_emergency_events(dt)
        
        # Update environmental conditions
        self._update_environmental_conditions(dt)
        
        # Update target area coverage
        self._update_target_coverage()
    
    def _update_disaster_zones(self, dt: float):
        """Update disaster zone dynamics using historical/simulated vast dataset."""
        self.timestep_counter += 1
        
        # Get dynamic zones from dataset
        dataset_zones = self.dataset_loader.get_active_zones(self.timestep_counter)
        
        # Merge dataset zones into existing scenario zones or replace them
        # For simplicity, we'll keep a mix of static config zones and dynamic dataset zones
        active_dataset_zones = []
        for d_zone in dataset_zones:
            zone = DisasterZone(
                x=d_zone["x"],
                y=d_zone["y"],
                width=d_zone["radius"]*2,
                height=d_zone["radius"]*2,
                severity=d_zone["severity"],
                priority=4 if d_zone["severity"] > 0.8 else 3,
                radius=d_zone["radius"]
            )
            active_dataset_zones.append(zone)
            
        # Update our main disaster zones list with dynamic data
        # We'll keep the first few from config and append dynamic ones
        self.disaster_zones = self.disaster_zones[:len(self.config.disaster_zones)] + active_dataset_zones

        import random
        for zone in self.disaster_zones:
            # Simulate zone evolution (e.g., fire spreading)
            if zone.severity < 1.0:
                zone.severity = min(1.0, zone.severity + random.uniform(0.001, 0.01) * dt)
            
            # Innovative feature: Dynamic spreading of critical zones
            if zone.severity > 0.8 and hasattr(zone, 'radius'):
                if random.random() < 0.02: # 2% chance per update
                    zone.radius += 1.0 * dt
                    zone.radius = min(zone.radius, 150.0) # Cap spread
            
            # Update priority based on severity
            if zone.severity > 0.8:
                zone.priority = 4  # Critical
            elif zone.severity > 0.6:
                zone.priority = 3  # High
            elif zone.severity > 0.4:
                zone.priority = 2  # Medium
            else:
                zone.priority = 1  # Low
    
    def _update_emergency_events(self, dt: float):
        """Update emergency events."""
        current_time = time.time()
        
        # Randomly spawn new emergency events
        if random.random() < 0.001 * dt:  # Low probability
            self._spawn_emergency_event()
        
        # Update existing events
        for event in self.emergency_events:
            if not event.resolved:
                # Simulate event evolution
                if event.event_type == "fire":
                    event.severity = min(1.0, event.severity + 0.01 * dt)
                elif event.event_type == "flood":
                    event.severity = min(1.0, event.severity + 0.005 * dt)
    
    def _spawn_emergency_event(self):
        """Spawn a new emergency event."""
        self.event_counter += 1
        event = EmergencyEvent(
            event_id=f"emergency_{self.event_counter}",
            x=random.uniform(0, self.width),
            y=random.uniform(0, self.height),
            event_type=random.choice(["fire", "flood", "earthquake", "medical_emergency"]),
            severity=random.uniform(0.3, 0.8),
            priority=random.randint(2, 4),
            discovered_time=time.time()
        )
        self.emergency_events.append(event)
    
    def _update_environmental_conditions(self, dt: float):
        """Update environmental conditions."""
        # Simulate wind changes
        wind_change = random.uniform(-0.1, 0.1) * dt
        self.wind_speed = max(0, min(10, self.wind_speed + wind_change))
        
        # Simulate weather changes
        if random.random() < 0.0001 * dt:  # Very low probability
            self.weather_conditions = random.choice(["clear", "cloudy", "rainy", "stormy"])
    
    def _update_target_coverage(self):
        """Update target area coverage based on UAV positions."""
        for target in self.target_areas:
            # This would be updated by the simulation when UAVs are in range
            # For now, we'll simulate some coverage
            target.current_coverage = min(1.0, target.current_coverage + random.uniform(0.001, 0.01))
            target.last_updated = time.time()
    
    def add_coverage(self, x: float, y: float, radius: float, coverage_amount: float = 1.0):
        """Add coverage to the coverage grid."""
        # Convert world coordinates to grid coordinates
        grid_x = int(x // 10)
        grid_y = int(y // 10)
        grid_radius = int(radius // 10)
        
        # Update grid cells within radius
        for dx in range(-grid_radius, grid_radius + 1):
            for dy in range(-grid_radius, grid_radius + 1):
                cell_x = grid_x + dx
                cell_y = grid_y + dy
                
                if (0 <= cell_x < self.coverage_grid.shape[1] and 
                    0 <= cell_y < self.coverage_grid.shape[0] and
                    dx*dx + dy*dy <= grid_radius*grid_radius):
                    self.coverage_grid[cell_y, cell_x] = min(1.0, 
                        self.coverage_grid[cell_y, cell_x] + coverage_amount)
    
    def get_coverage_percentage(self) -> float:
        """Get overall coverage percentage."""
        total_cells = self.coverage_grid.size
        covered_cells = np.sum(self.coverage_grid >= self.coverage_threshold)
        return (covered_cells / total_cells) * 100.0
    
    def get_disaster_zones_in_range(self, x: float, y: float, radius: float) -> List[DisasterZone]:
        """Get disaster zones within range of a position."""
        zones_in_range = []
        for zone in self.disaster_zones:
            if self._is_in_range(x, y, radius, zone.x, zone.y, zone.width, zone.height):
                zones_in_range.append(zone)
        return zones_in_range
    
    def get_obstacles_in_range(self, x: float, y: float, radius: float) -> List[Obstacle]:
        """Get obstacles within range of a position."""
        obstacles_in_range = []
        for obstacle in self.obstacles:
            if self._is_in_range(x, y, radius, obstacle.x, obstacle.y, obstacle.width, obstacle.height):
                obstacles_in_range.append(obstacle)
        return obstacles_in_range
    
    def get_target_areas_in_range(self, x: float, y: float, radius: float) -> List[TargetArea]:
        """Get target areas within range of a position."""
        targets_in_range = []
        for target in self.target_areas:
            if self._is_in_range(x, y, radius, target.x, target.y, target.width, target.height):
                targets_in_range.append(target)
        return targets_in_range
    
    def _is_in_range(self, x: float, y: float, radius: float, 
                     obj_x: float, obj_y: float, obj_w: float, obj_h: float) -> bool:
        """Check if a position is within range of an object."""
        # Calculate distance to object center
        obj_center_x = obj_x + obj_w / 2
        obj_center_y = obj_y + obj_h / 2
        distance = np.sqrt((x - obj_center_x)**2 + (y - obj_center_y)**2)
        return distance <= radius
    
    def check_collision(self, x: float, y: float, z: float, radius: float = 5.0) -> bool:
        """Check if a position collides with obstacles."""
        for obstacle in self.obstacles:
            if (obstacle.x <= x <= obstacle.x + obstacle.width and
                obstacle.y <= y <= obstacle.y + obstacle.height and
                z <= obstacle.height_3d):
                return True
        return False
    
    def get_environmental_conditions(self) -> Dict[str, Any]:
        """Get current environmental conditions."""
        return {
            "wind_speed": self.wind_speed,
            "wind_direction": self.wind_direction,
            "weather": self.weather_conditions,
            "temperature": 25.0,  # Simulated
            "humidity": 60.0,  # Simulated
            "visibility": 10.0 if self.weather_conditions == "clear" else 5.0
        }
    
    def get_priority_areas(self) -> List[Tuple[float, float, float, float, int]]:
        """Get all priority areas (disaster zones + target areas) with priorities."""
        priority_areas = []
        
        # Add disaster zones
        for zone in self.disaster_zones:
            priority_areas.append((zone.x, zone.y, zone.width, zone.height, zone.priority))
        
        # Add target areas
        for target in self.target_areas:
            priority_areas.append((target.x, target.y, target.width, target.height, target.priority))
        
        # Sort by priority (highest first)
        priority_areas.sort(key=lambda x: x[4], reverse=True)
        return priority_areas
    
    def get_scenario_summary(self) -> Dict[str, Any]:
        """Get summary of current scenario state."""
        return {
            "disaster_zones": len(self.disaster_zones),
            "obstacles": len(self.obstacles),
            "target_areas": len(self.target_areas),
            "emergency_events": len([e for e in self.emergency_events if not e.resolved]),
            "coverage_percentage": self.get_coverage_percentage(),
            "environmental_conditions": self.get_environmental_conditions()
        }
