"""Model Context Protocol configuration for swarm coordination."""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import yaml


@dataclass
class MCPConfig:
    """Configuration for Model Context Protocol server."""
    
    # Server settings
    host: str = "localhost"
    port: int = 8765
    max_connections: int = 50
    
    # Context update settings
    context_update_frequency: float = 1.0  # Hz
    context_retention_time: float = 60.0  # seconds
    
    # Message protocol settings
    max_message_size: int = 1024 * 1024  # 1MB
    message_timeout: float = 5.0  # seconds
    heartbeat_interval: float = 10.0  # seconds
    
    # Context aggregation settings
    coverage_grid_resolution: int = 10  # meters per grid cell
    battery_threshold: float = 20.0  # percentage
    communication_threshold: float = 0.8  # signal strength
    
    # Context types to track
    context_types: List[str] = None
    
    def __post_init__(self):
        if self.context_types is None:
            self.context_types = [
                "coverage_map",
                "battery_status",
                "communication_network",
                "environmental_conditions",
                "target_priorities",
                "obstacle_map",
                "wind_conditions",
                "emergency_events"
            ]
    
    @classmethod
    def from_yaml(cls, file_path: str) -> "MCPConfig":
        """Load MCP configuration from YAML file."""
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, file_path: str) -> None:
        """Save MCP configuration to YAML file."""
        data = {
            'host': self.host,
            'port': self.port,
            'max_connections': self.max_connections,
            'context_update_frequency': self.context_update_frequency,
            'context_retention_time': self.context_retention_time,
            'max_message_size': self.max_message_size,
            'message_timeout': self.message_timeout,
            'heartbeat_interval': self.heartbeat_interval,
            'coverage_grid_resolution': self.coverage_grid_resolution,
            'battery_threshold': self.battery_threshold,
            'communication_threshold': self.communication_threshold,
            'context_types': self.context_types,
        }
        
        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)


@dataclass
class ContextMessage:
    """Standard message format for MCP communication."""
    
    message_type: str  # "update", "query", "response", "heartbeat"
    sender_id: str
    timestamp: float
    context_type: str
    data: Dict[str, Any]
    priority: int = 1  # 1=low, 2=medium, 3=high, 4=critical
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "message_type": self.message_type,
            "sender_id": self.sender_id,
            "timestamp": self.timestamp,
            "context_type": self.context_type,
            "data": self.data,
            "priority": self.priority
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextMessage":
        """Create message from dictionary."""
        return cls(
            message_type=data["message_type"],
            sender_id=data["sender_id"],
            timestamp=data["timestamp"],
            context_type=data["context_type"],
            data=data["data"],
            priority=data.get("priority", 1)
        )


@dataclass
class SwarmContext:
    """Aggregated context information for the entire swarm."""
    
    coverage_map: Dict[str, Any]
    battery_status: Dict[str, float]  # UAV ID -> battery percentage
    uav_positions: Dict[str, Dict[str, float]]  # UAV ID -> {x, y, z}
    communication_network: Dict[str, List[str]]  # UAV ID -> connected UAVs
    environmental_conditions: Dict[str, Any]
    target_priorities: Dict[str, int]  # target ID -> priority
    obstacle_map: Dict[str, Any]
    wind_conditions: Dict[str, float]
    emergency_events: List[Dict[str, Any]]
    last_updated: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for transmission."""
        return {
            "coverage_map": self.coverage_map,
            "battery_status": self.battery_status,
            "uav_positions": self.uav_positions,
            "communication_network": self.communication_network,
            "environmental_conditions": self.environmental_conditions,
            "target_priorities": self.target_priorities,
            "obstacle_map": self.obstacle_map,
            "wind_conditions": self.wind_conditions,
            "emergency_events": self.emergency_events,
            "last_updated": self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SwarmContext":
        """Create context from dictionary."""
        return cls(
            coverage_map=data["coverage_map"],
            battery_status=data["battery_status"],
            uav_positions=data.get("uav_positions", {}),
            communication_network=data["communication_network"],
            environmental_conditions=data["environmental_conditions"],
            target_priorities=data["target_priorities"],
            obstacle_map=data["obstacle_map"],
            wind_conditions=data["wind_conditions"],
            emergency_events=data["emergency_events"],
            last_updated=data["last_updated"]
        )
