"""Simulation configuration for UAV swarm environment."""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import yaml


@dataclass
class UAVConfig:
    """Configuration for individual UAV parameters."""
    max_speed: float = 5.0  # m/s
    max_acceleration: float = 2.0  # m/s²
    battery_capacity: float = 100.0  # percentage
    battery_drain_rate: float = 0.1  # per second
    communication_range: float = 50.0  # meters
    sensor_range: float = 30.0  # meters
    payload_capacity: float = 1.0  # kg


@dataclass
class EnvironmentConfig:
    """Configuration for the disaster environment."""
    width: int = 1000  # meters
    height: int = 1000  # meters
    disaster_zones: List[Tuple[int, int, int, int]] = None  # (x, y, width, height)
    obstacles: List[Tuple[int, int, int, int]] = None  # (x, y, width, height)
    target_areas: List[Tuple[int, int, int, int]] = None  # (x, y, width, height)
    wind_conditions: Dict[str, float] = None  # wind speed and direction
    
    def __post_init__(self):
        if self.disaster_zones is None:
            self.disaster_zones = [(200, 200, 100, 100), (600, 400, 150, 120)]
        if self.obstacles is None:
            self.obstacles = [(300, 300, 50, 50), (700, 600, 80, 60)]
        if self.target_areas is None:
            self.target_areas = [(100, 100, 80, 80), (800, 800, 100, 100)]
        if self.wind_conditions is None:
            self.wind_conditions = {"speed": 2.0, "direction": 45.0}


@dataclass
class RLConfig:
    """Configuration for reinforcement learning agents."""
    algorithm: str = "PPO"  # PPO, SAC, TD3
    learning_rate: float = 3e-4
    batch_size: int = 64
    buffer_size: int = 100000
    gamma: float = 0.99
    tau: float = 0.005
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 1000
    target_update_frequency: int = 100
    training_frequency: int = 4
    max_episode_length: int = 5000
    total_timesteps: int = 1000000


@dataclass
class SimulationConfig:
    """Main simulation configuration."""
    num_uavs: int = 5
    simulation_time: float = 300.0  # seconds
    time_step: float = 0.1  # seconds
    render: bool = True
    render_fps: int = 30
    save_data: bool = True
    data_save_path: str = "data/simulation_data.json"
    
    # Sub-configurations
    uav_config: UAVConfig = None
    environment_config: EnvironmentConfig = None
    rl_config: RLConfig = None
    
    def __post_init__(self):
        if self.uav_config is None:
            self.uav_config = UAVConfig()
        if self.environment_config is None:
            self.environment_config = EnvironmentConfig()
        if self.rl_config is None:
            self.rl_config = RLConfig()
    
    @classmethod
    def from_yaml(cls, file_path: str) -> "SimulationConfig":
        """Load configuration from YAML file."""
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Create sub-configurations
        uav_config = UAVConfig(**data.get('uav', {}))
        env_config = EnvironmentConfig(**data.get('environment', {}))
        rl_config = RLConfig(**data.get('rl', {}))
        
        # Create main configuration
        main_data = {k: v for k, v in data.items() if k not in ['uav', 'environment', 'rl']}
        main_data['uav_config'] = uav_config
        main_data['environment_config'] = env_config
        main_data['rl_config'] = rl_config
        
        return cls(**main_data)
    
    def to_yaml(self, file_path: str) -> None:
        """Save configuration to YAML file."""
        data = {
            'num_uavs': self.num_uavs,
            'simulation_time': self.simulation_time,
            'time_step': self.time_step,
            'render': self.render,
            'render_fps': self.render_fps,
            'save_data': self.save_data,
            'data_save_path': self.data_save_path,
            'uav': {
                'max_speed': self.uav_config.max_speed,
                'max_acceleration': self.uav_config.max_acceleration,
                'battery_capacity': self.uav_config.battery_capacity,
                'battery_drain_rate': self.uav_config.battery_drain_rate,
                'communication_range': self.uav_config.communication_range,
                'sensor_range': self.uav_config.sensor_range,
                'payload_capacity': self.uav_config.payload_capacity,
            },
            'environment': {
                'width': self.environment_config.width,
                'height': self.environment_config.height,
                'disaster_zones': self.environment_config.disaster_zones,
                'obstacles': self.environment_config.obstacles,
                'target_areas': self.environment_config.target_areas,
                'wind_conditions': self.environment_config.wind_conditions,
            },
            'rl': {
                'algorithm': self.rl_config.algorithm,
                'learning_rate': self.rl_config.learning_rate,
                'batch_size': self.rl_config.batch_size,
                'buffer_size': self.rl_config.buffer_size,
                'gamma': self.rl_config.gamma,
                'tau': self.rl_config.tau,
                'epsilon_start': self.rl_config.epsilon_start,
                'epsilon_end': self.rl_config.epsilon_end,
                'epsilon_decay': self.rl_config.epsilon_decay,
                'target_update_frequency': self.rl_config.target_update_frequency,
                'training_frequency': self.rl_config.training_frequency,
                'max_episode_length': self.rl_config.max_episode_length,
                'total_timesteps': self.rl_config.total_timesteps,
            }
        }
        
        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)
