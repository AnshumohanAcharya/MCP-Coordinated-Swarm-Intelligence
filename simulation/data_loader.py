"""Dataset loader for real-world disaster dynamics."""

import numpy as np
import os
from typing import List, Dict, Any, Tuple
from loguru import logger

class DisasterDatasetLoader:
    """Loads and simulates vast disaster datasets (e.g., wildfire spread)."""
    
    def __init__(self, dataset_type: str = "wildfire"):
        self.dataset_type = dataset_type
        self.data_frames = []
        self._generate_synthetic_vast_dataset()
        
    def _generate_synthetic_vast_dataset(self):
        """Generates a vast synthetic dataset based on real-world wildfire patterns.
        In a real scenario, this would load from NetCDF or GeoJSON files.
        """
        logger.info(f"Generating vast {self.dataset_type} dataset (1000 time steps)...")
        
        # Simulate a 1000x1000 grid over 1000 time steps
        width, height = 1000, 1000
        num_steps = 1000
        
        # Start with a few ignition points
        current_map = np.zeros((100, 100)) # Downsampled for memory
        ignitions = [(20, 20), (80, 70), (50, 50)]
        for x, y in ignitions:
            current_map[y, x] = 1.0
            
        for t in range(num_steps):
            # Simple spread model: heat spreads to neighbors
            new_map = current_map.copy()
            # Random wind influence
            wind_x, wind_y = 1, 0 
            
            # Vectorized spread (simplified)
            spread_rate = 0.05
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0: continue
                    shifted = np.roll(np.roll(current_map, dy, axis=0), dx, axis=1)
                    # Bias spread by wind
                    bias = 1.2 if (dx == wind_x and dy == wind_y) else 0.8
                    new_map = np.maximum(new_map, shifted * spread_rate * bias)
            
            # Stochastic growth
            new_map *= (1.0 + np.random.uniform(-0.01, 0.02, size=new_map.shape))
            new_map = np.clip(new_map, 0, 1)
            
            self.data_frames.append(new_map.copy())
            current_map = new_map
            
        logger.info("Vast dataset generation complete.")

    def get_frame(self, timestep: int) -> np.ndarray:
        """Get disaster frame for a specific timestep."""
        idx = timestep % len(self.data_frames)
        return self.data_frames[idx]

    def get_active_zones(self, timestep: int, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Extract high-severity zones from the dataset frame."""
        frame = self.get_frame(timestep)
        # Find clusters/points above threshold
        y_indices, x_indices = np.where(frame > threshold)
        
        # Simplified: return top 5 intensity clusters as zones
        zones = []
        if len(x_indices) > 0:
            # Take representative points for efficiency
            step = max(1, len(x_indices) // 10)
            for i in range(0, len(x_indices), step):
                zones.append({
                    "x": float(x_indices[i] * 10), # Scale back to 1000x1000
                    "y": float(y_indices[i] * 10),
                    "severity": float(frame[y_indices[i], x_indices[i]]),
                    "radius": float(20 + frame[y_indices[i], x_indices[i]] * 50)
                })
        return zones
