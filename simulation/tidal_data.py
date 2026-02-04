"""Tidal data module for loading and processing Visakhapatnam tidal data."""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import os


class TidalDataLoader:
    """Load and process tidal data from CSV file."""
    
    def __init__(self, csv_path: str = "Visakhapatnam_UTide_full2024_hourly_IST.csv"):
        """Initialize tidal data loader.
        
        Args:
            csv_path: Path to the CSV file containing tidal data
        """
        self.csv_path = csv_path
        self.data = None
        self.start_time = None
        self.end_time = None
        self.load_data()
    
    def load_data(self):
        """Load tidal data from CSV file."""
        csv_file = Path(self.csv_path)
        
        if not csv_file.exists():
            # Try relative to project root
            project_root = Path(__file__).resolve().parents[2]
            csv_file = project_root / self.csv_path
            
            if not csv_file.exists():
                raise FileNotFoundError(f"Tidal data file not found: {self.csv_path}")
        
        try:
            # Read CSV with time column
            self.data = pd.read_csv(csv_file)
            
            # Parse time column
            if 'Time(IST)' in self.data.columns:
                self.data['Time(IST)'] = pd.to_datetime(self.data['Time(IST)'])
                self.data.set_index('Time(IST)', inplace=True)
            elif 'Time' in self.data.columns:
                self.data['Time'] = pd.to_datetime(self.data['Time'])
                self.data.set_index('Time', inplace=True)
            else:
                raise ValueError("Time column not found in CSV")
            
            # Get pressure column
            if 'prs(m)' in self.data.columns:
                self.pressure_col = 'prs(m)'
            elif 'pressure' in self.data.columns:
                self.pressure_col = 'pressure'
            else:
                raise ValueError("Pressure column not found in CSV")
            
            # Set time range
            self.start_time = self.data.index[0]
            self.end_time = self.data.index[-1]
            
            # Sort by time
            self.data.sort_index(inplace=True)
            
            print(f"Loaded {len(self.data)} tidal data points")
            print(f"Time range: {self.start_time} to {self.end_time}")
            print(f"Pressure range: {self.data[self.pressure_col].min():.2f}m to {self.data[self.pressure_col].max():.2f}m")
            
        except Exception as e:
            raise ValueError(f"Error loading tidal data: {e}")
    
    def get_pressure_at_time(self, simulation_time: float, start_datetime: Optional[datetime] = None) -> float:
        """Get tidal pressure at a specific simulation time.
        
        Args:
            simulation_time: Simulation time in seconds from start
            start_datetime: Starting datetime for simulation (defaults to data start)
        
        Returns:
            Tidal pressure in meters
        """
        if self.data is None or len(self.data) == 0:
            return 0.0
        
        # Determine actual datetime
        if start_datetime is None:
            start_datetime = self.start_time
        
        # Calculate target datetime
        target_datetime = start_datetime + timedelta(seconds=simulation_time)
        
        # Handle time wrapping (if simulation goes beyond data range)
        data_duration = (self.end_time - self.start_time).total_seconds()
        if simulation_time > data_duration:
            # Wrap around or use modulo
            wrapped_time = simulation_time % data_duration
            target_datetime = self.start_time + timedelta(seconds=wrapped_time)
        
        # Clamp to data range
        if target_datetime < self.start_time:
            target_datetime = self.start_time
        elif target_datetime > self.end_time:
            target_datetime = self.end_time
        
        # Interpolate to get pressure value
        try:
            # Find nearest time points
            idx = self.data.index.get_indexer([target_datetime], method='nearest')[0]
            pressure = self.data.iloc[idx][self.pressure_col]
            
            # Linear interpolation for smoother transitions
            if idx > 0 and idx < len(self.data) - 1:
                prev_time = self.data.index[idx - 1]
                next_time = self.data.index[idx + 1]
                
                if prev_time <= target_datetime <= next_time:
                    prev_pressure = self.data.iloc[idx - 1][self.pressure_col]
                    next_pressure = self.data.iloc[idx + 1][self.pressure_col]
                    
                    # Interpolate
                    time_diff = (next_time - prev_time).total_seconds()
                    if time_diff > 0:
                        weight = (target_datetime - prev_time).total_seconds() / time_diff
                        pressure = prev_pressure + weight * (next_pressure - prev_pressure)
            
            return float(pressure)
            
        except Exception as e:
            print(f"Error getting pressure at time {target_datetime}: {e}")
            return 0.0
    
    def get_pressure_range(self) -> Tuple[float, float]:
        """Get min and max pressure values.
        
        Returns:
            Tuple of (min_pressure, max_pressure)
        """
        if self.data is None or len(self.data) == 0:
            return (0.0, 0.0)
        
        min_pressure = float(self.data[self.pressure_col].min())
        max_pressure = float(self.data[self.pressure_col].max())
        return (min_pressure, max_pressure)
    
    def get_normalized_pressure(self, simulation_time: float, start_datetime: Optional[datetime] = None) -> float:
        """Get normalized tidal pressure (0.0 to 1.0).
        
        Args:
            simulation_time: Simulation time in seconds from start
            start_datetime: Starting datetime for simulation
        
        Returns:
            Normalized pressure value between 0.0 and 1.0
        """
        pressure = self.get_pressure_at_time(simulation_time, start_datetime)
        min_p, max_p = self.get_pressure_range()
        
        if max_p == min_p:
            return 0.5  # Default to middle if no variation
        
        normalized = (pressure - min_p) / (max_p - min_p)
        return float(np.clip(normalized, 0.0, 1.0))
    
    def get_wind_modification_factor(self, simulation_time: float, start_datetime: Optional[datetime] = None) -> float:
        """Get wind modification factor based on tidal pressure.
        
        Higher tidal pressure (high tide) typically correlates with stronger winds.
        
        Args:
            simulation_time: Simulation time in seconds from start
            start_datetime: Starting datetime for simulation
        
        Returns:
            Wind modification factor (0.5 to 1.5)
        """
        normalized = self.get_normalized_pressure(simulation_time, start_datetime)
        # Map normalized pressure (0-1) to wind factor (0.5-1.5)
        # Higher tide = stronger wind
        wind_factor = 0.5 + normalized * 1.0
        return float(wind_factor)
    
    def get_tidal_phase(self, simulation_time: float, start_datetime: Optional[datetime] = None) -> str:
        """Get current tidal phase.
        
        Args:
            simulation_time: Simulation time in seconds from start
            start_datetime: Starting datetime for simulation
        
        Returns:
            Tidal phase: 'low', 'rising', 'high', or 'falling'
        """
        normalized = self.get_normalized_pressure(simulation_time, start_datetime)
        
        # Get rate of change (simplified)
        if normalized < 0.25:
            return 'low'
        elif normalized < 0.5:
            return 'rising'
        elif normalized < 0.75:
            return 'high'
        else:
            return 'falling'
    
    def get_environmental_conditions(self, simulation_time: float, start_datetime: Optional[datetime] = None) -> dict:
        """Get environmental conditions based on tidal data.
        
        Args:
            simulation_time: Simulation time in seconds from start
            start_datetime: Starting datetime for simulation
        
        Returns:
            Dictionary with environmental conditions
        """
        pressure = self.get_pressure_at_time(simulation_time, start_datetime)
        normalized = self.get_normalized_pressure(simulation_time, start_datetime)
        wind_factor = self.get_wind_modification_factor(simulation_time, start_datetime)
        phase = self.get_tidal_phase(simulation_time, start_datetime)
        
        return {
            'tidal_pressure': pressure,
            'normalized_tidal_pressure': normalized,
            'wind_modification_factor': wind_factor,
            'tidal_phase': phase,
            'timestamp': simulation_time
        }


# Global instance for easy access
_tidal_loader: Optional[TidalDataLoader] = None


def get_tidal_loader(csv_path: str = "Visakhapatnam_UTide_full2024_hourly_IST.csv") -> TidalDataLoader:
    """Get or create global tidal data loader instance.
    
    Args:
        csv_path: Path to CSV file (only used on first call)
    
    Returns:
        TidalDataLoader instance
    """
    global _tidal_loader
    
    if _tidal_loader is None:
        try:
            _tidal_loader = TidalDataLoader(csv_path)
        except Exception as e:
            print(f"Warning: Could not load tidal data: {e}")
            print("Tidal effects will be disabled.")
            # Return a dummy loader that returns default values
            class DummyLoader:
                def get_pressure_at_time(self, *args, **kwargs):
                    return 0.0
                def get_wind_modification_factor(self, *args, **kwargs):
                    return 1.0
                def get_environmental_conditions(self, *args, **kwargs):
                    return {
                        'tidal_pressure': 0.0,
                        'normalized_tidal_pressure': 0.5,
                        'wind_modification_factor': 1.0,
                        'tidal_phase': 'unknown',
                        'timestamp': 0.0
                    }
            _tidal_loader = DummyLoader()
    
    return _tidal_loader

