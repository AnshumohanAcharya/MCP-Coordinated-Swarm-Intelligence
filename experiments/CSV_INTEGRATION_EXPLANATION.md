# CSV File Integration: Visakhapatnam Tidal Data

## What We Did with the CSV File

The `Visakhapatnam_UTide_full2024_hourly_IST.csv` file contains **real-world hourly tidal data** for Visakhapatnam, India, for the entire year 2024.

### 1. Created Tidal Data Module (`simulation/tidal_data.py`)
- **Purpose**: Load and process the CSV file
- **Features**:
  - Parses time stamps (IST - Indian Standard Time)
  - Extracts tidal pressure data (`prs(m)` column)
  - Provides time-based lookup for any simulation time
  - Calculates derived metrics:
    - `tidal_pressure`: Current tidal pressure in meters
    - `tidal_phase`: High tide, low tide, or transition
    - `wind_modification_factor`: How tides affect wind conditions

### 2. Integrated into Disaster Scenario (`simulation/disaster_scenario.py`)
- **Wind Speed Modification**: 
  - Instead of purely random wind changes, wind speed is now influenced by tidal pressure
  - Higher tidal pressure → stronger wind effects
  - Lower tidal pressure → calmer conditions
- **Real-World Dynamics**: Makes the simulation environment more realistic by using actual coastal data

### 3. Added to MCP Context (`mcp_server/context_manager.py`)
- **Context Broadcast**: The MCP server now includes tidal information in its context broadcasts:
  - `tidal_pressure`: Current pressure reading
  - `tidal_phase`: Current phase (high/low/transition)
  - `wind_modification_factor`: How tides are affecting wind
- **Agent Awareness**: Context-aware RL agents receive this information and can use it in decision-making

## Why This Matters

### 1. **Real-World Relevance**
- Uses actual data from a real location (Visakhapatnam)
- Demonstrates system's ability to integrate external data sources
- Makes simulation more realistic for coastal disaster scenarios

### 2. **Dynamic Environment**
- Environment changes based on real-world patterns
- Not just random - follows actual tidal cycles
- Agents must adapt to changing conditions

### 3. **MCP Innovation**
- Shows MCP can aggregate and broadcast real-world data
- Agents can make decisions based on environmental conditions
- Demonstrates the "context-aware" aspect of the system

### 4. **For Review 2**
- **Demonstrates**: System can integrate real-world data
- **Shows**: MCP can handle diverse data sources
- **Proves**: Architecture is flexible and extensible

## How It Works

1. **Simulation starts**: Tidal loader reads CSV and maps simulation time to real-world time
2. **Each step**: 
   - Disaster scenario queries tidal loader for current conditions
   - Wind speed is modified based on tidal pressure
   - Environmental conditions updated
3. **MCP broadcast**: 
   - Context manager includes tidal data in aggregated context
   - All agents receive tidal information
   - Agents can use this in their decision-making

## Data Format

The CSV contains:
- `Time(IST)`: Timestamp in Indian Standard Time
- `prs(m)`: Tidal pressure in meters (ranges from -0.03m to 2.11m)

## Example Usage

```python
from simulation.tidal_data import get_tidal_loader

loader = get_tidal_loader()
conditions = loader.get_environmental_conditions(simulation_time_seconds=3600)

# Returns:
# {
#   "tidal_pressure": 1.5,  # meters
#   "tidal_phase": "high_tide",
#   "wind_modification_factor": 1.2  # 20% stronger wind
# }
```

## Impact on Simulation

- **Wind patterns**: More realistic, following tidal cycles
- **Agent decisions**: Can adapt to changing environmental conditions
- **System demonstration**: Shows integration of real-world data sources

This integration demonstrates that your system is not just a simulation, but a **real-world applicable system** that can use actual environmental data!

