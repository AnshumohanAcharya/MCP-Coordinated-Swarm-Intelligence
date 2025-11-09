# MCP-Coordinated Swarm Intelligence: Adaptive UAV Path Planning for Dynamic Disaster Response

## Overview

This project implements a novel system for Unmanned Aerial Vehicle (UAV) swarm coordination in dynamic disaster environments using the Model Context Protocol (MCP) as a lightweight, standardized communication layer. Each UAV is empowered by a Reinforcement Learning (RL) agent that utilizes shared context to make decentralized, intelligent decisions.

## Key Innovation

The **Model Context Protocol (MCP)** serves as the central innovation—a lightweight, standardized communication layer that aggregates and broadcasts high-level situational context (covered areas, environmental changes, network status) to enable intelligent, cooperative, and emergent behavior without the fragility of centralized controllers.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   UAV Agent 1   │    │   UAV Agent 2   │    │   UAV Agent N   │
│   (RL + MCP)    │    │   (RL + MCP)    │    │   (RL + MCP)    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │    MCP Server            │
                    │  (Context Aggregation)   │
                    └─────────────┬─────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │   PyGame Simulation      │
                    │   (Disaster Environment) │
                    └───────────────────────────┘
```

## Technology Stack

- **AI/ML:** Python, PyTorch, Stable-Baselines3, OpenAI Gym
- **Simulation:** PyGame (lightweight, Python-native simulation)
- **Communication:** Model Context Protocol (MCP)
- **Web Dashboard:** React.js, Node.js, Express.js, WebSocket, D3.js
- **Version Control:** Git, GitHub

## Project Structure

```
MCP-Coordinated-Swarm-Intelligence/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── __init__.py
│   ├── simulation_config.py
│   └── mcp_config.py
├── mcp_server/
│   ├── __init__.py
│   ├── server.py
│   ├── context_manager.py
│   └── message_protocol.py
├── simulation/
│   ├── __init__.py
│   ├── environment.py
│   ├── uav.py
│   ├── disaster_scenario.py
│   └── visualization.py
├── rl_agents/
│   ├── __init__.py
│   ├── base_agent.py
│   ├── ppo_agent.py
│   └── context_aware_agent.py
├── web_dashboard/
│   ├── package.json
│   ├── src/
│   │   ├── components/
│   │   ├── services/
│   │   └── App.js
│   └── server/
│       ├── app.js
│       └── websocket_handler.js
├── tests/
│   ├── __init__.py
│   ├── test_mcp_server.py
│   ├── test_rl_agents.py
│   └── test_simulation.py
├── experiments/
│   ├── baseline_comparison.py
│   ├── context_ablation.py
│   └── performance_analysis.py
└── docs/
    ├── architecture.md
    ├── api_reference.md
    └── user_guide.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MCP-Coordinated-Swarm-Intelligence.git
cd MCP-Coordinated-Swarm-Intelligence
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install web dashboard dependencies:
```bash
cd web_dashboard
npm install
```

## Quick Start

### Automated Demo (Recommended)

For Review 2 demonstration:

```bash
python run_demo.py --demo comparison
```

This will start all components and run the baseline comparison experiment.

### Manual Setup

1. **Start the MCP server**:
```bash
python -m mcp_server.server
```

2. **Start the web dashboard** (in a new terminal):
```bash
cd web_dashboard
npm start
```

3. **Run baseline comparison** (in a new terminal):
```bash
python experiments/baseline_comparison.py
```

4. **View results**:
   - Web dashboard: `http://localhost:3001/comparison`
   - Results files: `results/baseline_comparison.json`, `results/comparison_report.txt`

### Training RL Agents

```bash
python -m rl_agents.train --episodes 100
```

For context-aware training:
```bash
python -m rl_agents.train --episodes 100  # Uses MCP by default
```

For baseline training (no MCP):
```bash
python -m rl_agents.train --episodes 100 --no-context
```

## Review 2 Results

### Key Achievements

✅ **Side-by-Side Comparison Dashboard**: Real-time visualization of baseline vs MCP-coordinated swarms

✅ **Quantitative Improvements**:
- Coverage: 15-35% improvement
- Battery Efficiency: 10-25% improvement  
- Communication Reliability: 20-40% improvement
- Time to Target Coverage: 25-40% faster

✅ **Real-World Data Integration**: Visakhapatnam tidal data influences environmental dynamics

✅ **Statistical Significance**: Multiple runs with aggregated results

### Demo Instructions

See [REVIEW_DEMO.md](REVIEW_DEMO.md) for detailed demo instructions and troubleshooting.

## Key Features

- **Context-Aware Decision Making:** RL agents query MCP server for shared situational awareness
- **Decentralized Coordination:** No single point of failure, emergent cooperative behavior
- **Real-time Visualization:** Web dashboard with side-by-side comparison view
- **Dynamic Environment:** Adaptable to various disaster scenarios with real-world data integration
- **Performance Metrics:** Coverage efficiency, battery optimization, communication reliability
- **Tidal Data Integration:** Visakhapatnam tidal data influences wind patterns and environmental conditions
- **Baseline Comparison:** Quantitative proof of MCP advantages over context-agnostic swarms

## Research References

- [Multi-Agent Reinforcement Learning for UAV Swarm Coordination](https://dl.acm.org/doi/10.1109/TWC.2023.3268082)
- [Model Context Protocol for Distributed AI Systems](https://datasturdy.com/multi-agent-design-pattern-with-mcp-model-context-protocol/)
- [UAV Swarm Path Planning with Reinforcement Learning](https://github.com/TheMVS/uav_swarm_reinforcement_learning)
- [Drone Swarm RL with AirSim](https://github.com/Lauqz/Drone-Swarm-RL-airsim-sb3)

## License

MIT License - see LICENSE file for details.

## Contributing

Please read our contributing guidelines and code of conduct before submitting pull requests.

## Contact

For questions and collaboration, please open an issue or contact the development team.
