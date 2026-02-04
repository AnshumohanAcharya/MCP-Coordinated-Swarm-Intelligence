# Dashboard Setup Guide - Getting Data to Show

## Problem
The dashboard is running but showing no data because:
1. **MCP server has no clients** (`num_clients: 0` in logs)
2. **Comparison experiment doesn't send live data** - it runs independently and generates plots
3. **Dashboard needs a running simulation** connected to MCP to receive data

## Solution: Run a Live Simulation

To see data on the dashboard, you need to run a **live simulation** (not the comparison experiment).

### Step-by-Step Setup

#### 1. Start MCP Server (Terminal 1)
```bash
python3 -m mcp_server.server
```
**Expected output:**
```
MCP server started successfully
Server listening on ws://localhost:8765
```

#### 2. Start Web Dashboard (Terminal 2)
```bash
cd web_dashboard
npm start
```
**Expected output:**
```
Dashboard server running on port 3001
Connected to MCP server
```

#### 3. Run Live Simulation (Terminal 3)
```bash
python3 -m simulation.main
```

**OR with trained agents:**
```bash
python3 -m rl_agents.train --episodes 10 --render
```

**OR run a quick demo:**
```bash
python3 run_demo.py --mode simulation
```

### What Happens

1. **Simulation connects to MCP** → MCP server shows `num_clients: 1+`
2. **Simulation sends updates** → Position, battery, coverage data
3. **MCP broadcasts to dashboard** → Dashboard receives `context_broadcast` messages
4. **Dashboard displays data** → Real-time visualization appears

## Understanding the Data Flow

```
Simulation (Python)
    ↓ sends updates
MCP Server (Python WebSocket)
    ↓ broadcasts context
Web Dashboard (Node.js)
    ↓ displays
Browser (React)
```

## Troubleshooting

### Dashboard shows "No data" or empty charts

**Check 1: MCP Server has clients**
- Look at MCP server logs
- Should see: `num_clients: 1` or higher
- If `num_clients: 0`, no simulation is connected

**Check 2: Simulation is connected**
- In simulation logs, look for: `Connected to MCP server`
- If you see connection errors, MCP server might not be running

**Check 3: Dashboard connected to MCP**
- In dashboard logs, look for: `Connected to MCP server`
- If you see reconnection attempts, MCP server might not be running

### MCP Server shows `num_clients: 0`

**Solution:** Run a simulation that connects to MCP:
```bash
python3 -m simulation.main
```

### Comparison Experiment Doesn't Show on Dashboard

**This is expected!** The comparison experiment (`baseline_comparison.py`) is designed to:
- Run experiments independently
- Generate plots and reports
- **NOT** send live data to dashboard

**To see comparison on dashboard:**
- You would need to modify `baseline_comparison.py` to send data during runs
- OR run two separate simulations (baseline and MCP) simultaneously
- OR use the comparison view with pre-recorded data

## Quick Test

Run this in order:

**Terminal 1:**
```bash
python3 -m mcp_server.server
```

**Terminal 2:**
```bash
cd web_dashboard && npm start
```

**Terminal 3:**
```bash
python3 -m simulation.main --headless
```

Then open `http://localhost:3001` in your browser. You should see:
- UAV positions updating
- Coverage percentage changing
- Battery levels
- Performance metrics

## For Review 2 Demo

To demonstrate the dashboard:

1. **Start all services:**
   ```bash
   python3 run_demo.py
   ```

2. **OR manually:**
   - Terminal 1: `python3 -m mcp_server.server`
   - Terminal 2: `cd web_dashboard && npm start`
   - Terminal 3: `python3 -m simulation.main`

3. **Open browser:** `http://localhost:3001`

4. **Navigate to different views:**
   - Dashboard: Overview
   - Simulation: Real-time visualization
   - Performance: Metrics and charts
   - Comparison: Side-by-side (if running dual simulations)

