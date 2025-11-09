# Quick Demo Guide

## Easy One-Command Demo

Run everything with a single command:

```bash
python3 demo_with_dashboard.py
```

This will:
1. ✅ Start MCP server (port 8765)
2. ✅ Start web dashboard (port 3001)
3. ✅ Run simulation with 5 agents
4. ✅ Display data on dashboard

## What You'll See

1. **Terminal Output**: 
   - Connection status
   - Step-by-step progress
   - Coverage and reward metrics

2. **Web Dashboard** (http://localhost:3001):
   - Real-time UAV positions
   - Coverage visualization
   - Performance metrics
   - Agent status

## Manual Setup (Alternative)

If you prefer to run components separately:

### Terminal 1: MCP Server
```bash
python3 -m mcp_server.server
```

### Terminal 2: Web Dashboard
```bash
cd web_dashboard
npm start
```

### Terminal 3: Simulation
```bash
python3 -m simulation.main --headless
```

## Troubleshooting

### "Port already in use"
- Kill existing processes:
  ```bash
  lsof -ti:8765 | xargs kill -9  # MCP server
  lsof -ti:3001 | xargs kill -9  # Dashboard
  ```

### "No clients connected"
- Wait a few seconds after starting MCP server
- Check that simulation is actually running
- Verify MCP server logs show "Client connected"

### Dashboard shows no data
- Make sure simulation is running
- Check browser console for errors
- Verify MCP server has clients (`num_clients > 0`)

## Demo Duration

Default demo runs for **60 seconds**. To change:
- Edit `demo_with_dashboard.py`
- Change `config.simulation_time = 60` to your desired duration

## Using Trained Agents

If you have trained models in `saved_models/`, the demo will automatically load them. Otherwise, it uses random agents for demonstration.

