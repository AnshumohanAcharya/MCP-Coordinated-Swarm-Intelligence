# Frontend Data Diagnosis

## Current Status: ❌ NO DATA

### Problem Analysis

**From your logs:**

1. **MCP Server**: `num_clients: 0` 
   - No simulation is connected to MCP
   - MCP only broadcasts data when clients exist

2. **Dashboard Server**: `Connected to MCP server` ✅
   - Dashboard is connected to MCP
   - But MCP has no data to send

3. **Frontend**: No data received
   - Dashboard never receives `context_broadcast` or `periodic_update`
   - Frontend never receives `simulation_update`

### Data Flow (What Should Happen)

```
Simulation (Python)
    ↓ connects & sends updates
MCP Server (Python)
    ↓ broadcasts periodic_update
Dashboard Server (Node.js)
    ↓ emits simulation_update
Frontend (React)
    ↓ displays data
```

### Current Data Flow (What's Happening)

```
❌ No Simulation Running
    ↓
MCP Server (num_clients: 0)
    ↓ no broadcasts (clients empty)
Dashboard Server (connected but idle)
    ↓ no events
Frontend (waiting for data)
```

## Solution: Run a Simulation

### Quick Test (3 Terminals)

**Terminal 1 - MCP Server:**
```bash
python3 -m mcp_server.server
```
**Expected:** `MCP server started successfully`

**Terminal 2 - Dashboard:**
```bash
cd web_dashboard && npm start
```
**Expected:** `Connected to MCP server`

**Terminal 3 - Simulation:**
```bash
python3 -m simulation.main --headless
```
**Expected:** 
- MCP server logs: `num_clients: 1+`
- Dashboard logs: `Received MCP message: periodic_update`
- Dashboard logs: `Broadcasting simulation_update: 3 UAVs`
- Frontend: Data appears!

## What to Look For

### ✅ Success Indicators

**MCP Server:**
```
num_clients: 1  (or higher)
```

**Dashboard Server:**
```
Received MCP message: periodic_update
Processing periodic_update, data keys: [...]
Broadcasting simulation_update: 3 UAVs, coverage: 1.2%
```

**Frontend:**
- UAV positions updating
- Coverage percentage changing
- Battery levels visible
- Charts showing data

### ❌ Failure Indicators

**MCP Server:**
```
num_clients: 0  ← No simulation connected!
```

**Dashboard Server:**
```
Connected to MCP server  ← Connected but no messages
(no "Received MCP message" logs)
```

**Frontend:**
- Empty charts
- "No data" messages
- Static/placeholder values

## Debugging Steps

### Step 1: Check MCP Server Has Clients

Look at MCP server logs:
- ✅ `num_clients: 1+` = Simulation connected
- ❌ `num_clients: 0` = No simulation running

### Step 2: Check Dashboard Receives Messages

Look at dashboard server logs:
- ✅ `Received MCP message: periodic_update` = Data flowing
- ❌ No "Received MCP message" = MCP not sending (no clients)

### Step 3: Check Frontend Receives Events

Open browser console (F12):
- ✅ `simulation_update` events = Data reaching frontend
- ❌ No events = Dashboard not emitting

## Common Issues

### Issue 1: MCP Server Has No Clients

**Symptom:** `num_clients: 0` in MCP logs

**Solution:** Run a simulation:
```bash
python3 -m simulation.main
```

### Issue 2: Dashboard Not Connected to MCP

**Symptom:** Dashboard logs show reconnection attempts

**Solution:** Start MCP server first:
```bash
python3 -m mcp_server.server
```

### Issue 3: Simulation Not Connecting to MCP

**Symptom:** Simulation runs but MCP shows `num_clients: 0`

**Solution:** Check simulation logs for:
- `Connected to MCP server` ✅
- `MCP connection failed` ❌ (start MCP server)

## Testing Checklist

- [ ] MCP server running (`python3 -m mcp_server.server`)
- [ ] Dashboard running (`cd web_dashboard && npm start`)
- [ ] Simulation running (`python3 -m simulation.main`)
- [ ] MCP logs show `num_clients: 1+`
- [ ] Dashboard logs show `Received MCP message`
- [ ] Frontend shows data updating

## Quick Command Sequence

```bash
# Terminal 1
python3 -m mcp_server.server

# Terminal 2
cd web_dashboard && npm start

# Terminal 3
python3 -m simulation.main --headless
```

Then open `http://localhost:3001` and check:
- Browser console for `simulation_update` events
- Dashboard server logs for message processing
- MCP server logs for client count

