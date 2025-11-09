const WebSocket = require('ws');
const axios = require('axios');

class WebSocketHandler {
  constructor(io) {
    this.io = io;
    this.mcpWebSocket = null;
    this.mcpConnected = false;
    this.reconnectDelayMs = 5000;
    this.maxReconnectDelayMs = 60000;
    this.reconnectTimer = null;
    this.lastErrorLogAt = 0;
    this.errorLogThrottleMs = 5000;
    this.simulationData = {
      uavs: [],
      scenario: {},
      performance: {},
      timestamp: null
    };
    this.agentsData = {};
    this.performanceMetrics = {
      coverage: [],
      battery: [],
      communication: [],
      rewards: []
    };
    
    // Dual simulation support for comparison
    this.comparisonMode = false;
    this.baselineSimulationData = {
      uavs: [],
      scenario: {},
      performance: {},
      timestamp: null
    };
    this.mcpSimulationData = {
      uavs: [],
      scenario: {},
      performance: {},
      timestamp: null
    };
    
    // Connect to MCP server (optional)
    if ((process.env.MCP_AUTO_CONNECT || 'true').toLowerCase() !== 'false') {
      this.connectToMCP();
    } else {
      console.log('MCP auto-connect disabled via MCP_AUTO_CONNECT=false');
    }
    
    // Start data collection
    this.startDataCollection();
    
    // Setup socket.io event handlers
    this.setupSocketHandlers();
  }
  
  setupSocketHandlers() {
    this.io.on('connection', (socket) => {
      console.log('Client connected:', socket.id);
      
      socket.on('request_comparison_data', () => {
        socket.emit('baseline_simulation_update', this.baselineSimulationData);
        socket.emit('mcp_simulation_update', this.mcpSimulationData);
      });
      
      socket.on('start_comparison', () => {
        this.comparisonMode = true;
        console.log('Comparison mode started');
        this.io.emit('comparison_started');
      });
      
      socket.on('stop_comparison', () => {
        this.comparisonMode = false;
        console.log('Comparison mode stopped');
        this.io.emit('comparison_stopped');
      });
      
      socket.on('reset_comparison', () => {
        this.baselineSimulationData = {
          uavs: [],
          scenario: {},
          performance: {},
          timestamp: null
        };
        this.mcpSimulationData = {
          uavs: [],
          scenario: {},
          performance: {},
          timestamp: null
        };
        this.io.emit('comparison_reset');
      });
      
      socket.on('disconnect', () => {
        console.log('Client disconnected:', socket.id);
      });
    });
  }
  
  connectToMCP() {
    const mcpUrl = process.env.MCP_SERVER_URL || 'ws://localhost:8765';
    // Prevent overlapping reconnect attempts
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    
    try {
      this.mcpWebSocket = new WebSocket(mcpUrl);
      
      this.mcpWebSocket.on('open', () => {
        console.log('Connected to MCP server');
        this.mcpConnected = true;
        this.broadcastStatusUpdate();
        // Reset backoff
        this.reconnectDelayMs = 5000;
      });
      
      this.mcpWebSocket.on('message', (data) => {
        try {
          const message = JSON.parse(data);
          console.log('Received MCP message:', message.message_type);
          this.handleMCPMessage(message);
        } catch (error) {
          console.error('Error parsing MCP message:', error);
        }
      });
      
      this.mcpWebSocket.on('close', () => {
        console.log('MCP server connection closed');
        this.mcpConnected = false;
        this.broadcastStatusUpdate();
        
        // Attempt to reconnect with backoff
        this.scheduleReconnect();
      });
      
      this.mcpWebSocket.on('error', (error) => {
        const now = Date.now();
        if (now - this.lastErrorLogAt >= this.errorLogThrottleMs) {
          const msg = (error && (error.code || error.message)) ? `${error.code || ''} ${error.message || ''}`.trim() : 'Unknown error';
          console.error('MCP WebSocket error:', msg);
          this.lastErrorLogAt = now;
        }
        this.mcpConnected = false;
        this.broadcastStatusUpdate();
        this.scheduleReconnect();
      });
      
    } catch (error) {
      console.error('Failed to connect to MCP server:', error);
      this.mcpConnected = false;
      this.scheduleReconnect();
    }
  }

  scheduleReconnect() {
    if ((process.env.MCP_AUTO_CONNECT || 'true').toLowerCase() === 'false') {
      return; // Do not reconnect if disabled
    }
    if (this.reconnectTimer) return;
    const delay = Math.min(this.reconnectDelayMs, this.maxReconnectDelayMs);
    console.log(`Reconnecting to MCP in ${Math.round(delay / 1000)}s...`);
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.reconnectDelayMs = Math.min(Math.floor(this.reconnectDelayMs * 1.5), this.maxReconnectDelayMs);
      this.connectToMCP();
    }, delay);
  }
  
  handleMCPMessage(message) {
    switch (message.message_type) {
      case 'context_broadcast':
        console.log('Processing context_broadcast, data keys:', Object.keys(message.data || {}));
        this.updateSimulationData(message.data);
        this.broadcastSimulationUpdate();
        break;
      
      case 'periodic_update':
        console.log('Processing periodic_update, data keys:', Object.keys(message.data || {}));
        this.updateSimulationData(message.data);
        this.broadcastSimulationUpdate();
        break;
      
      case 'agent_update':
        this.updateAgentData(message);
        this.broadcastAgentUpdate();
        break;
      
      default:
        console.log('Unknown MCP message type:', message.message_type);
    }
  }
  
  updateSimulationData(data) {
    this.simulationData = {
      uavs: this.extractUAVData(data),
      scenario: this.extractScenarioData(data),
      performance: this.extractPerformanceData(data),
      timestamp: new Date().toISOString()
    };
    
    // If in comparison mode, determine which simulation this is
    if (this.comparisonMode) {
      // Check if this is baseline (no MCP context) or MCP (has MCP context)
      const hasMcpContext = data.context && (
        data.context.coverage_map || 
        data.context.communication_network ||
        data.sender_type === 'mcp'
      );
      
      if (hasMcpContext) {
        this.mcpSimulationData = {
          uavs: this.extractUAVData(data),
          scenario: this.extractScenarioData(data),
          performance: this.extractPerformanceData(data),
          timestamp: new Date().toISOString()
        };
        this.io.emit('mcp_simulation_update', this.mcpSimulationData);
      } else {
        this.baselineSimulationData = {
          uavs: this.extractUAVData(data),
          scenario: this.extractScenarioData(data),
          performance: this.extractPerformanceData(data),
          timestamp: new Date().toISOString()
        };
        this.io.emit('baseline_simulation_update', this.baselineSimulationData);
      }
    }
  }
  
  extractUAVData(data) {
    // If upstream already provides structured UAVs
    if (Array.isArray(data.uavs)) {
      return data.uavs;
    }

    const uavs = [];
    const batteryMap = data.battery_status || data.batteries || (data.context && data.context.battery_status) || {};

    Object.keys(batteryMap).forEach(uavId => {
      uavs.push({
        id: uavId,
        battery: batteryMap[uavId],
        position: this.getUAVPosition(uavId, data),
        status: this.getUAVStatus(uavId, { battery_status: batteryMap }),
        communication_range: 50,
        sensor_range: 30
      });
    });

    return uavs;
  }
  
  getUAVPosition(uavId, data) {
    // Extract position from context if available
    const context = data.context || data;
    const uavPositions = context.uav_positions || {};
    
    if (uavPositions[uavId]) {
      return {
        x: uavPositions[uavId].x || 0,
        y: uavPositions[uavId].y || 0,
        z: uavPositions[uavId].z || 10
      };
    }
    
    // Fallback to placeholder if no position data
    return {
      x: Math.random() * 1000,
      y: Math.random() * 1000,
      z: 10 + Math.random() * 40
    };
  }
  
  getUAVStatus(uavId, data) {
    const battery = data.battery_status?.[uavId] || 100;
    if (battery < 10) return 'low_battery';
    if (battery <= 0) return 'emergency';
    return 'active';
  }
  
  extractScenarioData(data) {
    const context = data.context || data;
    return {
      coverage_map: context.coverage_map || [],
      disaster_zones: context.disaster_zones || [],
      obstacles: context.obstacles || [],
      target_areas: context.target_areas || [],
      environmental_conditions: context.environmental_conditions || {},
      wind_conditions: context.wind_conditions || {},
      emergency_events: context.emergency_events || []
    };
  }
  
  extractPerformanceData(data) {
    const context = data.context || data;
    const coveragePct =
      typeof context.coverage_percentage === 'number'
        ? context.coverage_percentage
        : this.calculateCoveragePercentage(context.coverage_map);
    return {
      coverage_percentage: coveragePct || 0,
      communication_network: context.communication_network || {},
      target_priorities: context.target_priorities || {},
      last_updated: context.last_updated || Date.now()
    };
  }
  
  calculateCoveragePercentage(coverageMap) {
    if (!coverageMap || !Array.isArray(coverageMap)) return 0;
    
    const totalCells = coverageMap.length * coverageMap[0].length;
    const coveredCells = coverageMap.flat().filter(cell => cell > 0.5).length;
    return (coveredCells / totalCells) * 100;
  }
  
  updateAgentData(message) {
    const agentId = message.sender_id;
    this.agentsData[agentId] = {
      ...this.agentsData[agentId],
      ...message.data,
      last_update: new Date().toISOString()
    };
  }
  
  startDataCollection() {
    // Collect performance metrics every 5 seconds
    setInterval(() => {
      this.collectPerformanceMetrics();
      // Push periodic updates to clients even without new MCP messages
      this.broadcastSimulationUpdate();
      this.broadcastPerformanceData();
    }, 5000);
  }
  
  collectPerformanceMetrics() {
    const now = new Date();
    
    // Add current performance data to metrics
    if (this.simulationData.performance) {
      this.performanceMetrics.coverage.push({
        timestamp: now,
        value: this.simulationData.performance.coverage_percentage
      });
      
      // Keep only last 100 data points
      if (this.performanceMetrics.coverage.length > 100) {
        this.performanceMetrics.coverage.shift();
      }
    }
    
    // Add UAV battery data
    if (this.simulationData.uavs.length > 0) {
      const avgBattery = this.simulationData.uavs.reduce((sum, uav) => sum + uav.battery, 0) / this.simulationData.uavs.length;
      this.performanceMetrics.battery.push({
        timestamp: now,
        value: avgBattery
      });
      
      if (this.performanceMetrics.battery.length > 100) {
        this.performanceMetrics.battery.shift();
      }
    }
  }
  
  broadcastStatusUpdate() {
    this.io.emit('status_update', {
      mcp_connected: this.mcpConnected,
      timestamp: new Date().toISOString()
    });
  }
  
  broadcastSimulationUpdate() {
    const uavCount = (this.simulationData.uavs || []).length;
    console.log(`Broadcasting simulation_update: ${uavCount} UAVs, coverage: ${this.simulationData.performance?.coverage_percentage || 0}%`);
    this.io.emit('simulation_update', this.simulationData);
  }
  
  broadcastAgentUpdate() {
    this.io.emit('agent_update', this.agentsData);
  }
  
  broadcastPerformanceData() {
    this.io.emit('performance_data', this.performanceMetrics);
  }
  
  sendSimulationData(socket) {
    socket.emit('simulation_data', this.simulationData);
  }
  
  sendAgentData(socket) {
    socket.emit('agent_data', this.agentsData);
  }
  
  sendPerformanceData(socket) {
    socket.emit('performance_data', this.performanceMetrics);
  }
  
  // Public methods for API endpoints
  isMCPConnected() {
    return this.mcpConnected;
  }
  
  getSimulationStatus() {
    return {
      mcp_connected: this.mcpConnected,
      simulation_running: this.simulationData.timestamp !== null,
      num_uavs: this.simulationData.uavs.length,
      last_update: this.simulationData.timestamp
    };
  }
  
  getAgentsInfo() {
    return Object.values(this.agentsData);
  }
  
  getPerformanceMetrics() {
    return this.performanceMetrics;
  }
}

module.exports = WebSocketHandler;
