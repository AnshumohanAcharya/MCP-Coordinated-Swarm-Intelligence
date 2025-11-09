import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { useSocket } from '../services/SocketService';
import MetricCard from './MetricCard';

const ComparisonContainer = styled.div`
  max-width: 1600px;
  margin: 0 auto;
`;

const Title = styled.h1`
  font-size: 2rem;
  font-weight: bold;
  color: ${props => props.theme.colors.text};
  margin-bottom: ${props => props.theme.spacing.xl};
  text-align: center;
`;

const ComparisonGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: ${props => props.theme.spacing.lg};
  margin-bottom: ${props => props.theme.spacing.xl};

  @media (max-width: 1200px) {
    grid-template-columns: 1fr;
  }
`;

const SwarmPanel = styled.div`
  background-color: ${props => props.theme.colors.surface};
  border-radius: ${props => props.theme.borderRadius};
  padding: ${props => props.theme.spacing.lg};
  box-shadow: ${props => props.theme.shadows.md};
  border: 2px solid ${props => props.isMCP ? '#10b981' : '#f59e0b'};
`;

const PanelHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: ${props => props.theme.spacing.md};
  padding-bottom: ${props => props.theme.spacing.md};
  border-bottom: 2px solid ${props => props.theme.colors.border};
`;

const PanelTitle = styled.h2`
  font-size: 1.5rem;
  font-weight: 600;
  color: ${props => props.theme.colors.text};
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
`;

const Badge = styled.span`
  padding: ${props => props.theme.spacing.xs} ${props => props.theme.spacing.sm};
  border-radius: ${props => props.theme.borderRadius};
  font-size: 0.75rem;
  font-weight: 600;
  background-color: ${props => props.isMCP ? '#10b981' : '#f59e0b'};
  color: white;
`;

const CoverageDisplay = styled.div`
  text-align: center;
  padding: ${props => props.theme.spacing.lg};
  background: linear-gradient(135deg, ${props => props.isMCP ? '#10b981' : '#f59e0b'}15, ${props => props.isMCP ? '#10b981' : '#f59e0b'}05);
  border-radius: ${props => props.theme.borderRadius};
  margin-bottom: ${props => props.theme.spacing.md};
`;

const CoverageValue = styled.div`
  font-size: 3rem;
  font-weight: bold;
  color: ${props => props.isMCP ? '#10b981' : '#f59e0b'};
  margin-bottom: ${props => props.theme.spacing.xs};
`;

const CoverageLabel = styled.div`
  font-size: 1rem;
  color: ${props => props.theme.colors.textSecondary};
  font-weight: 500;
`;

const SimulationCanvas = styled.div`
  background-color: ${props => props.theme.colors.background};
  border-radius: ${props => props.theme.borderRadius};
  padding: ${props => props.theme.spacing.md};
  margin-bottom: ${props => props.theme.spacing.md};
  height: 400px;
  position: relative;
  overflow: hidden;
  border: 1px solid ${props => props.theme.colors.border};
`;

const UAV = styled.div`
  position: absolute;
  width: 14px;
  height: 14px;
  border-radius: 50%;
  background-color: ${props => props.color || '#2563eb'};
  border: 2px solid white;
  transform: translate(-50%, -50%);
  transition: all 0.3s ease;
  cursor: pointer;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
  
  &:hover {
    transform: translate(-50%, -50%) scale(1.5);
    z-index: 10;
  }
`;

const CoverageOverlay = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  pointer-events: none;
  opacity: 0.3;
`;

const CoverageCell = styled.div`
  position: absolute;
  background-color: ${props => props.covered ? '#10b981' : 'transparent'};
  border: ${props => props.covered ? '1px solid #10b98140' : 'none'};
`;

const MetricsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: ${props => props.theme.spacing.md};
  margin-top: ${props => props.theme.spacing.md};
`;

const ComparisonSummary = styled.div`
  background-color: ${props => props.theme.colors.surface};
  border-radius: ${props => props.theme.borderRadius};
  padding: ${props => props.theme.spacing.lg};
  box-shadow: ${props => props.theme.shadows.md};
  margin-top: ${props => props.theme.spacing.xl};
`;

const SummaryTitle = styled.h3`
  font-size: 1.25rem;
  font-weight: 600;
  color: ${props => props.theme.colors.text};
  margin-bottom: ${props => props.theme.spacing.md};
`;

const SummaryGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: ${props => props.theme.spacing.md};
`;

const SummaryItem = styled.div`
  padding: ${props => props.theme.spacing.md};
  background-color: ${props => props.theme.colors.background};
  border-radius: ${props => props.theme.borderRadius};
`;

const SummaryLabel = styled.div`
  font-size: 0.875rem;
  color: ${props => props.theme.colors.textSecondary};
  margin-bottom: ${props => props.theme.spacing.xs};
`;

const SummaryValue = styled.div`
  font-size: 1.5rem;
  font-weight: bold;
  color: ${props => props.improvement > 0 ? '#10b981' : props.improvement < 0 ? '#ef4444' : props.theme.colors.text};
`;

const Controls = styled.div`
  display: flex;
  justify-content: center;
  gap: ${props => props.theme.spacing.md};
  margin-bottom: ${props => props.theme.spacing.lg};
`;

const Button = styled.button`
  padding: ${props => props.theme.spacing.sm} ${props => props.theme.spacing.lg};
  border: none;
  border-radius: ${props => props.theme.borderRadius};
  background-color: ${props => props.theme.colors.primary};
  color: white;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover {
    background-color: ${props => props.theme.colors.primary}dd;
    transform: translateY(-2px);
  }

  &:disabled {
    background-color: ${props => props.theme.colors.secondary};
    cursor: not-allowed;
    transform: none;
  }
`;

function ComparisonView() {
  const socket = useSocket();
  const [baselineData, setBaselineData] = useState({
    uavs: [],
    coverage: 0,
    battery: 0,
    efficiency: 0,
    timestamp: null
  });
  const [mcpData, setMcpData] = useState({
    uavs: [],
    coverage: 0,
    battery: 0,
    efficiency: 0,
    timestamp: null
  });
  const [comparisonRunning, setComparisonRunning] = useState(false);

  useEffect(() => {
    if (!socket) return;

    const handleBaselineUpdate = (data) => {
      setBaselineData({
        uavs: data.uavs || [],
        coverage: data.performance?.coverage_percentage || 0,
        battery: calculateAvgBattery(data.uavs || []),
        efficiency: calculateEfficiency(data),
        timestamp: data.timestamp
      });
    };

    const handleMcpUpdate = (data) => {
      setMcpData({
        uavs: data.uavs || [],
        coverage: data.performance?.coverage_percentage || 0,
        battery: calculateAvgBattery(data.uavs || []),
        efficiency: calculateEfficiency(data),
        timestamp: data.timestamp
      });
    };

    socket.on('baseline_simulation_update', handleBaselineUpdate);
    socket.on('mcp_simulation_update', handleMcpUpdate);

    socket.emit('request_comparison_data');

    return () => {
      socket.off('baseline_simulation_update', handleBaselineUpdate);
      socket.off('mcp_simulation_update', handleMcpUpdate);
    };
  }, [socket]);

  const calculateAvgBattery = (uavs) => {
    if (!uavs || uavs.length === 0) return 0;
    return uavs.reduce((sum, uav) => sum + (uav.battery || 0), 0) / uavs.length;
  };

  const calculateEfficiency = (data) => {
    // Calculate efficiency based on coverage per battery used
    const coverage = data.performance?.coverage_percentage || 0;
    const avgBattery = calculateAvgBattery(data.uavs || []);
    if (avgBattery === 0) return 0;
    return (coverage / (100 - avgBattery + 1)) * 100; // Coverage per battery consumed
  };

  const startComparison = () => {
    if (socket) {
      socket.emit('start_comparison');
      setComparisonRunning(true);
    }
  };

  const stopComparison = () => {
    if (socket) {
      socket.emit('stop_comparison');
      setComparisonRunning(false);
    }
  };

  const resetComparison = () => {
    if (socket) {
      socket.emit('reset_comparison');
      setBaselineData({
        uavs: [],
        coverage: 0,
        battery: 0,
        efficiency: 0,
        timestamp: null
      });
      setMcpData({
        uavs: [],
        coverage: 0,
        battery: 0,
        efficiency: 0,
        timestamp: null
      });
    }
  };

  const calculateImprovement = (baseline, mcp) => {
    if (baseline === 0) return mcp > 0 ? 100 : 0;
    return ((mcp - baseline) / baseline) * 100;
  };

  const coverageImprovement = calculateImprovement(baselineData.coverage, mcpData.coverage);
  const batteryImprovement = calculateImprovement(baselineData.battery, mcpData.battery);
  const efficiencyImprovement = calculateImprovement(baselineData.efficiency, mcpData.efficiency);

  return (
    <ComparisonContainer>
      <Title>Baseline vs MCP-Coordinated Swarm Comparison</Title>
      
      <Controls>
        <Button onClick={startComparison} disabled={comparisonRunning}>
          Start Comparison
        </Button>
        <Button onClick={stopComparison} disabled={!comparisonRunning}>
          Stop Comparison
        </Button>
        <Button onClick={resetComparison}>
          Reset
        </Button>
      </Controls>

      <ComparisonGrid>
        {/* Baseline Panel */}
        <SwarmPanel isMCP={false}>
          <PanelHeader>
            <PanelTitle>
              Baseline Swarm
              <Badge isMCP={false}>No MCP</Badge>
            </PanelTitle>
          </PanelHeader>
          
          <CoverageDisplay isMCP={false}>
            <CoverageValue isMCP={false}>
              {baselineData.coverage.toFixed(1)}%
            </CoverageValue>
            <CoverageLabel>Coverage</CoverageLabel>
          </CoverageDisplay>

          <SimulationCanvas>
            <CoverageOverlay>
              {/* Coverage visualization would go here */}
            </CoverageOverlay>
            {baselineData.uavs.map((uav, index) => (
              <UAV
                key={uav.id || index}
                color="#f59e0b"
                style={{
                  left: `${((uav.position?.x || 0) / 1000) * 100}%`,
                  top: `${((uav.position?.y || 0) / 1000) * 100}%`
                }}
                title={`${uav.id}: Battery ${uav.battery?.toFixed(1)}%`}
              />
            ))}
          </SimulationCanvas>

          <MetricsGrid>
            <MetricCard
              title="Average Battery"
              value={`${baselineData.battery.toFixed(1)}%`}
              icon="🔋"
              color="warning"
            />
            <MetricCard
              title="Efficiency"
              value={`${baselineData.efficiency.toFixed(1)}`}
              icon="⚡"
              color="warning"
            />
            <MetricCard
              title="Active UAVs"
              value={baselineData.uavs.filter(u => u.status === 'active').length}
              icon="🚁"
              color="primary"
            />
            <MetricCard
              title="Total UAVs"
              value={baselineData.uavs.length}
              icon="📊"
              color="primary"
            />
          </MetricsGrid>
        </SwarmPanel>

        {/* MCP Panel */}
        <SwarmPanel isMCP={true}>
          <PanelHeader>
            <PanelTitle>
              MCP-Coordinated Swarm
              <Badge isMCP={true}>With MCP</Badge>
            </PanelTitle>
          </PanelHeader>
          
          <CoverageDisplay isMCP={true}>
            <CoverageValue isMCP={true}>
              {mcpData.coverage.toFixed(1)}%
            </CoverageValue>
            <CoverageLabel>Coverage</CoverageLabel>
          </CoverageDisplay>

          <SimulationCanvas>
            <CoverageOverlay>
              {/* Coverage visualization would go here */}
            </CoverageOverlay>
            {mcpData.uavs.map((uav, index) => (
              <UAV
                key={uav.id || index}
                color="#10b981"
                style={{
                  left: `${((uav.position?.x || 0) / 1000) * 100}%`,
                  top: `${((uav.position?.y || 0) / 1000) * 100}%`
                }}
                title={`${uav.id}: Battery ${uav.battery?.toFixed(1)}%`}
              />
            ))}
          </SimulationCanvas>

          <MetricsGrid>
            <MetricCard
              title="Average Battery"
              value={`${mcpData.battery.toFixed(1)}%`}
              icon="🔋"
              color="success"
            />
            <MetricCard
              title="Efficiency"
              value={`${mcpData.efficiency.toFixed(1)}`}
              icon="⚡"
              color="success"
            />
            <MetricCard
              title="Active UAVs"
              value={mcpData.uavs.filter(u => u.status === 'active').length}
              icon="🚁"
              color="primary"
            />
            <MetricCard
              title="Total UAVs"
              value={mcpData.uavs.length}
              icon="📊"
              color="primary"
            />
          </MetricsGrid>
        </SwarmPanel>
      </ComparisonGrid>

      <ComparisonSummary>
        <SummaryTitle>Performance Comparison</SummaryTitle>
        <SummaryGrid>
          <SummaryItem>
            <SummaryLabel>Coverage Improvement</SummaryLabel>
            <SummaryValue improvement={coverageImprovement}>
              {coverageImprovement > 0 ? '+' : ''}{coverageImprovement.toFixed(1)}%
            </SummaryValue>
          </SummaryItem>
          <SummaryItem>
            <SummaryLabel>Battery Efficiency</SummaryLabel>
            <SummaryValue improvement={batteryImprovement}>
              {batteryImprovement > 0 ? '+' : ''}{batteryImprovement.toFixed(1)}%
            </SummaryValue>
          </SummaryItem>
          <SummaryItem>
            <SummaryLabel>Overall Efficiency</SummaryLabel>
            <SummaryValue improvement={efficiencyImprovement}>
              {efficiencyImprovement > 0 ? '+' : ''}{efficiencyImprovement.toFixed(1)}%
            </SummaryValue>
          </SummaryItem>
          <SummaryItem>
            <SummaryLabel>Time to Target Coverage</SummaryLabel>
            <SummaryValue improvement={-coverageImprovement}>
              {coverageImprovement > 0 ? 'Faster' : coverageImprovement < 0 ? 'Slower' : 'Equal'}
            </SummaryValue>
          </SummaryItem>
        </SummaryGrid>
      </ComparisonSummary>
    </ComparisonContainer>
  );
}

export default ComparisonView;

