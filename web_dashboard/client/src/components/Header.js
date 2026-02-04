import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import styled from 'styled-components';

const HeaderContainer = styled.header`
  background-color: ${props => props.theme.colors.surface};
  border-bottom: 1px solid ${props => props.theme.colors.border};
  box-shadow: ${props => props.theme.shadows.sm};
`;

const HeaderContent = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: ${props => props.theme.spacing.md} ${props => props.theme.spacing.lg};
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const Logo = styled.div`
  font-size: 1.5rem;
  font-weight: bold;
  color: ${props => props.theme.colors.primary};
`;

const Nav = styled.nav`
  display: flex;
  gap: ${props => props.theme.spacing.lg};
`;

const NavLink = styled(Link)`
  text-decoration: none;
  color: ${props => props.theme.colors.textSecondary};
  font-weight: 500;
  padding: ${props => props.theme.spacing.sm} ${props => props.theme.spacing.md};
  border-radius: ${props => props.theme.borderRadius};
  transition: all 0.2s ease;

  &:hover {
    color: ${props => props.theme.colors.primary};
    background-color: ${props => props.theme.colors.background};
  }

  &.active {
    color: ${props => props.theme.colors.primary};
    background-color: ${props => props.theme.colors.background};
  }
`;

const StatusIndicator = styled.div`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
  font-size: 0.875rem;
`;

const StatusDot = styled.div`
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background-color: ${props => 
    props.status === 'connected' ? props.theme.colors.success : props.theme.colors.error
  };
`;

function Header({ connectionStatus }) {
  const location = useLocation();

  return (
    <HeaderContainer>
      <HeaderContent>
        <Logo>MCP Swarm Intelligence</Logo>
        <Nav>
          <NavLink 
            to="/" 
            className={location.pathname === '/' ? 'active' : ''}
          >
            Dashboard
          </NavLink>
          <NavLink 
            to="/simulation" 
            className={location.pathname === '/simulation' ? 'active' : ''}
          >
            Simulation
          </NavLink>
          <NavLink 
            to="/performance" 
            className={location.pathname === '/performance' ? 'active' : ''}
          >
            Performance
          </NavLink>
          <NavLink 
            to="/agents" 
            className={location.pathname === '/agents' ? 'active' : ''}
          >
            Agents
          </NavLink>
          <NavLink 
            to="/comparison" 
            className={location.pathname === '/comparison' ? 'active' : ''}
          >
            Comparison
          </NavLink>
        </Nav>
        <StatusIndicator>
          <StatusDot status={connectionStatus} />
          <span>
            {connectionStatus === 'connected' ? 'Connected' : 'Disconnected'}
          </span>
        </StatusIndicator>
      </HeaderContent>
    </HeaderContainer>
  );
}

export default Header;
