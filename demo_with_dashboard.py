"""Easy demo script to run MCP server, dashboard, and simulation together."""

import asyncio
import subprocess
import time
import signal
import sys
import numpy as np
from pathlib import Path

def check_port(port):
    """Check if a port is available."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    return result != 0

def start_mcp_server():
    """Start MCP server in background."""
    print("Starting MCP server...")
    return subprocess.Popen(
        [sys.executable, "-m", "mcp_server.server"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

def start_dashboard():
    """Start web dashboard in background."""
    print("Starting web dashboard...")
    dashboard_dir = Path("web_dashboard")
    if not dashboard_dir.exists():
        print("ERROR: web_dashboard directory not found!")
        return None
    
    return subprocess.Popen(
        ["npm", "start"],
        cwd=dashboard_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

async def run_simulation_with_agents():
    """Run simulation with multiple agents."""
    from config.simulation_config import SimulationConfig
    from simulation.environment import SwarmEnvironment
    from rl_agents.context_aware_agent import ContextAwareAgent
    from rl_agents.ppo_agent import PPOAgent
    import torch
    
    # Set torch to evaluation mode for inference
    torch.set_grad_enabled(False)
    
    print("Starting simulation with agents...")
    
    # Load configuration
    config = SimulationConfig()
    config.simulation_time = 60  # 60 seconds for demo
    config.num_uavs = 5
    
    # Create environment
    env = SwarmEnvironment(config)
    
    # Start MCP connection
    print("Connecting to MCP server...")
    await env.start_mcp_connection()
    
    # Wait for connection to stabilize and register
    await asyncio.sleep(2)
    
    if not env.mcp_connected:
        print("WARNING: MCP connection failed. Continuing without MCP...")
    else:
        print(f"✓ Connected to MCP server with {config.num_uavs} UAVs")
    
    # Create agents (try to load trained models, fallback to random)
    from rl_agents.base_agent import BaseAgent
    
    agents = []
    state_dim = env.observation_space.shape[0] // config.num_uavs
    action_dim = env.action_space.shape[0] // config.num_uavs
    context_dim = 20  # Context dimension for ContextAwareAgent
    agent_config = config.rl_config.__dict__ if hasattr(config, 'rl_config') else {}
    
    for i in range(config.num_uavs):
        try:
            # Try to load trained model
            model_path = f"saved_models/agent_{i}_episode_100.pth"
            if Path(model_path).exists():
                agent = ContextAwareAgent(
                    agent_id=str(i),
                    state_dim=state_dim,
                    action_dim=action_dim,
                    context_dim=context_dim,
                    config=agent_config
                )
                # Load the saved model using the agent's load method
                agent.load(model_path)
                agent.set_training_mode(False)  # Set to eval mode for inference
                print(f"Loaded trained model for agent {i}")
            else:
                # Use random agent for demo
                agent = PPOAgent(
                    agent_id=str(i),
                    state_dim=state_dim,
                    action_dim=action_dim,
                    config=agent_config
                )
                print(f"Using random agent {i} (no trained model found)")
        except Exception as e:
            print(f"Error loading agent {i}: {e}, using random agent")
            agent = PPOAgent(
                agent_id=str(i),
                state_dim=state_dim,
                action_dim=action_dim,
                config=agent_config
            )
        agents.append(agent)
    
    print(f"Running simulation with {len(agents)} agents for {config.simulation_time} seconds...")
    print("Open http://localhost:3001 in your browser to see the dashboard!")
    
    try:
        observations, info = env.reset()
        start_time = time.time()
        step_count = 0
        episode_count = 0
        
        while time.time() - start_time < config.simulation_time:
            # Get actions from agents
            actions_list = []
            for i, agent in enumerate(agents):
                # Extract observation for this agent
                obs_dim = env.observation_space.shape[0] // config.num_uavs
                agent_obs = observations[i * obs_dim:(i + 1) * obs_dim]
                action = agent.select_action(agent_obs, deterministic=False)
                actions_list.append(action)
            
            # Combine actions
            actions = np.concatenate(actions_list)
            
            # Step environment
            observations, reward, terminated, truncated, info = env.step(actions)
            
            # Log progress
            if step_count % 50 == 0:
                coverage = info['performance_metrics']['coverage_percentage'][-1]
                print(f"Step {step_count}, Coverage: {coverage:.1f}%, Reward: {reward:.3f}")
            
            # Reset if episode ends
            if terminated or truncated:
                episode_count += 1
                print(f"Episode {episode_count} ended. Resetting...")
                observations, info = env.reset()
                step_count = 0
                continue
            
            step_count += 1
            
            # Allow event loop to process async tasks (MCP updates)
            # This is critical for MCP communication
            # Give the event loop time to process the async tasks created in env.step()
            await asyncio.sleep(0.1)  # 100ms delay to ensure async tasks execute
    
    except KeyboardInterrupt:
        print("\nSimulation interrupted")
    finally:
        env.close()
        print("Simulation completed")

def main():
    """Main demo function."""
    print("=" * 60)
    print("MCP-Coordinated Swarm Intelligence - Demo")
    print("=" * 60)
    
    # Check ports
    if not check_port(8765):
        print("ERROR: Port 8765 (MCP server) is already in use!")
        print("Please stop the existing MCP server first.")
        return
    
    if not check_port(3001):
        print("ERROR: Port 3001 (Dashboard) is already in use!")
        print("Please stop the existing dashboard first.")
        return
    
    processes = []
    
    try:
        # Start MCP server
        mcp_process = start_mcp_server()
        processes.append(mcp_process)
        print("Waiting for MCP server to start...")
        time.sleep(3)  # Wait for MCP server to start
        
        # Verify MCP server is running
        if not check_port(8765):
            print("✓ MCP server is running on port 8765")
        else:
            print("⚠ WARNING: MCP server may not have started properly")
        
        # Start dashboard
        dashboard_process = start_dashboard()
        if dashboard_process:
            processes.append(dashboard_process)
        print("Waiting for dashboard to start...")
        time.sleep(5)  # Wait for dashboard to start
        print("✓ Dashboard should be available at http://localhost:3001")
        
        # Run simulation
        print("\n" + "=" * 60)
        print("Starting simulation...")
        print("=" * 60 + "\n")
        
        asyncio.run(run_simulation_with_agents())
        
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        # Cleanup
        for process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()
        print("Demo completed")

if __name__ == "__main__":
    main()

