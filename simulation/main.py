"""Main simulation entry point for MCP-Coordinated Swarm Intelligence."""

import asyncio
import argparse
import time
from loguru import logger

from config.simulation_config import SimulationConfig
from .environment import SwarmEnvironment


async def run_simulation(config_path: str = None, headless: bool = False):
    """Run the main simulation."""
    
    # Load configuration
    if config_path:
        config = SimulationConfig.from_yaml(config_path)
    else:
        config = SimulationConfig()
    
    # Override render setting for headless mode
    if headless:
        config.render = False
    
    # Create environment
    env = SwarmEnvironment(config)
    
    # Start MCP connection
    await env.start_mcp_connection()
    
    logger.info("Starting UAV swarm simulation")
    logger.info(f"Configuration: {config.num_uavs} UAVs, {config.simulation_time}s duration")
    
    try:
        # Reset environment
        observations, info = env.reset()
        
        # Run simulation
        start_time = time.time()
        step_count = 0
        episode_count = 0
        
        while time.time() - start_time < config.simulation_time:
            # Generate random actions for demonstration
            # In practice, these would come from RL agents
            actions = env.action_space.sample()
            
            # Step environment
            observations, reward, terminated, truncated, info = env.step(actions)
            
            # Render if enabled
            if config.render:
                if not env.render():
                    break  # User closed window
            
            # Log progress
            if step_count % 100 == 0:
                logger.info(f"Episode {episode_count}, Step {step_count}, Reward: {reward:.3f}, "
                          f"Coverage: {info['performance_metrics']['coverage_percentage'][-1]:.1f}%")
            
            # Check termination - reset and continue instead of breaking
            if terminated or truncated:
                episode_count += 1
                logger.info(f"Episode {episode_count-1} terminated. Resetting and continuing...")
                observations, info = env.reset()
                step_count = 0
                continue
            
            step_count += 1
            
            # Small delay to allow async tasks (MCP updates) to execute
            # This is critical for MCP communication to work
            await asyncio.sleep(0.01)  # 10ms delay allows event loop to process tasks
            
            # Additional delay for visualization
            if config.render:
                await asyncio.sleep(1.0 / config.render_fps)
    
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    
    except Exception as e:
        logger.error(f"Simulation error: {e}")
    
    finally:
        # Cleanup
        env.close()
        logger.info("Simulation completed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MCP-Coordinated Swarm Intelligence Simulation")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    
    args = parser.parse_args()
    
    # Setup logging
    logger.add("logs/simulation.log", rotation="1 day", retention="7 days")
    
    # Create logs directory
    import os
    os.makedirs("logs", exist_ok=True)
    
    # Run simulation
    asyncio.run(run_simulation(args.config, args.headless))


if __name__ == "__main__":
    main()
