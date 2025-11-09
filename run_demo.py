#!/usr/bin/env python3
"""Demo script for MCP-Coordinated Swarm Intelligence."""

import asyncio
import subprocess
import time
import signal
import sys
import os
import socket
import requests
from pathlib import Path
from loguru import logger


class SwarmDemo:
    """Demo coordinator for the MCP-Coordinated Swarm Intelligence system."""
    
    def __init__(self):
        self.processes = []
        self.running = False
        self.health_check_interval = 5.0  # seconds
        self.max_startup_time = 30.0  # seconds
    
    def check_port_available(self, port: int) -> bool:
        """Check if a port is available."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            result = sock.connect_ex(('localhost', port))
            sock.close()
            return result != 0  # Port is available if connection fails
        except Exception:
            return False
    
    def wait_for_service(self, url: str, timeout: float = 10.0, name: str = "Service") -> bool:
        """Wait for a service to become available."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=1.0)
                if response.status_code == 200:
                    logger.info(f"{name} is ready")
                    return True
            except Exception:
                pass
            time.sleep(0.5)
        logger.warning(f"{name} did not become ready within {timeout}s")
        return False
    
    def start_mcp_server(self):
        """Start the MCP server with health checks."""
        logger.info("Starting MCP server...")
        
        # Check if port is available
        if not self.check_port_available(8765):
            logger.warning("Port 8765 is already in use. MCP server may already be running.")
        
        process = subprocess.Popen([
            sys.executable, "-m", "mcp_server.server"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.processes.append(("MCP Server", process))
        
        # Wait for server to start
        time.sleep(2)
        
        # Health check - try to connect to WebSocket port
        # (Simplified check - in production, use proper WebSocket health check)
        if process.poll() is None:
            logger.info("MCP server process started")
        else:
            logger.error("MCP server process failed to start")
            try:
                stderr = process.stderr.read().decode('utf-8')
                if stderr:
                    logger.error(f"MCP server error: {stderr}")
            except:
                pass
    
    def start_web_dashboard(self):
        """Start the web dashboard with health checks."""
        logger.info("Starting web dashboard...")
        
        # Check if port is available
        if not self.check_port_available(3001):
            logger.warning("Port 3001 is already in use. Dashboard may already be running.")
        
        dashboard_dir = Path("web_dashboard")
        if not dashboard_dir.exists():
            logger.error("web_dashboard directory not found!")
            return
        
        original_dir = os.getcwd()
        try:
            os.chdir(dashboard_dir)
            process = subprocess.Popen([
                "npm", "start"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.processes.append(("Web Dashboard", process))
        finally:
            os.chdir(original_dir)
        
        # Wait for dashboard to start
        time.sleep(5)
        
        # Health check
        if process.poll() is None:
            logger.info("Web dashboard process started")
            # Try to check if server is responding
            if self.wait_for_service("http://localhost:3001/api/health", timeout=10.0, name="Web Dashboard"):
                logger.info("Web dashboard is ready at http://localhost:3001")
        else:
            logger.error("Web dashboard process failed to start")
            try:
                stderr = process.stderr.read().decode('utf-8')
                if stderr:
                    logger.error(f"Dashboard error: {stderr[:500]}")  # Limit output
            except:
                pass
    
    def start_simulation(self, headless=False):
        """Start the simulation."""
        logger.info("Starting simulation...")
        cmd = [sys.executable, "-m", "simulation.main"]
        if headless:
            cmd.append("--headless")
        
        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.processes.append(("Simulation", process))
    
    def start_training(self, episodes=100, use_context=True):
        """Start training."""
        logger.info(f"Starting training with {episodes} episodes...")
        cmd = [sys.executable, "-m", "rl_agents.train", "--episodes", str(episodes)]
        if not use_context:
            cmd.append("--no-context")
        
        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.processes.append(("Training", process))
    
    def run_baseline_comparison(self, episodes=50):
        """Run baseline comparison experiment."""
        logger.info(f"Running baseline comparison with {episodes} episodes...")
        process = subprocess.Popen([
            sys.executable, "experiments/baseline_comparison.py"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.processes.append(("Baseline Comparison", process))
    
    def stop_all_processes(self):
        """Stop all running processes."""
        logger.info("Stopping all processes...")
        for name, process in self.processes:
            logger.info(f"Stopping {name}...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        self.processes.clear()
    
    def check_processes(self):
        """Check if all processes are still running."""
        active_processes = []
        for name, process in self.processes:
            if process.poll() is None:
                active_processes.append((name, process))
            else:
                # Process has stopped, check for errors
                return_code = process.returncode
                if return_code != 0:
                    try:
                        stderr_output = process.stderr.read().decode('utf-8')
                        if stderr_output:
                            logger.error(f"{name} failed with return code {return_code}: {stderr_output[:500]}")
                        else:
                            logger.error(f"{name} failed with return code {return_code}")
                    except:
                        logger.error(f"{name} failed with return code {return_code}")
                else:
                    logger.info(f"{name} completed successfully")
        
        self.processes = active_processes
        return len(self.processes) > 0
    
    def preflight_checks(self) -> bool:
        """Run pre-flight checks before starting demo."""
        logger.info("Running pre-flight checks...")
        checks_passed = True
        
        # Check Python version
        if sys.version_info < (3, 8):
            logger.error(f"Python 3.8+ required, found {sys.version}")
            checks_passed = False
        
        # Check required directories
        required_dirs = ["simulation", "mcp_server", "rl_agents", "config"]
        for dir_name in required_dirs:
            if not Path(dir_name).exists():
                logger.error(f"Required directory not found: {dir_name}")
                checks_passed = False
        
        # Check if CSV file exists (optional but recommended)
        csv_file = Path("Visakhapatnam_UTide_full2024_hourly_IST.csv")
        if not csv_file.exists():
            logger.warning(f"Tidal data file not found: {csv_file}. Tidal effects will be disabled.")
        else:
            logger.info("Tidal data file found")
        
        # Check ports
        if not self.check_port_available(8765):
            logger.warning("Port 8765 (MCP server) is in use")
        
        if not self.check_port_available(3001):
            logger.warning("Port 3001 (Web dashboard) is in use")
        
        if checks_passed:
            logger.info("Pre-flight checks passed")
        else:
            logger.error("Pre-flight checks failed")
        
        return checks_passed
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Received shutdown signal")
        self.running = False
        self.stop_all_processes()
        sys.exit(0)
    
    async def run_demo(self, demo_type="simulation", **kwargs):
        """Run the specified demo."""
        # Run pre-flight checks
        if not self.preflight_checks():
            logger.error("Pre-flight checks failed. Please fix issues before running demo.")
            return
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.running = True
        
        try:
            if demo_type == "simulation":
                await self._run_simulation_demo(**kwargs)
            elif demo_type == "training":
                await self._run_training_demo(**kwargs)
            elif demo_type == "comparison":
                await self._run_comparison_demo(**kwargs)
            elif demo_type == "full":
                await self._run_full_demo(**kwargs)
            else:
                logger.error(f"Unknown demo type: {demo_type}")
                return
            
        except KeyboardInterrupt:
            logger.info("Demo interrupted by user")
        except Exception as e:
            logger.error(f"Demo error: {e}")
        finally:
            self.stop_all_processes()
    
    async def _run_simulation_demo(self, headless=False, duration=60, **kwargs):
        """Run simulation demo."""
        logger.info("=== SIMULATION DEMO ===")
        
        # Start MCP server
        self.start_mcp_server()
        
        # Start web dashboard (if not headless)
        if not headless:
            self.start_web_dashboard()
        
        # Start simulation
        self.start_simulation(headless=headless)
        
        # Run for specified duration
        logger.info(f"Running simulation for {duration} seconds...")
        logger.info("Press Ctrl+C to stop early")
        
        start_time = time.time()
        while self.running and time.time() - start_time < duration:
            if not self.check_processes():
                logger.error("Some processes stopped unexpectedly")
                break
            await asyncio.sleep(1)
        
        logger.info("Simulation demo completed")
    
    async def _run_training_demo(self, episodes=100, use_context=True, **kwargs):
        """Run training demo."""
        logger.info("=== TRAINING DEMO ===")
        
        # Start MCP server
        self.start_mcp_server()
        
        # Start training
        self.start_training(episodes=episodes, use_context=use_context)
        
        # Monitor training
        logger.info(f"Running training for {episodes} episodes...")
        logger.info("Press Ctrl+C to stop early")
        
        while self.running:
            if not self.check_processes():
                logger.info("Training completed")
                break
            await asyncio.sleep(5)
        
        logger.info("Training demo completed")
    
    async def _run_comparison_demo(self, episodes=50, **kwargs):
        """Run baseline comparison demo."""
        logger.info("=== BASELINE COMPARISON DEMO ===")
        
        # Start MCP server
        self.start_mcp_server()
        
        # Run comparison
        self.run_baseline_comparison(episodes=episodes)
        
        # Monitor comparison
        logger.info(f"Running baseline comparison for {episodes} episodes...")
        logger.info("Press Ctrl+C to stop early")
        
        while self.running:
            if not self.check_processes():
                logger.info("Comparison completed")
                break
            await asyncio.sleep(5)
        
        logger.info("Comparison demo completed")
    
    async def _run_full_demo(self, duration=120, **kwargs):
        """Run full demo with all components."""
        logger.info("=== FULL DEMO ===")
        
        # Start MCP server
        self.start_mcp_server()
        
        # Start web dashboard
        self.start_web_dashboard()
        
        # Start simulation
        self.start_simulation(headless=False)
        
        # Run for specified duration
        logger.info(f"Running full demo for {duration} seconds...")
        logger.info("Access the web dashboard at http://localhost:3000")
        logger.info("Press Ctrl+C to stop early")
        
        start_time = time.time()
        while self.running and time.time() - start_time < duration:
            if not self.check_processes():
                logger.error("Some processes stopped unexpectedly")
                break
            await asyncio.sleep(1)
        
        logger.info("Full demo completed")


def main():
    """Main function for the demo script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MCP-Coordinated Swarm Intelligence Demo")
    parser.add_argument("--demo", choices=["simulation", "training", "comparison", "full"], 
                       default="simulation", help="Type of demo to run")
    parser.add_argument("--headless", action="store_true", help="Run simulation in headless mode")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--duration", type=int, default=60, help="Demo duration in seconds")
    parser.add_argument("--no-context", action="store_true", help="Train without MCP context")
    
    args = parser.parse_args()
    
    # Setup logging
    logger.add("logs/demo.log", rotation="1 day", retention="7 days")
    os.makedirs("logs", exist_ok=True)
    
    # Create and run demo
    demo = SwarmDemo()
    
    # Prepare demo arguments
    demo_kwargs = {
        "headless": args.headless,
        "episodes": args.episodes,
        "duration": args.duration,
        "use_context": not args.no_context
    }
    
    # Run demo
    asyncio.run(demo.run_demo(args.demo, **demo_kwargs))


if __name__ == "__main__":
    main()
