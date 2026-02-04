"""
Test installation and setup for Review III components.

This script verifies that all dependencies are correctly installed
and the system is ready for Review III demonstrations.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_imports():
    """Test basic Python imports."""
    print("Testing basic imports...")
    try:
        import numpy as np
        print("  ✓ NumPy", np.__version__)
    except ImportError as e:
        print(f"  ✗ NumPy: {e}")
        return False

    try:
        import scipy
        print("  ✓ SciPy", scipy.__version__)
    except ImportError as e:
        print(f"  ✗ SciPy: {e}")
        return False

    try:
        import matplotlib
        print("  ✓ Matplotlib", matplotlib.__version__)
    except ImportError as e:
        print(f"  ✗ Matplotlib: {e}")
        return False

    return True


def test_rl_dependencies():
    """Test RL-related dependencies."""
    print("\nTesting RL dependencies...")
    try:
        import torch
        print("  ✓ PyTorch", torch.__version__)
        print(f"    CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"    CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"  ✗ PyTorch: {e}")
        return False

    try:
        import gymnasium as gym
        print("  ✓ Gymnasium", gym.__version__)
    except ImportError as e:
        print(f"  ✗ Gymnasium: {e}")
        return False

    return True


def test_slam_dependencies():
    """Test SLAM-related dependencies."""
    print("\nTesting SLAM dependencies...")
    try:
        import cv2
        print("  ✓ OpenCV", cv2.__version__)
    except ImportError as e:
        print(f"  ✗ OpenCV: {e}")
        print("    Install with: pip install opencv-python")
        return False

    return True


def test_visualization_dependencies():
    """Test visualization dependencies."""
    print("\nTesting visualization dependencies...")
    try:
        import seaborn as sns
        print("  ✓ Seaborn", sns.__version__)
    except ImportError as e:
        print(f"  ✗ Seaborn: {e}")
        print("    Install with: pip install seaborn")
        return False

    try:
        import plotly
        print("  ✓ Plotly", plotly.__version__)
    except ImportError as e:
        print(f"  ✗ Plotly: {e}")
        return False

    return True


def test_project_modules():
    """Test project-specific modules."""
    print("\nTesting project modules...")

    # Test RL agents
    try:
        from rl_agents.ppo_agent import PPOAgent
        from rl_agents.advanced_agents import SACAgent, TD3Agent, A2CAgent, DQNAgent
        print("  ✓ RL agents (PPO, SAC, TD3, A2C, DQN)")
    except ImportError as e:
        print(f"  ✗ RL agents: {e}")
        return False

    # Test SLAM module
    try:
        from slam import EKF_SLAM, VisualSLAM, CollaborativeSLAM
        print("  ✓ SLAM module (EKF, Visual, Collaborative)")
    except ImportError as e:
        print(f"  ✗ SLAM module: {e}")
        return False

    # Test environment
    try:
        from simulation.environment import SwarmEnvironment
        print("  ✓ Simulation environment")
    except ImportError as e:
        print(f"  ✗ Simulation environment: {e}")
        return False

    # Test config
    try:
        from config.simulation_config import SimulationConfig
        print("  ✓ Configuration module")
    except ImportError as e:
        print(f"  ✗ Configuration module: {e}")
        return False

    return True


def test_comparison_scripts():
    """Test comparison scripts can be imported."""
    print("\nTesting comparison scripts...")

    try:
        from experiments.rl_comparison import RLAlgorithmComparison
        print("  ✓ RL comparison script")
    except ImportError as e:
        print(f"  ✗ RL comparison script: {e}")
        return False

    try:
        from experiments.slam_comparison import SLAMIntegrationDemo
        print("  ✓ SLAM comparison script")
    except ImportError as e:
        print(f"  ✗ SLAM comparison script: {e}")
        return False

    return True


def test_simple_agent_creation():
    """Test simple agent creation."""
    print("\nTesting agent creation...")

    try:
        from rl_agents.advanced_agents import SACAgent
        import numpy as np

        config = {
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "alpha": 0.2
        }

        agent = SACAgent("test", state_dim=10, action_dim=3, config=config)

        # Test action selection
        state = np.random.randn(10)
        action = agent.select_action(state, deterministic=True)

        print(f"  ✓ SAC agent created and tested")
        print(f"    State dim: 10, Action dim: 3")
        print(f"    Sample action: {action}")

        return True
    except Exception as e:
        print(f"  ✗ Agent creation failed: {e}")
        return False


def test_simple_slam():
    """Test simple SLAM initialization."""
    print("\nTesting SLAM initialization...")

    try:
        from slam import EKF_SLAM
        import numpy as np

        # Create SLAM
        initial_pose = np.array([0, 0, 10, 0, 0, 0])
        slam = EKF_SLAM(initial_pose)

        # Test prediction
        control = np.array([1, 0, 0, 0, 0, 0])
        slam.predict(control, dt=0.1)

        # Get pose
        pose = slam.get_pose()

        print(f"  ✓ EKF-SLAM created and tested")
        print(f"    Initial pose: {initial_pose[:3]}")
        print(f"    After prediction: {pose[:3]}")

        return True
    except Exception as e:
        print(f"  ✗ SLAM initialization failed: {e}")
        return False


def test_directory_structure():
    """Test that required directories exist or can be created."""
    print("\nTesting directory structure...")

    required_dirs = [
        "results",
        "results/rl_comparison",
        "results/slam_demo",
        "results/review_iii",
        "logs",
        "saved_models"
    ]

    for dir_path in required_dirs:
        path = project_root / dir_path
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
                print(f"  ✓ Created directory: {dir_path}")
            except Exception as e:
                print(f"  ✗ Failed to create {dir_path}: {e}")
                return False
        else:
            print(f"  ✓ Directory exists: {dir_path}")

    return True


def print_system_info():
    """Print system information."""
    print("\n" + "="*60)
    print("System Information")
    print("="*60)

    import platform
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")

    try:
        import torch
        print(f"\nPyTorch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except:
        pass

    print("="*60)


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Review III Installation Test")
    print("="*60)

    print_system_info()

    all_passed = True

    # Run tests
    tests = [
        ("Basic Imports", test_basic_imports),
        ("RL Dependencies", test_rl_dependencies),
        ("SLAM Dependencies", test_slam_dependencies),
        ("Visualization Dependencies", test_visualization_dependencies),
        ("Project Modules", test_project_modules),
        ("Comparison Scripts", test_comparison_scripts),
        ("Agent Creation", test_simple_agent_creation),
        ("SLAM Initialization", test_simple_slam),
        ("Directory Structure", test_directory_structure)
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"\n✗ {name} failed with exception: {e}")
            results.append((name, False))
            all_passed = False

    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:10} {name}")

    print("="*60)

    if all_passed:
        print("\n🎉 All tests passed! System is ready for Review III.")
        print("\nNext steps:")
        print("  1. Run quick demo: python run_review_iii_demo.py --quick")
        print("  2. Or use Make: make review3-quick")
        print("  3. Check results: make results")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please install missing dependencies:")
        print("  pip install -r requirements.txt")
        print("\nFor help, see REVIEW_III_GUIDE.md")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
