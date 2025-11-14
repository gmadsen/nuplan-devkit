#!/usr/bin/env python3
# ABOUTME: Health check script to verify nuPlan uv migration and environment setup
# ABOUTME: Tests imports, CUDA, nuPlan modules, and basic functionality without requiring dataset

import sys
from typing import Dict, List, Tuple


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_test(name: str, status: str, details: str = "") -> None:
    """Print a test result with status indicator."""
    symbols = {"pass": "✓", "fail": "✗", "warn": "⚠", "info": "ℹ"}
    colors = {
        "pass": "\033[92m",  # Green
        "fail": "\033[91m",  # Red
        "warn": "\033[93m",  # Yellow
        "info": "\033[94m",  # Blue
    }
    reset = "\033[0m"

    symbol = symbols.get(status, "?")
    color = colors.get(status, "")

    print(f"{color}{symbol}{reset} {name}")
    if details:
        print(f"  {details}")


def test_core_imports() -> Tuple[bool, List[str]]:
    """Test that all core packages can be imported."""
    print_header("1. Core Package Imports")

    packages = [
        ("nuplan", "nuPlan devkit"),
        ("torch", "PyTorch"),
        ("hydra", "Hydra configuration"),
        ("pytorch_lightning", "PyTorch Lightning"),
        ("geopandas", "GeoPandas (maps)"),
        ("ray", "Ray (distributed)"),
        ("bokeh", "Bokeh (visualization)"),
        ("cv2", "OpenCV"),
    ]

    all_passed = True
    failed_packages = []

    for module_name, description in packages:
        try:
            __import__(module_name)
            print_test(f"{description}", "pass", f"import {module_name}")
        except ImportError as e:
            print_test(f"{description}", "fail", f"import {module_name} failed: {e}")
            all_passed = False
            failed_packages.append(module_name)

    return all_passed, failed_packages


def test_cuda() -> Tuple[bool, Dict[str, str]]:
    """Test CUDA availability and configuration."""
    print_header("2. CUDA / GPU Configuration")

    try:
        import torch

        info = {
            "pytorch_version": torch.__version__,
            "cuda_available": str(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "device_count": str(torch.cuda.device_count()) if torch.cuda.is_available() else "0",
        }

        print_test("PyTorch version", "info", f"{info['pytorch_version']}")

        if torch.cuda.is_available():
            print_test("CUDA available", "pass", "GPU acceleration enabled")
            print_test("CUDA version", "info", f"{info['cuda_version']}")
            print_test("Device count", "info", f"{info['device_count']} GPU(s)")

            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                print_test(f"Device {i}", "info", device_name)
                info[f"device_{i}"] = device_name

            # Test a simple CUDA operation
            try:
                x = torch.randn(10, 10).cuda()
                y = x @ x.T
                print_test("CUDA compute test", "pass", "Matrix multiplication successful")
            except Exception as e:
                print_test("CUDA compute test", "fail", f"{e}")
                return False, info

            return True, info
        else:
            print_test("CUDA available", "warn", "Running in CPU mode (slower for training)")
            return True, info  # Not a failure, just a warning

    except Exception as e:
        print_test("CUDA test", "fail", f"Unexpected error: {e}")
        return False, {}


def test_nuplan_modules() -> Tuple[bool, List[str]]:
    """Test that nuPlan-specific modules can be imported."""
    print_header("3. nuPlan Module Imports")

    modules = [
        ("nuplan.planning.simulation.planner.abstract_planner", "AbstractPlanner"),
        ("nuplan.common.actor_state.ego_state", "EgoState"),
        ("nuplan.common.actor_state.state_representation", "StateSE2"),
        ("nuplan.planning.scenario_builder.abstract_scenario", "AbstractScenario"),
        ("nuplan.common.maps.maps_datatypes", "TrafficLightStatusData"),
        ("nuplan.planning.simulation.trajectory.abstract_trajectory", "AbstractTrajectory"),
        ("nuplan.planning.simulation.callback.abstract_callback", "AbstractCallback"),
    ]

    all_passed = True
    failed_modules = []

    for module_path, class_name in modules:
        try:
            module = __import__(module_path, fromlist=[class_name])
            getattr(module, class_name)  # Verify the class exists
            print_test(f"{class_name}", "pass", f"from {module_path}")
        except ImportError as e:
            print_test(f"{class_name}", "fail", f"{module_path}: {e}")
            all_passed = False
            failed_modules.append(module_path)
        except AttributeError as e:
            print_test(f"{class_name}", "fail", f"Class not found: {e}")
            all_passed = False
            failed_modules.append(module_path)

    return all_passed, failed_modules


def test_basic_functionality() -> bool:
    """Test basic nuPlan functionality."""
    print_header("4. Basic Functionality Tests")

    try:
        from nuplan.common.actor_state.state_representation import StateSE2
        from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
        import numpy as np

        # Test 1: Create a state
        state = StateSE2(x=0.0, y=0.0, heading=0.0)
        print_test("Create StateSE2", "pass", f"{state}")

        # Test 2: Get vehicle parameters
        params = get_pacifica_parameters()
        print_test("Vehicle parameters", "pass", f"wheelbase={params.wheel_base}m")

        # Test 3: Basic geometry operations
        state2 = StateSE2(x=10.0, y=5.0, heading=np.pi/4)
        distance = np.sqrt((state2.x - state.x)**2 + (state2.y - state.y)**2)
        print_test("Geometry operations", "pass", f"Distance: {distance:.2f}m")

        return True

    except Exception as e:
        print_test("Basic functionality", "fail", f"{e}")
        return False


def test_environment_variables() -> Tuple[bool, Dict[str, str]]:
    """Check if required environment variables are set."""
    print_header("5. Environment Variables")

    import os

    env_vars = {
        "NUPLAN_DATA_ROOT": "Dataset root directory",
        "NUPLAN_MAPS_ROOT": "Maps root directory",
        "NUPLAN_EXP_ROOT": "Experiments root directory",
    }

    env_status = {}
    all_set = True

    for var, description in env_vars.items():
        value = os.getenv(var)
        if value:
            print_test(f"{var}", "pass", f"{value}")
            env_status[var] = value
        else:
            print_test(f"{var}", "warn", f"Not set (required for dataset access)")
            all_set = False
            env_status[var] = "NOT SET"

    if not all_set:
        print("\n  💡 Tip: Copy .env.example to .env and set your paths")
        print("     then run: source .env")

    return all_set, env_status


def test_python_version() -> bool:
    """Check Python version compatibility."""
    print_header("0. Python Version")

    required_version = (3, 9)
    current_version = sys.version_info[:2]

    version_str = f"{current_version[0]}.{current_version[1]}"

    if current_version == required_version:
        print_test("Python version", "pass", f"{version_str} (required: 3.9)")
        return True
    elif current_version > required_version:
        print_test("Python version", "warn", f"{version_str} (recommended: 3.9, may have compatibility issues)")
        return True
    else:
        print_test("Python version", "fail", f"{version_str} (required: 3.9+)")
        return False


def print_summary(results: Dict[str, bool]) -> None:
    """Print a summary of all tests."""
    print_header("Summary")

    total = len(results)
    passed = sum(results.values())
    failed = total - passed

    print(f"\nTests run: {total}")
    print(f"✓ Passed: {passed}")
    if failed > 0:
        print(f"✗ Failed: {failed}")

    if all(results.values()):
        print("\n🎉 All tests passed! nuPlan environment is healthy.")
        print("\nNext steps:")
        print("  1. Set up .env with dataset paths: cp .env.example .env")
        print("  2. Download mini dataset: just cli download --mini")
        print("  3. Run tutorials: just tutorial")
    else:
        print("\n⚠️  Some tests failed. Please address the issues above.")
        print("\nTroubleshooting:")
        print("  - Reinstall: just clean-all && just setup")
        print("  - Check CUDA drivers: nvidia-smi")
        print("  - See MIGRATION_SUMMARY.md for more help")


def main() -> int:
    """Run all health checks."""
    print("\n🧪 nuPlan Health Check")
    print("Testing uv migration and environment setup...")

    results = {}

    # Test 0: Python version
    results["python_version"] = test_python_version()

    # Test 1: Core imports
    imports_passed, failed_packages = test_core_imports()
    results["core_imports"] = imports_passed

    # Test 2: CUDA
    cuda_passed, cuda_info = test_cuda()
    results["cuda"] = cuda_passed

    # Test 3: nuPlan modules
    modules_passed, failed_modules = test_nuplan_modules()
    results["nuplan_modules"] = modules_passed

    # Test 4: Basic functionality
    results["basic_functionality"] = test_basic_functionality()

    # Test 5: Environment variables (warning only, not a failure)
    env_set, env_status = test_environment_variables()
    # Don't count as failure since dataset might not be downloaded yet

    # Print summary
    print_summary(results)

    # Return exit code
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
