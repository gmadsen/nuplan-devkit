#!/usr/bin/env python3
"""
Profile script to identify where deepcopy calls are happening in nuPlan simulation.
This helps us understand what's triggering the 1.5M deepcopy calls per simulation.
"""

import cProfile
import pstats
import sys
from pathlib import Path

# Add nuplan to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    """Run a minimal simulation and profile deepcopy usage."""
    # Import here to avoid loading before path setup
    from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario
    from nuplan.planning.simulation.observation.tracks_observation import TracksObservation
    from collections import deque

    print("Setting up test scenario...")
    scenario = MockAbstractScenario(number_of_past_iterations=20)

    # Create history buffer
    print("Creating history buffer...")
    init_ego = EgoState.build_from_rear_axle(
        rear_axle_pose=StateSE2(x=0.0, y=0.0, heading=0.0),
        rear_axle_velocity_2d=StateVector2D(x=5.0, y=0.0),
        rear_axle_acceleration_2d=StateVector2D(x=0.0, y=0.0),
        tire_steering_angle=0.0,
        time_point=TimePoint(0),
        vehicle_parameters=get_pacifica_parameters(),
    )
    init_obs = scenario.initial_tracked_objects

    buffer = SimulationHistoryBuffer(
        ego_state_buffer=deque([init_ego], maxlen=200),
        observations_buffer=deque([init_obs], maxlen=200),
    )

    print("Starting profiling...")
    profiler = cProfile.Profile()
    profiler.enable()

    # Simulate 100 timesteps of appending
    for i in range(100):
        ego = EgoState.build_from_rear_axle(
            rear_axle_pose=StateSE2(x=float(i), y=float(i), heading=0.0),
            rear_axle_velocity_2d=StateVector2D(x=5.0, y=0.0),
            rear_axle_acceleration_2d=StateVector2D(x=0.0, y=0.0),
            tire_steering_angle=0.0,
            time_point=TimePoint(i * 100000),
            vehicle_parameters=get_pacifica_parameters(),
        )
        obs = scenario.initial_tracked_objects

        # This is where deepcopy might happen
        buffer.append(ego, obs)

        # Access properties (triggers list() copy)
        _ = buffer.ego_states
        _ = buffer.observations

    profiler.disable()

    print("\nProfiler results:")
    print("=" * 80)

    stats = pstats.Stats(profiler)

    # Find deepcopy calls
    print("\nSearching for 'deepcopy' in function calls...")
    stats.sort_stats('cumulative')
    stats.print_stats('deepcopy', 20)

    print("\nSearching for 'copy' in function calls...")
    stats.sort_stats('cumulative')
    stats.print_stats('copy', 20)

    print("\nTop 20 functions by cumulative time:")
    stats.sort_stats('cumulative')
    stats.print_stats(20)


if __name__ == "__main__":
    main()
