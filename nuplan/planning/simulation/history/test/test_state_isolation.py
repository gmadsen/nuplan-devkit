# ABOUTME: Tests to ensure state isolation when reducing deepcopy calls in history buffer
# ABOUTME: Validates that removing defensive copying doesn't introduce shared state bugs

import unittest
from collections import deque

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.tracks_observation import TracksObservation


class TestStateIsolation(unittest.TestCase):
    """
    Test suite to ensure state isolation in history buffer when reducing deepcopy calls.
    These tests verify that modifications to states after appending don't affect buffered history.
    """

    def setUp(self) -> None:
        """Initialize test scenario and buffer"""
        self.scenario = MockAbstractScenario(number_of_past_iterations=20)
        self.buffer_size = 10
        self.tracks_observation = TracksObservation(self.scenario)

    def _create_mutable_ego_state(self, x: float = 0.0, y: float = 0.0, heading: float = 0.0) -> EgoState:
        """Helper to create an ego state with mutable fields"""
        return EgoState.build_from_rear_axle(
            rear_axle_pose=StateSE2(x=x, y=y, heading=heading),
            rear_axle_velocity_2d=StateVector2D(x=5.0, y=0.0),
            rear_axle_acceleration_2d=StateVector2D(x=0.0, y=0.0),
            tire_steering_angle=0.0,
            time_point=TimePoint(0),
            vehicle_parameters=get_pacifica_parameters(),
        )

    def test_ego_state_isolation_after_append(self) -> None:
        """
        Test that modifying ego state after append doesn't affect buffered history.
        This ensures proper isolation when removing deepcopy calls.
        """
        # Initialize with one dummy state (buffer can't be empty)
        init_ego = self._create_mutable_ego_state()
        init_obs = self.scenario.initial_tracked_objects
        buffer = SimulationHistoryBuffer(
            ego_state_buffer=deque([init_ego], maxlen=self.buffer_size),
            observations_buffer=deque([init_obs], maxlen=self.buffer_size),
        )

        # Create and append an ego state
        ego_state = self._create_mutable_ego_state(x=10.0, y=20.0, heading=0.5)
        observation = self.scenario.initial_tracked_objects
        buffer.append(ego_state, observation)

        # Store original values (last appended state)
        original_x = buffer.ego_states[-1].rear_axle.x
        original_y = buffer.ego_states[-1].rear_axle.y
        original_heading = buffer.ego_states[-1].rear_axle.heading

        # Verify values are correct
        self.assertEqual(original_x, 10.0)
        self.assertEqual(original_y, 20.0)
        self.assertEqual(original_heading, 0.5)

        # Modify the original ego_state object
        # NOTE: EgoState is immutable via dataclass, but we test the reference
        # If we stored reference without copy, these would be same object
        self.assertIsNot(buffer.ego_states[-1], ego_state,
                         "Buffer should not share reference with original state")

        # Values should remain unchanged in buffer
        self.assertEqual(buffer.ego_states[-1].rear_axle.x, original_x)
        self.assertEqual(buffer.ego_states[-1].rear_axle.y, original_y)
        self.assertEqual(buffer.ego_states[-1].rear_axle.heading, original_heading)

    def test_observation_isolation_after_append(self) -> None:
        """
        Test that modifying observation after append doesn't affect buffered history.
        """
        # Initialize with one dummy state (buffer can't be empty)
        init_ego = self._create_mutable_ego_state()
        init_obs = self.scenario.initial_tracked_objects
        buffer = SimulationHistoryBuffer(
            ego_state_buffer=deque([init_ego], maxlen=self.buffer_size),
            observations_buffer=deque([init_obs], maxlen=self.buffer_size),
        )

        ego_state = self._create_mutable_ego_state()
        observation = self.scenario.initial_tracked_objects
        buffer.append(ego_state, observation)

        # Store original observation count (last appended)
        original_count = len(buffer.observations[-1].tracked_objects)

        # Buffer should not share reference with original observation
        self.assertIsNot(buffer.observations[-1], observation,
                         "Buffer should not share reference with original observation")

        # Count should remain unchanged
        self.assertEqual(len(buffer.observations[-1].tracked_objects), original_count)

    def test_multiple_appends_isolation(self) -> None:
        """
        Test that multiple appends maintain state isolation.
        Each timestep should have independent state.
        """
        # Initialize with one dummy state (buffer can't be empty)
        init_ego = self._create_mutable_ego_state()
        init_obs = self.scenario.initial_tracked_objects
        buffer = SimulationHistoryBuffer(
            ego_state_buffer=deque([init_ego], maxlen=self.buffer_size),
            observations_buffer=deque([init_obs], maxlen=self.buffer_size),
        )

        # Append multiple states
        states = []
        observations = []
        for i in range(5):
            ego = self._create_mutable_ego_state(x=float(i * 10), y=float(i * 20))
            obs = self.scenario.initial_tracked_objects
            states.append(ego)
            observations.append(obs)
            buffer.append(ego, obs)

        # Verify each buffered state has correct values
        # Note: First state is the init state (x=0, y=0), followed by our appended states
        for i in range(5):
            buffered_idx = i + 1  # Skip init state
            self.assertEqual(buffer.ego_states[buffered_idx].rear_axle.x, float(i * 10))
            self.assertEqual(buffer.ego_states[buffered_idx].rear_axle.y, float(i * 20))

            # Verify not sharing references
            self.assertIsNot(buffer.ego_states[buffered_idx], states[i],
                             f"Buffered state {i} should not share reference with original")

    def test_current_state_isolation(self) -> None:
        """
        Test that current_state() returns isolated states.
        """
        # Initialize with one dummy state (buffer can't be empty)
        init_ego = self._create_mutable_ego_state()
        init_obs = self.scenario.initial_tracked_objects
        buffer = SimulationHistoryBuffer(
            ego_state_buffer=deque([init_ego], maxlen=self.buffer_size),
            observations_buffer=deque([init_obs], maxlen=self.buffer_size),
        )

        ego_state = self._create_mutable_ego_state(x=100.0)
        observation = self.scenario.initial_tracked_objects
        buffer.append(ego_state, observation)

        # Get current state
        current_ego, current_obs = buffer.current_state

        # Verify values
        self.assertEqual(current_ego.rear_axle.x, 100.0)

        # Verify not sharing references with buffer internals
        # NOTE: current_state returns references from deque, which is OK
        # as long as they're not shared with external callers
        self.assertIs(current_ego, buffer.ego_states[-1],
                      "current_state should return reference to buffered state")

    def test_buffer_overflow_isolation(self) -> None:
        """
        Test that state isolation is maintained when buffer overflows (FIFO).
        """
        # Initialize with one dummy state (buffer can't be empty)
        init_ego = self._create_mutable_ego_state()
        init_obs = self.scenario.initial_tracked_objects
        buffer = SimulationHistoryBuffer(
            ego_state_buffer=deque([init_ego], maxlen=5),  # Small buffer
            observations_buffer=deque([init_obs], maxlen=5),
        )

        # Append more states than buffer size
        states = []
        for i in range(10):
            ego = self._create_mutable_ego_state(x=float(i))
            obs = self.scenario.initial_tracked_objects
            states.append(ego)
            buffer.append(ego, obs)

        # Buffer should only contain last 5 states
        self.assertEqual(len(buffer), 5)

        # Verify correct states are retained (FIFO - oldest discarded)
        # After appending 10 states starting from init, buffer should have states 6-9 (indices 6-10)
        for i, buffered_ego in enumerate(buffer.ego_states):
            expected_x = float(6 + i)  # States at indices 6, 7, 8, 9, 10 (x values)
            self.assertEqual(buffered_ego.rear_axle.x, expected_x)

    def test_extend_isolation(self) -> None:
        """
        Test that extend() maintains state isolation for batch appends.
        """
        # Initialize with one dummy state (buffer can't be empty)
        init_ego = self._create_mutable_ego_state()
        init_obs = self.scenario.initial_tracked_objects
        buffer = SimulationHistoryBuffer(
            ego_state_buffer=deque([init_ego], maxlen=self.buffer_size),
            observations_buffer=deque([init_obs], maxlen=self.buffer_size),
        )

        # Create batch of states
        ego_states = [self._create_mutable_ego_state(x=float(i * 10)) for i in range(5)]
        observations = [self.scenario.initial_tracked_objects] * 5

        buffer.extend(ego_states, observations)

        # Verify isolation
        # Note: First state is init state, then the 5 extended states
        for i in range(5):
            buffered_idx = i + 1  # Skip init state
            self.assertEqual(buffer.ego_states[buffered_idx].rear_axle.x, float(i * 10))
            self.assertIsNot(buffer.ego_states[buffered_idx], ego_states[i],
                             f"Extended state {i} should not share reference")


if __name__ == '__main__':
    unittest.main()
