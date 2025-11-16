# ABOUTME: WebSocket-based streaming callback for real-time simulation visualization
# ABOUTME: Sends simulation state to web dashboard at each timestep with minimal overhead

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

try:
    import websockets
    from websockets.sync.client import connect as ws_connect, ClientConnection
except ImportError:
    # Graceful degradation if websockets not installed
    websockets = None
    ws_connect = None
    ClientConnection = None

from nuplan.planning.simulation.callback.abstract_callback import AbstractCallback
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory

logger = logging.getLogger(__name__)


class StreamingVisualizationCallback(AbstractCallback):
    """
    Real-time streaming callback for web-based visualization.

    Streams simulation data via WebSocket at each timestep for live dashboard display.
    Designed for minimal overhead (< 1ms per step) via non-blocking sends and lightweight JSON.

    **Architecture**:
    - on_simulation_start: Send initial map/scenario metadata
    - on_step_end: Stream ego state, trajectory, and agents (every 0.1s)
    - on_simulation_end: Send completion signal

    **Performance characteristics**:
    - Target overhead: < 1ms per timestep
    - Fallback: Gracefully degrades if WebSocket server unavailable
    - Thread safety: Safe for sequential worker only (Ray pickling not supported)

    **AIDEV-NOTE**: Use with worker=sequential only - WebSocket connections not pickle-friendly!
    """

    def __init__(
        self,
        websocket_url: str = "ws://localhost:8765/ingest",
        enable_fallback: bool = True,
        max_trajectory_points: int = 50,
        max_tracked_objects: int = 100,
    ):
        """
        Initialize streaming visualization callback.

        :param websocket_url: WebSocket server URL (e.g., "ws://localhost:8765/ingest")
        :param enable_fallback: If True, continue simulation even if WebSocket fails
        :param max_trajectory_points: Downsample trajectory to this many points (performance)
        :param max_tracked_objects: Limit tracked objects sent per frame (performance)
        """
        if websockets is None:
            logger.warning(
                "websockets library not installed! Install with: uv add websockets\n"
                "StreamingVisualizationCallback will be disabled."
            )

        self.websocket_url = websocket_url
        self.enable_fallback = enable_fallback
        self.max_trajectory_points = max_trajectory_points
        self.max_tracked_objects = max_tracked_objects

        self._ws_client: Optional[ClientConnection] = None
        self._connection_failed = False
        self._step_count = 0

    def _connect(self) -> bool:
        """
        Establish WebSocket connection (lazy initialization).

        :return: True if connected, False if failed
        """
        if self._connection_failed:
            return False

        if self._ws_client is not None:
            return True

        if websockets is None:
            self._connection_failed = True
            return False

        try:
            logger.info(f"Connecting to WebSocket server at {self.websocket_url}")
            self._ws_client = ws_connect(self.websocket_url, open_timeout=2.0)
            logger.info("WebSocket connection established")
            return True
        except Exception as e:
            logger.warning(f"Failed to connect to WebSocket server: {e}")
            if not self.enable_fallback:
                raise
            self._connection_failed = True
            return False

    def _send_json(self, data: Dict[str, Any]) -> None:
        """
        Send JSON message to WebSocket server (non-blocking pattern).

        :param data: Dictionary to serialize and send
        """
        if not self._connect():
            return

        try:
            message = json.dumps(data)
            self._ws_client.send(message)
        except Exception as e:
            logger.warning(f"WebSocket send failed: {e}")
            if not self.enable_fallback:
                raise
            # Mark connection as failed to avoid repeated attempts
            self._connection_failed = True
            if self._ws_client:
                try:
                    self._ws_client.close()
                except:
                    pass
                self._ws_client = None

    def _extract_map_data(self, setup: SimulationSetup) -> Dict[str, Any]:
        """
        Extract map geometry for initial display.

        :param setup: Simulation setup containing map API
        :return: Dictionary with lane boundaries, roadblocks, etc.
        """
        # AIDEV-TODO: Implement map extraction from setup.scenario.map_api
        # For Phase 1, send minimal metadata - full map in Phase 3
        return {
            "map_name": setup.scenario.map_api.map_name if setup.scenario.map_api else "unknown",
            "scenario_name": setup.scenario.scenario_name,
            "scenario_type": setup.scenario.scenario_type,
        }

    def _extract_ego_state(self, sample: SimulationHistorySample) -> Dict[str, Any]:
        """
        Extract ego vehicle state at current timestep.

        :param sample: Simulation history sample containing ego state
        :return: Dictionary with position, velocity, acceleration, heading
        """
        ego = sample.ego_state

        return {
            "x": float(ego.rear_axle.x),
            "y": float(ego.rear_axle.y),
            "heading": float(ego.rear_axle.heading),
            "velocity": float(ego.dynamic_car_state.speed),
            "acceleration": float(ego.dynamic_car_state.acceleration),
            "steering_angle": float(ego.tire_steering_angle),
        }

    def _extract_trajectory(self, sample: SimulationHistorySample) -> List[Dict[str, float]]:
        """
        Extract planned trajectory (downsampled for performance).

        :param sample: Simulation history sample containing trajectory
        :return: List of waypoints [{x, y, heading, velocity}, ...]
        """
        trajectory = sample.trajectory.get_sampled_trajectory()

        # Downsample to max_trajectory_points
        step = max(1, len(trajectory) // self.max_trajectory_points)
        sampled_trajectory = trajectory[::step]

        return [
            {
                "x": float(state.rear_axle.x),
                "y": float(state.rear_axle.y),
                "heading": float(state.rear_axle.heading),
                "velocity": float(state.dynamic_car_state.speed),
            }
            for state in sampled_trajectory
        ]

    def _extract_tracked_objects(self, sample: SimulationHistorySample) -> List[Dict[str, Any]]:
        """
        Extract tracked agents (vehicles, pedestrians).

        :param sample: Simulation history sample containing observations
        :return: List of tracked objects with positions and velocities
        """
        if not sample.observation or not sample.observation.tracked_objects:
            return []

        tracked_objects = sample.observation.tracked_objects.tracked_objects

        # Limit to max_tracked_objects (performance)
        if len(tracked_objects) > self.max_tracked_objects:
            tracked_objects = tracked_objects[:self.max_tracked_objects]

        return [
            {
                "id": obj.track_token,
                "type": obj.tracked_object_type.fullname,
                "x": float(obj.center.x),
                "y": float(obj.center.y),
                "heading": float(obj.center.heading),
                "velocity_x": float(obj.velocity.x),
                "velocity_y": float(obj.velocity.y),
                "length": float(obj.box.length),
                "width": float(obj.box.width),
            }
            for obj in tracked_objects
        ]

    # Callback lifecycle hooks

    def on_initialization_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Called once before all scenarios run."""
        pass

    def on_initialization_end(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Called once after all scenarios complete."""
        # Close WebSocket connection
        if self._ws_client:
            try:
                self._ws_client.close()
                logger.info("WebSocket connection closed")
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")

    def on_simulation_start(self, setup: SimulationSetup) -> None:
        """
        Send initial scenario metadata and map data.

        Called once at the start of each scenario.
        """
        self._step_count = 0

        message = {
            "type": "simulation_start",
            "data": {
                **self._extract_map_data(setup),
                "initial_time": float(setup.scenario.start_time.time_s),
                "duration": float(setup.scenario.duration_s.time_s),
            }
        }

        self._send_json(message)

    def on_simulation_end(
        self,
        setup: SimulationSetup,
        planner: AbstractPlanner,
        history: SimulationHistory
    ) -> None:
        """
        Send simulation completion signal.

        Called once at the end of each scenario.
        """
        message = {
            "type": "simulation_end",
            "data": {
                "total_steps": self._step_count,
                "scenario_name": setup.scenario.scenario_name,
            }
        }

        self._send_json(message)

    def on_step_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Called before each simulation timestep."""
        pass

    def on_step_end(
        self,
        setup: SimulationSetup,
        planner: AbstractPlanner,
        sample: SimulationHistorySample
    ) -> None:
        """
        Stream current simulation state to WebSocket.

        Called after each simulation timestep (~200 times per scenario).
        Target overhead: < 1ms
        """
        self._step_count += 1

        message = {
            "type": "simulation_step",
            "data": {
                "iteration": sample.iteration.index,
                "time_s": float(sample.iteration.time_s),
                "ego_state": self._extract_ego_state(sample),
                "trajectory": self._extract_trajectory(sample),
                "tracked_objects": self._extract_tracked_objects(sample),
            }
        }

        self._send_json(message)

    def on_planner_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Called before planner computes trajectory."""
        pass

    def on_planner_end(
        self,
        setup: SimulationSetup,
        planner: AbstractPlanner,
        trajectory: AbstractTrajectory
    ) -> None:
        """Called after planner computes trajectory."""
        pass
