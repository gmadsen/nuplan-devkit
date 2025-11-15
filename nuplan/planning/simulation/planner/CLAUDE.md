# CLAUDE.md - nuplan/planning/simulation/planner

## Purpose & Responsibility

**THE core planning interface for nuPlan.** This module defines `AbstractPlanner` - the primary abstraction that all autonomous vehicle planners must implement. Every planner (rule-based, ML-based, hybrid) inherits from `AbstractPlanner` and implements `compute_planner_trajectory()` to generate ego vehicle trajectories from observations. This is where autonomous decision-making happens.

## Key Abstractions & Classes

### Core Interface
- **`AbstractPlanner`** - THE fundamental interface (ABC)
  - **`initialize(PlannerInitialization)`** - One-time setup with map, route, goal
  - **`compute_planner_trajectory(PlannerInput)`** - Generate trajectory (called every 0.1s typically)
  - **`observation_type()`** - Declare expected observation format
  - **`name()`** - Planner identifier string
  - **`generate_planner_report()`** - Runtime statistics
  - Class attribute: `requires_scenario` (bool) - Oracle planners only (not for competition submission)
  - Auto-tracked: `_compute_trajectory_runtimes` - Performance profiling

### Input Data Structures
- **`PlannerInitialization`** - One-time initialization data (frozen dataclass)
  - `route_roadblock_ids: List[str]` - Roadblock IDs comprising the route to goal
  - `mission_goal: StateSE2` - Final destination pose (may not be reachable in scenario)
  - `map_api: AbstractMap` - Map query interface

- **`PlannerInput`** - Per-timestep input to planner (frozen dataclass)
  - `iteration: SimulationIteration` - Current timestep and simulation progress
  - `history: SimulationHistoryBuffer` - Rolling buffer of past states/observations
  - `traffic_light_data: Optional[List[TrafficLightStatusData]]` - Traffic light statuses

### Concrete Implementations (Examples)
- **`SimplePlanner`** - Baseline planner (goes straight with constant acceleration)
  - Parameters: `horizon_seconds`, `sampling_time`, `acceleration`, `max_velocity`, `steering_angle`
  - Uses `KinematicBicycleModel` for motion propagation
  - Returns `InterpolatedTrajectory` from EgoState sequence

- **`MLPlanner`** (in ml_planner/) - ML-based planning
  - Loads PyTorch Lightning checkpoint
  - Feature extraction → model inference → trajectory decoding

- **`IDMPlanner`** - Intelligent Driver Model planner
  - Car-following behavior with IDM dynamics

- **`LogFuturePlanner`** - Oracle planner (replays log data)
  - Uses `requires_scenario=True`
  - NOT valid for submissions!

- **`RemotePlanner`** - Dockerized planner (for competition submission)

### Supporting Classes
- **`PlannerReport`** - Runtime statistics dataclass
  - `compute_trajectory_runtimes: List[float]` - Time series of planning latencies

## Architecture & Design Patterns

1. **Strategy Pattern**: `AbstractPlanner` is a strategy interface
   - Planners are interchangeable via polymorphism
   - Hydra configuration selects planner at runtime

2. **Template Method**: `compute_trajectory()` wraps `compute_planner_trajectory()`
   - Adds timing instrumentation automatically
   - Exception handling with timing preservation
   - User implements `compute_planner_trajectory()`, framework calls `compute_trajectory()`

3. **Initialization Separation**: Two-phase construction
   - `__init__()` - Planner-specific parameters
   - `initialize()` - Scenario-specific data (route, map, goal)
   - Enables planner reuse across scenarios

4. **Type Declaration**: `observation_type()` declares expected input format
   - Simulation validates observation matches expected type
   - Enables heterogeneous observation types across planners

5. **Performance Monitoring**: Built-in runtime tracking
   - Every `compute_trajectory()` call is timed
   - `generate_planner_report()` extracts statistics
   - Critical for real-time constraint analysis (must finish in < 0.1s typically)

6. **Frozen Dataclasses**: `PlannerInitialization` and `PlannerInput` are immutable
   - Prevents accidental mutation during planning
   - Thread-safe for parallel simulations

## Dependencies (What We Import)

### Internal nuPlan
- `nuplan.common.actor_state.state_representation` - StateSE2 (goal pose)
- `nuplan.common.maps.abstract_map` - AbstractMap (map queries)
- `nuplan.common.maps.maps_datatypes` - TrafficLightStatusData
- `nuplan.planning.simulation.history.simulation_history_buffer` - SimulationHistoryBuffer
- `nuplan.planning.simulation.observation.observation_type` - Observation types
- `nuplan.planning.simulation.simulation_time_controller.simulation_iteration` - SimulationIteration
- `nuplan.planning.simulation.trajectory.abstract_trajectory` - AbstractTrajectory (output)

### For Concrete Implementations
- SimplePlanner:
  - `nuplan.planning.simulation.controller.motion_model.kinematic_bicycle` - KinematicBicycleModel
  - `nuplan.planning.simulation.trajectory.interpolated_trajectory` - InterpolatedTrajectory
- MLPlanner:
  - PyTorch Lightning, model loading utilities

## Dependents (Who Imports Us)

**Central to all nuPlan workflows:**
- **`nuplan/planning/simulation/runner/`** - Simulation loop calls `compute_trajectory()` each timestep
- **`nuplan/planning/script/run_simulation.py`** - Main entry point instantiates planners
- **`nuplan/planning/training/`** - Training compares ML planner outputs to expert
- **`nuplan/planning/metrics/`** - Metrics evaluate planner-generated trajectories
- **Hydra configs** (`config/planner/`) - Each planner has YAML config for instantiation

## Critical Files (Prioritized)

1. **`abstract_planner.py`** (124 lines) - **MUST READ FIRST!**
   - Defines `AbstractPlanner`, `PlannerInput`, `PlannerInitialization`
   - Core interface every planner implements
   - Timing instrumentation logic

2. **`simple_planner.py`** (96 lines) - **Reference implementation**
   - Minimal working example of AbstractPlanner
   - Shows initialization, trajectory propagation, InterpolatedTrajectory construction
   - Use as template for custom planners!

3. **`ml_planner/ml_planner.py`** - ML-based planning
   - Feature extraction from observations
   - Model inference
   - Trajectory decoding
   - Critical for understanding ML planning workflow

4. **`idm_planner.py`** - Intelligent Driver Model
   - Car-following logic
   - Demonstrates reactive planning

5. **`abstract_idm_planner.py`** - Base class for IDM variants

6. **`log_future_planner.py`** - Oracle/replay planner
   - Shows how to use scenario data
   - Example of `requires_scenario=True`

7. **`remote_planner.py`** - Dockerized planner wrapper
   - Competition submission mechanism

8. **`planner_report.py`** - Runtime statistics

9. **`ml_planner/model_loader.py`** - PyTorch Lightning checkpoint loading
10. **`ml_planner/transform_utils.py`** - Coordinate transformations for ML

## Common Usage Patterns

### Implementing a Custom Planner
```python
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner, PlannerInitialization, PlannerInput
)
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from typing import Type, List
from nuplan.common.actor_state.ego_state import EgoState

class MyPlanner(AbstractPlanner):
    def __init__(self, param1: float, param2: int):
        """Constructor for planner-specific parameters"""
        self.param1 = param1
        self.param2 = param2
        self._map_api = None  # Set in initialize()
        self._route = None

    def initialize(self, initialization: PlannerInitialization) -> None:
        """One-time setup with scenario-specific data"""
        self._map_api = initialization.map_api
        self._route = initialization.route_roadblock_ids
        self._goal = initialization.mission_goal
        # Precompute anything route-specific here

    def name(self) -> str:
        return "MyPlanner"

    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks  # or your custom observation type

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """Core planning logic - called every timestep (typically 0.1s)"""
        # Access current state
        ego_state, observation = current_input.history.current_state

        # Access past history
        past_ego_states = current_input.history.ego_states
        past_observations = current_input.history.observations

        # Access traffic lights
        traffic_lights = current_input.traffic_light_data

        # Query map
        nearby_lanes = self._map_api.get_proximal_map_objects(
            ego_state.center, radius=50.0
        )

        # YOUR PLANNING LOGIC HERE
        future_states: List[EgoState] = self._plan(
            ego_state, observation, nearby_lanes, traffic_lights
        )

        # Return trajectory
        return InterpolatedTrajectory(future_states)

    def _plan(self, ego_state, observation, map_data, traffic_lights):
        """Your core planning algorithm"""
        # ...
        return future_ego_states
```

### Accessing History Buffer
```python
# Current state
ego_state, observation = current_input.history.current_state

# All past ego states (including current)
all_ego_states = current_input.history.ego_states  # List[EgoState], newest last

# Past N states
last_5_states = current_input.history.ego_states[-5:]

# Observations
all_observations = current_input.history.observations  # List[Observation]
current_tracked_objects = observation.tracked_objects  # TrackedObjects
```

### Map Queries During Planning
```python
# Query lanes near ego
proximal_lanes = self._map_api.get_proximal_map_objects(
    point=ego_state.center,
    radius=100.0,  # meters
    layers=["lanes", "intersections"]
)

# Get roadblock from route
roadblock = self._map_api.get_map_object(
    self._route[0],  # roadblock ID
    SemanticMapLayer.ROADBLOCK
)
```

### Trajectory Construction
```python
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

# Option 1: From EgoState list
trajectory = InterpolatedTrajectory(ego_states)

# Option 2: From Waypoint list
from nuplan.common.actor_state.waypoint import Waypoint
trajectory = InterpolatedTrajectory([state.waypoint for state in ego_states])
```

## Gotchas & Pitfalls

1. **Timing Constraints**: Planners typically have **0.1 seconds** to compute trajectory
   - Simulation fails if planner exceeds time budget
   - Use `generate_planner_report()` to profile your planner
   - Optimize hot paths! Map queries, collision checks, trajectory propagation

2. **Initialization vs Construction**: Two-phase setup is critical
   - `__init__()` receives planner parameters (from Hydra config)
   - `initialize()` receives scenario data (route, map, goal)
   - Don't query map in `__init__()` - it's not available yet!

3. **History Buffer Indexing**: `history.ego_states` includes current state
   - `history.ego_states[-1]` == `current_input.history.current_state[0]`
   - Be careful with off-by-one errors!

4. **Observation Type Mismatch**: Declare correct observation type
   - `observation_type()` must match what simulation provides
   - Mismatch causes runtime error
   - Most planners use `DetectionsTracks`

5. **Trajectory Horizon**: Must return meaningful trajectory length
   - Too short → controller has no future to track
   - Too long → wasted computation
   - Typical: 8-10 seconds @ 0.1s sampling = 80-100 waypoints

6. **Immutable Inputs**: `PlannerInput` and `PlannerInitialization` are frozen
   - Can't modify them! Read-only data
   - Create new objects if you need to transform data

7. **Thread Safety**: Planners may be called in parallel (multiple scenarios)
   - Don't use mutable class-level state!
   - Instance variables are safe (one instance per simulation)

8. **Route vs Goal**: Different concepts!
   - `route_roadblock_ids`: Sequence of roadblocks to follow (like GPS route)
   - `mission_goal`: Final destination pose (may be beyond scenario end)
   - Route guides intermediate navigation, goal is ultimate destination

9. **Map Coordinates**: All map queries use global frame
   - ego_state.center is in global UTM coordinates
   - Map returns global coordinates
   - Transform to ego frame if needed for planning

10. **SimplePlanner Velocity Clamping**: Note the max_velocity logic
    - If speed exceeds max, sets deceleration to exactly reach max
    - Common pattern for velocity-controlled planners

11. **Exception Handling**: `compute_trajectory()` wrapper catches exceptions
    - Still records timing even if planning fails
    - But exception still propagates - simulation will crash
    - Handle edge cases gracefully in `compute_planner_trajectory()`!

12. **`requires_scenario=True`**: Only for oracle planners
    - Gives access to full scenario object (including future)
    - Cheating for evaluation! Use only for debugging/baselines
    - Not allowed in competition submissions

## Test Coverage Notes

Test directory: `nuplan/planning/simulation/planner/test/`
- `test_simple_planner.py` - SimplePlanner unit tests
- `test_planner_report.py` - Runtime statistics
- Mock scenarios for planner testing
- See tests for edge cases and usage examples

## Related Documentation

### Parent Module
- `nuplan/planning/simulation/CLAUDE.md` - Simulation infrastructure overview

### Critical Dependencies
- `nuplan/common/actor_state/CLAUDE.md` - EgoState, TrackedObjects (planner inputs/outputs)
- `nuplan/common/maps/CLAUDE.md` - AbstractMap (map queries during planning)
- `nuplan/planning/simulation/trajectory/CLAUDE.md` - AbstractTrajectory (planner output format)
- `nuplan/planning/simulation/observation/CLAUDE.md` - Observation types (planner input format)
- `nuplan/planning/simulation/history/CLAUDE.md` - SimulationHistoryBuffer (past state access)

### Dependents
- `nuplan/planning/simulation/runner/CLAUDE.md` - Simulation loop that calls planners
- `nuplan/planning/simulation/controller/CLAUDE.md` - Controllers track planner trajectories
- `nuplan/planning/metrics/CLAUDE.md` - Metrics evaluate planner performance
- `nuplan/planning/training/CLAUDE.md` - ML training for learned planners

### Sibling Modules
- `nuplan/planning/simulation/callback/CLAUDE.md` - Callbacks can inspect planner output
- `nuplan/planning/simulation/predictor/CLAUDE.md` - Agent prediction (often used by planners)

---

**AIDEV-NOTE**: AbstractPlanner is THE interface to implement for autonomous driving. Everything else in nuPlan exists to support running and evaluating planners. Start here when learning nuPlan!

**AIDEV-NOTE**: SimplePlanner is the best reference implementation - under 100 lines, shows all required methods, uses standard components (KinematicBicycleModel, InterpolatedTrajectory).

**AIDEV-TODO**: Consider adding AbstractPlanner.validate() method for planner-specific input validation before compute_trajectory()

**AIDEV-QUESTION**: Why does compute_trajectory() record time even on exception? For debugging flaky planners that intermittently fail?
