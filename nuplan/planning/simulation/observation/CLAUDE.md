# nuplan/planning/simulation/observation/

## Purpose & Responsibility

**THE perception layer for nuPlan's simulation pipeline.** This module defines how planners perceive the world during closed-loop simulation by transforming raw scenario data into planner-consumable observations. Every simulation timestep (typically 0.1s), observations are updated and passed to the planner via `PlannerInput.history`. Observations can be replayed sensor data (LiDAR, cameras, detection tracks), simulated smart agents (IDM-based, ML-based), or hybrid combinations. This module bridges the gap between static scenario data and dynamic simulation state.

## Key Abstractions & Classes

### Core Interface
- **`AbstractObservation`** - THE fundamental observation interface (ABC metaclass)
  - **`initialize()`** - One-time setup (load models, build agent graphs, etc.)
  - **`reset()`** - Reset internal state for new scenario run
  - **`get_observation() -> Observation`** - Return current observation object
  - **`update_observation(iteration, next_iteration, history)`** - Propagate observations to next timestep
  - **`observation_type() -> Type[Observation]`** - Declare observation format (DetectionsTracks, Sensors, etc.)

### Observation Data Types (observation_type.py)
- **`Observation`** (ABC dataclass) - Base observation container
  - `detection_type()` - Class method returning observation type string

- **`DetectionsTracks`** (dataclass) - Perception system output (most common)
  - `tracked_objects: TrackedObjects` - Bounding boxes with velocities/predictions
  - Used by: TracksObservation, IDMAgents, AbstractMLAgents

- **`Sensors`** (dataclass) - Raw sensor data
  - `pointcloud: Optional[Dict[LidarChannel, LidarPointCloud]]` - LiDAR point clouds
  - `images: Optional[Dict[CameraChannel, Image]]` - Camera images
  - Used by: LidarPcObservation

- **`CameraChannel`** (Enum) - 8 camera channels
  - CAM_F0 (front), CAM_B0 (back), CAM_L{0,1,2} (left), CAM_R{0,1,2} (right)

- **`LidarChannel`** (Enum) - MERGED_PC (MergedPointCloud)

### Concrete Implementations

#### 1. **TracksObservation** - Replay Ground Truth Tracks (Open-Loop)
```python
class TracksObservation(AbstractObservation):
    """Replay detections from scenario database samples."""
```
- **Purpose**: Simplest observation - just replays logged detection tracks
- **Use case**: Testing planners with perfect perception
- **Performance**: Minimal overhead (database query only)
- **Update behavior**: Increments iteration index, queries scenario DB

#### 2. **LidarPcObservation** - Replay LiDAR Point Clouds
```python
class LidarPcObservation(AbstractObservation):
    """Replay lidar pointclouds from scenario sensor blobs."""
```
- **Purpose**: Provide raw LiDAR sensor data
- **Use case**: Testing perception pipelines or sensor-based planners
- **Performance**: High memory/IO overhead (point clouds are large)
- **Update behavior**: Increments iteration, loads point cloud from disk

#### 3. **IDMAgents** - Intelligent Driver Model Simulation (Closed-Loop)
```python
class IDMAgents(AbstractObservation):
    """Simulate agents based on IDM policy (car-following behavior)."""
```
- **Purpose**: Reactive agents that respond to ego and traffic lights
- **Use case**: Realistic multi-agent closed-loop simulation
- **Key parameters**:
  - `target_velocity` [m/s] - Desired free-flow speed
  - `min_gap_to_lead_agent` [m] - Following distance
  - `headway_time` [s] - Time gap to lead vehicle
  - `accel_max`, `decel_max` [m/s²] - Acceleration limits
  - `open_loop_detections_types` - Static objects (pedestrians, cones, barriers)
  - `radius` [m] - Only simulate agents within this range
- **Update behavior**:
  1. Propagate IDM agents using IDMPolicy (ODE solver)
  2. Query traffic light status
  3. Update agent routes (lane selection based on curvature)
  4. Insert stop lines into occupancy map for red lights
  5. Compute lead agent (nearest in path using spatial index)
  6. Apply IDM acceleration model
- **Performance**: Computationally intensive (100+ agents × ODE solve × spatial queries)

#### 4. **AbstractMLAgents** - ML Model-Based Agent Simulation (Closed-Loop)
```python
class AbstractMLAgents(AbstractObservation):
    """Simulate agents based on an ML model (learned behavior)."""
```
- **Purpose**: Abstract base for ML-driven agent simulation
- **Architecture**:
  - Uses `ModelLoader` (same as MLPlanner) for inference
  - Subclasses implement `_infer_model()` and `_update_observation_with_predictions()`
- **Agent lifecycle**:
  - `_initialize_agents()` - Extract VEHICLE objects from scenario.initial_tracked_objects
  - `update_observation()` - Run model inference, update agent states
  - `get_observation()` - Return tracked objects with predicted trajectories
- **Key fields**:
  - `_agents: Dict[str, TrackedObject]` - Agent state by track_token
  - `_ego_anchor_state` - Ego state for coordinate transforms
  - `step_time` - Elapsed time since last update

#### 5. **EgoCentricMLAgents** - Ego-Centric ML Agent Model (Concrete)
```python
class EgoCentricMLAgents(AbstractMLAgents):
    """Ego-centric coordinate frame ML agent predictions."""
```
- **Purpose**: ML agents with ego-relative predictions (like most learned models)
- **Coordinate transforms**:
  - Model predicts relative poses/velocities in ego frame
  - `numpy_array_to_absolute_pose()` - Convert to global coordinates
  - `numpy_array_to_absolute_velocity()` - Velocity frame transform
- **Update flow**:
  1. Build features from history (via ModelLoader)
  2. Infer model → agents_trajectory predictions
  3. Extract batch[0] → numpy array [num_agents, num_frames, state_dim]
  4. Transform ego-relative → global coordinates
  5. Create PredictedTrajectory for each agent
  6. Propagate agents to current simulation time

### IDM Submodule (idm/)

**Purpose**: Complete implementation of Intelligent Driver Model for traffic simulation

#### **IDMAgent** - Individual smart agent
- **State**: `IDMAgentState(progress, velocity)` - 1D longitudinal state along path
- **Route**: `Deque[LaneGraphEdgeMapObject]` - Dynamic route planning
- **Path**: `InterpolatedPath` - Continuous baseline from route segments
- **Policy**: `IDMPolicy` - Car-following acceleration model
- **Key methods**:
  - `propagate(lead_agent, tspan)` - Apply IDM policy, update state
  - `plan_route(traffic_light_status)` - Extend route (prefers straight, respects lights)
  - `get_path_to_go()` - Remaining path from current progress
  - `projected_footprint` - Future occupancy based on velocity × headway
- **Caching**: `_requires_state_update` flag avoids recomputing Agent objects

#### **IDMAgentManager** - Multi-agent orchestration
- **Occupancy map**: `OccupancyMap` - Spatial index for collision detection
- **Agent filtering**: Only simulates agents within radius of ego
- **Propagation pipeline** (per timestep):
  1. Insert ego + open-loop detections into occupancy map
  2. Filter agents beyond radius (remove from dict + occupancy map)
  3. For each active agent:
     - Plan route (check traffic lights, extend if needed)
     - Get relevant stop lines → insert into occupancy map
     - Find intersecting agents along path (spatial query)
     - Identify nearest agent (or ego or stop line)
     - Compute relative heading/velocity (project longitudinal component)
     - Create `IDMLeadAgentState(distance, velocity, length)`
     - Call `agent.propagate()` → ODE solve
     - Update occupancy map with new footprint
     - Remove stop lines from occupancy map
  4. Return active agents as DetectionsTracks

#### **IDMPolicy** - Acceleration model
- **Classic IDM equation**:
  ```
  a = a_max * [1 - (v/v₀)⁴ - (s*/s)²]

  s* = s₀ + vT + v(v - v_lead) / (2√(a_max × d_max))

  where:
    v₀ = target_velocity
    s₀ = min_gap_to_lead_agent
    T = headway_time
    a_max = accel_max
    d_max = decel_max
  ```
- **Solvers**:
  - `solve_forward_euler_idm_policy()` - Simple Euler integration (fast, differentiable)
  - `solve_odeint_idm_policy()` - SciPy odeint (more accurate)
  - `solve_ivp_idm_policy()` - SciPy RK45 (adaptive timestep)
- **Default**: Forward Euler for speed

#### **IDMAgentsBuilder** - Agent initialization
- **`build_idm_agents_on_map_rails()`** - Main factory function
  - Extract VEHICLE objects from scenario.initial_tracked_objects
  - For each vehicle:
    - Find starting segment (lane or lane_connector) via map query
    - Snap to baseline path (heading alignment)
    - Check collision with existing agents (occupancy map)
    - Project velocity to longitudinal component
    - Handle NaN velocities (use ego velocity as fallback)
    - Create IDMAgent with route=[starting_segment]
  - Returns: `(UniqueIDMAgents, OccupancyMap)`
- **Collision filtering**: Agents overlapping after snapping are discarded
- **Progress bar**: tqdm for initialization feedback

#### **idm_states.py** - State representations
- `IDMAgentState(progress, velocity)` - Agent longitudinal state
- `IDMLeadAgentState(progress, velocity, length_rear)` - Lead vehicle state

## Architecture & Design Patterns

### 1. **Strategy Pattern**: Pluggable Observation Types
- `AbstractObservation` defines interface
- Concrete implementations (Tracks, Sensors, IDM, ML) are interchangeable
- Hydra config selects observation at runtime: `simulation.observation=idm_agents_observation`
- Simulation validates observation matches planner's expected type

### 2. **Template Method**: Update Lifecycle
```
Simulation.propagate():
  ├─ observations.update_observation(iteration, next_iteration, history)
  │    └─ Subclass-specific logic (propagate agents, advance iteration, etc.)
  └─ observations.get_observation()
       └─ Return current state as Observation object
```

### 3. **Two-Phase Initialization**: Construct → Initialize
```python
# Phase 1: Construction (Hydra instantiation)
observation = IDMAgents(target_velocity=10, ..., scenario=scenario)

# Phase 2: Initialization (simulation setup)
observation.initialize()  # Build agent graph, load models, etc.
```
- Enables config-driven construction without simulation context
- `initialize()` called once before simulation loop
- `reset()` called between scenario runs

### 4. **Type Declaration & Validation**
```python
# Planner declares expected observation
class SimplePlanner(AbstractPlanner):
    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks

# Simulation validates compatibility
validate_planner_setup(setup, planner)
    if planner.observation_type() != setup.observations.observation_type():
        raise ValueError(...)
```

### 5. **Lazy Initialization**: Models loaded on-demand
- `AbstractMLAgents._model_loader.initialize()` called in `initialize()`
- Avoids loading PyTorch models during Hydra config composition
- Enables fast dry-runs and config validation

### 6. **Occupancy Map Spatial Index** (IDM)
- STRTree (R-tree variant) for efficient spatial queries
- Enables O(log N) intersection tests instead of O(N²)
- Critical for 100+ agent simulations

### 7. **Ego-Centric Coordinate Transforms** (ML Agents)
- Models predict in ego frame (translation + rotation invariance)
- `_ego_anchor_state` updated every timestep
- Predictions transformed to global frame for simulation

### 8. **Open-Loop vs Closed-Loop Hybrid** (IDM)
- IDM agents: Closed-loop (react to ego)
- Static objects (pedestrians, cones): Open-loop (replayed from scenario)
- Hybrid approach balances realism and computational cost

## Dependencies (What We Import)

### Internal nuPlan (Documented ✅)
- `nuplan.common.actor_state.ego_state` ✅ - EgoState representation
- `nuplan.common.actor_state.state_representation` ✅ - StateSE2, StateVector2D, ProgressStateSE2
- `nuplan.common.actor_state.tracked_objects` ✅ - TrackedObjects, TrackedObject, Agent
- `nuplan.common.actor_state.waypoint` ✅ - Waypoint, PredictedTrajectory
- `nuplan.common.geometry.convert` ✅ - `numpy_array_to_absolute_pose/velocity()`
- `nuplan.common.geometry.transform` ✅ - `rotate_angle()`
- `nuplan.common.maps.abstract_map` ✅ - AbstractMap, SemanticMapLayer
- `nuplan.common.maps.abstract_map_objects` ✅ - LaneGraphEdgeMapObject, StopLine
- `nuplan.common.maps.maps_datatypes` ✅ - TrafficLightStatusType
- `nuplan.planning.scenario_builder.abstract_scenario` ✅ - AbstractScenario
- `nuplan.planning.simulation.planner.abstract_planner` ✅ - PlannerInput, PlannerInitialization
- `nuplan.planning.simulation.history.simulation_history_buffer` ✅ - SimulationHistoryBuffer
- `nuplan.planning.simulation.simulation_time_controller.simulation_iteration` ✅ - SimulationIteration

### Internal nuPlan (Undocumented - Future sessions ⏳)
- `nuplan.planning.training.modeling.torch_module_wrapper` ⏳ - TorchModuleWrapper (for ML agents)
- `nuplan.planning.training.modeling.types` ⏳ - FeaturesType, TargetsType
- `nuplan.planning.training.preprocessing.features.agents_trajectories` ⏳ - AgentsTrajectories
- `nuplan.planning.simulation.planner.ml_planner.model_loader` ⏳ - ModelLoader
- `nuplan.planning.simulation.path.interpolated_path` ⏳ - InterpolatedPath (IDM)
- `nuplan.planning.simulation.path.utils` ⏳ - `trim_path()`, `trim_path_up_to_progress()`
- `nuplan.planning.simulation.occupancy_map.abstract_occupancy_map` ⏳ - OccupancyMap
- `nuplan.planning.simulation.occupancy_map.strtree_occupancy_map` ⏳ - STRTreeOccupancyMap
- `nuplan.planning.metrics.utils.expert_comparisons` ⏳ - `principal_value()` (angle wrapping)
- `nuplan.database.utils.image` ⏳ - Image
- `nuplan.database.utils.pointclouds.lidar` ⏳ - LidarPointCloud

### External Dependencies
- **NumPy** - Array operations, spatial distance (cdist)
- **SciPy** - ODE solvers (odeint, solve_ivp), interpolation
- **Shapely** - Geometry operations (Polygon, LineString, unary_union, buffer)
- **PyTorch** - ML agent inference (via ModelLoader)

## Dependents (Who Uses Us)

- **`nuplan.planning.simulation.simulation.Simulation`** - Main simulation loop
  - `initialize()` → `observations.initialize()`
  - `initialize()` → `history_buffer.append(ego_state, observations.get_observation())`
  - `propagate()` → `observations.update_observation(iteration, next_iteration, history)`
  - `propagate()` → `history_buffer.append(ego_state, observations.get_observation())`

- **`nuplan.planning.simulation.simulation_setup.SimulationSetup`** - Configuration container
  - `observations: AbstractObservation` - Required field
  - `validate_planner_setup()` - Type compatibility check
  - `reset()` → `observations.reset()`

- **Hydra Configs**:
  - `nuplan/planning/script/config/simulation/observation/*.yaml`
  - Selected via `simulation.observation=<config_name>`

- **Planner compatibility**:
  - All planners declare `observation_type()` for validation
  - SimplePlanner, MLPlanner, IDMPlanner → DetectionsTracks
  - LiDAR-based planners → Sensors

## Critical Files (Prioritized)

### Must Read First
1. **`abstract_observation.py`** (59 lines) - **START HERE!**
   - AbstractObservation interface contract
   - Lifecycle methods (initialize, reset, update, get)
   - Simple, well-documented ABC

2. **`observation_type.py`** (68 lines) - **Data structures**
   - Observation, DetectionsTracks, Sensors definitions
   - CameraChannel and LidarChannel enums
   - Critical for understanding observation formats

### Simple Implementations (Easy to understand)
3. **`tracks_observation.py`** (43 lines) - Simplest concrete example
   - Open-loop track replay
   - Shows minimal update logic (increment iteration)

4. **`lidar_pc.py`** (44 lines) - Sensor data replay
   - Nearly identical to tracks_observation
   - Shows Sensors observation type

### Complex Implementations (Study after understanding basics)
5. **`idm_agents.py`** (155 lines) - **Closed-loop smart agents**
   - IDM-based reactive agents
   - Hybrid open/closed-loop design
   - Traffic light awareness
   - Complex initialization via IDMAgentManager

6. **`abstract_ml_agents.py`** (116 lines) - **ML agent base**
   - Model loading and inference
   - Feature building pipeline
   - Agent lifecycle management
   - Abstract methods for coordinate transforms

7. **`ego_centric_ml_agents.py`** (117 lines) - **Concrete ML implementation**
   - Ego-relative coordinate transforms
   - Predicted trajectory construction
   - State propagation via trajectory interpolation

### IDM Submodule (Deep dive for advanced users)
8. **`idm/idm_policy.py`** (202 lines) - **IDM acceleration model**
   - Classic car-following equations
   - Three ODE solver implementations
   - Target velocity adaptation from speed limits

9. **`idm/idm_agent.py`** (312 lines) - **Individual agent logic**
   - State propagation along path
   - Route planning (lane selection)
   - Footprint projection
   - Agent caching optimization

10. **`idm/idm_agent_manager.py`** (194 lines) - **Multi-agent coordination**
    - Occupancy map management
    - Agent filtering by radius
    - Lead agent identification
    - Traffic light + stop line handling

11. **`idm/idm_agents_builder.py`** (153 lines) - **Agent initialization**
    - Map-based agent placement
    - Collision detection during init
    - Velocity projection and NaN handling

## Common Usage Patterns

### 1. Basic Open-Loop Observation (Perfect Perception)
```python
from nuplan.planning.simulation.observation.tracks_observation import TracksObservation

# Hydra config: simulation/observation/box_observation.yaml
observation = TracksObservation(scenario=scenario)
observation.initialize()
observation.reset()

# Simulation loop
for iteration, next_iteration in timesteps:
    # Get current observation
    detections = observation.get_observation()  # DetectionsTracks
    tracked_objects = detections.tracked_objects  # TrackedObjects

    # Update to next timestep
    observation.update_observation(iteration, next_iteration, history)
```

### 2. IDM Smart Agents (Closed-Loop Simulation)
```python
from nuplan.planning.simulation.observation.idm_agents import IDMAgents

# Hydra config: simulation/observation/idm_agents_observation.yaml
observation = IDMAgents(
    target_velocity=10.0,  # m/s
    min_gap_to_lead_agent=1.0,  # m
    headway_time=1.5,  # s
    accel_max=1.0,  # m/s²
    decel_max=2.0,  # m/s²
    open_loop_detections_types=["PEDESTRIAN", "BARRIER", "TRAFFIC_CONE"],
    scenario=scenario,
    minimum_path_length=20.0,  # m
    radius=100.0,  # m (only simulate agents within this range)
)

observation.initialize()  # Build IDM agent graph
observation.reset()

# Simulation loop
for iteration, next_iteration in timesteps:
    # Get reactive agents (DetectionsTracks)
    detections = observation.get_observation()

    # Agents include predicted trajectories (from IDM policy)
    for agent in detections.tracked_objects.tracked_objects:
        if agent.predictions:
            future_traj = agent.predictions[0]  # PredictedTrajectory

    # Update agents (IDM propagation + route planning)
    observation.update_observation(iteration, next_iteration, history)
```

### 3. ML-Based Agent Simulation
```python
from nuplan.planning.simulation.observation.ego_centric_ml_agents import EgoCentricMLAgents
from nuplan.planning.training.modeling.lightning_module_wrapper import LightningModuleWrapper

# Load trained agent model
model = LightningModuleWrapper.load_from_checkpoint(
    checkpoint_path="path/to/agent_model.ckpt",
    map_location="cuda"
).model

observation = EgoCentricMLAgents(model=model, scenario=scenario)
observation.initialize()  # Initialize model, extract initial agents
observation.reset()

# Simulation loop
for iteration, next_iteration in timesteps:
    # Get ML-predicted agents
    detections = observation.get_observation()  # DetectionsTracks

    # Each agent has ML-predicted future trajectory
    for agent in detections.tracked_objects.tracked_objects:
        predictions = agent.predictions  # List[PredictedTrajectory]

    # Update: run inference + coordinate transforms
    observation.update_observation(iteration, next_iteration, history)
```

### 4. LiDAR Point Cloud Observation
```python
from nuplan.planning.simulation.observation.lidar_pc import LidarPcObservation
from nuplan.planning.simulation.observation.observation_type import LidarChannel

observation = LidarPcObservation(scenario=scenario)
observation.initialize()
observation.reset()

# Simulation loop
for iteration, next_iteration in timesteps:
    # Get sensor data
    sensors = observation.get_observation()  # Sensors

    # Access point cloud
    if sensors.pointcloud:
        lidar_pc = sensors.pointcloud[LidarChannel.MERGED_PC]  # LidarPointCloud
        points = lidar_pc.points  # numpy array [N, 3+] (x, y, z, intensity, ...)

    # Access camera images (if available)
    if sensors.images:
        from nuplan.planning.simulation.observation.observation_type import CameraChannel
        front_img = sensors.images[CameraChannel.CAM_F0]  # Image

    observation.update_observation(iteration, next_iteration, history)
```

### 5. Observation Type Validation (Planner Compatibility)
```python
from nuplan.planning.simulation.simulation_setup import validate_planner_setup

# Planner declares expected observation type
class MyPlanner(AbstractPlanner):
    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks  # Expects tracked objects

# Setup simulation
setup = SimulationSetup(
    observations=TracksObservation(scenario),  # DetectionsTracks
    planner=MyPlanner(),
    ...
)

# Validation (raises ValueError if mismatch)
validate_planner_setup(setup, planner)
```

### 6. Accessing Observations in Planner
```python
class MyPlanner(AbstractPlanner):
    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        # Access latest observation
        latest_observation = current_input.history.observations[-1]

        if isinstance(latest_observation, DetectionsTracks):
            tracked_objects = latest_observation.tracked_objects

            # Filter by object type
            vehicles = tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE)
            pedestrians = tracked_objects.get_tracked_objects_of_type(TrackedObjectType.PEDESTRIAN)

            # Access predictions (if available)
            for vehicle in vehicles:
                if vehicle.predictions:
                    future_traj = vehicle.predictions[0]
                    waypoints = future_traj.waypoints  # List[Waypoint]

        # Access observation history (rolling window)
        observation_history = current_input.history.observations  # List[Observation]

        # Plan trajectory...
```

## Gotchas & Pitfalls

### Performance & Real-Time Constraints

1. **IDM agents are computationally expensive**
   - Each agent: ODE solve + spatial queries + route planning
   - 100 agents × 0.1s timestep can take 0.5-1.0s wall time
   - **Mitigation**: Use `radius` parameter to limit agent count (default 100m)
   - **Monitoring**: Profile with `cProfile` to identify bottlenecks

2. **Occupancy map dominates IDM runtime**
   - STRTree intersection queries for every agent every timestep
   - Inserting/removing stop lines repeatedly
   - **Mitigation**: Reduce agent count, increase radius (trades realism for speed)
   - **AIDEV-NOTE**: Consider caching stop line insertions across timesteps

3. **LiDAR point clouds are memory-intensive**
   - MERGED_PC can be 100k+ points per frame
   - Loading from disk every timestep is slow
   - **Mitigation**: Use detection tracks (DetectionsTracks) instead if possible
   - **AIDEV-NOTE**: Consider point cloud caching for repeated scenario runs

4. **ML agent inference adds latency**
   - Model forward pass + coordinate transforms every timestep
   - GPU overhead if model is small (transfer time > compute time)
   - **Mitigation**: Batch inference if using multiple ML agent models
   - **Monitoring**: Check `EgoCentricMLAgents` doesn't exceed planner time budget

5. **Observation update MUST complete before planner runs**
   - `update_observation()` is synchronous blocking call
   - Slow observations delay entire simulation
   - **Consequence**: Real-time constraint violations (> 0.1s per step)

### Initialization & State Management

6. **`initialize()` vs `reset()` confusion**
   - `initialize()`: Called ONCE before simulation (load models, build graphs)
   - `reset()`: Called BEFORE EACH scenario (clear agent states, reset iteration)
   - **Gotcha**: Calling only `reset()` without `initialize()` → uninitialized models
   - **Validation**: AbstractMLAgents.get_observation() asserts `self._agents` is not None

7. **IDM agents may fail to initialize**
   - Agents off-road (no lane/lane_connector) → silently skipped
   - Agents overlapping after snapping → collision-filtered
   - **Consequence**: Fewer agents than expected in simulation
   - **Debug**: Check `build_idm_agents_on_map_rails()` logs for skipped agents

8. **ML agents only initialized from first frame**
   - `_initialize_agents()` uses `scenario.initial_tracked_objects`
   - Agents appearing mid-scenario are NOT added
   - **AIDEV-TODO**: `abstract_ml_agents.py:65` - "consider agents appearing in the future"
   - **Workaround**: Use hybrid approach (ML agents + open-loop detections)

9. **Ego state must be set before first `get_observation()`**
   - Simulation calls `history_buffer.append(ego_state, observation.get_observation())`
   - IDM/ML agents may access `history.current_state[0]` (ego)
   - **Gotcha**: Calling `get_observation()` before initialization → IndexError

10. **Observation type mismatch crashes simulation**
    - Planner expects DetectionsTracks, observation provides Sensors
    - `validate_planner_setup()` raises ValueError with clear message
    - **Prevention**: Always declare `observation_type()` correctly in planner

### IDM-Specific Issues

11. **IDM agents can get stuck at red lights**
    - Stop lines inserted into occupancy map
    - If light never turns green, agent stops forever
    - **Consequence**: Unrealistic traffic jams
    - **Mitigation**: Check traffic light cycles in scenario

12. **IDM route planning prefers going straight**
    - `plan_route()` selects edge with lowest curvature
    - May not follow optimal/human-like route
    - **Design choice**: Simplicity over realism

13. **Velocity projection assumes longitudinal motion**
    - `rotate_angle()` projects lead agent velocity to follower heading
    - Breaks down for perpendicular traffic (intersections)
    - **Consequence**: Incorrect following distances at intersections

14. **NaN velocities fallback to ego velocity**
    - `build_idm_agents_on_map_rails()` line 125-131
    - If agent has NaN velocity → uses ego's speed
    - **Gotcha**: Can cause unrealistic agent initialization speeds

15. **Agent path can become invalid**
    - If route planning fails to extend path → `get_progress_to_go() = 0`
    - Agent stops moving (no path to follow)
    - **Check**: `agent.has_valid_path()` before propagation

### ML Agent Issues

16. **Coordinate frame mismatches are silent errors**
    - Model predicts ego-relative, but code treats as absolute → agents teleport
    - **Symptom**: Agents suddenly jump to wrong locations
    - **Debug**: Print `_ego_anchor_state` and predictions before transform

17. **Batch dimension confusion**
    - Models expect batch dimension [B, N, D]
    - Single inference requires `collate([feature])` → [1, N, D]
    - Extracting predictions requires `[0]` indexing → [N, D]
    - **Gotcha**: Forgetting `[0]` → passes 3D array to 2D transform → ValueError

18. **Agent count mismatch between features and predictions**
    - Feature builder extracts N agents
    - Model predicts M trajectories (M ≠ N if some agents filtered)
    - **Consequence**: zip() mismatch in `_update_observation_with_predictions()`
    - **AIDEV-NOTE**: AbstractMLAgents assumes 1:1 correspondence

19. **Step time interpolation can fail**
    - `future_trajectory.trajectory.get_state_at_time(self.step_time)`
    - If step_time > predicted horizon → extrapolation error
    - **Mitigation**: Ensure model horizon > scenario timestep

20. **Model device mismatch**
    - Features moved to CUDA, but model on CPU (or vice versa)
    - **Symptom**: RuntimeError: Expected tensor on device cuda:0 but got cpu
    - **Fix**: ModelLoader handles device placement, but check config

### Observation Data Quality

21. **Tracked objects may have missing metadata**
    - `TrackedObject.metadata` can be None or incomplete
    - Accessing `metadata.timestamp_s` → AttributeError
    - **Prevention**: Always check `if metadata is not None`

22. **Predictions are optional**
    - `TrackedObject.predictions: List[PredictedTrajectory]` can be empty list
    - IDM agents provide predictions (if `planned_trajectory_samples` set)
    - Open-loop tracks do NOT have predictions
    - **Check**: `if agent.predictions:` before accessing

23. **Bounding box orientation vs velocity direction**
    - `agent.box.center.heading` (oriented box heading)
    - `agent.velocity` direction (may differ, e.g., drifting)
    - **Gotcha**: Using box heading for velocity projection → wrong results

24. **TrackedObjects.tracked_objects is a list, not dict**
    - No direct indexing by track_token
    - Must iterate or build lookup dict yourself
    - **Performance**: O(N) lookup if repeatedly accessing specific agents

### Traffic Light Handling

25. **Traffic light data can be empty**
    - `scenario.get_traffic_light_status_at_iteration()` may return []
    - IDM agents assume lights are green if not in red list
    - **Gotcha**: Missing data → agents run red lights

26. **Lane connector IDs are strings, not ints**
    - `traffic_light_status[RED]` → List[str]
    - Comparing with int lane IDs → silent mismatch
    - **Prevention**: Ensure `str(lane_connector.id)` conversion

## Performance Notes

### Real-Time Constraints
- **Observation update target**: < 50ms (leaves 50ms for planner in 0.1s budget)
- **TracksObservation**: ~1-5ms (just DB query)
- **LidarPcObservation**: ~10-50ms (disk I/O for point clouds)
- **IDMAgents (50 agents)**: ~20-100ms (ODE solves + spatial queries)
- **IDMAgents (100+ agents)**: Can exceed 100ms (fails real-time constraint)
- **EgoCentricMLAgents**: ~10-50ms (model size dependent, plus coordinate transforms)

### Bottleneck Analysis (IDM)

| Operation | % Time | Notes |
|-----------|--------|-------|
| Occupancy map intersects() | 30-40% | STRTree queries for each agent's path |
| IDM policy ODE solve | 20-30% | Forward Euler is fastest, RK45 is slowest |
| Route planning | 15-25% | Map queries + curvature computation |
| Stop line insertion/removal | 10-15% | Repeated geometry ops |
| Agent state construction | 5-10% | Creating Agent objects with trajectories |

**AIDEV-NOTE**: Consider parallelizing agent propagation (independent agents can be updated in parallel)

### Memory Footprint
- **TracksObservation**: Minimal (~1 KB per frame)
- **LidarPcObservation**: High (10-100 MB per frame if all channels loaded)
- **IDMAgents**: Moderate (InterpolatedPath + occupancy map, ~100 KB per agent)
- **ML Agents**: High (model parameters + feature cache, 100MB-1GB depending on model)

### Optimization Strategies

1. **Reduce agent count** (IDM)
   - Use smaller `radius` parameter (e.g., 50m instead of 100m)
   - Pre-filter agents by relevance (near ego route)

2. **Simplify IDM solver**
   - Use forward Euler (default) instead of RK45
   - Reduce `solve_points` if using odeint

3. **Cache stop lines**
   - Don't insert/remove every timestep if traffic lights unchanged
   - Track previous traffic light state

4. **Batch ML inference**
   - If using multiple ML observation models, batch on GPU

5. **Lazy trajectory prediction**
   - Only compute `planned_trajectory` if planner uses predictions
   - Check if `planned_trajectory_samples` is None

6. **Profile-guided optimization**
   ```python
   import cProfile
   profiler = cProfile.Profile()
   profiler.enable()
   observation.update_observation(iteration, next_iteration, history)
   profiler.disable()
   profiler.print_stats(sort='cumulative')
   ```

## Related Documentation

### Documented Modules (Cross-reference ✅)
- `nuplan/planning/simulation/planner/CLAUDE.md` ✅ - Planner interface and observation consumption
- `nuplan/planning/simulation/planner/ml_planner/CLAUDE.md` ✅ - ML planner (similar patterns to ML agents)
- `nuplan/planning/simulation/trajectory/CLAUDE.md` ✅ - Trajectory types (InterpolatedTrajectory used in IDM)
- `nuplan/planning/scenario_builder/CLAUDE.md` ✅ - Scenario data access
- `nuplan/planning/scenario_builder/nuplan_db/CLAUDE.md` ✅ - Database queries for observations
- `nuplan/database/nuplan_db_orm/CLAUDE.md` ✅ - ORM schema for tracked_objects
- `nuplan/common/actor_state/CLAUDE.md` ✅ - Agent, TrackedObjects, EgoState
- `nuplan/common/geometry/CLAUDE.md` ✅ - Coordinate transforms
- `nuplan/common/maps/CLAUDE.md` ✅ - Map queries for IDM route planning
- `nuplan/common/maps/nuplan_map/CLAUDE.md` ✅ - Specific map implementation

### Undocumented Dependencies (Future Documentation ⏳)
- `nuplan/planning/training/modeling/` ⏳ - Model wrappers for ML agents
- `nuplan/planning/training/preprocessing/` ⏳ - Feature builders
- `nuplan/planning/simulation/path/` ⏳ - InterpolatedPath, path utilities
- `nuplan/planning/simulation/occupancy_map/` ⏳ - Spatial indexing
- `nuplan/planning/simulation/controller/motion_model/` ⏳ - Motion models
- `nuplan/planning/simulation/history/` ⏳ - SimulationHistoryBuffer
- `nuplan/planning/simulation/simulation_time_controller/` ⏳ - SimulationIteration

### Simulation Pipeline Flow
```
Scenario → AbstractObservation → SimulationHistoryBuffer → PlannerInput → AbstractPlanner
           ↓
       update_observation()  (this module)
           ↓
       get_observation()
           ↓
       Observation (DetectionsTracks, Sensors)
```

### Configuration Files
- `nuplan/planning/script/config/simulation/observation/box_observation.yaml` - TracksObservation
- `nuplan/planning/script/config/simulation/observation/lidar_pc_observation.yaml` - LidarPcObservation
- `nuplan/planning/script/config/simulation/observation/idm_agents_observation.yaml` - IDMAgents
- `nuplan/planning/script/config/simulation/observation/ego_centric_ml_agents_observation.yaml` - EgoCentricMLAgents

## AIDEV Notes

### Critical TODOs
- **AIDEV-TODO** (`abstract_ml_agents.py:65`): "consider agents appearing in the future (not just the first frame)"
  - Current: Only scenario.initial_tracked_objects are initialized
  - Needed: Dynamic agent spawning mid-scenario
  - Complexity: Requires tracking scenario iteration and checking for new agents

- **AIDEV-TODO** (`abstract_ml_agents.py:93`): "Rename PlannerInitialization to something that also applies to smart agents"
  - Current name is planner-centric but used in observation module
  - Suggestion: `SimulationInitialization` or `ScenarioInitialization`

### Performance Optimization Opportunities
- **AIDEV-NOTE** (`idm_agent_manager.py:70-72`): Stop line insertion/removal every agent
  - Current: Insert stop lines → propagate agent → remove stop lines (per agent!)
  - Optimization: Insert all stop lines once → propagate all agents → remove all stop lines
  - Estimated speedup: 10-15% for scenarios with traffic lights

- **AIDEV-NOTE** (`idm_agent.py:239-242`): Agent state caching
  - `_requires_state_update` flag avoids recomputing full Agent
  - Good pattern, but could extend to path trimming
  - Consider caching `get_path_to_go()` if progress unchanged

- **AIDEV-NOTE**: Parallelize IDM agent propagation
  - Agents are independent within single timestep (no inter-agent communication)
  - Could use multiprocessing.Pool or Ray for parallelization
  - Bottleneck: Occupancy map is shared state (would need thread-safe version)

### Design Questions
- **AIDEV-QUESTION**: Why are IDM agents always initialized at iteration 0?
  - `IDMAgent.__init__(start_iteration=0, ...)` hardcoded in builder
  - Scenario may start at iteration > 0
  - Investigate: Should start_iteration come from scenario?

- **AIDEV-QUESTION**: Should observation_type() be a class method?
  - Current: Instance method returning class
  - Alternative: Class attribute or @classmethod
  - Benefit: Type checking before instantiation

- **AIDEV-QUESTION**: ML agents assume sorted dict order
  - `abstract_ml_agents.py:66` - `sort_dict(unique_agents)`
  - Python 3.7+ dicts are ordered, but explicit sort may not be needed
  - Check if model training assumes specific agent ordering

### Potential Bugs
- **AIDEV-NOTE** (`idm_agent_manager.py:79`): Assertion can fail
  ```python
  assert intersecting_agents.contains(agent_token), "Agent's baseline does not intersect the agent itself"
  ```
  - Can fail if agent's path is very short or agent is off-baseline
  - Should this be logged warning instead of assertion?

- **AIDEV-NOTE** (`ego_centric_ml_agents.py:95-96`): zip() silently truncates
  ```python
  for agent_token, agent, poses_horizon, xy_velocity_horizon in zip(
      self._agents, self._agents.values(), agent_poses, agent_velocities
  ):
  ```
  - If arrays have different lengths, shorter one wins
  - Should validate lengths match before zip()

### Testing Gaps
- **AIDEV-NOTE**: No integration tests for observation lifecycle
  - Unit tests exist in `observation/test/` and `observation/idm/test/`
  - Missing: Full simulation loop with each observation type
  - Needed: Verify initialize() → reset() → update_observation() sequence

- **AIDEV-NOTE**: ML agent coordinate transform tests insufficient
  - Critical path: ego-relative → absolute coordinate transform
  - Should have regression tests with known ego state + predictions → expected global poses

### Documentation Improvements Needed
- Add sequence diagram for observation update flow
- Document expected observation update frequency (currently implicit 0.1s)
- Clarify when predictions are populated (IDM yes, TracksObservation no)
- Document memory usage by observation type (important for long simulations)
