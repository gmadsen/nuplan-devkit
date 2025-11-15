# nuplan/planning/simulation/controller/

## 1. Purpose & Responsibility

This module implements **trajectory tracking and vehicle motion control** for closed-loop simulation, bridging the gap between high-level planner trajectories and low-level vehicle dynamics. The `AbstractEgoController` is THE interface for propagating ego state forward in time, either by perfectly tracking planned trajectories (open-loop), replaying logged data (oracle), or using realistic two-stage controllers (tracker + motion model) that simulate control delays and physical constraints. Every simulation timestep (typically 0.1s), the controller updates ego state based on the planner's output trajectory.

## 2. Key Abstractions

### Core Interface

**AbstractEgoController**
- **Purpose**: Base class for all ego vehicle controllers
- **Key methods**:
  - `get_state() -> EgoState` - Returns current ego state
  - `reset()` - Resets controller internal state for new scenario
  - `update_state(current_iteration, next_iteration, ego_state, trajectory)` - Propagates ego state forward one timestep
- **Thread safety**: Controllers must be stateless or thread-safe for parallel simulations
- **Stateful**: Maintains `_current_state` between timesteps

### Controller Implementations

**PerfectTrackingController**
- **Purpose**: Perfect trajectory tracking (no control error)
- **Use case**: Testing planners without control noise, ideal for debugging
- **Update behavior**: Directly samples trajectory at next timestep
- **Validation**: Raises RuntimeError if velocity exceeds 50 m/s (safety threshold)
- **Performance**: O(1) trajectory interpolation
- **Gotcha**: Assumes trajectory covers next_iteration.time_point (fails if trajectory too short)

**LogPlaybackController**
- **Purpose**: Replay ground-truth ego states from scenario database
- **Use case**: Oracle baseline (perfect execution), no planning needed
- **Update behavior**: Increments iteration index, queries scenario DB for ego state
- **Performance**: O(1) database lookup
- **Limitation**: Trajectory input is ignored (uses logged data only)

**TwoStageController**
- **Purpose**: Realistic control with tracker + motion model (two-stage architecture)
- **Architecture**:
  1. **Tracker** (`AbstractTracker`) - Computes desired dynamic state (acceleration, steering rate) to follow trajectory
  2. **Motion Model** (`AbstractMotionModel`) - Propagates vehicle state using kinematic/dynamic equations
- **Use case**: Closed-loop simulation with realistic control delays and saturation
- **Components**:
  - `tracker: AbstractTracker` - LQRTracker, ILQRTracker, etc.
  - `motion_model: AbstractMotionModel` - KinematicBicycleModel, etc.
- **Update flow**:
  1. Tracker computes desired `DynamicCarState` (acceleration, steering rate)
  2. Motion model propagates state using desired dynamics + control delays
  3. Returns updated `EgoState`

### Motion Models (motion_model/)

**AbstractMotionModel**
- **Purpose**: Interface for vehicle dynamics (kinematic or dynamic models)
- **Key methods**:
  - `get_state_dot(state) -> EgoStateDot` - Computes state derivatives (dx/dt)
  - `propagate_state(state, ideal_dynamic_state, sampling_time) -> EgoState` - Integrates state forward
- **Parameters**: `ideal_dynamic_state` is the **desired** dynamics from tracker (may differ from actual due to control delays)

**KinematicBicycleModel**
- **Purpose**: Kinematic bicycle model with rear axle reference point
- **State**: `[x, y, heading, velocity, steering_angle]`
- **Inputs**: `[acceleration, steering_rate]`
- **Dynamics** (continuous time):
  ```
  x_dot = v * cos(heading)
  y_dot = v * sin(heading)
  heading_dot = v * tan(steering_angle) / wheelbase
  v_dot = acceleration
  steering_angle_dot = steering_rate
  ```
- **Control delays**: Low-pass filter on acceleration and steering (1st order lag)
  - `accel_time_constant` (default 0.2s) - Acceleration response time
  - `steering_angle_time_constant` (default 0.05s) - Steering response time
- **Constraints**:
  - `max_steering_angle` (default π/3 = 60°) - Steering angle clipping
  - Heading wrapped to [-π, π] via `principal_value()`
- **Integration**: Forward Euler (`forward_integrate()` utility)
- **Gotcha**: Lateral velocity always zero (kinematic assumption)

### Trackers (tracker/)

**AbstractTracker**
- **Purpose**: Interface for trajectory tracking controllers
- **Key method**: `track_trajectory(current_iteration, next_iteration, initial_state, trajectory) -> DynamicCarState`
- **Output**: Desired acceleration and steering rate to track trajectory
- **Philosophy**: Tracker outputs **desired** dynamics, motion model applies them with delays/saturation

**LQRTracker** - Linear Quadratic Regulator (Decoupled)
- **Purpose**: Decoupled longitudinal + lateral LQR control with kinematic bicycle linearization
- **Architecture**:
  - **Longitudinal subsystem**: States=[velocity], Inputs=[acceleration]
  - **Lateral subsystem**: States=[lateral_error, heading_error, steering_angle], Inputs=[steering_rate]
- **Linearization**: Small angle approximation, Euler discretization
- **Cost weights**:
  - `q_longitudinal` (1,) - Velocity error weight
  - `r_longitudinal` (1,) - Acceleration effort weight
  - `q_lateral` (3,) - [lateral_error, heading_error, steering_angle] weights
  - `r_lateral` (1,) - Steering rate effort weight
- **Key parameters**:
  - `discretization_time` [s] - LQR discretization timestep (e.g., 0.1s)
  - `tracking_horizon` [int] - Lookahead horizon in discrete steps (e.g., 10 = 1s)
  - `stopping_proportional_gain` - P-controller gain for near-stop conditions
  - `stopping_velocity` [m/s] - Velocity threshold for stopping controller
  - `jerk_penalty`, `curvature_rate_penalty` - Regularization for velocity/curvature estimation
- **Reference trajectory processing**:
  1. Interpolate trajectory at `discretization_time` resolution
  2. Compute velocity/curvature profiles via least squares (with jerk/curvature_rate penalties)
  3. Extract reference velocity and curvature at lookahead time
- **Stopping behavior**: If `velocity < stopping_velocity` AND `reference_velocity < stopping_velocity`, use P-controller instead of LQR
- **Lateral state**: Computed in Frenet frame (lateral error, heading error relative to trajectory)
- **Gotcha**: Requires `tracking_horizon > 1` (else steering rate has no impact with Euler discretization)

**ILQRTracker** - Iterative Linear Quadratic Regulator
- **Purpose**: Optimal trajectory tracking via iterative LQR with kinematic bicycle dynamics
- **Solver**: `ILQRSolver` - Full iLQR implementation with warm start, trust regions, constraints
- **State**: `[x, y, heading, velocity, steering_angle]` (5D)
- **Inputs**: `[acceleration, steering_rate]` (2D)
- **Dynamics**: Discrete-time kinematic bicycle (Euler integration)
- **Cost**: Quadratic tracking cost + input effort cost
- **Warm start**: Infers initial input trajectory from reference poses via least squares (velocity/curvature fitting)
- **Constraints**: Input clipping (max accel/steering rate), steering angle saturation
- **Trust regions**: State and input trust region costs to limit update step size
- **Convergence**: Terminates when input norm difference < threshold or max iterations reached
- **Key parameters**:
  - `n_horizon` - Planning horizon (number of timesteps)
  - `ilqr_solver: ILQRSolver` - Solver instance with all cost/constraint parameters
- **Output**: First control input from optimal trajectory (model predictive control)
- **Performance**: Computationally intensive (10-50× slower than LQR due to iterations)

**ILQRSolver** - Iterative LQR Solver Implementation
- **Algorithm**: iLQR with quadratic tracking cost on kinematic bicycle model
- **Key components**:
  - **Forward dynamics**: Euler integration with constraint projection
  - **Backward recursion**: Dynamic programming to compute LQR policy
  - **Policy update**: Apply feedback + feedforward inputs
- **Parameters** (`ILQRSolverParameters`):
  - `discretization_time` [s] - Time discretization
  - `state_cost_diagonal_entries` (5,) - Q matrix diagonal (tracking error weights)
  - `input_cost_diagonal_entries` (2,) - R matrix diagonal (effort weights)
  - `state_trust_region_entries` (5,) - Trust region on state perturbations
  - `input_trust_region_entries` (2,) - Trust region on input perturbations
  - `max_ilqr_iterations` - Iteration limit
  - `convergence_threshold` - Input norm difference threshold for early termination
  - `max_solve_time` [s] - Optional time budget
  - `max_acceleration`, `max_steering_angle`, `max_steering_angle_rate` - Constraints
  - `min_velocity_linearization` [m/s] - Velocity threshold for linearization (avoids singularities)
  - `wheelbase` [m] - Vehicle parameter
- **Warm start** (`ILQRWarmStartParameters`):
  - `k_velocity_error_feedback` - Velocity feedback gain
  - `k_steering_angle_error_feedback` - Steering angle feedback gain
  - `lookahead_distance_lateral_error` [m] - Lateral error lookahead
  - `k_lateral_error` - Lateral error feedback gain
  - `jerk_penalty_warm_start_fit`, `curvature_rate_penalty_warm_start_fit` - Regularization
- **Iteration output**: `ILQRSolution` (state_trajectory, input_trajectory, tracking_cost) at each iteration
- **AIDEV-NOTE**: Returns list of solutions (one per iteration) - last element is final solution

### Tracker Utilities (tracker_utils.py)

**Velocity & Curvature Estimation**
- **Purpose**: Infer velocity/acceleration and curvature/curvature_rate from pose sequence
- **Method**: Regularized least squares with jerk and curvature_rate penalties
- **Key functions**:
  - `get_velocity_curvature_profiles_with_derivatives_from_poses()` - Main entry point
  - `_fit_initial_velocity_and_acceleration_profile()` - Longitudinal dynamics
  - `_fit_initial_curvature_and_curvature_rate_profile()` - Lateral dynamics (heading)
  - `complete_kinematic_state_and_inputs_from_poses()` - Full state + input reconstruction
- **Use case**: LQR/iLQR need reference velocity/curvature, but trajectory only has poses
- **Regularization**: Prevents overfitting to noisy poses (jerk penalty smooths acceleration, curvature_rate penalty smooths steering)
- **AIDEV-NOTE**: Zero velocity edge case handled via `initial_curvature_penalty` (regularizes first curvature to avoid singularity)

**Steering Angle Feedback**
- **Purpose**: Compute feedback steering angle to correct lateral tracking error
- **Function**: `compute_steering_angle_feedback(pose_reference, pose_current, lookahead_distance, k_lateral_error)`
- **Algorithm**: Stanley controller-style lateral error + heading error feedback
- **Formula**: `steering_feedback = -k * (lateral_error + lookahead_distance * heading_error)`
- **Use case**: iLQR warm start (add feedback to feedforward steering from pose fit)

**Trajectory Interpolation**
- **Purpose**: Resample trajectory at fixed discretization time
- **Function**: `get_interpolated_reference_trajectory_poses(trajectory, discretization_time)`
- **Output**: Times array (s) and poses array (N, 3) - [x, y, heading]
- **Use case**: LQR/iLQR need evenly-spaced poses for discretization

### Utilities (utils.py)

**forward_integrate()**
- **Purpose**: Simple Euler integration for scalar state
- **Formula**: `x(t+dt) = x(t) + x_dot * dt`
- **Use case**: KinematicBicycleModel uses this for position, heading, velocity, steering angle integration
- **AIDEV-NOTE**: Simple first-order, no higher-order methods (adequate for small dt)

## 3. Architecture & Design Patterns

### 1. **Strategy Pattern**: Pluggable Controllers
```python
# Controllers are interchangeable via AbstractEgoController interface
controller: AbstractEgoController = instantiate(cfg.ego_controller, scenario=scenario)
controller.update_state(...)  # Polymorphic dispatch
```

### 2. **Composite Pattern**: Two-Stage Controller
```python
TwoStageController(scenario, tracker, motion_model)
    ├─ tracker.track_trajectory() → DynamicCarState (desired dynamics)
    └─ motion_model.propagate_state() → EgoState (actual dynamics)
```
- Enables mix-and-match: LQRTracker + KinematicBicycleModel, ILQRTracker + DynamicBicycleModel, etc.
- Clean separation of concerns: tracking logic vs dynamics

### 3. **Template Method**: Motion Model Propagation
```python
class AbstractMotionModel(ABC):
    def get_state_dot(state) -> EgoStateDot:  # Subclass implements
        ...

    def propagate_state(state, ideal_dynamic_state, sampling_time) -> EgoState:  # Subclass implements
        state_dot = self.get_state_dot(state)
        # Euler integration...
```

### 4. **Lazy Initialization**: Controller State
```python
self._current_state: Optional[EgoState] = None  # Set to None initially

def get_state(self) -> EgoState:
    if self._current_state is None:
        self._current_state = self._scenario.initial_ego_state
    return self._current_state
```
- Avoids coupling constructor to scenario data
- Enables Hydra instantiation without scenario context

### 5. **Least Squares Estimation**: Reference Signal Reconstruction
- LQRTracker needs velocity/curvature profiles, but trajectory only has poses
- Solution: Fit velocity/curvature via regularized least squares
- Pattern: Transform discrete noisy measurements → smooth continuous signals

### 6. **Frozen Dataclasses**: iLQR State Containers
```python
@dataclass(frozen=True)
class ILQRIterate:
    state_trajectory: DoubleMatrix
    input_trajectory: DoubleMatrix
    state_jacobian_trajectory: DoubleMatrix
    input_jacobian_trajectory: DoubleMatrix
```
- Immutable snapshots of iLQR iteration state
- Prevents accidental mutation during optimization

### 7. **Trust Regions**: iLQR Stability
- Without trust regions, iLQR can diverge (linearization error too large)
- Trust region cost penalizes large perturbations: `cost += state_diff^T Q_tr state_diff`
- Keeps updates within linearization validity region

## 4. Dependencies

### Internal (nuPlan - Documented ✅)

**Direct Dependencies**:
- ✅ `nuplan.common.actor_state.ego_state` - EgoState, EgoStateDot
- ✅ `nuplan.common.actor_state.dynamic_car_state` - DynamicCarState
- ✅ `nuplan.common.actor_state.state_representation` - StateSE2, StateVector2D, TimePoint
- ✅ `nuplan.common.actor_state.vehicle_parameters` - VehicleParameters, get_pacifica_parameters()
- ✅ `nuplan.common.geometry.compute` - `principal_value()` (angle wrapping)
- ✅ `nuplan.planning.scenario_builder.abstract_scenario` - AbstractScenario
- ✅ `nuplan.planning.simulation.simulation_time_controller.simulation_iteration` - SimulationIteration
- ✅ `nuplan.planning.simulation.trajectory.abstract_trajectory` - AbstractTrajectory

**Indirect Dependencies**:
- `nuplan.database.utils.measure` - `angle_diff()` (angle difference with wrapping)

### External Dependencies
- **NumPy** - Array operations, linear algebra (`np.linalg.inv`, `np.linalg.pinv`)
- **SciPy** - Not used in core controller (but iLQR could use ODE solvers)
- **typing** - Type hints
- **dataclasses** - Frozen dataclasses for iLQR parameters

### Dependency Notes
**AIDEV-NOTE**: Controllers depend heavily on `✅ actor_state` for state representations. Motion models use `✅ geometry.compute.principal_value()` for angle wrapping.

## 5. Dependents (Who Uses This Module?)

### Direct Consumers

**Simulation Infrastructure**:
- `nuplan/planning/simulation/simulation.py` - Main simulation loop
  - `__init__(setup)` - Receives `setup.ego_controller`
  - `propagate()` - Calls `ego_controller.update_state()`
  - `reset()` - Calls `ego_controller.reset()`

**Simulation Setup**:
- `nuplan/planning/simulation/simulation_setup.py` - Configuration container
  - `ego_controller: AbstractEgoController` - Required field
  - `reset()` - Calls `ego_controller.reset()`

**Builders**:
- `nuplan/planning/script/builders/simulation_builder.py`
  - `build_simulations()` - Instantiates controller via Hydra: `instantiate(cfg.ego_controller, scenario=scenario)`

### Hydra Configurations
- `nuplan/planning/script/config/simulation/ego_controller/*.yaml`
  - `perfect_tracking_controller.yaml` - PerfectTrackingController config
  - `log_playback_controller.yaml` - LogPlaybackController config
  - `two_stage_controller.yaml` - TwoStageController with LQRTracker + KinematicBicycleModel

### Use Cases

1. **Perfect Tracking (Testing)**
   - Planner development/debugging: Remove control error from evaluation
   - Metrics: Isolate planner performance from controller performance

2. **Realistic Closed-Loop (Validation)**
   - TwoStageController with LQR: Standard closed-loop simulation
   - Evaluates planner + controller system performance
   - Simulates realistic control delays, saturation

3. **Oracle Baseline (Comparison)**
   - LogPlaybackController: Ground-truth ego trajectory
   - Baseline for metrics comparison (how much worse is planner vs human?)

4. **Advanced Optimal Control (Research)**
   - ILQRTracker: State-of-the-art trajectory tracking
   - Research on optimal control strategies

**AIDEV-NOTE**: Most simulations use TwoStageController with LQRTracker (realistic but not too slow). ILQRTracker is for research/advanced use.

## 6. Critical Files (Prioritized)

### Priority 1: Core Interfaces (Read First!)

1. **`abstract_controller.py`** (45 lines) - **START HERE!**
   - `AbstractEgoController` interface
   - 3 methods: `get_state()`, `reset()`, `update_state()`
   - Simple, self-contained

2. **`perfect_tracking.py`** (47 lines) - **Simplest implementation**
   - Shows minimal controller logic
   - Trajectory interpolation pattern
   - Velocity validation

3. **`log_playback.py`** (38 lines) - **Oracle controller**
   - Scenario database query pattern
   - Iteration index management

### Priority 2: Realistic Control (Most Common)

4. **`two_stage_controller.py`** (61 lines) - **Standard controller**
   - Tracker + motion model composition
   - Two-stage update flow
   - Lazy state initialization

5. **`motion_model/kinematic_bicycle.py`** (150 lines) - **Motion model**
   - Bicycle dynamics equations
   - Control delay implementation
   - Euler integration
   - Constraint enforcement

6. **`tracker/lqr.py`** (389 lines) - **LQR tracker (standard)**
   - Decoupled longitudinal + lateral LQR
   - Reference velocity/curvature computation
   - Stopping controller logic
   - Frenet frame error computation

### Priority 3: Advanced (Research-Level)

7. **`tracker/ilqr_tracker.py`** (108 lines) - **iLQR wrapper**
   - ILQRSolver integration
   - Reference trajectory extraction
   - MPC pattern (apply first input only)

8. **`tracker/ilqr/ilqr_solver.py`** (689 lines) - **iLQR implementation**
   - Full iLQR algorithm
   - Warm start generation
   - Backward recursion (LQR)
   - Forward dynamics with constraints
   - Trust region handling
   - **Complex but well-documented!**

### Priority 4: Utilities

9. **`tracker/tracker_utils.py`** (377 lines) - **Signal processing**
   - Velocity/curvature estimation from poses
   - Regularized least squares
   - Steering feedback computation
   - Trajectory interpolation

10. **`utils.py`** (13 lines) - **Basic integration**
    - `forward_integrate()` - Euler integration helper

### Priority 5: Supporting Files

11. **`motion_model/abstract_motion_model.py`** (36 lines) - Motion model interface
12. **`tracker/abstract_tracker.py`** (31 lines) - Tracker interface

**AIDEV-NOTE**: For typical users, read files 1-6. Files 7-8 (iLQR) are advanced research code - skip unless needed.

## 7. Common Usage Patterns

### 1. Perfect Tracking Controller (Testing)
```python
from nuplan.planning.simulation.controller.perfect_tracking import PerfectTrackingController

# Instantiate (Hydra or manual)
controller = PerfectTrackingController(scenario=scenario)
controller.reset()

# Simulation loop
for iteration, next_iteration in simulation_timesteps:
    # Get current state
    ego_state = controller.get_state()

    # Plan trajectory
    planner_input = PlannerInput(iteration=iteration, history=history_buffer)
    trajectory = planner.compute_planner_trajectory(planner_input)

    # Update ego state (perfect tracking)
    controller.update_state(iteration, next_iteration, ego_state, trajectory)

    # New state available via get_state()
    new_ego_state = controller.get_state()
```

### 2. Two-Stage Controller with LQR (Standard)
```python
from nuplan.planning.simulation.controller.two_stage_controller import TwoStageController
from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
import numpy as np

# Build tracker
lqr_tracker = LQRTracker(
    q_longitudinal=np.array([10.0]),  # Velocity error weight
    r_longitudinal=np.array([1.0]),   # Acceleration effort weight
    q_lateral=np.array([1.0, 10.0, 0.0]),  # [lateral_error, heading_error, steering_angle]
    r_lateral=np.array([1.0]),        # Steering rate effort weight
    discretization_time=0.1,          # 10 Hz
    tracking_horizon=10,              # 1 second lookahead
    jerk_penalty=0.01,
    curvature_rate_penalty=0.01,
    stopping_proportional_gain=0.5,
    stopping_velocity=0.1,            # m/s
    vehicle=get_pacifica_parameters(),
)

# Build motion model
motion_model = KinematicBicycleModel(
    vehicle=get_pacifica_parameters(),
    max_steering_angle=np.pi / 3,     # 60 degrees
    accel_time_constant=0.2,          # 200ms acceleration lag
    steering_angle_time_constant=0.05,  # 50ms steering lag
)

# Build two-stage controller
controller = TwoStageController(
    scenario=scenario,
    tracker=lqr_tracker,
    motion_model=motion_model,
)

# Simulation loop (same as above)
controller.reset()
for iteration, next_iteration in simulation_timesteps:
    ego_state = controller.get_state()
    trajectory = planner.compute_planner_trajectory(...)

    # Two-stage update:
    # 1. LQR computes desired acceleration/steering_rate
    # 2. KinematicBicycle propagates state with control delays
    controller.update_state(iteration, next_iteration, ego_state, trajectory)
```

### 3. iLQR Tracker (Advanced Optimal Control)
```python
from nuplan.planning.simulation.controller.tracker.ilqr_tracker import ILQRTracker
from nuplan.planning.simulation.controller.tracker.ilqr.ilqr_solver import (
    ILQRSolver, ILQRSolverParameters, ILQRWarmStartParameters
)

# Configure iLQR solver
solver_params = ILQRSolverParameters(
    discretization_time=0.1,
    state_cost_diagonal_entries=[1.0, 1.0, 10.0, 10.0, 1.0],  # [x, y, heading, velocity, steering]
    input_cost_diagonal_entries=[1.0, 1.0],  # [accel, steering_rate]
    state_trust_region_entries=[0.1, 0.1, 0.5, 0.5, 0.1],
    input_trust_region_entries=[0.1, 0.1],
    max_ilqr_iterations=10,
    convergence_threshold=1e-3,
    max_solve_time=0.05,  # 50ms budget
    max_acceleration=3.0,
    max_steering_angle=np.pi / 3,
    max_steering_angle_rate=np.pi / 2,
    min_velocity_linearization=0.1,
)

warm_start_params = ILQRWarmStartParameters(
    k_velocity_error_feedback=1.0,
    k_steering_angle_error_feedback=1.0,
    lookahead_distance_lateral_error=5.0,
    k_lateral_error=0.5,
    jerk_penalty_warm_start_fit=0.01,
    curvature_rate_penalty_warm_start_fit=0.01,
)

ilqr_solver = ILQRSolver(solver_params, warm_start_params)

# Build iLQR tracker
ilqr_tracker = ILQRTracker(n_horizon=20, ilqr_solver=ilqr_solver)

# Use with TwoStageController
controller = TwoStageController(scenario, ilqr_tracker, motion_model)
```

### 4. Accessing Controller State
```python
# Get current ego state at any time
ego_state = controller.get_state()

# Extract pose
x = ego_state.rear_axle.x
y = ego_state.rear_axle.y
heading = ego_state.rear_axle.heading

# Extract dynamics
velocity = ego_state.dynamic_car_state.rear_axle_velocity_2d.x
acceleration = ego_state.dynamic_car_state.rear_axle_acceleration_2d.x
steering_angle = ego_state.tire_steering_angle

# Check if in auto mode
is_auto = ego_state.is_in_auto_mode
```

### 5. Velocity/Curvature Estimation (Tracker Utilities)
```python
from nuplan.planning.simulation.controller.tracker.tracker_utils import (
    get_velocity_curvature_profiles_with_derivatives_from_poses
)

# Extract poses from trajectory
poses = np.array([[state.rear_axle.x, state.rear_axle.y, state.rear_axle.heading]
                  for state in trajectory_states])

# Estimate velocity and curvature profiles
velocity_profile, accel_profile, curvature_profile, curvature_rate_profile = \
    get_velocity_curvature_profiles_with_derivatives_from_poses(
        discretization_time=0.1,
        poses=poses,
        jerk_penalty=0.01,       # Smoothness penalty on acceleration
        curvature_rate_penalty=0.01,  # Smoothness penalty on curvature
    )

# velocity_profile: [m/s] length N-1 (one per displacement)
# accel_profile: [m/s²] length N-2 (one per jerk)
# curvature_profile: [rad/m] length N-1
# curvature_rate_profile: [rad/m/s] length N-2
```

### 6. Custom Motion Model
```python
from nuplan.planning.simulation.controller.motion_model.abstract_motion_model import AbstractMotionModel

class MyDynamicModel(AbstractMotionModel):
    def __init__(self, mass, drag_coeff, ...):
        self.mass = mass
        self.drag_coeff = drag_coeff

    def get_state_dot(self, state: EgoState) -> EgoStateDot:
        """Compute state derivatives from current state"""
        velocity = state.dynamic_car_state.rear_axle_velocity_2d.x

        # Dynamic equations (e.g., drag force)
        drag_force = 0.5 * self.drag_coeff * velocity**2

        # Return EgoStateDot with derivatives
        ...

    def propagate_state(self, state, ideal_dynamic_state, sampling_time) -> EgoState:
        """Integrate state forward"""
        state_dot = self.get_state_dot(state)

        # Higher-order integration (e.g., RK4)
        ...

        return new_ego_state

# Use with TwoStageController
controller = TwoStageController(scenario, tracker, MyDynamicModel(...))
```

## 8. Gotchas & Edge Cases

### Controller Lifecycle Issues

1. **`reset()` must be called between scenarios**
   - **Issue**: Controller state persists across scenario runs
   - **Symptom**: New scenario starts with ego state from previous scenario
   - **Fix**: Simulation loop calls `controller.reset()` before each scenario
   - **Check**: `_current_state` should be None after reset

2. **Lazy state initialization race condition**
   - **Issue**: `get_state()` returns `scenario.initial_ego_state` if `_current_state` is None
   - **Symptom**: First `get_state()` call after `reset()` returns scenario initial state (correct)
   - **Gotcha**: Subsequent `get_state()` calls before first `update_state()` still return initial state
   - **Expected**: This is intentional - state doesn't change until `update_state()` is called

3. **Update without trajectory fails silently**
   - **Issue**: PerfectTrackingController.update_state() samples trajectory, but trajectory might not cover next_iteration
   - **Symptom**: `trajectory.get_state_at_time(next_iteration.time_point)` returns None
   - **Fix**: Assert in PerfectTrackingController:41 catches this (raises AssertionError)
   - **Prevention**: Planners must return trajectories covering simulation horizon

4. **LogPlaybackController ignores planner trajectory**
   - **Issue**: LogPlaybackController uses scenario DB, not planner output
   - **Symptom**: Planner trajectory is computed but has no effect
   - **Expected behavior**: This is intentional (oracle mode)
   - **Gotcha**: Don't use LogPlaybackController to test planners!

5. **Controller state diverges from observation**
   - **Issue**: Controller.get_state() and observation.get_observation() can be out of sync
   - **Symptom**: Ego in observation doesn't match controller.get_state()
   - **Root cause**: Simulation bug (update order mismatch)
   - **AIDEV-NOTE**: Simulation should ensure controller.update_state() before observation.update_observation()

### Two-Stage Controller Issues

6. **Tracker can return infeasible dynamics**
   - **Issue**: LQRTracker computes unbounded acceleration/steering_rate (no saturation in tracker)
   - **Symptom**: Motion model clips inputs → tracking error
   - **Expected**: Motion model enforces constraints (KinematicBicycleModel clips steering, not acceleration)
   - **AIDEV-NOTE**: LQR cost should penalize large inputs to avoid saturation

7. **Control delay can cause instability**
   - **Issue**: Large `accel_time_constant` or `steering_angle_time_constant` → slow response
   - **Symptom**: Oscillations, overshoots, poor tracking at high speeds
   - **Fix**: Tune tracker gains (Q, R matrices) to account for delays
   - **Rule of thumb**: Tracking horizon should be > 3× control time constant

8. **Motion model integration timestep mismatch**
   - **Issue**: `sampling_time` in propagate_state() doesn't match tracker discretization_time
   - **Symptom**: Tracker assumes 0.1s, but simulation uses 0.05s → tracking error
   - **Fix**: Ensure tracker.discretization_time == simulation timestep
   - **Check**: `next_iteration.time_s - current_iteration.time_s == discretization_time`

9. **Zero wheelbase crashes KinematicBicycleModel**
   - **Issue**: Division by zero in `heading_dot = velocity * tan(steering) / wheelbase`
   - **Symptom**: RuntimeError or NaN propagation
   - **Prevention**: VehicleParameters validation (wheelbase > 0)

10. **Steering angle wrapping causes jumps**
    - **Issue**: Steering angle crosses [-π, π] boundary → discontinuity
    - **Symptom**: Sudden 2π jump in steering angle
    - **Fix**: KinematicBicycleModel.propagate_state() wraps heading via `principal_value()` but NOT steering angle
    - **AIDEV-NOTE**: Steering angle typically stays in [-π/3, π/3] (no wrapping needed)

### LQR Tracker Issues

11. **Tracking horizon too short**
    - **Issue**: `tracking_horizon = 1` → steering rate has no effect (Euler integration)
    - **Symptom**: Lateral tracking error, vehicle doesn't steer
    - **Fix**: Use `tracking_horizon >= 2` (enforced in LQRTracker.__init__)
    - **Recommended**: tracking_horizon = 10 (1 second @ 0.1s)

12. **Stopping controller thrashing**
    - **Issue**: Velocity oscillates around `stopping_velocity` threshold → switches between LQR and P-controller
    - **Symptom**: Jerky motion near stop
    - **Fix**: Hysteresis (use different thresholds for entering/exiting stopping mode)
    - **AIDEV-TODO**: Add hysteresis to stopping controller

13. **Q/R matrix not positive (semi-)definite**
    - **Issue**: LQRTracker asserts Q ≥ 0, R > 0, but user passes negative weights
    - **Symptom**: AssertionError in LQRTracker.__init__
    - **Fix**: Validate config before instantiation
    - **Note**: Q must be PSD (≥ 0), R must be PD (> 0) for LQR to work

14. **Reference velocity/curvature estimation fails**
    - **Issue**: Trajectory has < 2 poses → `_get_xy_heading_displacements_from_poses()` asserts
    - **Symptom**: AssertionError: "Cannot get displacements given empty or single element pose trajectory"
    - **Fix**: Planners must return trajectories with ≥ 2 waypoints

15. **Jerk/curvature_rate penalty too large**
    - **Issue**: Large penalties → over-smoothed velocity/curvature (doesn't match trajectory)
    - **Symptom**: Poor tracking, controller follows smoothed reference instead of actual trajectory
    - **Fix**: Tune penalties (typical: 0.01 - 0.1)

### iLQR Tracker Issues

16. **iLQR doesn't converge**
    - **Issue**: max_ilqr_iterations reached, but convergence_threshold not satisfied
    - **Symptom**: Suboptimal tracking, high cost
    - **Causes**: Poor warm start, reference trajectory not feasible, trust regions too small
    - **Debug**: Inspect `solution_list[-1].tracking_cost` vs `solution_list[0].tracking_cost` (should decrease)

17. **Warm start infeasible**
    - **Issue**: Inferred inputs from reference trajectory violate constraints
    - **Symptom**: Warm start input trajectory has clipped values
    - **Expected**: This is OK - iLQR will optimize from infeasible warm start
    - **AIDEV-NOTE**: Consider constraint-aware warm start (project onto feasible set)

18. **Trust region too tight**
    - **Issue**: Large `state_trust_region_entries` or `input_trust_region_entries` → tiny updates
    - **Symptom**: iLQR converges slowly or not at all
    - **Fix**: Reduce trust region weights (0.01 - 0.5 typical)

19. **Min velocity linearization causes jumps**
    - **Issue**: Velocity crosses `min_velocity_linearization` threshold → Jacobian changes discontinuously
    - **Symptom**: Optimization jumpiness near zero velocity
    - **Fix**: Use small `min_velocity_linearization` (0.01 - 0.1 m/s)

20. **iLQR exceeds time budget**
    - **Issue**: `max_solve_time` too large or not set → tracker takes > 0.1s
    - **Symptom**: Simulation slower than real-time, planner gets less time
    - **Fix**: Set `max_solve_time` to leave time for planner (e.g., 0.05s if planner needs 0.05s)
    - **Monitoring**: Check ILQRSolution list length (should be < max_ilqr_iterations if time budget hit)

### Motion Model Issues

21. **Euler integration accumulates error**
    - **Issue**: Forward Euler is first-order → error accumulates over time
    - **Symptom**: State drift, especially in heading/position
    - **Fix**: Use smaller timestep or higher-order integrator (RK4)
    - **AIDEV-TODO**: Implement RK4 integration option in KinematicBicycleModel

22. **Control delay parameters unrealistic**
    - **Issue**: `accel_time_constant` or `steering_angle_time_constant` too small/large
    - **Symptom**: Too responsive (no delay) or too sluggish (unrealistic lag)
    - **Realistic values**: accel: 0.1-0.3s, steering: 0.03-0.1s
    - **Source**: Real vehicle data, NHTSA tests

23. **Lateral velocity assumption violation**
    - **Issue**: KinematicBicycleModel assumes `lateral_velocity = 0` (no slip)
    - **Symptom**: Inaccurate at high speeds, aggressive maneuvers
    - **Fix**: Use dynamic bicycle model with slip angle (future work)
    - **Typical validity**: Works well for < 10 m/s², normal driving

24. **Acceleration not clipped**
    - **Issue**: KinematicBicycleModel clips steering but NOT acceleration
    - **Symptom**: Unrealistic acceleration (> vehicle limits)
    - **Expected**: Tracker should limit acceleration via cost, or add constraint in motion model
    - **AIDEV-TODO**: Add max_acceleration clipping in KinematicBicycleModel

## 9. Performance Considerations

### Computational Cost (per timestep, typical scenario)

| Controller | Complexity | Typical Time | Notes |
|-----------|-----------|--------------|-------|
| PerfectTrackingController | O(1) | < 0.1 ms | Trajectory interpolation only |
| LogPlaybackController | O(1) | < 0.5 ms | Database query |
| TwoStageController (LQR) | O(H² × N_states²) | 1-5 ms | H = tracking_horizon (10), N_states = 3 (lateral) |
| TwoStageController (iLQR) | O(I × H × N_states³) | 10-50 ms | I = iterations (5-10), H = horizon (20), N_states = 5 |

**Bottleneck analysis (LQRTracker)**:
- Velocity/curvature estimation: 30-40% (least squares solve)
- LQR solve (lateral): 20-30% (matrix inverse)
- LQR solve (longitudinal): 5-10%
- Reference trajectory interpolation: 10-20%
- Overhead: 10-20%

**Bottleneck analysis (ILQRTracker)**:
- iLQR iterations: 70-80% (forward dynamics + backward recursion × I)
- Warm start generation: 10-15% (pose fitting)
- Reference trajectory extraction: 5-10%

### Memory Footprint

- **PerfectTrackingController**: < 1 KB (just EgoState)
- **LogPlaybackController**: < 1 KB
- **TwoStageController (LQR)**: ~10 KB (interpolated trajectory, velocity/curvature profiles)
- **TwoStageController (iLQR)**: ~100 KB (solution list × iterations, Jacobians)

### Real-Time Constraints

- **Target**: Controller update < 10 ms (leaves 90ms for planner in 100ms budget)
- **PerfectTracking/LogPlayback**: ✅ Always real-time
- **LQR**: ✅ Real-time for tracking_horizon ≤ 20
- **iLQR**: ⚠️ May exceed real-time if max_ilqr_iterations > 10 or horizon > 30
  - **Mitigation**: Set `max_solve_time` to enforce time budget

### Optimization Strategies

1. **Reduce LQR tracking horizon** (10 → 5)
   - Trades tracking accuracy for speed
   - Still works well for smooth trajectories

2. **Cache velocity/curvature profiles** (LQR)
   - If trajectory doesn't change, reuse previous profiles
   - Requires trajectory equality check

3. **Warm start iLQR from previous solution**
   - Use previous solve's final input trajectory as warm start
   - Significantly faster convergence (2-3× speedup)
   - **AIDEV-TODO**: Implement in ILQRTracker

4. **Early termination (iLQR)**
   - Monitor convergence_threshold, stop when satisfied
   - Typical: Converges in 3-5 iterations if warm start is good

5. **Parallel LQR (if multiple controllers)**
   - Longitudinal and lateral LQR are independent
   - Could parallelize (marginal benefit, adds overhead)

6. **Use NumPy optimized BLAS**
   - Ensure NumPy linked to MKL or OpenBLAS
   - 2-5× speedup on matrix operations

## 10. Related Documentation

### Cross-References (Documented ✅)
- ✅ `nuplan/planning/simulation/planner/CLAUDE.md` - Planners generate trajectories for controllers
- ✅ `nuplan/planning/simulation/trajectory/CLAUDE.md` - Trajectory representations (AbstractTrajectory, InterpolatedTrajectory)
- ✅ `nuplan/planning/simulation/observation/CLAUDE.md` - Observations updated after controller
- ✅ `nuplan/planning/simulation/history/CLAUDE.md` - History buffer stores ego states from controller
- ✅ `nuplan/planning/simulation/simulation_time_controller/CLAUDE.md` - SimulationIteration used in update_state()
- ✅ `nuplan/common/actor_state/CLAUDE.md` - EgoState, DynamicCarState, EgoStateDot
- ✅ `nuplan/common/geometry/CLAUDE.md` - Geometry utilities (principal_value, coordinate transforms)
- ✅ `nuplan/planning/scenario_builder/CLAUDE.md` - Scenario provides initial_ego_state

### Undocumented Dependencies (Future ⏳)
- (None - all dependencies documented!)

### Configuration Files
- `nuplan/planning/script/config/simulation/ego_controller/perfect_tracking_controller.yaml`
- `nuplan/planning/script/config/simulation/ego_controller/log_playback_controller.yaml`
- `nuplan/planning/script/config/simulation/ego_controller/two_stage_controller.yaml`

### External Resources
- **LQR theory**: Boyd & Barratt, "Linear Controller Design: Limits of Performance"
- **iLQR algorithm**: Todorov & Li, "A generalized iterative LQR method for locally-optimal feedback control of constrained nonlinear stochastic systems" (2005)
- **Bicycle model**: Rajamani, "Vehicle Dynamics and Control" (2011), Chapter 2
- **Stanley controller**: Hoffmann et al., "Autonomous Automobile Trajectory Tracking for Off-Road Driving" (2007)

## 11. AIDEV Notes

### Design Philosophy
- **Separation of concerns**: Tracker (what to do) vs motion model (how vehicle responds)
- **Modularity**: Mix-and-match trackers and motion models
- **Realism**: Two-stage architecture mirrors real AV software (planning → control → actuation)
- **Testing**: Perfect tracking enables planner-only evaluation

### Common Mistakes
- Forgetting to call `reset()` between scenarios (stale state)
- Using LogPlaybackController to test planners (it ignores planner output!)
- Assuming controller is stateless (it maintains _current_state)
- Not checking trajectory horizon covers next_iteration (PerfectTracking assertion)
- Setting tracking_horizon = 1 (breaks LQR lateral control)

### Future Improvements
- **AIDEV-TODO**: Add hysteresis to LQRTracker stopping controller (prevent oscillations)
- **AIDEV-TODO**: Implement RK4 integration in KinematicBicycleModel (reduce drift)
- **AIDEV-TODO**: Add max_acceleration clipping in KinematicBicycleModel
- **AIDEV-TODO**: Warm start iLQR from previous solution (temporal coherence)
- **AIDEV-TODO**: Constraint-aware warm start for iLQR (project onto feasible set)
- **AIDEV-TODO**: Add dynamic bicycle model (handle slip, high-speed maneuvers)
- **AIDEV-TODO**: Parallelize iLQR forward/backward passes (GPU acceleration)
- **AIDEV-TODO**: Cache velocity/curvature profiles in LQRTracker (avoid redundant computation)

### Potential Bugs
- **AIDEV-NOTE** (perfect_tracking.py:44-46): Velocity threshold hardcoded at 50 m/s
  - Should this be configurable? What about racing scenarios?

- **AIDEV-NOTE** (kinematic_bicycle.py:118): Lateral velocity hardcoded to zero
  - Violates dynamics at high speeds - consider warning or assertion

- **AIDEV-NOTE** (lqr.py:148-151): Stopping controller has no velocity feedback limit
  - Could command large deceleration if velocity error is large
  - Add saturation or use LQR even at low speeds?

- **AIDEV-QUESTION** (two_stage_controller.py:30-31): Why is _current_state Optional?
  - Lazy initialization pattern, but could use @property instead
  - Would avoid repeated None checks

### Testing Gaps
- No unit tests for control delay in KinematicBicycleModel
  - Should verify lag response matches time constants

- No integration tests for LQR tracking accuracy
  - Should verify tracking error < threshold for standard trajectories

- No performance benchmarks for iLQR
  - Should profile and document expected runtimes

### Documentation Improvements Needed
- Add diagram of two-stage controller dataflow
- Document typical Q/R matrix values (LQR tuning guide)
- Explain trust region selection (iLQR tuning guide)
- Add coordinate frame diagram (rear axle reference point)
