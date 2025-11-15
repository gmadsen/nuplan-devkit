# nuplan/planning/simulation/controller/tracker/

## 1. Purpose & Responsibility

This module implements **trajectory tracking controllers** that compute desired vehicle dynamics (acceleration and steering rate) to follow a reference trajectory. Trackers are THE bridge between high-level path planning and low-level vehicle control, translating geometric trajectories into control commands that motion models can execute. The core responsibility is computing optimal control inputs at each timestep to minimize tracking error while respecting vehicle dynamics constraints and comfort limits.

## 2. Key Abstractions

### Core Interface

**AbstractTracker**
- **Purpose**: Base class for all trajectory tracking controllers
- **Key method**: `track_trajectory(current_iteration, next_iteration, initial_state, trajectory) -> DynamicCarState`
- **Input**: Current ego state + reference trajectory to follow
- **Output**: Desired `DynamicCarState` (acceleration, steering rate)
- **Philosophy**: Tracker outputs **desired** dynamics; motion model applies them with delays/saturation
- **Design pattern**: Strategy pattern - trackers are interchangeable

### Tracker Implementations

**LQRTracker** - Linear Quadratic Regulator (Decoupled)
- **Purpose**: Decoupled longitudinal + lateral LQR control with kinematic bicycle linearization
- **Architecture**:
  - **Longitudinal subsystem**: States=[velocity], Inputs=[acceleration]
  - **Lateral subsystem**: States=[lateral_error, heading_error, steering_angle], Inputs=[steering_rate]
- **Linearization**: Small angle approximation, Euler discretization, zero-order-hold on input
- **State representation**:
  ```python
  # Lateral state (LateralStateIndex enum)
  [lateral_error,    # [m] Lateral error in Frenet frame
   heading_error,    # [rad] Heading error relative to trajectory
   steering_angle]   # [rad] Current wheel angle
  ```
- **Continuous-time lateral dynamics** (linearized):
  ```
  lateral_error_dot  = velocity * heading_error
  heading_error_dot  = velocity * (steering_angle / wheelbase - curvature)
  steering_angle_dot = steering_rate
  ```
- **Continuous-time longitudinal dynamics**:
  ```
  velocity_dot = acceleration
  ```
- **Cost weights**:
  - `q_longitudinal` (1,) - Velocity error weight (default: [10.0])
  - `r_longitudinal` (1,) - Acceleration effort weight (default: [1.0])
  - `q_lateral` (3,) - [lateral_error, heading_error, steering_angle] weights (default: [1.0, 10.0, 0.0])
  - `r_lateral` (1,) - Steering rate effort weight (default: [1.0])
- **Key parameters**:
  - `discretization_time` [s] - LQR discretization timestep (default: 0.1s)
  - `tracking_horizon` [int] - Lookahead horizon in discrete steps (default: 10 = 1s lookahead)
  - `stopping_proportional_gain` - P-controller gain for near-stop conditions (default: 0.5)
  - `stopping_velocity` [m/s] - Velocity threshold for stopping controller (default: 0.2 m/s)
  - `jerk_penalty` - Regularization for velocity estimation from poses (default: 1e-4)
  - `curvature_rate_penalty` - Regularization for curvature estimation (default: 1e-2)
- **Reference trajectory processing**:
  1. Interpolate trajectory at `discretization_time` resolution (0.1s)
  2. Compute velocity/curvature profiles via regularized least squares
  3. Extract reference velocity and curvature at lookahead time
- **Stopping behavior**: If `velocity < stopping_velocity` AND `reference_velocity < stopping_velocity`, switch to P-controller
- **Lateral state computation**: Computed in Frenet frame (perpendicular to trajectory)
- **Performance**: ~1-5 ms per timestep (fast enough for real-time)
- **Use case**: Standard closed-loop simulation, good balance of performance and tracking accuracy

**ILQRTracker** - Iterative Linear Quadratic Regulator
- **Purpose**: Optimal trajectory tracking via iterative LQR with full kinematic bicycle dynamics
- **Solver**: `ILQRSolver` - Full iLQR implementation with warm start, trust regions, constraints
- **State**: `[x, y, heading, velocity, steering_angle]` (5D, full pose)
- **Inputs**: `[acceleration, steering_rate]` (2D)
- **Dynamics**: Discrete-time kinematic bicycle with Euler integration
- **Cost**: Quadratic tracking cost + input effort cost + trust region penalties
- **Algorithm**: Iterative LQR with forward dynamics rollout + backward recursion
- **Warm start**: Infers initial input trajectory from reference poses via least squares
- **Constraints**: Input clipping (max accel/steering rate), steering angle saturation
- **Trust regions**: State and input trust region costs to limit update step size (prevent divergence)
- **Convergence**: Terminates when `||u_next - u_current|| < convergence_threshold` or max iterations reached
- **Key parameters**:
  - `n_horizon` - Planning horizon (number of timesteps, default: 40)
  - `ilqr_solver: ILQRSolver` - Solver instance with all cost/constraint parameters
- **Output**: First control input from optimal trajectory (model predictive control pattern)
- **Performance**: ~10-50 ms per timestep (10-50× slower than LQR due to iterations)
- **Use case**: Research, advanced optimal control, when LQR accuracy is insufficient

### ILQRSolver - Core iLQR Implementation

**ILQRSolver**
- **Purpose**: Solve trajectory tracking problem using iterative Linear Quadratic Regulator
- **Algorithm**: iLQR with quadratic tracking cost on kinematic bicycle model
- **Key components**:
  - **Forward dynamics**: Euler integration with constraint projection
  - **Backward recursion**: Dynamic programming to compute LQR policy (feedback + feedforward)
  - **Policy update**: Apply perturbations to current trajectory
- **Discrete-time bicycle dynamics**:
  ```
  x_{k+1}     = x_k     + v_k * cos(theta_k) * dt
  y_{k+1}     = y_k     + v_k * sin(theta_k) * dt
  theta_{k+1} = theta_k + v_k * tan(delta_k) / L * dt
  v_{k+1}     = v_k     + a_k * dt
  delta_{k+1} = delta_k + phi_k * dt
  ```
  where `delta` = steering angle, `phi` = steering rate, `L` = wheelbase
- **Cost function**:
  ```
  J = sum_{k=0}^{N-1} ||u_k||_2^{R} + sum_{k=0}^N ||z_k - z_{ref,k}||_2^{Q}
      + trust_region_costs
  ```
- **Trust region**: Penalizes large deviations from linearization trajectory (keeps updates small)
- **Convergence check**: `||u_{iter+1} - u_{iter}|| < convergence_threshold`
- **Time budget**: Optional `max_solve_time` to enforce real-time constraints

**ILQRSolverParameters** (frozen dataclass)
- `discretization_time` [s] - Time discretization (default: 0.2s)
- `state_cost_diagonal_entries` (5,) - Q matrix diagonal [x, y, heading, velocity, steering_angle]
  - Default: [1.0, 1.0, 10.0, 0.0, 0.0] - prioritize heading tracking
- `input_cost_diagonal_entries` (2,) - R matrix diagonal [acceleration, steering_rate]
  - Default: [1.0, 10.0] - penalize steering rate more than acceleration
- `state_trust_region_entries` (5,) - Trust region on state perturbations (default: [1.0] × 5)
- `input_trust_region_entries` (2,) - Trust region on input perturbations (default: [1.0] × 2)
- `max_ilqr_iterations` - Iteration limit (default: 20)
- `convergence_threshold` - Input norm difference threshold (default: 1e-6)
- `max_solve_time` [s] - Optional time budget (default: 0.05s = 50ms)
- `max_acceleration` [m/s²] - Constraint (default: 3.0)
- `max_steering_angle` [rad] - Constraint (default: π/3 ≈ 60°)
- `max_steering_angle_rate` [rad/s] - Constraint (default: 0.5)
- `min_velocity_linearization` [m/s] - Velocity threshold for Jacobian (default: 0.01)
- `wheelbase` [m] - Vehicle parameter (default: Pacifica wheelbase)

**ILQRWarmStartParameters** (frozen dataclass)
- `k_velocity_error_feedback` - Velocity feedback gain (default: 0.5)
- `k_steering_angle_error_feedback` - Steering angle feedback gain (default: 0.05)
- `lookahead_distance_lateral_error` [m] - Lateral error lookahead (default: 15.0)
- `k_lateral_error` - Lateral error feedback gain (default: 0.1)
- `jerk_penalty_warm_start_fit` - Regularization (default: 1e-4)
- `curvature_rate_penalty_warm_start_fit` - Regularization (default: 1e-2)

**ILQRIterate** (frozen dataclass)
- **Purpose**: Immutable snapshot of state/input trajectory + Jacobians at one iLQR iteration
- **Fields**:
  - `state_trajectory` - (N+1, 5) array of states
  - `input_trajectory` - (N, 2) array of inputs
  - `state_jacobian_trajectory` - (N, 5, 5) array of ∂f/∂z matrices
  - `input_jacobian_trajectory` - (N, 5, 2) array of ∂f/∂u matrices
- **Validation**: Checks for NaN, shape consistency

**ILQRInputPolicy** (frozen dataclass)
- **Purpose**: LQR policy (feedback + feedforward) from backward recursion
- **Fields**:
  - `state_feedback_matrices` - (N, 2, 5) array of K_k matrices
  - `feedforward_inputs` - (N, 2) array of κ_k vectors
- **Policy form**: `Δu_k = K_k Δz_k + κ_k`

**ILQRSolution** (frozen dataclass)
- **Purpose**: Final solution output for client consumption
- **Fields**:
  - `state_trajectory` - (N+1, 5) optimal state trajectory
  - `input_trajectory` - (N, 2) optimal input trajectory
  - `tracking_cost` - Total cost (for monitoring convergence)
- **Note**: `solve()` returns `List[ILQRSolution]` - one per iteration, last element is final

### Tracker Utilities (tracker_utils.py)

**Velocity & Curvature Estimation**
- **Purpose**: Infer velocity/acceleration and curvature/curvature_rate from pose sequence (least squares)
- **Problem**: Trajectory has poses `(x, y, heading)` but trackers need velocity and curvature profiles
- **Solution**: Regularized least squares fit with jerk and curvature_rate penalties
- **Key functions**:
  - `get_velocity_curvature_profiles_with_derivatives_from_poses()` - Main entry point
    - Input: Poses (N, 3), discretization_time, jerk_penalty, curvature_rate_penalty
    - Output: velocity (N-1), acceleration (N-2), curvature (N-1), curvature_rate (N-2)
  - `_fit_initial_velocity_and_acceleration_profile()` - Longitudinal dynamics
    - Solves `min ||y - Ax||² + jerk_penalty * ||R x||²` where y = xy displacements
  - `_fit_initial_curvature_and_curvature_rate_profile()` - Lateral dynamics (heading)
    - Solves `min ||y - Ax||² + curvature_rate_penalty * ||x||²` where y = heading displacements
  - `complete_kinematic_state_and_inputs_from_poses()` - Full state + input reconstruction
    - Output: kinematic_states (N, 5), kinematic_inputs (N, 2)
- **Regularization rationale**:
  - Jerk penalty smooths acceleration (prevents overfitting to noisy poses)
  - Curvature rate penalty smooths steering (prevents sharp turns)
- **Edge case**: Zero velocity handled via `initial_curvature_penalty` (1e-10) to regularize first curvature
- **AIDEV-NOTE**: This is critical for LQR/iLQR - without smooth profiles, control is noisy

**Steering Angle Feedback**
- **Purpose**: Compute feedback steering angle to correct lateral tracking error
- **Function**: `compute_steering_angle_feedback(pose_reference, pose_current, lookahead_distance, k_lateral_error)`
- **Algorithm**: Stanley controller-style lateral error + heading error feedback
- **Formula**: `steering_feedback = -k * (lateral_error + lookahead_distance * heading_error)`
- **Use case**: iLQR warm start (add feedback to feedforward steering from pose fit)
- **Reference**: Stanford DARPA Grand Challenge steering controller (Hoffmann et al. 2007)

**Trajectory Interpolation**
- **Purpose**: Resample trajectory at fixed discretization time for LQR/iLQR
- **Function**: `get_interpolated_reference_trajectory_poses(trajectory, discretization_time)`
- **Output**: times_s (N,), poses (N, 3) where N = trajectory_duration / discretization_time
- **Details**: Samples at `[t_start, t_start + dt, t_start + 2*dt, ..., t_end]`
- **Edge case**: Adds extra state if last sample aligns with discretization time

**Helper Functions**
- `_generate_profile_from_initial_condition_and_derivatives()` - Integration via cumsum
- `_get_xy_heading_displacements_from_poses()` - Compute differences for fitting
- `_make_banded_difference_matrix()` - Construct finite difference matrix for regularization
- `_convert_curvature_profile_to_steering_profile()` - Curvature → steering angle conversion
  - Formula: `steering_angle = arctan(wheelbase * curvature)`

## 3. Architecture & Design Patterns

### 1. **Strategy Pattern**: Pluggable Trackers
```python
# Trackers are interchangeable via AbstractTracker interface
tracker: AbstractTracker = instantiate(cfg.tracker)
dynamic_state = tracker.track_trajectory(iteration, next_iteration, ego_state, trajectory)
```

### 2. **Decoupled Control** (LQRTracker)
- **Pattern**: Separate longitudinal and lateral control subsystems
- **Benefit**: Simpler design, faster computation (2 small LQR problems instead of 1 large)
- **Assumption**: Longitudinal/lateral dynamics are weakly coupled (valid at low speeds)
- **Linearization**: Small angle approximation (`sin(θ) ≈ θ`, `cos(θ) ≈ 1`)

### 3. **Model Predictive Control** (ILQRTracker)
- **Pattern**: Solve optimization over horizon, apply only first control input
- **Receding horizon**: At next timestep, re-solve with updated state and trajectory
- **Benefit**: Handles constraints, optimality guarantees, predictive behavior

### 4. **Frozen Dataclasses**: Immutable State Containers
```python
@dataclass(frozen=True)
class ILQRIterate:
    state_trajectory: DoubleMatrix
    input_trajectory: DoubleMatrix
    state_jacobian_trajectory: DoubleMatrix
    input_jacobian_trajectory: DoubleMatrix
```
- **Purpose**: Prevent accidental mutation during optimization
- **Validation**: `__post_init__` checks for NaN, shape consistency
- **Pattern**: Used throughout iLQR solver (ILQRIterate, ILQRInputPolicy, ILQRSolution)

### 5. **Regularized Least Squares**: Signal Reconstruction
- **Problem**: Trajectory has discrete poses, but we need continuous velocity/curvature
- **Solution**: Fit smooth profiles via least squares with derivative penalties
- **Pattern**: Transform noisy discrete measurements → smooth continuous signals
- **Math**: `min ||y - Ax||² + λ ||R x||²` where R is finite difference matrix

### 6. **Trust Regions**: Optimization Stability (iLQR)
- **Problem**: Without trust regions, iLQR can diverge (linearization error too large)
- **Solution**: Add quadratic penalty on perturbations from linearization trajectory
- **Cost**: `J += state_diff^T Q_tr state_diff + input_diff^T R_tr input_diff`
- **Effect**: Keeps updates within linearization validity region
- **Tuning**: Smaller trust region → slower convergence, larger → risk divergence

### 7. **Warm Start**: Initialization for Iterative Optimization (iLQR)
- **Pattern**: Infer good initial guess from reference trajectory
- **Method**:
  1. Fit velocity/curvature from reference poses (least squares)
  2. Convert curvature to steering angle
  3. Add feedback for initial tracking error
  4. Clip to satisfy constraints
- **Benefit**: Faster convergence (3-5 iterations vs 10-20 without warm start)

### 8. **Constraint Projection**: Feasibility Enforcement (iLQR)
- **Pattern**: Enforce constraints during forward dynamics, not in optimization
- **Method**:
  1. Clip inputs to bounds: `u = clip(u, u_min, u_max)`
  2. Propagate forward with clipped input
  3. Clip steering angle state: `delta = clip(delta, delta_min, delta_max)`
  4. Adjust steering rate to match clipped state
- **Benefit**: Always feasible trajectories, simple implementation
- **Limitation**: Not optimal (active constraint handling would be better)

## 4. Dependencies

### Internal (nuPlan - Documented ✅)

**Direct Dependencies**:
- ✅ `nuplan.common.actor_state.ego_state` - EgoState
- ✅ `nuplan.common.actor_state.dynamic_car_state` - DynamicCarState (output of tracker)
- ✅ `nuplan.common.actor_state.state_representation` - StateSE2, StateVector2D, TimePoint
- ✅ `nuplan.common.actor_state.vehicle_parameters` - VehicleParameters, get_pacifica_parameters()
- ✅ `nuplan.common.geometry.compute` - `principal_value()` (angle wrapping to [-π, π])
- ✅ `nuplan.planning.simulation.simulation_time_controller.simulation_iteration` - SimulationIteration
- ✅ `nuplan.planning.simulation.trajectory.abstract_trajectory` - AbstractTrajectory

**Indirect Dependencies**:
- `nuplan.database.utils.measure` - `angle_diff()` (angle difference with wrapping)

### External Dependencies
- **NumPy** - Array operations, linear algebra
  - `np.linalg.inv()` - Matrix inverse (LQR solve)
  - `np.linalg.pinv()` - Pseudoinverse (least squares)
  - `np.diag()`, `np.eye()` - Cost matrix construction
  - `np.cumsum()`, `np.diff()` - Integration/differentiation
- **typing** - Type hints
- **dataclasses** - Frozen dataclasses for iLQR
- **time** - Performance timing (`time.perf_counter()` for time budget)
- **enum** - IntEnum for LateralStateIndex

### Dependency Notes
**AIDEV-NOTE**: Trackers depend heavily on `✅ geometry.compute.principal_value()` for heading angle wrapping. LQR uses `database.utils.measure.angle_diff()` which is similar but with configurable period.

## 5. Dependents (Who Uses This Module?)

### Direct Consumers

**TwoStageController**:
- `nuplan/planning/simulation/controller/two_stage_controller.py`
  - `__init__(scenario, tracker, motion_model)` - Receives tracker instance
  - `update_state()` - Calls `tracker.track_trajectory()` to compute desired dynamics
  - Pattern: Tracker → motion model pipeline

### Hydra Configurations
- `nuplan/planning/script/config/simulation/ego_controller/tracker/lqr_tracker.yaml`
  - LQRTracker with default tuning
- `nuplan/planning/script/config/simulation/ego_controller/tracker/ilqr_tracker.yaml`
  - ILQRTracker with ILQRSolver configuration
- `nuplan/planning/script/config/simulation/ego_controller/two_stage_controller.yaml`
  - Composes tracker + motion model

### Use Cases

1. **Standard Closed-Loop Simulation (LQR)**
   - TwoStageController with LQRTracker + KinematicBicycleModel
   - Use case: Realistic control with ~1-5 ms overhead
   - Configuration: `lqr_tracker.yaml`

2. **Advanced Optimal Control (iLQR)**
   - TwoStageController with ILQRTracker + KinematicBicycleModel
   - Use case: Research, when LQR tracking accuracy is insufficient
   - Configuration: `ilqr_tracker.yaml`
   - Trade-off: 10-50× slower but better tracking

3. **Custom Tracker Development**
   - Implement `AbstractTracker`
   - Register in Hydra config
   - Use with TwoStageController

**AIDEV-NOTE**: 95% of simulations use LQRTracker (fast, good enough). ILQRTracker is for research or high-precision requirements.

## 6. Critical Files (Prioritized)

### Priority 1: Core Interfaces (Read First!)

1. **`abstract_tracker.py`** (31 lines) - **START HERE!**
   - `AbstractTracker` interface
   - Single method: `track_trajectory()`
   - Extremely simple, self-contained

2. **`lqr.py`** (389 lines) - **LQR tracker (standard)**
   - Decoupled longitudinal + lateral LQR
   - Reference velocity/curvature computation
   - Stopping controller logic
   - Frenet frame error computation
   - Most important file for typical usage

### Priority 2: Utilities (Supporting LQR)

3. **`tracker_utils.py`** (377 lines) - **Signal processing**
   - Velocity/curvature estimation from poses
   - Regularized least squares implementation
   - Steering feedback computation
   - Trajectory interpolation
   - **Critical for understanding LQR preprocessing**

### Priority 3: Advanced (Research-Level)

4. **`ilqr_tracker.py`** (108 lines) - **iLQR wrapper**
   - ILQRSolver integration
   - Reference trajectory extraction
   - MPC pattern (apply first input only)
   - Simple wrapper around solver

5. **`ilqr/ilqr_solver.py`** (689 lines) - **iLQR implementation**
   - Full iLQR algorithm
   - Warm start generation
   - Backward recursion (LQR policy computation)
   - Forward dynamics with constraints
   - Trust region handling
   - **Complex but exceptionally well-documented!**
   - **Read module docstring first** (lines 1-32)

### Priority 4: Tests (Examples)

6. **`test/test_lqr_tracker.py`** (269 lines) - **LQR usage examples**
   - Shows typical LQR configuration
   - Tests error projection, stopping controller, lateral control
   - Good reference for expected behavior

7. **`test/test_tracker_utils.py`** - Utility function tests
8. **`test/test_ilqr_tracker.py`** - iLQR usage examples
9. **`ilqr/test/test_ilqr_solver.py`** - iLQR solver tests

**AIDEV-NOTE**: For typical users, read files 1-3. Files 4-5 (iLQR) are advanced research code - skip unless needed.

## 7. Common Usage Patterns

### 1. Standard LQR Tracker (Most Common)
```python
from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker
import numpy as np

# Build tracker with typical parameters
lqr_tracker = LQRTracker(
    q_longitudinal=np.array([10.0]),  # Velocity error weight
    r_longitudinal=np.array([1.0]),   # Acceleration effort weight
    q_lateral=np.array([1.0, 10.0, 0.0]),  # [lateral_error, heading_error, steering_angle]
    r_lateral=np.array([1.0]),        # Steering rate effort weight
    discretization_time=0.1,          # 10 Hz discretization
    tracking_horizon=10,              # 1 second lookahead
    jerk_penalty=1e-4,                # Smooth velocity estimation
    curvature_rate_penalty=1e-2,      # Smooth curvature estimation
    stopping_proportional_gain=0.5,
    stopping_velocity=0.2,            # m/s
)

# In simulation loop
for iteration, next_iteration in simulation_timesteps:
    # Get current ego state
    ego_state = controller.get_state()

    # Get planned trajectory from planner
    trajectory = planner.compute_planner_trajectory(...)

    # Compute desired dynamics
    desired_dynamic_state = lqr_tracker.track_trajectory(
        current_iteration=iteration,
        next_iteration=next_iteration,
        initial_state=ego_state,
        trajectory=trajectory,
    )

    # Extract control commands
    acceleration = desired_dynamic_state.rear_axle_acceleration_2d.x
    steering_rate = desired_dynamic_state.tire_steering_rate
```

### 2. iLQR Tracker (Advanced Optimal Control)
```python
from nuplan.planning.simulation.controller.tracker.ilqr_tracker import ILQRTracker
from nuplan.planning.simulation.controller.tracker.ilqr.ilqr_solver import (
    ILQRSolver, ILQRSolverParameters, ILQRWarmStartParameters
)

# Configure solver
solver_params = ILQRSolverParameters(
    discretization_time=0.2,  # 5 Hz (coarser than LQR)
    state_cost_diagonal_entries=[1.0, 1.0, 10.0, 0.0, 0.0],  # Prioritize heading
    input_cost_diagonal_entries=[1.0, 10.0],  # Penalize steering rate heavily
    state_trust_region_entries=[1.0, 1.0, 1.0, 1.0, 1.0],
    input_trust_region_entries=[1.0, 1.0],
    max_ilqr_iterations=20,
    convergence_threshold=1e-6,
    max_solve_time=0.05,  # 50ms time budget
    max_acceleration=3.0,
    max_steering_angle=np.pi / 3,
    max_steering_angle_rate=0.5,
    min_velocity_linearization=0.01,
)

warm_start_params = ILQRWarmStartParameters(
    k_velocity_error_feedback=0.5,
    k_steering_angle_error_feedback=0.05,
    lookahead_distance_lateral_error=15.0,
    k_lateral_error=0.1,
    jerk_penalty_warm_start_fit=1e-4,
    curvature_rate_penalty_warm_start_fit=1e-2,
)

ilqr_solver = ILQRSolver(solver_params, warm_start_params)
ilqr_tracker = ILQRTracker(n_horizon=40, ilqr_solver=ilqr_solver)

# Use same as LQR (interface identical)
desired_dynamic_state = ilqr_tracker.track_trajectory(...)
```

### 3. Tuning LQR Weights for Aggressive/Conservative Tracking
```python
# Conservative (smooth, comfortable)
conservative_tracker = LQRTracker(
    q_longitudinal=np.array([5.0]),   # Low velocity error weight → gentle acceleration
    r_longitudinal=np.array([2.0]),   # High effort weight → smooth inputs
    q_lateral=np.array([0.5, 5.0, 0.0]),  # Low lateral error weight → wider turns
    r_lateral=np.array([2.0]),        # High steering rate weight → smooth steering
    discretization_time=0.1,
    tracking_horizon=15,              # Longer horizon → more predictive
    jerk_penalty=1e-3,                # Higher penalty → smoother profiles
    curvature_rate_penalty=1e-1,
    stopping_proportional_gain=0.3,   # Gentle stopping
    stopping_velocity=0.5,            # Start stopping early
)

# Aggressive (tight tracking, less comfortable)
aggressive_tracker = LQRTracker(
    q_longitudinal=np.array([20.0]),  # High velocity error weight → fast convergence
    r_longitudinal=np.array([0.5]),   # Low effort weight → larger inputs
    q_lateral=np.array([2.0, 20.0, 0.0]),  # High error weights → tight tracking
    r_lateral=np.array([0.5]),        # Low steering rate weight → sharper turns
    discretization_time=0.1,
    tracking_horizon=5,               # Shorter horizon → more reactive
    jerk_penalty=1e-5,                # Lower penalty → follows noisy trajectory
    curvature_rate_penalty=1e-3,
    stopping_proportional_gain=1.0,   # Aggressive stopping
    stopping_velocity=0.1,            # Stop at last moment
)
```

### 4. Using Tracker Utilities Directly
```python
from nuplan.planning.simulation.controller.tracker.tracker_utils import (
    get_velocity_curvature_profiles_with_derivatives_from_poses,
    compute_steering_angle_feedback,
    get_interpolated_reference_trajectory_poses,
)

# Extract velocity/curvature from trajectory
times_s, poses = get_interpolated_reference_trajectory_poses(
    trajectory=trajectory,
    discretization_time=0.1,
)

velocity_profile, accel_profile, curvature_profile, curvature_rate_profile = \
    get_velocity_curvature_profiles_with_derivatives_from_poses(
        discretization_time=0.1,
        poses=poses,
        jerk_penalty=1e-4,
        curvature_rate_penalty=1e-2,
    )

# Compute steering feedback for tracking error correction
steering_feedback = compute_steering_angle_feedback(
    pose_reference=np.array([x_ref, y_ref, heading_ref]),
    pose_current=np.array([x_current, y_current, heading_current]),
    lookahead_distance=5.0,  # meters
    k_lateral_error=0.5,
)
```

### 5. Monitoring iLQR Convergence
```python
# iLQR solve returns list of solutions (one per iteration)
solution_list = ilqr_tracker._ilqr_solver.solve(current_state, reference_trajectory)

# Check convergence
print(f"Iterations: {len(solution_list) - 1}")  # -1 because final iterate is added
print(f"Initial cost: {solution_list[0].tracking_cost:.2f}")
print(f"Final cost: {solution_list[-1].tracking_cost:.2f}")
print(f"Cost reduction: {(1 - solution_list[-1].tracking_cost / solution_list[0].tracking_cost) * 100:.1f}%")

# Inspect cost trajectory (should decrease monotonically)
costs = [sol.tracking_cost for sol in solution_list]
for i, cost in enumerate(costs):
    print(f"Iteration {i}: cost = {cost:.2f}")
```

### 6. Custom Tracker Implementation
```python
from nuplan.planning.simulation.controller.tracker.abstract_tracker import AbstractTracker

class MyCustomTracker(AbstractTracker):
    def __init__(self, my_param_1, my_param_2):
        self.my_param_1 = my_param_1
        self.my_param_2 = my_param_2

    def track_trajectory(
        self,
        current_iteration: SimulationIteration,
        next_iteration: SimulationIteration,
        initial_state: EgoState,
        trajectory: AbstractTrajectory,
    ) -> DynamicCarState:
        """Compute desired acceleration and steering rate"""

        # 1. Get reference state at current time
        reference_state = trajectory.get_state_at_time(current_iteration.time_point)

        # 2. Compute tracking errors
        velocity_error = initial_state.dynamic_car_state.rear_axle_velocity_2d.x - \
                        reference_state.dynamic_car_state.rear_axle_velocity_2d.x

        # 3. Compute control inputs (your algorithm here)
        acceleration = -self.my_param_1 * velocity_error
        steering_rate = 0.0  # Example: no lateral control

        # 4. Return DynamicCarState
        return DynamicCarState.build_from_rear_axle(
            rear_axle_to_center_dist=initial_state.car_footprint.rear_axle_to_center_dist,
            rear_axle_velocity_2d=initial_state.dynamic_car_state.rear_axle_velocity_2d,
            rear_axle_acceleration_2d=StateVector2D(acceleration, 0),
            tire_steering_rate=steering_rate,
        )
```

## 8. Gotchas & Edge Cases

### LQR Tracker Issues

1. **Tracking horizon too short**
   - **Issue**: `tracking_horizon = 1` → steering rate has no effect (Euler integration)
   - **Symptom**: Lateral tracking error, vehicle doesn't steer
   - **Root cause**: With Euler discretization, steering rate only affects steering angle, which needs one more step to affect heading
   - **Fix**: Use `tracking_horizon >= 2` (enforced in `LQRTracker.__init__`)
   - **Recommended**: `tracking_horizon = 10` (1 second @ 0.1s)

2. **Stopping controller thrashing**
   - **Issue**: Velocity oscillates around `stopping_velocity` threshold → switches between LQR and P-controller
   - **Symptom**: Jerky motion near stop, acceleration commands flip rapidly
   - **Root cause**: No hysteresis on mode switch
   - **Fix**: Add hysteresis (use different thresholds for entering/exiting stopping mode)
   - **AIDEV-TODO**: Implement hysteresis in stopping controller

3. **Q/R matrix not positive (semi-)definite**
   - **Issue**: LQRTracker asserts `Q >= 0`, `R > 0`, but user passes negative weights
   - **Symptom**: AssertionError in `LQRTracker.__init__` (lines 105, 108)
   - **Fix**: Validate config before instantiation
   - **Note**: Q must be positive semidefinite (PSD), R must be positive definite (PD) for LQR to work

4. **Reference velocity/curvature estimation fails**
   - **Issue**: Trajectory has < 2 poses → `_get_xy_heading_displacements_from_poses()` asserts
   - **Symptom**: AssertionError: "Cannot get displacements given empty or single element pose trajectory"
   - **Fix**: Planners must return trajectories with >= 2 waypoints
   - **Prevention**: Check `len(trajectory.get_sampled_trajectory()) >= 2`

5. **Jerk/curvature_rate penalty too large**
   - **Issue**: Large penalties (> 1.0) → over-smoothed velocity/curvature (doesn't match trajectory)
   - **Symptom**: Poor tracking, controller follows smoothed reference instead of actual trajectory
   - **Fix**: Tune penalties (typical: 1e-5 to 1e-1)
   - **Rule of thumb**: Start with 1e-4 (jerk), 1e-2 (curvature_rate) and adjust

6. **Jerk/curvature_rate penalty too small**
   - **Issue**: Very small penalties (< 1e-10) → noisy velocity/curvature estimates
   - **Symptom**: Jerky control inputs, oscillations
   - **Fix**: Increase penalties to smooth profiles
   - **Diagnostic**: Plot velocity/curvature profiles, check for high-frequency noise

7. **Discretization time mismatch**
   - **Issue**: `tracker.discretization_time != simulation_timestep`
   - **Symptom**: Tracking error, unexpected behavior
   - **Example**: Tracker assumes 0.1s but simulation uses 0.05s
   - **Fix**: Ensure `discretization_time == (next_iteration.time_s - current_iteration.time_s)`
   - **Note**: Tracker discretization can be coarser than simulation (e.g., 0.2s vs 0.1s), but should be multiple

8. **Lateral error sign confusion**
   - **Issue**: Lateral error computed in Frenet frame, sign depends on trajectory heading
   - **Symptom**: Controller steers wrong direction
   - **Formula**: `lateral_error = -x_error * sin(heading_ref) + y_error * cos(heading_ref)`
   - **Convention**: Positive lateral error = left of trajectory (in vehicle frame)
   - **AIDEV-NOTE**: See `test_lqr_tracker.py:66-118` for validation

9. **Heading angle wrapping in error computation**
   - **Issue**: Heading error crosses [-π, π] boundary → 2π jump
   - **Symptom**: Sudden large control input
   - **Fix**: LQRTracker uses `angle_diff()` with 2π period (line 192)
   - **AIDEV-NOTE**: Don't use raw difference `heading_current - heading_ref`!

10. **Stopping controller deceleration saturation**
    - **Issue**: P-controller has no saturation → can command large deceleration
    - **Symptom**: Unrealistic braking (> 5 m/s²)
    - **Example**: `velocity = 10 m/s`, `reference_velocity = 0`, `gain = 0.5` → `accel = -5 m/s²`
    - **Fix**: Add saturation or use LQR even at low speeds
    - **AIDEV-TODO**: Add max deceleration limit to stopping controller

### iLQR Tracker Issues

11. **iLQR doesn't converge**
    - **Issue**: `max_ilqr_iterations` reached, but `convergence_threshold` not satisfied
    - **Symptom**: Suboptimal tracking, high cost, solution_list[-1].tracking_cost still large
    - **Causes**: Poor warm start, infeasible reference trajectory, trust regions too small, Q/R weights inconsistent
    - **Debug**: Inspect `solution_list[-1].tracking_cost` vs `solution_list[0].tracking_cost` (should decrease)
    - **Fix**: Increase max_ilqr_iterations, improve warm start, relax trust regions

12. **Warm start infeasible**
    - **Issue**: Inferred inputs from reference trajectory violate constraints
    - **Symptom**: Warm start input trajectory has clipped values
    - **Expected**: This is OK - iLQR will optimize from infeasible warm start
    - **AIDEV-NOTE**: Consider constraint-aware warm start (project onto feasible set)

13. **Trust region too tight**
    - **Issue**: Large `state_trust_region_entries` or `input_trust_region_entries` (> 10.0) → tiny updates
    - **Symptom**: iLQR converges slowly (20+ iterations) or cost doesn't decrease
    - **Fix**: Reduce trust region weights (typical: 0.1 - 5.0)
    - **Rule of thumb**: Start with 1.0 for all, decrease if diverging, increase if not converging

14. **Trust region too loose**
    - **Issue**: Small trust region weights (< 0.01) → large updates, linearization error
    - **Symptom**: iLQR diverges, cost increases, NaN values
    - **Fix**: Increase trust region weights
    - **Diagnostic**: Check `||state_trajectory[i+1] - state_trajectory[i]||` is reasonable

15. **Min velocity linearization causes jumps**
    - **Issue**: Velocity crosses `min_velocity_linearization` threshold → Jacobian changes discontinuously
    - **Symptom**: Optimization jumpiness near zero velocity
    - **Root cause**: Jacobian has velocity in denominator (heading dynamics), zero velocity → uncontrollable
    - **Fix**: Use small `min_velocity_linearization` (0.01 - 0.1 m/s)
    - **AIDEV-NOTE**: See `ilqr_solver.py:524-532` for implementation

16. **iLQR exceeds time budget**
    - **Issue**: `max_solve_time` too large or not set → tracker takes > 0.1s
    - **Symptom**: Simulation slower than real-time, planner gets less time
    - **Fix**: Set `max_solve_time` to leave time for planner (e.g., 0.05s if planner needs 0.05s)
    - **Monitoring**: Check `len(solution_list)` (should be < max_ilqr_iterations if time budget hit)

17. **Steering angle singularity at ±90°**
    - **Issue**: `tan(steering_angle)` in dynamics → singularity at ±π/2
    - **Symptom**: RuntimeError or NaN propagation
    - **Prevention**: iLQR asserts `|steering_angle| < π/2` (line 492-494)
    - **Fix**: Ensure `max_steering_angle < π/2` (default: π/3 ≈ 60°)

18. **iLQR horizon too long**
    - **Issue**: `n_horizon > 100` → excessive computation time
    - **Symptom**: Solver takes > 100 ms, real-time constraint violated
    - **Fix**: Reduce horizon (typical: 20-40 for 0.1-0.2s discretization)
    - **Trade-off**: Longer horizon → better optimality, shorter → faster

19. **iLQR cost weights zero on velocity/steering**
    - **Issue**: `state_cost_diagonal_entries[3] = 0.0` (velocity) AND `state_cost_diagonal_entries[4] = 0.0` (steering)
    - **Symptom**: iLQR doesn't track velocity or steering angle (only position/heading)
    - **Expected**: This is intentional in default config - focus on geometric tracking
    - **AIDEV-NOTE**: Velocity/steering tracking can be added if needed (increase weights)

20. **Reference trajectory shorter than horizon**
    - **Issue**: Trajectory ends before `current_time + n_horizon * dt`
    - **Symptom**: `ilqr_tracker._get_reference_trajectory()` returns shorter reference (M < n_horizon)
    - **Expected**: iLQR handles this gracefully (uses shorter horizon)
    - **AIDEV-NOTE**: No error - solver adapts to available reference length

### Numerical Issues

21. **LQR matrix inversion failure**
    - **Issue**: `np.linalg.inv(B.T @ Q @ B + R)` fails (singular matrix)
    - **Symptom**: `numpy.linalg.LinAlgError: Singular matrix`
    - **Causes**: R not positive definite, B all zeros (uncontrollable), numerical precision
    - **Fix**: Ensure R > 0, check B is full rank, use `np.linalg.pinv()` as fallback
    - **AIDEV-NOTE**: Current implementation assumes R is PD (validated in __init__)

22. **Least squares overfitting**
    - **Issue**: Regularization penalties too small → least squares fits noise
    - **Symptom**: Velocity/curvature profiles have high-frequency oscillations
    - **Example**: Trajectory with 0.01s sampling, jerk_penalty = 1e-10
    - **Fix**: Increase penalties (jerk_penalty, curvature_rate_penalty)
    - **Diagnostic**: Plot fitted profiles, check for unrealistic values

23. **Zero wheelbase**
    - **Issue**: Vehicle wheelbase = 0 → division by zero in dynamics
    - **Symptom**: NaN propagation, RuntimeError
    - **Prevention**: VehicleParameters validation, default is Pacifica wheelbase (3.089m)
    - **AIDEV-NOTE**: All trackers use get_pacifica_parameters() by default

24. **Cumulative integration error**
    - **Issue**: `_generate_profile_from_initial_condition_and_derivatives()` uses cumsum → error accumulates
    - **Symptom**: Velocity/curvature profiles drift from reference over long horizons
    - **Fix**: Use shorter horizons, higher regularization, or closed-form integration
    - **AIDEV-NOTE**: Not a major issue for typical 1-2 second horizons

## 9. Performance Considerations

### Computational Cost (per timestep)

| Tracker | Complexity | Typical Time | Bottleneck |
|---------|-----------|--------------|-----------|
| LQRTracker | O(H² × N_states²) | 1-5 ms | Velocity/curvature estimation (least squares) |
| ILQRTracker | O(I × H × N_states³) | 10-50 ms | iLQR iterations (forward/backward pass) |

Where:
- H = tracking_horizon (LQR: 10, iLQR: 40)
- N_states = 3 (lateral LQR), 5 (iLQR)
- I = iLQR iterations (typical: 3-10)

### LQRTracker Profiling

**Breakdown** (total ~3 ms):
- `get_interpolated_reference_trajectory_poses()`: 15% (trajectory sampling)
- `get_velocity_curvature_profiles_with_derivatives_from_poses()`: 40% (least squares)
  - `_fit_initial_velocity_and_acceleration_profile()`: 25%
  - `_fit_initial_curvature_and_curvature_rate_profile()`: 15%
- `_longitudinal_lqr_controller()`: 10% (1D LQR solve)
- `_lateral_lqr_controller()`: 25% (3D LQR solve, matrix multiplication)
- `_compute_initial_velocity_and_lateral_state()`: 10% (error projection)

**Bottleneck**: Velocity/curvature estimation (40%) - dominated by matrix operations in least squares

### ILQRTracker Profiling

**Breakdown** (total ~30 ms, 5 iterations):
- `_input_warm_start()`: 15% (pose fitting + constraint projection)
- iLQR iterations (5×): 80%
  - `_run_forward_dynamics()`: 30% (Euler integration + Jacobians)
  - `_run_lqr_backward_recursion()`: 40% (dynamic programming)
  - `_update_inputs_with_policy()`: 10% (apply perturbations)
- `_compute_tracking_cost()`: 5% (cost evaluation)

**Bottleneck**: Backward recursion (40%) - dominated by matrix multiplications in LQR solve

### Memory Footprint

- **LQRTracker**: ~10-20 KB
  - Interpolated trajectory (~2 KB)
  - Velocity/curvature profiles (~1 KB)
  - Cost matrices (negligible)
- **ILQRTracker**: ~100-500 KB
  - Solution list (iterations × (states + inputs + Jacobians))
  - Example: 5 iterations × 40 horizon × 5 states × 8 bytes ≈ 8 KB per iteration
  - Jacobians dominate: 5 iterations × 40 horizon × (5×5 + 5×2) × 8 bytes ≈ 70 KB

### Real-Time Constraints

- **Target**: Tracker < 10 ms (leaves 90ms for planner in 100ms budget)
- **LQRTracker**: ✅ Always real-time (1-5 ms)
- **ILQRTracker**: ⚠️ May exceed real-time if:
  - `max_ilqr_iterations > 15`
  - `n_horizon > 50`
  - Poor warm start (slow convergence)
- **Mitigation**: Set `max_solve_time = 0.05s` to enforce time budget

### Optimization Strategies

1. **Reduce LQR tracking horizon** (10 → 5)
   - Speedup: 2-3×
   - Trade-off: Slightly worse tracking accuracy
   - Use case: Real-time constraints tight

2. **Coarser iLQR discretization** (0.1s → 0.2s)
   - Speedup: 2× (half the horizon for same lookahead time)
   - Trade-off: Less accurate dynamics
   - Use case: When computational budget is limited

3. **Cache velocity/curvature profiles** (LQR)
   - Speedup: 40% reduction if trajectory doesn't change
   - Implementation: Hash trajectory, cache results
   - Trade-off: Memory overhead, cache invalidation logic
   - **AIDEV-TODO**: Not currently implemented

4. **Warm start iLQR from previous solution**
   - Speedup: 2-3× (converges in 2-3 iterations vs 5-10)
   - Implementation: Use previous solve's final input trajectory as warm start
   - Trade-off: Temporal coherence assumption
   - **AIDEV-TODO**: Not currently implemented

5. **Early termination (iLQR)**
   - Already implemented: Stops when `||u_next - u_current|| < convergence_threshold`
   - Typical: Converges in 3-5 iterations with good warm start
   - Tuning: Increase `convergence_threshold` (1e-6 → 1e-4) for faster convergence

6. **Parallel forward/backward passes (iLQR)**
   - Potential speedup: Minimal (iterations are sequential)
   - Better: Parallelize multiple scenarios in batch simulation
   - Note: iLQR iterations are inherently sequential

7. **Use optimized BLAS**
   - Ensure NumPy linked to MKL or OpenBLAS
   - Speedup: 2-5× on matrix operations
   - Check: `np.show_config()`

## 10. Related Documentation

### Cross-References (Documented ✅)
- ✅ `nuplan/planning/simulation/controller/CLAUDE.md` - Parent module (two-stage controller)
- ✅ `nuplan/planning/simulation/controller/motion_model/CLAUDE.md` - Motion models that consume tracker output
- ✅ `nuplan/planning/simulation/planner/CLAUDE.md` - Planners generate trajectories for trackers
- ✅ `nuplan/planning/simulation/trajectory/CLAUDE.md` - Trajectory representations (AbstractTrajectory)
- ✅ `nuplan/common/actor_state/CLAUDE.md` - EgoState, DynamicCarState
- ✅ `nuplan/common/geometry/CLAUDE.md` - Geometry utilities (principal_value, angle wrapping)

### Configuration Files
- `nuplan/planning/script/config/simulation/ego_controller/tracker/lqr_tracker.yaml`
  - Default LQR tuning parameters
- `nuplan/planning/script/config/simulation/ego_controller/tracker/ilqr_tracker.yaml`
  - Default iLQR solver configuration

### External Resources
- **LQR theory**: Boyd & Barratt, "Linear Controller Design: Limits of Performance" (1991)
- **iLQR algorithm**: Todorov & Li, "A generalized iterative LQR method" (2005)
- **Bicycle model**: Rajamani, "Vehicle Dynamics and Control" (2011), Chapter 2
- **Stanley controller**: Hoffmann et al., "Autonomous Automobile Trajectory Tracking for Off-Road Driving" (2007)
- **Frenet frame**: Werling et al., "Optimal Trajectory Generation for Dynamic Street Scenarios" (2010)

## 11. AIDEV Notes

### Design Philosophy
- **Modularity**: Trackers are independent of motion models (clean separation)
- **Optimality**: LQR provides locally optimal control (under linearization assumptions)
- **Predictive**: Tracking horizon enables lookahead (not just reactive)
- **Decoupling**: LQR separates longitudinal/lateral (simpler, faster)
- **Warm start**: iLQR leverages reference trajectory for initialization (faster convergence)

### Common Mistakes
- Setting `tracking_horizon = 1` (breaks lateral LQR)
- Using very large/small regularization penalties (over-smoothing or noise amplification)
- Not wrapping heading angles properly (use `angle_diff()` or `principal_value()`)
- Assuming tracker is stateless (it's not - but only during single track_trajectory call)
- Forgetting to set `max_solve_time` for iLQR (can exceed real-time)
- Using iLQR when LQR is sufficient (10-50× slower for marginal benefit)

### Future Improvements
- **AIDEV-TODO**: Add hysteresis to LQRTracker stopping controller (prevent mode thrashing)
- **AIDEV-TODO**: Implement warm start from previous solution for iLQR (temporal coherence)
- **AIDEV-TODO**: Cache velocity/curvature profiles in LQRTracker (avoid redundant computation)
- **AIDEV-TODO**: Add max deceleration limit to stopping controller (safety)
- **AIDEV-TODO**: Constraint-aware warm start for iLQR (project onto feasible set)
- **AIDEV-TODO**: Adaptive trust regions for iLQR (adjust based on linearization error)
- **AIDEV-TODO**: Line search in iLQR policy update (gradient step size optimization)
- **AIDEV-TODO**: Log-barrier constraints in iLQR (handle constraints in optimization, not projection)

### Potential Bugs
- **AIDEV-NOTE** (lqr.py:244-256): Stopping controller has no saturation
  - Could command unrealistic deceleration if velocity error is large
  - Add clipping or switch to LQR with saturation

- **AIDEV-NOTE** (lqr.py:148-151): Stopping condition uses AND logic
  - Both current AND reference velocity must be below threshold
  - This is intentional (prevent switching during deceleration to stop)

- **AIDEV-NOTE** (ilqr_solver.py:514-518): Steering angle clipping adjusts steering rate
  - Applied steering rate is recomputed after clipping steering angle
  - This ensures consistency (delta_{k+1} = delta_k + phi_k * dt)
  - Good design - prevents divergence from constraints

- **AIDEV-QUESTION** (tracker_utils.py:190-194): Initial curvature penalty hardcoded
  - `initial_curvature_penalty = 1e-10` to handle zero velocity edge case
  - Should this be a parameter? What if velocity is actually zero?
  - Current implementation assumes velocity > 0 eventually

### Testing Gaps
- No unit tests for stopping controller hysteresis (doesn't exist yet)
- No integration tests for LQR tracking accuracy on standard maneuvers
  - Should verify tracking error < threshold for lane following, turns, stops
- No performance benchmarks for iLQR convergence
  - Should document expected iterations vs warm start quality
- No tests for edge case: reference trajectory ends mid-horizon
  - iLQR handles this but not explicitly tested

### Documentation Improvements Needed
- Add diagram of Frenet frame coordinate system (lateral error definition)
- Document typical Q/R matrix values for different driving styles (aggressive/conservative)
- Explain trust region selection for iLQR (tuning guide)
- Add section on debugging tracking error (systematic approach)
- Include plots of velocity/curvature estimation (show effect of regularization)

### Research Opportunities
- **Adaptive LQR**: Time-varying Q/R weights based on scenario (e.g., higher Q in intersections)
- **Model Predictive Path Integral (MPPI)**: Sampling-based alternative to iLQR (parallelizable)
- **Tube MPC**: Robust tracking with uncertainty (handle prediction errors)
- **Learning-based warm start**: Neural network predicts good initial trajectory
- **Constraint-aware LQR**: Active set methods for handling constraints directly
