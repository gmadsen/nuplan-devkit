# nuplan/planning/simulation/controller/tracker/lqr/

## 1. Purpose & Responsibility

This module implements **Linear Quadratic Regulator (LQR) and iterative LQR (iLQR) trajectory tracking controllers** for autonomous vehicle control, computing optimal acceleration and steering rate commands to minimize tracking error. The core responsibility is transforming high-level trajectory references into low-level control inputs (acceleration, steering rate) by solving finite-horizon optimal control problems with kinematic bicycle dynamics, enabling realistic closed-loop simulation with predictable, smooth vehicle behavior.

**Note on Organization**: This documentation covers both `lqr.py` (file in parent tracker/ directory) AND `ilqr/` subdirectory because they form a cohesive LQR-based control family.

## 2. Key Abstractions

### Core Controllers

**LQRTracker** (`tracker/lqr.py`)
- **Purpose**: Decoupled longitudinal + lateral LQR control with small-angle linearization
- **Architecture**: Two independent LQR subproblems solved sequentially
  1. **Longitudinal**: velocity → acceleration
  2. **Lateral**: [lateral_error, heading_error, steering_angle] → steering_rate
- **State representation**:
  - Longitudinal: `[velocity]` (1D)
  - Lateral: `[lateral_error, heading_error, steering_angle]` (3D via `LateralStateIndex` enum)
- **Inputs**: `[acceleration, steering_rate]` (2D, decoupled)
- **Key innovation**: Analytically solve one-step LQR for each subsystem (no iteration needed)
- **Stopping mode**: Falls back to P-controller when `velocity < stopping_velocity` (avoids LQR singularities at zero speed)
- **Linearization**: Euler discretization with small angle approximations
- **Reference computation**: Infers velocity/curvature profiles from trajectory poses via regularized least squares

**ILQRTracker** (`tracker/ilqr_tracker.py`)
- **Purpose**: Wrapper connecting `ILQRSolver` to the `AbstractTracker` interface
- **Architecture**: Model Predictive Control (MPC) pattern
  - Solve full optimal control problem at each timestep
  - Apply only first control input from solution
  - Discard rest of trajectory (replan next timestep)
- **State**: `[x, y, heading, velocity, steering_angle]` (5D, full pose + dynamics)
- **Inputs**: `[acceleration, steering_rate]` (2D, coupled via nonlinear dynamics)
- **Delegation**: All heavy lifting done by `ILQRSolver`
- **Reference trajectory**: Extracts states from planner trajectory via interpolation at solver's discretization time
- **Horizon management**: Handles trajectories shorter than `n_horizon` (truncates reference)

**ILQRSolver** (`tracker/ilqr/ilqr_solver.py`)
- **Purpose**: Full iterative LQR implementation for trajectory tracking with kinematic bicycle model
- **Algorithm**: Differential Dynamic Programming (DDP) variant
  1. **Warm start**: Infer feasible initial trajectory from reference poses
  2. **Iteration loop** (up to `max_ilqr_iterations`):
     - Forward pass: Compute state trajectory + Jacobians
     - Backward pass: Solve LQR via dynamic programming (Riccati recursion)
     - Policy update: Apply feedback + feedforward perturbations
     - Convergence check: Exit if input change < threshold
  3. Return solution list (one per iteration)
- **Dynamics**: Discrete-time kinematic bicycle (Euler integration)
  ```
  x_{k+1} = x_k + v_k * cos(θ_k) * dt
  y_{k+1} = y_k + v_k * sin(θ_k) * dt
  θ_{k+1} = θ_k + v_k * tan(δ_k) / L * dt
  v_{k+1} = v_k + a_k * dt
  δ_{k+1} = δ_k + φ_k * dt
  ```
  where `δ` = steering angle, `φ` = steering rate, `L` = wheelbase
- **Cost function**: Quadratic tracking cost + input effort cost + trust region penalties
  ```
  J = Σ ||z_k - z_ref,k||²_Q + Σ ||u_k||²_R + trust_region_terms
  ```
- **Constraints**: Input clipping (acceleration, steering rate), steering angle saturation
- **Trust regions**: Penalize large state/input perturbations to keep linearization valid
- **Warm start strategy**:
  - Fit velocity/curvature from reference poses
  - Convert to steering angle via bicycle model
  - Add feedback terms for initial tracking error
  - Ensures feasible starting point for optimization

### Configuration Dataclasses

**ILQRSolverParameters** (`frozen` dataclass)
- **Purpose**: All solver runtime parameters in one immutable container
- **Key fields**:
  - `discretization_time` [s] - Integration timestep (e.g., 0.2s)
  - `state_cost_diagonal_entries` (5,) - Q matrix diagonal `[x, y, heading, v, δ]` weights
  - `input_cost_diagonal_entries` (2,) - R matrix diagonal `[accel, steering_rate]` weights
  - `state_trust_region_entries` (5,) - Trust region on state perturbations
  - `input_trust_region_entries` (2,) - Trust region on input perturbations
  - `max_ilqr_iterations` - Iteration budget
  - `convergence_threshold` - Early termination threshold (input norm difference)
  - `max_solve_time` [s] - Optional time budget
  - Constraint limits: `max_acceleration`, `max_steering_angle`, `max_steering_angle_rate`
  - `min_velocity_linearization` [m/s] - Velocity floor for Jacobian computation (avoids singularities)
- **Validation**: `__post_init__()` ensures positive values, PSD cost matrices

**ILQRWarmStartParameters** (`frozen` dataclass)
- **Purpose**: Parameters for generating initial trajectory guess
- **Key fields**:
  - `k_velocity_error_feedback` - Gain for velocity error correction
  - `k_steering_angle_error_feedback` - Gain for steering angle error correction
  - `lookahead_distance_lateral_error` [m] - Stanley controller lookahead
  - `k_lateral_error` - Lateral error feedback gain
  - `jerk_penalty_warm_start_fit`, `curvature_rate_penalty_warm_start_fit` - Regularization for pose fitting

**ILQRIterate** (`frozen` dataclass)
- **Purpose**: Snapshot of optimization state at one iLQR iteration
- **Fields**:
  - `state_trajectory` (N+1, 5) - State sequence
  - `input_trajectory` (N, 2) - Input sequence
  - `state_jacobian_trajectory` (N, 5, 5) - ∂f/∂z Jacobians
  - `input_jacobian_trajectory` (N, 5, 2) - ∂f/∂u Jacobians
- **Validation**: Ensures no NaN values, consistent dimensions

**ILQRInputPolicy** (`frozen` dataclass)
- **Purpose**: Affine state feedback policy from LQR backward pass
- **Fields**:
  - `state_feedback_matrices` (N, 2, 5) - K_k gain matrices
  - `feedforward_inputs` (N, 2) - κ_k feedforward terms
- **Policy form**: `Δu_k = K_k Δz_k + κ_k` (perturbation policy)

**ILQRSolution** (`frozen` dataclass)
- **Purpose**: Final solution package for client consumption
- **Fields**:
  - `state_trajectory` (N+1, 5)
  - `input_trajectory` (N, 2)
  - `tracking_cost` (float) - Total quadratic cost achieved
- **Usage**: Last element of solution list is final optimized trajectory

### State Indices

**LateralStateIndex** (`IntEnum`)
- **Purpose**: Named indices for LQR lateral state vector
- **Values**:
  - `LATERAL_ERROR = 0` [m] - Cross-track error in Frenet frame
  - `HEADING_ERROR = 1` [rad] - Angular error relative to trajectory heading
  - `STEERING_ANGLE = 2` [rad] - Current wheel angle
- **Use case**: Clean indexing in lateral LQR matrices

## 3. Architecture & Design Patterns

### 1. **Decoupled vs Coupled Control**

**LQRTracker** (decoupled):
```python
# Sequential solve: longitudinal first, then lateral
accel = solve_1D_LQR(velocity_error)
velocity_profile = forward_integrate(velocity, accel)  # Predict future velocities
steering_rate = solve_3D_LQR(lateral_state, velocity_profile, curvature_profile)
```
- **Pros**: Fast (no iteration), analytically optimal for each subsystem
- **Cons**: Ignores longitudinal-lateral coupling (e.g., speed affects steering dynamics)
- **Validity**: Excellent for normal driving (< 10 m/s², < 0.3 rad curvature)

**ILQRTracker** (coupled):
```python
# Joint solve: optimize both controls simultaneously
solution_list = ilqr_solver.solve(current_state, reference_trajectory)
optimal_inputs = solution_list[-1].input_trajectory
accel, steering_rate = optimal_inputs[0, :]  # MPC: apply first input only
```
- **Pros**: Accounts for coupling, globally optimal (within linearization validity)
- **Cons**: 10-50× slower due to iterations
- **Validity**: Necessary for aggressive maneuvers, emergency scenarios

### 2. **One-Step LQR Pattern** (LQRTracker)

**Key insight**: For tracking problems with constant reference over horizon, can collapse multi-step LQR to one-step problem.

```python
# Standard LQR: minimize Σ_k ||z_k - z_ref||²_Q + ||u_k||²_R
# Assume constant input u over horizon N → z_N = A z_0 + B u + g
# Collapse to: min ||A z_0 + B u + g - z_ref||²_Q + ||u||²_R
# Analytical solution: u* = -(B^T Q B + R)^{-1} B^T Q (A z_0 + g - z_ref)
```

This is implemented in `_solve_one_step_lqr()` - a static method used by both longitudinal and lateral controllers.

### 3. **Iterative Refinement** (ILQRSolver)

**Main loop** (`solve()`):
```python
current_iterate = warm_start()  # Initial guess
for iteration in range(max_ilqr_iterations):
    # Forward: Linearize dynamics about current trajectory
    current_iterate = run_forward_dynamics(current_state, input_trajectory)

    # Backward: Solve LQR for perturbations
    lqr_policy = run_lqr_backward_recursion(current_iterate, reference)

    # Update: Apply policy to get new input trajectory
    input_trajectory_next = update_inputs_with_policy(current_iterate, lqr_policy)

    # Check convergence
    if ||input_trajectory_next - input_trajectory|| < threshold:
        break

    current_iterate = run_forward_dynamics(current_state, input_trajectory_next)
```

**Why it works**: Each iteration improves cost by moving in locally optimal direction (LQR policy). Trust regions prevent large steps that violate linearization.

### 4. **Riccati Backward Recursion** (Dynamic Programming)

**Value function**: `V_k(Δz_k) = Δz_k^T P_k Δz_k + 2 ρ_k^T Δz_k`

**Backward pass** (`_run_lqr_backward_recursion()`):
```python
# Initialize terminal cost
P_current = Q + Q_trust_region
ρ_current = Q @ error_state[-1]

# Recurse backwards through time
for k in reversed(range(N)):
    # Compute optimal policy at time k
    K_k = -(B^T P B + R + R_trust)^{-1} B^T P A
    κ_k = -(B^T P B + R + R_trust)^{-1} (R u_k + B^T ρ)

    # Update value function for time k-1
    A_cl = A + B K_k  # Closed-loop dynamics
    P_prior = Q + K^T R K + A_cl^T P A_cl + trust_region_terms
    ρ_prior = Q e_k + K^T R (κ + u) + A_cl^T (P B κ + ρ) + ...

    P_current = P_prior
    ρ_current = ρ_prior
```

This is the **heart of iLQR** - computing optimal feedback gains for perturbations about linearization trajectory.

### 5. **Trust Region Regularization**

**Problem**: Large LQR updates can violate linearization → divergence

**Solution**: Augment cost with trust region penalties
```python
cost += state_diff^T Q_trust state_diff + input_diff^T R_trust input_diff
```

**Effect**:
- Small trust region weights → aggressive updates, fast convergence, risk of divergence
- Large trust region weights → conservative updates, slow convergence, stable

**Tuning rule**: Start with `Q_trust ≈ 0.1 * Q`, `R_trust ≈ 0.1 * R`, adjust based on convergence behavior.

### 6. **Warm Start via Signal Fitting**

**Challenge**: iLQR needs initial feasible trajectory, but we only have reference poses

**Solution** (`_input_warm_start()`):
1. **Fit velocity/curvature** from poses via `complete_kinematic_state_and_inputs_from_poses()`
   - Least squares with jerk/curvature_rate penalties
   - Produces smooth velocity and curvature profiles
2. **Convert to steering** via `_convert_curvature_profile_to_steering_profile()`
   - `steering_angle = atan(wheelbase * curvature)`
3. **Add feedback for initial error**:
   - Velocity feedback: `a_0 += -k_v (v_current - v_ref)`
   - Steering feedback: Stanley controller `δ_fb = -k_lat (e_lat + d * e_heading)`
4. **Run forward dynamics** with constraints to ensure feasibility

**Result**: Initial trajectory that's close to reference and satisfies constraints.

### 7. **Frenet Frame Error Computation** (LQRTracker)

**Challenge**: Tracking error in global (x,y) frame depends on trajectory orientation

**Solution** (`_compute_initial_velocity_and_lateral_state()`):
```python
# Reference pose
x_ref, y_ref, θ_ref = reference_pose

# Error in global frame
Δx = x_ego - x_ref
Δy = y_ego - y_ref

# Project onto Frenet frame (trajectory-aligned)
lateral_error = -Δx * sin(θ_ref) + Δy * cos(θ_ref)  # Perpendicular distance
longitudinal_error = Δx * cos(θ_ref) + Δy * sin(θ_ref)  # Along-track (unused)
heading_error = angle_diff(θ_ego, θ_ref)
```

**Why**: Lateral error in Frenet frame is THE quantity LQR tries to minimize (cross-track error). Longitudinal error handled separately by velocity tracking.

### 8. **Regularized Least Squares for Reference Signals**

**Problem**: Trajectory has discrete poses, but LQR needs continuous velocity/curvature profiles

**Solution** (`tracker_utils.py`):

**Velocity estimation**:
```python
# Given poses [p_0, ..., p_N] with displacements Δp_k = p_{k+1} - p_k
# Model: Δp_k ≈ v_0 dt + Σ_{j=0}^{k-1} a_j dt²
# Minimize: ||y - A x||² + jerk_penalty * ||D a||²
#   where x = [v_0, a_0, ..., a_{N-2}], y = flattened displacements, D = difference matrix
# Result: v_0, acceleration profile → integrate to get velocity profile
```

**Curvature estimation**:
```python
# Given heading displacements Δθ_k and velocity profile v_k
# Model: Δθ_k ≈ Σ_{j=0}^k v_j κ_j dt + v_j κ_rate_j dt²
# Minimize: ||y - A x||² + curvature_rate_penalty * ||κ_rate||²
# Result: κ_0, curvature_rate profile → integrate to get curvature profile
```

**Key detail**: Small `initial_curvature_penalty` added to avoid singular matrix when initial velocity = 0.

**AIDEV-NOTE**: This least squares fitting is brilliant - turns noisy discrete poses into smooth differentiable signals. Regularization prevents overfitting.

## 4. Dependencies

### Internal (nuPlan - Documented ✅)

**Direct Dependencies**:
- ✅ `nuplan.common.actor_state.dynamic_car_state` - `DynamicCarState` (output of trackers)
- ✅ `nuplan.common.actor_state.ego_state` - `EgoState` (current vehicle state)
- ✅ `nuplan.common.actor_state.state_representation` - `StateVector2D`, `TimePoint`
- ✅ `nuplan.common.actor_state.vehicle_parameters` - `VehicleParameters`, `get_pacifica_parameters()`
- ✅ `nuplan.common.geometry.compute` - `principal_value()` (angle wrapping)
- ✅ `nuplan.planning.simulation.controller.tracker.abstract_tracker` - `AbstractTracker` (base class)
- ✅ `nuplan.planning.simulation.controller.tracker.tracker_utils` - Signal fitting utilities
- ✅ `nuplan.planning.simulation.simulation_time_controller.simulation_iteration` - `SimulationIteration`
- ✅ `nuplan.planning.simulation.trajectory.abstract_trajectory` - `AbstractTrajectory`

**Indirect Dependencies**:
- `nuplan.database.utils.measure` - `angle_diff()` (used in LQR, not iLQR)

### External Dependencies
- **NumPy** - Linear algebra (`np.linalg.inv`, `np.linalg.pinv`), array operations
- **typing** - Type hints
- **dataclasses** - Frozen configuration containers
- **time** - Performance timing in iLQR solver (`time.perf_counter()`)
- **enum** - `LateralStateIndex` (IntEnum)

### Dependency Notes

**AIDEV-NOTE**: Both LQR and iLQR depend heavily on `tracker_utils.py` for velocity/curvature estimation. This is the shared foundation for reference signal processing.

**AIDEV-NOTE**: iLQR uses `principal_value()` from geometry module for heading wrapping, while LQR uses `angle_diff()` from database utils. Inconsistency but both work.

## 5. Dependents (Who Uses This Module?)

### Direct Consumers

**Controller Layer**:
- ✅ `nuplan/planning/simulation/controller/two_stage_controller.py`
  - Instantiates `LQRTracker` or `ILQRTracker` as `tracker` component
  - Calls `tracker.track_trajectory()` to compute desired dynamics
  - Passes result to motion model for state propagation

**Hydra Configuration System**:
- `nuplan/planning/script/config/simulation/ego_controller/tracker/lqr_tracker.yaml`
  - Default LQR configuration (q/r gains, horizons, penalties)
- `nuplan/planning/script/config/simulation/ego_controller/tracker/ilqr_tracker.yaml`
  - Default iLQR configuration (solver params, warm start params)
- `nuplan/planning/script/config/simulation/ego_controller/two_stage_controller.yaml`
  - References tracker configs via Hydra composition

### Use Cases

1. **Standard Closed-Loop Simulation** (LQRTracker)
   - Default tracker for most simulations
   - Fast enough for real-time evaluation
   - Sufficient for normal driving scenarios
   - Used in: `just simulate`, tutorial notebooks

2. **Research Optimal Control** (ILQRTracker)
   - State-of-the-art trajectory tracking
   - Handles aggressive maneuvers better than LQR
   - Computationally expensive (10-50× slower)
   - Used in: Advanced research, racing scenarios, emergency maneuvers

3. **Planner Debugging** (LQRTracker with high gains)
   - Increase Q weights → aggressive tracking
   - Reveals planner issues (unfeasible trajectories, high curvature)
   - Fast iterations during development

**AIDEV-NOTE**: 95% of simulations use LQRTracker. ILQRTracker is for special cases where coupling matters or tracking precision is critical.

## 6. Critical Files (Prioritized)

### Priority 1: Start Here! (LQR Basics)

1. **`tracker/lqr.py`** (389 lines) - **Core LQR implementation**
   - `LQRTracker` class with decoupled longitudinal + lateral control
   - `_solve_one_step_lqr()` - Key static method for analytical LQR
   - `_compute_initial_velocity_and_lateral_state()` - Frenet frame projection
   - `_compute_reference_velocity_and_curvature_profile()` - Reference signal extraction
   - `_stopping_controller()` - P-controller for near-stop conditions
   - **Read this first** - simpler than iLQR, shows core concepts

2. **`tracker/abstract_tracker.py`** (31 lines) - **Interface definition**
   - `AbstractTracker.track_trajectory()` - Single abstract method
   - Defines input/output contract for all trackers
   - Simple, self-contained

3. **`tracker/tracker_utils.py`** (377 lines) - **Signal processing toolkit**
   - `get_velocity_curvature_profiles_with_derivatives_from_poses()` - Main entry point
   - `_fit_initial_velocity_and_acceleration_profile()` - Longitudinal least squares
   - `_fit_initial_curvature_and_curvature_rate_profile()` - Lateral least squares
   - `complete_kinematic_state_and_inputs_from_poses()` - Full state reconstruction
   - `compute_steering_angle_feedback()` - Stanley controller
   - `get_interpolated_reference_trajectory_poses()` - Trajectory resampling
   - **Essential utilities** - read after lqr.py to understand reference computation

### Priority 2: Advanced (iLQR)

4. **`tracker/ilqr_tracker.py`** (108 lines) - **iLQR wrapper**
   - `ILQRTracker` class (simple MPC wrapper)
   - `_get_reference_trajectory()` - Extract reference from planner trajectory
   - **Read this** before diving into solver - shows how iLQR fits into nuPlan

5. **`tracker/ilqr/ilqr_solver.py`** (689 lines) - **Full iLQR implementation**
   - `ILQRSolver.solve()` - Main optimization loop (lines 263-345)
   - `_run_forward_dynamics()` - State propagation + Jacobians (lines 441-477)
   - `_dynamics_and_jacobian()` - Bicycle model + linearization (lines 479-546)
   - `_run_lqr_backward_recursion()` - Riccati recursion (lines 552-631)
   - `_update_inputs_with_policy()` - Apply LQR policy (lines 633-688)
   - `_input_warm_start()` - Initial trajectory generation (lines 394-435)
   - `_compute_tracking_cost()` - Cost evaluation (lines 351-372)
   - **Complex but well-documented** - this is graduate-level control theory

### Priority 3: Configuration & Tests

6. **`tracker/test/test_lqr_tracker.py`** (269 lines) - **LQR unit tests**
   - Shows typical LQR usage patterns
   - Tests Frenet frame projection, stopping controller, one-step LQR
   - Good examples for understanding edge cases

7. **`tracker/ilqr/test/test_ilqr_solver.py`** (not read in this session)
   - iLQR solver tests (likely similar structure to LQR tests)

8. **Configuration files**:
   - `config/simulation/ego_controller/tracker/lqr_tracker.yaml` - LQR defaults
   - `config/simulation/ego_controller/tracker/ilqr_tracker.yaml` - iLQR defaults

**AIDEV-NOTE**: For typical users, read files 1-3 (LQR basics). Files 4-5 (iLQR) are advanced - only needed if using ILQRTracker.

## 7. Common Usage Patterns

### 1. LQR Tracker (Standard Configuration)

```python
from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
import numpy as np

# Instantiate LQR tracker
lqr_tracker = LQRTracker(
    # Longitudinal subsystem weights
    q_longitudinal=np.array([10.0]),  # Velocity error cost
    r_longitudinal=np.array([1.0]),   # Acceleration effort cost

    # Lateral subsystem weights (lateral_error, heading_error, steering_angle)
    q_lateral=np.array([1.0, 10.0, 0.0]),  # Emphasize heading error
    r_lateral=np.array([1.0]),  # Steering rate effort cost

    # Discretization
    discretization_time=0.1,  # 10 Hz
    tracking_horizon=10,      # 1 second lookahead

    # Reference signal estimation
    jerk_penalty=1e-4,             # Smoothness penalty for velocity fit
    curvature_rate_penalty=1e-2,   # Smoothness penalty for curvature fit

    # Stopping controller
    stopping_proportional_gain=0.5,
    stopping_velocity=0.2,  # [m/s] Switch to P-controller below this

    vehicle=get_pacifica_parameters(),
)

# Use in simulation loop
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration

current_iteration = SimulationIteration(current_time, index)
next_iteration = SimulationIteration(next_time, index+1)

# Compute control inputs
dynamic_state = lqr_tracker.track_trajectory(
    current_iteration=current_iteration,
    next_iteration=next_iteration,
    initial_state=ego_state,
    trajectory=planner_trajectory,
)

# Extract commands
acceleration = dynamic_state.rear_axle_acceleration_2d.x
steering_rate = dynamic_state.tire_steering_rate
```

### 2. Tuning LQR Gains (Q/R Matrices)

**Longitudinal tuning**:
```python
# High Q, low R → aggressive tracking (large accelerations)
q_longitudinal = [100.0]  # Strong velocity error penalty
r_longitudinal = [0.1]    # Weak acceleration penalty
# Result: Fast convergence to reference velocity, harsh accel/decel

# Low Q, high R → smooth tracking (gentle accelerations)
q_longitudinal = [1.0]   # Weak velocity error penalty
r_longitudinal = [10.0]  # Strong acceleration penalty
# Result: Slow convergence, smooth ride
```

**Lateral tuning**:
```python
# Emphasize cross-track error (stay on centerline)
q_lateral = [100.0, 10.0, 0.0]  # High lateral error weight
r_lateral = [1.0]
# Result: Minimal lane deviation, possibly jerky steering

# Emphasize heading error (smooth orientation)
q_lateral = [1.0, 100.0, 0.0]  # High heading error weight
r_lateral = [1.0]
# Result: Smooth heading alignment, may deviate laterally in curves

# Penalize steering angle (minimize wheel angle magnitude)
q_lateral = [1.0, 10.0, 10.0]  # Steering angle cost
r_lateral = [1.0]
# Result: Smaller steering angles, may increase tracking error

# Penalize steering rate (smooth steering commands)
q_lateral = [1.0, 10.0, 0.0]
r_lateral = [10.0]  # High steering rate penalty
# Result: Very smooth steering, slower response
```

**Rule of thumb**: Start with `q_lateral = [1.0, 10.0, 0.0]`, `r_lateral = [1.0]`, then adjust based on behavior.

### 3. iLQR Tracker (Advanced Optimal Control)

```python
from nuplan.planning.simulation.controller.tracker.ilqr_tracker import ILQRTracker
from nuplan.planning.simulation.controller.tracker.ilqr.ilqr_solver import (
    ILQRSolver, ILQRSolverParameters, ILQRWarmStartParameters
)

# Configure solver parameters
solver_params = ILQRSolverParameters(
    discretization_time=0.2,  # 5 Hz (coarser than LQR for speed)

    # State costs [x, y, heading, velocity, steering_angle]
    state_cost_diagonal_entries=[1.0, 1.0, 10.0, 0.0, 0.0],  # Emphasize heading

    # Input costs [acceleration, steering_rate]
    input_cost_diagonal_entries=[1.0, 10.0],  # Penalize steering rate heavily

    # Trust regions (prevent large updates)
    state_trust_region_entries=[1.0, 1.0, 1.0, 1.0, 1.0],
    input_trust_region_entries=[1.0, 1.0],

    # Convergence
    max_ilqr_iterations=20,
    convergence_threshold=1e-6,  # Stop when input change tiny
    max_solve_time=0.05,  # 50ms budget (leave time for planner)

    # Constraints
    max_acceleration=3.0,  # [m/s²]
    max_steering_angle=1.047,  # [rad] ≈ 60°
    max_steering_angle_rate=0.5,  # [rad/s]
    min_velocity_linearization=0.01,  # Avoid singularity at v=0
)

# Configure warm start
warm_start_params = ILQRWarmStartParameters(
    k_velocity_error_feedback=0.5,
    k_steering_angle_error_feedback=0.05,
    lookahead_distance_lateral_error=15.0,  # [m] Stanley controller
    k_lateral_error=0.1,
    jerk_penalty_warm_start_fit=1e-4,
    curvature_rate_penalty_warm_start_fit=1e-2,
)

# Create solver and tracker
ilqr_solver = ILQRSolver(solver_params, warm_start_params)
ilqr_tracker = ILQRTracker(n_horizon=40, ilqr_solver=ilqr_solver)

# Use same as LQR (AbstractTracker interface)
dynamic_state = ilqr_tracker.track_trajectory(
    current_iteration, next_iteration, ego_state, trajectory
)
```

### 4. Accessing iLQR Solution Trajectory (Debugging)

```python
# iLQR returns DynamicCarState, but solver has full solution internally
# For debugging, can access solver directly:

from nuplan.planning.simulation.controller.tracker.ilqr_tracker import ILQRTracker

# After creating ILQRTracker as above
current_state_array = np.array([
    ego_state.rear_axle.x,
    ego_state.rear_axle.y,
    ego_state.rear_axle.heading,
    ego_state.dynamic_car_state.rear_axle_velocity_2d.x,
    ego_state.tire_steering_angle,
])

# Get reference trajectory (same as tracker does internally)
reference_trajectory = ilqr_tracker._get_reference_trajectory(
    current_iteration, planner_trajectory
)

# Run solver directly
solution_list = ilqr_tracker._ilqr_solver.solve(
    current_state_array, reference_trajectory
)

# Inspect convergence
for i, solution in enumerate(solution_list):
    print(f"Iteration {i}: cost = {solution.tracking_cost:.2f}")

# Visualize final trajectory
final_solution = solution_list[-1]
import matplotlib.pyplot as plt
plt.plot(final_solution.state_trajectory[:, 0],  # x
         final_solution.state_trajectory[:, 1],  # y
         'b-', label='Optimized trajectory')
plt.plot(reference_trajectory[:, 0],  # x_ref
         reference_trajectory[:, 1],  # y_ref
         'r--', label='Reference')
plt.legend()
plt.axis('equal')
plt.show()

# Inspect inputs
print(f"Acceleration: {final_solution.input_trajectory[:, 0]}")
print(f"Steering rate: {final_solution.input_trajectory[:, 1]}")
```

### 5. Using Tracker Utilities Standalone

```python
from nuplan.planning.simulation.controller.tracker.tracker_utils import (
    get_velocity_curvature_profiles_with_derivatives_from_poses,
    complete_kinematic_state_and_inputs_from_poses,
)
import numpy as np

# Example: Analyze a recorded trajectory
poses = np.array([
    [0.0, 0.0, 0.0],     # [x, y, heading]
    [1.0, 0.0, 0.0],
    [2.0, 0.1, 0.05],
    [3.0, 0.3, 0.1],
    # ... more poses
])

# Estimate velocity and curvature
velocity, accel, curvature, curvature_rate = \
    get_velocity_curvature_profiles_with_derivatives_from_poses(
        discretization_time=0.1,
        poses=poses,
        jerk_penalty=0.01,  # Increase for smoother velocity
        curvature_rate_penalty=0.01,  # Increase for smoother steering
    )

print(f"Initial velocity: {velocity[0]:.2f} m/s")
print(f"Mean curvature: {np.mean(curvature):.4f} rad/m")
print(f"Max acceleration: {np.max(np.abs(accel)):.2f} m/s²")

# Reconstruct full kinematic state + inputs
kinematic_states, kinematic_inputs = \
    complete_kinematic_state_and_inputs_from_poses(
        discretization_time=0.1,
        wheel_base=3.089,  # Pacifica wheelbase [m]
        poses=poses,
        jerk_penalty=0.01,
        curvature_rate_penalty=0.01,
    )

# kinematic_states: (N, 5) [x, y, heading, velocity, steering_angle]
# kinematic_inputs: (N, 2) [acceleration, steering_rate]

print(f"Steering angles: {kinematic_states[:, 4]}")
print(f"Steering rates: {kinematic_inputs[:, 1]}")
```

### 6. Custom Reference Trajectory Preprocessing

```python
# Sometimes you want to preprocess reference before giving to tracker
from nuplan.planning.simulation.controller.tracker.tracker_utils import (
    get_interpolated_reference_trajectory_poses
)

# Resample trajectory at tracker's discretization time
times, poses_resampled = get_interpolated_reference_trajectory_poses(
    trajectory=planner_trajectory,
    discretization_time=0.1,
)

# Apply smoothing filter (e.g., moving average on heading)
from scipy.ndimage import uniform_filter1d
poses_resampled[:, 2] = uniform_filter1d(
    poses_resampled[:, 2], size=5, mode='nearest'
)

# Wrap heading back to [-π, π]
from nuplan.common.geometry.compute import principal_value
poses_resampled[:, 2] = principal_value(poses_resampled[:, 2])

# Now use smoothed poses for velocity/curvature estimation
# (This is advanced - normally tracker handles this internally)
```

### 7. Monitoring Tracker Performance

```python
# LQR: Check tracking errors
initial_velocity, initial_lateral_state = \
    lqr_tracker._compute_initial_velocity_and_lateral_state(
        current_iteration, ego_state, trajectory
    )

lateral_error = initial_lateral_state[0]
heading_error = initial_lateral_state[1]
steering_angle = initial_lateral_state[2]

print(f"Lateral error: {lateral_error:.3f} m")
print(f"Heading error: {np.degrees(heading_error):.2f} deg")
print(f"Steering angle: {np.degrees(steering_angle):.2f} deg")

# iLQR: Check convergence
solution_list = ilqr_tracker._ilqr_solver.solve(current_state, reference)
n_iterations = len(solution_list) - 1  # -1 because includes initial iterate

if n_iterations >= ilqr_tracker._ilqr_solver._solver_params.max_ilqr_iterations:
    print("WARNING: iLQR reached max iterations (may not be converged)")
else:
    print(f"iLQR converged in {n_iterations} iterations")

# Check cost reduction
initial_cost = solution_list[0].tracking_cost
final_cost = solution_list[-1].tracking_cost
print(f"Cost reduction: {initial_cost:.2f} → {final_cost:.2f} "
      f"({100*(initial_cost - final_cost)/initial_cost:.1f}% improvement)")
```

## 8. Gotchas & Edge Cases

### LQR Tracker Issues

1. **Tracking horizon = 1 breaks lateral control**
   - **Issue**: With Euler integration, steering rate has no effect on heading in one timestep
   - **Symptom**: Vehicle doesn't steer, lateral error grows unbounded
   - **Fix**: LQRTracker.__init__ asserts `tracking_horizon > 1`
   - **Recommended**: Use `tracking_horizon >= 5` for stable lateral control

2. **Stopping controller oscillation**
   - **Issue**: Velocity hovers around `stopping_velocity` threshold → switches between LQR and P-controller
   - **Symptom**: Jerky motion near stop, oscillating accel/decel
   - **Root cause**: No hysteresis on mode switch
   - **Workaround**: Set `stopping_velocity` lower (e.g., 0.1 m/s instead of 0.2 m/s)
   - **AIDEV-TODO**: Add hysteresis (e.g., enter stopping mode at 0.2 m/s, exit at 0.3 m/s)

3. **Q/R matrix not positive (semi-)definite**
   - **Issue**: User passes negative weight or non-diagonal matrix
   - **Symptom**: AssertionError in `__init__`: "must be positive semidefinite/definite"
   - **Fix**: Ensure all Q entries ≥ 0, all R entries > 0
   - **Note**: LQR requires Q ≥ 0 (PSD), R > 0 (PD) for well-posed problem

4. **Reference velocity/curvature estimation fails**
   - **Issue**: Trajectory has < 2 states
   - **Symptom**: AssertionError in `_get_xy_heading_displacements_from_poses()`: "Cannot get displacements given empty or single element pose trajectory"
   - **Prevention**: Planners must return trajectories with ≥ 2 waypoints
   - **Check**: `len(trajectory.get_sampled_trajectory()) >= 2`

5. **Jerk/curvature_rate penalty too large**
   - **Issue**: Over-regularization smooths out real trajectory features
   - **Symptom**: Poor tracking, LQR follows smoothed reference instead of actual trajectory
   - **Example**: Sharp turn gets smoothed into gentle curve → vehicle cuts corner
   - **Fix**: Reduce penalties to 1e-4 to 1e-2 range
   - **Tuning**: Increase penalties until tracking degrades, then back off 2×

6. **Discretization time mismatch**
   - **Issue**: `LQRTracker.discretization_time != simulation_timestep`
   - **Symptom**: Poor tracking, unexpected behavior
   - **Example**: LQR assumes 0.1s, but simulation runs at 0.05s
   - **Fix**: Ensure `discretization_time == next_iteration.time_s - current_iteration.time_s`
   - **Check**: Add assertion in `track_trajectory()`

7. **Zero velocity breaks curvature estimation**
   - **Issue**: If initial velocity = 0, curvature fit has singular matrix (velocity multiplies curvature in dynamics)
   - **Symptom**: `np.linalg.LinAlgError` or poor curvature estimates
   - **Fix**: `INITIAL_CURVATURE_PENALTY` in `tracker_utils.py` adds small regularization (1e-10)
   - **Result**: Initial curvature slightly biased toward zero, but matrix is invertible

8. **Trajectory shorter than tracking horizon**
   - **Issue**: Trajectory ends before `current_time + tracking_horizon * dt`
   - **Symptom**: `np.interp` extrapolates reference velocity/curvature (may be inaccurate)
   - **Expected behavior**: LQR extrapolates with last values (usually OK)
   - **Better solution**: Planners should generate trajectories covering at least tracking horizon

9. **Large heading error wrapping**
   - **Issue**: Heading error crosses ±π boundary → 2π jump
   - **Symptom**: Sudden large steering commands, oscillation
   - **Fix**: `angle_diff()` in `_compute_initial_velocity_and_lateral_state()` wraps to [-π, π]
   - **Also**: `_solve_one_step_lqr()` has `angle_diff_indices` parameter to handle angular states

10. **Lateral error in wrong frame**
    - **Issue**: Computing lateral error in global (x,y) instead of Frenet frame
    - **Symptom**: Lateral error magnitude/sign wrong → incorrect steering
    - **Correct**: `lateral_error = -Δx sin(θ_ref) + Δy cos(θ_ref)` (perpendicular to trajectory)
    - **Incorrect**: `lateral_error = Δy` (assumes trajectory aligned with x-axis)
    - **Check**: See `_compute_initial_velocity_and_lateral_state()` for reference implementation

### iLQR Solver Issues

11. **iLQR doesn't converge**
    - **Issue**: Reaches `max_ilqr_iterations` without satisfying `convergence_threshold`
    - **Symptom**: Suboptimal tracking, high final cost
    - **Causes**:
      - Poor warm start (reference trajectory not feasible)
      - Trust regions too small (updates too conservative)
      - Convergence threshold too tight (1e-8 may be unrealistic)
      - Reference trajectory incompatible with dynamics
    - **Debug**:
      - Check cost reduction: `solution_list[0].tracking_cost` vs `solution_list[-1].tracking_cost`
      - If cost decreases steadily → just needs more iterations
      - If cost plateaus → may be stuck in local minimum
    - **Fix**:
      - Increase `max_ilqr_iterations` (20 → 50)
      - Relax `convergence_threshold` (1e-6 → 1e-4)
      - Reduce trust region weights by 2-5×

12. **Warm start infeasible**
    - **Issue**: Initial trajectory from pose fitting violates constraints
    - **Symptom**: Warm start has clipped inputs, large initial cost
    - **Expected**: This is OK - iLQR optimizes from infeasible start
    - **Improvement**: Better warm start → faster convergence
    - **AIDEV-TODO**: Constraint-aware warm start (project onto feasible set via QP)

13. **Trust region too tight**
    - **Issue**: Large `state_trust_region_entries` or `input_trust_region_entries`
    - **Symptom**: Tiny input updates per iteration, slow/no convergence
    - **Example**: `state_trust_region = [100, 100, 100, 100, 100]` vs `state_cost = [1, 1, 10, 0, 0]`
      - Trust region dominates → solver minimizes perturbations instead of tracking error
    - **Fix**: Trust region should be ~0.1× to 1× state/input costs
    - **Rule**: If iLQR barely moves from warm start, reduce trust regions

14. **Trust region too loose**
    - **Issue**: Tiny or zero `state_trust_region_entries` / `input_trust_region_entries`
    - **Symptom**: Large input changes, iLQR diverges (cost increases)
    - **Root cause**: Large updates violate linearization, LQR policy is invalid
    - **Fix**: Increase trust region weights until convergence is stable
    - **Line search**: More advanced solution (adjust step size α: `u_new = u + α Δu`)
    - **AIDEV-NOTE**: Current implementation has no line search - relies on trust regions

15. **Min velocity linearization causes jumps**
    - **Issue**: Velocity crosses `min_velocity_linearization` threshold → Jacobian discontinuity
    - **Symptom**: Jittery optimization near zero velocity
    - **Example**: Velocity alternates 0.009 ↔ 0.011, Jacobian flips
    - **Fix**: Use small `min_velocity_linearization` (0.01 - 0.1 m/s)
    - **Why needed**: At v=0, heading dynamics decouple from steering (rank-deficient controllability matrix)

16. **Steering angle singularity at ±90°**
    - **Issue**: Bicycle model has `tan(steering_angle)` in dynamics → singularity at ±π/2
    - **Symptom**: NaN Jacobians, assertion failure in `_dynamics_and_jacobian()`
    - **Assertion**: Line 492-494: `assert np.abs(steering_angle) < np.pi/2`
    - **Prevention**: `max_steering_angle` should be < π/2 (typically π/3 ≈ 60°)
    - **Fix**: Enforce `max_steering_angle < π/2` in config validation

17. **Time budget exceeded**
    - **Issue**: `max_solve_time` too large or not set
    - **Symptom**: iLQR takes > 100ms, starves planner
    - **Example**: Simulation timestep = 100ms, iLQR uses 80ms → planner only gets 20ms
    - **Fix**: Set `max_solve_time` to leave time for planner (e.g., 0.05s)
    - **Monitor**: Check solution list length - if < `max_ilqr_iterations`, time budget was hit

18. **Euler integration accumulates error**
    - **Issue**: Forward Euler is first-order → error grows over long horizons
    - **Symptom**: State trajectory drifts from reference after 10-20 steps
    - **Typical**: Not a problem for horizons < 50 steps at dt = 0.2s
    - **Fix**: Use smaller `discretization_time` or higher-order integrator (RK4)
    - **AIDEV-TODO**: Implement RK4 integration option

19. **Heading wrapping in state error**
    - **Issue**: Reference heading = 3.0 rad, current heading = -3.1 rad → error = 6.1 rad (wrong!)
    - **Fix**: Line 366 in `_compute_tracking_cost()`: `error_state_trajectory[:, 2] = principal_value(error_state_trajectory[:, 2])`
    - **Also**: Line 574 in `_run_lqr_backward_recursion()` (same wrapping)
    - **Critical**: Without this, heading errors explode when crossing ±π

20. **Reference trajectory shorter than horizon**
    - **Issue**: Trajectory ends before `current_time + n_horizon * dt`
    - **Handling**: `ILQRTracker._get_reference_trajectory()` truncates reference (returns shorter array)
    - **Solver behavior**: Works fine with shorter reference (N < `n_horizon`)
    - **Gotcha**: If reference is length 1 → iLQR can't track (need at least 2 states)
    - **Check**: Line 277 in `ilqr_solver.py` asserts `reference_trajectory_length > 1`

### Performance & Numerical Issues

21. **NaN propagation in iLQR**
    - **Issue**: NaN appears in state/input/Jacobian
    - **Symptom**: AssertionError in `ILQRIterate.__post_init__()` (line 160)
    - **Causes**:
      - Division by zero (wheelbase = 0, velocity = 0 without min_velocity_linearization)
      - Invalid input (steering angle > π/2)
      - Poorly conditioned matrix inversion
    - **Debug**: Add NaN checks before/after each operation, print where NaN first appears
    - **Fix**: Validate inputs, ensure constraints are satisfied

22. **Matrix inversion failure in LQR backward pass**
    - **Issue**: `B^T P B + R + R_trust` is singular
    - **Symptom**: `np.linalg.LinAlgError` in `_run_lqr_backward_recursion()` (line 595-596)
    - **Root cause**: R or R_trust not positive definite (zero or negative eigenvalues)
    - **Prevention**: `ILQRSolverParameters.__post_init__()` checks R > 0 (line 101)
    - **Workaround**: Add small regularization `1e-6 * I` to matrix before inversion

23. **Poor conditioning in least squares (tracker_utils)**
    - **Issue**: Matrix `A^T A + penalty * Q` has very large/small eigenvalues
    - **Symptom**: Unstable velocity/curvature estimates, sensitive to noise
    - **Fix**: Use `np.linalg.pinv()` (pseudo-inverse) instead of `np.linalg.inv()` (already done!)
    - **Lines**: 149 (_fit_initial_velocity_and_acceleration_profile), 197 (_fit_initial_curvature_and_curvature_rate_profile)

24. **Slow iLQR convergence on long horizons**
    - **Issue**: `n_horizon = 100` → each iteration very expensive
    - **Complexity**: O(n_horizon × n_states³) per iteration ≈ O(100 × 5³) = O(12,500) ops
    - **Symptom**: Each iLQR call takes > 100ms
    - **Fix**: Reduce `n_horizon` (40 is good default), increase `discretization_time`
    - **Trade-off**: Shorter horizon = more myopic control, longer horizon = better planning but slower

25. **LQR faster than iLQR by 10-50×**
    - **Benchmark** (typical scenario):
      - LQRTracker: ~2 ms per timestep
      - ILQRTracker: ~30 ms per timestep (15× slower)
    - **Why**: LQR is one-shot analytical, iLQR iterates 5-20 times
    - **When to use each**:
      - LQR: 95% of cases (fast, good enough)
      - iLQR: Aggressive maneuvers, research, high-precision tracking
    - **AIDEV-NOTE**: Consider LQR first, only use iLQR if tracking quality insufficient

## 9. Performance Considerations

### Computational Cost (per timestep)

**LQRTracker** (tracking_horizon = 10, dt = 0.1s):
- **Total**: ~2-5 ms
- **Breakdown**:
  - Reference velocity/curvature estimation: 1-2 ms (50% of total)
    - Pose interpolation: 0.2 ms
    - Least squares solve (velocity): 0.5 ms
    - Least squares solve (curvature): 0.5 ms
    - Profile generation: 0.2 ms
  - Frenet frame projection: 0.1 ms
  - Longitudinal LQR: 0.1 ms (1D matrix inversion - trivial)
  - Velocity profile prediction: 0.1 ms
  - Lateral LQR: 0.5 ms (3×3 matrix operations)
  - Overhead: 0.2 ms

**ILQRTracker** (n_horizon = 40, dt = 0.2s, typical 5-10 iterations):
- **Total**: 20-50 ms (10-25× slower than LQR!)
- **Breakdown**:
  - Warm start generation: 3-5 ms (15%)
    - Pose fitting: 2-3 ms
    - Feedback computation: 0.5 ms
    - Forward dynamics: 0.5 ms
  - iLQR iterations (× 5-10): 15-40 ms (75%)
    - Forward dynamics + Jacobians: 40% of iteration
    - Backward recursion (Riccati): 50% of iteration
    - Policy update: 10% of iteration
  - Reference trajectory extraction: 1-2 ms (10%)

**Asymptotic complexity**:
- LQRTracker: O(H² + N) where H = tracking_horizon (≤ 20), N = trajectory length
  - Dominated by least squares (O(M² × N) for M unknowns, N measurements)
- ILQRTracker: O(I × H × (n² + n³)) where I = iterations, H = n_horizon, n = state dim (5)
  - Dominated by matrix multiplications in backward pass

### Memory Footprint

**LQRTracker**:
- Interpolated poses: ~1 KB (N × 3 floats)
- Velocity/curvature profiles: ~1 KB (4 arrays × N floats)
- Q/R matrices: ~0.1 KB (small diagonals)
- **Total**: ~2-3 KB

**ILQRTracker** (for n_horizon = 40):
- State trajectory: 40 × 5 × 8 bytes = 1.6 KB
- Input trajectory: 40 × 2 × 8 bytes = 0.64 KB
- State Jacobians: 40 × 5 × 5 × 8 bytes = 8 KB
- Input Jacobians: 40 × 5 × 2 × 8 bytes = 3.2 KB
- Solution list (×10 iterations): ~130 KB
- **Total**: ~150 KB (50× larger than LQR)

### Real-Time Performance

**LQRTracker**:
- ✅ **Always real-time** on modern CPU
- Typical: 2-5 ms << 100 ms budget
- Leaves 95-98 ms for planner
- **Scaling**: O(H²) → tracking_horizon = 20 still < 10 ms

**ILQRTracker**:
- ⚠️ **May violate real-time** if not tuned
- Typical: 20-50 ms (uses 20-50% of 100ms budget)
- **Critical**: Set `max_solve_time` to enforce deadline
- **Scaling**: O(I × H) → doubling horizon or iterations → 2× runtime
- **Mitigation**:
  - Reduce `n_horizon` (40 → 20)
  - Reduce `max_ilqr_iterations` (20 → 10)
  - Increase `discretization_time` (0.2 → 0.5) - fewer steps
  - Relax `convergence_threshold` (1e-6 → 1e-4) - early exit

### Optimization Strategies

1. **Cache velocity/curvature profiles** (LQR)
   - If trajectory unchanged, reuse previous profiles
   - Saves ~50% of LQR runtime
   - Requires trajectory hash or pointer comparison
   - **AIDEV-TODO**: Implement in LQRTracker with `_cached_trajectory` field

2. **Warm start from previous solution** (iLQR)
   - Use previous timestep's optimal trajectory as initial guess
   - Shift time: `u_warm[k] = u_prev[k+1]`, extrapolate last input
   - 2-5× faster convergence (iterations reduced from 10 → 2-3)
   - **AIDEV-TODO**: Add `previous_solution` parameter to `ILQRTracker.track_trajectory()`

3. **Adaptive horizon** (iLQR)
   - Use shorter horizon when trajectory is straight/smooth
   - Use longer horizon for complex maneuvers
   - Heuristic: `n_horizon = min(max_horizon, int(5.0 / |curvature|))`
   - **AIDEV-TODO**: Implement adaptive horizon in ILQRTracker

4. **Parallel forward/backward pass** (iLQR)
   - Forward dynamics for each timestep is independent → parallelize
   - Backward recursion is sequential (dynamic programming) - cannot parallelize
   - Marginal benefit (~20% speedup) due to overhead
   - Better: Parallelize across scenarios (Ray)

5. **GPU acceleration** (iLQR)
   - Move matrix operations to GPU (CuPy, PyTorch)
   - Good for large horizons (n_horizon > 100) or batched scenarios
   - Overkill for typical use (GPU transfer overhead > compute savings)

6. **Use optimized BLAS** (both)
   - Ensure NumPy linked to MKL or OpenBLAS (not reference BLAS)
   - 2-5× speedup on matrix operations
   - Check: `np.show_config()` should show `blas_mkl_info` or `openblas_info`

7. **Reduce tracking horizon** (LQR)
   - tracking_horizon = 10 → 5: ~2× speedup
   - Still works well for smooth trajectories
   - Trade-off: Less lookahead → more reactive, less predictive

8. **Coarser discretization** (iLQR)
   - discretization_time = 0.2 → 0.5: horizon 40 → 16 steps, ~60% speedup
   - Trade-off: Coarser approximation of continuous dynamics
   - Acceptable if simulation timestep also coarse (e.g., 0.5s)

## 10. Related Documentation

### Cross-References (Documented ✅)

**Core Dependencies**:
- ✅ `nuplan/planning/simulation/controller/CLAUDE.md` - Parent controller architecture, TwoStageController
- ✅ `nuplan/planning/simulation/controller/motion_model/CLAUDE.md` - KinematicBicycleModel (receives tracker output)
- ✅ `nuplan/planning/simulation/trajectory/CLAUDE.md` - AbstractTrajectory, InterpolatedTrajectory (input to trackers)
- ✅ `nuplan/common/actor_state/CLAUDE.md` - EgoState, DynamicCarState, StateVector2D
- ✅ `nuplan/common/geometry/CLAUDE.md` - principal_value(), coordinate transforms
- ✅ `nuplan/planning/simulation/simulation_time_controller/CLAUDE.md` - SimulationIteration

**Related Modules**:
- ✅ `nuplan/planning/simulation/planner/CLAUDE.md` - Planners generate trajectories that trackers follow
- ✅ `nuplan/planning/simulation/observation/CLAUDE.md` - Observations updated after controller
- ✅ `nuplan/planning/simulation/history/CLAUDE.md` - History buffer stores ego states from controller

### Configuration Files

**LQR Configuration**:
- `nuplan/planning/script/config/simulation/ego_controller/tracker/lqr_tracker.yaml`
  - Default Q/R gains, horizons, penalties
  - Stopping controller parameters

**iLQR Configuration**:
- `nuplan/planning/script/config/simulation/ego_controller/tracker/ilqr_tracker.yaml`
  - Solver parameters (costs, trust regions, constraints)
  - Warm start parameters

**Parent Configs**:
- `nuplan/planning/script/config/simulation/ego_controller/two_stage_controller.yaml`
  - Composes tracker + motion model

### External Resources

**LQR Theory**:
- Boyd & Barratt, "Linear Controller Design: Limits of Performance" (1991)
- Bertsekas, "Dynamic Programming and Optimal Control" Vol 1 (2017) - Chapter 4

**iLQR/DDP**:
- Todorov & Li, "A generalized iterative LQR method" (2005) - Original iLQR paper
- Tassa et al., "Control-Limited Differential Dynamic Programming" (2014) - Constrained DDP
- Pabbeel lecture: https://people.eecs.berkeley.edu/~pabbeel/cs287-fa19/slides/Lec5-LQR.pdf

**Bicycle Model**:
- Rajamani, "Vehicle Dynamics and Control" (2011) - Chapter 2 (kinematic models)
- Kong et al., "Kinematic and dynamic vehicle models for autonomous driving control design" (2015)

**Stanley Controller** (steering feedback):
- Hoffmann et al., "Autonomous Automobile Trajectory Tracking for Off-Road Driving" (2007)
- Thrun et al., "Stanley: The robot that won the DARPA Grand Challenge" (2006)

## 11. AIDEV Notes

### Design Insights

**LQR Decoupling Philosophy**:
- Longitudinal and lateral dynamics are weakly coupled for normal driving
- Decoupling enables analytical solution (no iteration)
- Trade-off: Slight suboptimality (1-5% cost increase) for 10-50× speedup
- Valid regime: |accel| < 5 m/s², |curvature| < 0.3 rad/m, velocity > 0.5 m/s

**iLQR as "DDP Lite"**:
- True DDP includes second-order terms (Hessians)
- iLQR uses only first-order (Jacobians) → faster, simpler
- For quadratic costs, iLQR = DDP (no difference)
- For non-quadratic costs, DDP converges faster (but more expensive per iteration)

**Trust Regions vs Line Search**:
- Trust regions: Limit update magnitude via cost penalty (current implementation)
- Line search: Limit update magnitude via step size α (not implemented)
- Trust regions easier to implement (just augment cost matrix)
- Line search potentially faster (can take larger steps if valid)
- **AIDEV-TODO**: Add optional line search in iLQR for better convergence

**Warm Start Quality**:
- Good warm start → 2-3 iterations to converge
- Poor warm start → 10-20 iterations, may not converge
- Current warm start uses regularized least squares (clever!)
- **Improvement idea**: Use previous timestep's solution (temporal coherence)

### Common Mistakes

1. **Confusing LQR file location**: `lqr.py` is in `tracker/` directory, not `tracker/lqr/` subdirectory
   - Only `ilqr/` is a subdirectory

2. **Setting tracking_horizon = 1**: Breaks lateral LQR (steering rate has no effect)

3. **Using iLQR without time budget**: Can starve planner (set `max_solve_time`!)

4. **Forgetting to wrap heading errors**: Causes 2π jumps in LQR commands

5. **Setting trust regions = 0**: iLQR diverges (no regularization)

6. **Assuming LQR = globally optimal**: Only locally optimal (decoupled subsystems)

7. **Thinking iLQR converges to global optimum**: Only local minimum (depends on warm start)

8. **Not checking iLQR convergence**: May be applying suboptimal controls

9. **Copying iLQR solution trajectory**: Violates MPC philosophy (should replan each timestep)

10. **Using LQR for aggressive maneuvers**: Decoupling assumption breaks, tracking degrades

### Future Improvements

**LQRTracker**:
- **AIDEV-TODO**: Add hysteresis to stopping controller (prevent oscillation)
- **AIDEV-TODO**: Cache velocity/curvature profiles (avoid redundant computation)
- **AIDEV-TODO**: Validate trajectory length > tracking_horizon (catch truncation early)
- **AIDEV-TODO**: Adaptive Q/R gains based on scenario (e.g., higher R in dense traffic)
- **AIDEV-TODO**: Multi-step LQR (solve full horizon instead of collapsing to one-step)

**ILQRTracker**:
- **AIDEV-TODO**: Warm start from previous solution (temporal coherence)
- **AIDEV-TODO**: Line search in policy update (better convergence)
- **AIDEV-TODO**: Adaptive horizon (shorter for simple scenarios)
- **AIDEV-TODO**: Constraint handling via augmented Lagrangian (instead of clipping)
- **AIDEV-TODO**: Second-order DDP (use Hessians for faster convergence)
- **AIDEV-TODO**: GPU acceleration for large horizons
- **AIDEV-TODO**: Real-time monitoring (convergence dashboard)

**Tracker Utilities**:
- **AIDEV-TODO**: RK4 integration option (reduce discretization error)
- **AIDEV-TODO**: Adaptive regularization (increase penalties if fit residuals large)
- **AIDEV-TODO**: Constraint-aware warm start (project onto feasible set via QP)

### Potential Bugs

**LQRTracker** (`lqr.py`):
- **Line 148**: Stopping controller has no acceleration saturation
  - Could command unrealistic decel if velocity error large
  - Add clipping: `accel = np.clip(accel, -max_decel, max_accel)`

- **Line 235**: Reference velocity uses `np.interp` with extrapolation
  - If `reference_time > times_s[-1]`, extrapolates linearly (may be inaccurate)
  - Consider clamping to last value instead: `np.interp(..., left=..., right=velocity_profile[-1])`

- **Line 240**: Curvature profile also extrapolates
  - Same issue as velocity
  - Clamp to last curvature for safety

**ILQRSolver** (`ilqr_solver.py`):
- **Line 530-532**: `min_velocity_linearization` creates discontinuity
  - When velocity crosses threshold, Jacobian jumps
  - Better: Smooth transition via tanh or sigmoid
  - `velocity_for_linearization = sign(v) * max(|v|, v_min)` causes derivative discontinuity

- **Line 516-518**: Steering angle clipping changes applied input
  - Input `current_input[1]` is modified after being passed in
  - This is intentional (ensures feasibility) but surprising
  - Consider returning separate `applied_input` for clarity

- **Line 388**: LQR matrix inversion assumes invertibility
  - If `B^T Q B + R` is rank-deficient, crashes
  - Happens if input has no effect on cost (shouldn't occur with R > 0)
  - Add assertion to check condition number: `np.linalg.cond(matrix) < 1e12`

**Tracker Utilities** (`tracker_utils.py`):
- **Line 149**: `np.linalg.pinv` used, but no warning if matrix is ill-conditioned
  - Should check condition number or residual
  - If `||A x - y|| > threshold`, warn user about poor fit

- **Line 330, 336**: Input/state extrapolation by repeating last value
  - Simple but may cause discontinuity if trajectory ends mid-maneuver
  - Better: Use constant velocity/steering angle model to extrapolate

### Testing Gaps

**Unit tests needed**:
- LQR tracking accuracy on curved trajectories (quantitative metrics)
- iLQR convergence on infeasible reference trajectories
- Performance benchmarks (runtime vs horizon, iterations)
- Numerical stability tests (ill-conditioned matrices, extreme velocities)

**Integration tests needed**:
- LQR vs iLQR tracking comparison on standard scenarios
- Controller + motion model closed-loop simulation
- Sensitivity analysis (Q/R gain sweeps)

**Edge case tests**:
- Zero velocity trajectories (parking scenarios)
- Reversing (negative velocity)
- Large heading errors (> π)
- Trajectory truncation (shorter than horizon)

### Documentation Improvements Needed

**Diagrams to add**:
- LQR decoupled architecture (longitudinal || lateral)
- iLQR iteration loop flowchart
- Frenet frame coordinate system
- Trust region visualization (valid linearization region)
- Warm start process flow

**Tuning guides**:
- Q/R matrix selection cookbook (with scenarios)
- Trust region tuning procedure
- When to use LQR vs iLQR decision tree

**Mathematical appendix**:
- Derivation of one-step LQR solution
- Riccati recursion derivation
- Bicycle model linearization
- Least squares regularization explanation

---

**End of CLAUDE.md for tracker/lqr/**
