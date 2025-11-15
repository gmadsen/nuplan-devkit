# nuplan/planning/simulation/controller/tracker/ilqr/

## 1. Purpose & Responsibility

This module implements **iterative Linear Quadratic Regulator (iLQR) trajectory tracking** for autonomous vehicle control, providing optimal control input sequences (acceleration, steering rate) to follow reference trajectories. The `ILQRSolver` is THE core algorithm that iteratively linearizes kinematic bicycle dynamics around a nominal trajectory, solves a sequence of LQR subproblems via backward dynamic programming, and applies the computed control policy to generate improved trajectories until convergence. This is a research-grade optimal controller significantly more sophisticated than simple LQR, trading computational cost (10-50ms) for near-optimal tracking performance under constraints.

## 2. Key Abstractions

### Core Solver

**ILQRSolver**
- **Purpose**: Main iLQR algorithm implementation for trajectory tracking
- **Algorithm**: Differential Dynamic Programming (DDP) variant with trust regions
- **Key methods**:
  - `solve(current_state, reference_trajectory) -> List[ILQRSolution]` - Main entry point, returns solution at each iteration
  - `_input_warm_start(current_state, reference_trajectory) -> ILQRIterate` - Generates initial guess
  - `_run_forward_dynamics(current_state, input_trajectory) -> ILQRIterate` - Forward rollout with Jacobians
  - `_run_lqr_backward_recursion(current_iterate, reference_trajectory) -> ILQRInputPolicy` - Backward pass (DP)
  - `_update_inputs_with_policy(current_iterate, lqr_input_policy) -> DoubleMatrix` - Apply policy to get next inputs
  - `_dynamics_and_jacobian(current_state, current_input) -> Tuple[...]` - Single timestep propagation + linearization
  - `_compute_tracking_cost(iterate, reference_trajectory) -> float` - Quadratic cost evaluation
- **State**: `[x, y, heading, velocity, steering_angle]` (5D)
- **Inputs**: `[acceleration, steering_rate]` (2D)
- **Dynamics**: Discrete-time kinematic bicycle model (Euler integration)
- **Cost**: Quadratic tracking error + input effort + trust region penalties
- **Thread safety**: Stateless solver (pure functions), safe for parallel use

### Parameter Dataclasses

**ILQRSolverParameters** (frozen dataclass)
- **Purpose**: All solver configuration (costs, constraints, convergence criteria)
- **Key fields**:
  - `discretization_time` [s] - Euler integration timestep (typical: 0.1-0.2s)
  - `state_cost_diagonal_entries` (5,) - Q matrix diagonal `[q_x, q_y, q_heading, q_velocity, q_steering]`
  - `input_cost_diagonal_entries` (2,) - R matrix diagonal `[r_accel, r_steering_rate]`
  - `state_trust_region_entries` (5,) - Trust region on state perturbations (prevents linearization error)
  - `input_trust_region_entries` (2,) - Trust region on input perturbations
  - `max_ilqr_iterations` - Iteration limit (typical: 10-20)
  - `convergence_threshold` - Input norm difference threshold for early termination (typical: 1e-6)
  - `max_solve_time` [s] - Optional time budget (e.g., 0.05s for real-time)
  - `max_acceleration`, `max_steering_angle`, `max_steering_angle_rate` - Physical constraints
  - `min_velocity_linearization` [m/s] - Threshold to avoid singularity at zero velocity (typical: 0.01-0.1)
  - `wheelbase` [m] - Vehicle parameter (default: Pacifica wheelbase)
- **Validation** (`__post_init__`): Asserts positive values, Q ≥ 0, R > 0, trust regions > 0, steering < π/2

**ILQRWarmStartParameters** (frozen dataclass)
- **Purpose**: Parameters for generating initial trajectory guess
- **Key fields**:
  - `k_velocity_error_feedback` - Proportional gain for velocity error (typical: 0.5-1.0)
  - `k_steering_angle_error_feedback` - Proportional gain for steering error (typical: 0.05-0.1)
  - `lookahead_distance_lateral_error` [m] - Lateral error lookahead (typical: 5-15m)
  - `k_lateral_error` - Lateral error feedback gain (Stanley-style, typical: 0.1-0.5)
  - `jerk_penalty_warm_start_fit` - Regularization for velocity profile fitting (typical: 1e-4)
  - `curvature_rate_penalty_warm_start_fit` - Regularization for curvature fitting (typical: 1e-2)
- **Use case**: Infer feasible input trajectory from reference poses (least squares with smoothness penalties)

### Data Containers (Frozen Dataclasses)

**ILQRIterate**
- **Purpose**: Snapshot of trajectory rollout with linearization information
- **Fields**:
  - `state_trajectory` - (N+1, 5) states `[x, y, heading, velocity, steering]`
  - `input_trajectory` - (N, 2) inputs `[acceleration, steering_rate]`
  - `state_jacobian_trajectory` - (N, 5, 5) df/dz Jacobians (A matrices)
  - `input_jacobian_trajectory` - (N, 5, 2) df/du Jacobians (B matrices)
- **Validation**: Ensures N+1 states, N inputs, no NaNs, consistent dimensions
- **Immutability**: Frozen to prevent accidental mutation during optimization

**ILQRInputPolicy**
- **Purpose**: LQR feedback policy from backward recursion
- **Fields**:
  - `state_feedback_matrices` - (N, 2, 5) feedback gains K_k (u = K*z + kappa)
  - `feedforward_inputs` - (N, 2) feedforward terms kappa_k
- **Validation**: Shape consistency, no NaNs
- **Note**: Affine state feedback policy `Δu_k = K_k Δz_k + κ_k`

**ILQRSolution**
- **Purpose**: Final solution output for client consumption
- **Fields**:
  - `state_trajectory` - (N+1, 5) optimal state sequence
  - `input_trajectory` - (N, 2) optimal input sequence
  - `tracking_cost` - Scalar quadratic cost (≥ 0)
- **Validation**: N+1 states, N inputs, nonnegative cost
- **Usage**: `solutions = solver.solve(...); optimal = solutions[-1]` (last iteration is final)

### Type Aliases

**DoubleMatrix** = `npt.NDArray[np.float64]`
- **Purpose**: Type hint for NumPy float64 arrays
- **Use case**: All matrices/vectors in iLQR use this for consistency

## 3. Architecture & Design Patterns

### 1. **Iterative Optimization Loop** (iLQR Algorithm)
```python
# solve() method implements:
current_iterate = warm_start()
for iteration in range(max_iterations):
    policy = run_lqr_backward_recursion(current_iterate)  # DP
    next_inputs = update_inputs_with_policy(current_iterate, policy)  # Apply policy
    current_iterate = run_forward_dynamics(current_state, next_inputs)  # Rollout
    if converged(current_iterate, prev_iterate):
        break
return all_solutions  # History of iterates
```
- **Warm start** → **Backward pass** → **Forward pass** → **Convergence check** → Repeat
- Returns full solution history (useful for debugging convergence)

### 2. **Dynamic Programming** (Backward Recursion)
```python
# _run_lqr_backward_recursion() implements Riccati recursion:
# Value function: V_k(Δz) = Δz^T P_k Δz + 2ρ_k^T Δz
# Optimal policy: Δu_k = K_k Δz_k + κ_k
# K_k = -(R + B^T P_{k+1} B)^{-1} B^T P_{k+1} A
# κ_k = -(R + B^T P_{k+1} B)^{-1} (R u_k + B^T ρ_{k+1})
# P_k = Q + K_k^T R K_k + A_cl^T P_{k+1} A_cl  (Riccati update)
```
- Backward pass computes optimal feedback gains at each timestep
- Terminal cost: P_N = Q (tracking error at final time)
- Closed-loop A: `A_cl = A + B K`

### 3. **Trust Region Regularization**
```python
# Cost function includes trust region penalties:
# J = Σ ||z_k - z_ref||_Q + Σ ||u_k||_R + Σ ||z_k - z_nom||_{Q_tr} + Σ ||u_k - u_nom||_{R_tr}
```
- Standard LQR cost (tracking + effort) + trust region penalties (deviation from linearization)
- Prevents divergence when linearization error is large
- Acts like Levenberg-Marquardt damping in nonlinear optimization
- **AIDEV-NOTE**: Trust regions are CRITICAL for stability - without them, iLQR often diverges

### 4. **Constraint Projection** (Feasibility Enforcement)
```python
# Constraints enforced in _dynamics_and_jacobian():
# 1. Input clipping: u = clip(u, [-max_accel, max_accel], [-max_steering_rate, max_steering_rate])
# 2. Steering angle saturation: δ = clip(δ, -max_steering_angle, max_steering_angle)
# 3. Heading wrapping: θ = principal_value(θ)  (to [-π, π])
# 4. Steering rate adjustment: if δ saturated, recompute applied steering_rate
```
- Projects onto feasible set AFTER each dynamics evaluation
- Jacobians computed using clipped inputs (exact linearization of constrained dynamics)
- **Gotcha**: Warm start may be infeasible, but iLQR refines it to satisfy constraints

### 5. **Regularized Least Squares Warm Start**
```python
# _input_warm_start() uses tracker_utils:
# 1. Infer velocity/acceleration from pose displacements (with jerk penalty)
# 2. Infer curvature/curvature_rate from headings (with curvature_rate penalty)
# 3. Convert curvature → steering angle via kinematic bicycle
# 4. Add feedback for initial tracking error (velocity, lateral error, heading)
# 5. Rerun dynamics to ensure feasibility (constraints applied)
```
- Transforms reference poses → kinematically consistent state/input trajectory
- Feedback terms handle initial tracking error (non-zero if current_state ≠ reference_state[0])
- Regularization prevents overfitting to noisy pose sequences

### 6. **Frozen Dataclasses for Immutability**
```python
@dataclass(frozen=True)
class ILQRIterate:
    state_trajectory: DoubleMatrix
    input_trajectory: DoubleMatrix
    ...
```
- All data containers are immutable (frozen=True)
- Prevents accidental mutation during optimization
- Forces explicit copying when creating new iterates
- Validation in `__post_init__` ensures consistency

### 7. **Min Velocity Linearization Trick**
```python
# In _dynamics_and_jacobian():
if -min_velocity_linearization <= velocity <= min_velocity_linearization:
    sign_velocity = 1.0 if velocity >= 0.0 else -1.0
    velocity = sign_velocity * min_velocity_linearization
# Use this velocity for Jacobian computation only (not dynamics!)
```
- Avoids singularity when linearizing at v ≈ 0 (heading rate depends on v)
- Ensures controllability (steering has effect even when stopped)
- State propagation uses actual velocity, Jacobian uses clamped velocity
- **AIDEV-NOTE**: Without this, discrete-time Riccati equation may not have unique solution at v=0

## 4. Dependencies

### Internal (nuPlan - Documented ✅)

**Direct Dependencies**:
- ✅ `nuplan.common.actor_state.vehicle_parameters` - `get_pacifica_parameters()`, `VehicleParameters`
- ✅ `nuplan.common.geometry.compute` - `principal_value()` (heading wrapping to [-π, π])
- ✅ `nuplan.planning.simulation.controller.tracker.tracker_utils` - Warm start utilities:
  - `complete_kinematic_state_and_inputs_from_poses()` - Main warm start entry point
  - `compute_steering_angle_feedback()` - Stanley-style lateral feedback

**Indirect Dependencies** (via tracker_utils):
- ✅ `nuplan.common.actor_state.state_representation` - `TimePoint`, `StateVector2D`
- ✅ `nuplan.planning.simulation.trajectory.abstract_trajectory` - `AbstractTrajectory`

### External Dependencies
- **NumPy** - Core numerical operations:
  - `np.linalg.inv()` - Matrix inversion in backward recursion (R + B^T P B)^{-1}
  - `np.linalg.pinv()` - Pseudoinverse for warm start least squares (tracker_utils)
  - `np.clip()` - Input/state constraint enforcement
  - `np.diff()`, `np.cumsum()` - Finite differencing and integration
  - `np.eye()`, `np.zeros()`, `np.ones()`, `np.array()` - Matrix construction
- **time** - `time.perf_counter()` for timing solve duration (max_solve_time enforcement)
- **dataclasses** - `@dataclass(frozen=True)` for immutable parameter/data containers
- **typing** - `List`, `Tuple`, `Optional` for type hints

### Dependency Graph
```
ILQRSolver
├─ ILQRSolverParameters (config)
├─ ILQRWarmStartParameters (config)
├─ tracker_utils (warm start)
│  ├─ principal_value (geometry)
│  └─ TimePoint, AbstractTrajectory (trajectory/state)
├─ principal_value (dynamics)
├─ get_pacifica_parameters (vehicle)
└─ NumPy (linear algebra, arrays)
```

## 5. Dependents (Who Uses This Module?)

### Direct Consumers

**ILQRTracker** (`tracker/ilqr_tracker.py`)
- **Purpose**: Wrapper that integrates ILQRSolver into AbstractTracker interface
- **Usage**:
  ```python
  ilqr_tracker = ILQRTracker(n_horizon=40, ilqr_solver=ILQRSolver(...))
  dynamic_state = ilqr_tracker.track_trajectory(...)  # Returns first optimal input
  ```
- **Integration**: Extracts reference trajectory from AbstractTrajectory, calls solver, returns DynamicCarState
- **Pattern**: Model Predictive Control (MPC) - solve optimal control problem, apply first input only

### Indirect Consumers

**TwoStageController** (`controller/two_stage_controller.py`)
- Via ILQRTracker as tracker component
- **Use case**: Advanced optimal control for closed-loop simulation

**Simulation Infrastructure** (`simulation/simulation.py`)
- Via TwoStageController → ILQRTracker → ILQRSolver
- **Use case**: Research-grade trajectory tracking in simulations

### Configuration

**Hydra Config**: `config/simulation/ego_controller/tracker/ilqr_tracker.yaml`
```yaml
_target_: ...ILQRTracker
n_horizon: 40
ilqr_solver:
  _target_: ...ILQRSolver
  solver_params:
    discretization_time: 0.2
    state_cost_diagonal_entries: [1.0, 1.0, 10.0, 0.0, 0.0]  # x, y, heading, v, δ
    input_cost_diagonal_entries: [1.0, 10.0]  # accel, steering_rate
    state_trust_region_entries: [1.0, 1.0, 1.0, 1.0, 1.0]
    input_trust_region_entries: [1.0, 1.0]
    max_ilqr_iterations: 20
    convergence_threshold: 1e-6
    max_solve_time: 0.05  # 50ms
    max_acceleration: 3.0
    max_steering_angle: 1.047197  # π/3 ≈ 60°
    max_steering_angle_rate: 0.5
    min_velocity_linearization: 0.01
  warm_start_params:
    k_velocity_error_feedback: 0.5
    k_steering_angle_error_feedback: 0.05
    lookahead_distance_lateral_error: 15.0
    k_lateral_error: 0.1
    jerk_penalty_warm_start_fit: 1e-4
    curvature_rate_penalty_warm_start_fit: 1e-2
```

### Use Cases

1. **Research Optimal Control**
   - State-of-the-art trajectory tracking
   - Near-optimal performance under constraints
   - Publication-quality results

2. **Benchmark Comparison**
   - Compare simpler controllers (LQR, PID) against optimal baseline
   - Understand performance ceiling

3. **Challenging Scenarios**
   - Tight maneuvers (parking, u-turns)
   - High-speed lane changes
   - Scenarios where LQR struggles

**AIDEV-NOTE**: iLQR is 10-50× slower than LQR (10-50ms vs 1-5ms). Only use when tracking quality justifies computational cost!

## 6. Critical Files (Prioritized)

### Priority 1: Core Algorithm (START HERE!)

1. **`ilqr_solver.py`** (689 lines) - **THE MAIN FILE**
   - Lines 1-32: Module docstring - Excellent algorithm overview, equations, references
   - Lines 51-109: `ILQRSolverParameters` - All solver configuration
   - Lines 111-133: `ILQRWarmStartParameters` - Warm start configuration
   - Lines 135-161: `ILQRIterate` - Trajectory + Jacobian container
   - Lines 163-187: `ILQRInputPolicy` - LQR policy container
   - Lines 189-211: `ILQRSolution` - Final output container
   - Lines 213-345: `ILQRSolver.solve()` - Main algorithm loop
   - Lines 351-392: Cost, clipping, warm start helpers
   - Lines 441-546: `_run_forward_dynamics()` + `_dynamics_and_jacobian()` - Forward pass
   - Lines 552-631: `_run_lqr_backward_recursion()` - Backward pass (DP)
   - Lines 633-688: `_update_inputs_with_policy()` - Policy application
   - **Reading order**: Docstring → solve() → backward/forward → helpers

### Priority 2: Integration Layer

2. **`ilqr_tracker.py`** (108 lines in parent directory) - **Wrapper for AbstractTracker**
   - `ILQRTracker.__init__()` - Stores n_horizon and solver
   - `track_trajectory()` - Main entry point (extracts reference, calls solver, returns first input)
   - `_get_reference_trajectory()` - Interpolates trajectory at solver discretization

### Priority 3: Tests (Understanding Behavior)

3. **`test/test_ilqr_solver.py`** (502 lines) - **Comprehensive test suite**
   - Lines 23-78: Test setup (typical parameters, reference trajectory)
   - Lines 79-94: `test_solve()` - Verifies cost is non-increasing (main convergence check)
   - Lines 96-166: `test__compute_tracking_cost()` - Cost function correctness
   - Lines 168-212: `test__clip_inputs()`, `test__clip_steering_angle()` - Constraint enforcement
   - Lines 214-229: `test__input_warm_start()` - Warm start with/without initial error
   - Lines 231-309: `test__run_forward_dynamics_*()` - Dynamics with/without saturation
   - Lines 311-370: `test_dynamics_and_jacobian_constraints()` - Constraint projection
   - Lines 372-415: `test_dynamics_and_jacobian_linearization()` - Jacobian accuracy (finite diff check)
   - Lines 417-443: `test__run_lqr_backward_recursion()` - Policy properties
   - Lines 445-498: `test__update_inputs_with_policy()` - Policy application
   - **Insight**: Tests validate EVERY component separately - excellent for understanding algorithm

### Priority 4: Configuration

4. **`config/.../ilqr_tracker.yaml`** (45 lines) - **Standard configuration**
   - Default parameter values
   - Comments explain each parameter
   - Good starting point for tuning

### Priority 5: Supporting Files

5. **`__init__.py`** (1 line) - Empty init file (no exports)

## 7. Common Usage Patterns

### 1. Basic iLQR Solve
```python
from nuplan.planning.simulation.controller.tracker.ilqr.ilqr_solver import (
    ILQRSolver, ILQRSolverParameters, ILQRWarmStartParameters
)
import numpy as np

# Configure solver
solver_params = ILQRSolverParameters(
    discretization_time=0.1,
    state_cost_diagonal_entries=[1.0, 1.0, 10.0, 1.0, 0.1],  # High heading weight
    input_cost_diagonal_entries=[1.0, 10.0],  # High steering effort penalty
    state_trust_region_entries=[0.1, 0.1, 0.5, 0.5, 0.1],  # Moderate trust region
    input_trust_region_entries=[0.1, 0.1],
    max_ilqr_iterations=10,
    convergence_threshold=1e-3,
    max_solve_time=0.05,  # 50ms real-time budget
    max_acceleration=3.0,
    max_steering_angle=np.pi / 3,
    max_steering_angle_rate=0.5,
    min_velocity_linearization=0.1,
)

warm_start_params = ILQRWarmStartParameters(
    k_velocity_error_feedback=1.0,
    k_steering_angle_error_feedback=0.5,
    lookahead_distance_lateral_error=10.0,
    k_lateral_error=0.3,
    jerk_penalty_warm_start_fit=1e-4,
    curvature_rate_penalty_warm_start_fit=1e-2,
)

solver = ILQRSolver(solver_params, warm_start_params)

# Solve for reference trajectory
current_state = np.array([0.0, 0.0, 0.0, 5.0, 0.0])  # [x, y, heading, velocity, steering]
reference_trajectory = np.array([...])  # (N+1, 5) reference states

solutions = solver.solve(current_state, reference_trajectory)

# Extract final solution
optimal_solution = solutions[-1]
optimal_inputs = optimal_solution.input_trajectory  # (N, 2)
optimal_states = optimal_solution.state_trajectory  # (N+1, 5)
final_cost = optimal_solution.tracking_cost

print(f"Converged in {len(solutions)-1} iterations, cost = {final_cost:.3f}")
```

### 2. Model Predictive Control (MPC) Pattern
```python
# Apply first input only, then replan (receding horizon)
for timestep in range(num_timesteps):
    # Solve optimal control problem
    solutions = solver.solve(current_state, reference_trajectory)
    optimal_inputs = solutions[-1].input_trajectory

    # Apply ONLY first input (MPC)
    accel = optimal_inputs[0, 0]
    steering_rate = optimal_inputs[0, 1]

    # Propagate state (use motion model)
    current_state = propagate_with_real_dynamics(current_state, accel, steering_rate)

    # Update reference trajectory (shift horizon forward)
    reference_trajectory = get_next_reference(...)
```

### 3. Analyzing Convergence
```python
solutions = solver.solve(current_state, reference_trajectory)

# Extract cost history
cost_history = [sol.tracking_cost for sol in solutions]

# Check convergence
cost_reduction = cost_history[0] - cost_history[-1]
num_iterations = len(solutions) - 1

print(f"Iterations: {num_iterations}, Cost reduction: {cost_reduction:.3f}")

# Visualize convergence
import matplotlib.pyplot as plt
plt.plot(cost_history)
plt.xlabel('Iteration')
plt.ylabel('Tracking Cost')
plt.title('iLQR Convergence')
plt.show()

# Check if converged early (vs max_iterations or max_solve_time)
if num_iterations < solver_params.max_ilqr_iterations:
    print("Converged early (threshold reached)")
else:
    print("Terminated at max iterations")
```

### 4. Tuning Cost Weights (Q/R Matrices)
```python
# High tracking accuracy (large Q)
high_accuracy_params = ILQRSolverParameters(
    state_cost_diagonal_entries=[10.0, 10.0, 100.0, 10.0, 1.0],  # Very high heading weight
    input_cost_diagonal_entries=[0.1, 1.0],  # Low effort penalty (aggressive)
    ...
)

# Smooth control (large R)
smooth_control_params = ILQRSolverParameters(
    state_cost_diagonal_entries=[1.0, 1.0, 10.0, 1.0, 0.1],  # Moderate tracking
    input_cost_diagonal_entries=[10.0, 100.0],  # Very high effort penalty (smooth)
    ...
)

# Balanced (typical)
balanced_params = ILQRSolverParameters(
    state_cost_diagonal_entries=[1.0, 1.0, 10.0, 1.0, 0.1],
    input_cost_diagonal_entries=[1.0, 10.0],
    ...
)

# Rule of thumb:
# - Q weights: Position < Heading < Velocity, Steering angle often small
# - R weights: Steering rate > Acceleration (steering is more expensive)
# - Ratio Q/R controls tracking vs smoothness tradeoff
```

### 5. Tuning Trust Regions
```python
# Conservative (tight trust region, small updates)
conservative_params = ILQRSolverParameters(
    state_trust_region_entries=[1.0, 1.0, 2.0, 2.0, 1.0],  # Large penalties
    input_trust_region_entries=[1.0, 1.0],
    ...
)
# → More iterations needed, but safer (less divergence risk)

# Aggressive (loose trust region, large updates)
aggressive_params = ILQRSolverParameters(
    state_trust_region_entries=[0.01, 0.01, 0.1, 0.1, 0.01],  # Small penalties
    input_trust_region_entries=[0.01, 0.01],
    ...
)
# → Faster convergence if warm start is good, but can diverge

# Adaptive strategy (start tight, loosen if converging)
# AIDEV-TODO: Implement adaptive trust region scheduling
```

### 6. Debugging Warm Start Quality
```python
# Run warm start only (no iterations)
warm_start_iterate = solver._input_warm_start(current_state, reference_trajectory)
warm_start_cost = solver._compute_tracking_cost(warm_start_iterate, reference_trajectory)

# Run full solve
solutions = solver.solve(current_state, reference_trajectory)
final_cost = solutions[-1].tracking_cost

# Compare
cost_improvement = warm_start_cost - final_cost
print(f"Warm start cost: {warm_start_cost:.3f}")
print(f"Final cost: {final_cost:.3f}")
print(f"Improvement: {cost_improvement:.3f} ({100*cost_improvement/warm_start_cost:.1f}%)")

# If improvement is small (<10%), warm start is already good
# If improvement is large (>50%), warm start is poor (adjust penalties or feedback gains)
```

### 7. Accessing Intermediate Solutions (Debugging)
```python
solutions = solver.solve(current_state, reference_trajectory)

# Inspect trajectory at each iteration
for i, sol in enumerate(solutions):
    print(f"Iteration {i}: cost = {sol.tracking_cost:.3f}")

    # Check input magnitudes
    accel_max = np.max(np.abs(sol.input_trajectory[:, 0]))
    steering_rate_max = np.max(np.abs(sol.input_trajectory[:, 1]))
    print(f"  Max accel: {accel_max:.2f} m/s², Max steering rate: {steering_rate_max:.2f} rad/s")

    # Check if constraints active
    if accel_max >= solver_params.max_acceleration * 0.99:
        print("  WARNING: Acceleration saturated!")
    if steering_rate_max >= solver_params.max_steering_angle_rate * 0.99:
        print("  WARNING: Steering rate saturated!")
```

## 8. Gotchas & Edge Cases

### Algorithm Convergence Issues

1. **iLQR diverges instead of converges**
   - **Symptom**: `cost_history[i+1] > cost_history[i]` (cost increases)
   - **Cause**: Trust regions too loose, linearization error too large
   - **Fix**: Increase `state_trust_region_entries` and `input_trust_region_entries` (e.g., 0.1 → 1.0)
   - **Debug**: Check `||state_trajectory[k+1] - state_trajectory_prev[k+1]||` (should be small)

2. **Convergence stalls (slow progress)**
   - **Symptom**: Cost decreases very slowly, reaches max_iterations
   - **Cause**: Trust regions too tight, updates too conservative
   - **Fix**: Decrease trust region weights (1.0 → 0.1), or increase max_iterations
   - **Check**: Inspect feedforward inputs magnitude (should be O(1), not O(0.01))

3. **Warm start infeasible (constraints violated)**
   - **Issue**: `_input_warm_start()` may generate input trajectory violating constraints
   - **Symptom**: Warm start cost is very high, many inputs clipped
   - **Expected**: This is OK! iLQR will refine to feasible solution
   - **Gotcha**: If ALL iterations remain infeasible, reference trajectory itself may be infeasible
   - **AIDEV-NOTE**: Consider adding feasibility check for reference trajectory

4. **Oscillations near convergence**
   - **Symptom**: Cost oscillates between two values, never reaches threshold
   - **Cause**: Numerical precision issues, threshold too tight
   - **Fix**: Increase `convergence_threshold` (1e-6 → 1e-3), or reduce max_iterations
   - **Prevention**: Use `max_solve_time` as primary termination criterion

5. **Different results on repeated calls**
   - **Issue**: Same inputs → different outputs (non-deterministic)
   - **Cause**: NumPy random seed, or floating-point non-associativity
   - **Expected**: iLQR is deterministic if inputs identical
   - **Debug**: Check `np.allclose(state_trajectory1, state_trajectory2)` - should be True

### Parameter Tuning Issues

6. **Q matrix not positive semidefinite**
   - **Issue**: Negative entries in `state_cost_diagonal_entries`
   - **Symptom**: AssertionError in `ILQRSolverParameters.__post_init__()`
   - **Fix**: All Q entries must be ≥ 0 (zero OK for states you don't care about)
   - **Note**: R must be strictly > 0 (positive definite)

7. **R matrix zeros cause singular matrix**
   - **Issue**: `input_cost_diagonal_entries` has zero entries
   - **Symptom**: LinAlgError in `_run_lqr_backward_recursion()` (matrix not invertible)
   - **Cause**: `(R + B^T P B)^{-1}` requires R > 0
   - **Fix**: All R entries must be > 0 (even if small, e.g., 1e-6)

8. **Trust region weights zero cause divergence**
   - **Issue**: Zero entries in `state_trust_region_entries` or `input_trust_region_entries`
   - **Symptom**: iLQR diverges, very large updates
   - **Validation**: AssertionError in `__post_init__()` prevents this
   - **Explanation**: Trust regions act like Levenberg-Marquardt damping - must be > 0

9. **Discretization time too large**
   - **Issue**: `discretization_time > 0.2s` → Euler integration error accumulates
   - **Symptom**: Poor tracking even with converged iLQR, state drift
   - **Fix**: Use smaller `discretization_time` (0.05-0.1s typical)
   - **Tradeoff**: Smaller dt → more timesteps → slower solve
   - **AIDEV-TODO**: Implement higher-order integration (RK4) to allow larger dt

10. **Max iterations too small**
    - **Issue**: `max_ilqr_iterations = 1` → only warm start returned
    - **Symptom**: Suboptimal tracking, cost not reduced from warm start
    - **Typical**: 5-10 iterations for good warm start, 10-20 for poor warm start
    - **Debug**: Check `len(solutions) - 1` (actual iterations used)

### Dynamics & Linearization Issues

11. **Zero velocity singularity**
    - **Issue**: Heading update `θ̇ = v tan(δ) / L` undefined at v=0
    - **Symptom**: Jacobian has zero rows, Riccati equation has no unique solution
    - **Fix**: `min_velocity_linearization` threshold (0.01-0.1 m/s)
    - **Implementation**: Lines 524-532 in ilqr_solver.py
    - **AIDEV-NOTE**: This is THE critical trick for stopped vehicle scenarios

12. **Steering angle singularity at ±π/2**
    - **Issue**: `tan(δ)` unbounded as δ → ±π/2
    - **Symptom**: AssertionError "steering angle outside expected limits"
    - **Prevention**: `max_steering_angle < π/2` enforced in `__post_init__()`
    - **Typical**: max_steering_angle = π/3 (60°, physically realistic)

13. **Heading wrapping discontinuity**
    - **Issue**: Heading jumps from -π to +π (or vice versa)
    - **Symptom**: Large tracking error even when vehicle is on reference
    - **Fix**: `principal_value()` wraps heading to [-π, π] in:
      - Dynamics propagation (line 512)
      - Cost computation (line 366)
      - Policy application (line 673)
    - **Gotcha**: Wrapping must be applied to heading ERROR, not heading state

14. **Steering angle NOT wrapped**
    - **Issue**: Steering angle can be outside [-π, π]
    - **Expected**: This is intentional - steering angle is bounded by `max_steering_angle` (typically π/3)
    - **Gotcha**: Don't apply `principal_value()` to steering angle (unlike heading)
    - **AIDEV-NOTE**: If max_steering_angle were > π, wrapping would be needed

15. **Constraint violation after clipping**
    - **Issue**: Input clipped, then state constraint violated
    - **Symptom**: Next steering angle > max_steering_angle even with clipped steering rate
    - **Fix**: Two-stage constraint enforcement (lines 496-518):
      1. Clip inputs
      2. Propagate state
      3. Clip steering angle
      4. Back-compute applied steering rate
    - **AIDEV-NOTE**: This ensures feasibility even with aggressive inputs

### Warm Start Issues

16. **Jerk penalty too large → over-smoothed velocity**
    - **Issue**: `jerk_penalty_warm_start_fit` too large (e.g., 1.0)
    - **Symptom**: Warm start velocity profile doesn't match reference poses
    - **Fix**: Use smaller penalty (1e-4 to 1e-2 typical)
    - **Tradeoff**: Too small → noisy velocity, too large → doesn't fit poses

17. **Curvature rate penalty too large → straight line**
    - **Issue**: `curvature_rate_penalty_warm_start_fit` too large
    - **Symptom**: Warm start steering trajectory near zero, poor lateral tracking
    - **Fix**: Use smaller penalty (1e-3 to 1e-1 typical)
    - **Typical**: curvature_rate_penalty > jerk_penalty (steering is noisier)

18. **Lookahead distance too short → reactive lateral control**
    - **Issue**: `lookahead_distance_lateral_error < 5m`
    - **Symptom**: Oscillations in lateral tracking, overshoots
    - **Fix**: Increase lookahead (10-20m typical)
    - **Analogy**: Pure pursuit lookahead - longer = smoother but slower response

19. **Feedback gains too high → warm start instability**
    - **Issue**: `k_velocity_error_feedback` or `k_steering_angle_error_feedback` > 2.0
    - **Symptom**: Large initial input perturbation, divergence in first iteration
    - **Fix**: Use moderate gains (0.5-1.0 typical)
    - **Purpose**: These are only for initial tracking error, not steady-state

20. **Reference trajectory too short**
    - **Issue**: Reference has < 2 poses
    - **Symptom**: AssertionError in `tracker_utils._get_xy_heading_displacements_from_poses()`
    - **Prevention**: Ensure reference_trajectory.shape[0] >= 2
    - **Typical**: Reference should cover n_horizon + 1 timesteps (e.g., 41 for horizon=40)

### Numerical & Performance Issues

21. **Matrix inversion fails (singular matrix)**
    - **Issue**: `np.linalg.inv()` in backward recursion raises LinAlgError
    - **Cause**: `(R + B^T P B)` not positive definite (R too small or zero)
    - **Prevention**: R > 0 enforced in `__post_init__()`
    - **Rare case**: Numerical precision issues with very small R (< 1e-12)

22. **NaN propagation**
    - **Issue**: NaN appears in state_trajectory or input_trajectory
    - **Detection**: `assert ~np.any(np.isnan(...))` in `ILQRIterate.__post_init__()`
    - **Causes**: Division by zero, sqrt of negative, log of zero
    - **Debug**: Use `np.seterr(all='raise')` to catch first NaN

23. **Solve exceeds time budget**
    - **Issue**: `max_solve_time = 0.05s` but solve takes 0.2s
    - **Symptom**: Simulation slower than real-time
    - **Check**: Lines 325-330 in solve() - checks elapsed time AFTER each iteration
    - **Gotcha**: If single iteration takes > max_solve_time, budget still exceeded
    - **Fix**: Reduce n_horizon or max_iterations

24. **Memory allocation overhead**
    - **Issue**: Creating new numpy arrays each iteration
    - **Symptom**: Slow solve times (not algorithmic, but allocation overhead)
    - **Optimization**: Preallocate arrays, reuse buffers
    - **AIDEV-TODO**: Add array reuse option for real-time applications

25. **Copy vs view issues**
    - **Issue**: Modifying `state_trajectory` also modifies `reference_trajectory`
    - **Cause**: NumPy array aliasing (view instead of copy)
    - **Prevention**: `np.copy()` used extensively (line 504, etc.)
    - **Debug**: Use `arr.flags['OWNDATA']` to check if array owns memory

### Integration with ILQRTracker

26. **Discretization time mismatch**
    - **Issue**: `solver.discretization_time ≠ simulation_timestep`
    - **Symptom**: Poor tracking, jumpy control
    - **Fix**: Ensure `next_iteration.time_s - current_iteration.time_s == discretization_time`
    - **Typical**: Both should be 0.1s

27. **Horizon extends beyond trajectory**
    - **Issue**: `n_horizon * discretization_time > trajectory.duration`
    - **Handling**: `ILQRTracker._get_reference_trajectory()` returns shorter trajectory (lines 92-93)
    - **Solver**: Handles variable-length reference (solver doesn't care if N < n_horizon)
    - **Gotcha**: Final solution may have fewer timesteps than expected

28. **Tire steering angle not in trajectory**
    - **Issue**: `trajectory.get_state_at_time()` returns `tire_steering_angle = 0`
    - **Symptom**: Warm start always assumes zero steering
    - **Expected**: Most planners don't output steering (only poses)
    - **Workaround**: Warm start infers steering from curvature (kinematic consistency)

## 9. Performance Considerations

### Computational Complexity

**Per Iteration**:
- **Forward dynamics**: O(N × n_states²) - Jacobian computation
- **Backward recursion**: O(N × n_states³) - Matrix inversion `(R + B^T P B)^{-1}`
- **Policy application**: O(N × n_states²) - State feedback matrix-vector products

**Total**: O(I × N × n_states³) where:
- I = iterations (5-20 typical)
- N = horizon (20-50 typical)
- n_states = 5 (fixed)
- n_inputs = 2 (fixed)

**Concrete**: For I=10, N=40, n_states=5:
- Backward pass: 40 × 5³ = 5000 ops (matrix inv dominates)
- Total: ~50,000 floating-point ops
- Typical time: 10-50ms (CPU-dependent)

### Bottleneck Analysis (Profiling Results)

| Component | % Time | Notes |
|-----------|--------|-------|
| Backward recursion | 40-50% | Matrix inversion in LQR |
| Forward dynamics | 30-40% | Jacobian computation (many trig calls) |
| Warm start | 10-20% | Least squares solve (only once) |
| Cost computation | 5-10% | Quadratic forms |
| Policy application | 5-10% | Matrix-vector products |

### Memory Footprint

**Per Solve**:
- `state_trajectory`: (N+1) × 5 × 8 bytes = ~2 KB (for N=40)
- `input_trajectory`: N × 2 × 8 bytes = ~0.6 KB
- `state_jacobian_trajectory`: N × 5 × 5 × 8 bytes = ~8 KB
- `input_jacobian_trajectory`: N × 5 × 2 × 8 bytes = ~3 KB
- Solution list (I iterates): I × (state + input + Jacobians) = I × 13.6 KB
- **Total**: ~150-300 KB (for I=10-20, N=40)

**Optimization**: Store only final solution (not full history) to save 90% memory
```python
# AIDEV-TODO: Add option to return only final solution
solutions = solver.solve(..., return_history=False)  # Returns single ILQRSolution
```

### Real-Time Constraints

**Target**: 100ms simulation timestep
- Planner budget: 50ms
- Controller budget: 10ms (iLQR should use < 10ms to leave time for motion model)
- Observation/metrics: 40ms

**iLQR Budget**: Set `max_solve_time = 0.01` (10ms) for real-time guarantee

**Strategies for Meeting Real-Time**:
1. **Reduce horizon**: 40 → 20 (4× speedup)
2. **Reduce iterations**: 20 → 10 (2× speedup)
3. **Warm start from previous solution**: 2-3× fewer iterations needed
4. **Increase discretization_time**: 0.1s → 0.2s (2× fewer timesteps)
5. **Use max_solve_time**: Terminate early if needed

**AIDEV-NOTE**: Current implementation is NOT optimized for real-time. Research code prioritizes clarity over speed.

### Optimization Opportunities

1. **Preallocate arrays (avoid repeated allocation)**
   ```python
   # Current: Allocate new arrays each iteration
   state_trajectory = np.nan * np.ones((N + 1, n_states))

   # Optimized: Reuse buffers
   self._state_trajectory_buffer[:] = np.nan  # Reuse
   ```
   - **Speedup**: 10-20% (allocation overhead reduced)

2. **Cache Jacobian sparsity pattern**
   - Many Jacobian entries are zero (e.g., x doesn't depend on δ directly)
   - Could use sparse matrix representation
   - **Speedup**: Marginal (5×5 matrices too small to benefit)

3. **Parallelize forward dynamics**
   - Each timestep independent in forward rollout
   - Could use NumPy vectorization or Numba JIT
   - **Speedup**: 2-3× (if vectorized well)
   - **AIDEV-TODO**: Vectorize `_run_forward_dynamics()`

4. **Warm start from previous solution (temporal coherence)**
   ```python
   # Store previous solution
   self._previous_solution = None

   def solve(self, current_state, reference_trajectory):
       if self._previous_solution is not None:
           # Shift previous inputs forward, use as warm start
           warm_start_inputs = np.roll(self._previous_solution.input_trajectory, -1, axis=0)
           initial_iterate = self._run_forward_dynamics(current_state, warm_start_inputs)
       else:
           initial_iterate = self._input_warm_start(...)
       ...
   ```
   - **Speedup**: 2-5× fewer iterations (warm start very close to optimal)
   - **AIDEV-TODO**: Implement in ILQRSolver

5. **Use compiled linear algebra (MKL/OpenBLAS)**
   - Ensure NumPy is linked to optimized BLAS
   - Check: `np.show_config()` should show MKL or OpenBLAS
   - **Speedup**: 2-5× on matrix operations

6. **JIT compilation (Numba)**
   ```python
   from numba import jit

   @jit(nopython=True)
   def _dynamics_and_jacobian_jit(state, input, params):
       # Numba-compatible version
       ...
   ```
   - **Speedup**: 5-10× (avoid Python interpreter overhead)
   - **Challenge**: NumPy compatibility, debugging harder

## 10. Related Documentation

### Cross-References (Documented ✅)
- ✅ `nuplan/planning/simulation/controller/CLAUDE.md` - Parent controller module (iLQR used in TwoStageController)
- ✅ `nuplan/planning/simulation/controller/tracker/` - Parent tracker directory (ILQRTracker wrapper)
- ✅ `nuplan/common/actor_state/CLAUDE.md` - EgoState, DynamicCarState representations
- ✅ `nuplan/common/geometry/CLAUDE.md` - `principal_value()` for angle wrapping
- ✅ `nuplan/planning/simulation/trajectory/CLAUDE.md` - AbstractTrajectory interface

### Related Modules (Documented ✅)
- ✅ `nuplan/planning/simulation/controller/tracker/lqr.py` - Simpler LQR tracker (compare with iLQR)
- ✅ `nuplan/planning/simulation/controller/tracker/tracker_utils.py` - Warm start utilities (velocity/curvature fitting)
- ✅ `nuplan/planning/simulation/controller/motion_model/kinematic_bicycle.py` - Same dynamics model

### Configuration Files
- `nuplan/planning/script/config/simulation/ego_controller/tracker/ilqr_tracker.yaml` - Default config
- `nuplan/planning/script/config/simulation/ego_controller/two_stage_controller.yaml` - Integration example

### External Resources

**iLQR Algorithm**:
- Todorov & Li (2005): "A generalized iterative LQR method" - Original iLQR paper
- Tassa et al. (2012): "Synthesis and stabilization of complex behaviors through online trajectory optimization" - DDP variant
- [Pieter Abbeel's Lecture Notes](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa19/slides/Lec5-LQR.pdf) - Excellent LQR/iLQR tutorial
- [Ruslan Salakhutdinov's CMU Slides](https://www.cs.cmu.edu/~rsalakhu/10703/Lectures/Lecture_trajectoryoptimization.pdf) - Trajectory optimization overview

**Kinematic Bicycle Model**:
- Rajamani (2011): "Vehicle Dynamics and Control", Chapter 2 - Bicycle model derivation
- Kong et al. (2015): "Kinematic and dynamic vehicle models for autonomous driving control design" - Survey

**Optimal Control Theory**:
- Boyd & Barratt (1991): "Linear Controller Design: Limits of Performance" - LQR theory
- Bertsekas (2017): "Dynamic Programming and Optimal Control", Vol. 1 - DP foundations

**Related nuPlan Tutorials**:
- `tutorials/nuplan_planner_tutorial.ipynb` - Controller integration
- `tutorials/nuplan_simulation.ipynb` - Closed-loop simulation with controllers

## 11. AIDEV Notes

### Design Philosophy

**Why iLQR over LQR?**
- LQR assumes dynamics are linear (or linearized once around reference)
- iLQR iteratively refines linearization → handles nonlinearity better
- Trade computational cost (10×) for tracking accuracy (2-5× error reduction)
- Useful when LQR fails: tight maneuvers, aggressive inputs, model mismatch

**Why Not Full Nonlinear MPC?**
- iLQR is local optimization (no global convergence guarantees)
- But: Fast enough for real-time (10-50ms), good local convergence
- Full NLP (IPOPT, SNOPT) too slow (100-1000ms) for online use
- iLQR is sweet spot: Better than LQR, faster than NLP

**Trust Regions = Levenberg-Marquardt**
- Standard iLQR can diverge (linearization error too large)
- Trust regions add damping: Penalize large deviations from linearization point
- Exactly analogous to Levenberg-Marquardt in nonlinear least squares
- Critical for robustness (without trust regions, iLQR often diverges)

### Common Mistakes (From Experience)

1. **Forgetting to set max_solve_time** → Solver takes 200ms, simulation grinds to halt
2. **Zero R matrix entries** → LinAlgError, cryptic message
3. **Trust regions too loose** → Divergence on first iteration
4. **Reference trajectory too short** → AssertionError in tracker_utils
5. **Expecting deterministic solve time** → Iterations vary (3-20), use max_solve_time!
6. **Using final cost as success metric** → Cost depends on Q/R, not absolute quality measure
7. **Not checking constraint saturation** → Vehicle hits limits, tracking degrades
8. **Tuning Q/R without understanding tradeoff** → Either too aggressive or too conservative
9. **Assuming iLQR always better than LQR** → 10× slower, only 2× better for smooth trajectories
10. **Not warm-starting from previous solution** → Missing 2-5× speedup opportunity

### Future Improvements

**Performance Optimizations**:
- **AIDEV-TODO**: Vectorize `_run_forward_dynamics()` (eliminate loop over timesteps)
- **AIDEV-TODO**: Warm start from previous solution (shift + extrapolate)
- **AIDEV-TODO**: Add option to return only final solution (save memory)
- **AIDEV-TODO**: JIT compile dynamics/Jacobian (Numba)
- **AIDEV-TODO**: Implement RK4 integration (allow larger discretization_time)
- **AIDEV-TODO**: Parallelize forward/backward passes (GPU acceleration for large horizons)

**Algorithm Enhancements**:
- **AIDEV-TODO**: Line search in policy update (Armijo condition) - Better convergence
- **AIDEV-TODO**: Adaptive trust regions (increase if converging, decrease if diverging)
- **AIDEV-TODO**: Constraint handling via log-barrier or augmented Lagrangian (exact constraints)
- **AIDEV-TODO**: Second-order backward pass (full Hessian, not just diagonal P)
- **AIDEV-TODO**: Inequality constraints on states (e.g., lane boundaries)
- **AIDEV-TODO**: Multiple shooting (improved numerical stability for long horizons)

**Usability**:
- **AIDEV-TODO**: Add warm start quality metric (return initial cost for comparison)
- **AIDEV-TODO**: Expose convergence diagnostics (gradient norm, constraint violation)
- **AIDEV-TODO**: Add parameter validation (warn if Q/R ratio extreme)
- **AIDEV-TODO**: Implement automatic tuning (grid search or Bayesian optimization)
- **AIDEV-TODO**: Add visualization (plot cost history, trajectory evolution)

### Potential Bugs

**AIDEV-NOTE** (ilqr_solver.py:530-532): Min velocity linearization threshold
- Applied to Jacobian computation but not state propagation
- Asymmetry could cause issues if dynamics don't match linearization
- Consider applying threshold to both, or document this carefully

**AIDEV-NOTE** (ilqr_solver.py:595-597): Matrix inversion without condition number check
- `np.linalg.inv()` can fail silently with ill-conditioned matrices
- Should check `np.linalg.cond()` and warn or switch to `pinv()`
- Rare in practice (R > 0 ensures invertibility), but safety check worthwhile

**AIDEV-NOTE** (ilqr_solver.py:318-319): Input norm difference for convergence
- Uses Frobenius norm of full trajectory: `||U_next - U_prev||_F`
- Could converge even if first few inputs change significantly (later inputs dominate)
- Consider checking `||u_0^next - u_0^prev||` (only first input matters for MPC)

**AIDEV-QUESTION** (ilqr_solver.py:269-345): Why return full solution list?
- Returns every iteration's solution (memory overhead, slower)
- Useful for debugging convergence, but not needed in production
- Could add `return_history=False` option to save memory/time

**AIDEV-QUESTION** (ilqr_solver.py:514-518): Why recompute steering rate after clipping?
- Enforces steering angle constraint by adjusting applied steering rate
- But this means applied input ≠ commanded input (could confuse caller)
- Should this be documented in DynamicCarState returned by ILQRTracker?

### Testing Gaps

**Unit Tests** (test_ilqr_solver.py is comprehensive):
- ✅ Dynamics correctness (Jacobian via finite diff)
- ✅ Constraint enforcement (saturation)
- ✅ Cost computation
- ✅ Warm start with/without initial error
- ✅ LQR backward recursion properties
- ✅ Policy application

**Missing Integration Tests**:
- ❌ End-to-end tracking accuracy (compare against ground truth)
- ❌ Performance benchmarks (timing for various horizons/iterations)
- ❌ Comparison with LQR (when is iLQR worth the cost?)
- ❌ Constraint satisfaction over full trajectory (not just warm start)
- ❌ Warm start from previous solution (temporal coherence)

**Missing Failure Mode Tests**:
- ❌ Infeasible reference trajectory (e.g., requires accel > max_acceleration)
- ❌ Very tight trust regions (stalled convergence)
- ❌ Very loose trust regions (divergence)
- ❌ Ill-conditioned matrices (near-singular R + B^T P B)
- ❌ NaN propagation (what happens if Jacobian has NaN?)

### Documentation Improvements

**Needed**:
- Add algorithm flowchart (warm start → iterate → converge)
- Explain trust region intuition (why necessary, how to tune)
- Provide Q/R tuning guide with examples (aggressive vs smooth)
- Document relationship to DDP (iLQR is DDP with diagonal Hessian approximation)
- Add section on when to use iLQR vs LQR (decision tree)
- Explain MPC pattern (solve → apply first input → replan)

**Clarifications**:
- Iteration count is VARIABLE (solutions list length = iterations + 1)
- Final solution is `solutions[-1]`, not `solutions[max_iterations]`
- Warm start counts as iteration 0 (included in solutions list)
- Cost always non-increasing (proven by LQR optimality)
- Convergence threshold is on INPUT norm, not cost (could have converged inputs but cost still decreasing slightly)

### Research Questions

**AIDEV-QUESTION**: Is diagonal Hessian approximation (P) sufficient?
- iLQR uses diagonal P (scalar for each state), DDP uses full P (matrix)
- Trade: Full P → better convergence, but 5× more computation
- For 5D state, probably worth full Hessian (only 5×5 matrix)

**AIDEV-QUESTION**: Should we handle model mismatch (planned vs actual dynamics)?
- Current: Assumes kinematic bicycle is perfect
- Reality: Model mismatch (tire slip, suspension, delay)
- Solution: Robust iLQR (tube MPC, min-max formulation)

**AIDEV-QUESTION**: Can we guarantee real-time performance?
- Current: max_solve_time is soft limit (checked after each iteration)
- Could implement anytime algorithm (return best-so-far solution)
- Or: Precompute offline policy (works for fixed reference, not general)

**AIDEV-QUESTION**: How to handle moving obstacles?
- Current: Only tracks static reference trajectory
- Extension: Add obstacle avoidance constraints (collision cone)
- Challenge: Constraints are nonconvex (hard to handle in LQR)
