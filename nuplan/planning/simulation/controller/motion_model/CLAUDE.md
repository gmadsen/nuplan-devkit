# CLAUDE.md - nuplan/planning/simulation/controller/motion_model

## Purpose & Responsibility

**THE physics engine for ego vehicle dynamics.** This module defines `AbstractMotionModel` - the interface for propagating ego state forward in time according to vehicle physics. Motion models convert control commands (acceleration, steering rate) into updated ego states (position, velocity, heading) by integrating equations of motion. This is the "reality simulator" that determines how the vehicle actually moves in closed-loop simulation.

## Key Abstractions & Classes

### Core Interface
- **`AbstractMotionModel`** - THE fundamental physics interface (ABC)
  - **`get_state_dot(EgoState) -> EgoStateDot`** - Compute state derivatives (dx/dt, dy/dt, dθ/dt, etc.)
  - **`propagate_state(EgoState, DynamicCarState, TimePoint) -> EgoState`** - Forward integrate state over timestep
  - Used by `TwoStageController` to evolve ego vehicle state in simulation loop

### Concrete Implementations
- **`KinematicBicycleModel`** - THE standard vehicle motion model (only implementation in codebase)
  - Models vehicle as bicycle with front steering wheel
  - Rear axle is reference point (no lateral slip at rear)
  - Parameters:
    - `vehicle: VehicleParameters` - Wheelbase, dimensions, rear axle offset
    - `max_steering_angle: float = π/3` - Steering limit (±60°)
    - `accel_time_constant: float = 0.2` - Acceleration low-pass filter time constant (seconds)
    - `steering_angle_time_constant: float = 0.05` - Steering low-pass filter time constant (seconds)
  - **Key constraint**: Lateral velocity at rear axle is always zero (kinematic bicycle assumption)

### Supporting Utilities
- **`forward_integrate(init, delta, sampling_time)`** - Simple Euler integration
  - `result = init + delta * dt`
  - Used for all state propagation (position, velocity, heading, steering)
  - Located in `nuplan/planning/simulation/controller/utils.py`

## Architecture & Design Patterns

1. **State Derivative Pattern**: Separate derivative computation from integration
   - `get_state_dot()` computes rates of change (velocities → accelerations)
   - `propagate_state()` numerically integrates over time
   - Clean separation enables different integration schemes (Euler, RK4, etc.)

2. **Control Delay Modeling**: Low-pass filters simulate actuator lag
   - `_update_commands()` applies first-order lag to acceleration and steering
   - Time constants model real AV response times
   - Formula: `updated = (dt / (dt + τ)) * (ideal - current) + current`
   - Acceleration lag (τ=0.2s): Models engine/brake response
   - Steering lag (τ=0.05s): Models steering actuator response
   - Setting τ=0 disables delay (instantaneous response)

3. **Kinematic Bicycle Physics**:
   - **Reference point**: Rear axle (not center of gravity!)
   - **State variables**: (x, y, θ, v_x, a_x, δ) - position, heading, velocity, accel, steering angle
   - **Kinematic equations**:
     ```
     ẋ = v * cos(θ)                  # Global x velocity
     ẏ = v * sin(θ)                  # Global y velocity
     θ̇ = v * tan(δ) / L              # Yaw rate (L = wheelbase)
     v̇ = a_x                          # Longitudinal acceleration
     ω = v * tan(δ) / L              # Angular velocity
     ```
   - **Constraint**: Lateral velocity = 0 at rear axle (no-slip condition)

4. **Saturation & Clipping**:
   - Steering angle clipped to ±max_steering_angle
   - Prevents unrealistic steering commands from planner/tracker
   - Applied AFTER integration, before building next state

5. **Frozen Timepoint Arithmetic**: Uses `TimePoint.time_s` for numerical integration
   - Converts microsecond timestamps to float seconds
   - All integration uses seconds (not microseconds!)

## Dependencies (What We Import)

### Internal nuPlan (Documented)
- ✅ `nuplan.common.actor_state.dynamic_car_state` - DynamicCarState (control inputs)
- ✅ `nuplan.common.actor_state.ego_state` - EgoState, EgoStateDot (state representation)
- ✅ `nuplan.common.actor_state.state_representation` - StateSE2, StateVector2D, TimePoint
- ✅ `nuplan.common.actor_state.vehicle_parameters` - VehicleParameters (wheelbase, dimensions)
- ✅ `nuplan.common.geometry.compute` - principal_value() (wrap angle to [-π, π])

### Internal nuPlan (Controller Module)
- `nuplan.planning.simulation.controller.utils` - forward_integrate() (Euler integration)

### External
- `numpy` - Numerical operations (cos, sin, tan, clip)
- `abc` - Abstract base class definition

## Dependents (Who Uses This Module)

**Critical for closed-loop simulation:**
- **`nuplan.planning.simulation.controller.two_stage_controller`** - TwoStageController uses motion model to propagate ego state
  - Workflow: Planner → Trajectory → Tracker → DynamicCarState → MotionModel → EgoState
  - Motion model is stage 2 of "two stage" controller

- **`nuplan.planning.simulation.planner.simple_planner`** - SimplePlanner uses KinematicBicycleModel directly
  - Propagates constant-acceleration trajectory
  - Bypasses tracker (direct motion model usage)

- **Hydra configs** (`config/simulation/ego_controller/motion_model/`) - Instantiation via dependency injection
  - `kinematic_bicycle_model.yaml` - Default configuration

- **Unit tests** - Extensive coverage in `test/test_kinematic_motion_model.py`

## Critical Files (Prioritized)

1. **`abstract_motion_model.py`** (36 lines) - **MUST READ FIRST!**
   - Defines AbstractMotionModel interface
   - Two methods: `get_state_dot()` and `propagate_state()`
   - Simple, clean interface

2. **`kinematic_bicycle.py`** (150 lines) - **THE implementation**
   - KinematicBicycleModel - only concrete motion model in codebase
   - Three methods: `get_state_dot()`, `_update_commands()`, `propagate_state()`
   - Complete vehicle dynamics implementation
   - Reference for understanding vehicle physics in nuPlan

3. **`utils.py`** (13 lines) - Integration utility
   - forward_integrate() - Euler integration helper
   - Used throughout motion model

4. **`test/test_kinematic_motion_model.py`** (167 lines) - Validation suite
   - Tests state derivative computation
   - Tests state propagation
   - Tests steering angle limits
   - Tests control delay filters
   - Excellent examples of motion model usage

5. **`BUILD`** - Bazel build configuration

6. **`__init__.py`** - Module initialization (empty)

## Common Usage Patterns

### Instantiating Motion Model
```python
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel

# Standard vehicle (Pacifica parameters)
vehicle = get_pacifica_parameters()
motion_model = KinematicBicycleModel(vehicle)

# Custom parameters
motion_model = KinematicBicycleModel(
    vehicle=vehicle,
    max_steering_angle=np.pi / 4,  # ±45° max steering
    accel_time_constant=0.1,        # Faster acceleration response
    steering_angle_time_constant=0.02  # Faster steering response
)

# No control delay (instantaneous response)
motion_model = KinematicBicycleModel(
    vehicle=vehicle,
    accel_time_constant=0.0,
    steering_angle_time_constant=0.0
)
```

### Computing State Derivative
```python
from nuplan.common.actor_state.ego_state import EgoState

# Given current ego state
ego_state: EgoState = ...

# Compute state derivative
state_dot = motion_model.get_state_dot(ego_state)

# Access derivatives
dx_dt = state_dot.rear_axle.x       # Global x velocity [m/s]
dy_dt = state_dot.rear_axle.y       # Global y velocity [m/s]
dtheta_dt = state_dot.rear_axle.heading  # Yaw rate [rad/s]
dv_dt = state_dot.dynamic_car_state.rear_axle_velocity_2d.x  # Longitudinal accel [m/s²]
```

### Propagating State (Full Workflow)
```python
from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.state_representation import StateVector2D, TimePoint

# Current state
current_state: EgoState = ...

# Desired control inputs (from tracker)
ideal_dynamic_state = DynamicCarState.build_from_rear_axle(
    rear_axle_to_center_dist=vehicle.rear_axle_to_center,
    rear_axle_velocity_2d=StateVector2D(5.0, 0.0),  # 5 m/s forward
    rear_axle_acceleration_2d=StateVector2D(1.0, 0.0),  # 1 m/s² accel
    tire_steering_rate=0.1  # 0.1 rad/s steering rate
)

# Timestep duration
dt = TimePoint(100000)  # 0.1 seconds (100,000 microseconds)

# Propagate state
next_state = motion_model.propagate_state(
    state=current_state,
    ideal_dynamic_state=ideal_dynamic_state,
    sampling_time=dt
)

# Result: next_state contains updated position, velocity, heading at t+dt
```

### Using in Two-Stage Controller Context
```python
from nuplan.planning.simulation.controller.two_stage_controller import TwoStageController
from nuplan.planning.simulation.controller.tracker.ilqr_tracker import ILQRTracker

# Setup (happens in controller initialization)
tracker = ILQRTracker(...)
motion_model = KinematicBicycleModel(vehicle)
controller = TwoStageController(scenario, tracker, motion_model)

# Simulation loop (simplified)
# 1. Planner generates trajectory
trajectory = planner.compute_planner_trajectory(current_input)

# 2. Tracker computes control commands
dynamic_state = tracker.track_trajectory(
    current_iteration, next_iteration, ego_state, trajectory
)

# 3. Motion model propagates state
next_ego_state = motion_model.propagate_state(
    state=ego_state,
    ideal_dynamic_state=dynamic_state,
    sampling_time=dt
)
```

### Direct Integration in Planner (SimplePlanner Pattern)
```python
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner

class MyPlanner(AbstractPlanner):
    def __init__(self, ...):
        self.vehicle = get_pacifica_parameters()
        self.motion_model = KinematicBicycleModel(self.vehicle)
        self.sampling_time = TimePoint(100000)  # 0.1s

    def compute_planner_trajectory(self, current_input):
        ego_state, _ = current_input.history.current_state

        # Build trajectory by forward propagation
        trajectory = [ego_state]
        state = ego_state

        for _ in range(80):  # 8 seconds @ 0.1s
            # Create desired dynamic state
            dynamic_state = DynamicCarState.build_from_rear_axle(
                rear_axle_to_center_dist=state.car_footprint.rear_axle_to_center_dist,
                rear_axle_velocity_2d=state.dynamic_car_state.rear_axle_velocity_2d,
                rear_axle_acceleration_2d=StateVector2D(0.5, 0.0),  # Constant accel
                tire_steering_rate=0.0  # Straight
            )

            # Propagate
            state = self.motion_model.propagate_state(
                state, dynamic_state, self.sampling_time
            )
            trajectory.append(state)

        return InterpolatedTrajectory(trajectory)
```

## Gotchas & Edge Cases

1. **Rear Axle Reference Point**: Motion model uses rear axle, NOT center of gravity!
   - `EgoState.rear_axle` is the integration reference
   - `EgoState.center` is derived from rear_axle + vehicle geometry
   - Kinematic equations assume no lateral slip at rear wheels
   - Tracker outputs must be in rear axle frame!

2. **Steering Angle vs Steering Rate**: Input is steering RATE, state tracks steering ANGLE
   - `DynamicCarState.tire_steering_rate` [rad/s] - input to propagate_state()
   - `EgoState.tire_steering_angle` [rad] - integrated state variable
   - Integration: `next_angle = current_angle + rate * dt`
   - Don't confuse them - they're different units!

3. **Control Delay Is Always Applied**: Low-pass filters run even if time constants are default
   - Default `accel_time_constant=0.2` adds ~0.2s lag to acceleration
   - Default `steering_angle_time_constant=0.05` adds ~0.05s lag to steering
   - Set to 0.0 for instantaneous response (testing/debugging)
   - Real AVs have actuator delays - model captures this!

4. **Lateral Velocity Always Zero**: Kinematic assumption, not dynamic reality
   - `next_point_velocity_y = 0.0` is hardcoded (line 118)
   - No tire slip, no drifting, no lateral dynamics
   - For high-speed or aggressive maneuvers, this is WRONG!
   - Use dynamic model (not in codebase) for racing/emergency maneuvers

5. **Heading Wrapping**: Angles wrapped to [-π, π] after integration
   - Uses `principal_value()` from geometry module
   - Prevents heading from growing unbounded
   - Critical for multi-lap scenarios or long routes

6. **Steering Saturation Timing**: Clipping happens AFTER integration
   - Ideal steering is integrated first
   - Then np.clip() enforces ±max_steering_angle
   - Means steering can "saturate" - rate is ignored when at limit
   - Controller should avoid commanding saturated steering!

7. **TimePoint Unit Confusion**: Integration uses `.time_s`, not `.time_us`
   - TimePoint stores microseconds internally
   - `sampling_time.time_s` converts to float seconds
   - Forgetting `.time_s` causes 1e6x scaling errors!

8. **Angular Velocity Recomputation**: Not integrated, recomputed from velocity and steering
   - `next_point_angular_velocity = v * tan(δ) / L` (line 128-130)
   - Not: `angular_velocity + angular_accel * dt`
   - Kinematic constraint (not a free state variable)
   - angular_accel is derived (line 136), not integrated

9. **State Derivative Return Type**: `get_state_dot()` returns EgoStateDot, not EgoState
   - EgoStateDot is a special state representation for derivatives
   - Has same structure as EgoState but semantics are rates
   - Don't use EgoStateDot as EgoState - type system allows it but semantics differ!

10. **Zero Acceleration Derivative**: Jerk is assumed zero in state_dot
    - `rear_axle_acceleration_2d=StateVector2D(0.0, 0.0)` (line 47)
    - Means acceleration is piecewise constant over timestep
    - No jerk modeling - acceleration jumps instantaneously
    - For comfort metrics, this is unrealistic!

11. **Control Command Update Before Propagation**: `_update_commands()` runs first
    - Creates `propagating_state` with filtered commands
    - THEN computes derivatives and integrates
    - Means applied commands lag ideal commands by filter delay
    - Two-step process: filter → integrate

12. **Pacifica Hardcoded in SimplePlanner**: SimplePlanner uses Pacifica vehicle params
    - `get_pacifica_parameters()` returns fixed wheelbase, dimensions
    - Motion model is vehicle-agnostic, but planner isn't
    - For custom vehicles, pass VehicleParameters explicitly

13. **Max Steering Angle Default (π/3 = 60°)**: Conservative for many vehicles
    - Real cars: ~30-45° typical max steering
    - Trucks/buses: ~20-30°
    - Sports cars: ~45-50°
    - Default π/3 may be too permissive - check your vehicle!

14. **Euler Integration Accuracy**: First-order integration has O(dt²) error
    - Acceptable for dt=0.1s typical simulation
    - For dt>0.5s, errors accumulate rapidly
    - For high-fidelity simulation, need RK4 or higher-order integrator
    - forward_integrate() is simple Euler - no adaptive stepping!

15. **No Validation of DynamicCarState**: Inputs are not checked for physical feasibility
    - Can pass infinite acceleration - motion model will integrate it!
    - Can pass conflicting steering rate and angle - filters apply blindly
    - Garbage in, garbage out - tracker must provide valid commands

## Performance Considerations

### Computational Cost
- **Extremely fast**: ~10-20 μs per propagate_state() call (pure Python + NumPy)
- Dominated by trigonometric functions (sin, cos, tan)
- No matrix operations, no iterative solvers
- Much faster than dynamic models or MPC controllers

### Memory Footprint
- Tiny: ~1 KB per KinematicBicycleModel instance
- No state history, no caching
- Stateless computation (except model parameters)

### Numerical Stability
- **Stable for normal driving**: dt ≤ 0.1s, speeds < 50 m/s, steering < 60°
- **Unstable at high speeds with tight steering**: tan(δ) → ∞ as δ → π/2
  - Yaw rate formula: θ̇ = v·tan(δ)/L blows up at δ=90°
  - max_steering_angle prevents this, but be careful!
- **Stable integration**: Euler method is stable for kinematic systems (no stiffness)

### Optimization Opportunities
- **None needed!** Already extremely fast
- Bottleneck is NOT motion model - it's planning, perception, map queries
- Profiling shows <0.1% of simulation time in propagate_state()

## Related Documentation

### Parent Module
- `nuplan/planning/simulation/controller/CLAUDE.md` - Controller architecture (if exists)

### Critical Dependencies (Documented)
- ✅ `nuplan/common/actor_state/CLAUDE.md` - EgoState, DynamicCarState, VehicleParameters
- ✅ `nuplan/common/geometry/CLAUDE.md` - principal_value() for angle wrapping
- ✅ `nuplan/common/actor_state/state_representation` - StateSE2, StateVector2D, TimePoint

### Dependents (Documented)
- ✅ `nuplan/planning/simulation/planner/CLAUDE.md` - SimplePlanner uses motion model directly
- `nuplan/planning/simulation/controller/tracker/CLAUDE.md` - Trackers provide DynamicCarState inputs (if exists)

### Sibling Modules
- `nuplan/planning/simulation/controller/tracker/` - Tracker computes control commands
- `nuplan/planning/simulation/controller/two_stage_controller.py` - Integrates tracker + motion model

### Related Concepts
- **Kinematic vs Dynamic Models**: Kinematic = no forces, dynamic = tire forces, inertia, slip
  - nuPlan uses kinematic (faster, simpler, sufficient for planning research)
  - For vehicle dynamics research, need dynamic model (not in codebase)
- **Bicycle vs Ackermann Model**: Bicycle = single front wheel, Ackermann = left/right steering
  - nuPlan uses bicycle (standard simplification)
  - Ackermann needed for differential steering (not in codebase)

---

## AIDEV Notes

**AIDEV-NOTE**: KinematicBicycleModel is THE only motion model in nuPlan. All simulation uses this. No dynamic models, no particle models, no learned models. If you need different physics, extend AbstractMotionModel!

**AIDEV-NOTE**: Rear axle reference point is critical - many bugs come from confusion between rear_axle and center coordinates. Always check which frame you're in!

**AIDEV-NOTE**: Control delay modeling (_update_commands) is subtle - it applies BEFORE state propagation, not after. This means the delay is in the loop, not post-integration.

**AIDEV-TODO**: Consider adding RK4 integration option for higher-fidelity simulation. Euler is fast but accumulates error over long horizons.

**AIDEV-TODO**: Add input validation to propagate_state() - check for NaN, inf, unrealistic acceleration/steering. Silent failures are hard to debug.

**AIDEV-TODO**: Consider parameterizing max acceleration (currently unbounded). Real AVs have comfort limits (~2-3 m/s²).

**AIDEV-QUESTION**: Why is angular_accel computed via finite difference (line 136) instead of using kinematic derivative? Is this for numerical stability?

**AIDEV-QUESTION**: Should lateral velocity tolerance be added? Currently hardcoded zero, but numerical integration might accumulate small errors.

**AIDEV-QUESTION**: Why does get_state_dot() return acceleration in rear_axle_velocity_2d field (line 46)? Semantics are confusing - it's acceleration, not velocity!

## Physics Equations Reference

For reference, the complete kinematic bicycle model equations:

### State Vector
```
x = [x, y, θ, v, δ]ᵀ
  where:
    x, y = rear axle position (global frame) [m]
    θ = heading (global frame) [rad]
    v = longitudinal velocity at rear axle [m/s]
    δ = front wheel steering angle [rad]
```

### Control Inputs
```
u = [a, δ̇]ᵀ
  where:
    a = longitudinal acceleration [m/s²]
    δ̇ = steering rate [rad/s]
```

### Kinematic Equations (Continuous Time)
```
ẋ = v·cos(θ)
ẏ = v·sin(θ)
θ̇ = (v/L)·tan(δ)     [L = wheelbase]
v̇ = a
δ̇ = steering_rate
```

### Discrete-Time Integration (Euler)
```
x[k+1] = x[k] + ẋ[k]·Δt
y[k+1] = y[k] + ẏ[k]·Δt
θ[k+1] = wrap(θ[k] + θ̇[k]·Δt)
v[k+1] = v[k] + v̇[k]·Δt
δ[k+1] = clip(δ[k] + δ̇[k]·Δt, -δ_max, δ_max)
```

### Derived Quantities
```
ω = (v/L)·tan(δ)      [angular velocity]
α = (ω - ω_prev)/Δt   [angular acceleration]
v_y = 0               [lateral velocity, kinematic constraint]
```

### Control Delay Filter (First-Order Lag)
```
a_filtered = a_prev + (Δt/(Δt + τ_a))·(a_ideal - a_prev)
δ_filtered = δ_prev + (Δt/(Δt + τ_δ))·(δ_ideal - δ_prev)
  where:
    τ_a = accel_time_constant [s]
    τ_δ = steering_angle_time_constant [s]
```

This is the standard kinematic bicycle model used in robotics and automotive control!
