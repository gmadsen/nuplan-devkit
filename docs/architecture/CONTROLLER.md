# Controller & Trajectory Execution Architecture

## Overview

The **controller system** bridges high-level planner trajectories and low-level vehicle dynamics, executing planned trajectories while respecting physical constraints and control delays. Controllers transform the planner's desired future trajectory into actual ego vehicle motion, either by perfectly tracking (idealized), replaying logged data (oracle), or using realistic two-stage controllers (tracker + motion model) that simulate control latency and actuation limits.

## Purpose

Controllers enable:
1. **Trajectory tracking** - Convert planned trajectory to ego state updates
2. **Realistic simulation** - Model control delays, steering saturation, acceleration limits
3. **Flexible execution** - Support perfect tracking (debugging), logged playback (baseline), or realistic controllers
4. **Validation** - Ensure planned trajectories are feasible given vehicle dynamics

## Architecture Overview

### Controller Role in Simulation Loop

```
Simulation Loop:
├─ planner.compute_planner_trajectory() → trajectory_t
│  └─ Returns: Future ego states from t=0 to t=8 seconds
│
└─ controller.update_state(iteration_t, iteration_t+1, ego_t, trajectory_t)
   │
   ├─ INPUT:
   │  ├─ current_iteration: Timestep at t
   │  ├─ next_iteration: Timestep at t+dt (t+0.1s)
   │  ├─ ego_state: Current ego pose and velocity at t
   │  └─ trajectory: Planned trajectory covering t to t+8s
   │
   ├─ PROCESSING:
   │  ├─ Query trajectory at next_iteration time
   │  └─ Apply vehicle dynamics (kinematics/dynamics, delays, saturation)
   │
   └─ OUTPUT:
      └─ ego_state is updated: ego_state_t → ego_state_t+1
         ├─ Position updated
         ├─ Velocity updated
         ├─ Heading updated
         └─ Acceleration, steering updated
```

## Key Abstractions

### AbstractEgoController - Base Interface

```python
class AbstractEgoController(abc.ABC):
    """Base class for ego vehicle control."""
    
    @abstractmethod
    def reset(self) -> None:
        """Reset controller state for new scenario."""
        # Clear any accumulated state
        # Reset to initial conditions
    
    @abstractmethod
    def get_state(self) -> EgoState:
        """Return current ego state (pose, velocity, acceleration)."""
        # Called to read current state
        # Used in get_planner_input() to construct history
    
    @abstractmethod
    def update_state(
        self,
        current_iteration: SimulationIteration,
        next_iteration: SimulationIteration,
        ego_state: EgoState,
        trajectory: AbstractTrajectory,
    ) -> None:
        """
        Advance ego state to next timestep based on trajectory.
        
        Args:
            current_iteration: Timestep at t
            next_iteration: Timestep at t+dt
            ego_state: Current ego state at t (for some controllers)
            trajectory: Planned trajectory (covers t to t+horizon)
        
        Called once per simulation step (typically 0.1s).
        This is where ego motion is actually computed!
        """
        pass
```

## Concrete Controller Implementations

### 1. PerfectTrackingController - Idealized Control

```python
class PerfectTrackingController(AbstractEgoController):
    """Perfect trajectory tracking without control error."""
    
    def __init__(self):
        super().__init__()
        self._current_state = None
    
    def reset(self):
        self._current_state = None
    
    def get_state(self) -> EgoState:
        return self._current_state
    
    def update_state(
        self,
        current_iteration,
        next_iteration,
        ego_state,
        trajectory,
    ):
        """
        Sample trajectory at next_iteration time.
        
        Process:
        1. Extract target time from next_iteration
        2. Query trajectory at this time
        3. Directly set ego state (no control dynamics!)
        4. Validate velocity < 50 m/s (safety check)
        """
        # Get state at next timestep from trajectory
        target_time = next_iteration.time_point
        
        # Query trajectory (with interpolation)
        self._current_state = trajectory.get_state_at_time(target_time)
        
        # Safety check: velocity must be reasonable
        if self._current_state.velocity > 50.0:  # m/s
            raise RuntimeError(
                f"Velocity exceeds limit: {self._current_state.velocity} m/s"
            )
```

**Use case**: 
- Testing planners without control noise
- Debugging planner logic in isolation
- Baseline for comparison

**Assumptions**: 
- Trajectory covers entire planning horizon
- Vehicle can execute any desired acceleration/steering
- No control delays or actuation limits

**Gotcha**: Fails if trajectory is too short or doesn't cover next_iteration time.

### 2. LogPlaybackController - Oracle Control

```python
class LogPlaybackController(AbstractEgoController):
    """Replay ego states from scenario database (oracle)."""
    
    def __init__(self, scenario):
        super().__init__()
        self._scenario = scenario
        self._current_state = None
    
    def reset(self):
        self._current_state = None
    
    def get_state(self) -> EgoState:
        return self._current_state
    
    def update_state(
        self,
        current_iteration,
        next_iteration,
        ego_state,
        trajectory,  # IGNORED! Uses logged data instead
    ):
        """
        Query ego state from scenario database.
        
        Note: trajectory parameter is IGNORED.
        This controller doesn't execute planned trajectories!
        """
        # Load ego state from scenario DB at next_iteration
        self._current_state = self._scenario.get_ego_state_at_iteration(
            next_iteration.index
        )
```

**Use case**: 
- Oracle baseline (perfect execution)
- Measuring observation quality in isolation (without planner errors)
- Debugging scenarios without planner influence

**Key property**: Does NOT execute planner's trajectory - uses logged data only.

**Limitation**: Cannot be used for closed-loop evaluation (no feedback from planner to agents).

### 3. TwoStageController - Realistic Control

```python
class TwoStageController(AbstractEgoController):
    """Realistic trajectory tracking with tracker + motion model."""
    
    def __init__(self, tracker: AbstractTracker, motion_model: AbstractMotionModel):
        super().__init__()
        self._tracker = tracker            # Computes desired dynamics
        self._motion_model = motion_model  # Propagates state with delays
        self._current_state = None
    
    def reset(self):
        self._current_state = None
        self._tracker.reset()
        self._motion_model.reset()
    
    def get_state(self) -> EgoState:
        return self._current_state
    
    def update_state(
        self,
        current_iteration,
        next_iteration,
        ego_state,
        trajectory,
    ):
        """
        Two-stage control: tracker → motion model
        
        Stage 1: Tracker
        ├─ Inputs: current ego state, trajectory, iteration info
        ├─ Computes: desired acceleration, steering rate
        └─ Outputs: DynamicCarState (ideal_acceleration, ideal_steering)
        
        Stage 2: Motion Model
        ├─ Inputs: current state, desired dynamics
        ├─ Applies: control delays, saturation
        └─ Propagates: state forward using kinematics/dynamics
        """
        # Stage 1: Compute desired dynamics via tracker
        ideal_dynamics = self._tracker.track_trajectory(
            current_iteration=current_iteration,
            next_iteration=next_iteration,
            initial_state=ego_state,
            trajectory=trajectory,
        )
        # Returns: DynamicCarState with acceleration, steering_rate
        
        # Stage 2: Propagate state with motion model
        self._current_state = self._motion_model.propagate_state(
            current_state=ego_state,
            ideal_dynamic_state=ideal_dynamics,  # What tracker wants
            sampling_time=(next_iteration.time_s - current_iteration.time_s),
        )
        # Returns: Updated EgoState with control delays applied
```

**Use case**: 
- Closed-loop evaluation with realistic control behavior
- Testing robustness to control delays and actuation limits
- Competition submission (often required for realism)

**Two-stage philosophy**:
- **Tracker**: "What should the vehicle do to follow this trajectory?"
- **Motion model**: "Given those commands, what actually happens?"

## Tracker Implementations

### AbstractTracker - Trajectory Tracking Interface

```python
class AbstractTracker(abc.ABC):
    """Base class for trajectory tracking controllers."""
    
    @abstractmethod
    def track_trajectory(
        self,
        current_iteration: SimulationIteration,
        next_iteration: SimulationIteration,
        initial_state: EgoState,
        trajectory: AbstractTrajectory,
    ) -> DynamicCarState:
        """
        Compute desired acceleration and steering to track trajectory.
        
        Returns:
            DynamicCarState with:
            ├─ acceleration: m/s² (positive = forward)
            └─ steering_rate: rad/s (positive = left turn)
        
        The motion model will apply these with delays and saturation.
        """
        pass
```

### LQRTracker - Linear Quadratic Regulator

```python
class LQRTracker(AbstractTracker):
    """
    Decoupled LQR trajectory tracking.
    
    Approach:
    ├─ Split into 2 independent control problems:
    │  ├─ Longitudinal (velocity control): minimize acceleration effort
    │  └─ Lateral (steering control): minimize lateral error + heading error
    │
    └─ Solve each as linear quadratic regulator (LQR) problem
    """
    
    def __init__(
        self,
        wheelbase: float = 2.7,  # Vehicle wheelbase
        q_longitudinal: np.ndarray = np.array([1.0]),  # Velocity error weight
        r_longitudinal: np.ndarray = np.array([0.1]),  # Acceleration effort weight
        q_lateral: np.ndarray = np.array([1.0, 1.0, 0.1]),  # Error weights
        r_lateral: np.ndarray = np.array([0.5]),  # Steering effort weight
    ):
        self.wheelbase = wheelbase
        self.q_longitudinal = q_longitudinal
        self.r_longitudinal = r_longitudinal
        self.q_lateral = q_lateral
        self.r_lateral = r_lateral
    
    def track_trajectory(self, current_iteration, next_iteration, initial_state, trajectory):
        """
        Compute optimal acceleration and steering.
        
        Process:
        1. Sample desired state from trajectory at next_iteration
        2. Compute lateral errors:
           ├─ Position error: ego_y - desired_y (in ego frame)
           ├─ Heading error: ego_heading - desired_heading
           └─ Steering angle error: current_steering - desired_steering
        3. Formulate LQR problem for lateral control
        4. Solve LQR (closed-form or iterative)
        5. Return control input: steering rate
        6. Similarly for longitudinal (velocity control → acceleration)
        
        Assumptions:
        ├─ Small angle approximation (heading ~= 0)
        ├─ Kinematic bicycle model
        └─ Linear system dynamics
        """
        # Get desired state from trajectory
        desired_state = trajectory.get_state_at_time(next_iteration.time_point)
        
        # Compute tracking errors
        # (Transform to ego-centric frame for control)
        position_error = self._compute_position_error(initial_state, desired_state)
        heading_error = self._compute_heading_error(initial_state, desired_state)
        
        # Solve LQR for steering
        # Minimize: ||[lateral_error, heading_error, steering_angle]||_Q + ||steering_rate||_R
        steering_rate = self._solve_lateral_lqr(position_error, heading_error)
        
        # Solve LQR for acceleration  
        # Minimize: ||velocity_error||_Q + ||acceleration||_R
        acceleration = self._solve_longitudinal_lqr(initial_state.velocity, desired_state.velocity)
        
        return DynamicCarState(acceleration=acceleration, steering_rate=steering_rate)
```

**Strengths**:
- Theoretically sound (optimal control)
- Fast computation (closed-form solution)
- Handles tracking constraints naturally

**Weaknesses**:
- Requires linearization (assumes small errors)
- May diverge for large tracking errors
- Needs tuning of Q/R weights

### ILQRTracker - Iterative LQR (Advanced)

```python
class ILQRTracker(AbstractTracker):
    """
    Iterative Linear Quadratic Regulator (iLQR).
    
    Improvement over LQR:
    ├─ Linearizes around actual trajectory (not just current state)
    ├─ Iteratively refines solution
    └─ Better for large tracking errors
    
    Trade-off: Slower but more robust
    """
```

## Motion Model Implementations

### AbstractMotionModel - Vehicle Dynamics Interface

```python
class AbstractMotionModel(abc.ABC):
    """Base class for vehicle dynamics models."""
    
    @abstractmethod
    def propagate_state(
        self,
        current_state: EgoState,
        ideal_dynamic_state: DynamicCarState,
        sampling_time: float,
    ) -> EgoState:
        """
        Advance vehicle state given desired dynamics.
        
        Input:
            current_state: Position, velocity, heading at t
            ideal_dynamic_state: Desired acceleration, steering_rate
            sampling_time: dt (usually 0.1s)
        
        Output:
            Updated EgoState at t+dt with:
            ├─ Position advanced
            ├─ Velocity updated (with acceleration limit)
            ├─ Heading rotated (with steering rate limit)
            └─ Control delays applied (1st order lag)
        """
        pass
```

### KinematicBicycleModel - Most Common

```python
class KinematicBicycleModel(AbstractMotionModel):
    """
    Kinematic bicycle model with control delays.
    
    Vehicle model:
    ├─ 5 states: [x, y, heading, velocity, steering_angle]
    ├─ 2 inputs: [acceleration, steering_rate]
    └─ Dynamics:
       ├─ x_dot = v * cos(heading)
       ├─ y_dot = v * sin(heading)
       ├─ heading_dot = (v / wheelbase) * tan(steering_angle)
       ├─ v_dot = acceleration (with clipping)
       └─ steering_angle_dot = steering_rate (with clipping)
    
    Control delays:
    ├─ Acceleration lags with time constant (e.g., 0.2s)
    └─ Steering lags with time constant (e.g., 0.05s)
    
    Saturation:
    ├─ max_steering_angle: ±60° (π/3 radians)
    ├─ max_acceleration: ±4.0 m/s²
    └─ max_steering_rate: Limited by actuators
    """
    
    def __init__(
        self,
        wheelbase: float = 2.7,
        accel_time_constant: float = 0.2,  # Response time
        steering_angle_time_constant: float = 0.05,  # Response time
        max_steering_angle: float = np.pi / 3,  # 60 degrees
        max_acceleration: float = 4.0,
        max_steering_rate: float = np.pi,  # rad/s
    ):
        self.wheelbase = wheelbase
        self.accel_time_constant = accel_time_constant
        self.steering_angle_time_constant = steering_angle_time_constant
        self.max_steering_angle = max_steering_angle
        self.max_acceleration = max_acceleration
        self.max_steering_rate = max_steering_rate
    
    def propagate_state(
        self,
        current_state: EgoState,
        ideal_dynamic_state: DynamicCarState,
        sampling_time: float,
    ) -> EgoState:
        """
        Propagate state using kinematic bicycle model.
        
        Steps:
        1. Apply low-pass filter to control inputs (simulate delays)
           ├─ acceleration_actual = current_accel + (ideal_accel - current_accel) * (dt / tau)
           └─ Similar for steering_rate
        
        2. Clip actuator limits
           ├─ acceleration_actual = clip(acceleration_actual, -4.0, 4.0)
           └─ steering_angle = clip(steering_angle, -π/3, π/3)
        
        3. Integrate state forward using Euler method:
           ├─ x += v * cos(heading) * dt
           ├─ y += v * sin(heading) * dt
           ├─ heading += (v / wheelbase) * tan(steering_angle) * dt
           ├─ v += acceleration * dt
           └─ steering_angle += steering_rate * dt
        
        4. Wrap heading to [-π, π]
        
        5. Return updated EgoState
        """
        # Apply control delays (1st order lag)
        actual_accel = self._apply_lag(
            current_state.acceleration,
            ideal_dynamic_state.acceleration,
            sampling_time,
            self.accel_time_constant,
        )
        
        actual_steering_rate = self._apply_lag(
            self._current_steering_rate,
            ideal_dynamic_state.steering_rate,
            sampling_time,
            self.steering_angle_time_constant,
        )
        
        # Clip actuator saturation
        actual_accel = np.clip(actual_accel, -self.max_acceleration, self.max_acceleration)
        actual_steering_rate = np.clip(
            actual_steering_rate,
            -self.max_steering_rate,
            self.max_steering_rate
        )
        
        # Propagate using kinematic bicycle model
        next_state = self._integrate_state(
            current_state,
            actual_accel,
            actual_steering_rate,
            sampling_time,
        )
        
        return next_state
    
    def _apply_lag(self, current, desired, dt, time_constant):
        """
        1st order low-pass filter: x_new = x + (x_desired - x) * (dt / tau)
        
        Interpretation:
        ├─ tau = 0: Instantaneous response (no delay)
        ├─ tau = 0.2: 20% response per 0.1s timestep
        └─ tau = large: Slow response, significant lag
        """
        if time_constant <= 0:
            return desired  # No delay
        
        gain = min(1.0, dt / time_constant)  # Clamp to [0, 1]
        return current + (desired - current) * gain
```

**Key parameters**:
- `wheelbase` [m]: Distance from front to rear axle
- `accel_time_constant` [s]: How quickly acceleration reaches desired value
- `steering_angle_time_constant` [s]: How quickly steering responds
- `max_steering_angle` [rad]: ±60° = π/3 typical for cars

**Typical response**:
- Acceleration step: Takes ~200ms to reach steady state (with 0.2s time constant)
- Steering step: Takes ~50ms to reach steady state (with 0.05s time constant)
- Very realistic! Models actual vehicle actuators.

## Integration with Simulation Loop

```
propagate(trajectory):
├─ controller.update_state(current_iteration, next_iteration, ego_state, trajectory)
│  │
│  └─ TwoStageController:
│     ├─ tracker.track_trajectory()
│     │  └─ Returns: DynamicCarState(acceleration, steering_rate)
│     │
│     └─ motion_model.propagate_state()
│        ├─ Apply control delays (1st order lag)
│        ├─ Integrate kinematics (Euler method)
│        └─ Returns: Updated EgoState
│
└─ Updated self._ego_controller._current_state now reflects executed trajectory
   (with control delays and actuator limits applied)
```

## Performance Timing

| Operation | Time | Notes |
|-----------|------|-------|
| PerfectTrackingController | <1ms | Direct trajectory interpolation |
| LQRTracker | 2-5ms | LQR solve + integration |
| KinematicBicycleModel | 1-2ms | Forward Euler integration |
| **Total per step** | **3-10ms** | Part of 85-100ms total step time |

**Controller time is small** (~5-10% of total), dominated by planner compute time.

## Common Patterns

### Pattern 1: Tuning LQR Weights

```python
# Conservative tracking (low error tolerance)
controller = TwoStageController(
    tracker=LQRTracker(
        q_lateral=np.array([10.0, 10.0, 1.0]),  # Heavy penalty on lateral error
        r_lateral=np.array([0.1]),               # Cheap steering effort
    ),
    motion_model=KinematicBicycleModel()
)

# Aggressive tracking (prioritize smooth motion)
controller = TwoStageController(
    tracker=LQRTracker(
        q_lateral=np.array([1.0, 1.0, 0.1]),   # Light penalty
        r_lateral=np.array([1.0]),              # Expensive steering effort
    ),
    motion_model=KinematicBicycleModel()
)
```

### Pattern 2: Adjusting Control Delays

```python
# Responsive vehicle (sports car)
motion_model = KinematicBicycleModel(
    accel_time_constant=0.05,      # 50ms acceleration response
    steering_angle_time_constant=0.02,  # 20ms steering response
)

# Sluggish vehicle (truck)
motion_model = KinematicBicycleModel(
    accel_time_constant=0.5,       # 500ms acceleration response
    steering_angle_time_constant=0.2,   # 200ms steering response
)
```

## Gotchas & Anti-Patterns

### Gotcha 1: Trajectory Too Short

```python
# ❌ WRONG - Trajectory only covers 2 seconds ahead
trajectory = InterpolatedTrajectory(states_t0_t2)

# In propagate(), when we ask for state at t=3s:
next_state = trajectory.get_state_at_time(time_t3)  # Extrapolation error!

# ✅ CORRECT - Trajectory covers 8+ seconds
trajectory = InterpolatedTrajectory(states_t0_t8)  # 80 states
```

### Gotcha 2: Forgetting Heading Wrapping

```python
# ❌ WRONG - Heading accumulates past 2π
heading = heading + heading_dot * dt  # If heading_dot large, heading grows unbounded

# ✅ CORRECT - Wrap to [-π, π]
heading = heading + heading_dot * dt
heading = principal_value(heading)  # Wraps to [-π, π]
```

### Gotcha 3: Wrong Wheelbase Value

```python
# ❌ WRONG - Incorrect vehicle geometry
kinematic_model = KinematicBicycleModel(wheelbase=2.0)  # Wrong!

# ✅ CORRECT - Match actual vehicle
kinematic_model = KinematicBicycleModel(wheelbase=2.7)  # Typical car
```

## Cross-References

- **[SIMULATION_CORE.md](./SIMULATION_CORE.md)** - How controller fits in simulation loop
- **[controller/CLAUDE.md](../nuplan/planning/simulation/controller/CLAUDE.md)** - Complete controller API
- **[controller/motion_model/CLAUDE.md](../nuplan/planning/simulation/controller/motion_model/CLAUDE.md)** - Motion model details
- **[controller/tracker/CLAUDE.md](../nuplan/planning/simulation/controller/tracker/CLAUDE.md)** - Tracker implementations

---

**AIDEV-NOTE**: For 0.51x → 1.0x realtime improvement, controller is likely NOT the bottleneck (3-10ms vs 80ms planner). Focus optimization efforts on planner.

**AIDEV-NOTE**: PerfectTrackingController is great for debugging planner logic without control error confounding results. Switch to TwoStageController for realistic evaluation.

