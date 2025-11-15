# CLAUDE.md - nuplan/planning/simulation/trajectory

## Purpose & Responsibility

Trajectory representations for planned and predicted motion. Planners output `AbstractTrajectory` to represent ego's future path; predictors output trajectories for other agents. The primary implementation `InterpolatedTrajectory` provides temporal interpolation of states (EgoState, Waypoint, etc.) using scipy for linear states and angular interpolation for heading.

## Key Abstractions & Classes

### Core Interface
- **`AbstractTrajectory`** - Generic trajectory interface (ABC)
  - `start_time` / `end_time` → TimePoint boundaries
  - `duration` / `duration_us` → Trajectory length in seconds / microseconds
  - `get_state_at_time(time_point)` → Interpolated state at specific time
  - `get_state_at_times(time_points)` → Batch interpolation
  - `get_sampled_trajectory()` → Original discrete waypoints
  - `is_in_range(time_point)` → Check if time is within trajectory bounds

### Concrete Implementations
- **`InterpolatedTrajectory`** - Main implementation (uses scipy interpolation)
  - Constructor: `InterpolatedTrajectory(trajectory: List[InterpolatableState])`
  - Requires >= 2 states minimum
  - Automatically splits states into linear (x, y, velocity) and angular (heading) components
  - Linear: `scipy.interpolate.interp1d` (piecewise linear)
  - Angular: `AngularInterpolator` (handles heading wraparound at ±π)
  - Supports any `InterpolatableState` (EgoState, Waypoint, AgentState)

- **`PredictedTrajectory`** - For agent predictions
  - Adds probability/confidence to trajectory
  - Used in `Agent.predictions`

### Supporting Classes
- **`TrajectoryS

ampling`** - Time sampling utilities
  - `AbstractTrajectoryS ampling` - Interface for sampling strategies
  - Used to generate time points for trajectory queries

## Architecture & Design Patterns

1. **Interpolation Strategy**: Separate linear vs angular interpolation
   - Linear states (x, y, vx, vy, etc.): scipy.interpolate.interp1d
   - Angular states (heading): Custom `AngularInterpolator` handles ±π wraparound
   - Prevents heading "unwrapping" artifacts (e.g., -π → +π discontinuity)

2. **State Decomposition**: `InterpolatableState` protocol
   - `to_split_state()` → SplitState(linear, angular, fixed)
   - `from_split_state(split)` → Reconstruct state
   - Fixed states (vehicle dimensions) not interpolated

3. **Type Polymorphism**: Works with any `InterpolatableState`
   - EgoState, Waypoint, AgentState all supported
   - Trajectory remembers original class type for reconstruction

4. **Lazy Evaluation**: Interpolation on-demand
   - Stores original waypoints + interpolation functions
   - Only compute intermediate states when queried

## Dependencies

### Internal nuPlan
- `nuplan.common.actor_state.state_representation` - TimePoint
- `nuplan.common.utils.interpolatable_state` - InterpolatableState protocol
- `nuplan.common.utils.split_state` - SplitState decomposition
- `nuplan.common.geometry.compute` - AngularInterpolator

### External
- `scipy.interpolate` - Linear interpolation (interp1d)
- `numpy` - Array operations

## Dependents

- `nuplan/planning/simulation/planner/` - Planners return AbstractTrajectory
- `nuplan/planning/simulation/controller/` - Controllers track trajectories
- `nuplan/planning/simulation/predictor/` - Predictors output agent trajectories
- `nuplan/planning/metrics/` - Metrics evaluate trajectories
- `nuplan/planning/training/` - ML models predict/generate trajectories

## Critical Files

1. **`abstract_trajectory.py`** (82 lines) - Core interface
2. **`interpolated_trajectory.py`** (109 lines) - Main implementation
3. **`predicted_trajectory.py`** - Agent prediction trajectories
4. **`trajectory_sampling.py`** - Time sampling utilities

## Common Usage Patterns

### Create Trajectory from EgoStates
```python
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

# Plan sequence of future ego states
future_states = [ego_t0, ego_t1, ego_t2, ...]  # List[EgoState]
trajectory = InterpolatedTrajectory(future_states)
```

### Query Trajectory at Specific Time
```python
from nuplan.common.actor_state.state_representation import TimePoint, TimeDuration

# Get state at 1.5 seconds
t = TimePoint(int(1.5e6))  # microseconds
state = trajectory.get_state_at_time(t)

# Batch query
times = [TimePoint(int(t * 1e6)) for t in [1.0, 1.5, 2.0, 2.5]]
states = trajectory.get_state_at_times(times)
```

### Common Pattern in Controllers
```python
# Controller tracking a planner trajectory
current_time = ego_state.time_point
lookahead = TimeDuration.from_s(0.5)  # 500ms lookahead
target_time = current_time + lookahead

if trajectory.is_in_range(target_time):
    target_state = trajectory.get_state_at_time(target_time)
    control = compute_control_to_reach(ego_state, target_state)
```

## Gotchas & Pitfalls

1. **Minimum 2 States Required**: `InterpolatedTrajectory` needs >= 2 points
   - Single-point trajectory raises ValueError
   - Need start + end minimum for interpolation

2. **Time Range Assertions**: Interpolation only within `[start_time, end_time]`
   - Querying outside range raises AssertionError
   - Always check `is_in_range()` before querying!
   - No extrapolation - design choice for safety

3. **Heading Interpolation**: Uses custom angular interpolator
   - Handles ±π wraparound correctly
   - Don't interpolate heading manually - use trajectory's interpolation!
   - Example: heading from -170° to +170° interpolates through ±180°, not through 0°

4. **State Type Consistency**: All states must be same class
   - Can't mix EgoState and Waypoint in same trajectory
   - Trajectory remembers first state's class type
   - Reconstruction uses original class's `from_split_state()`

5. **Fixed States**: Not interpolated, taken from first waypoint
   - Vehicle dimensions, parameters, etc.
   - Assumption: These don't change during trajectory
   - All waypoints should have same fixed states!

6. **Time Units**: TimePoint uses microseconds
   - Convert carefully: `TimePoint(int(seconds * 1e6))`
   - Floating point precision can cause off-by-one microsecond errors

7. **Scipy Interp1d Behavior**: Piecewise linear by default
   - Not smooth! No continuity in derivatives
   - For smooth trajectories, use higher-order planners or post-process
   - Controllers may need to handle discontinuities

8. **Pickling**: Custom `__reduce__()` for serialization
   - Stores original waypoints, not interpolation functions
   - Interpolators rebuilt on unpickling

## Related Documentation

- `nuplan/planning/simulation/planner/CLAUDE.md` - Planners generate trajectories
- `nuplan/planning/simulation/controller/CLAUDE.md` - Controllers track trajectories
- `nuplan/common/actor_state/CLAUDE.md` - InterpolatableState types (EgoState, Waypoint)
- `nuplan/common/geometry/CLAUDE.md` - AngularInterpolator implementation
- `nuplan/common/utils/CLAUDE.md` - InterpolatableState protocol, SplitState

---

**AIDEV-NOTE**: Trajectory interpolation is critical for controller tracking. Planners output sparse waypoints (e.g., every 0.1s), controllers query at arbitrary times (e.g., 0.05s lookahead).

**AIDEV-NOTE**: Angular interpolation handles heading wraparound - key for correct vehicle orientation interpolation.
