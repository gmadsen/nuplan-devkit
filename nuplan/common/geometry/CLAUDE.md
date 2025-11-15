# CLAUDE.md - nuplan/common/geometry

## Purpose & Responsibility

**Foundational geometric operations for autonomous vehicle planning.** This module provides low-level utilities for coordinate transformations, distance calculations, angular arithmetic, and interpolation in SE(2) space. These functions are used pervasively throughout nuPlan for converting between reference frames, computing spatial relationships between objects, and handling trajectory mathematics. Critical performance-sensitive code - many functions are called in tight loops during simulation.

## Key Functions & Utilities

### Coordinate Frame Transformations (`transform.py`)

**2D Euclidean Transformations:**
- **`translate(pose, translation)`** - Apply 2D translation vector to StateSE2
- **`rotate(pose, rotation_matrix)`** - Apply 2D rotation matrix to StateSE2
- **`rotate_angle(pose, theta)`** - Rotate StateSE2 by angle (convenience wrapper)
- **`transform(pose, transform_matrix)`** - Apply SE(2) or SE(3) transformation matrix

**Vehicle-Centric Transformations (Critical!):**
- **`translate_longitudinally(pose, distance)`** - Move along heading direction
  - Positive distance = forward, negative = backward
  - Uses `distance * [cos(heading), sin(heading)]`
- **`translate_laterally(pose, distance)`** - Move perpendicular to heading
  - Positive distance = left, negative = right
  - Uses `distance * [cos(heading + π/2), sin(heading + π/2)]`
- **`translate_longitudinally_and_laterally(pose, lon, lat)`** - Combined translation
  - More efficient than separate calls (single trig evaluation)

**Point Transformations:**
- **`rotate_2d(point, rotation_matrix)`** - Rotate Point2D with 2x2 matrix

### Conversion Between Representations (`convert.py`)

**Transformation Matrix ↔ StateSE2:**
- **`matrix_from_pose(pose)`** - StateSE2 → 3x3 homogeneous transform matrix
  - Returns: `[[cos(θ), -sin(θ), x], [sin(θ), cos(θ), y], [0, 0, 1]]`
- **`pose_from_matrix(transform_matrix)`** - 3x3 matrix → StateSE2
  - Extracts: x = M[0,2], y = M[1,2], θ = atan2(M[1,0], M[0,0])

**Reference Frame Conversion (Batch Operations):**
- **`absolute_to_relative_poses(absolute_poses)`** - Convert list to relative coordinates
  - First pose becomes origin [0, 0, 0]
  - Uses inverse of first transform: `inv(T[0]) @ T[i]`
- **`relative_to_absolute_poses(origin_pose, relative_poses)`** - Convert back to global
  - Applies origin transform: `T_origin @ T_rel[i]`
- **`numpy_array_to_absolute_pose(origin, poses)`** - Nx3 array → List[StateSE2] (absolute)
- **`numpy_array_to_absolute_velocity(origin, velocities)`** - Nx2 array → List[StateVector2D] (absolute)

**Polar ↔ Cartesian:**
- **`vector_2d_from_magnitude_angle(magnitude, angle)`** - (r, θ) → StateVector2D(x, y)

### Distance Computations (`compute.py`)

**Point-to-Pose Distances (Projected):**
- **`longitudinal_distance(reference, other)`** - Distance along heading direction
  - Formula: `cos(θ)·Δx + sin(θ)·Δy` (dot product with heading vector)
  - Returns signed distance (positive = ahead, negative = behind)
- **`lateral_distance(reference, other)`** - Distance perpendicular to heading
  - Formula: `-sin(θ)·Δx + cos(θ)·Δy` (dot product with lateral vector)
  - Returns signed distance (positive = left, negative = right)

**Ego-to-Polygon Distances (Vehicle-Aware):**
- **`signed_lateral_distance(ego_state, polygon)`** - Lateral clearance to polygon
  - Accounts for ego vehicle width (Pacifica parameters)
  - Returns: positive = polygon on left, negative = polygon on right
  - Zero = just touching
  - **AIDEV-NOTE**: Uses `get_pacifica_parameters()` - assumes ego is always Pacifica!
- **`signed_longitudinal_distance(ego_state, polygon)`** - Longitudinal clearance
  - Accounts for ego vehicle length
  - Returns: positive = polygon ahead, negative = polygon behind

**Euclidean Distances:**
- **`compute_distance(lhs, rhs)`** - Simple Euclidean distance between StateSE2 (ignores heading)
  - Formula: `hypot(Δx, Δy)`

**Oriented Box Distances (Advanced):**
- **`l2_euclidean_corners_distance(box1, box2)`** - L2 norm of corner-to-corner distances
  - Computes Euclidean distance for each corner pair
  - Returns norm of distance vector: `||[d1, d2, d3, d4]||`
  - Sensitive to rotation differences
- **`se2_box_distances(query, targets, box_size, consider_flipped=True)`** - Batch box distances
  - Spawns boxes at each pose with given dimensions
  - Optionally checks query rotated 180° (for symmetric vehicles)
  - Returns list of distances to each target
  - **Use case**: Matching predicted poses to ground truth (handles 180° ambiguity)

**Lateral Displacement:**
- **`compute_lateral_displacements(poses)`** - Extract Δy from pose sequence
  - Returns: `[poses[i].y - poses[i-1].y for i in 1..N]`
  - **AIDEV-NOTE**: This is global frame Δy, NOT lateral in vehicle frame!

### Angular Arithmetic (`compute.py`)

- **`principal_value(angle, min_=-π)`** - Wrap angle to [min_, min_ + 2π)
  - Formula: `(angle - min_) % 2π + min_`
  - Default: wrap to [-π, π)
  - **Asserts angle is finite** - catches NaN/inf bugs early!
  - Vectorized: accepts floats or numpy arrays

### Interpolation (`interpolate_state.py`, `interpolate_tracked_object.py`)

**State Interpolation (Generic):**
- **`interpolate_future_waypoints(waypoints, horizon_len_s, interval_s)`** - Interpolate forward in time
  - Input: List of InterpolatableState (must be monotonically increasing timestamps)
  - Output: Uniformly sampled states at `interval_s` spacing for `horizon_len_s` duration
  - Pads with `None` if waypoints don't cover full horizon
  - Edge case: Single waypoint → [waypoint, None, None, ...]
- **`interpolate_past_waypoints(waypoints, horizon_len_s, interval_s)`** - Interpolate backward in time
  - Similar to future, but samples backward from last waypoint
  - Last state guaranteed to exist (not None)

**Tracked Object Interpolation:**
- **`interpolate_agent(agent, horizon_len_s, interval_s)`** - Interpolate AgentTemporalState
  - Interpolates both predictions (future) and past_trajectory
  - Preserves prediction probabilities
  - Returns AgentTemporalState with uniformly sampled trajectories
- **`interpolate_tracks(tracked_objects, horizon_len_s, interval_s)`** - Batch interpolation
  - Applies to all agents in TrackedObjects collection
  - Static objects pass through unmodified

**Internal Helpers:**
- **`_validate_waypoints(waypoints)`** - Check non-empty and monotonically increasing
- **`_compute_desired_time_steps(start, end, horizon, interval)`** - Generate sample times
- **`_interpolate_waypoints(waypoints, target_timestamps)`** - Core linear interpolation
  - Uses `InterpolatedTrajectory.get_state_at_time()`

**Angular Interpolation (Special Case):**
- **`AngularInterpolator`** - Handles angle wrapping during interpolation
  - Constructor: `AngularInterpolator(states, angular_states)`
  - Uses `np.unwrap()` to remove discontinuities before interpolation
  - `interpolate(sampled_state)` - Returns interpolated angle wrapped to [-π, π)
  - **Use case**: Interpolating headings without sudden jumps at ±π boundary

### PyTorch Geometry (`torch_geometry.py`)

**Tensor Validation:**
- **`_validate_state_se2_tensor_shape(tensor, expected_first_dim)`** - Check (3,) or (N, 3) shape
- **`_validate_state_se2_tensor_batch_shape(tensor)`** - Check (N, 3) shape
- **`_validate_transform_matrix_shape(tensor)`** - Check (3, 3) shape
- **`_validate_transform_matrix_batch_shape(tensor)`** - Check (N, 3, 3) shape

**StateSE2 Tensor ↔ Transform Matrix:**
- **`state_se2_tensor_to_transform_matrix(input_data, precision)`** - [x, y, h] → 3x3 matrix
  - Input: shape (3,) torch.Tensor
  - Output: 3x3 homogeneous transform
- **`state_se2_tensor_to_transform_matrix_batch(input_data, precision)`** - (N, 3) → (N, 3, 3)
  - Vectorized version (much faster than loop!)
  - Clever matrix multiply: `[x, y, cos(h), sin(h), 1] @ reshaping_tensor → (N, 9) → (N, 3, 3)`
- **`transform_matrix_to_state_se2_tensor(input_data, precision)`** - 3x3 → [x, y, h]
- **`transform_matrix_to_state_se2_tensor_batch(input_data)`** - (N, 3, 3) → (N, 3)

**Reference Frame Transformation (Torch):**
- **`global_state_se2_tensor_to_local(global_states, local_state, precision)`**
  - Transforms (N, 3) global poses to local frame defined by (3,) anchor
  - Formula: `inv(T_local) @ T_global[i]`
  - **Critical for ML models**: converts global map/agent poses to ego-centric
- **`coordinates_to_local_frame(coords, anchor_state, precision)`**
  - Transforms (N, 2) coordinates (no heading) to local frame
  - Uses homogeneous coordinates: `[x, y, 1]` → apply transform → `[x', y', 1]`
  - **Edge case**: Empty input (N=0) returns immediately (torch.nn.functional.pad crashes!)
- **`vector_set_coordinates_to_local_frame(coords, avails, anchor_state, output_precision)`**
  - Transforms (num_elements, num_points, 2) map element coordinates
  - Uses availability mask to zero out padding
  - **Always uses float64 internally** for numerical stability (outputs specified precision)
  - **Use case**: Transforming vector map representations for ML features

## Architecture & Design Patterns

1. **Functional API**: Pure functions, no stateful objects (except AngularInterpolator)
   - Easy to test, reason about, parallelize
   - Compose operations naturally

2. **NumPy + SciPy Foundation**: Leverages mature numerical libraries
   - `np.hypot`, `np.arctan2`, `scipy.interpolate.interp1d`
   - Vectorization for performance

3. **Shapely Integration**: Seamless interop with Shapely geometry
   - Polygon-based distance calculations
   - Collision detection support (via OrientedBox.geometry)

4. **PyTorch Parallel**: Torch equivalents for ML pipelines
   - Batch operations critical for training (1000s of states)
   - GPU-compatible (all torch tensors, no numpy)
   - Precision control for memory/accuracy tradeoffs

5. **Type Safety**: Extensive use of numpy.typing and type hints
   - `npt.NDArray[np.float64]` for arrays
   - `Union[float, npt.NDArray]` for scalar-or-array functions

6. **Validation-First**: Assertions catch shape mismatches early
   - Better error messages than cryptic numpy broadcast errors
   - `principal_value` asserts finiteness (catches NaN propagation)

7. **Coordinate System Conventions (CRITICAL!):**
   - **Heading**: 0 = East, counter-clockwise (mathematical convention)
   - **Longitudinal**: Along heading direction (vehicle forward)
   - **Lateral**: Perpendicular to heading (vehicle left = positive)
   - **Units**: meters (m), radians (rad), seconds (s), microseconds (us for timestamps)

## Dependencies / Dependents

### Dependencies (What We Import)
- **Standard Library**: `typing`, `math`
- **NumPy/SciPy**: `numpy`, `numpy.typing`, `scipy.interpolate.interp1d`
- **Shapely**: `shapely.geometry.Polygon` (for polygon distances)
- **PyTorch**: `torch` (torch_geometry.py only)
- **nuPlan Internal**:
  - `nuplan.common.actor_state.state_representation` - StateSE2, Point2D, StateVector2D, TimePoint
  - `nuplan.common.actor_state.oriented_box` - OrientedBox, Dimension
  - `nuplan.common.actor_state.vehicle_parameters` - `get_pacifica_parameters()`
  - `nuplan.common.actor_state.tracked_objects` - TrackedObject, TrackedObjects
  - `nuplan.common.actor_state.agent_temporal_state` - AgentTemporalState
  - `nuplan.common.utils.interpolatable_state` - InterpolatableState protocol
  - `nuplan.planning.simulation.trajectory.interpolated_trajectory` - InterpolatedTrajectory
  - `nuplan.planning.simulation.trajectory.predicted_trajectory` - PredictedTrajectory

### Dependents (Who Imports Us)
**Used everywhere in nuPlan - TIER 0 dependency!**
- **`nuplan/common/actor_state/`** - OrientedBox, CarFootprint, transform utilities
- **`nuplan/planning/simulation/planner/`** - All planners use transforms, distance checks
- **`nuplan/planning/simulation/observation/`** - Coordinate transformations for sensors
- **`nuplan/planning/training/preprocessing/`** - ML feature extraction (torch_geometry heavily used!)
- **`nuplan/planning/metrics/`** - Distance-based metrics
- **`nuplan/planning/scenario_builder/`** - Coordinate conversions
- **`nuplan/planning/utils/`** - Serialization, visualization

## Critical Files (Prioritized)

1. **`transform.py`** (105 lines) - **MUST READ FIRST!**
   - Core SE(2) transformations
   - `translate_longitudinally` and `translate_laterally` used everywhere
   - Vehicle-centric coordinate operations

2. **`convert.py`** (105 lines) - **Reference frame conversions**
   - Absolute ↔ relative pose conversions
   - Matrix ↔ pose conversions
   - Batch operations for efficiency

3. **`compute.py`** (170 lines) - **Distance calculations**
   - Longitudinal/lateral distance projections
   - Ego-to-polygon clearances
   - Angular arithmetic (principal_value)

4. **`torch_geometry.py`** (299 lines) - **ML pipeline support**
   - Batch tensor operations
   - GPU-compatible transformations
   - Critical for training performance

5. **`interpolate_state.py`** (120 lines) - **Temporal interpolation**
   - Future/past waypoint interpolation
   - Trajectory sampling
   - Handles missing data (None padding)

6. **`interpolate_tracked_object.py`** (57 lines) - **Agent interpolation**
   - Tracked object temporal resampling
   - Prediction trajectory interpolation

7. **`__init__.py`** (1 line) - Empty (no public API surface)

## Common Usage Patterns

### Transform Ego Pose to Local Frame
```python
from nuplan.common.geometry.transform import translate_longitudinally, translate_laterally

# Move ego forward 10m
future_pose = translate_longitudinally(ego_state.center, distance=10.0)

# Shift left 2m (e.g., lane change)
left_lane_pose = translate_laterally(ego_state.center, distance=2.0)

# Get position 5m ahead, 1m right
lane_center = translate_longitudinally_and_laterally(
    ego_state.center, lon=5.0, lat=-1.0  # negative = right
)
```

### Convert Between Reference Frames
```python
from nuplan.common.geometry.convert import (
    absolute_to_relative_poses,
    relative_to_absolute_poses
)

# Convert trajectory to ego-centric
absolute_trajectory = [state1, state2, state3, ...]
relative_trajectory = absolute_to_relative_poses(absolute_trajectory)
# relative_trajectory[0] is now StateSE2(0, 0, 0)

# Convert back to global
reconstructed = relative_to_absolute_poses(
    origin_pose=absolute_trajectory[0],
    relative_poses=relative_trajectory
)
```

### Compute Spatial Relationships
```python
from nuplan.common.geometry.compute import (
    longitudinal_distance,
    lateral_distance,
    compute_distance
)

# Check if agent is ahead of ego
lon_dist = longitudinal_distance(ego_state.center, agent.center)
if lon_dist > 0:
    print(f"Agent is {lon_dist:.1f}m ahead")

# Check lane offset
lat_dist = lateral_distance(lane_center_pose, ego_state.center)
print(f"Ego is {abs(lat_dist):.2f}m {'left' if lat_dist > 0 else 'right'} of lane center")

# Simple distance (ignores heading)
dist = compute_distance(ego_state.center, target_state.center)
```

### Check Clearances (Collision Avoidance)
```python
from nuplan.common.geometry.compute import signed_lateral_distance

# Get lateral clearance to obstacle
clearance = signed_lateral_distance(ego_state, obstacle_polygon)

if abs(clearance) < 0.5:  # Less than 50cm clearance
    print("WARNING: Tight squeeze!")
# Positive = obstacle on left, negative = obstacle on right
```

### Angular Wrapping
```python
from nuplan.common.geometry.compute import principal_value
import numpy as np

# Wrap angle to [-π, π)
heading = principal_value(7.5)  # Returns ~1.78 (7.5 - 2π)

# Wrap to [0, 2π)
heading_positive = principal_value(angle, min_=0.0)

# Vectorized (critical for performance!)
headings = np.array([0.5, 3.5, -4.0, 10.0])
wrapped = principal_value(headings)  # All in [-π, π)
```

### Interpolate Trajectories
```python
from nuplan.common.geometry.interpolate_state import interpolate_future_waypoints

# Sample trajectory at 0.5s intervals for 8 seconds
waypoints = [ego_state1, ego_state2, ...]  # List[EgoState]
sampled = interpolate_future_waypoints(
    waypoints=waypoints,
    horizon_len_s=8.0,
    interval_s=0.5
)
# Returns 17 states: [0.0s, 0.5s, 1.0s, ..., 8.0s]
# Pads with None if waypoints don't cover full horizon
```

### PyTorch ML Feature Extraction
```python
from nuplan.common.geometry.torch_geometry import (
    global_state_se2_tensor_to_local,
    coordinates_to_local_frame
)
import torch

# Convert global agent poses to ego-centric frame
global_agents = torch.tensor([[10.0, 5.0, 0.5], [15.0, 2.0, 0.3]])  # (N, 3)
ego_pose = torch.tensor([0.0, 0.0, 0.0])  # (3,)
local_agents = global_state_se2_tensor_to_local(global_agents, ego_pose)
# Now in ego frame: local_agents[0] ≈ [10.0, 5.0, 0.5]

# Transform map polylines (no heading)
map_coords = torch.randn(20, 50, 2)  # (num_lanes, points_per_lane, 2)
avails = torch.ones(20, 50, dtype=torch.bool)
local_map = coordinates_to_local_frame(
    coords=map_coords.reshape(-1, 2),  # Flatten to (1000, 2)
    anchor_state=ego_pose,
    precision=torch.float32
).reshape(20, 50, 2)
```

### Batch Transform Matrix Conversion (Fast!)
```python
from nuplan.common.geometry.torch_geometry import (
    state_se2_tensor_to_transform_matrix_batch,
    transform_matrix_to_state_se2_tensor_batch
)

# Convert batch of poses to matrices (for composition)
poses = torch.randn(100, 3)  # (N, 3)
matrices = state_se2_tensor_to_transform_matrix_batch(poses)  # (N, 3, 3)

# Apply transformation via matrix multiply
transformed = torch.matmul(reference_matrix, matrices)

# Convert back to poses
result_poses = transform_matrix_to_state_se2_tensor_batch(transformed)
```

## Gotchas & Pitfalls

1. **Heading Convention (Critical!)**: 0 radians = **East**, counter-clockwise
   - NOT North! This is mathematical convention, not geographic
   - Longitudinal direction: `[cos(θ), sin(θ)]`
   - Lateral direction: `[cos(θ + π/2), sin(θ + π/2)] = [-sin(θ), cos(θ)]`
   - **When visualizing**: Rotate mental model 90° from GPS convention

2. **Lateral Sign Convention**: Positive = **left** of vehicle
   - `translate_laterally(pose, +2.0)` moves LEFT
   - Intuitive for driver ("pull left 2 meters")
   - Opposite of some robotics conventions (right-hand rule)

3. **Longitudinal vs Global Displacement**:
   - `longitudinal_distance()` projects onto heading vector (vehicle forward/back)
   - `compute_lateral_displacements()` computes **global** Δy, NOT vehicle lateral!
   - **Don't confuse them!** One is ego-centric, one is world-frame

4. **Pacifica Hardcoding**: `signed_lateral_distance()` and `signed_longitudinal_distance()` assume ego is Pacifica
   - Calls `get_pacifica_parameters()` internally
   - **Will be wrong** if ego vehicle changes (different width/length)
   - **AIDEV-TODO**: Parameterize vehicle dimensions

5. **Principal Value Domain**: Default is [-π, π), but can specify custom min
   - `principal_value(angle)` → [-π, π)
   - `principal_value(angle, min_=0)` → [0, 2π)
   - **Be consistent** across codebase to avoid wrap-around bugs

6. **Angular Interpolation Discontinuities**: Use `AngularInterpolator` for headings!
   - `scipy.interpolate.interp1d` doesn't handle angle wrapping
   - Interpolating from -170° to +170° gives wrong result (goes through 0° instead of 180°)
   - `AngularInterpolator` uses `np.unwrap()` internally

7. **Torch Precision Control**: Many functions accept `precision` parameter
   - **Default**: inherits from input tensor dtype
   - **Mixed dtypes**: Must explicitly specify precision or raises ValueError
   - **Numerical stability**: Use float64 for transformations, cast to float32 after
   - Example: `vector_set_coordinates_to_local_frame` always uses float64 internally

8. **Batch Operations Are Fast**: Always prefer batch functions!
   - `state_se2_tensor_to_transform_matrix_batch` is **10-100x faster** than loop
   - Uses clever matrix algebra to eliminate for-loops
   - Critical for ML training (batch size = 32-256)

9. **Coordinate Transform Order**: Matrix multiplication is **not commutative**!
   - `T_world_to_ego @ T_world_to_object` = `T_ego_to_object` ✅
   - `T_world_to_object @ T_world_to_ego` = WRONG ❌
   - **Remember**: Right-multiply to apply transform, left-multiply inverse to change frame

10. **Interpolation Padding with None**: `interpolate_future_waypoints` returns `List[Optional[State]]`
    - **Must check for None** before using!
    - Common pattern: `[state for state in interpolated if state is not None]`
    - Single waypoint → all except first are None

11. **Empty Tensor Edge Case**: `coordinates_to_local_frame` short-circuits on (0, 2) input
    - `torch.nn.functional.pad` crashes on zero-length tensors
    - Returns empty tensor immediately
    - **Check shape before calling** if you expect empty input

12. **Homogeneous Coordinate Padding**: Transform functions use `[x, y, 1]` representation
    - Last column/row of transform matrix is translation
    - Don't manually construct matrices - use `matrix_from_pose()`!
    - Easy to get signs wrong (especially rotation part)

13. **Shapely Polygon Vertex Winding**: `signed_lateral_distance` extracts vertices from polygon
    - Assumes Shapely Polygon convention (exterior ring)
    - Holes (interior rings) are **ignored**!
    - Use `polygon.exterior.coords.xy` pattern

14. **Monotonic Timestamp Requirement**: Interpolation functions assert monotonicity
    - Will **crash** if timestamps are out of order or duplicated
    - Common bug: appending to history buffer in wrong order
    - Validate with `np.all(np.diff(timestamps) > 0)`

15. **Performance-Critical Paths**: This module is in **tight simulation loops**
    - `translate_longitudinally`, `lateral_distance` called **millions** of times
    - Avoid allocations in hot paths
    - Prefer in-place operations when possible
    - Profile before optimizing (use cProfile)

## Test Coverage Notes

Test directory: `nuplan/common/geometry/test/`
- **`test_compute.py`** (8650 bytes) - Comprehensive distance/angular tests
  - Tests principal_value wrapping at boundaries
  - Mocks Pacifica parameters for signed distance tests
  - Angular interpolator edge cases
- **`test_convert.py`** (6967 bytes) - Reference frame conversion tests
  - Absolute ↔ relative roundtrip validation
  - Matrix conversion accuracy
- **`test_torch_geometry.py`** (10086 bytes) - **Largest test suite**
  - Batch operation correctness
  - Precision handling
  - Shape validation
- **`test_transform.py`** (4215 bytes) - Transformation correctness
  - Longitudinal/lateral translation
  - Rotation composition
- **`test_interpolate_tracked_object.py`** (2753 bytes) - Agent interpolation

**Test patterns to learn from:**
- Use `np.allclose()` for float comparisons (not `==`)
- Mock `get_pacifica_parameters()` for unit tests
- Validate both shape and values
- Test edge cases (empty input, single element, wrapping at ±π)

## Related Documentation

### Parent Module
- `nuplan/common/CLAUDE.md` - Common utilities overview (when it exists)

### Critical Dependencies
- **`nuplan/common/actor_state/CLAUDE.md`** - StateSE2, Point2D, EgoState (data structures we transform)
  - **Read this first** to understand SE(2) representation
  - Explains coordinate frames (rear axle vs center)

### Dependents (Many!)
- `nuplan/planning/simulation/planner/CLAUDE.md` - Planners use transforms extensively
- `nuplan/planning/training/preprocessing/CLAUDE.md` - ML features use torch_geometry
- `nuplan/planning/metrics/CLAUDE.md` - Metrics use distance calculations
- `nuplan/common/maps/CLAUDE.md` - Map queries return coordinates (often transformed)

### Sibling Modules
- `nuplan/common/utils/interpolatable_state.py` - InterpolatableState protocol
- `nuplan/planning/simulation/trajectory/CLAUDE.md` - Trajectory representations (use interpolation)

---

**AIDEV-NOTE**: This module is foundational - every planner, metric, and ML model uses these functions. Master the coordinate conventions (heading, longitudinal, lateral) before writing planning code!

**AIDEV-NOTE**: Performance matters here! `translate_longitudinally` is called in tight loops. Any optimization should profile first.

**AIDEV-NOTE**: Torch geometry functions are critical for ML training speed. The batch operations use clever matrix algebra - see line 130-135 in torch_geometry.py for the reshaping trick.

**AIDEV-TODO**: Parameterize `signed_lateral_distance` and `signed_longitudinal_distance` to accept VehicleParameters instead of hardcoding Pacifica.

**AIDEV-QUESTION**: Why does `compute_lateral_displacements` return global Δy instead of vehicle lateral displacement? Seems inconsistent with naming convention.
