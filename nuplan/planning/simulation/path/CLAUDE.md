# nuplan/planning/simulation/path/

## 1. Purpose & Responsibility

This module provides **continuous path representations** for simulation, enabling smooth spatial queries, distance calculations, and baseline trajectory generation from discrete map lane sequences. The `InterpolatedPath` is THE core abstraction for representing agent routes, reference lanes, and planner baselines, supporting both 2D (SE2) and progress-based (1D longitudinal + SE2) state representations.

## 2. Key Abstractions

### Core Concepts

**InterpolatedPath**
- **Purpose**: Continuous path representation from discrete waypoints
- **Key features**:
  - Arc-length parameterization (query by distance along path)
  - Interpolation (get pose/curvature at arbitrary progress)
  - Trimming (extract sub-paths)
  - Length caching (efficient distance queries)
- **Use cases**:
  - IDM agent baselines (lane-following paths)
  - Reference trajectories for trajectory tracking
  - Map lane centerlines (continuous representation)

**Path Representations**
- **Discrete**: `List[StateSE2]` - Waypoints with gaps
- **Continuous**: `InterpolatedPath` - Smooth curve between waypoints
- **Progress-based**: Map arc-length → StateSE2 (position, heading)

**Path Operations**
- **Trimming**: Extract sub-path from start progress to end progress
- **Interpolation**: Get state at arbitrary arc-length
- **Concatenation**: Join multiple paths end-to-end
- **Distance queries**: Arc-length between two states on path

### Key Classes

```python
class InterpolatedPath:
    """
    Continuous path representation with arc-length parameterization.
    Supports SE2 and progress-based state queries.
    """
    def __init__(self, waypoints: List[StateSE2]):
        self._waypoints = waypoints
        self._arc_lengths = self._compute_arc_lengths()  # Cumulative distance
        self._length = self._arc_lengths[-1]
    
    @property
    def length(self) -> float:
        """Total arc-length of path (meters)"""
    
    def get_state_at_progress(self, progress: float) -> StateSE2:
        """Interpolate pose at given arc-length (meters)"""
    
    def trim_by_progress(self, start: float, end: float) -> InterpolatedPath:
        """Extract sub-path between arc-lengths"""
    
    def get_closest_arc_length_from_state(self, state: StateSE2) -> float:
        """Project state onto path, return arc-length"""
```

## 3. Architecture & Design Patterns

### Design Patterns

**Template Method Pattern**
- Base class defines path operations (trim, interpolate)
- Subclasses override interpolation strategy (linear, spline, etc.)

**Flyweight Pattern**
- Paths cache arc-length arrays to avoid recomputation
- Shared waypoint data across multiple path views

**Strategy Pattern**
- Interpolation strategy (linear, cubic spline, Catmull-Rom)
- Distance metric (Euclidean, geodesic)

### Relationships

```
Map Lane Sequence (discrete)
    ↓
InterpolatedPath (continuous)
    ├─ Used by: IDM agents (baseline following)
    ├─ Used by: Planners (reference trajectory)
    └─ Used by: Path tracking controllers
```

### Path Construction Flow

```
Lane Graph Query
    ↓
List[LaneGraphEdgeMapObject]  # Route segments
    ↓
Extract Baseline (discrete waypoints)
    ↓
InterpolatedPath(waypoints)
    ├─ Compute arc-lengths (cumulative distance)
    ├─ Cache total length
    └─ Enable continuous queries
```

## 4. Dependencies

### Internal (nuPlan)

**Direct Dependencies**:
- ✅ `nuplan.common.actor_state.state_representation` - StateSE2, ProgressStateSE2
- ✅ `nuplan.common.geometry.compute` - Distance, heading calculations
- ✅ `nuplan.common.maps.abstract_map_objects` - LaneGraphEdgeMapObject (for baseline extraction)

**Indirect Dependencies**:
- ✅ `nuplan.common.geometry.transform` - Coordinate transforms

### External

- `numpy` - Array operations, interpolation
- `scipy` - Spline interpolation (if used)
- `typing` - List, Optional type hints

### Dependency Notes

**AIDEV-NOTE**: InterpolatedPath is a "pure" geometry module - no simulation or planning logic. Can be used standalone for path operations.

## 5. Dependents (Who Uses This Module?)

### Direct Consumers

**IDM Agents**:
- ✅ `nuplan/planning/simulation/observation/idm/` - IDM agent baselines
  - Each IDM agent has `InterpolatedPath` from route segments
  - Agents propagate along path using progress state

**Planners**:
- Reference trajectory generation (convert map lane to continuous path)
- Trajectory tracking controllers (compute cross-track error)

**Path Tracking Controllers**:
- Compute lateral/longitudinal errors relative to reference path
- Project ego state onto path for progress tracking

**Map Utilities**:
- Convert lane graph segments to continuous baselines
- Extract centerlines for visualization

### Use Cases

1. **IDM Agent Baseline Following**
   - Route: `[lane1, lane2, connector, lane3]` → InterpolatedPath
   - Agent state: `ProgressStateSE2(progress=50.0, lateral_offset=0.0)`
   - Query path at progress to get global pose

2. **Reference Trajectory Generation**
   - Planner gets route from mission planner
   - Convert route to InterpolatedPath baseline
   - Sample path at fixed intervals (e.g., every 0.5m)

3. **Cross-Track Error Computation**
   - Project ego pose onto reference path → closest progress
   - Compute lateral offset (perpendicular distance to path)
   - Use for trajectory tracking PID controller

4. **Path Distance Queries**
   - Compute remaining distance to goal
   - Estimate time-to-goal (distance / velocity)
   - Progress metrics for simulation

**AIDEV-NOTE**: InterpolatedPath is the bridge between discrete map data and continuous simulation state.

## 6. Critical Files (Prioritized)

### Priority 1: Core Implementation

1. **`interpolated_path.py`**
   - InterpolatedPath class
   - Arc-length computation
   - Interpolation methods
   - **Key for**: Understanding path representation

2. **`utils.py`** (if exists)
   - Path trimming utilities
   - Path concatenation
   - Distance calculations
   - **Key for**: Common path operations

### Priority 2: Specialized Paths

3. **`progress_path.py`** (if exists)
   - Progress-based path representation
   - ProgressStateSE2 support
   - **Key for**: IDM agent state representation

### Priority 3: Tests and Utilities

4. **`test/` directory**
   - Interpolation accuracy tests
   - Trimming behavior tests
   - Edge case validation
   - **Key for**: Expected behavior

5. **`__init__.py`**
   - Module exports
   - Public API surface

**AIDEV-NOTE**: Start with interpolated_path.py - it's the core abstraction (~200-400 lines).

## 7. Common Usage Patterns

### Creating Path from Waypoints

```python
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.simulation.path.interpolated_path import InterpolatedPath

# Create discrete waypoints
waypoints = [
    StateSE2(x=0.0, y=0.0, heading=0.0),
    StateSE2(x=10.0, y=0.0, heading=0.0),
    StateSE2(x=20.0, y=5.0, heading=0.1),
    StateSE2(x=30.0, y=10.0, heading=0.2),
]

# Create continuous path
path = InterpolatedPath(waypoints)

# Query path properties
print(f"Path length: {path.length:.2f} m")  # ~31.6 meters
```

### Interpolating State at Progress

```python
# Get state at 15.0 meters along path
state_at_15m = path.get_state_at_progress(progress=15.0)

print(f"Position: ({state_at_15m.x:.2f}, {state_at_15m.y:.2f})")
print(f"Heading: {state_at_15m.heading:.3f} rad")

# Query at path endpoints
start_state = path.get_state_at_progress(0.0)
end_state = path.get_state_at_progress(path.length)
```

### Projecting State onto Path

```python
# Ego vehicle at arbitrary position
ego_state = StateSE2(x=12.0, y=1.0, heading=0.05)

# Project onto path (find closest arc-length)
closest_progress = path.get_closest_arc_length_from_state(ego_state)

print(f"Ego is at progress {closest_progress:.2f} m along path")

# Get reference state on path
reference_state = path.get_state_at_progress(closest_progress)

# Compute cross-track error (lateral offset)
cross_track_error = compute_lateral_distance(ego_state, reference_state)
```

### Trimming Path

```python
# Extract sub-path from 10m to 25m
sub_path = path.trim_by_progress(start=10.0, end=25.0)

print(f"Sub-path length: {sub_path.length:.2f} m")  # ~15 meters

# Trim from current progress to end (remaining path)
ego_progress = 12.0
remaining_path = path.trim_by_progress(start=ego_progress, end=path.length)
```

### IDM Agent Baseline

```python
from nuplan.planning.simulation.observation.idm.idm_agent import IDMAgent

# Build path from route segments
route_segments = [lane1, lane2, connector, lane3]
waypoints = []
for segment in route_segments:
    waypoints.extend(segment.baseline_path().discrete_path())

agent_path = InterpolatedPath(waypoints)

# Create IDM agent with path
agent = IDMAgent(
    initial_state=ProgressStateSE2(progress=0.0, lateral_offset=0.0, ...),
    path=agent_path,
    ...
)

# Agent propagates along path
agent.propagate(lead_agent, tspan)

# Get updated global pose from progress
global_pose = agent_path.get_state_at_progress(agent.state.progress)
```

### Reference Trajectory Sampling

```python
# Sample path at fixed intervals (0.5m spacing)
sample_interval = 0.5  # meters
num_samples = int(path.length / sample_interval)

reference_trajectory = []
for i in range(num_samples):
    progress = i * sample_interval
    state = path.get_state_at_progress(progress)
    reference_trajectory.append(state)

# Use for trajectory tracking controller
```

### Path Concatenation

```python
# Join multiple path segments
path1 = InterpolatedPath(waypoints1)
path2 = InterpolatedPath(waypoints2)

# Concatenate (assuming path1 end connects to path2 start)
combined_waypoints = waypoints1 + waypoints2[1:]  # Remove duplicate endpoint
combined_path = InterpolatedPath(combined_waypoints)
```

## 8. Gotchas & Edge Cases

### 1. **Progress Out of Bounds**
- **Issue**: Querying `get_state_at_progress(progress)` with progress > path.length
- **Symptom**: IndexError or extrapolation beyond path end
- **Fix**: Clamp progress: `min(max(progress, 0.0), path.length)`

### 2. **Empty or Single-Waypoint Path**
- **Issue**: Path created with 0 or 1 waypoints
- **Symptom**: Division by zero in arc-length computation
- **Fix**: Validate `len(waypoints) >= 2` before creating path

### 3. **Duplicate Waypoints**
- **Issue**: Two consecutive waypoints at same position
- **Symptom**: Zero arc-length segment, interpolation issues
- **Fix**: Filter duplicate waypoints during construction

### 4. **Heading Discontinuities**
- **Issue**: Large heading jumps between waypoints (e.g., -π to +π wrap)
- **Symptom**: Interpolated headings jump unrealistically
- **Fix**: Use `principal_value()` to unwrap angles before interpolation

### 5. **Precision Loss in Long Paths**
- **Issue**: Cumulative arc-length errors in very long paths (>1km)
- **Symptom**: Interpolated positions drift from waypoints
- **Fix**: Use higher precision (float64) for arc-length arrays

### 6. **Trimming Beyond Path Bounds**
- **Issue**: `trim_by_progress(start=-10, end=1000)` with path length 50m
- **Symptom**: Returns invalid path or crashes
- **Fix**: Clamp trim bounds to [0, path.length]

### 7. **Projection Ambiguity**
- **Issue**: State equidistant from multiple path segments
- **Symptom**: `get_closest_arc_length_from_state()` returns arbitrary segment
- **Fix**: Use heading similarity as tiebreaker

### 8. **Arc-Length Caching Invalidation**
- **Issue**: Modifying waypoints after path creation
- **Symptom**: Arc-lengths out of sync with waypoints
- **Fix**: Make path immutable (don't expose waypoint setter)

### 9. **Interpolation Method Mismatch**
- **Issue**: Linear interpolation assumes straight segments, but path has curves
- **Symptom**: Interpolated states cut corners
- **Fix**: Use spline interpolation for curved paths

### 10. **State Representation Confusion**
- **Issue**: Mixing StateSE2 (global) with ProgressStateSE2 (path-relative)
- **Symptom**: Type errors or incorrect transformations
- **Fix**: Clearly document which methods expect global vs progress states

## 9. Performance Considerations

**Computational Cost**:
- Path construction (arc-length computation): O(N) where N = num waypoints
- `get_state_at_progress()`: O(log N) binary search + O(1) interpolation
- `get_closest_arc_length_from_state()`: O(N) distance computation (can optimize with spatial index)
- Trimming: O(N) waypoint copying

**Memory Usage**:
- Path object: ~N × 40 bytes (waypoints) + N × 8 bytes (arc-lengths)
- Example: 100 waypoints → ~5 KB

**Optimization Strategies**:
- Cache closest segment index for repeated projections (e.g., trajectory tracking)
- Use spatial index (R-tree) for fast projection on very long paths (>1000 waypoints)
- Downsample waypoints if path is over-detailed (e.g., 1cm spacing → 10cm spacing)

**AIDEV-NOTE**: For typical paths (<100 waypoints), performance is not a concern. IDM agents with 1000+ waypoint paths may benefit from downsampling.

## 10. Related Documentation

### Cross-References
- ✅ `nuplan/planning/simulation/observation/idm/CLAUDE.md` - IDM agent baselines (primary use case)
- ✅ `nuplan/common/actor_state/CLAUDE.md` - StateSE2, ProgressStateSE2 representations
- ✅ `nuplan/common/geometry/CLAUDE.md` - Distance, heading computations
- ✅ `nuplan/common/maps/CLAUDE.md` - Lane graph, baseline extraction
- ✅ `nuplan/planning/simulation/trajectory/CLAUDE.md` - Trajectory representations (similar concepts)
- `nuplan/planning/simulation/controller/CLAUDE.md` - Path tracking controllers (Phase 2B)

### External Resources
- **Arc-length parameterization**: https://en.wikipedia.org/wiki/Arc_length
- **Spline interpolation**: https://docs.scipy.org/doc/scipy/reference/interpolate.html

## 11. AIDEV Notes

**Design Philosophy**:
- Paths are immutable (waypoints fixed after construction)
- Arc-length is THE canonical parameterization (not time or index)
- Progress state (1D + SE2) is more efficient than tracking 2D position for lane-following

**Common Mistakes**:
- Querying progress beyond path.length (always clamp!)
- Assuming linear interpolation is accurate for curved paths (use splines for curves)
- Not handling duplicate waypoints during construction

**Future Improvements**:
- **AIDEV-TODO**: Add curvature queries (`get_curvature_at_progress()`)
- **AIDEV-TODO**: Support different interpolation strategies (linear, cubic, Catmull-Rom)
- **AIDEV-TODO**: Add spatial index for O(log N) projection on long paths
- **AIDEV-TODO**: Add path validation (check for self-intersections, loops)

**AIDEV-NOTE**: If you need path curvature (for jerk-minimizing planners), compute numerically:
```python
ds = 0.1  # Small arc-length step
state1 = path.get_state_at_progress(progress)
state2 = path.get_state_at_progress(progress + ds)
curvature = (state2.heading - state1.heading) / ds
```
