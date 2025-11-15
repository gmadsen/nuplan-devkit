# CLAUDE.md - nuplan/common/actor_state

## Purpose & Responsibility

State representations for all actors in the nuPlan autonomous vehicle environment. This module provides the core data structures for representing ego vehicle states, other agents (vehicles, pedestrians, bicyclists), static objects (cones, barriers), and their geometric/kinematic properties. These are the foundational types used throughout planning, simulation, and evaluation.

## Key Abstractions & Classes

### Time Representations
- **`TimePoint`** - Absolute timestamp in microseconds since epoch (dataclass)
- **`TimeDuration`** - Relative time delta with microsecond resolution
  - Constructors: `from_us()`, `from_ms()`, `from_s()`
  - Supports arithmetic operations (+, -, *, /, comparisons)

### Geometric State Representations
- **`Point2D`** - Simple 2D point (x, y) in meters
- **`StateSE2`** - SE(2) state [x, y, heading] - inherits from Point2D
  - `as_matrix()` → 3x3 transformation matrix
  - `as_matrix_3d()` → 4x4 transformation matrix (SE3 projection)
  - `distance_to(state)` → Euclidean distance
  - Heading in radians (0 = East, counter-clockwise)
- **`ProgressStateSE2`** - StateSE2 + progress along path (meters)
- **`TemporalStateSE2`** - StateSE2 + TimePoint (state at specific time)
- **`StateVector2D`** - 2D velocity/acceleration vector with magnitude

### Oriented Bounding Box
- **`OrientedBox`** - 3D axis-aligned box in vehicle frame
  - Properties: `center` (StateSE2), `length`, `width`, `height`
  - `corner(point_type)` → Extract specific corner (FL, FR, RL, RR, etc.)
  - `all_corners()` → List of 4 corners
  - `geometry` → Shapely Polygon (cached property for collision checks)
  - Utility functions: `in_collision()`, `collision_by_radius_check()`
- **`OrientedBoxPointType`** - Enum for box reference points (CENTER, FRONT_BUMPER, REAR_BUMPER, corners)

### Ego Vehicle State
- **`EgoState`** - Complete state of the ego vehicle (implements InterpolatableState)
  - Components:
    - `car_footprint` (CarFootprint) - geometric footprint
    - `dynamic_car_state` (DynamicCarState) - velocities, accelerations
    - `tire_steering_angle` (float) - steering wheel angle
    - `is_in_auto_mode` (bool) - autonomous vs manual
    - `time_point` (TimePoint) - timestamp
  - Reference frames:
    - `build_from_rear_axle()` - construct from rear axle coordinates
    - `build_from_center()` - construct from center of mass coordinates
  - Key properties:
    - `center` → center of mass pose
    - `rear_axle` → rear axle pose
    - `waypoint` → convert to Waypoint representation
    - `agent` → convert to AgentState representation
  - Serialization: `serialize()`, `deserialize()`, `to_split_state()`, `from_split_state()`
- **`EgoStateDot`** - Derivative of EgoState (for dynamics equations, mostly semantic)

### Agent (Other Vehicles/Pedestrians) State
- **`AgentState`** - Single timestep state of a tracked object
  - `tracked_object_type` (TrackedObjectType) - VEHICLE, PEDESTRIAN, BICYCLE, etc.
  - `oriented_box` (OrientedBox) - geometric extent
  - `velocity` (StateVector2D) - velocity vector
  - `metadata` (SceneObjectMetadata) - token, track_id, timestamp
  - `angular_velocity` (optional float)
- **`AgentTemporalState`** - AgentState with temporal history/prediction
  - `predictions` - List of PredictedTrajectory (future motion)
  - `past_trajectory` - PredictedTrajectory (historical motion)
- **`Agent`** - Combines AgentState + AgentTemporalState (full representation with past/future)

### Static Objects
- **`StaticObject`** - Non-moving objects (cones, barriers, construction zones)
  - Similar to AgentState but for static elements
  - `tracked_object_type` from STATIC_OBJECT_TYPES

### Collections & Utilities
- **`TrackedObjects`** - Collection of all agents/static objects in a scene
  - Internally sorted by `TrackedObjectType` for efficient querying
  - `get_tracked_objects_of_type(type)` → filtered list
  - `get_agents()` → all dynamic agents (vehicles, peds, bikes)
  - `get_static_objects()` → all static objects
  - Uses cached `_ranges_per_type` for O(1) type filtering
- **`TrackedObjectType`** - Enum defining object categories (VEHICLE, PEDESTRIAN, BICYCLE, BARRIER, CONES, etc.)
  - `AGENT_TYPES` - subset of dynamic objects
  - `STATIC_OBJECT_TYPES` - subset of static objects

### Waypoint
- **`Waypoint`** - Single point along a trajectory (implements InterpolatableState)
  - `time_point` (TimePoint) - when to reach this point
  - `oriented_box` (OrientedBox) - vehicle pose and dimensions
  - `velocity` (Optional[StateVector2D]) - velocity at waypoint
  - Used to construct trajectories from sequence of waypoints

### Vehicle Parameters
- **`VehicleParameters`** - Physical parameters of ego vehicle (not shown in files read, but referenced)
  - Dimensions, axle distances, etc.

## Architecture & Design Patterns

1. **Dataclass-heavy**: Leverage Python dataclasses for lightweight state containers
2. **SE(2) Group Theory**: StateSE2 represents rigid body transformations in 2D
3. **Reference Frame Conversions**: EgoState supports both rear-axle and center-of-mass frames
4. **Interpolation Support**: InterpolatableState protocol enables trajectory interpolation
5. **Immutability where appropriate**: OrientedBox uses frozen dataclass for Dimension
6. **Cached Properties**: Expensive computations (geometry Polygon, waypoint) are cached
7. **Type Safety**: Union types and enums for tracked object classification

## Dependencies (What We Import)

### Internal nuPlan
- `nuplan.common.geometry.transform` - Coordinate transformations
- `nuplan.common.utils.interpolatable_state` - Interpolation protocol
- `nuplan.common.utils.split_state` - State decomposition for interpolation

### External
- `numpy` - Array operations, geometric calculations
- `shapely.geometry.Polygon` - 2D collision detection
- Standard library: `dataclasses`, `functools.cached_property`, `typing`

## Dependents (Who Imports Us)

**Used everywhere in nuPlan!** Critical dependencies:
- **Planning**:
  - `nuplan/planning/simulation/planner/` - PlannerInput contains ego_state, observations
  - `nuplan/planning/simulation/trajectory/` - Trajectories built from Waypoints
  - `nuplan/planning/simulation/observation/` - Observations contain TrackedObjects
  - `nuplan/planning/simulation/controller/` - Controllers track EgoState
- **Metrics**:
  - `nuplan/planning/metrics/` - All metrics evaluate based on EgoState trajectories and agent interactions
- **Scenario Builder**:
  - `nuplan/planning/scenario_builder/` - Scenarios provide initial EgoState and TrackedObjects
- **Database**:
  - `nuplan/database/` - ORM models serialize to these state representations
- **Training**:
  - `nuplan/planning/training/preprocessing/features/` - ML features extracted from states

## Critical Files (Prioritized)

1. **`state_representation.py`** (576 lines) - FOUNDATIONAL
   - TimePoint, TimeDuration, Point2D, StateSE2, StateVector2D
   - All geometric state primitives
   - Read this FIRST to understand coordinate systems

2. **`ego_state.py`** (318 lines) - CRITICAL for planning
   - EgoState class (main ego vehicle representation)
   - build_from_rear_axle(), build_from_center() constructors
   - Serialization and interpolation support

3. **`oriented_box.py`** (234 lines) - Core geometry
   - OrientedBox class (bounding boxes for all actors)
   - Collision detection utilities
   - Shapely integration

4. **`tracked_objects.py`** (127 lines) - Scene management
   - TrackedObjects collection
   - Efficient type-based filtering
   - Used in every observation

5. **`agent.py`** (70 lines) - Other vehicles/pedestrians
   - Agent class combining state + temporal info
   - from_agent_state() factory method

6. **`waypoint.py`** (182 lines) - Trajectory building block
   - Waypoint class for trajectory points
   - Serialization for trajectory storage

7. **`tracked_objects_types.py`** - Enum definitions
   - TrackedObjectType enum
   - AGENT_TYPES, STATIC_OBJECT_TYPES constants

8. **`car_footprint.py`** - Ego vehicle geometry
   - CarFootprint class
   - Axle-to-center conversions

9. **`dynamic_car_state.py`** - Vehicle dynamics
   - DynamicCarState (velocities, accelerations)
   - Reference frame transformations

10. **`vehicle_parameters.py`** - Physical vehicle properties
11. **`scene_object.py`** - Base scene object
12. **`agent_state.py`** - Single-timestep agent representation
13. **`agent_temporal_state.py`** - Temporal agent history
14. **`static_object.py`** - Static scene elements
15. **`transform_state.py`** - State transformation utilities
16. **`ego_temporal_state.py`** - Temporal ego history

## Common Usage Patterns

### Creating EgoState (from rear axle)
```python
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters

ego = EgoState.build_from_rear_axle(
    rear_axle_pose=StateSE2(x=0.0, y=0.0, heading=0.0),
    rear_axle_velocity_2d=StateVector2D(10.0, 0.0),  # 10 m/s forward
    rear_axle_acceleration_2d=StateVector2D(0.0, 0.0),
    tire_steering_angle=0.1,  # radians
    time_point=TimePoint(time_us=1000000),  # 1 second
    vehicle_parameters=vehicle_params,
)
```

### Querying TrackedObjects by type
```python
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

# Get all vehicles
vehicles = tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE)

# Get all dynamic agents (vehicles + peds + bikes)
all_agents = tracked_objects.get_agents()

# Get static objects
barriers = tracked_objects.get_tracked_objects_of_type(TrackedObjectType.BARRIER)
```

### Collision checking
```python
from nuplan.common.actor_state.oriented_box import in_collision

if in_collision(ego.car_footprint.oriented_box, agent.box):
    print("Collision detected!")
```

### Time arithmetic
```python
from nuplan.common.actor_state.state_representation import TimePoint, TimeDuration

t0 = TimePoint(time_us=1000000)  # 1 second
dt = TimeDuration.from_s(0.1)    # 100ms
t1 = t0 + dt                      # 1.1 seconds

time_diff = t1.diff(t0)  # TimeDuration(0.1s)
```

## Gotchas & Pitfalls

1. **Heading Convention**: Heading is in **radians**, **0 = East**, **counter-clockwise positive**
   - NOT degrees! NOT North-up!
   - Use `nuplan.common.geometry` for conversions if needed

2. **Reference Frames**: Ego has TWO reference frames
   - **Rear axle**: Physically at middle of rear axle
   - **Center**: Center of mass (CoM)
   - Always check which frame you're in! Use `build_from_rear_axle()` vs `build_from_center()`
   - Velocities/accelerations transform between frames (see `dynamic_car_state.py`)

3. **Time Units**: TimePoint uses **microseconds** internally
   - Always use constructors: `TimePoint.from_s()`, `TimeDuration.from_ms()`, etc.
   - Don't construct directly (it will raise RuntimeError!)

4. **TrackedObjects Sorting**: Internal list is sorted by TrackedObjectType
   - Don't assume insertion order is preserved!
   - Use `get_tracked_objects_of_type()` for filtering

5. **Interpolation**: EgoState and Waypoint implement `InterpolatableState`
   - Supports `to_split_state()` → separate linear/angular/fixed states
   - Angular states (heading) interpolated differently than linear (x, y)
   - This is critical for trajectory interpolation!

6. **Polygon Geometry**: `OrientedBox.geometry` is a **cached_property**
   - First access builds Shapely Polygon
   - Subsequent accesses reuse cached value
   - Immutable! Don't modify box after creating geometry

7. **Serialization Format**: `EgoState.deserialize()` expects exactly 9 values
   - Ordering: [time_us, x, y, heading, vx, vy, ax, ay, steering_angle]
   - Backward compatibility - don't change!

8. **StateSE2 Equality**: Uses `math.isclose()` with tolerances
   - x, y: 1e-3 (1mm)
   - heading: 1e-4 (0.0001 rad ≈ 0.006 degrees)
   - Don't use `==` for exact comparison!

9. **Agent vs AgentState vs AgentTemporalState**:
   - `AgentState`: Single timestep, no history/prediction
   - `AgentTemporalState`: Has predictions/past_trajectory attributes
   - `Agent`: Multiple inheritance - full representation
   - Choose appropriate type based on whether you need temporal info!

10. **Vehicle Parameters**: Required for EgoState construction
    - Contains wheelbase, dimensions, etc.
    - Usually loaded from scenario metadata
    - Don't create ad-hoc - use scenario's vehicle_parameters

## Test Coverage Notes

Test files in `nuplan/common/actor_state/test/`:
- Unit tests for each state class
- Serialization/deserialization roundtrips
- Reference frame conversion validation
- Collision detection edge cases
- Time arithmetic correctness

See tests for usage examples and edge cases!

## Related Documentation

### Parent Module
- `nuplan/common/CLAUDE.md` - Overview of common utilities

### Sibling Modules
- `nuplan/common/geometry/CLAUDE.md` - Geometric transformations (imports from here)
- `nuplan/common/maps/CLAUDE.md` - Map API (uses StateSE2 for queries)
- `nuplan/common/utils/CLAUDE.md` - Interpolation, split state utilities

### Dependent Modules
- `nuplan/planning/simulation/planner/CLAUDE.md` - Uses EgoState, TrackedObjects
- `nuplan/planning/simulation/trajectory/CLAUDE.md` - Built from Waypoints
- `nuplan/planning/simulation/observation/CLAUDE.md` - Wraps TrackedObjects
- `nuplan/planning/scenario_builder/CLAUDE.md` - Provides initial states
- `nuplan/planning/metrics/CLAUDE.md` - Evaluates state trajectories

---

**AIDEV-NOTE**: This module is THE foundation of nuPlan. Every state in every scenario, simulation, and evaluation uses these classes. Master this before diving into planning/simulation code!

**AIDEV-TODO**: Consider adding type aliases (e.g., `EgoTrajectory = List[EgoState]`) for common patterns

**AIDEV-QUESTION**: Why does EgoStateDot exist as an empty subclass? Semantic marker for derivatives in dynamics equations?
