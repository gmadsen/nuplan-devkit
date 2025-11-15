# Scenario Builder Module

## Purpose & Key Abstractions

The `scenario_builder` module is the **primary interface for accessing and filtering scenarios from the nuPlan dataset**. It defines how temporal slices (~20 seconds) of driving logs are extracted, classified, and made available for training and simulation. The module provides two core abstractions: `AbstractScenario` (a time slice of a log containing ego state, tracked objects, map, and sensor data) and `AbstractScenarioBuilder` (a factory for constructing and filtering scenarios from database files).

This module bridges the raw database layer (`nuplan/database/nuplan_db_orm/`) and the planning/simulation layers, transforming SQLite records into rich, queryable scenario objects.

## Architecture & Design Patterns

### Factory Pattern
- **AbstractScenarioBuilder**: Factory interface for scenario construction
- **NuPlanScenarioBuilder**: Concrete factory that loads from SQLite DBs and applies filters
- Supports concurrent scenario loading via WorkerPool for parallelism

### Repository Pattern
- **AbstractScenario**: Rich domain object representing a scenario
- **NuPlanScenario**: Concrete implementation with lazy-loaded properties (`@cached_property`)
- Database queries encapsulated in `nuplan_scenario_queries.py`

### Strategy Pattern
- **ScenarioFilter**: Declarative filter specification (types, tokens, limits, etc.)
- **FilterWrapper**: Composable filter functions applied sequentially
- Multiple filtering strategies: per-type limits, timestamp clustering, ego motion filters

### Caching Pattern
- **CachedScenario**: Minimal scenario stub for precomputed features (training only)
- **ScenarioExtractionInfo**: Configurable temporal extraction (duration, offset, subsampling)
- Blob stores (LocalStore/S3Store) for sensor data with download-on-demand

## Dependencies

### Documented Dependencies (Session 1) ✅
- **nuplan/common/actor_state/**: EgoState, VehicleParameters, state representations
- **nuplan/common/geometry/**: Coordinate transforms (not heavily used here)
- **nuplan/common/maps/abstract_map.py**: Map API interface for querying map topology
- **nuplan/common/maps/nuplan_map/**: Concrete map implementation (get_maps_api, roadblock extraction)
- **nuplan/planning/simulation/planner/**: Not directly used, but scenarios feed planners
- **nuplan/planning/simulation/trajectory/**: TrajectorySampling for future waypoint queries

### Undocumented Dependencies (This Session) ⏳
- **nuplan/database/nuplan_db_orm/**: SQLAlchemy ORM models (LidarPc, Image, Track, etc.)
- **nuplan/database/nuplan_db/nuplan_scenario_queries.py**: SQL query functions
- **nuplan/database/common/blob_store/**: LocalStore/S3Store for sensor blobs
- **nuplan/planning/simulation/observation/**: DetectionsTracks, Sensors data types

### Future Dependencies (Session 3-4) ⏳
- **nuplan/planning/simulation/runner/**: Simulation orchestration (consumes scenarios)
- **nuplan/planning/training/**: Training pipeline (consumes scenarios via caching)

## Critical Files (Prioritized)

### Tier 1: Core Abstractions
1. **abstract_scenario.py** (399 lines) - Scenario interface with 30+ query methods
2. **abstract_scenario_builder.py** (52 lines) - Builder interface and repartition strategy
3. **scenario_filter.py** (71 lines) - Declarative filter specification

### Tier 2: nuPlan Concrete Implementation
4. **nuplan_db/nuplan_scenario.py** (510 lines) - Concrete scenario with DB queries
5. **nuplan_db/nuplan_scenario_builder.py** (268 lines) - Scenario factory with filtering
6. **nuplan_db/nuplan_scenario_filter_utils.py** (677 lines) - Filter implementations
7. **nuplan_db/nuplan_scenario_utils.py** (485 lines) - Extraction utilities, blob loading

### Tier 3: Support & Testing
8. **scenario_utils.py** (32 lines) - Time horizon sampling utility
9. **cache/cached_scenario.py** (197 lines) - Training-only cached scenario stub
10. **test/mock_abstract_scenario.py** (440 lines) - Mock for unit testing

## Usage Patterns

### Pattern 1: Building Scenarios for Simulation

```python
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter

# Initialize builder
builder = NuPlanScenarioBuilder(
    data_root="/data/sets/nuplan",
    map_root="/data/sets/nuplan/maps",
    sensor_root="/data/sets/nuplan/nuplan-v1.1/sensor_blobs",
    db_files="/data/sets/nuplan/nuplan-v1.1/splits/mini/*.db",  # Can be list, dir, or single file
    map_version="nuplan-maps-v1.0",
    vehicle_parameters=get_pacifica_parameters()
)

# Define filter
scenario_filter = ScenarioFilter(
    scenario_types=["starting_left_turn", "near_multiple_vehicles"],  # Filter by type
    num_scenarios_per_type=10,  # Limit per type
    limit_total_scenarios=100,  # Total limit
    remove_invalid_goals=True,  # Remove scenarios without valid mission goal
    shuffle=True
)

# Get scenarios (parallelized via worker pool)
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
worker = WorkerPool(max_workers=8)
scenarios = builder.get_scenarios(scenario_filter, worker)

# scenarios is a List[NuPlanScenario]
print(f"Retrieved {len(scenarios)} scenarios")
```

### Pattern 2: Querying Scenario Data

```python
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario

scenario: NuPlanScenario = scenarios[0]

# Basic properties
print(f"Token: {scenario.token}")
print(f"Log: {scenario.log_name}")
print(f"Type: {scenario.scenario_type}")  # e.g., "starting_left_turn"
print(f"Duration: {scenario.duration_s}s")
print(f"Iterations: {scenario.get_number_of_iterations()}")

# Ego state queries
initial_ego = scenario.initial_ego_state  # EgoState at iteration 0
ego_at_10 = scenario.get_ego_state_at_iteration(10)

# Get expert trajectory (generator for memory efficiency)
expert_trajectory = list(scenario.get_expert_ego_trajectory())

# Get future trajectory from a specific iteration
future_traj = list(scenario.get_ego_future_trajectory(
    iteration=0,
    time_horizon=8.0,  # 8 seconds into future
    num_samples=16     # 16 sampled states
))

# Tracked objects (other vehicles, pedestrians)
detections = scenario.get_tracked_objects_at_iteration(0)
for tracked_obj in detections.tracked_objects:
    print(f"{tracked_obj.tracked_object_type}: {tracked_obj.center}")

# Map queries
map_api = scenario.map_api
roadblock_ids = scenario.get_route_roadblock_ids()
mission_goal = scenario.get_mission_goal()  # Optional[StateSE2]
```

### Pattern 3: Advanced Filtering

```python
# AIDEV-NOTE: Filters are applied sequentially in NuPlanScenarioBuilder._create_filter_wrappers()
# Order: num_scenarios_per_type → limit_total_scenarios → timestamp → ego_displacement → starts/stops → token_set → route

scenario_filter = ScenarioFilter(
    # Type-based filtering
    scenario_types=["starting_left_turn", "starting_right_turn"],
    map_names=["us-nv-las-vegas-strip"],
    log_names=["2021.07.16.20.45.29_veh-35_01095_01486"],

    # Quantity limits
    num_scenarios_per_type=50,  # Per-type limit (applied first)
    limit_total_scenarios=0.1,  # 10% of total (float) OR 100 scenarios (int)

    # Temporal filtering
    timestamp_threshold_s=5.0,  # Remove scenarios within 5s of each other

    # Ego motion filters
    ego_displacement_minimum_m=10.0,  # Ego must move at least 10m
    ego_start_speed_threshold=2.0,  # Ego must accelerate above 2 m/s (from below)
    ego_stop_speed_threshold=0.5,   # Ego must decelerate below 0.5 m/s (from above)
    speed_noise_tolerance=0.1,      # Ignore speed changes < 0.1 m/s

    # Route filtering
    ego_route_radius=50.0,  # Must have on-route lane segment within 50m

    # Advanced token filtering
    token_set_path="/path/to/lidarpc_tokens.json",
    fraction_in_token_set_threshold=0.8,  # 80% of lidarpc tokens must be in set

    # Options
    expand_scenarios=False,  # If True, ignore ScenarioMapping durations
    remove_invalid_goals=True,
    shuffle=True
)
```

## Critical Gotchas & Solutions

### 1. Scenario vs Log Confusion
**Problem**: Scenarios are NOT full logs—they're temporal slices extracted from logs.

**Details**:
- A log file (e.g., `2021.07.16.20.45.29_veh-35_01095_01486.db`) contains ~30-60 minutes of data
- Scenarios are extracted at labeled events (e.g., "starting_left_turn" at timestamp X)
- Each scenario is ~20 seconds by default (configurable via `ScenarioExtractionInfo`)

**Solution**:
```python
# WRONG: Assuming scenario.log_name gives you the scenario
log_name = scenario.log_name  # e.g., "2021.07.16.20.45.29_veh-35_01095_01486"

# RIGHT: scenario.token is the unique identifier
scenario_id = scenario.token  # Unique lidarpc token (e.g., "a1b2c3d4...")
```

### 2. Filter Combinatorics Explosion
**Problem**: Multiple filters are multiplicative, not additive. Setting `num_scenarios_per_type=10` and `limit_total_scenarios=100` with 20 types doesn't guarantee 100 scenarios.

**Details** (nuplan_scenario_builder.py:252-262):
```python
# Filters applied sequentially:
# 1. num_scenarios_per_type: 20 types × 10 = 200 scenarios
# 2. limit_total_scenarios: 200 → 100 scenarios (removes 100)
# Result: 100 scenarios, but distribution may be uneven!
```

**Solution**: Use `limit_total_scenarios` as primary limiter, `num_scenarios_per_type` as balancer:
```python
# Good: Balanced sampling
scenario_filter = ScenarioFilter(
    scenario_types=["type_a", "type_b", "type_c"],
    num_scenarios_per_type=50,  # 150 total
    shuffle=True  # Randomize selection within types
)

# Better: Explicit total control
scenario_filter = ScenarioFilter(
    limit_total_scenarios=100,  # Hard limit
    shuffle=True  # Random sampling across all types
)
```

### 3. Lazy Loading Pitfalls
**Problem**: `NuPlanScenario` uses `@cached_property` for expensive queries. Accessing these in tight loops can cause unpredictable delays.

**Example** (nuplan_scenario.py:152-169):
```python
@cached_property
def _lidarpc_tokens(self) -> List[str]:
    # First access: queries DB, extracts tokens (slow!)
    # Subsequent access: returns cached list (fast)
    ...

@cached_property
def _route_roadblock_ids(self) -> List[str]:
    # First access: extracts expert trajectory, queries map (very slow!)
    ...
```

**Solution**: Pre-warm caches in parallel before main processing:
```python
from nuplan.planning.utils.multithreading.worker_utils import worker_map

def warm_scenario_cache(scenario: NuPlanScenario):
    """Pre-compute expensive cached properties."""
    _ = scenario.get_route_roadblock_ids()  # Triggers _route_roadblock_ids
    _ = scenario.get_number_of_iterations()  # Triggers _lidarpc_tokens
    return scenario

# Warm caches in parallel
worker = WorkerPool(max_workers=8)
scenarios = worker_map(worker, warm_scenario_cache, scenarios)
```

### 4. Scenario Extraction Subsampling Edge Case
**Problem**: `ScenarioExtractionInfo.subsample_ratio` must result in integer skip rows, but this isn't validated until runtime.

**Example** (nuplan_scenario.py:104-109):
```python
# subsample_ratio=0.3 → skip_rows=1/0.3=3.333... → ValueError!
# subsample_ratio=0.5 → skip_rows=1/0.5=2.0 → OK
```

**Solution**: Use ratios that divide evenly: 1.0, 0.5, 0.25, 0.2, 0.1, etc.
```python
scenario_mapping = ScenarioMapping(
    scenario_map={
        "starting_left_turn": (20.0, -2.0, 0.5),  # (duration, offset, subsample_ratio)
        # subsample_ratio=0.5 → 20Hz → 10Hz (every 2nd frame)
    },
    subsample_ratio_override=0.5
)
```

### 5. Database Interval vs Scenario Interval Confusion
**Problem**: `database_interval` (0.05s = 20Hz) is fixed, but scenario interval varies with subsampling.

**Details** (nuplan_scenario.py:216-220):
```python
@property
def database_interval(self) -> float:
    if self._scenario_extraction_info is None:
        return 0.05  # 20Hz
    return float(0.05 / self._scenario_extraction_info.subsample_ratio)
    # subsample_ratio=0.5 → interval=0.1s (10Hz)
```

**Solution**: Always use `scenario.database_interval`, never hardcode 0.05:
```python
# WRONG:
num_samples = int(time_horizon / 0.05)  # Breaks with subsampling!

# RIGHT:
num_samples = int(time_horizon / scenario.database_interval)
```

### 6. Memory Explosion with Large Scenario Sets
**Problem**: Loading 10,000+ scenarios into memory can OOM (each scenario holds DB connection, map reference).

**Solution 1**: Use streaming via generators
```python
# Don't do this:
scenarios = builder.get_scenarios(scenario_filter, worker)  # Loads all into memory

# Do this instead:
for batch in chunk_scenarios(scenarios, batch_size=100):
    process_batch(batch)
    del batch  # Free memory
```

**Solution 2**: Use `limit_total_scenarios` aggressively
```python
scenario_filter = ScenarioFilter(
    limit_total_scenarios=500,  # Conservative limit for prototyping
    shuffle=True
)
```

### 7. Sensor Data Download Blocking
**Problem**: `get_sensors_at_iteration()` downloads sensor blobs on-demand, blocking for 10-60 seconds per scenario.

**Details** (nuplan_scenario.py:487-509):
```python
def _get_sensor_data_from_lidar_pc(self, lidar_pc, channels):
    local_store, remote_store = self._create_blob_store_if_needed()
    # If sensor blob not local, downloads from S3 (slow!)
    images = {channel: load_image(image, local_store, remote_store) ...}
```

**Solution**: Pre-download sensor blobs before running experiments:
```bash
# Use enhanced CLI to generate download scripts
just download-tutorial  # For camera_0 + lidar_0
./download_nuplan_aria2c.sh  # Run generated script
```

Or disable sensor loading if not needed:
```python
builder = NuPlanScenarioBuilder(
    ...,
    include_cameras=False  # Don't load camera images
)
```

### 8. Traffic Light Status Generator Exhaustion
**Problem**: `get_traffic_light_status_at_iteration()` returns a generator that can only be consumed once.

**Example**:
```python
traffic_lights = scenario.get_traffic_light_status_at_iteration(0)
list(traffic_lights)  # Consume generator
list(traffic_lights)  # Returns [] - generator exhausted!
```

**Solution**: Convert to list immediately or re-query:
```python
# Option 1: Convert to list
traffic_lights = list(scenario.get_traffic_light_status_at_iteration(0))

# Option 2: Re-query each time
for iteration in range(scenario.get_number_of_iterations()):
    tl_status = list(scenario.get_traffic_light_status_at_iteration(iteration))
```

### 9. Scenario Token is NOT Scenario Name
**Problem**: `scenario.token` is a lidarpc token (hex string), `scenario.scenario_name` is same as token, `scenario.scenario_type` is the semantic type.

**Example** (nuplan_scenario.py:180-198):
```python
@property
def token(self) -> str:
    return self._initial_lidar_token  # e.g., "a1b2c3d4e5f6..."

@property
def scenario_name(self) -> str:
    return self.token  # Same as token!

@property
def scenario_type(self) -> str:
    return self._scenario_type  # e.g., "starting_left_turn"
```

**Solution**: Use `scenario_type` for categorization, `token` for uniqueness:
```python
# Group by scenario type
from collections import defaultdict
scenarios_by_type = defaultdict(list)
for scenario in scenarios:
    scenarios_by_type[scenario.scenario_type].append(scenario)
```

### 10. Route Roadblock IDs May Be Empty
**Problem**: `get_route_roadblock_ids()` can return empty list for scenarios without route annotations.

**Details** (nuplan_scenario.py:238-242):
```python
def get_route_roadblock_ids(self) -> List[str]:
    roadblock_ids = get_roadblock_ids_for_lidarpc_token_from_db(...)
    assert roadblock_ids is not None, "Unable to find Roadblock ids"
    return cast(List[str], roadblock_ids)
    # Can return [] if no route in DB!
```

**Solution**: Always check before use:
```python
roadblock_ids = scenario.get_route_roadblock_ids()
if not roadblock_ids:
    print(f"Warning: Scenario {scenario.token} has no route!")
    # Use alternative navigation strategy or skip
```

Or filter upfront:
```python
scenario_filter = ScenarioFilter(
    ego_route_radius=50.0,  # Only keeps scenarios with route
    ...
)
```

### 11. Mission Goal vs Expert Goal Confusion
**Problem**: `get_mission_goal()` returns far future goal (100m+), `get_expert_goal_state()` returns scenario end state.

**Details**:
- **Mission goal**: High-level destination (e.g., "exit parking lot"), can be `None`
- **Expert goal**: Where expert actually ended up at scenario end (always available)

**Solution**: Use expert goal for trajectory evaluation, mission goal for high-level planning:
```python
expert_final_state = scenario.get_expert_goal_state()  # Never None
mission_goal = scenario.get_mission_goal()  # Optional[StateSE2]

if mission_goal:
    # Use for high-level route planning
    pass
else:
    # Fall back to expert goal or use roadblock IDs
    pass
```

### 12. Timestamp Filtering is Log-Local
**Problem**: `timestamp_threshold_s` only filters within each log, not across logs.

**Details** (nuplan_scenario_filter_utils.py:236-280):
```python
def filter_scenarios_by_timestamp(scenario_dict, timestamp_threshold_s):
    for scenario_type in scenario_dict:
        scenario_dict[scenario_type] = _filter_scenarios_by_timestamp(...)
    # Filters per-type, but scenarios from different logs can still cluster
```

**Solution**: If you need global timestamp spacing, post-process:
```python
def global_timestamp_filter(scenarios, threshold_s):
    scenarios.sort(key=lambda s: s.start_time.time_s)
    filtered = [scenarios[0]]
    for scenario in scenarios[1:]:
        if scenario.start_time.time_s - filtered[-1].start_time.time_s >= threshold_s:
            filtered.append(scenario)
    return filtered
```

### 13. Cached Scenario Stub Limitations
**Problem**: `CachedScenario` only implements 3 methods (token, log_name, scenario_type). Using it outside training crashes.

**Details** (cached_scenario.py:12-197):
```python
class CachedScenario(AbstractScenario):
    # Only 3 implemented:
    def token(self) -> str: ...
    def log_name(self) -> str: ...
    def scenario_type(self) -> str: ...

    # All others raise NotImplementedError!
    def get_ego_state_at_iteration(self, iteration):
        raise NotImplementedError(...)
```

**Solution**: Only use `CachedScenario` in training pipeline with precomputed features:
```python
# Training: OK (features pre-cached)
cached_scenario = CachedScenario(log_name, token, scenario_type)

# Simulation: NOT OK (needs live queries)
scenario = builder.get_scenarios(...)  # Use NuPlanScenario
```

### 14. Worker Pool Repartitioning Strategy
**Problem**: `RepartitionStrategy.REPARTITION_FILE_DISK` loads all scenarios then redistributes, causing memory spikes.

**Details** (abstract_scenario_builder.py:13-17):
```python
class RepartitionStrategy(Enum):
    REPARTITION_FILE_DISK = 1  # Load all, then balance (memory spike!)
    INLINE = 2  # Build on each worker, distribute evenly (slower)
```

**Solution**: For large datasets, use `INLINE` strategy (requires custom builder):
```python
# Currently hardcoded in NuPlanScenarioBuilder.repartition_strategy
# Future: Make configurable via constructor
```

### 15. S3 Path Detection Edge Cases
**Problem**: Remote path detection relies on `s3://` prefix, but legacy paths may use environment variables.

**Details** (nuplan_scenario_utils.py:108-148):
```python
def download_file_if_necessary(data_root, potentially_remote_path, verbose):
    if os.path.exists(potentially_remote_path):  # Local path
        return potentially_remote_path
    # Otherwise, tries to infer S3 path from NUPLAN_DATA_ROOT_S3_URL
```

**Solution**: Always set `NUPLAN_DATA_ROOT_S3_URL` when using S3:
```bash
export NUPLAN_DATA_ROOT_S3_URL="s3://motional-nuplan/public/nuplan-v1.1"
```

## Cross-References

### To Documented Modules (Session 1) ✅
- **nuplan/common/actor_state/ego_state.py**: `scenario.get_ego_state_at_iteration()` returns EgoState
- **nuplan/common/actor_state/vehicle_parameters.py**: `scenario.ego_vehicle_parameters` property
- **nuplan/common/maps/abstract_map.py**: `scenario.map_api` returns AbstractMap for topology queries
- **nuplan/common/maps/nuplan_map/utils.py**: `get_roadblock_ids_from_trajectory()` extracts route
- **nuplan/planning/simulation/trajectory/trajectory_sampling.py**: TrajectorySampling for future waypoints

### To Undocumented Modules (This Session) ⏳
- **nuplan/database/nuplan_db_orm/**: SQLAlchemy models (LidarPc, Image, Track, Ego)
- **nuplan/database/nuplan_db/nuplan_scenario_queries.py**: SQL query functions (get_ego_state_for_lidarpc_token_from_db, etc.)
- **nuplan/database/common/blob_store/**: LocalStore/S3Store for sensor blob management
- **nuplan/planning/simulation/observation/observation_type.py**: DetectionsTracks, Sensors data types

### To Future Modules (Session 3-4) ⏳
- **nuplan/planning/simulation/runner/**: Simulation loop consumes scenarios
- **nuplan/planning/training/**: Training pipeline uses scenario builders
- **nuplan/planning/metrics/**: Metrics computed over scenario expert trajectories

## Key Design Decisions

### Why Generators for Temporal Queries?
**Rationale**: Methods like `get_ego_future_trajectory()` return generators to avoid loading entire trajectories into memory. This is critical for long scenarios (60+ seconds) with high frequency (20Hz = 1200+ frames).

**AIDEV-NOTE**: Always consume generators carefully—they can only be iterated once!

### Why Cached Properties?
**Rationale**: Database queries are expensive (disk I/O, SQL parsing). Properties like `_lidarpc_tokens` and `_route_roadblock_ids` are computed once and cached for scenario lifetime.

**Trade-off**: First access is slow, subsequent access is fast. Pre-warm in parallel for best performance.

### Why ScenarioFilter Dataclass?
**Rationale**: Hydra configuration system requires declarative filter specs. Using dataclasses enables YAML-based configuration:

```yaml
# config/scenario_filter/default_scenario_filter.yaml
scenario_types: ["starting_left_turn", "near_multiple_vehicles"]
num_scenarios_per_type: 10
shuffle: true
```

### Why Sequential Filter Application?
**Rationale**: Filters have different computational costs. Applying cheap filters first (num_scenarios_per_type) reduces dataset size before expensive filters (ego_route_radius requires map queries).

**Filter Order** (nuplan_scenario_builder.py:168-250):
1. `num_scenarios_per_type` (cheap: list slicing)
2. `limit_total_scenarios` (cheap: list slicing)
3. `timestamp_threshold_s` (medium: timestamp sorting)
4. `ego_displacement_minimum_m` (expensive: trajectory extraction)
5. `ego_start/stop_speed_threshold` (expensive: trajectory + speed checks)
6. `token_set_threshold` (medium: set intersection)
7. `ego_route_radius` (very expensive: map queries)

## Common Workflow: End-to-End Scenario Usage

```python
# 1. Initialize builder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool

builder = NuPlanScenarioBuilder(
    data_root="/data/sets/nuplan",
    map_root="/data/sets/nuplan/maps",
    sensor_root="/data/sets/nuplan/nuplan-v1.1/sensor_blobs",
    db_files="/data/sets/nuplan/nuplan-v1.1/splits/mini/*.db",
    map_version="nuplan-maps-v1.0"
)

# 2. Define filter
scenario_filter = ScenarioFilter(
    scenario_types=["starting_left_turn"],
    num_scenarios_per_type=5,
    remove_invalid_goals=True,
    shuffle=True
)

# 3. Build scenarios
worker = WorkerPool(max_workers=4)
scenarios = builder.get_scenarios(scenario_filter, worker)

# 4. Use scenarios
for scenario in scenarios:
    # Get initial state
    initial_ego = scenario.initial_ego_state

    # Get map context
    map_api = scenario.map_api
    nearby_lanes = map_api.get_proximal_map_objects(
        initial_ego.center, radius=50.0, layers=[SemanticMapLayer.LANE]
    )

    # Get expert trajectory
    expert_traj = list(scenario.get_ego_future_trajectory(
        iteration=0, time_horizon=8.0, num_samples=16
    ))

    # Get tracked objects
    detections = scenario.get_tracked_objects_at_iteration(0)

    # Process scenario...
```

## AIDEV Notes & TODOs

**AIDEV-NOTE**: The scenario_builder module is the dataset access layer. Any performance issues in training/simulation often trace back to inefficient scenario queries. Profile scenario loading first!

**AIDEV-TODO**: Document scenario type taxonomy (starting_left_turn, near_multiple_vehicles, etc.) - see nuplan/database/nuplan_db/scenario_tag.py

**AIDEV-TODO**: Add example of custom ScenarioMapping for competition scenarios with different durations

**AIDEV-QUESTION**: Should RepartitionStrategy be configurable in NuPlanScenarioBuilder constructor? Currently hardcoded to REPARTITION_FILE_DISK.

**AIDEV-NOTE**: When debugging "scenario not found" errors, check:
1. NUPLAN_DATA_ROOT env var is set
2. DB files exist and are readable
3. Filters aren't too restrictive (check logs for "Extracted X scenarios")
4. Map version matches (nuplan-maps-v1.0 vs v1.1)
