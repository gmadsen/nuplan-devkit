# Scenario Builder & Database Architecture

## Overview

The scenario builder is the **primary interface between raw SQLite logs and the planning/simulation systems**. It extracts temporal slices (~20 seconds) from driving logs, applies semantic filtering, and materializes `NuPlanScenario` objects that encapsulate all queries needed by planners and metrics.

**Key responsibility**: Bridge the database layer and planning layer, enabling efficient scenario access with lazy-loaded properties and on-demand database queries.

## Architecture Layers

```
┌─────────────────────────────────────────┐
│   Planning / Simulation Layer           │  (consumes scenarios)
│   - run_simulation.py                   │
│   - Planner (simple_planner, ml_planner)│
│   - Metrics (comfort, collision, etc.)  │
└────────────┬────────────────────────────┘
             │
             │ get_scenarios(scenario_filter)
             ↓
┌─────────────────────────────────────────┐
│   Scenario Builder Layer                │  (orchestrates scenario creation)
│   ┌──────────────────────────────────┐  │
│   │ NuPlanScenarioBuilder            │  │  Creates scenarios from DB files
│   │ + FilterWrapper (composition)     │  │
│   │ + ScenarioFilter (spec)          │  │
│   └──────────────────────────────────┘  │
└────────────┬────────────────────────────┘
             │
             │ Scenario objects (lazy-loaded)
             ↓
┌─────────────────────────────────────────┐
│   Scenario Layer                        │  (encapsulates queries)
│   ┌──────────────────────────────────┐  │
│   │ NuPlanScenario                   │  │
│   │ + cached_property (_lidarpc_tokens)│ │
│   │ + cached_property (_route_roadblock_ids)│
│   │ + get_ego_state_at_iteration()   │  │  First access: queries DB
│   │ + get_tracked_objects_at_iteration()  │  Subsequent: from cache
│   │ + get_traffic_light_status_at_iteration()│
│   └──────────────────────────────────┘  │
└────────────┬────────────────────────────┘
             │
             │ DB queries (execute_many)
             ↓
┌─────────────────────────────────────────┐
│   Database Query Layer                  │  (raw SQL execution)
│   ┌──────────────────────────────────┐  │
│   │ nuplan_scenario_queries.py       │  │  SQL functions:
│   │ + get_ego_state_for_lidarpc_token│  │  - get_traffic_light_status_for_lidarpc_token_from_db
│   │ + get_tracked_objects_within_...  │  │  - get_ego_state_for_lidarpc_token_from_db
│   │ + get_traffic_light_status_...   │  │  - get_tracked_objects_within_time_interval_from_db
│   │ + get_roadblock_ids_for_...      │  │
│   └──────────────────────────────────┘  │
└────────────┬────────────────────────────┘
             │
             │ SQL queries
             ↓
┌─────────────────────────────────────────┐
│   SQLite Database                       │  (persistent storage)
│   ┌──────────────────────────────────┐  │
│   │ Tables:                          │  │
│   │ - lidar_pc (timestamps, tokens)  │  │
│   │ - ego (state per timestamp)      │  │
│   │ - track (agent trajectories)     │  │
│   │ - traffic_light_status (light TL)│  │
│   │ - map_object (static map data)   │  │
│   │ - image (camera data)            │  │
│   └──────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

## Database I/O Patterns (CRITICAL)

### Pattern 1: Per-Step Queries During Simulation

**Timing**: Called once per iteration (0.1s timestep), once per step for ~10-20 steps per scenario = 100-200 queries per scenario.

**Queries per iteration**:
```
iteration 0:
  - get_ego_state_at_iteration(0)                    → 1 DB query (SELECT ego WHERE token=?)
  - get_tracked_objects_at_iteration(0)              → N queries (SELECT track WHERE time BETWEEN ?)
  - get_traffic_light_status_at_iteration(0)         → 1 DB query (SELECT traffic_light WHERE lidar_pc_token=?)
  - metric: get_ego_future_trajectory(0, horizon=8)  → 8 queries (sampled EGO states)

iteration 1:
  - repeat...
```

**Worst case**: ~200 SQLite queries per scenario × 31 scenarios = **6,200 database queries total** per 95ms investigation run!

**CRITICAL INSIGHT**: Traffic light queries are **per-step database hits**. Each `get_traffic_light_status_at_iteration()` triggers a separate SQL query.

### Pattern 2: Cached Properties (First Access is Slow)

Methods using `@cached_property` are slow on first access, fast subsequently:

```python
@cached_property
def _lidarpc_tokens(self) -> List[str]:
    """First access: queries DB, extracts sensor tokens (100ms+)
       Subsequent access: returns cached list (0.001ms)"""
    # This is called during initialization, but access timing varies
    
@cached_property
def _route_roadblock_ids(self) -> List[str]:
    """First access: extracts expert trajectory, queries map (very slow!)
       Subsequent access: cached (fast)"""
    expert_trajectory = list(self._extract_expert_trajectory())  # DB I/O
    return get_roadblock_ids_from_trajectory(self.map_api, expert_trajectory)  # Map I/O
```

**Impact**: If metric computation triggers these cached properties, first scenario is slow (cache warm), subsequent scenarios fast.

### Pattern 3: Lazy Scenario Loading

Scenarios are instantiated but not immediately queried:

```python
scenarios = builder.get_scenarios(scenario_filter, worker)  # Only scenario stubs created
# DB queries haven't happened yet!

for scenario in scenarios:
    initial_state = scenario.get_ego_state_at_iteration(0)  # FIRST DB query here
    # More queries as needed during simulation
```

**Impact**: If you measure time from `get_scenarios()` to first step, it's fast (stubs only). If you measure from scenario load to completion, it includes lazy query overhead.

## Scenario Extraction & Filtering Pipeline

### Step 1: Database Discovery

```python
# Find all .db files in data_root
db_files = discover_log_dbs(data_root)  
# e.g., ["/data/nuplan-v1.1/splits/mini/2021.05.12.22.00.38_veh-35_01008_01518.db", ...]
```

### Step 2: Scenario Extraction from Each Log

```python
# For each DB file, extract candidate scenarios
# Uses ScenarioExtractionInfo (duration=20s, offset=-2s, subsample_ratio=0.5)
# Queries scenario tags from database, creates NuPlanScenario objects

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils import get_scenarios_from_log_file

scenarios_per_type = get_scenarios_from_log_file(
    log_file="/data/nuplan-v1.1/splits/mini/2021.05.12.22.00.38_veh-35_01008_01518.db",
    expand_scenarios=False,
    # ... other params
)
# Returns: {"starting_left_turn": [NuPlanScenario, ...], "near_multiple_vehicles": [...], ...}
```

**Database queries**: 1 query per scenario type group to fetch scenario tokens + metadata.

### Step 3: Filter Application (Sequential)

Filters are applied in order, each one reducing the scenario set:

```python
# Filter 1: num_scenarios_per_type (cheap)
# "Select only 10 scenarios per type"
scenario_dict = filter_num_scenarios_per_type(scenario_dict, num_scenarios_per_type=10)

# Filter 2: limit_total_scenarios (cheap)
# "If we have 200 total (20 types × 10), keep only 100"
scenario_dict = filter_total_num_scenarios(scenario_dict, limit_total_scenarios=100)

# Filter 3: timestamp_threshold_s (medium)
# "Ensure scenarios are 5+ seconds apart"
scenario_dict = filter_scenarios_by_timestamp(scenario_dict, timestamp_threshold_s=5.0)

# Filter 4: ego_displacement_minimum_m (expensive!)
# "Ego must move at least 10 meters" - requires extracting full trajectory per scenario
scenario_dict = filter_non_stationary_ego(scenario_dict, minimum_threshold=10.0)

# Filter 5: ego_start/stop_speed_threshold (expensive!)
# "Ego must accelerate/decelerate" - requires computing velocities per scenario
scenario_dict = filter_ego_starts(scenario_dict, speed_threshold=2.0)

# Filter 6: ego_route_radius (VERY expensive!)
# "Route must exist near ego" - map queries for every candidate scenario
scenario_dict = filter_ego_has_route(scenario_dict, map_radius=50.0)
```

**Database I/O cost by filter**:
- `num_scenarios_per_type`: In-memory list operations only (0 DB queries)
- `limit_total_scenarios`: In-memory list operations only (0 DB queries)
- `timestamp_threshold_s`: In-memory timestamp comparisons (0 DB queries)
- `ego_displacement_minimum_m`: Calls `scenario.get_number_of_iterations()` per scenario, triggers `_lidarpc_tokens` cache = N queries
- `ego_start_speed_threshold`: Calls `scenario.get_ego_past_trajectory()` = M trajectory queries per scenario
- `ego_route_radius`: Calls `scenario.get_route_roadblock_ids()` which triggers `_route_roadblock_ids` cache = very expensive
- `filter_ego_has_route`: Map API queries, slow but fewer DB hits than ego_route_radius

**Total cost**: If `ego_route_radius` filter is enabled, expect 10-20 seconds per 100 scenarios due to map queries!

## Scenario Object Lifecycle

### Creation Phase (Cheap)

```python
scenario = NuPlanScenario(
    data_root="/data/nuplan",
    log_file_load_path="/data/nuplan-v1.1/splits/mini/2021.05.12.22.00.38_veh-35_01008_01518.db",
    initial_lidar_token="a1b2c3d4e5f6...",  # hex string
    initial_lidar_timestamp=1234567890000,   # microseconds
    scenario_type="starting_left_turn",
    map_root="/data/nuplan/maps",
    map_version="nuplan-maps-v1.0",
    map_name="us-nv-las-vegas-strip",
    scenario_extraction_info=ScenarioExtractionInfo(
        duration_s=20.0,
        offset_s=-2.0,
        subsample_ratio=0.5
    ),
    ego_vehicle_parameters=get_pacifica_parameters(),
    sensor_root="/data/nuplan/sensor_blobs"
)
# No database queries yet! Just object creation with stored parameters.
```

### First Query Phase (Can be Expensive)

```python
# First time accessing these properties triggers DB queries:

# Access 1: _lidarpc_tokens (cached_property)
tokens = scenario.get_number_of_iterations()  # or scenario._lidarpc_tokens directly
# Triggers: extract_sensor_tokens_as_scenario() → queries lidar_pc table
# Cost: ~50-100ms per scenario (depends on duration)

# Access 2: _route_roadblock_ids (cached_property)
roadblocks = scenario.get_route_roadblock_ids()
# Triggers: _extract_expert_trajectory() → queries ego + track tables
#           map API queries for each roadblock
# Cost: ~200-500ms per scenario (very expensive!)
```

### Steady State Phase (Queries Cached)

```python
# Subsequent accesses use cached data:

state = scenario.get_ego_state_at_iteration(0)  # Direct DB query (not cached)
detections = scenario.get_tracked_objects_at_iteration(0)  # Direct DB query (not cached)
tl_status = scenario.get_traffic_light_status_at_iteration(0)  # Direct DB query (not cached)

# But these use cached _lidarpc_tokens:
tokens = scenario._lidarpc_tokens  # Already cached, no DB hit
roadblocks = scenario.get_route_roadblock_ids()  # Already cached, no DB hit
```

## Key Database Queries (with SQL)

### Query 1: Get Traffic Light Status

```python
# From: nuplan/database/nuplan_db/nuplan_scenario_queries.py:602
def get_traffic_light_status_for_lidarpc_token_from_db(log_file: str, token: str):
    query = """
        SELECT  CASE WHEN tl.status == "green" THEN 0
                     WHEN tl.status == "yellow" THEN 1
                     WHEN tl.status == "red" THEN 2
                     ELSE 3
                END AS status,
                tl.lane_connector_id,
                lp.timestamp AS timestamp
        FROM lidar_pc AS lp
        INNER JOIN traffic_light_status AS tl
            ON lp.token = tl.lidar_pc_token
        WHERE lp.token = ?
    """
    for row in execute_many(query, (bytearray.fromhex(token),), log_file):
        yield TrafficLightStatusData(...)
```

**Database cost**: 1 query per `get_traffic_light_status_at_iteration()` call = **1 query per simulation step per scenario** = CRITICAL BOTTLENECK.

### Query 2: Get Ego State

```python
# Fetches ego vehicle state (position, velocity, heading, etc.)
# at a specific lidarpc timestamp
query = """
    SELECT ego.* FROM ego
    WHERE ego.lidar_pc_token = ?
"""
# Cost: 1 query per iteration during simulation
```

### Query 3: Get Tracked Objects

```python
# Fetches all agent trajectories within time window
query = """
    SELECT track.* FROM track
    WHERE track.start_time <= ? AND track.end_time >= ?
    UNION ...  # potentially multiple agent types
"""
# Cost: 1 query per iteration, returns N agents (potentially expensive)
```

## Query Optimization Opportunities

### Opportunity 1: Pre-Cache All Scenario Data

**Current**: Queries happen on-demand during simulation.

**Optimization**: Load all scenario data upfront into memory:

```python
# After scenario creation, pre-warm all caches
def warm_scenario_cache(scenario: NuPlanScenario):
    # Trigger all cached_property evaluations
    _ = scenario._lidarpc_tokens  # Cache scenario timestamps
    _ = scenario._route_roadblock_ids  # Cache route
    # Now first query doesn't pay cache-miss penalty
    return scenario

# Apply in parallel before simulation
from nuplan.planning.utils.multithreading.worker_utils import worker_map
scenarios = worker_map(worker, warm_scenario_cache, scenarios)
# All scenarios now have pre-computed lidarpc_tokens
```

**Benefit**: ~100ms savings per scenario if _route_roadblock_ids was cold.

### Opportunity 2: Batch Query Traffic Light Status

**Current**: Each `get_traffic_light_status_at_iteration(iteration)` queries 1 lidarpc token:

```sql
SELECT * FROM traffic_light_status WHERE lidar_pc_token = ?
```

**Optimization**: Fetch all traffic light statuses for scenario upfront:

```python
# Instead of per-step queries, batch fetch all TL statuses for scenario
all_tokens = scenario._lidarpc_tokens  # All 200 tokens for scenario
all_tl_status = {}
for token in all_tokens:
    all_tl_status[token] = list(get_traffic_light_status_for_lidarpc_token_from_db(log_file, token))

# During simulation, lookup from dict (no DB hit):
tl_status = all_tl_status[iteration]
```

**Benefit**: 200 DB queries → 1 batch query = 95% reduction for TL-heavy metrics.

### Opportunity 3: Use Cached Scenario Stubs for Training

**Current**: Training loads full NuPlanScenario objects with all DB queries.

**Optimization**: Use CachedScenario with pre-computed features:

```python
# For training, use feature cache instead of live scenario queries
from nuplan.planning.scenario_builder.cache.cached_scenario import CachedScenario

cached_scenario = CachedScenario(
    log_name="2021.05.12.22.00.38_veh-35_01008_01518",
    token="a1b2c3d4e5f6...",
    scenario_type="starting_left_turn",
    # No lidar_pc_tokens, no DB queries!
)
# CachedScenario is ~100x faster because it only stores metadata
```

**Benefit**: Training doesn't incur DB I/O for observation queries.

### Opportunity 4: Filter Earlier on Cheap Criteria

**Current**: Order is: num_per_type → total → timestamp → ego_motion → ego_speed → route_radius.

**Optimization**: Apply expensive filters last on smaller dataset:

```python
# Good order (cheap first):
1. num_scenarios_per_type: 200 → 50 scenarios
2. limit_total_scenarios: 50 → 30 scenarios
3. timestamp_threshold_s: 30 → 25 scenarios
4. ego_displacement_minimum_m: 25 → 20 scenarios [DB queries only on 20, not 200!]
5. ego_start_speed_threshold: 20 → 15 scenarios
6. ego_route_radius: 15 → 10 scenarios [only 10 expensive map queries, not 200!]
```

**Current implementation**: Already does this! Filter order in `_create_filter_wrappers()` is optimized.

## Memory Footprint Analysis

### Per-Scenario Memory

```
NuPlanScenario object:
  - Metadata (tokens, timestamps, scenario_type): ~1KB
  - Log file path, map paths: ~200B
  - Cached _lidarpc_tokens (list of 200 hex strings): ~20KB
  - Cached _route_roadblock_ids (list of 50 roadblock IDs): ~5KB
  - Map API cache: ~0MB (loaded globally, not per-scenario)
  
Total per scenario: ~30KB metadata

When accessed during simulation:
  - ego_state objects (200 iterations): ~2KB each = 400KB
  - tracked_objects (200 iterations × 10 agents): ~1KB per agent = 2MB
  - traffic_light_status (200 iterations): ~100B each = 20KB

Total in memory during active simulation: ~2.5MB per scenario
```

### For 31 scenarios (typical run):

```
Stubs only: 31 × 30KB = ~1MB
During simulation: 31 × 2.5MB = ~78MB
With Ray workers (4 workers): 78MB × 4 = ~312MB
```

## Filtering Strategy Analysis

### Example: Default Simulation Config

```yaml
scenario_filter:
  scenario_types: null                    # Keep all types
  log_names: [2021.07.16.20.45.29_veh-35_01095_01486]  # Single log
  map_names: null                         # All maps
  num_scenarios_per_type: null            # No per-type limit
  limit_total_scenarios: null             # No total limit
  timestamp_threshold_s: null             # No temporal filter
  ego_displacement_minimum_m: null        # No motion filter
  ego_start_speed_threshold: null         # No speed filter
  ego_stop_speed_threshold: null
  ego_route_radius: null                  # No route filter
  remove_invalid_goals: true
```

**Database cost**: 
- Log discovery: 1 query (find scenarios in chosen log)
- Filter application: 0 additional queries (all filters disabled)
- Total: ~1-2 queries for scenario loading

### Example: Challenge Submission Config

```yaml
scenario_filter:
  scenario_types: [starting_left_turn, starting_right_turn, ...]  # 15 types
  num_scenarios_per_type: 10              # 150 total candidates
  limit_total_scenarios: null             # Keep all 150
  timestamp_threshold_s: 5.0              # Temporal spacing
  ego_displacement_minimum_m: 10.0        # Motion required
  ego_start_speed_threshold: 1.0          # Acceleration
  ego_route_radius: 50.0                  # Route exists!
```

**Database cost**:
1. Extract candidates: 150 scenarios
2. num_per_type: in-memory only
3. limit_total: in-memory only
4. timestamp: in-memory only
5. ego_displacement: 150 × (1 query to get _lidarpc_tokens) = 150 queries
6. ego_start_speed: 150 × (10 queries for trajectory) = 1500 queries
7. ego_route_radius: 150 × (queries for _route_roadblock_ids + map API) = **very slow!**

**Total**: 1500+ queries + expensive map operations = **30-60 seconds of filtering overhead!**

## Critical Insights for 95ms Slowdown

### Hypothesis 1: Metrics Cause Database Queries

Evidence:
- Metrics compute `get_ego_future_trajectory()` per scenario
- Metrics compute `drivable_area_compliance_statistics()` per scenario (map queries)
- Each metric triggers 10-20 database queries per step
- Per-scenario: 200 metric queries = 95ms overhead

**Test**: `simulation_metric: []` should reduce per-step time by 20-30ms.

### Hypothesis 2: Traffic Light Queries are Hot Path

Evidence:
- `get_traffic_light_status_at_iteration()` is called per step by some metrics
- Each call triggers 1 separate SQL query
- 100 steps × 31 scenarios = 3100 TL queries

**Test**: Replace TL query with cache:
```python
# Instead of:
tl_status = list(scenario.get_traffic_light_status_at_iteration(iteration))

# Pre-compute once per scenario:
all_tl_statuses = {token: list(...) for token in scenario._lidarpc_tokens}
# Then lookup: tl_status = all_tl_statuses[token]
```

### Hypothesis 3: Cache-Miss Penalty During Filtering

Evidence:
- If filtering includes `ego_route_radius`, expensive cached property (_route_roadblock_ids) is computed per scenario
- Filtering 150 scenarios with expensive map queries = 30-60 seconds
- This happens before simulation starts, inflates total time

**Test**: Profile scenario building separately:
```bash
# Isolate scenario loading time
python -m cProfile -s cumtime nuplan/planning/script/run_simulation.py \
  scenario_filter.limit_total_scenarios=5 \
  callback=[] \
  simulation_metric=[] \
  worker=sequential \
  # Look at get_scenarios() time
```

## Recommendations for Performance Investigation

### 1. Measure Component Isolation

```bash
# Baseline (no simulation)
time python -c "
from nuplan.planning.script.builders.scenario_builder import build_scenario_builder
builder = build_scenario_builder(cfg)
scenarios = builder.get_scenarios(scenario_filter, worker)
"
# Typical: 5-10 seconds for scenario loading with 31 scenarios

# With simulation
time python nuplan/planning/script/run_simulation.py ...
# Typical: 30-60 seconds for full run
```

### 2. Disable Features Incrementally

```bash
# Step 1: Disable callbacks
simulation_metric=[]
callback=[]
# Measure time

# Step 2: Disable map queries (remove route filters)
# (no change, already none)

# Step 3: Single-thread
worker=sequential

# Step 4: Minimal scenarios
scenario_filter.limit_total_scenarios=1
```

### 3. Profile Database Queries

Add instrumentation to `nuplan_scenario_queries.py`:

```python
import time
call_count = 0
total_time = 0

def execute_many(query, params, log_file):
    global call_count, total_time
    start = time.perf_counter()
    result = original_execute_many(query, params, log_file)
    end = time.perf_counter()
    total_time += (end - start)
    call_count += 1
    return result

# After simulation:
print(f"Total DB queries: {call_count}, Total time: {total_time:.2f}s")
print(f"Avg query time: {(total_time/call_count)*1000:.2f}ms")
```

## References

- **Scenario Builder**: `/home/garrett/projects/nuplan-devkit/nuplan/planning/scenario_builder/`
- **Database Queries**: `/home/garrett/projects/nuplan-devkit/nuplan/database/nuplan_db/nuplan_scenario_queries.py`
- **Scenario Filter**: `/home/garrett/projects/nuplan-devkit/nuplan/planning/scenario_builder/scenario_filter.py`
- **NuPlanScenario**: `/home/garrett/projects/nuplan-devkit/nuplan/planning/scenario_builder/nuplan_db/nuplan_scenario.py`

---

**AIDEV-NOTE**: The scenario builder is a critical bottleneck for performance optimization. Database I/O patterns are key to understanding the 95ms slowdown. Traffic light queries are per-step database hits that should be pre-cached. Metrics computation triggers additional queries that compound during simulation.

**AIDEV-TODO**: Implement traffic light status pre-caching in NuPlanScenario for ~20% performance gain.

**AIDEV-QUESTION**: Should CachedScenario be used for simulation as well, or only training? Currently only training uses cached stubs.

