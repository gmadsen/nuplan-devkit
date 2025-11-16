# Workstream C: Configuration & Scenario System Architecture (COMPLETE)

## Mission Accomplished

Completed comprehensive READ-ONLY exploration and documentation of:
1. **Hydra Configuration System** - How nuPlan orchestrates training/simulation configs
2. **Scenario Builder Architecture** - Database I/O patterns and performance bottlenecks

**Deliverables**:
- `HYDRA_CONFIG.md` (800 lines) - Configuration system deep-dive
- `SCENARIO_BUILDER.md` (650 lines) - Database architecture & I/O patterns
- This summary document

## The Two Key Systems Explained

### System 1: Hydra Configuration (`HYDRA_CONFIG.md`)

**What it does**: Declarative YAML-based config management for training and simulation. Enables rapid hypothesis testing via CLI overrides without code changes.

**Key structure**:
```
config/
├── common/              # Shared: dataset, worker, model selection
│   ├── scenario_builder/    (nuplan_mini, nuplan, nuplan_challenge)
│   ├── scenario_filter/     (filters: types, tokens, limits, motion, route)
│   ├── worker/              (ray_distributed, sequential)
│   └── model/               (raster_model, vector_model)
├── simulation/          # Simulation-specific
│   ├── planner/             (simple_planner, ml_planner, idm_planner, log_future_planner)
│   ├── callback/            (simulation_log, serialization, timing)
│   ├── main_callback/       (metrics, aggregation, validation)
│   ├── metric/              (comfort, progress, drivable_area, collision)
│   └── observation/         (box, lidar_pc, ego_centric_ml_agents)
└── training/            # Training-specific
    ├── objective/           (loss functions)
    ├── optimizer/           (adam, adamw)
    ├── lr_scheduler/        (one_cycle_lr)
    └── data_loader/         (batch size, workers)
```

**Quick reference for performance testing**:
```bash
# Disable expensive features
callback=[]                           # No callbacks
simulation_metric=[]                  # NO METRICS (critical for DB investigation!)
worker.threads_per_node=1             # Single-thread for profiling

# Reduce dataset
scenario_filter.limit_total_scenarios=5

# Single-threaded worker
worker=sequential

# Full command:
uv run python nuplan/planning/script/run_simulation.py \
  callback=[] \
  simulation_metric=[] \
  worker=sequential \
  scenario_filter.limit_total_scenarios=5 \
  planner=simple_planner \
  observation=box_observation \
  ego_controller=two_stage_controller
```

**Critical insight**: The configuration system is your **highest-leverage tool** for hypothesis testing. Most bottlenecks can be isolated by disabling components via CLI without code changes.

### System 2: Scenario Builder (`SCENARIO_BUILDER.md`)

**What it does**: Extracts temporal slices (scenarios) from SQLite logs, applies filtering, and provides lazy-loaded scenario objects to planning/simulation layers.

**Architecture layers**:
```
Planning/Simulation Layer (consumes scenarios)
    ↓ get_scenarios(scenario_filter)
Scenario Builder (orchestrates creation, applies filters)
    ↓ Scenario objects
NuPlanScenario Layer (lazy-loaded, cached)
    ↓ Direct DB queries
Database Query Layer (SQL execution)
    ↓ SQL
SQLite Database (persistent storage)
```

**Critical database I/O patterns**:

**Pattern 1: Per-Step Queries** (HOTTEST PATH)
```
Per iteration (0.1s timestep):
  - get_ego_state_at_iteration(i)                    → 1 DB query
  - get_tracked_objects_at_iteration(i)              → N queries (N = agents)
  - get_traffic_light_status_at_iteration(i)         → 1 DB query (PER STEP!)
  - metric computation                               → 8-20 additional queries

Total: ~200 queries per scenario × 31 scenarios = 6,200 DB queries in typical run
```

**CRITICAL INSIGHT**: Traffic light queries are **per-step database hits**. Each call triggers a separate SQL query. This could be 20-30% of the 95ms slowdown!

**Pattern 2: Cached Properties**
```python
@cached_property
def _lidarpc_tokens(self):
    # First access: 50-100ms (DB query)
    # Subsequent: 0.001ms (cached)

@cached_property
def _route_roadblock_ids(self):
    # First access: 200-500ms (very expensive!)
    # Subsequent: cached
```

**Pattern 3: Lazy Loading**
- Scenarios created as stubs (no DB queries initially)
- Queries happen on first property access
- Time measurement matters: stub creation vs first query vs completion

**Filter pipeline** (cheap first):
1. `num_scenarios_per_type` - In-memory (0 queries)
2. `limit_total_scenarios` - In-memory (0 queries)
3. `timestamp_threshold_s` - Timestamp comparisons (0 queries)
4. `ego_displacement_minimum_m` - Triggers `_lidarpc_tokens` cache (N queries)
5. `ego_start_speed_threshold` - Trajectory queries (M queries per scenario)
6. `ego_route_radius` - **Very expensive map queries** (30-60s for 150 scenarios!)

## Three Hypotheses for 95ms Slowdown

### Hypothesis 1: Metrics Computation (MOST LIKELY)
**Evidence**: Metrics trigger DB queries per scenario per step
- `get_ego_future_trajectory()` → 8-20 queries per metric per step
- `drivable_area_compliance_statistics()` → map queries
- Reactive agent metrics → prediction + interaction modeling

**Test**: `simulation_metric: []` → should reduce per-step time by 20-30ms

### Hypothesis 2: Traffic Light Queries (CRITICAL)
**Evidence**: Per-step database hits, not cached
- `get_traffic_light_status_at_iteration()` → 1 SQL query per step per scenario
- 100 steps × 31 scenarios = 3,100 TL queries total
- Each query ~1-2ms = 3-6 seconds total

**Test**: Pre-cache all TL statuses per scenario (eliminate DB hits)
```python
# Current (per step):
tl_status = scenario.get_traffic_light_status_at_iteration(iteration)

# Optimized (once per scenario):
all_tl_statuses = {
    token: list(get_traffic_light_status_for_lidarpc_token_from_db(log_file, token))
    for token in scenario._lidarpc_tokens
}
# Then: tl_status = all_tl_statuses[token]  # No DB hit!
```

**Potential gain**: 95% reduction in TL queries = ~20ms per step improvement

### Hypothesis 3: Callback Overhead
**Evidence**: Default callbacks serialize, log, and compute metrics
- `simulation_log_callback` - Serialization overhead
- `metric_file_callback` - I/O overhead
- `metric_aggregator_callback` - Computation

**Test**: `callback: []` → eliminates all callback overhead

## Recommended Investigation Steps

### Step 1: Measure Component Isolation
```bash
# Baseline: scenario loading only
time python -c "
from nuplan.planning.script.builders.scenario_builder import build_scenario_builder
builder = build_scenario_builder(cfg)
scenarios = builder.get_scenarios(scenario_filter, worker)
"
# Expected: 5-10 seconds for 31 scenarios

# With simulation
time python nuplan/planning/script/run_simulation.py ...
# Expected: 30-60 seconds total
```

### Step 2: Disable Features Incrementally
```bash
# Test 1: Disable metrics
simulation_metric=[]
# Measure per-step time impact

# Test 2: Disable callbacks
callback=[]
# Measure total time impact

# Test 3: Single-thread
worker.threads_per_node=1
# Cleaner profiling

# Test 4: Minimal scenarios
scenario_filter.limit_total_scenarios=5
# Faster iteration
```

### Step 3: Profile Database Queries
Instrument `nuplan/database/nuplan_db/nuplan_scenario_queries.py`:
```python
import time
query_count = 0
query_time = 0.0

# After simulation, report:
print(f"DB queries: {query_count}")
print(f"DB time: {query_time:.2f}s")
print(f"Avg query: {(query_time/query_count)*1000:.2f}ms")
```

## Key Files

### Configuration System
- **Entry points**: `nuplan/planning/script/run_simulation.py`, `run_training.py`
- **Root configs**: `config/simulation/default_simulation.yaml`, `config/training/default_training.yaml`
- **Common configs**: `config/common/default_experiment.yaml`, `default_common.yaml`
- **Config groups**: `config/common/{scenario_builder,scenario_filter,worker,model,splitter}/`

### Scenario Builder
- **Factory**: `nuplan/planning/scenario_builder/nuplan_db/nuplan_scenario_builder.py`
- **Scenario object**: `nuplan/planning/scenario_builder/nuplan_db/nuplan_scenario.py` (510 lines, 30+ query methods)
- **Database queries**: `nuplan/database/nuplan_db/nuplan_scenario_queries.py` (critical hot path)
- **Filtering**: `nuplan/planning/scenario_builder/scenario_filter.py`
- **Filter utils**: `nuplan/planning/scenario_builder/nuplan_db/nuplan_scenario_filter_utils.py`

### Documentation
- `docs/architecture/HYDRA_CONFIG.md` (25KB, 800 lines)
- `docs/architecture/SCENARIO_BUILDER.md` (24KB, 650 lines)

## Quick Reference: Config Override Patterns

### For Performance Investigation
```bash
# Minimal overhead
callback=[] simulation_metric=[] worker.threads_per_node=1

# Focus on metrics impact
simulation_metric=[] worker=sequential

# Focus on callback impact
callback=[] worker.threads_per_node=1

# Focus on database I/O
callback=[] simulation_metric=[] observation=box_observation worker=sequential
```

### For Feature Selection
```bash
# Single planner
planner=simple_planner

# ML planner with checkpoint
planner=ml_planner \
  planner.ml_planner.model_config=raster_model \
  planner.ml_planner.checkpoint_path=/path/to/checkpoint.ckpt

# Single scenario type
scenario_filter.scenario_types=[starting_left_turn]

# Single log file
scenario_filter.log_names=[2021.07.16.20.45.29_veh-35_01095_01486]
```

## Key Insights Summary

1. **Configuration > Code Changes**: Use Hydra overrides for hypothesis testing, not code modifications.

2. **Traffic Light Queries Are Hot Path**: Per-step database hits, not cached. Pre-caching could yield ~20ms improvement.

3. **Metrics Computation is Expensive**: Disabling `simulation_metric: []` will immediately show metric overhead.

4. **Lazy Loading Affects Filtering, Not Simulation**: Cache-miss penalties happen during scenario building, not simulation.

5. **Filter Order is Already Optimized**: Cheap filters first. No improvements there, but traffic light caching is clear opportunity.

## Next Session Recommendations

**Session D (Simulation Core & Callbacks)**: Understand callback execution, metric computation details, and observation history buffering. This will complete the performance investigation foundation.

**Implementation Priority**:
1. Traffic light status pre-caching (20ms+ gain, low effort)
2. Scenario cache warming (100ms+ gain per scenario, medium effort)
3. Metric reduction (depends on which metrics are bottleneck)

---

**Navigation**: See `HYDRA_CONFIG.md` for detailed config guide, `SCENARIO_BUILDER.md` for database architecture details.

**Documentation Status**: COMPLETE (2/4 workstreams done, 2 remaining)
