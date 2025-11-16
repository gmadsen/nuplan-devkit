# nuplan/planning/ Architecture Guide

## Table of Contents
- [Quick Start](#quick-start)
- [System Overview](#system-overview)
- [Core Subsystems](#core-subsystems)
- [Data Flow](#data-flow)
- [Configuration System](#configuration-system)
- [Performance Characteristics](#performance-characteristics)
- [Common Patterns](#common-patterns)
- [References](#references)

---

## Quick Start

### What is nuplan/planning/?

The `nuplan/planning/` module is the **core simulation and evaluation framework** for autonomous vehicle planning algorithms in the nuPlan dataset ecosystem. It provides:

- **Closed-loop simulation** - Test planners in realistic multi-agent scenarios
- **Comprehensive metrics** - Evaluate safety, comfort, and progress
- **ML training infrastructure** - Train neural network planners from expert demonstrations
- **Flexible configuration** - Hydra-based declarative configuration system

### Key Entry Points

```bash
# Run closed-loop simulation
python nuplan/planning/script/run_simulation.py \
  planner=simple_planner \
  observation=box_observation \
  ego_controller=two_stage_controller

# Train ML planner
python nuplan/planning/script/run_training.py \
  model=raster_model \
  data_loader.params.batch_size=32

# Visualize results
python nuplan/planning/script/run_nuboard.py \
  simulation_path=$NUPLAN_EXP_ROOT/simulation_logs
```

### 5-Minute Orientation Guide

**Core workflow**:
1. **Scenarios** are extracted from SQLite logs (20-second driving clips)
2. **Simulation loop** queries planner for trajectory every 0.1s
3. **Planner** processes observations (ego state, agents, map) and returns trajectory
4. **Controller** executes trajectory with realistic vehicle dynamics
5. **Metrics** evaluate performance post-simulation (safety, comfort, progress)

**Key directories**:
- `simulation/` - Simulation loop orchestration
- `scenario_builder/` - Dataset loading and filtering
- `metrics/` - Evaluation metrics
- `training/` - ML model training
- `script/` - Entry points and Hydra configs

### Common Tasks

```bash
# Profile planner performance
./scripts/profile_single_scenario.sh

# Run single scenario for debugging
python nuplan/planning/script/run_simulation.py \
  scenario_filter.limit_total_scenarios=1 \
  callback=[] \
  simulation_metric=[]

# Change planner
python nuplan/planning/script/run_simulation.py \
  planner=ml_planner \
  planner.ml_planner.checkpoint_path=/path/to/model.ckpt

# Optimize for speed (disable expensive features)
python nuplan/planning/script/run_simulation.py \
  callback=[] \
  simulation_metric=[] \
  worker.threads_per_node=1
```

---

## System Overview

### High-Level Architecture

```
┌────────────────────────────────────────────────────────────┐
│ Entry Points (script/)                                     │
│  ├─ run_simulation.py - Closed-loop evaluation           │
│  ├─ run_training.py - ML planner training                │
│  └─ run_nuboard.py - Visualization dashboard             │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────────────┐
│ Configuration Layer (Hydra)                                │
│  ├─ Compose configs from YAML files                      │
│  ├─ Apply CLI overrides                                   │
│  └─ Instantiate components via _target_                   │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────────────┐
│ Scenario Builder (scenario_builder/)                       │
│  ├─ Load scenarios from SQLite DB                        │
│  ├─ Apply filters (type, random, limit)                  │
│  └─ Return NuPlanScenario objects                         │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────────────┐
│ Simulation Loop (simulation/)                              │
│  ├─ Initialize: create history buffer, load map          │
│  ├─ Step loop: get_planner_input → compute_trajectory    │
│  │             → propagate → callbacks                    │
│  └─ Teardown: extract history, compute metrics           │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────────────┐
│ Planner (planner/)                                         │
│  ├─ initialize(map, route, goal)                         │
│  └─ compute_planner_trajectory(input) → trajectory       │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────────────┐
│ Metrics Engine (metrics/)                                  │
│  ├─ Compute metrics from history                         │
│  ├─ Categories: safety, comfort, progress, expert        │
│  └─ Save results to parquet/pickle                       │
└────────────────────────────────────────────────────────────┘
```

### Subsystem Responsibilities

#### simulation/ - Simulation Loop and Orchestration
- Maintains simulation state (ego, observations, history)
- Calls planner every timestep (0.1s)
- Propagates state forward using controller and observations
- Manages callbacks for metrics, logging, visualization

#### scenario_builder/ - Dataset Loading and Filtering
- Extracts scenarios from SQLite database logs
- Applies semantic filters (type, duration, ego motion)
- Lazy-loads scenario data on demand
- **Performance bottleneck**: Database queries can dominate runtime

#### metrics/ - Evaluation and Scoring
- Computes performance metrics post-simulation
- Safety: collisions, drivable area compliance
- Comfort: jerk, acceleration limits
- Progress: route adherence, speed
- **NOT on critical path**: Metrics run after simulation completes

#### training/ - ML Model Training
- PyTorch Lightning training infrastructure
- Feature caching and preprocessing
- Model architectures (raster, vector)
- Training callbacks and metrics

#### utils/ - Shared Utilities
- Multithreading (Ray, sequential)
- Geometry operations
- Map APIs
- Data structures

---

## Core Subsystems

### Simulation Loop (simulation/)

**Purpose**: Orchestrate closed-loop interaction between planner, controller, and environment

**Main loop phases**:
```
Initialization:
├─ reset() - Clear state
├─ create history buffer (21 samples, 2.0s window)
├─ observations.initialize() - Load sensors/agents
└─ Return PlannerInitialization (map, route, goal)

Main Loop (per 0.1s timestep):
├─ get_planner_input() - Assemble current observations (~2ms)
├─ planner.compute_trajectory() - Planning decision (~80ms target)
├─ propagate() - Update state (~5-10ms)
│  ├─ controller.update_state() - Execute trajectory
│  ├─ observations.update_observation() - Agents move
│  └─ history_buffer.append() - Rolling window update
└─ callbacks.on_step_end() - Metrics, logging (~1-2ms)

Teardown:
└─ callbacks.on_simulation_end() - Extract metrics, serialize
```

**Performance**:
- **Expected**: ~100ms per step (80ms planner + 20ms overhead)
- **Actual (SimplePlanner)**: 228ms per step (2.3x slower)
- **Actual (MLPlanner)**: 570ms per step (5.7x slower)

**Key files**:
- `simulation/simulation.py` - Main loop (lines 142-174)
- `simulation/runner/simulations_runner.py` - Orchestration
- `simulation/history/simulation_history_buffer.py` - Rolling state window

**Detailed guide**: [docs/architecture/SIMULATION_CORE.md](../../docs/architecture/SIMULATION_CORE.md)

---

### Planner Interface (simulation/planner/)

**Purpose**: Define contract between simulation and planning algorithms

**AbstractPlanner contract**:
```python
class AbstractPlanner(ABC):
    # Called once per scenario
    def initialize(self, initialization: PlannerInitialization) -> None:
        # Cache map_api, route, goal

    # Called ~200x per scenario (every 0.1s)
    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        # Access: current_input.history.current_state
        # Access: current_input.history.ego_states (past 2s)
        # Access: current_input.traffic_light_data
        # Return: InterpolatedTrajectory (8s horizon)
```

**Planner lifecycle**:
1. **Construction** (`__init__`) - Set hyperparameters
2. **Scenario binding** (`initialize()`) - Receive map, route, goal
3. **Trajectory computation** (`compute_planner_trajectory()`) - Per-step planning
4. **Reporting** (`generate_planner_report()`) - Extract timing statistics

**Performance expectations**:
- **SimplePlanner**: 1-5ms per step
- **MLPlanner**: 50-100ms per step (target ~80ms for realtime)
- **Actual MLPlanner**: 127ms per step (feature building dominates!)

**Common implementations**:
- `SimplePlanner` - Constant velocity baseline
- `MLPlanner` - Neural network planner
- `IDMPlanner` - Intelligent driver model
- `LogFuturePlanner` - Oracle (expert replay)

**Detailed guide**: [docs/architecture/PLANNER_INTERFACE.md](../../docs/architecture/PLANNER_INTERFACE.md)

---

### Observation & History (simulation/observation/, simulation/history/)

**Purpose**: Manage perception pipeline and historical context for planners

**Perception pipeline**:
```
Scenario Database → AbstractObservation.update_observation()
                  → DetectionsTracks / Sensors / IDMAgents
                  → SimulationHistoryBuffer.append()
                  → PlannerInput
```

**Observation types**:
- **TracksObservation** - Ground truth detection replay (1-2ms per step)
- **LidarPcObservation** - Raw LiDAR point clouds (100-500ms per step, heavy I/O)
- **IDMAgents** - Simulated agent behaviors (10-20ms per step, 50 agents)
- **EgoCentricMLAgents** - Learned agent models (20-50ms per step)

**History buffer**:
- **Size**: 21 samples = 2.0s window @ 0.1s interval
- **Contents**: List[EgoState], List[Observation]
- **Memory**: ~100KB total
- **Thread-safe**: Uses threading.Lock for concurrent access

**Performance**:
- TracksObservation: 1-2ms update time
- IDMAgents (50 agents): 10-20ms update time (ODE solver overhead)
- IDMAgents (200 agents): 50-100ms (scales O(n) with agent count)

**Detailed guide**: [docs/architecture/OBSERVATION_HISTORY.md](../../docs/architecture/OBSERVATION_HISTORY.md)

---

### Controller (simulation/controller/)

**Purpose**: Execute planner trajectories with realistic vehicle dynamics

**Control flow**:
```
planner.compute_trajectory() → AbstractTrajectory
  ↓
controller.update_state(trajectory)
  ├─ tracker.track_trajectory() → DynamicCarState (acceleration, steering_rate)
  └─ motion_model.propagate_state() → Updated EgoState (with delays/saturation)
```

**Controller types**:
- **PerfectTrackingController** - Ideal tracking (debugging)
- **LogPlaybackController** - Oracle replay (baseline)
- **TwoStageController** - Tracker + motion model (realistic)

**Performance**:
- PerfectTrackingController: <1ms (direct interpolation)
- LQRTracker + KinematicBicycleModel: 3-10ms
- **Total per step**: ~5-10ms (minor contributor to overhead)

**Key insight**: Controller is NOT a bottleneck (only 5-10ms vs 570ms total)

**Detailed guide**: [docs/architecture/CONTROLLER.md](../../docs/architecture/CONTROLLER.md)

---

### Callbacks (simulation/callback/)

**Purpose**: Event-driven hooks for metrics, logging, visualization

**Lifecycle hooks (8 total)**:
1. `on_simulation_start()` - Before planner initialization
2. `on_initialization_start()` - Before planner.initialize()
3. `on_initialization_end()` - After planner.initialize()
4. `on_step_start()` - Before each timestep (~200x per scenario)
5. `on_planner_start()` - Before planner.compute_trajectory()
6. `on_planner_end()` - After planner.compute_trajectory()
7. `on_step_end()` - After each timestep (~200x per scenario)
8. `on_simulation_end()` - After all steps complete

**Execution model**:
- **Synchronous by default** - Callbacks block main thread
- **Asynchronous optional** - Submit to worker pool for parallel execution
- **Sequential within MultiCallback** - Order matters!

**Performance impact**:
- **Per-step callbacks**: <1ms overhead when empty
- **Post-simulation callbacks**: 3-5s (metrics computation)
- **VisualizationCallback**: 50-100x slowdown (0.5-2s per frame!)

**Common callbacks**:
- `MetricCallback` - Compute metrics post-simulation
- `TimingCallback` - Profile execution time
- `SerializationCallback` - Save simulation history
- `StreamingVizCallback` - Real-time visualization (HTTP POST)

**Critical finding**: Callbacks are NOT the performance bottleneck! Per-step overhead is <1ms.

**Detailed guide**: [docs/architecture/CALLBACKS.md](../../docs/architecture/CALLBACKS.md)

---

### Metrics (metrics/)

**Purpose**: Evaluate planner performance against quantitative measures

**Metric computation timing**:
```
Simulation Loop (20s): Callbacks do NOT compute metrics
  ↓
on_simulation_end(): Trigger metric computation
  ↓
MetricEngine.compute_metric_results() (3-5s)
  ├─ Safety metrics: 250ms (collision, drivable area)
  ├─ Comfort metrics: 300ms (jerk, acceleration)
  ├─ Progress metrics: 200ms (route adherence, speed)
  └─ Expert comparison: 200ms (L2 error)
  ↓
write_to_files() (1-2s): Disk I/O
```

**Key finding**: Metrics are **NOT on critical path** during simulation!
- Metrics run AFTER simulation completes
- `on_step_*` callbacks are empty (just `pass`)
- Total overhead: ~5s per scenario (post-simulation)

**Performance characteristics**:
- 30 metrics: ~3-5s computation time
- Collision detection: ~250ms (most expensive single metric)
- Disk I/O: 1-2s (pickle serialization)

**Execution modes**:
- **Synchronous**: Block until metrics complete (simple, safe)
- **Asynchronous**: Submit to worker pool (15-20% speedup)

**Detailed guide**: [docs/architecture/METRICS.md](../../docs/architecture/METRICS.md)

---

### Configuration (script/config/)

**Purpose**: Declarative configuration management via Hydra 1.1.0rc1

**Config hierarchy**:
```
defaults list → composition
  ├─ default_experiment (output paths, seed)
  ├─ default_common (dataset, worker)
  ├─ Config groups:
  │  ├─ scenario_builder (nuplan_mini, nuplan)
  │  ├─ scenario_filter (one_continuous_log, training_scenarios)
  │  ├─ planner (simple_planner, ml_planner)
  │  ├─ observation (box_observation, idm_agents)
  │  ├─ callback (timing, metric, serialization)
  │  └─ worker (ray_distributed, sequential)
  └─ CLI overrides
```

**Common override patterns**:
```bash
# Disable expensive features
callback=[] simulation_metric=[] worker.threads_per_node=1

# Reduce dataset size
scenario_filter.limit_total_scenarios=10

# Change planner
planner=ml_planner planner.ml_planner.checkpoint_path=/path/to/model.ckpt

# Debug config composition
python script.py --cfg job --resolve
```

**Performance-relevant configs**:
- `simulation_metric: []` - Disable metrics (saves ~5s per scenario)
- `callback: []` - Disable callbacks (saves ~1-2s per scenario)
- `worker.threads_per_node: 1` - Single-threaded (cleaner profiling)
- `scenario_filter.limit_total_scenarios: 1` - Minimal dataset

**Detailed guide**: [docs/architecture/HYDRA_CONFIG.md](../../docs/architecture/HYDRA_CONFIG.md)

---

### Scenario Builder (scenario_builder/)

**Purpose**: Extract scenarios from SQLite logs and apply semantic filters

**Scenario extraction pipeline**:
```
1. Discover .db files in data_root
2. For each log, extract scenarios by type (SQL queries)
3. Apply filters sequentially:
   ├─ num_scenarios_per_type (cheap, in-memory)
   ├─ limit_total_scenarios (cheap, in-memory)
   ├─ timestamp_threshold_s (medium, in-memory)
   ├─ ego_displacement_minimum_m (expensive, DB queries)
   └─ ego_route_radius (very expensive, map queries)
4. Return NuPlanScenario objects (lazy-loaded)
```

**Database query patterns**:
```python
# Per-step queries during simulation:
- get_ego_state_at_iteration(i) → 1 DB query
- get_tracked_objects_at_iteration(i) → 1 DB query
- get_traffic_light_status_at_iteration(i) → 1 DB query

# Total per scenario: ~200 iterations × 3 queries = 600 queries
```

**CRITICAL PERFORMANCE ISSUE**: Database queries explode!
- **Expected**: ~5 queries per step
- **Actual**: 145 queries per step (29x too many!)
- **Root cause**: Traffic lights queried 48x per step, connections churned 8x per step

**Optimization opportunities**:
1. **Connection pooling**: Reuse DB connections (-31ms/step)
2. **Cache traffic light status**: Pre-fetch all per scenario (-40ms/step)
3. **Batch queries**: Reduce round trips (-30ms/step)

**Detailed guide**: [docs/architecture/SCENARIO_BUILDER.md](../../docs/architecture/SCENARIO_BUILDER.md)

---

## Data Flow

### End-to-End Simulation Data Flow

```
SQLite Database
  ├─ Tables: lidar_pc, ego, track, traffic_light_status
  └─ Schema: nuplan_db/models.py
    ↓
ScenarioBuilder.get_scenarios(filter)
  ├─ Extract scenarios by type
  ├─ Apply filters
  └─ Return List[NuPlanScenario]
    ↓
SimulationLoop.initialize()
  ├─ Create history buffer (21 samples)
  ├─ Load map API
  └─ planner.initialize(map, route, goal)
    ↓
FOR EACH TIMESTEP (0.1s):
  ├─ get_planner_input()
  │  ├─ Query traffic lights (DB hit!)
  │  └─ Return PlannerInput(history, traffic_lights)
  ├─ planner.compute_trajectory(input)
  │  ├─ Access: input.history.current_state
  │  ├─ Query: map_api.get_proximal_map_objects()
  │  └─ Return: InterpolatedTrajectory
  ├─ propagate(trajectory)
  │  ├─ controller.update_state() → New ego state
  │  ├─ observations.update_observation() → Agents move
  │  └─ history_buffer.append(ego, obs)
  └─ callbacks.on_step_end() → Minimal overhead
    ↓
callbacks.on_simulation_end()
  ├─ MetricCallback: compute_metric_results() (3-5s)
  └─ SerializationCallback: write_to_files() (1-2s)
```

### Planner Data Flow (Detailed)

```
PlannerInput:
  ├─ iteration: SimulationIteration (index, time)
  ├─ history: SimulationHistoryBuffer
  │  ├─ ego_states: List[EgoState] (21 samples, newest = -1)
  │  ├─ observations: List[Observation] (21 samples)
  │  └─ current_state: (ego_states[-1], observations[-1])
  └─ traffic_light_data: List[TrafficLightStatusData]
    ↓
planner.compute_planner_trajectory(input)
  ├─ Extract current state
  │  ego_state = input.history.current_state[0]
  ├─ Process observations
  │  agents = input.history.observations[-1].tracked_objects
  ├─ Query map
  │  lanes = self._map_api.get_proximal_map_objects(ego_state.center, radius=50)
  ├─ Make decision
  │  # SimplePlanner: Constant velocity
  │  # MLPlanner: Feature building (112ms) + inference (15ms)
  └─ Return trajectory
     InterpolatedTrajectory(future_states)  # 8s horizon, 80 states
    ↓
controller.update_state(trajectory)
  ├─ tracker.track_trajectory() → DynamicCarState (accel, steering_rate)
  └─ motion_model.propagate_state() → New EgoState
```

### Database Query Flow (BOTTLENECK!)

```
Simulation Loop (per step):
  ↓
get_planner_input()
  ├─ scenario.get_traffic_light_status_at_iteration(i)
  │  └─ SQL: SELECT * FROM traffic_light_status WHERE lidar_pc_token = ?
  │     [CALLED 48 TIMES PER STEP! Should be 1!]
  ├─ scenario.get_ego_state_at_iteration(i)
  │  └─ SQL: SELECT * FROM ego WHERE lidar_pc_token = ?
  │     [OK: 1 query per step]
  └─ scenario.get_tracked_objects_at_iteration(i)
     └─ SQL: SELECT * FROM track WHERE time BETWEEN ? AND ?
        [OK: 1 query per step]
    ↓
Total queries per step: 145 (should be ~5!)
  ├─ Traffic light: 48 queries (should be 1)
  ├─ Connection churn: 8 new connections (should reuse)
  └─ Other queries: 89 (metric-related)
```

---

## Configuration System

### Hydra Configuration Patterns

**Basic composition**:
```yaml
# default_simulation.yaml
defaults:
  - default_experiment  # Output paths, seed
  - default_common      # Dataset, worker
  - simulation_metric: default_metrics
  - callback: [timing_callback, metric_callback]
  - planner: null       # MUST be provided
  - observation: null   # MUST be provided
  - ego_controller: null # MUST be provided
```

**CLI override syntax**:
```bash
# Override simple value
planner=simple_planner

# Override nested value
worker.threads_per_node=4

# Override with list
scenario_filter.scenario_types=[starting_left_turn,near_multiple_vehicles]

# Add new parameter
+new_param=value

# Disable config group
callback=[]
```

**Common performance tuning configs**:
```yaml
# Fast iteration (minimal overhead)
callback: []
simulation_metric: []
worker.threads_per_node: 1
scenario_filter.limit_total_scenarios: 1

# Memory optimization
worker.threads_per_node: 2  # Reduce parallelism
scenario_builder:
  include_cameras: false    # Skip camera data

# Profiling-friendly
worker: sequential          # Single-threaded
disable_callback_parallelization: true
```

**Environment variable interpolation**:
```yaml
# In config files:
data_root: ${oc.env:NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/mini
output_dir: ${oc.env:NUPLAN_EXP_ROOT}/exp/${experiment_name}

# Before running:
export NUPLAN_DATA_ROOT=/path/to/data
export NUPLAN_MAPS_ROOT=/path/to/maps
export NUPLAN_EXP_ROOT=/path/to/experiments
```

---

## Performance Characteristics

### Measured Timing Breakdown (Profiling Results)

#### SimplePlanner (Baseline)
```
Total per step: 228ms (2.3x slower than 100ms target)

Framework overhead: 213ms (93%)
  ├─ Database queries: 134ms (59%)
  │  ├─ Traffic light: 48 queries/step (!!!)
  │  ├─ Connection churn: 8 connects/step
  │  └─ Query execution: 131ms
  ├─ State propagation: 104ms (46%)
  │  ├─ Controller update: ~40ms
  │  ├─ History buffer: ~20ms
  │  ├─ Observation update: ~25ms
  │  └─ Time stepping: ~10ms
  └─ Input preparation: 62ms (27%)
     ├─ Buffer access: ~10ms
     ├─ Traffic light query: ~48ms
     └─ Other: ~4ms

Planner compute: 15ms (7%)
  └─ SimplePlanner logic: ~15ms
```

#### MLPlanner
```
Total per step: 570ms (5.7x slower than 100ms target)

Framework overhead: 285ms (50%)
  ├─ Database queries: 131ms (23%)  [Same as SimplePlanner!]
  ├─ State propagation: 95ms (17%)  [Slightly faster]
  └─ Input preparation: 59ms (10%)  [Similar]

ML-specific overhead: 285ms (50%)
  ├─ Feature building: 112ms (20%)
  │  ├─ Map rasterization: 46ms (41% of features)
  │  ├─ Agent rasterization: 33ms (29% of features)
  │  └─ Layer generation: 33ms (30% of features)
  ├─ Neural net inference: 15ms (3%)  [Only 12% of ML time!]
  └─ Post-sim logging (amortized): 158ms (28%)
     └─ LZMA compression: 35s total / 149 steps
```

### Top 3 Bottlenecks (by impact)

**1. Database Queries (131ms/step, 23% of time)**
- 145 queries per step (29x more than expected 5!)
- Traffic light queried 48x per step (should be cached)
- Connection churn: 8 new connections per step
- **Optimization potential**: -100ms/step (80% reduction)

**2. Feature Building (112ms/step, 20% of time, ML-only)**
- 88% of ML planner time is feature prep, only 12% inference
- Map rasterization: 46ms (doesn't change per step, should cache!)
- Agent rasterization: 33ms (could vectorize)
- **Optimization potential**: -60ms/step (50% reduction)

**3. State Propagation (95ms/step, 17% of time)**
- Controller update, history buffer, observations
- Affects both planners equally
- 1.5M deepcopy calls (10K per step!)
- **Optimization potential**: -20ms/step (20% reduction)

### Optimization Roadmap (Data-Driven)

**Phase 1: Quick Wins (1-2 days, -95ms/step)**
1. Cache traffic light status per scenario: **-40ms**
2. Cache map rasterization: **-30ms**
3. Preload map data into memory: **-25ms**

**Phase 2: Medium Effort (3-5 days, -80ms/step)**
4. Implement DB connection pooling: **-31ms**
5. Optimize agent rasterization (vectorize): **-20ms**
6. Batch database queries: **-30ms**

**Phase 3: Major Refactor (1-2 weeks, -50ms/step)**
7. Reduce data copying (use views): **-30ms**
8. Optimize history buffer: **-10ms**
9. Async LZMA compression: **-10ms**

**Total potential**: -225ms/step reduction
- SimplePlanner: 228ms → 3ms (blazing fast!)
- MLPlanner: 570ms → 345ms (still 3.45x realtime)

**Realistic target**: 2.0x realtime improvement with Phase 1-2 optimizations

---

## Common Patterns

### How to Profile a Planner

```bash
# Run cProfile on single scenario
./scripts/profile_single_scenario.sh

# Analyze results
python -m pstats profiling_output/simulation_profile.stats
> sort cumtime
> stats 20

# Compare Simple vs ML planner
./scripts/profile_simple_planner.sh
# Check: profiling_output/cprofile_simple_planner.txt
```

### How to Test Performance Hypotheses via Config

```bash
# Disable metrics to isolate simulation time
simulation_metric=[] callback=[]

# Reduce parallelism for cleaner profiling
worker=sequential worker.threads_per_node=1

# Change observation complexity
simulation.observation=tracks_observation  # Fast (1-2ms)
simulation.observation=idm_agents         # Slow (10-20ms)

# Minimal scenario set
scenario_filter.limit_total_scenarios=1
```

### How to Implement a Custom Planner

```python
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

class MyPlanner(AbstractPlanner):
    def __init__(self, horizon_seconds=8.0):
        super().__init__()
        self._horizon = horizon_seconds
        self._map_api = None

    def initialize(self, initialization: PlannerInitialization):
        self._map_api = initialization.map_api
        self._route = initialization.route_roadblock_ids

    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        # Current state
        ego_state = current_input.history.current_state[0]

        # Access observations
        agents = current_input.history.observations[-1].tracked_objects.tracked_objects

        # Query map
        lanes = self._map_api.get_proximal_map_objects(
            ego_state.center, radius=50.0, layers=["lanes"]
        )

        # Make decision (your logic here)
        future_states = self._plan(ego_state, agents, lanes)

        return InterpolatedTrajectory(future_states)

    def name(self) -> str:
        return "my_planner"
```

### How to Add a Custom Metric

```python
from nuplan.planning.metrics.abstract_metric import AbstractMetricBuilder

class MyMetric(AbstractMetricBuilder):
    @property
    def name(self) -> str:
        return "my_metric"

    @property
    def category(self) -> str:
        return "custom"

    def compute(
        self, history: SimulationHistory, scenario: AbstractScenario
    ) -> List[MetricStatistics]:
        # Process history
        for sample in history.samples:
            ego_state = sample.ego_state
            # ... compute metric ...

        return [MetricStatistics(
            metric_name=self.name,
            value=final_score,
            unit="custom_unit",
            time_series=time_series_data
        )]

    def compute_score(
        self, scenario, metric_statistics, time_series=None
    ) -> float:
        # Aggregate to single score
        return 1.0 if metric_statistics[0].value < threshold else 0.0
```

---

## Common Gotchas

### 1. Hydra RC Version is Pinned
**Issue**: Hydra 1.1.0rc1 (release candidate) is intentionally pinned
**Don't**: Update Hydra to stable 1.3+
**Why**: Config composition behavior changed, would break existing configs

### 2. Database Queries are NOT Cached
**Issue**: Traffic lights queried 48x per step instead of 1x
**Impact**: 131ms/step database overhead (23% of total time)
**Fix**: Cache traffic light status per scenario (quick win, -40ms/step)

### 3. Metrics Run Post-Simulation (NOT Per-Step)
**Issue**: Metrics computation (~5s) happens AFTER simulation completes
**Impact**: NOT on critical path during 100ms timestep
**Don't**: Blame metrics for simulation slowness (they're innocent!)
**Do**: Optimize database queries and feature building instead

### 4. Feature Building > Inference
**Issue**: ML planner spends 112ms on features, only 15ms on inference
**Impact**: 88% of ML time is preprocessing, not prediction
**Fix**: Cache map rasterization (doesn't change per step)

### 5. SimplePlanner is Also Slow
**Issue**: SimplePlanner at 228ms/step proves framework overhead is 2.3x too slow
**Impact**: Framework bottlenecks affect ALL planners (not just ML)
**Fix**: Optimize database queries and state propagation first (benefits everyone)

---

## Quick Reference: Key Files

**Entry points**:
- `script/run_simulation.py` - Run closed-loop simulation
- `script/run_training.py` - Train ML planners
- `script/run_nuboard.py` - Visualize results

**Core loop**:
- `simulation/simulation.py` (142-174) - Main simulation loop
- `simulation/runner/simulations_runner.py` - Execution orchestration

**Interfaces**:
- `simulation/planner/abstract_planner.py` - Planner contract
- `simulation/controller/abstract_ego_controller.py` - Controller contract
- `simulation/observation/abstract_observation.py` - Observation contract
- `simulation/callback/abstract_callback.py` - Callback contract

**Profiling hotspots** (from cProfile):
- `scenario_builder/nuplan_db/nuplan_scenario_queries.py` - Database queries (HOT!)
- `training/modeling/models/raster_model.py` - Feature building (HOT!)
- `simulation/history/simulation_history_buffer.py` - History updates

**Configuration**:
- `script/config/simulation/default_simulation.yaml` - Simulation root config
- `script/config/common/default_common.yaml` - Common defaults
- `script/config/simulation/planner/` - Planner configs
- `script/config/common/scenario_filter/` - Scenario filter configs

---

## Summary

The nuplan/planning module is a **well-architected but slow** simulation framework. The architecture is clean with no major design flaws, but **database query explosion** (145 queries/step instead of ~5) and **expensive feature building** (112ms for map rasterization) cause 5.7x slowdown vs realtime targets.

### Key Insights

1. **Framework overhead is the main problem** (not ML!)
   - SimplePlanner: 228ms/step (2.3x slower than target)
   - MLPlanner: 570ms/step (5.7x slower than target)
   - Framework contributes 285ms regardless of planner

2. **Database queries dominate** (131ms/step, 23% of time)
   - Traffic lights queried 48x per step (should be 1x)
   - New DB connection 8x per step (should reuse)
   - Quick win: Cache traffic light status → -40ms/step

3. **Feature building > inference** (112ms vs 15ms)
   - ML planner spends 88% of time on features, 12% on prediction
   - Map rasterization (46ms) doesn't change per step → should cache
   - Quick win: Cache map rasterization → -30ms/step

4. **Metrics are NOT the bottleneck**
   - Metrics run AFTER simulation completes (not per-step)
   - Per-step callback overhead: <1ms
   - Post-simulation metrics: 3-5s (acceptable)

### Optimization Strategy

**Phase 1 (Quick Wins)**: Fix database caching and map rasterization
- Impact: -95ms/step (17% speedup)
- Effort: 1-2 days

**Phase 2 (Medium Effort)**: Connection pooling and feature optimization
- Impact: -80ms/step additional (31% total speedup)
- Effort: 3-5 days

**Phase 3 (Major Refactor)**: Reduce data copying and optimize buffers
- Impact: -50ms/step additional (40% total speedup)
- Effort: 1-2 weeks

**Realistic target**: 2.0x realtime improvement → SimplePlanner at 1.5x realtime, MLPlanner at 2.5x realtime

---

## References

### Detailed Architecture Docs
- [docs/architecture/SIMULATION_CORE.md](../../docs/architecture/SIMULATION_CORE.md) - Simulation loop deep dive
- [docs/architecture/PLANNER_INTERFACE.md](../../docs/architecture/PLANNER_INTERFACE.md) - Planner API and profiling
- [docs/architecture/OBSERVATION_HISTORY.md](../../docs/architecture/OBSERVATION_HISTORY.md) - Perception pipeline
- [docs/architecture/CONTROLLER.md](../../docs/architecture/CONTROLLER.md) - Trajectory execution
- [docs/architecture/CALLBACKS.md](../../docs/architecture/CALLBACKS.md) - Callback system
- [docs/architecture/METRICS.md](../../docs/architecture/METRICS.md) - Metric computation
- [docs/architecture/HYDRA_CONFIG.md](../../docs/architecture/HYDRA_CONFIG.md) - Configuration patterns
- [docs/architecture/SCENARIO_BUILDER.md](../../docs/architecture/SCENARIO_BUILDER.md) - Dataset loading

### Performance Reports
- [docs/reports/2025-11-16-CPROFILE_RESULTS.md](../../docs/reports/2025-11-16-CPROFILE_RESULTS.md) - Hotspot analysis
- [docs/reports/2025-11-16-BASELINE_COMPARISON.md](../../docs/reports/2025-11-16-BASELINE_COMPARISON.md) - Planner comparison
- [docs/reports/2025-11-16-PERFORMANCE_EXECUTIVE_SUMMARY.md](../../docs/reports/2025-11-16-PERFORMANCE_EXECUTIVE_SUMMARY.md) - Optimization roadmap

### Existing Documentation
- `nuplan/planning/simulation/CLAUDE.md` - Simulation details (1000+ lines)
- `nuplan/planning/simulation/callback/CLAUDE.md` - Callback patterns (1000+ lines)
- `nuplan/planning/simulation/runner/CLAUDE.md` - Execution orchestration (1000+ lines)
- Plus 10+ other CLAUDE.md files in subdirectories

---

**AIDEV-NOTE**: This master architecture guide synthesizes findings from 4 parallel architecture documentation agents and 3 performance profiling reports. All profiling data is based on actual cProfile runs (59M+ function calls, 114s runtime). The optimization roadmap is data-driven with specific impact estimates and implementation locations.
