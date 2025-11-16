# Simulation Module (Root)

## Purpose & Key Abstractions

The `simulation` root module is the **top-level orchestrator for closed-loop simulation execution**. It manages the complete simulation lifecycle: initialization, state propagation, history tracking, and serialization. The module provides three core classes:

1. **Simulation** - Main simulation runner that coordinates time stepping, state updates, and history logging
2. **SimulationSetup** - Configuration container bundling scenario, planner, controllers, and observations
3. **SimulationLog** - Serialization layer for saving/loading simulation results (msgpack/pickle + lzma compression)

This module bridges the **scenario layer** (what to simulate) and the **planner/controller layers** (how to simulate), orchestrating all components into a closed-loop execution. It is the **entry point for all simulation runs** in nuPlan.

## Package Structure

The simulation package contains **12 submodules** organized by responsibility:

```
nuplan/planning/simulation/
├── simulation.py              # Top-level orchestrator (this file)
├── simulation_setup.py        # Configuration container
├── simulation_log.py          # Serialization layer
│
├── planner/                   # Planning algorithms (simple, ML, etc.)
├── controller/                # Motion control (perfect tracking, log playback, etc.)
├── observation/               # Sensor data processing (tracks, IDM agents, etc.)
├── trajectory/                # Trajectory representations (interpolated, kinematic)
├── history/                   # State history tracking (buffer, samples, queries)
│
├── callback/                  # Lifecycle hooks (metrics, visualization, serialization)
├── main_callback/             # Process-level callbacks (aggregation, S3 upload)
├── runner/                    # Execution orchestration (sequential, Ray, parallel)
│
├── simulation_time_controller/  # Time stepping logic
├── path/                      # Path utilities (convert to linestring, etc.)
├── occupancy_map/             # Spatial reasoning (drivable area, collisions)
├── predictor/                 # Agent prediction (future trajectories)
└── visualization/             # Rendering utilities
```

**Component relationships**:
- **Simulation** coordinates all components via SimulationSetup
- **Runner** manages simulation lifecycle and parallelization
- **Callback** observes simulation events without modifying behavior
- **Planner** computes trajectories from observations
- **Controller** propagates ego state based on trajectory
- **Observation** processes sensor data for planner input
- **History** maintains rolling window of past states

## Architecture & Design Patterns

### Command Pattern
- **Simulation.initialize()**: One-time setup phase returning PlannerInitialization
- **Simulation.get_planner_input()**: Per-step query for current state → PlannerInput
- **Simulation.propagate(trajectory)**: Per-step state update based on planner output
- Enables clear separation of read vs. write operations in simulation loop

### Builder/Factory Pattern
- **SimulationSetup**: Configuration object passed to Simulation constructor
- **validate_planner_setup()**: Validation function ensuring observation type compatibility
- Decouples simulation construction from execution logic

### Memento Pattern
- **SimulationLog**: Captures complete simulation state (scenario, planner, history)
- **Simulation.reset()**: Restores simulation to initial state
- Supports reproducibility and debugging

### Strategy Pattern
- **AbstractCallback**: Pluggable callbacks for metrics, visualization, logging
- **MultiCallback**: Composite pattern for combining multiple callbacks
- Time/ego/observation controllers are strategies injected via SimulationSetup

### Rolling Window Pattern
- **SimulationHistoryBuffer**: Fixed-size buffer (default 2 seconds) for recent states
- Provides planner with recent context without unbounded memory growth
- Buffer size calculated as `(duration / database_interval) + 1`

### Simulation Loop Architecture

**High-level flow** (canonical closed-loop execution):

```
┌─────────────────────────────────────────────────────────────┐
│ Runner (runner/)                                            │
│  ├─ Parallel execution (Ray/Sequential)                    │
│  ├─ Error handling & retry logic                           │
│  └─ Result aggregation                                     │
└───────────────┬─────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│ Simulation (simulation.py)  ◄──── SimulationSetup          │
│  │                                                           │
│  ├─► initialize()                                           │
│  │    ├─ Create history buffer (history/)                  │
│  │    ├─ Initialize observations (observation/)            │
│  │    └─ Return PlannerInitialization                      │
│  │                                                           │
│  └─► LOOP: while is_simulation_running()                    │
│       │                                                      │
│       ├─► get_planner_input()                               │
│       │    ├─ Current iteration (simulation_time_controller/)│
│       │    ├─ History buffer (history/)                     │
│       │    └─ Traffic lights (scenario_builder/)            │
│       │                                                      │
│       ├─► Planner.compute_trajectory()  ◄─── planner/       │
│       │    ├─ Process observations                          │
│       │    ├─ Query map (path/, occupancy_map/)             │
│       │    └─ Predict agents (predictor/)                   │
│       │                                                      │
│       ├─► propagate(trajectory)                             │
│       │    ├─ Add to history (history/)                     │
│       │    ├─ Update ego state (controller/)                │
│       │    ├─ Update observations (observation/)            │
│       │    └─ Advance time (simulation_time_controller/)    │
│       │                                                      │
│       └─► Callback.on_step_end()  ◄─── callback/            │
│            ├─ Compute metrics                               │
│            ├─ Serialize data                                │
│            └─ Render visualization                          │
└─────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│ MainCallback (main_callback/)                               │
│  ├─ Aggregate results across scenarios                     │
│  ├─ Upload to S3 (MetricAggregatorCallback)                │
│  └─ Generate parquet summaries                             │
└─────────────────────────────────────────────────────────────┘
```

**AIDEV-NOTE**: The simulation loop is a tight integration of 8 submodules - understanding this flow is critical for debugging!

## Dependencies

### Documented Internal Dependencies ✅

**All simulation submodules are now documented** (Phase 2 complete):

- **planner/** - Planning algorithm interface (AbstractPlanner, PlannerInitialization, PlannerInput)
- **controller/** - Motion control (AbstractEgoController, PerfectTrackingController, LogPlaybackController)
- **observation/** - Sensor processing (AbstractObservation, TracksObservation, IDMAgents)
- **trajectory/** - Trajectory representations (AbstractTrajectory, InterpolatedTrajectory)
- **history/** - State tracking (SimulationHistory, SimulationHistoryBuffer, SimulationHistorySample)
- **callback/** - Lifecycle hooks (AbstractCallback, MultiCallback, MetricCallback, SerializationCallback)
- **main_callback/** - Process-level callbacks (AbstractMainCallback, MetricAggregatorCallback)
- **runner/** - Execution orchestration (SimulationRunner, MetricRunner, execute_runners())
- **simulation_time_controller/** - Time stepping (AbstractSimulationTimeController, StepSimulationTimeController)
- **path/** - Path utilities (convert_to_interp_traj_from_se2_path)
- **occupancy_map/** - Spatial reasoning (STRTreeOccupancyMapFactory, collision detection)
- **predictor/** - Agent prediction (SimplePredictor for future trajectories)

### External Dependencies

**Scenario Layer**:
- **nuplan/planning/scenario_builder/abstract_scenario.py**: Scenario interface, map API, mission goals

**Script Layer** (undocumented):
- **nuplan/planning/script/run_simulation.py**: CLI entry point that constructs Simulation instances
- **nuplan/planning/script/run_workers.py**: Ray-based parallel execution

**Training Layer** (undocumented):
- **nuplan/planning/training/**: Training pipeline may use simulation logs for data augmentation

## Dependents

### Direct Dependents
- **nuplan/planning/script/run_simulation.py**: Main CLI script for running simulations
- **nuplan/planning/script/run_workers.py**: Ray-based parallel simulation execution
- **nuplan/planning/simulation/runner/**: Simulation runners that wrap this module

### Indirect Dependents
- All planner implementations (simple_planner, ml_planner, etc.) - executed via Simulation
- nuBoard visualization - loads SimulationLog for playback
- Metrics computation - operates on SimulationHistory from completed runs

## Critical Files (Prioritized)

### Root Module Files
1. **simulation.py** (217 lines) - Main Simulation orchestrator class
2. **simulation_setup.py** (60 lines) - Configuration container and validation
3. **simulation_log.py** (109 lines) - Serialization/deserialization for results
4. **__init__.py** (1 line) - Empty init file

### Key Subdirectories (All Documented ✅)

**Core Simulation Components**:
1. **planner/** - Planning algorithms (simple, ML, IDM) - See `planner/CLAUDE.md`
2. **controller/** - Motion control and state propagation - See `controller/CLAUDE.md`
3. **observation/** - Sensor data processing - See `observation/CLAUDE.md`
4. **trajectory/** - Trajectory representations - See `trajectory/CLAUDE.md`
5. **history/** - State history tracking - See `history/CLAUDE.md`

**Execution & Orchestration**:
6. **runner/** - Parallel execution and error handling - See `runner/CLAUDE.md`
7. **callback/** - Per-scenario lifecycle hooks - See `callback/CLAUDE.md`
8. **main_callback/** - Process-level aggregation - See `main_callback/CLAUDE.md`

**Utilities**:
9. **simulation_time_controller/** - Time stepping logic - See `simulation_time_controller/CLAUDE.md`
10. **path/** - Path conversion utilities - See `path/CLAUDE.md`
11. **occupancy_map/** - Spatial reasoning - See `occupancy_map/CLAUDE.md`
12. **predictor/** - Agent future prediction - See `predictor/CLAUDE.md`
13. **visualization/** - Rendering utilities (undocumented)

## Usage Patterns

### Pattern 1: Complete Simulation Lifecycle

```python
from nuplan.planning.simulation.simulation import Simulation
from nuplan.planning.simulation.simulation_setup import SimulationSetup, validate_planner_setup

# Assume setup and planner are configured (typically via Hydra)
validate_planner_setup(setup, planner)

# Create simulation
sim = Simulation(
    simulation_setup=setup,
    callback=callback,
    simulation_history_buffer_duration=2.0  # 2 seconds of history
)

# Initialize
initialization = sim.initialize()
planner.initialize(initialization)

# Run simulation loop
while sim.is_simulation_running():
    # Get current state
    planner_input = sim.get_planner_input()

    # Compute trajectory
    trajectory = planner.compute_planner_trajectory(planner_input)

    # Propagate state
    sim.propagate(trajectory)

    # Callback events
    sim.callback.on_step_end(sim.setup, planner, sim.history.last_sample)

# Access results
history = sim.history
```

**AIDEV-NOTE**: This is the canonical simulation loop - see run_simulation.py for production implementation

### Pattern 2: Simulation Log Serialization

```python
from nuplan.planning.simulation.simulation_log import SimulationLog
from pathlib import Path

# Save simulation results
log = SimulationLog(
    file_path=Path("results/sim_001.msgpack.xz"),
    scenario=scenario,
    planner=planner,
    simulation_history=history
)
log.save_to_file()  # Auto-detects msgpack from suffix

# Load simulation results
data = SimulationLog.load_data(Path("results/sim_001.msgpack.xz"))
# Returns deserialized SimulationLog object
```

**AIDEV-NOTE**: File suffix determines serialization: `.pkl.xz` = pickle, `.msgpack.xz` = msgpack

### Pattern 3: Reset and Reuse

```python
# Run first simulation
sim.initialize()
# ... run simulation ...

# Reset and run with same scenario, different planner
sim.reset()
new_initialization = sim.initialize()
new_planner.initialize(new_initialization)
# ... run simulation again ...
```

**When to use**: Benchmarking multiple planners on same scenario without reconstruction overhead

## Critical Implementation Details

### 1. History Buffer Size Calculation (simulation.py:56-61)

```python
# Add database_interval to ensure minimum buffer duration is satisfied
self._simulation_history_buffer_duration = (
    simulation_history_buffer_duration + self._scenario.database_interval
)

# +1 accounts for duration vs. step count (20 steps @ 0.1s = 1.9s, need 21 for 2.0s)
self._history_buffer_size = (
    int(self._simulation_history_buffer_duration / self._scenario.database_interval) + 1
)
```

**AIDEV-NOTE**: Off-by-one guard ensures buffer duration meets minimum requirement

### 2. Dual Serialization Support (simulation_log.py:46-53)

```python
serialization_type = self.simulation_log_type(self.file_path)

if serialization_type == "pickle":
    self._dump_to_pickle()
elif serialization_type == "msgpack":
    self._dump_to_msgpack()  # pickle → msgpack → lzma
else:
    raise ValueError(f"Unknown option: {serialization_type}")
```

**Why both formats?**
- **Pickle**: Python-native, handles complex objects, slightly slower
- **Msgpack**: Cross-language compatible, faster serialization, smaller files
- Both use lzma compression (preset=0 for speed over compression ratio)

**AIDEV-NOTE**: Msgpack still uses pickle internally (line 38) - it's pickle wrapped in msgpack!

### 3. Observation Type Validation (simulation_setup.py:43-59)

```python
def validate_planner_setup(setup: SimulationSetup, planner: AbstractPlanner) -> None:
    type_observation_planner = planner.observation_type()
    type_observation = setup.observations.observation_type()

    if type_observation_planner != type_observation:
        raise ValueError(
            f"Error: The planner did not receive the right observations:"
            f"{type_observation} != {type_observation_planner} planner."
        )
```

**Why critical?**: Prevents runtime failures where planner expects DetectionsTracks but receives Sensors (or vice versa)

### 4. Simulation State Machine (simulation.py:74-79, 142-153)

```python
def is_simulation_running(self) -> bool:
    return not self._time_controller.reached_end() and self._is_simulation_running

def propagate(self, trajectory):
    if not self.is_simulation_running():
        raise RuntimeError("Simulation is not running, simulation can not be propagated!")
    # ...
    if not next_iteration:
        self._is_simulation_running = False  # Explicit stop
```

**Two stop conditions**:
1. Time controller reaches scenario end
2. Manual stop via `_is_simulation_running` flag

**AIDEV-NOTE**: Check both conditions before calling propagate() to avoid runtime errors

### 5. Buffer Initialization Timing (simulation.py:106-114)

```python
self._history_buffer = SimulationHistoryBuffer.initialize_from_scenario(
    self._history_buffer_size, self._scenario, self._observations.observation_type()
)

self._observations.initialize()  # Must happen after buffer creation

# Add current state AFTER observation initialization
self._history_buffer.append(self._ego_controller.get_state(), self._observations.get_observation())
```

**Order matters**:
1. Create buffer from scenario (loads past states from DB)
2. Initialize observations (sets up sensors)
3. Append current state (t=0) to buffer

**AIDEV-NOTE**: Buffer pre-loads past states from scenario DB, then adds current live state

### 6. Pickle Reconstruction Hook (simulation.py:67-72)

```python
def __reduce__(self) -> Tuple[Type[Simulation], Tuple[Any, ...]]:
    """Hints on how to reconstruct the object when pickling."""
    return self.__class__, (self._setup, self._callback, self._simulation_history_buffer_duration)
```

**Why needed?**: Ray workers serialize Simulation objects for parallel execution. This ensures proper reconstruction without internal state corruption.

### 7. File Type Deduction (simulation_log.py:56-89)

```python
@staticmethod
def simulation_log_type(file_path: Path) -> str:
    """
    Deduce log type from last two suffixes. Must be *.{pkl|msgpack}.xz
    Examples:
    - "/foo/bar.pkl.xz" -> "pickle"
    - "/foo/bar.msgpack.xz" -> "msgpack"
    - "/foo/bar.msgpack.pkl.xz" -> "pickle" (second-to-last takes precedence)
    - "/foo/bar.msgpack" -> Error (must end in .xz)
    """
    if len(file_path.suffixes) < 2:
        raise ValueError(f"Inconclusive file type: {file_path}")

    last_suffix = file_path.suffixes[-1]
    if last_suffix != ".xz":
        raise ValueError(f"Inconclusive file type: {file_path}")

    second_to_last_suffix = file_path.suffixes[-2]
    if second_to_last_suffix not in {".msgpack", ".pkl"}:
        raise ValueError(f"Inconclusive file type: {file_path}")

    return {"msgpack": "msgpack", ".pkl": "pickle"}[second_to_last_suffix]
```

**Edge case**: `foo.msgpack.pkl.xz` → "pickle" (rightmost inner suffix wins)

**AIDEV-NOTE**: Always use double suffix (.pkl.xz or .msgpack.xz) - single suffix fails validation!

## Common Gotchas & Anti-patterns

### Gotcha 1: Forgetting to Initialize Before Stepping
```python
# ❌ WRONG - will raise RuntimeError
sim = Simulation(setup)
planner_input = sim.get_planner_input()  # RuntimeError: Simulation was not initialized!

# ✅ CORRECT
sim = Simulation(setup)
initialization = sim.initialize()
planner.initialize(initialization)
planner_input = sim.get_planner_input()
```

**Why**: History buffer is created in initialize(), get_planner_input() requires it (simulation.py:128-129)

### Gotcha 2: Buffer Duration Shorter Than Database Interval
```python
# ❌ WRONG - will raise ValueError
sim = Simulation(
    simulation_setup=setup,
    simulation_history_buffer_duration=0.05  # scenario.database_interval = 0.1s
)

# ✅ CORRECT
sim = Simulation(
    simulation_setup=setup,
    simulation_history_buffer_duration=2.0  # Must be >= database_interval
)
```

**Why**: Buffer needs at least one database sample to initialize (simulation.py:36-39)

### Gotcha 3: Modifying Setup After Simulation Creation
```python
# ❌ WRONG - changes won't be reflected
sim = Simulation(setup)
setup.ego_controller = new_controller  # Too late, already proxied!

# ✅ CORRECT
setup.ego_controller = new_controller
sim = Simulation(setup)  # Create new simulation
```

**Why**: Simulation proxies controllers in __init__ (simulation.py:45-48), later changes ignored

### Gotcha 4: Reusing Simulation Without Reset
```python
# ❌ WRONG - history accumulates across runs
sim.initialize()
# ... run simulation 1 ...
sim.initialize()  # Reset is called, but confusing pattern
# ... run simulation 2 with history from run 1 ...

# ✅ CORRECT
sim.initialize()
# ... run simulation 1 ...
sim.reset()  # Explicit reset
sim.initialize()
# ... run simulation 2 with clean state ...
```

**Why**: initialize() calls reset() (line 103), but explicit reset is clearer. History is cleared in reset().

### Gotcha 5: Calling propagate() After Simulation Ends
```python
# ❌ WRONG - will raise RuntimeError
while some_condition:  # Not checking is_simulation_running()
    planner_input = sim.get_planner_input()
    trajectory = planner.compute_planner_trajectory(planner_input)
    sim.propagate(trajectory)  # May fail after scenario ends!

# ✅ CORRECT
while sim.is_simulation_running():
    planner_input = sim.get_planner_input()
    trajectory = planner.compute_planner_trajectory(planner_input)
    sim.propagate(trajectory)
```

**Why**: propagate() checks is_simulation_running() and raises RuntimeError if false (simulation.py:152-153)

### Gotcha 6: Wrong File Suffix for Serialization
```python
# ❌ WRONG - will raise ValueError
log = SimulationLog(file_path=Path("results.pkl"), ...)  # Missing .xz
log.save_to_file()

# ❌ WRONG - will raise ValueError
log = SimulationLog(file_path=Path("results.xz"), ...)  # No inner suffix
log.save_to_file()

# ✅ CORRECT
log = SimulationLog(file_path=Path("results.pkl.xz"), ...)  # Double suffix
log.save_to_file()
```

**Why**: simulation_log_type() requires exactly 2 suffixes, last must be .xz (simulation_log.py:72-78)

### Gotcha 7: Accessing history_buffer Property Before Initialize
```python
# ❌ WRONG - will raise RuntimeError
sim = Simulation(setup)
buffer = sim.history_buffer  # RuntimeError: buffer is None, call initialize()

# ✅ CORRECT
sim = Simulation(setup)
sim.initialize()
buffer = sim.history_buffer  # Now safe
```

**Why**: history_buffer property raises if _history_buffer is None (simulation.py:212-215)

### Gotcha 8: Mismatched Observation Types
```python
# ❌ WRONG - will raise ValueError at validation
setup = SimulationSetup(
    observations=IDMAgents(),  # Returns DetectionsTracks
    # ... other params ...
)
planner = SimplePlanner()  # Expects Sensors
validate_planner_setup(setup, planner)  # ValueError!

# ✅ CORRECT
setup = SimulationSetup(
    observations=TracksObservation(),  # Returns DetectionsTracks
    # ... other params ...
)
planner = MLPlanner()  # Also expects DetectionsTracks
validate_planner_setup(setup, planner)  # Passes
```

**Why**: Planner's observation_type() must match setup.observations.observation_type() (simulation_setup.py:54-58)

### Gotcha 9: Ignoring next_iteration Return Value
```python
# ❌ WRONG - may index out of bounds in controllers
next_iteration = self._time_controller.next_iteration()
self._ego_controller.update_state(iteration, next_iteration, ...)  # May be None!

# ✅ CORRECT (from simulation.py:170-174)
next_iteration = self._time_controller.next_iteration()
if next_iteration:
    self._ego_controller.update_state(iteration, next_iteration, ...)
else:
    self._is_simulation_running = False
```

**Why**: next_iteration() returns None when scenario ends, controllers can't handle None iteration

### Gotcha 10: Loading Logs with Wrong Format Assumption
```python
# ❌ WRONG - assumes pickle, may be msgpack
with lzma.open("results.pkl.xz") as f:
    log = pickle.load(f)  # Fails if file is actually msgpack!

# ✅ CORRECT
log = SimulationLog.load_data(Path("results.pkl.xz"))  # Auto-detects format
```

**Why**: load_data() checks file suffix and uses correct deserialization path (simulation_log.py:92-108)

### Gotcha 11: Forgetting +1 in Buffer Size Calculation
```python
# ❌ WRONG - buffer will be 1 sample too short
buffer_size = int(duration / interval)  # 2.0s / 0.1s = 20 samples

# ✅ CORRECT (from simulation.py:61)
buffer_size = int(duration / interval) + 1  # 21 samples for 2.0s duration
```

**Why**: 20 steps at 0.1s intervals span only 1.9s (t=0.0 to t=1.9), need 21 for 2.0s

### Gotcha 12: Using uv run with Ray Workers
```python
# ❌ WRONG - Ray creates minimal worker envs, missing dependencies
uv run python nuplan/planning/script/run_simulation.py worker=ray

# ✅ CORRECT - use direct venv python for Ray scripts
.venv/bin/python nuplan/planning/script/run_simulation.py worker=ray
```

**Why**: Ray's uv integration doesn't propagate extras (torch-cuda11, etc.) to workers. See CLAUDE.md:432-457.

**AIDEV-NOTE**: This is a production lesson learned 2025-11-14, documented in root CLAUDE.md

## Testing & Validation

### Unit Testing
```python
from nuplan.planning.simulation.simulation import Simulation
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario

# Use mock scenario for deterministic testing
mock_scenario = MockAbstractScenario()
setup = SimulationSetup(
    scenario=mock_scenario,
    time_controller=...,
    ego_controller=...,
    observations=...
)

sim = Simulation(setup)
sim.initialize()
assert sim.is_simulation_running()
```

### Integration Testing
```bash
# Test with mini dataset scenario
just test-path nuplan/planning/simulation/test

# Full simulation test with simple planner
just simulate

# Test serialization round-trip
just test-path nuplan/planning/simulation/test/test_simulation_log.py
```

### Performance Profiling
```python
import cProfile
import pstats

# Profile simulation loop
with cProfile.Profile() as pr:
    while sim.is_simulation_running():
        planner_input = sim.get_planner_input()
        trajectory = planner.compute_planner_trajectory(planner_input)
        sim.propagate(trajectory)

stats = pstats.Stats(pr)
stats.sort_stats('cumtime').print_stats(20)
```

**Common bottlenecks**:
- Planner compute_trajectory (80-90% of time)
- History buffer append (observation processing)
- Scenario queries (traffic lights, map objects)

## Cross-References

### All Simulation Submodule Documentation

**Core Components** (see Package Structure above):
- **[planner/CLAUDE.md](planner/CLAUDE.md)** - Planning algorithms, SimplePlanner, MLPlanner, IDMPlanner
- **[controller/CLAUDE.md](controller/CLAUDE.md)** - Motion control, PerfectTrackingController, LogPlaybackController
- **[observation/CLAUDE.md](observation/CLAUDE.md)** - Sensor processing, TracksObservation, IDMAgents, LQRAgents
- **[trajectory/CLAUDE.md](trajectory/CLAUDE.md)** - Trajectory types, InterpolatedTrajectory, kinematic constraints
- **[history/CLAUDE.md](history/CLAUDE.md)** - State tracking, SimulationHistoryBuffer, time-based queries

**Execution & Callbacks**:
- **[runner/CLAUDE.md](runner/CLAUDE.md)** - SimulationRunner, MetricRunner, Ray parallelization
- **[callback/CLAUDE.md](callback/CLAUDE.md)** - 8 lifecycle hooks, MetricCallback, SerializationCallback
- **[main_callback/CLAUDE.md](main_callback/CLAUDE.md)** - Process-level aggregation, S3 upload, competition tracking

**Utilities**:
- **[simulation_time_controller/CLAUDE.md](simulation_time_controller/CLAUDE.md)** - Time stepping, StepSimulationTimeController
- **[path/CLAUDE.md](path/CLAUDE.md)** - Path conversion, interpolation utilities
- **[occupancy_map/CLAUDE.md](occupancy_map/CLAUDE.md)** - Collision detection, drivable area validation
- **[predictor/CLAUDE.md](predictor/CLAUDE.md)** - Agent prediction, SimplePredictor

**Test Documentation**:
- **[callback/test/CLAUDE.md](callback/test/CLAUDE.md)** - Callback testing patterns
- **[main_callback/test/CLAUDE.md](main_callback/test/CLAUDE.md)** - Main callback testing
- **[runner/test/CLAUDE.md](runner/test/CLAUDE.md)** - Runner testing strategies

### Related Project Documentation

**Project-Level**:
- **[CLAUDE.md (root)](../../../CLAUDE.md)** - Complete ML planning workflow, Ray integration gotchas, dataset management
- **[nuplan/planning/CLAUDE.md](../CLAUDE.md)** - Planning module overview (if exists)

**Dependencies**:
- **nuplan/planning/scenario_builder/** - Scenario interface, log data access (undocumented)
- **nuplan/planning/script/run_simulation.py** - Production CLI entry point (undocumented)

**Related Systems**:
- **nuplan/planning/training/modeling/** - ML model training (documented in separate session)
- **nuplan/planning/metrics/** - Metric computation (documented in separate session)

### External Resources
- **nuPlan Devkit Docs**: https://nuplan-devkit.readthedocs.io/en/latest/simulation.html
- **Hydra Config System**: https://hydra.cc/docs/1.1/intro/ (note: nuPlan uses RC version 1.1.0rc1)
- **Ray Documentation**: https://docs.ray.io/en/latest/ (for parallel execution)

## Quick Reference Commands

```bash
# Run simulation with simple planner
just simulate

# Run simulation with trained ML planner
just simulate-ml /path/to/checkpoint.ckpt

# Launch nuBoard to visualize results
just nuboard

# Test simulation module
just test-path nuplan/planning/simulation/test

# Profile simulation performance
uv run python -m cProfile -o sim.prof nuplan/planning/script/run_simulation.py
uv run python -m pstats sim.prof
```

## Changelog

- **2025-11-15**: Phase 2C completion - Enhanced top-level integration documentation
  - Added comprehensive Package Structure section with component relationships
  - Created high-level Simulation Loop Architecture diagram showing submodule integration
  - Updated Dependencies section to reflect all 12 documented submodules
  - Added complete Cross-References linking to all submodule CLAUDE.md files
  - Documented relationship between simulation/, callback/, main_callback/, and runner/ layers

- **2025-11-15**: Initial Tier 2 documentation (Session 3 Batch 2)
  - Covered root simulation module (4 files: simulation.py, simulation_setup.py, simulation_log.py)
  - Added 12 critical gotchas from production usage
  - Documented Ray/uv integration issues
  - Included buffer size calculation edge cases (off-by-one guards)
