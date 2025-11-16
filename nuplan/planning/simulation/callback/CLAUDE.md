# nuplan/planning/simulation/callback/

## Purpose & Scope

**What are callbacks?**

Callbacks are event-driven hooks that execute at specific points in the simulation lifecycle. They enable modular observability, metrics collection, serialization, and visualization without modifying the core simulation loop.

The callback system provides **8 lifecycle hooks** that fire at defined execution points:
- **Metrics collection**: Compute and aggregate performance metrics per scenario
- **Data serialization**: Save simulation history, trajectories, and results to disk
- **Visualization**: Render frames and create videos of scenario execution
- **Performance profiling**: Track timing and resource usage per simulation step
- **Logging**: Capture simulation metadata and execution details

**How callbacks fit in the simulation loop:**

```
SimulationRunner
│
├─► on_initialization_start/end    [Once per runner, wraps planner.initialize()]
│
├─► on_simulation_start/end         [Once per scenario, setup/teardown]
│
└─► FOR EACH TIMESTEP (~200x):
    ├─► on_step_start/end           [Every simulation timestep, ~0.1s]
    └─► on_planner_start/end        [Every planner compute call, nested in step]
```

**CRITICAL**: Callbacks are **observers, not controllers** - they cannot modify simulation behavior, only observe and record state.

## Key Abstractions

### AbstractCallback (Base Interface)

All callbacks implement this protocol defining 8 lifecycle hooks:

```python
from nuplan.planning.simulation.callback.abstract_callback import AbstractCallback

class AbstractCallback:
    """Base interface for simulation lifecycle hooks."""

    def on_initialization_start(self, setup: SimulationSetup) -> None:
        """Called once before all scenarios run (per runner)."""
        pass

    def on_initialization_end(self, setup: SimulationSetup) -> None:
        """Called once after all scenarios complete (per runner)."""
        pass

    def on_simulation_start(self, setup: SimulationSetup) -> None:
        """Called at start of each scenario."""
        pass

    def on_simulation_end(self, setup: SimulationSetup, planner: AbstractPlanner,
                         history: SimulationHistory) -> None:
        """Called at end of each scenario with full history."""
        pass

    def on_step_start(self, setup: SimulationSetup, planner_input: PlannerInput) -> None:
        """Called before each simulation timestep."""
        pass

    def on_step_end(self, setup: SimulationSetup, planner_input: PlannerInput,
                   planner_output: AbstractTrajectory) -> None:
        """Called after each simulation timestep."""
        pass

    def on_planner_start(self, setup: SimulationSetup, planner_input: PlannerInput) -> None:
        """Called before planner computes trajectory."""
        pass

    def on_planner_end(self, setup: SimulationSetup, planner_input: PlannerInput,
                      planner_output: AbstractTrajectory) -> None:
        """Called after planner computes trajectory."""
        pass
```

**AIDEV-NOTE**: All hooks return `None` - callbacks cannot influence simulation execution!

### Built-in Callback Implementations

**1. MetricCallback** (`metric_callback.py`)
- Computes planning metrics (collision, comfort, progress, etc.)
- Aggregates statistics across scenarios
- Saves metric results to disk (parquet/msgpack)
- **Activation**: Only `on_simulation_end` hook is active

**2. TimingCallback** (`timing_callback.py`)
- Profiles execution time per callback hook
- Tracks total simulation runtime
- Validates hook pairs are called (start/end assertions)
- **Output**: TensorBoard scalars for performance analysis

**3. SerializationCallback** (`serialization_callback.py`)
- Saves simulation history to disk (msgpack/pickle)
- Persists trajectories, observations, ego states
- Can save per-step or per-scenario
- **Usage**: nuBoard visualization, experiment analysis

**4. SimulationLogCallback** (`simulation_log_callback.py`)
- Logs metadata per scenario (planner, config, duration)
- Creates CSV/parquet file with run details
- Tracks success/failure status
- **Usage**: Experiment tracking and batch analysis

**5. VisualizationCallback** (`visualization_callback.py`)
- Renders bird's-eye view frames per timestep
- Creates MP4 videos of scenario execution
- Can render multiple agents, map overlays
- **WARNING**: Extremely slow (10-100x overhead), debugging only!

### Composite and Async Patterns

**MultiCallback** (`multi_callback.py`)
- Wraps multiple callbacks as one
- Executes in sequence (order-dependent!)
- Propagates exceptions immediately
- Pattern: `MultiCallback([timing, metrics, serialization])`

**AsyncCallback** (via WorkerPool)
- Executes callbacks asynchronously via WorkerPool
- Returns futures for parallel execution
- Errors surface when awaiting results
- Pattern: Used internally by runner for parallelism

## Architecture

### Event-Driven Execution Flow

```
Simulation Runner (per scenario)
│
├─► on_initialization_start(setup)          [ONCE per runner]
│   └─► Initialize resources, create dirs
│
├─► FOR EACH SCENARIO:
│   │
│   ├─► on_simulation_start(setup)          [Per scenario]
│   │   └─► Load scenario, reset state
│   │
│   ├─► FOR EACH TIMESTEP:
│   │   │
│   │   ├─► on_step_start(setup, input)    [~200 times/scenario]
│   │   │
│   │   ├─► on_planner_start(setup, input)
│   │   ├─► planner.compute_trajectory()   [User code]
│   │   ├─► on_planner_end(setup, input, output)
│   │   │
│   │   └─► on_step_end(setup, input, output)
│   │
│   └─► on_simulation_end(setup, planner, history)
│       └─► Compute metrics, save results
│
└─► on_initialization_end(setup)            [ONCE per runner]
    └─► Cleanup, aggregate statistics
```

**Execution guarantees**:
- Hooks are **synchronous** by default (blocking)
- MultiCallback executes **sequentially** (not parallel)
- AsyncCallback returns **futures** (must await or results lost)
- Exceptions in callbacks **crash the simulation** (no safety net)

**AIDEV-NOTE**: `on_step_start/end` runs ~200x per scenario - keep lightweight (< 1ms each)!

### Hook Execution Order Dependencies

**Critical ordering**:
1. **TimingCallback FIRST**: Must wrap other callbacks to measure their overhead
2. **MetricCallback BEFORE SerializationCallback**: Metrics may need history access before serialization
3. **VisualizationCallback LAST**: Slowest, don't block critical callbacks

**Typical configuration**:
```yaml
# config/callback/default_callback.yaml
callbacks:
  - timing_callback          # First: Profile everything
  - metric_callback          # Second: Compute metrics
  - serialization_callback   # Third: Save history
  # - visualization_callback # Last: Render (optional, VERY slow)
```

**AIDEV-NOTE**: Hydra list order determines execution order in MultiCallback!

### Async Execution with WorkerPool

The runner uses `WorkerPool` for parallel scenario execution:

```python
from nuplan.planning.simulation.runner.executor import WorkerPool

# In runner.py
with WorkerPool(num_workers=8) as pool:
    # Submit callback tasks
    futures = [
        pool.submit(callback.on_simulation_end, setup, planner, history)
        for setup, planner, history in scenarios
    ]

    # Wait for completion
    for future in futures:
        try:
            future.result()  # Blocks until done, raises if exception
        except Exception as e:
            # Exception from callback crashes here
            raise
```

**Pickling requirements**: Callbacks must be serializable (no open file handles, lambdas, local classes).

## Dependencies

### Imports From (Callback Dependencies)

**Core simulation types**:
- `nuplan.planning.simulation.history.simulation_history` - SimulationHistory, SimulationHistorySample
- `nuplan.planning.simulation.planner.abstract_planner` - AbstractPlanner
- `nuplan.planning.simulation.observation.observation_type` - Observation
- `nuplan.planning.simulation.trajectory.abstract_trajectory` - AbstractTrajectory
- `nuplan.planning.simulation.simulation_setup` - SimulationSetup

**Metrics computation**:
- `nuplan.planning.metrics.abstract_metric` - AbstractMetric
- `nuplan.planning.metrics.metric_engine` - MetricsEngine
- `nuplan.planning.metrics.metric_result` - MetricStatistics

**Visualization rendering**:
- `nuplan.planning.simulation.visualization.abstract_visualization` - AbstractVisualization

**Worker parallelism**:
- `nuplan.planning.simulation.runner.executor` - WorkerPool

### Used By (Callback Consumers)

**Simulation runners**:
- `nuplan.planning.simulation.runner.abstract_runner` - Invokes callbacks per scenario
- `nuplan.planning.simulation.runner.runner` - Main simulation orchestrator

**Main callback builders**:
- `nuplan.planning.simulation.main_callback.metric_callback` - Creates MetricCallback with metrics
- `nuplan.planning.simulation.main_callback.multi_callback_builder` - Composes MultiCallback from config

**CRITICAL**: `MetricCallback` and `SimulationLogCallback` are **NOT** in `build_simulation_callbacks()` - they're built separately in main entry points!

## Common Usage Patterns

### 1. Implementing a Custom Callback

```python
from nuplan.planning.simulation.callback.abstract_callback import AbstractCallback
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.simulation.history.simulation_history import SimulationHistory
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
import logging

logger = logging.getLogger(__name__)

class CustomDebugCallback(AbstractCallback):
    """Logs ego velocity at each timestep for debugging."""

    def __init__(self, velocity_threshold: float = 10.0):
        """
        :param velocity_threshold: Log warning if velocity exceeds this (m/s)
        """
        self.velocity_threshold = velocity_threshold
        self._step_count = 0

    def on_simulation_start(self, setup: SimulationSetup) -> None:
        """Reset counter per scenario."""
        self._step_count = 0
        logger.info(f"Starting scenario: {setup.scenario.scenario_name}")

    def on_step_end(self, setup, planner_input, planner_output) -> None:
        """Log ego velocity after each step."""
        self._step_count += 1
        ego_state = planner_input.history.ego_states[-1]
        velocity = ego_state.dynamic_car_state.speed  # m/s

        if velocity > self.velocity_threshold:
            logger.warning(
                f"Step {self._step_count}: High velocity {velocity:.2f} m/s "
                f"at position ({ego_state.center.x:.1f}, {ego_state.center.y:.1f})"
            )

    def on_simulation_end(self, setup, planner, history) -> None:
        """Log final statistics."""
        final_ego = history.ego_states[-1]
        logger.info(
            f"Scenario complete: {self._step_count} steps, "
            f"final velocity {final_ego.dynamic_car_state.speed:.2f} m/s"
        )

    # Unused hooks (can be omitted or pass)
    def on_initialization_start(self, setup): pass
    def on_initialization_end(self, setup): pass
    def on_step_start(self, setup, planner_input): pass
    def on_planner_start(self, setup, planner_input): pass
    def on_planner_end(self, setup, planner_input, planner_output): pass
```

**Usage**:
```python
from nuplan.planning.simulation.callback.multi_callback import MultiCallback

callbacks = MultiCallback([
    CustomDebugCallback(velocity_threshold=15.0),
    # ... other callbacks
])
```

### 2. Registering Callbacks via Hydra Config

**Create config file**: `config/callback/custom_debug_callback.yaml`

```yaml
_target_: path.to.custom_debug_callback.CustomDebugCallback

# Constructor arguments
velocity_threshold: 15.0
```

**Compose in simulation config**: `config/simulation/my_simulation.yaml`

```yaml
defaults:
  - base_simulation
  - override /callback:
      - timing_callback
      - custom_debug_callback  # Your callback
      - metric_callback
      - serialization_callback

# Callback-specific overrides
callback:
  custom_debug_callback:
    velocity_threshold: 20.0  # Override default
```

**Run simulation**:
```bash
uv run python nuplan/planning/script/run_simulation.py \
    +simulation=my_simulation \
    planner=simple_planner
```

**AIDEV-NOTE**: Callback order in defaults list determines execution order!

### 3. Async Execution Pattern with WorkerPool

**Scenario**: Save simulation history to disk asynchronously (don't block main thread)

```python
from nuplan.planning.simulation.runner.executor import WorkerPool
from nuplan.planning.simulation.callback.serialization_callback import SerializationCallback
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class AsyncSerializationCallback(AbstractCallback):
    """Saves history asynchronously to avoid blocking simulation loop."""

    def __init__(self, output_dir: str, num_workers: int = 4):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._worker_pool = WorkerPool(num_workers=num_workers)
        self._pending_futures = []

    def on_simulation_end(self, setup, planner, history) -> None:
        """Submit save task to worker pool (non-blocking)."""
        scenario_name = setup.scenario.scenario_name
        output_path = self.output_dir / f"{scenario_name}.msgpack"

        # Submit async task
        future = self._worker_pool.submit(
            self._save_history,
            history,
            output_path
        )
        self._pending_futures.append((scenario_name, future))
        logger.info(f"Queued save for {scenario_name}")

    def on_initialization_end(self, setup) -> None:
        """Wait for all pending saves to complete."""
        logger.info(f"Waiting for {len(self._pending_futures)} saves to complete...")

        for scenario_name, future in self._pending_futures:
            try:
                future.result()  # Blocks until done
                logger.info(f"Saved {scenario_name}")
            except Exception as e:
                logger.error(f"Failed to save {scenario_name}: {e}")

        self._worker_pool.shutdown()  # Cleanup
        logger.info("All saves complete")

    @staticmethod
    def _save_history(history: SimulationHistory, output_path: Path) -> None:
        """Worker task: Serialize history to msgpack."""
        import msgpack

        # Convert history to dict (must be picklable!)
        data = {
            'ego_states': [state.serialize() for state in history.ego_states],
            'observations': [obs.serialize() for obs in history.observations],
            # ... other fields
        }

        with open(output_path, 'wb') as f:
            msgpack.pack(data, f)

    # Unused hooks
    def on_initialization_start(self, setup): pass
    def on_simulation_start(self, setup): pass
    def on_step_start(self, setup, planner_input): pass
    def on_step_end(self, setup, planner_input, planner_output): pass
    def on_planner_start(self, setup, planner_input): pass
    def on_planner_end(self, setup, planner_input, planner_output): pass
```

**CRITICAL**: Worker tasks must be picklable - no lambdas, open files, or local class references!

### 4. Multi-Callback Composition Pattern

**Scenario**: Combine timing, metrics, and serialization with controlled ordering

```python
from nuplan.planning.simulation.callback.multi_callback import MultiCallback
from nuplan.planning.simulation.callback.timing_callback import TimingCallback
from nuplan.planning.simulation.callback.metric_callback import MetricCallback
from nuplan.planning.simulation.callback.serialization_callback import SerializationCallback
from pathlib import Path

def build_evaluation_callbacks(
    output_dir: str,
    metrics: List[AbstractMetric],
    enable_timing: bool = True,
    enable_serialization: bool = True
) -> MultiCallback:
    """Build callback stack for evaluation runs.

    Execution order:
    1. Timing (profiles everything else)
    2. Metrics (compute before serialization)
    3. Serialization (save results to disk)
    """
    callbacks = []

    # Always first: Timing callback
    if enable_timing:
        callbacks.append(TimingCallback())

    # Always second: Metrics callback
    callbacks.append(MetricCallback(
        metric_engine=MetricsEngine(metrics=metrics),
        output_directory=Path(output_dir) / 'metrics'
    ))

    # Optional: Serialization callback
    if enable_serialization:
        callbacks.append(SerializationCallback(
            output_directory=Path(output_dir) / 'history',
            serialize_into_single_file=False,  # Per-scenario files
            serialization_type='msgpack'  # Faster than pickle
        ))

    return MultiCallback(callbacks)

# Usage
callbacks = build_evaluation_callbacks(
    output_dir='/tmp/eval_results',
    metrics=[collision_metric, comfort_metric, progress_metric],
    enable_timing=True,
    enable_serialization=True
)
```

**AIDEV-NOTE**: MultiCallback executes sequentially, not in parallel - order matters!

## Gotchas & Pitfalls

### 1. Hook Execution Order Dependencies

**Problem**: Callbacks depend on execution order but MultiCallback doesn't enforce it.

**Example failure**:
```python
# BAD: Serialization before metrics
callbacks = MultiCallback([
    SerializationCallback(...),  # Saves history first
    MetricCallback(...),         # Computes metrics second
])

# on_simulation_end order:
# 1. SerializationCallback saves history (missing metrics!)
# 2. MetricCallback computes metrics (too late)
```

**Solution**: Always put TimingCallback first, MetricCallback before serialization:
```python
# GOOD: Proper ordering
callbacks = MultiCallback([
    TimingCallback(),            # First: Profile everything
    MetricCallback(...),         # Second: Compute metrics
    SerializationCallback(...),  # Third: Save with metrics
])
```

**AIDEV-NOTE**: Document callback order requirements in config comments!

### 2. State Mutation Risks in on_simulation_end

**Problem**: Multiple callbacks receive same `history` object - mutations propagate!

**Example failure**:
```python
class BadMetricCallback(AbstractCallback):
    def on_simulation_end(self, setup, planner, history):
        # DANGER: Mutates shared history object
        history.ego_states = self._filter_invalid_states(history.ego_states)
        # Now SerializationCallback saves corrupted history!
```

**Solution**: Never mutate shared objects - copy if modification needed:
```python
class GoodMetricCallback(AbstractCallback):
    def on_simulation_end(self, setup, planner, history):
        # SAFE: Work on copy
        valid_states = list(history.ego_states)  # Copy
        filtered = self._filter_invalid_states(valid_states)
        metrics = self._compute_metrics(filtered)
        # Original history unchanged
```

**AIDEV-NOTE**: Treat all callback arguments as read-only unless explicitly designed for mutation!

### 3. File Handle Leaks with Async Callbacks

**Problem**: WorkerPool pickles callbacks - open file handles cause errors or leaks.

**Example failure**:
```python
class LeakyCallback(AbstractCallback):
    def __init__(self, log_file: str):
        self.log_handle = open(log_file, 'w')  # DANGER: Open file

    def on_step_end(self, setup, planner_input, planner_output):
        # WorkerPool tries to pickle self -> pickle.PicklingError!
        self.log_handle.write(f"Step complete\n")
```

**Solution**: Open files lazily and close in `__del__`:
```python
class SafeCallback(AbstractCallback):
    def __init__(self, log_file: str):
        self.log_file = log_file
        self._log_handle = None

    def on_step_end(self, setup, planner_input, planner_output):
        if self._log_handle is None:
            self._log_handle = open(self.log_file, 'w')
        self._log_handle.write(f"Step complete\n")

    def __del__(self):
        if self._log_handle:
            self._log_handle.close()
```

**Better solution**: Use context managers per hook:
```python
def on_simulation_end(self, setup, planner, history):
    with open(self.log_file, 'a') as f:
        f.write(f"Scenario {setup.scenario.scenario_name} complete\n")
```

### 4. MetricCallback/SimulationLogCallback Not in build_simulation_callbacks

**Problem**: `build_simulation_callbacks()` only creates Timing, Serialization, Visualization.

**Example failure**:
```python
# In run_simulation.py
from nuplan.planning.simulation.main_callback.multi_callback_builder import build_simulation_callbacks

callbacks = build_simulation_callbacks(cfg)  # Only 3 callbacks!
# MetricCallback missing -> no metrics computed!
```

**Root cause**: Metrics require `metric_engine` argument, built separately in main scripts.

**Solution**: Build MetricCallback explicitly:
```python
from nuplan.planning.simulation.main_callback.metric_callback import build_metrics_callback

# Build metrics separately
metric_callback = build_metrics_callback(
    metric_engine=MetricsEngine(metrics=metrics),
    output_directory=Path(output_dir) / 'metrics'
)

# Build other callbacks
other_callbacks = build_simulation_callbacks(cfg)

# Combine
all_callbacks = MultiCallback([metric_callback] + other_callbacks.callbacks)
```

**AIDEV-NOTE**: See `run_simulation.py` for reference pattern!

### 5. Callback Exceptions Crash Entire Simulation

**Problem**: No try/catch around callback invocations - exceptions propagate immediately.

**Example failure**:
```python
class BuggyCallback(AbstractCallback):
    def on_step_end(self, setup, planner_input, planner_output):
        result = 1 / 0  # ZeroDivisionError!
        # Simulation crashes, all progress lost!
```

**Impact**: 100 scenarios complete, 1 callback error -> lose all 100 results!

**Solution**: Wrap risky code in try/except internally:
```python
class RobustCallback(AbstractCallback):
    def on_step_end(self, setup, planner_input, planner_output):
        try:
            self._process_step(planner_input, planner_output)
        except Exception as e:
            logger.error(f"Callback error in step: {e}", exc_info=True)
            # Continue simulation, just skip this callback
```

**AIDEV-TODO**: Consider patching MultiCallback to add optional exception isolation!

### 6. Timing Assertion Failures from Unpaired Hooks

**Problem**: TimingCallback expects start/end pairs - missing call triggers assertion.

**Example failure**:
```python
# In custom runner (buggy)
callback.on_planner_start(setup, planner_input)
trajectory = planner.compute_trajectory(planner_input)
# FORGOT: callback.on_planner_end(...)
# TimingCallback._planner_start assertion fails on next call!
```

**Error**:
```python
AssertionError: on_planner_end called without on_planner_start
```

**Solution**: Always call hooks in pairs, use try/finally:
```python
callback.on_planner_start(setup, planner_input)
try:
    trajectory = planner.compute_trajectory(planner_input)
finally:
    callback.on_planner_end(setup, planner_input, trajectory)
```

**AIDEV-NOTE**: Runner code already does this correctly - only issue if writing custom runner!

### 7. WorkerPool Pickling Requirements

**Problem**: Callbacks submitted to WorkerPool must be serializable (pickle compatible).

**Unpicklable patterns**:
- Open file handles
- Lambda functions
- Local/nested class definitions
- Thread locks
- Database connections

**Example failure**:
```python
class UnpicklableCallback(AbstractCallback):
    def __init__(self):
        self.db_conn = sqlite3.connect('metrics.db')  # DANGER: Connection object
        self.process_fn = lambda x: x * 2  # DANGER: Lambda

# WorkerPool.submit(callback.on_simulation_end, ...) -> PicklingError!
```

**Solution**: Use picklable alternatives:
```python
class PicklableCallback(AbstractCallback):
    def __init__(self, db_path: str):
        self.db_path = db_path  # Store path, not connection
        # No lambda - use named function instead

    def on_simulation_end(self, setup, planner, history):
        # Open connection in worker process
        conn = sqlite3.connect(self.db_path)
        try:
            self._save_metrics(conn, history)
        finally:
            conn.close()

    @staticmethod
    def _process_fn(x):  # Named function, not lambda
        return x * 2
```

**AIDEV-NOTE**: Test pickleability: `pickle.dumps(callback)` before submitting to WorkerPool!

### 8. Futures Not Awaited = Silent Failures

**Problem**: AsyncCallback returns futures - if not awaited, errors disappear.

**Example failure**:
```python
# In runner (buggy)
future = worker_pool.submit(callback.on_simulation_end, setup, planner, history)
# FORGOT: future.result()
# Callback crashes but error never surfaces!
```

**Solution**: Always await futures before continuing:
```python
future = worker_pool.submit(callback.on_simulation_end, setup, planner, history)
try:
    result = future.result()  # Blocks until done, raises if error
except Exception as e:
    logger.error(f"Callback failed: {e}", exc_info=True)
    raise
```

**Best practice**: Use context manager to ensure cleanup:
```python
with WorkerPool(num_workers=4) as pool:
    futures = [pool.submit(callback.on_simulation_end, ...) for ... in scenarios]

    # Wait for all
    for future in futures:
        future.result()  # Raises if any failed
# Pool auto-shutdown on exit
```

### 9. Directory Creation Race Conditions in Parallel Runs

**Problem**: Multiple workers create same directory simultaneously -> FileExistsError.

**Example failure**:
```python
class RacyCallback(AbstractCallback):
    def on_simulation_start(self, setup):
        output_dir = Path('/tmp/results') / setup.scenario.scenario_name
        output_dir.mkdir(parents=True)  # DANGER: Race condition!
        # Multiple workers -> FileExistsError (sometimes)
```

**Solution**: Use `exist_ok=True` or create directories once in initialization:
```python
class SafeCallback(AbstractCallback):
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)

    def on_initialization_start(self, setup):
        # Create base directory once (single-threaded)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def on_simulation_start(self, setup):
        scenario_dir = self.base_dir / setup.scenario.scenario_name
        scenario_dir.mkdir(exist_ok=True)  # Safe: exist_ok=True
```

**AIDEV-NOTE**: Always use `exist_ok=True` for directories created in parallel hooks!

### 10. Serialization Format File Size Explosion

**Problem**: Pickle format saves redundant metadata - msgpack 2-5x smaller.

**Example sizes** (200-step scenario):
- Pickle: ~150 MB (includes Python metadata, type info)
- Msgpack: ~40 MB (binary only, no metadata)
- JSON: ~200 MB (human-readable, slow)

**Solution**: Use msgpack for production, pickle only for debugging:
```python
SerializationCallback(
    output_directory=output_dir,
    serialization_type='msgpack',  # Not 'pickle'
    serialize_into_single_file=False  # Per-scenario files for parallelism
)
```

**Trade-offs**:
- Msgpack: Fast, compact, requires schema knowledge to deserialize
- Pickle: Slow, large, easy to deserialize (preserves Python types)

**AIDEV-NOTE**: For 1000s of scenarios, msgpack saves 100s of GBs!

### 11. Visualization Callback Blocking Performance

**Problem**: Rendering frames is slow (~0.5-2s per frame) - blocks simulation loop.

**Example impact**:
```python
# Without visualization: 200 steps * 0.01s = 2s per scenario
# With visualization: 200 steps * 0.5s = 100s per scenario (50x slower!)
```

**Solution**: Disable visualization for large batches, use only for debugging:
```yaml
# config/callback/default_callback.yaml
callbacks:
  - timing_callback
  - metric_callback
  - serialization_callback
  # - visualization_callback  # Comment out for batch runs
```

**Alternative**: Render offline from saved history:
```bash
# 1. Run simulation without visualization (fast)
just simulate planner=ml_planner +callback=no_viz

# 2. Render videos offline from saved history (parallel)
uv run python nuplan/planning/script/render_history.py \
    history_dir=/tmp/results/history \
    output_dir=/tmp/results/videos \
    num_workers=8
```

**AIDEV-NOTE**: VisualizationCallback should never run in production evals!

### 12. AsyncCallback Error Handling Updates Report Post-Facto

**Problem**: Errors in async tasks surface when awaiting future, not when submitted.

**Example timeline**:
```python
# T=0: Submit 100 scenario tasks
futures = [pool.submit(callback.on_simulation_end, ...) for _ in range(100)]

# T=0-60s: All scenarios running in parallel

# T=60s: First 99 succeed, #100 crashes
# T=61s: future[99].result() -> Exception raised HERE (not at submission!)
```

**Impact**: Wasted 60s of computation before discovering error in scenario #100.

**Solution**: Check futures periodically with timeout:
```python
import concurrent.futures

futures = [pool.submit(callback.on_simulation_end, ...) for ...]

# Check every 5 seconds for early failures
for i, future in enumerate(futures):
    try:
        result = future.result(timeout=5.0)  # Wait max 5s
    except concurrent.futures.TimeoutError:
        continue  # Still running, check next
    except Exception as e:
        logger.error(f"Scenario {i} failed early: {e}")
        # Cancel remaining futures?
        raise
```

**AIDEV-NOTE**: Fast-fail pattern prevents wasted computation on doomed batches!

### 13. Memory Footprint from Redundant History Storage

**Problem**: Each callback stores reference to SimulationHistory - huge memory waste.

**Example**:
```python
class MemoryHogCallback(AbstractCallback):
    def __init__(self):
        self.all_histories = []  # DANGER: Accumulates memory

    def on_simulation_end(self, setup, planner, history):
        self.all_histories.append(history)  # 100 scenarios * 50 MB = 5 GB!
```

**Solution**: Extract only needed data, discard history reference:
```python
class MemoryEfficientCallback(AbstractCallback):
    def __init__(self):
        self.scenario_summaries = []  # Lightweight

    def on_simulation_end(self, setup, planner, history):
        # Extract only what you need
        summary = {
            'name': setup.scenario.scenario_name,
            'duration': len(history.ego_states) * 0.1,
            'final_position': history.ego_states[-1].center,
        }
        self.scenario_summaries.append(summary)
        # history garbage-collected after this function returns
```

**AIDEV-NOTE**: 1000 scenarios * 50 MB history = 50 GB RAM if stored naively!

### 14. Cached Scenario Data in on_simulation_start Pattern

**Problem**: Loading heavy scenario data per simulation is slow - cache in initialization.

**Anti-pattern**:
```python
class SlowCallback(AbstractCallback):
    def on_simulation_start(self, setup):
        # SLOW: Loads map from disk every scenario
        self.map_api = MapFactory.build_map(setup.map_name)
```

**Better pattern**:
```python
class FastCallback(AbstractCallback):
    def __init__(self):
        self._map_cache = {}

    def on_simulation_start(self, setup):
        # FAST: Load map once, cache for reuse
        if setup.map_name not in self._map_cache:
            self._map_cache[setup.map_name] = MapFactory.build_map(setup.map_name)

        self.map_api = self._map_cache[setup.map_name]
```

**AIDEV-NOTE**: Map loading takes ~2-5s - cache saves 100s of seconds over 1000 scenarios!

### 15. Hook Budget Timing Constraints

**Problem**: Callbacks in on_step_* must complete within timestep budget (0.1s).

**Example failure**:
```python
class ExpensiveCallback(AbstractCallback):
    def on_step_end(self, setup, planner_input, planner_output):
        # SLOW: 0.5s per step
        self._render_frame(planner_input)
        self._compute_complex_metric(planner_output)
        # Simulation can't keep real-time (0.1s budget exceeded!)
```

**Impact**: Simulation takes 5x longer than real-time (0.5s per 0.1s step).

**Solution**: Offload expensive work to on_simulation_end:
```python
class EfficientCallback(AbstractCallback):
    def __init__(self):
        self._step_data = []

    def on_step_end(self, setup, planner_input, planner_output):
        # FAST: Just store data (~0.001s)
        self._step_data.append({
            'input': planner_input,
            'output': planner_output,
        })

    def on_simulation_end(self, setup, planner, history):
        # SLOW: Process all steps here (not time-critical)
        for data in self._step_data:
            self._render_frame(data['input'])
            self._compute_complex_metric(data['output'])
        self._step_data.clear()
```

**AIDEV-NOTE**: on_step_* hooks run 200x per scenario - keep under 1ms each!

## Cross-References

### Related Modules

**Simulation execution** (`nuplan/planning/simulation/`):
- `runner/` - Orchestrates simulation loop, invokes callbacks
- `simulation_setup.py` - Setup context passed to all callbacks
- `history/` - SimulationHistory objects passed to on_simulation_end

**Metrics computation** (`nuplan/planning/training/modeling/metrics/`):
- `abstract_metric.py` - Metric interface used by MetricCallback
- `metric_engine.py` - Aggregates multiple metrics
- `planning_metrics/` - Collision, comfort, progress metrics

**Main callback builders** (`nuplan/planning/simulation/main_callback/`):
- `metric_callback.py` - Builds MetricCallback with metrics
- `multi_callback_builder.py` - Composes MultiCallback from Hydra config

**Visualization** (`nuplan/planning/simulation/visualization/`):
- `abstract_visualization.py` - Renderer used by VisualizationCallback

### See Also

- **Tutorial**: `tutorials/nuplan_simulation.ipynb` - Callback configuration examples
- **Entry point**: `nuplan/planning/script/run_simulation.py` - Callback orchestration
- **Config**: `config/callback/` - Hydra callback configurations
- **Tests**: `nuplan/planning/simulation/callback/test/` - Callback unit tests

**AIDEV-NOTE**: For custom callbacks, start with `test/test_timing_callback.py` as template!
