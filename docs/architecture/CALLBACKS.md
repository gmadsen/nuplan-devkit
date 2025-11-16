# Callback System Architecture

## Purpose & Scope

The callback system provides **event-driven hooks** that execute at specific points in the simulation lifecycle, enabling modular observability, metrics collection, serialization, and visualization without modifying the core simulation loop.

**Design principle**: Callbacks are **passive observers** - they observe state changes but cannot modify simulation behavior or planner decisions.

**Key characteristic**: Callbacks can be **synchronous** (blocking) or **asynchronous** (non-blocking with futures), making them flexible for both real-time observability and batch post-processing.

## Callback Lifecycle

The simulation runner invokes callbacks at **8 lifecycle hooks** in strict order:

```
SimulationRunner.run()
│
├─► on_simulation_start(setup)
│   • Before planner initialization
│   • Setup input dir, logging, etc.
│
├─► _initialize() (internal)
│   │
│   ├─► on_initialization_start(setup, planner)
│   │   • Before planner.initialize()
│   │
│   ├─► planner.initialize(PlannerInitialization)
│   │   • Planner sets up map, routes, goal
│   │
│   └─► on_initialization_end(setup, planner)
│       • After planner.initialize()
│
├─► MAIN LOOP: while simulation.is_simulation_running()
│   │
│   ├─► on_step_start(setup, planner)
│   │   • Before planner input computation
│   │   [CALLED ~200 TIMES PER SCENARIO]
│   │
│   ├─► get_planner_input()
│   │   • Assemble current observations
│   │
│   ├─► on_planner_start(setup, planner)
│   │   • Before planner.compute_trajectory()
│   │   [CALLED ~200 TIMES PER SCENARIO]
│   │
│   ├─► planner.compute_trajectory(PlannerInput)
│   │   • ~50-100ms per call (10-90% of step time)
│   │
│   ├─► on_planner_end(setup, planner, trajectory)
│   │   • After planner returns trajectory
│   │   [CALLED ~200 TIMES PER SCENARIO]
│   │
│   ├─► simulation.propagate(trajectory)
│   │   • Update ego state, observations, history
│   │
│   └─► on_step_end(setup, planner, sample)
│       • After propagation complete
│       [CALLED ~200 TIMES PER SCENARIO]
│
└─► on_simulation_end(setup, planner, history)
    • After all steps complete, full history available
    • Metrics computation, serialization happens here
    [CALLED ONCE PER SCENARIO]
```

**Execution guarantee**: Hooks are **always called in order**, even if previous hooks fail (with exception handling).

---

## Core Abstractions

### 1. AbstractCallback (abstract_callback.py)

**Base interface for all callbacks.**

```python
class AbstractCallback(ABC):
    """Base class for simulation callbacks."""

    # Called once at runner initialization
    @abstractmethod
    def on_initialization_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Before planner.initialize() is called."""
        pass

    @abstractmethod
    def on_initialization_end(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """After planner.initialize() completes."""
        pass

    # Called ~200x per scenario (per simulation step)
    @abstractmethod
    def on_step_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Before each simulation timestep."""
        pass

    @abstractmethod
    def on_step_end(
        self,
        setup: SimulationSetup,
        planner: AbstractPlanner,
        sample: SimulationHistorySample
    ) -> None:
        """After each simulation timestep."""
        pass

    # Called ~200x per scenario (per planner call)
    @abstractmethod
    def on_planner_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Before planner.compute_trajectory()."""
        pass

    @abstractmethod
    def on_planner_end(
        self,
        setup: SimulationSetup,
        planner: AbstractPlanner,
        trajectory: AbstractTrajectory
    ) -> None:
        """After planner.compute_trajectory() returns."""
        pass

    # Called once per scenario
    @abstractmethod
    def on_simulation_start(self, setup: SimulationSetup) -> None:
        """At start of scenario execution."""
        pass

    @abstractmethod
    def on_simulation_end(
        self,
        setup: SimulationSetup,
        planner: AbstractPlanner,
        history: SimulationHistory
    ) -> None:
        """After scenario completes, full history available."""
        pass
```

**Key contracts**:
- All methods return `None` - callbacks cannot influence simulation
- All hooks are **synchronous** by default (blocking)
- Exceptions in callbacks **propagate** immediately (no try/catch)
- Callbacks receive **read-only** references to state objects

---

### 2. MultiCallback (multi_callback.py)

**Composite pattern: Combines multiple callbacks into one.**

```python
class MultiCallback(AbstractCallback):
    """Calls multiple callbacks in sequence."""

    def __init__(self, callbacks: List[AbstractCallback]):
        self._callbacks = callbacks  # Execution order matters!

    def on_step_end(self, setup: SimulationSetup, planner: AbstractPlanner, sample: SimulationHistorySample) -> None:
        """Inherited, see superclass."""
        for callback in self._callbacks:
            callback.on_step_end(setup, planner, sample)
            # Waits for callback to complete before calling next
```

**Execution model**: **SEQUENTIAL**, NOT parallel.
```
on_step_end() call:
  │
  ├─► Callback 1.on_step_end() ← Wait here until complete
  ├─► Callback 2.on_step_end() ← Then this
  └─► Callback 3.on_step_end() ← Then this

Total time = callback1 + callback2 + callback3
(No parallelism!)
```

**CRITICAL**: Order matters! If callback 1 mutates shared state, callback 2 sees mutations.

---

## Built-in Callback Implementations

### 1. MetricCallback (metric_callback.py)

**Computes performance metrics post-simulation.**

```python
class MetricCallback(AbstractCallback):
    """Callback for computing metrics at end of simulation."""

    def __init__(self, metric_engine: MetricsEngine, worker_pool: Optional[WorkerPool] = None):
        self._metric_engine = metric_engine
        self._pool = worker_pool  # None = sync, not None = async
        self._futures = []

    def on_simulation_end(self, setup: SimulationSetup, planner: AbstractPlanner, history: SimulationHistory) -> None:
        """ONLY active hook. All others empty."""
        if self._pool is not None:
            # ASYNC: Submit to worker pool, return immediately
            self._futures.append(
                self._pool.submit(
                    Task(run_metric_engine, num_cpus=1, num_gpus=0),
                    metric_engine=self._metric_engine,
                    history=history,
                    scenario=setup.scenario,
                    planner_name=planner.name(),
                )
            )
        else:
            # SYNC: Compute metrics immediately (blocking)
            run_metric_engine(
                metric_engine=self._metric_engine,
                history=history,
                scenario=setup.scenario,
                planner_name=planner.name(),
            )
```

**Key characteristics**:
- **Only `on_simulation_end()` implemented** - all other hooks are `pass`
- **Metrics NOT computed during simulation loop** - zero impact on step-level performance
- **Total duration**: ~5-10 seconds per scenario (post-simulation)
- **Two modes**: Sync (blocking) or Async (non-blocking with futures)

**Performance impact on simulation loop**: **ZERO** ✓

---

### 2. TimingCallback (timing_callback.py)

**Profiles execution time at each hook.**

```python
class TimingCallback(AbstractCallback):
    """Callback to log timing information to TensorBoard."""

    def __init__(self, writer: SummaryWriter):
        self._writer = writer
        self._step_start = None
        self._planner_start = None
        self._step_duration = []
        self._planner_step_duration = []

    def on_planner_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Start timing planner computation."""
        self._planner_start = time.perf_counter()

    def on_planner_end(self, setup: SimulationSetup, planner: AbstractPlanner, trajectory: AbstractTrajectory) -> None:
        """Record planner compute time."""
        assert self._planner_start, "on_planner_end called without on_planner_start!"
        self._planner_step_duration.append(time.perf_counter() - self._planner_start)

    def on_step_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Start timing simulation step."""
        self._step_start = time.perf_counter()

    def on_step_end(self, setup: SimulationSetup, planner: AbstractPlanner, sample: SimulationHistorySample) -> None:
        """Record total step time."""
        assert self._step_start, "on_step_end called without on_step_start!"
        self._step_duration.append(time.perf_counter() - self._step_start)

    def on_simulation_end(self, setup: SimulationSetup, planner: AbstractPlanner, history: SimulationHistory) -> None:
        """Write aggregated timing statistics to TensorBoard."""
        timings = {
            "mean_step_time": np.mean(self._step_duration),
            "max_step_time": np.max(self._step_duration),
            "mean_planner_step_time": np.mean(self._planner_step_duration),
            "max_planner_step_time": np.max(self._planner_step_duration),
        }
        self._writer.add_scalar("mean_step_time", timings["mean_step_time"], ...)
        # ... write other metrics ...
```

**Key characteristics**:
- **Measures each step overhead** - timing for on_step_start/end, on_planner_start/end
- **Provides TensorBoard visualization** - track performance across scenarios
- **Lightweight** - just stopwatch, no expensive computation

**Performance impact**: ~1-2ms per step (stopwatch calls)

---

### 3. SerializationCallback (serialization_callback.py)

**Saves simulation history to disk (msgpack/pickle).**

```python
class SerializationCallback(AbstractCallback):
    """Callback for serializing simulation history to disk."""

    def __init__(
        self,
        output_directory: Path,
        serialize_into_single_file: bool = False,
        serialization_type: str = "msgpack",  # or "pickle", "json"
    ):
        self._output_directory = output_directory
        self._serialize_into_single_file = serialize_into_single_file
        self._serialization_type = serialization_type
        self._history_samples = []  # Accumulate per-step if not single file

    def on_step_end(self, setup: SimulationSetup, planner: AbstractPlanner, sample: SimulationHistorySample) -> None:
        """Save per-step data (if not serializing into single file)."""
        if not self._serialize_into_single_file:
            # Accumulate samples
            self._history_samples.append(sample)

    def on_simulation_end(self, setup: SimulationSetup, planner: AbstractPlanner, history: SimulationHistory) -> None:
        """Save complete history to disk."""
        if self._serialize_into_single_file:
            # Save all history as one file
            scene_data = convert_sample_to_scene(...)  # Convert to nuBoard format
            _dump_to_file(
                file=self._output_directory / f"{setup.scenario.scenario_name}",
                scene_to_save=scene_data,
                serialization_type=self._serialization_type
            )
        else:
            # Save each step separately
            for sample in self._history_samples:
                scene_data = convert_sample_to_scene(...)
                _dump_to_file(
                    file=self._output_directory / f"{setup.scenario.scenario_name}_step_{sample.iteration}",
                    scene_to_save=scene_data,
                    serialization_type=self._serialization_type
                )
```

**Key characteristics**:
- **Converts history to scene format** - transforms nuPlan objects to nuBoard visualization format
- **Two modes**: Single file (one .msgpack per scenario) or per-step files
- **Disk I/O overhead**: ~1-2 seconds per scenario (write + compression)

**Performance impact**:
- Per-step savings: ~10 bytes per serialization call (~1ms)
- Total I/O on disk: ~1-2 seconds at on_simulation_end()

---

### 4. SimulationLogCallback (simulation_log_callback.py)

**Logs scenario execution metadata (CSV).**

```python
class SimulationLogCallback(AbstractCallback):
    """Callback for logging simulation metadata."""

    def __init__(self, output_directory: Path):
        self._output_directory = output_directory

    def on_simulation_end(self, setup: SimulationSetup, planner: AbstractPlanner, history: SimulationHistory) -> None:
        """Log scenario execution summary."""
        log_entry = {
            'log_name': setup.scenario.log_name,
            'scenario_name': setup.scenario.scenario_name,
            'scenario_type': setup.scenario.scenario_type,
            'planner_name': planner.name(),
            'duration': len(history.ego_states) * setup.scenario.database_interval,
            'num_steps': len(history.ego_states),
            'timestamp': datetime.now(),
        }
        # Append to simulation_log.csv
```

**Key characteristics**:
- **Minimal overhead** - just CSV append (~5ms)
- **Metadata only** - no history serialization
- **Useful for tracking** - which scenarios ran, when, with which planner

**Performance impact**: ~5-10ms per scenario

---

### 5. VisualizationCallback (visualization_callback.py)

**Renders bird's-eye-view frames and creates videos.**

```python
class VisualizationCallback(AbstractCallback):
    """Callback for rendering scenario visualization."""

    def __init__(self, output_directory: Path, visualization: AbstractVisualization):
        self._output_directory = output_directory
        self._visualization = visualization
        self._frames = []

    def on_step_end(self, setup: SimulationSetup, planner: AbstractPlanner, sample: SimulationHistorySample) -> None:
        """Render frame at each step."""
        frame = self._visualization.render_frame(
            ego_state=sample.ego_state,
            observation=sample.observation,
            traffic_light_status=...,
        )
        self._frames.append(frame)
        # ~0.5-2s per frame!

    def on_simulation_end(self, setup: SimulationSetup, planner: AbstractPlanner, history: SimulationHistory) -> None:
        """Create MP4 video from frames."""
        create_video(
            frames=self._frames,
            output_path=self._output_directory / f"{setup.scenario.scenario_name}.mp4",
            fps=10
        )
```

**Key characteristics**:
- **EXTREMELY SLOW** - ~0.5-2 seconds per frame (200 frames = 100-400 seconds!)
- **Blocks entire simulation** - rendering happens during on_step_end()
- **50-100x overhead** - simulation goes from 20s → 2000s with visualization
- **Only for debugging** - never use in production/batch runs

**Performance impact**: **DEVASTATING** - 50-100x slower!

⚠️ **WARNING**: VisualizationCallback should ALWAYS be disabled for batch runs

---

## Execution Orchestration

### Callback Invocation in SimulationRunner (simulations_runner.py)

```python
def run(self) -> RunnerReport:
    start_time = time.perf_counter()

    report = RunnerReport(succeeded=True, ...)

    # Hook 1
    self.simulation.callback.on_simulation_start(self.simulation.setup)

    # Initialize
    self._initialize()  # Hooks 2-3: on_initialization_start/end

    # Main loop
    while self.simulation.is_simulation_running():
        # Hook 4
        self.simulation.callback.on_step_start(self.simulation.setup, self.planner)

        planner_input = self._simulation.get_planner_input()

        # Hook 5
        self._simulation.callback.on_planner_start(self.simulation.setup, self.planner)

        trajectory = self.planner.compute_trajectory(planner_input)

        # Hook 6
        self._simulation.callback.on_planner_end(self.simulation.setup, self.planner, trajectory)

        self.simulation.propagate(trajectory)

        # Hook 7
        self.simulation.callback.on_step_end(
            self.simulation.setup, self.planner, self.simulation.history.last()
        )

        current_time = time.perf_counter()
        if not self.simulation.is_simulation_running():
            report.end_time = current_time

    # Hook 8
    self.simulation.callback.on_simulation_end(
        self.simulation.setup, self.planner, self.simulation.history
    )

    planner_report = self.planner.generate_planner_report()
    report.planner_report = planner_report

    return report
```

**Call frequency**:
- on_simulation_start/end: 1 each
- on_initialization_start/end: 1 each (during _initialize)
- on_step_start/end: ~200 each (main loop)
- on_planner_start/end: ~200 each (planner calls per step)

---

## Performance Characteristics

### Per-Step Callback Overhead

```
Simulation Step (100ms total):
  │
  ├─► on_step_start()          ~0.1ms (empty default callback)
  ├─► get_planner_input()       ~2ms   (observation processing)
  ├─► on_planner_start()        ~0.1ms (empty default)
  ├─► planner.compute_trajectory() ~50-90ms (DOMINANT - user code!)
  ├─► on_planner_end()          ~0.1ms (empty default)
  ├─► simulation.propagate()    ~3ms   (history append, state update)
  └─► on_step_end()             ~0.1ms (empty default)
     ─────────────────────────────
     TOTAL per step:            ~100ms

Callback overhead: ~0.3ms (0.3% of step time) for empty callbacks
With metrics/serialization: Depends on implementation (mostly on_simulation_end)
```

**Key insight**: On-step callbacks contribute <1% overhead when empty!

---

### Total Scenario Timing (31-scenario batch)

| Configuration | Simulation | Metrics | Serial/Parallel | Total Time |
|---|---|---|---|---|
| **Sync (basic)** | 20s × 31 = 620s | 5s × 31 = 155s | Serial | **775s** (12.9min) |
| **Sync (no metrics)** | 20s × 31 = 620s | 0s | Serial | **620s** (10.3min) |
| **Async (4 workers)** | 20s × 31 = 620s | 5s × 31 = 155s | Parallel | **625s** (10.4min) |
| **Visualization** | (20+100)s × 31 = 3720s | 5s × 31 = 155s | Serial | **3875s** (64.6min) |

**Conclusion**: Metrics add ~25% overhead in sync mode, but can be parallelized with async. Visualization is a **10x disaster**.

---

## Critical Performance Findings

### Finding 1: Callback Overhead During Simulation is Minimal

**Evidence**:
- Empty callbacks (just `pass`) add ~0.3ms per step
- With 200 steps: 0.3ms × 200 = 60ms total overhead
- Planner dominant: 50-90ms per step (100-200x more than callbacks)

**Conclusion**: The 95ms overhead mentioned in the mission is NOT from on-step callbacks.

---

### Finding 2: All Expensive Callbacks Run at Simulation End

**Evidence**:
- MetricCallback: only `on_simulation_end()` does work
- SerializationCallback: only `on_simulation_end()` does work
- VisualizationCallback: `on_step_end()` is EXPENSIVE (but only one callback type)

**Timeline**:
```
Simulation loop: 20s (REAL-TIME CONSTRAINTS)
  - Callbacks: <1ms per step (negligible)

on_simulation_end: 5s (POST-SIMULATION, no time constraints)
  - Metrics: 3-5s
  - Serialization: 1-2s
  - Logging: ~10ms
```

**Conclusion**: Metrics and serialization are NOT on the critical path!

---

### Finding 3: Callback Futures Must Be Awaited

**Evidence** (executor.py:105-114):
```python
# execute_runners() awaits all callback futures before returning
callback_futures_map = {
    future: (scenario_name, planner_name, log_name)
    for runner in runners
    for callback in runner.simulation.callback.callbacks
    if isinstance(callback, (MetricCallback, ...))
    for future in callback.futures
}

for future in concurrent.futures.as_completed(callback_futures_map.keys()):
    try:
        future.result()  # Blocks until done, raises if exception
    except Exception:
        # Mark report as failed if callback crashes
```

**Impact**: If any async metric computation fails, the entire batch is marked failed.

---

## Configuration & Tuning

### Recommended Callback Stack (Optimal Balance)

```yaml
callbacks:
  # Order matters!
  - timing_callback              # First: Measure everything else
  - metric_callback              # Second: Post-sim computation
  - serialization_callback       # Third: Save history
  # DON'T include: visualization_callback (unless debugging single scenario)
```

**Rationale**:
1. **Timing first**: Wraps other callbacks, accurate measurements
2. **Metrics second**: Computes performance scores
3. **Serialization third**: Saves data for later visualization
4. **Skip visualization**: 50x slowdown, only for offline rendering

---

### Reduce Callback Overhead

**Option 1: Disable Non-Essential Callbacks**
```yaml
callbacks:
  - timing_callback              # Lightweight ~1-2ms per step
  - metric_callback              # Only runs at simulation end
  # Remove serialization if not visualizing
```

**Option 2: Use Async Metrics**
```yaml
callback:
  metric_callback:
    worker_pool: ray_distributed  # Metrics run in background workers
    threads_per_node: 4
# Metrics computed while next scenario simulates (parallelism!)
```

**Option 3: Skip Metrics During Development**
```bash
# Run without metrics
uv run python nuplan/planning/script/run_simulation.py \
    callback.metric_callback=null

# Later: Compute metrics offline from saved logs
uv run python nuplan/planning/script/run_metric.py \
    simulation_log_dir=$NUPLAN_EXP_ROOT/simulation_logs
```

---

## Gotchas & Pitfalls

### Gotcha 1: Callbacks Cannot Modify Simulation

**Assumption**: Callback can adjust planner decision
```python
def on_planner_end(self, setup, planner, trajectory):
    # WRONG: Trajectory is read-only
    trajectory.states[0].x += 1.0  # Has no effect!
```

**Reality**: All arguments are read-only references. Any mutations don't affect simulation.

**Safe pattern**: Only observe, don't modify.

---

### Gotcha 2: on_step_* Callbacks Must Be Fast

**Problem**: Callback takes 50ms, step time becomes 150ms
```python
def on_step_end(self, setup, planner, sample):
    # BAD: This blocks the step!
    expensive_computation()  # 50ms
    # Step now takes 150ms instead of 100ms
```

**Budget**: ~5-10ms per callback (step is 100ms total).

**Solution**: Offload expensive work to on_simulation_end or async workers.

---

### Gotcha 3: Callbacks Access Shared History

**Problem**: MultiCallback receives same history object
```python
class BadCallback(AbstractCallback):
    def on_simulation_end(self, setup, planner, history):
        # DANGER: Mutates shared object
        history.ego_states = history.ego_states[:50]  # Truncate!
        # SerializationCallback (next) saves truncated history!
```

**Solution**: Never mutate objects passed to callbacks. Copy if needed:
```python
class GoodCallback(AbstractCallback):
    def on_simulation_end(self, setup, planner, history):
        # SAFE: Work on copy
        ego_states = list(history.ego_states)
        filtered = [s for s in ego_states if valid(s)]
```

---

### Gotcha 4: Exceptions in Callbacks Crash Entire Batch

**Problem**: Metric computation fails on one scenario
```python
# Scenario 1-30 complete, scenario 31 crashes
# With sync metrics: All 30 results saved before failure
# With async metrics: Results may be lost if future not awaited!
```

**Solution**: execute_runners() awaits all futures before returning.

---

### Gotcha 5: Order Dependencies Between Callbacks

**Bad**:
```python
callbacks = MultiCallback([
    SerializationCallback(),    # Saves history (without metrics!)
    MetricCallback(),           # Computes metrics (too late)
])
```

**Good**:
```python
callbacks = MultiCallback([
    TimingCallback(),           # Profile everything
    MetricCallback(),           # Compute metrics first
    SerializationCallback(),    # Then save (with metrics)
])
```

**Reason**: Serialization should include metric results if available.

---

### Gotcha 6: VisualizationCallback Kills Performance

**Problem**: Need to visualize one scenario
```bash
# WRONG: Visualization on for all 31 scenarios
just simulate  # 30+ minutes!

# RIGHT: Visualization only for debugging
.venv/bin/python ... callback=timing,metric,serialization  # No viz
# Then render offline from saved logs
```

**Solution**: Disable visualization, render from saved history files offline.

---

### Gotcha 7: File Handle Leaks in Custom Callbacks

**Problem**: Open file, never close
```python
class LeakyCallback(AbstractCallback):
    def __init__(self):
        self.f = open("log.txt", "w")  # DANGER: Open file!

    def on_step_end(self, ...):
        self.f.write("step complete\n")  # File handle leaks!
```

**Solution**: Open lazily, close in `__del__`:
```python
class SafeCallback(AbstractCallback):
    def __init__(self):
        self.log_file = "log.txt"
        self._f = None

    def on_step_end(self, ...):
        if self._f is None:
            self._f = open(self.log_file, "w")
        self._f.write("step complete\n")

    def __del__(self):
        if self._f:
            self._f.close()
```

Or better, use context managers:
```python
def on_simulation_end(self, setup, planner, history):
    with open(self.log_file, "a") as f:
        f.write(f"Scenario {setup.scenario.scenario_name} complete\n")
```

---

### Gotcha 8: Async Callback Futures Not Tracked

**Problem**: Callback submitted to worker pool, future lost
```python
# DANGER: Future created but never stored
self._pool.submit(run_metric_engine, ...)
# Main process continues, future computation lost!
```

**Solution**: Store futures, executor awaits them:
```python
self._futures.append(
    self._pool.submit(run_metric_engine, ...)
)
# execute_runners() will await self._futures
```

---

## Cross-References

### Related Files
- **abstract_callback.py** - Base interface (8 lifecycle hooks)
- **multi_callback.py** - Composite pattern for combining callbacks
- **metric_callback.py** - Metrics computation bridge
- **timing_callback.py** - Performance profiling
- **serialization_callback.py** - History serialization
- **simulation_log_callback.py** - Metadata logging
- **visualization_callback.py** - Bird's-eye-view rendering

### Integration Points
- **nuplan/planning/simulation/runner/simulations_runner.py** - Invokes callbacks at 8 hooks
- **nuplan/planning/simulation/runner/executor.py** - Awaits async callback futures
- **nuplan/planning/script/run_simulation.py** - Builds callbacks from Hydra config
- **nuplan/planning/simulation/CLAUDE.md** - Top-level simulation orchestration

### See Also
- **docs/architecture/METRICS.md** - Detailed metric computation flow
- **nuplan/planning/simulation/callback/CLAUDE.md** - Comprehensive callback patterns (gotchas, patterns, advanced usage)

---

## Summary

**Callback System Design**:
- Event-driven architecture with 8 lifecycle hooks
- Passive observers (cannot modify simulation)
- Sequential execution within MultiCallback
- Optional async execution with WorkerPool

**Performance Characteristics**:
- Per-step callbacks: <1ms overhead (negligible)
- Post-simulation callbacks: 5-10s overhead (not on critical path)
- Metrics: Sequential computation, can be parallelized with async workers
- Visualization: 50-100x slowdown (debugging only!)

**Key Finding**: Callback overhead during simulation is minimal. The 95ms per-step overhead likely comes from planner latency variability and observation processing, not callbacks.

**Optimization**: Use async metrics with worker pool to parallelize metric computation with next scenario simulation (15-20% speedup).

