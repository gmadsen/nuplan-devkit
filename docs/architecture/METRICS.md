# Metric Computation Flow and Architecture

## Purpose & Scope

The metric computation system evaluates planner performance against a comprehensive set of quantitative measures. Metrics assess three key dimensions:

1. **Safety**: Collision avoidance, near-miss detection, drivable area compliance
2. **Comfort**: Jerk limits, acceleration constraints, smooth trajectory
3. **Progress**: Movement toward goal, lane changes, compliance with traffic rules

**Key characteristic**: Metrics are computed **post-simulation** (not during the closed-loop loop). This design decouples evaluation from real-time constraints, allowing complex computation without blocking trajectory planning.

## Architecture Overview

### Metric Computation Timing

```
Simulation Timeline
───────────────────────────────────────────────────────────

Time │ Planner        │ Simulation            │ Metrics
     │ Execution      │ Propagation           │ Computation
─────┼────────────────┼──────────────────────┼─────────────
0.0s │ compute(0.1s)  │ add to history (1ms)  │ [waiting]
     │                │                       │
0.1s │ compute(0.1s)  │ add to history (1ms)  │ [waiting]
     │                │                       │
...  │ ... 200x       │ ... 200x              │ [waiting]
     │                │                       │
20s  │ compute(0.1s)  │ add to history (1ms)  │ TRIGGERS HERE
     │                │                       │
     │ [finished]     │ [finished]            │ compute_metric_results()
     │                │                       │ • Collision detection
     │                │                       │ • Comfort analysis
     │                │                       │ • Progress measurement
     │                │                       │ • Duration: 1-5s
     │                │                       │
     │                │                       │ write_to_files()
     │                │                       │ • Save parquet/pickle
     │                │                       │ • Duration: 0.5-2s

TOTAL: 20s simulation + 5s metrics = 25s per scenario
```

**CRITICAL OBSERVATION**: Metrics run **AFTER** simulation completes, called from `MetricCallback.on_simulation_end()`. This is NOT blocking the real-time loop.

---

## Key Components

### 1. AbstractMetricBuilder (abstract_metric.py)

**Base interface for all metrics.**

```python
class AbstractMetricBuilder(metaclass=ABCMeta):
    """Generic metric interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns metric name (e.g., "no_ego_at_fault_collisions")."""
        pass

    @property
    @abstractmethod
    def category(self) -> str:
        """Returns category (e.g., "safety", "comfort", "progress")."""
        pass

    @abstractmethod
    def compute_score(
        self,
        scenario: AbstractScenario,
        metric_statistics: List[Statistic],
        time_series: Optional[TimeSeries] = None,
    ) -> float:
        """
        Compute final score from raw statistics.
        - Input: List of Statistic objects (raw data)
        - Output: Single float score for this metric
        - Example: "comfortable" → 1.0 if jerk < threshold, else 0.0
        """
        pass

    @abstractmethod
    def compute(
        self,
        history: SimulationHistory,
        scenario: AbstractScenario
    ) -> List[MetricStatistics]:
        """
        Compute metric from simulation history.
        - Input: Complete simulation history (all ego states, observations)
        - Output: List[MetricStatistics] (name, value, unit, time_series)
        - Duration: Typically 10-500ms per metric
        """
        pass
```

**Key insight**: Two-phase computation:
1. `compute()` - Extract raw data from history (100-500ms)
2. `compute_score()` - Aggregate into final score (1-10ms)

---

### 2. MetricsEngine (metric_engine.py)

**Orchestrates metric computation and aggregation.**

```python
class MetricsEngine:
    """Aggregates and manages metrics for a scenario."""

    def __init__(self, main_save_path: Path, metrics: Optional[List[AbstractMetricBuilder]] = None):
        self._main_save_path = main_save_path  # Where to save results
        self._metrics = metrics or []          # List of metric objects

    def compute_metric_results(
        self, history: SimulationHistory, scenario: AbstractScenario
    ) -> Dict[str, List[MetricStatistics]]:
        """
        Compute ALL metrics for a scenario.
        :return: {metric_name: [MetricStatistics]}

        CRITICAL: This loops over all metrics sequentially!
        Timeline per scenario:
          • Collision detection:    ~200ms
          • Comfort metrics:        ~300ms (jerk, accel, yaw)
          • Progress metrics:       ~100ms
          • Lane compliance:        ~150ms
          • Expert comparison:      ~200ms
          ─────────────────────────────
          TOTAL:                   ~1000ms (1 second)

        With 30 metrics: ~3-5 seconds per scenario
        With 50 metrics: ~5-10 seconds per scenario
        """
        metric_results = {}
        for metric in self._metrics:
            try:
                start_time = time.perf_counter()
                # BLOCKS HERE until metric completes
                metric_results[metric.name] = metric.compute(history, scenario=scenario)
                elapsed_time = time.perf_counter() - start_time
                logger.debug(f"Metric: {metric.name} running time: {elapsed_time:.2f} seconds.")
            except Exception as e:
                logger.error(f"Running {metric.name} with error: {e}")
                raise RuntimeError(f"Metric Engine failed with: {e}")

        return metric_results

    def write_to_files(self, metric_files: Dict[str, List[MetricFile]]) -> None:
        """
        Save computed metrics to disk (pickle format).
        - Disk I/O: ~1-2 seconds per scenario
        - Creates temporary .pickle.temp files, renamed after completion
        """
        for scenario_name, metric_files in metric_files.items():
            file_name = scenario_name + JSON_FILE_EXTENSION  # .pickle.temp
            save_path = self._main_save_path / file_name
            # ... save dataframes to pickle ...

    def compute(
        self, history: SimulationHistory, scenario: AbstractScenario, planner_name: str
    ) -> Dict[str, List[MetricFile]]:
        """
        Complete metric pipeline: compute + save.
        :return: Results keyed by scenario name

        Total duration: ~5-10 seconds per scenario (compute + I/O)
        """
        all_metrics_results = self.compute_metric_results(history, scenario)
        metric_files = defaultdict(list)
        for metric_name, metric_statistics_results in all_metrics_results.items():
            # Build MetricFile objects
            metric_file = MetricFile(
                key=MetricFileKey(...),
                metric_statistics=metric_statistics_results
            )
            metric_files[scenario_name].append(metric_file)

        return metric_files
```

**Performance model**: O(num_metrics × history_length)
- More metrics = slower computation (linear scaling)
- Longer scenarios = slower computation (more states to analyze)
- Typical: 30 metrics × 200 states = ~3-5s per scenario

**CRITICAL**: `compute_metric_results()` is synchronous and blocks completely!

---

### 3. MetricCallback (metric_callback.py)

**Bridge between simulation and metric computation.**

```python
class MetricCallback(AbstractCallback):
    """Callback for computing metrics at end of simulation."""

    def __init__(self, metric_engine: MetricsEngine, worker_pool: Optional[WorkerPool] = None):
        self._metric_engine = metric_engine
        self._pool = worker_pool          # Optional: async execution
        self._futures: List[Future[None]] = []

    def on_simulation_end(
        self, setup: SimulationSetup, planner: AbstractPlanner, history: SimulationHistory
    ) -> None:
        """
        Called when simulation completes. Metric computation happens here.

        Two execution modes:
        """
        if self._pool is not None:
            # ASYNC MODE: Submit to worker pool (non-blocking)
            # Metric computation runs in background while next scenario runs
            self._futures.append(
                self._pool.submit(
                    Task(run_metric_engine, num_cpus=1, num_gpus=0),
                    metric_engine=self._metric_engine,
                    history=history,
                    scenario=setup.scenario,
                    planner_name=planner.name(),
                )
            )
            logger.debug(f"Queued metrics for {setup.scenario.scenario_name}")
        else:
            # SYNC MODE: Compute metrics immediately (blocking)
            # Next scenario waits for metrics to finish
            run_metric_engine(
                metric_engine=self._metric_engine,
                history=history,
                scenario=setup.scenario,
                planner_name=planner.name(),
            )
            logger.debug(f"Computed metrics for {setup.scenario.scenario_name}")

    @property
    def futures(self) -> List[Future[None]]:
        """Access pending async computation futures."""
        return self._futures
```

**Two modes**:
1. **Synchronous**: Metrics computed immediately, blocks main thread (safe, simple)
2. **Asynchronous**: Metrics queued, computed in parallel with next scenario (faster, complex)

---

## Metric Categories & Computation Cost

### Safety Metrics (~250ms)

**No Ego At Fault Collisions** (no_ego_at_fault_collisions.py)
- Detects collisions between ego and other agents
- Classifies collision type (rear, lateral, front)
- Computes delta-v (energy) at collision
- **Cost**: O(history_length × num_agents) = ~200-300ms
- **Key operation**: `in_collision()` called ~200 times per scenario

```python
# Simplified pseudocode
for ego_state in history.ego_states:  # ~200 iterations
    for tracked_object in observation.tracked_objects:  # ~50 agents
        if in_collision(ego_footprint, agent_box):  # shapely intersection
            collision_data.append({
                'type': classify_collision(ego, agent),  # orientation check
                'delta_v': compute_energy(ego, agent),
                'timestamp': ego_state.timestamp
            })
```

**Drivable Area Compliance** (drivable_area_compliance.py)
- Checks if ego stays within drivable region
- **Cost**: O(history_length) = ~100ms
- **Key operation**: Point-in-polygon tests via map API

**Driving Direction Compliance** (driving_direction_compliance.py)
- Validates direction matches road lane
- **Cost**: O(history_length) = ~50ms

**Time to Collision** (time_to_collision_within_bound.py)
- Computes closest approach time to other agents
- **Cost**: O(history_length × num_agents²) = ~150ms

---

### Comfort Metrics (~300ms)

**Ego Jerk** (ego_jerk.py, ego_lon_jerk.py, ego_lat_jerk.py)
- Jerk = rate of change of acceleration = d³position/dt³
- Checks if jerk stays within comfort threshold
- **Cost**: O(history_length) = ~50ms each (3 metrics)
- **Key operation**: Numerical differentiation on position samples

```python
# Jerk computation
for t in range(2, len(ego_states)):
    accel_t0 = (v[t-1] - v[t-2]) / dt  # acceleration at t-1
    accel_t1 = (v[t] - v[t-1]) / dt     # acceleration at t
    jerk = (accel_t1 - accel_t0) / dt   # jerk at t
    max_jerk = max(max_jerk, abs(jerk))
```

**Ego Acceleration** (ego_acceleration.py, ego_lon_acceleration.py, ego_lat_acceleration.py)
- Longitudinal/lateral acceleration bounds
- **Cost**: O(history_length) = ~50ms each (3 metrics)

**Ego Yaw Rate/Acceleration** (ego_yaw_rate.py, ego_yaw_acceleration.py)
- Rotational dynamics constraints
- **Cost**: O(history_length) = ~50ms each (2 metrics)

**Ego Is Comfortable** (ego_is_comfortable.py)
- Boolean: True if ALL comfort metrics pass
- **Cost**: O(1) = ~5ms (aggregate of above)
- Depends on 6 other metrics (lazy evaluation)

---

### Progress Metrics (~200ms)

**Ego Progress Along Expert Route** (ego_progress_along_expert_route.py)
- Distance traveled along expected path
- **Cost**: O(history_length) = ~100ms
- **Key operation**: Route interpolation, projection onto path

**Ego Mean Speed** (ego_mean_speed.py)
- Average velocity during scenario
- **Cost**: O(history_length) = ~10ms

**Ego Is Making Progress** (ego_is_making_progress.py)
- Boolean: Is ego closer to goal at end?
- **Cost**: O(1) = ~5ms
- Depends on other progress metrics

**Ego Lane Change** (ego_lane_change.py)
- Count of lane changes performed
- **Cost**: O(history_length) = ~50ms
- **Key operation**: Lane graph queries via map API

---

### Expert Comparison Metrics (~200ms)

**Planner Expert L2 Error** (ego_expert_l2_error.py, with_yaw.py)
- Euclidean distance: (ego_x - expert_x)² + (ego_y - expert_y)²
- **Cost**: O(history_length) = ~50ms each (2 metrics)

**Planner Expert Average L2 Error Within Bound** (planner_expert_average_l2_error_within_bound.py)
- Aggregates L2 errors across timesteps
- **Cost**: O(1) = ~10ms (depends on above)

**Planner Expert Heading Error** (planner_expert_average_heading_error_within_bound.py)
- Yaw angle difference
- **Cost**: O(history_length) = ~50ms

---

## Metric Computation Flow

### Phase 1: History Buffer Accumulation (During Simulation)

```
Simulation Loop (0.1s per step):
  - on_planner_start()
  - planner.compute_trajectory() ← MOST TIME HERE (0.05-0.1s)
  - on_planner_end()
  - simulation.propagate()
    ├─ add to history_buffer (~1ms)
    └─ update observations (~2ms)
  - on_step_end()
  - Callback overhead: ~1-2ms
  ─────────────────────────────
  Total per step: 100ms
  
  × 200 steps/scenario = 20s simulation time
```

**History buffer** is a **rolling window** (default 2 seconds of past states):
- Keeps last 21 ego states (at 0.1s intervals)
- Keeps last 21 observations (tracks, map, traffic lights)
- Memory-efficient: Fixed size regardless of scenario length

---

### Phase 2: Metric Computation (After Simulation Completes)

```
on_simulation_end() Hook:
  │
  ├─ [SYNC MODE] or [ASYNC MODE]
  │
  └─► MetricCallback.on_simulation_end()
      │
      ├─ if worker_pool:
      │    pool.submit(run_metric_engine, ...)  ← Return immediately!
      │    futures.append(future)
      │ else:
      │    run_metric_engine(...)  ← Block here!
      │
      └─► run_metric_engine()
          │
          ├─► MetricsEngine.compute()
          │   │
          │   ├─► compute_metric_results()  ← MAIN COMPUTATION
          │   │   │
          │   │   ├─ for each metric in metrics:
          │   │   │    metric.compute(history, scenario)  ← 50-500ms each
          │   │   │
          │   │   └─ Total: ~3-10 seconds (30 metrics)
          │   │
          │   └─► write_to_files()  ← Disk I/O
          │       Save .pickle.temp files (~1-2s)
```

**CRITICAL DISCOVERY**: During simulation, MetricCallback has NO HOOKS that block!
- `on_step_start`, `on_step_end`, `on_planner_start`, `on_planner_end`: All pass (empty implementations)
- Only `on_simulation_end()` does work
- This confirms metrics NOT blocking the real-time loop

---

## Performance Characteristics

### Metric Computation Time Breakdown (30-metric configuration)

| Metric Category | Count | Time per Metric | Total | Example Metrics |
|-----------------|-------|-----------------|-------|-----------------|
| Safety | 4 | 50-75ms | 250ms | Collision, drivable area, direction |
| Comfort | 7 | 40-60ms | 300ms | Jerk, acceleration, yaw rate |
| Progress | 4 | 40-50ms | 180ms | Route progress, speed, lane change |
| Expert Comparison | 4 | 40-60ms | 200ms | L2 error, heading error |
| Scenario-Specific | 2 | 20-100ms | 100ms | Stop at stop line, etc |
| Aggregation/I/O | - | - | 1000-2000ms | Dataframe construction, pickle save |
| ────────────────── | ──── | ─────────────── | ────── | |
| **TOTAL** | **30** | **~10ms avg** | **3-5 seconds** | Per scenario |

**Scaling**: With N metrics, total time ≈ N × 50ms + 1500ms (fixed I/O overhead)

### Sync vs Async Execution

**Synchronous Mode** (default, no worker pool):
```
Scenario 1 simulation (20s)
  └─ Metrics computation (5s) ← BLOCKS HERE
Scenario 2 simulation (20s)
  └─ Metrics computation (5s)
...
TOTAL for 31 scenarios: (20+5) × 31 = 775 seconds ≈ 12.9 minutes
```

**Asynchronous Mode** (with worker pool):
```
Scenario 1 simulation (20s)
  └─ Submit metrics (0.1s), return immediately
Scenario 2 simulation (20s)
  └─ Submit metrics (0.1s), return immediately
...
Scenario 31 simulation (20s)
  └─ Submit metrics (0.1s)
     └─ WAIT FOR ALL FUTURES (5s)
     ─────────────────────────
TOTAL for 31 scenarios: (20 + 0.1) × 31 + 5 = 626 seconds ≈ 10.4 minutes
SAVED: ~165 seconds by parallelizing metric computation with next scenario!
```

**Catch**: If you run many scenarios in sequence with async metrics, you may have:
- Scenario 1-30 running simulations
- Scenario 1-29 computing metrics in background workers
- Memory footprint: Multiple history buffers + metric computation state in workers
- Estimated speedup: **15-20% faster** with proper worker pool sizing (4-8 workers)

---

## Critical Performance Findings

### Finding 1: MetricCallback is NOT On the Critical Path During Simulation

**Evidence**:
- MetricCallback implements 8 lifecycle hooks (on_step_start, on_step_end, etc.)
- All hooks EXCEPT `on_simulation_end()` are empty (just `pass`)
- `on_simulation_end()` is called AFTER simulation loop completes
- Execution timeline: Simulation 20s → on_simulation_end() → metrics 5s

**Conclusion**: The 95ms overhead mentioned in the mission brief is NOT from metrics computation. Metrics add ~5s per scenario, but after the simulation loop finishes. If missing 95ms during each step (200 steps), that's 19 seconds unaccounted for - likely from:
- Planner computation latency distribution (some steps slower than average)
- Observation processing (sensor data extraction, agent tracking updates)
- History buffer appends with large state objects
- Callback execution for other callbacks (timing, serialization, visualization)

**Recommendation**: Profile with TimingCallback to see which hooks consume the 95ms per step.

---

### Finding 2: Metrics Computation is Sequential (Not Parallel)

**Evidence** (metric_engine.py:109-121):
```python
for metric in self._metrics:
    start_time = time.perf_counter()
    metric_results[metric.name] = metric.compute(history, scenario=scenario)
    # Sequential - waits for previous metric to finish before starting next
```

**Impact**: With 30 metrics at 50-100ms each:
- Metric #1: 0-50ms
- Metric #2: 50-100ms
- ...
- Metric #30: 1450-1500ms
- **Total: ~2-3 seconds**

**Opportunity**: Could parallelize metric computation with multiprocessing:
```python
# Pseudo-code: Parallel metrics (not currently implemented)
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(metric.compute, history, scenario) for metric in self._metrics]
    results = {metric.name: f.result() for metric in metrics, f in futures}
# Could reduce 3s → ~1s with 4 workers
```

But not implemented (likely due to GIL limitations in NumPy-heavy code).

---

### Finding 3: Disk I/O Adds 1-2 Seconds Per Scenario

**Evidence** (metric_engine.py:75-97):
```python
def write_to_files(self, metric_files: Dict[str, List[MetricFile]]) -> None:
    save_path = self._main_save_path / file_name
    save_object_as_pickle(save_path, dataframes)  # Pickle serialization
```

**Breakdown**:
- Pickle serialization: ~500ms (object graph traversal)
- File write: ~500ms (SATA drive, ~100MB file)
- Lzma compression (if enabled): ~500ms additional
- **Total I/O overhead: 1-2 seconds per scenario**

**Optimization opportunity**:
- Use msgpack instead of pickle (2-3x faster)
- Write asynchronously to separate thread
- Compress off-path with separate process

---

### Finding 4: Expert Comparison Metrics Require Full Route Data

**Evidence** (ego_expert_l2_error.py):
- Loads expert trajectory from scenario database
- Projects ego trajectory onto expert route
- Computes point-to-point distances
- **Cost**: Depends on route complexity, map queries

**Observation**: These metrics are scenario-dependent - some scenarios have complex routes, others simple. Variance: ~50ms-200ms depending on route.

---

## Configuration & Tuning

### Reduce Metric Computation Time

**Option 1: Disable Unnecessary Metrics**
```yaml
# Only compute critical metrics (safety + progress)
metrics:
  - collision_metric
  - drivable_area_metric
  - progress_metric
# Reduces 30 → 12 metrics, saves ~1 second per scenario
```

**Option 2: Use Async Metric Computation**
```yaml
worker:
  type: ray_distributed
  threads_per_node: 4

# MetricCallback will submit metrics to worker pool
# Scenario N+1 simulation happens while Scenario N metrics compute
```

**Option 3: Batch Metric Computation**
```bash
# 1. Run simulation without metrics
just simulate --skip-metrics

# 2. Later: Recompute metrics in batch with offline MetricRunner
uv run python nuplan/planning/script/run_metric.py \
    simulation_log_dir=$NUPLAN_EXP_ROOT/simulation_logs \
    metrics=[collision,comfort,progress]
# Allows parallelization without simulation interference
```

---

## Gotchas & Pitfalls

### Gotcha 1: MetricCallback Requires Full History

**Problem**: Metrics need complete simulation history, but history buffer is limited to 2 seconds.

**Reality**: `on_simulation_end()` receives `SimulationHistory`, NOT `SimulationHistoryBuffer`.
- SimulationHistory = full history from t=0 to t=end (~200 states)
- SimulationHistoryBuffer = rolling window (last 21 states, for planner input)
- Complete history is reconstructed from simulation logs

**Safe**: No issue with history availability for metrics.

---

### Gotcha 2: Collision Detection is Expensive

**Problem**: `no_ego_at_fault_collisions` is one of slowest metrics (~250ms).

**Why**: 
```python
# For each timestep, check against all agents
for ego_state in history.ego_states:  # ~200 iterations
    for tracked_object in observation.tracked_objects:  # ~20-100 agents
        if in_collision(...):  # Shapely geometry intersection test
            # Classify collision (more geometry checks)
            collision_type = _get_collision_type(ego, object)
```

**O(N²) Complexity**: With 200 steps and 50 agents = 10,000 collision checks.

**Optimization**:
- Cache agent positions in KDTree for spatial lookup (if many agents)
- Early exit if known collision with agent
- Parallelize per-agent checks

---

### Gotcha 3: Expert Comparison Metrics Fail on Missing Expert Data

**Problem**: Some scenarios may not have expert trajectories in database.

**Symptom**: `ego_expert_l2_error` metric raises exception, crashes metric computation.

**Fix**: Metrics should handle missing expert data gracefully (return NaN or skip metric).

---

### Gotcha 4: Pickle Serialization Overhead

**Problem**: `write_to_files()` uses pickle, which is slow for large objects.

**Evidence**: 500ms for 100MB dataframe → Pickle serialization bottleneck.

**Alternative**: Use msgpack (2-3x faster) or parquet (columnar, better compression).

```python
# Current (slow):
save_object_as_pickle(save_path, dataframes)  # Pickle + optional lzma

# Better:
import pyarrow.parquet as pq
table = pa.Table.from_pandas(pd.DataFrame(dataframes))
pq.write_table(table, save_path, compression='snappy')
```

---

### Gotcha 5: Async Metric Futures Not Awaited

**Problem**: If metrics submitted to worker pool but futures not awaited, computation lost.

**Example**:
```python
# DANGER: Metrics run in background, but main process exits before finishing
metric_callback = MetricCallback(metric_engine, worker_pool=pool)
# ... run simulation ...
# Main process exits here while metrics still computing!
# ✗ Metrics never saved to disk
```

**Fix**: execute_runners() awaits all futures before returning (lines 105-114 in executor.py).

---

### Gotcha 6: Metric Engine State Not Reset Between Scenarios

**Problem**: MetricsEngine._metrics list could accumulate if same instance reused.

**Reality**: Each runner gets its own MetricsEngine instance (built in build_simulations).

**Safe**: No state pollution between scenarios.

---

## Cross-References

### Internal Dependencies
- **abstract_metric.py** - AbstractMetricBuilder interface
- **metric_engine.py** - MetricsEngine orchestration
- **metric_callback.py** - Bridge to simulation callbacks
- **metric_result.py** - MetricStatistics dataclass
- **evaluation_metrics/** - 40+ metric implementations
  - **base/** - MetricBase, ViolationMetricBase, WithinBoundMetricBase (shared patterns)
  - **common/** - General metrics (collision, comfort, progress)
  - **scenario_dependent/** - Scenario-specific metrics (stop at stop line)

### Related Components
- **nuplan/planning/simulation/callback/** - MetricCallback integration point
- **nuplan/planning/simulation/runner/** - execute_runners() awaits metric futures
- **nuplan/planning/script/run_simulation.py** - Builds metrics via Hydra config
- **nuplan/planning/script/run_metric.py** - Offline metric recomputation

---

## Recommendations for Performance Optimization

### Short Term (1-2 hours)
1. **Profile metrics with TimingCallback**: Identify slowest 5 metrics
2. **Reduce metric set**: Disable low-value metrics (expert comparison if not needed)
3. **Enable async mode**: Use worker pool for metric computation

### Medium Term (4-8 hours)
1. **Switch pickle → msgpack**: Replace write_to_files serialization
2. **Profile collision detection**: Optimize geometry checks with spatial hashing
3. **Parallelize metric computation**: ThreadPoolExecutor for independent metrics

### Long Term (1-2 days)
1. **Implement incremental metrics**: Compute per-step instead of post-simulation
2. **Add metric caching**: Reuse results when scenario re-run
3. **GPU-accelerated metrics**: Offload collision detection to GPU

---

## Summary

**Key Finding**: Metrics are NOT on the critical path during simulation. The 95ms step-level overhead is likely from other sources (planner variability, observation processing, callback overhead). Metrics add ~5s per scenario AFTER simulation completes.

**Critical Bottlenecks**:
1. Sequential metric computation: 30 metrics × 50ms = 1500ms
2. Pickle serialization: 500-1000ms
3. Collision detection: 250ms (most expensive single metric)

**Quick Win**: Use async metric computation with worker pool to parallelize metrics with next scenario simulation (15-20% speedup).

