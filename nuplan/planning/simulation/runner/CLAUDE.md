# nuplan/planning/simulation/runner/

## Purpose & Scope

The **runner system** is the orchestration layer that manages simulation execution, parallelization, error handling, and result aggregation. It acts as the bridge between scenario selection (what to simulate) and simulation execution (how to simulate), providing resilient batch execution with sophisticated error isolation and recovery.

**Core responsibilities**:
- **Lifecycle control**: Initialize → Run → Report for each simulation
- **Parallelization**: Distribute scenarios across workers (Sequential/Thread/Ray)
- **Error handling**: Isolate failures, capture tracebacks, continue batch execution
- **Result aggregation**: Collect RunnerReports, await async callbacks, save metrics
- **Resource management**: GPU/CPU allocation, memory limits, disk space monitoring

**Key characteristic**: Runners are **thin orchestration wrappers** around Simulation objects, NOT business logic containers. The simulation loop lives in `Simulation.propagate()`, while runners handle batch execution, error recovery, and reporting.

## Key Abstractions

### 1. AbstractRunner (abstract_runner.py)
**Base interface for all runner types.**

```python
class AbstractRunner(metaclass=ABCMeta):
    """Interface for a generic runner."""

    @abstractmethod
    def run(self) -> RunnerReport:
        """
        Run through all runners with simulation history.
        :return A runner report with success status, timing, and planner stats.
        """
        pass

    @property
    @abstractmethod
    def scenario(self) -> AbstractScenario:
        """Return the scenario being executed."""
        pass

    @property
    @abstractmethod
    def planner(self) -> AbstractPlanner:
        """Return the planner being evaluated."""
        pass
```

**Key contracts**:
- `run()` must return `RunnerReport` (NEVER raise - exceptions caught by executor)
- `scenario`/`planner` properties enable error logging (log_name, scenario_name)
- No parameters to `run()` - all state encapsulated in runner instance

**AIDEV-NOTE**: AbstractRunner is intentionally minimal - orchestration logic lives in executor.py

---

### 2. SimulationRunner (simulations_runner.py)
**Executes closed-loop simulation with a planner.**

```python
class SimulationRunner(AbstractRunner):
    """Manager which executes simulations with a planner."""

    def __init__(self, simulation: Simulation, planner: AbstractPlanner):
        self._simulation = simulation
        self._planner = planner

    def run(self) -> RunnerReport:
        """
        Run simulation until complete.
        Steps:
          1. Initialize planner (route, map, goal)
          2. Loop until simulation.is_simulation_running() returns False
          3. Trigger callbacks at each lifecycle point
          4. Return report with timing and planner stats
        """
        # ... see architecture section for full flow
```

**Responsibilities**:
1. **Lifecycle management**: Initialize planner, run loop, collect reports
2. **Callback orchestration**: Trigger 8 lifecycle hooks at correct times
3. **Timing measurement**: Record start_time/end_time for performance analysis
4. **Metadata extraction**: Populate report with scenario_name, log_name, planner_name

**Lifecycle hooks** (in order):
1. `on_simulation_start(setup)` - Before planner initialization
2. `on_initialization_start(setup, planner)` - Before planner.initialize()
3. `on_initialization_end(setup, planner)` - After planner.initialize()
4. `on_step_start(setup, planner)` - Before each timestep
5. `on_planner_start(setup, planner)` - Before compute_trajectory()
6. `on_planner_end(setup, planner, trajectory)` - After compute_trajectory()
7. `on_step_end(setup, planner, sample)` - After propagate()
8. `on_simulation_end(setup, planner, history)` - After loop completes

**AIDEV-NOTE**: SimulationRunner does NOT catch exceptions - that's executor.py's job!

---

### 3. MetricRunner (metric_runner.py)
**Computes metrics from pre-saved simulation logs (offline evaluation).**

```python
class MetricRunner(AbstractRunner):
    """Manager which executes metrics with simulation logs."""

    def __init__(self, simulation_log: SimulationLog, metric_callback: MetricCallback):
        self._simulation_log = simulation_log
        self._metric_callback = metric_callback

    def run(self) -> RunnerReport:
        """
        Run metric engine on simulation log.
        :return Runner report (planner_report is always None).
        """
        # Compute metrics without re-running simulation
        run_metric_engine(
            metric_engine=self._metric_callback.metric_engine,
            scenario=self._simulation_log.scenario,
            history=self._simulation_log.simulation_history,
            planner_name=self._simulation_log.planner.name(),
        )
        # Return report
```

**Use case**: Re-run metrics without re-running expensive simulations
```bash
# 1. Run simulations with SimulationLogCallback
just simulate-ml  # Saves SimulationLog files

# 2. Later: re-compute metrics with different config
uv run python nuplan/planning/script/run_metric.py \
    simulation_log_dir=$NUPLAN_EXP_ROOT/simulation_logs \
    metric=new_metric_config
```

**Key differences from SimulationRunner**:
| Aspect | SimulationRunner | MetricRunner |
|--------|-----------------|--------------|
| Input | Scenario + Planner | SimulationLog (scenario + history) |
| Executes | Closed-loop simulation | Metric computation only |
| Output | History + Metrics | Metrics only |
| Duration | ~30s per scenario | ~5s per scenario |
| planner_report | Present | None (no planner ran) |

---

### 4. RunnerReport (runner_report.py)
**Dataclass containing execution results and metadata.**

```python
@dataclass
class RunnerReport:
    """Report for a runner."""

    # Execution status
    succeeded: bool  # True if simulation completed without exceptions
    error_message: Optional[str]  # None if succeeded, else full traceback
    start_time: float  # time.perf_counter() when run() started
    end_time: Optional[float]  # when run() returned (or None temporarily)

    # Planner performance stats
    planner_report: Optional[PlannerReport]  # Contains compute times, None if failed

    # Metadata for aggregation/filtering
    scenario_name: str  # e.g., "starting_left_turn_001"
    planner_name: str   # e.g., "simple_planner", "ml_planner"
    log_name: str       # e.g., "2021.05.12.22.00.38_veh-35_01008_01518"
```

**Typical values**:
```python
# Successful simulation
RunnerReport(
    succeeded=True,
    error_message=None,
    start_time=123456.789,
    end_time=123486.234,  # ~30s duration
    planner_report=PlannerReport(
        compute_trajectory_runtimes=[0.002, 0.003, ...],  # 200 values
        mean_step_time=0.0025,
        max_step_time=0.015
    ),
    scenario_name="starting_left_turn_001",
    planner_name="simple_planner",
    log_name="2021.05.12.22.00.38_veh-35_01008_01518"
)

# Failed simulation
RunnerReport(
    succeeded=False,
    error_message="Traceback (most recent call last):\n  File ...\nKeyError: 'velocity'",
    start_time=123456.789,
    end_time=123457.123,  # Failed fast (~0.3s)
    planner_report=None,
    scenario_name="near_multiple_vehicles_042",
    planner_name="ml_planner",
    log_name="2021.06.01.18.30.12_veh-42_02134_02987"
)
```

**AIDEV-NOTE**: error_message contains FULL traceback (can be >10KB for deep stacks)

---

### 5. Executor Pattern (executor.py)
**Handles parallelization and error isolation for batch execution.**

**Key functions**:

#### run_simulation()
**Wraps runner.run() to catch exceptions and convert to failed RunnerReport.**

```python
def run_simulation(sim_runner: AbstractRunner, exit_on_failure: bool = False) -> RunnerReport:
    """
    Proxy for calling simulation.
    :param sim_runner: A simulation runner.
    :param exit_on_failure: If true, raises exception (for debugging).
    :return: Report for the simulation.
    """
    start_time = time.perf_counter()
    try:
        return sim_runner.run()  # May raise exception
    except Exception as e:
        error = traceback.format_exc()

        # Log to console
        logger.warning("----------- Simulation failed with trace:")
        traceback.print_exc()
        logger.warning(f"Simulation failed with error:\n{e}")
        logger.warning(f"Failed simulation [log,token]:\n[{sim_runner.scenario.log_name}, {sim_runner.scenario.scenario_name}]")

        # Optionally crash (debugging mode)
        if exit_on_failure:
            raise RuntimeError('Simulation failed')

        # Return failed report
        return RunnerReport(
            succeeded=False,
            error_message=error,  # Full traceback
            start_time=start_time,
            end_time=time.perf_counter(),
            planner_report=None,
            scenario_name=sim_runner.scenario.scenario_name,
            planner_name=sim_runner.planner.name(),
            log_name=sim_runner.scenario.log_name,
        )
```

**Error handling strategy**:
```
┌─────────────────────────────────────────────────────┐
│  execute_runners()                                  │
│    │                                                │
│    ├─> WorkerPool.map(run_simulation, runners)     │
│    │     │                                          │
│    │     ├─> Runner 1: succeeded=True  ✓           │
│    │     ├─> Runner 2: EXCEPTION → report failed   │
│    │     ├─> Runner 3: succeeded=True  ✓           │
│    │     └─> Runner 4: EXCEPTION → report failed   │
│    │                                                │
│    └─> All 4 reports returned (2 success, 2 fail)  │
│                                                     │
│  Batch job completes even if some scenarios fail   │
└─────────────────────────────────────────────────────┘
```

---

#### execute_runners()
**Execute multiple runners with parallelization and callback awaiting.**

```python
def execute_runners(
    runners: List[AbstractRunner],
    worker: WorkerPool,
    num_gpus: Optional[Union[int, float]],
    num_cpus: Optional[int],
    exit_on_failure: bool = False,
    verbose: bool = False,
) -> List[RunnerReport]:
    """
    Execute multiple simulation runners or metric runners.
    :param runners: List of simulations to run.
    :param worker: WorkerPool for submitting tasks.
    :param num_gpus: Number (or fractional) of GPUs per simulation.
    :param num_cpus: Number of CPU threads per simulation.
    :param exit_on_failure: If true, crash on first error.
    :return: List of RunnerReports.
    """
    # 1. Run all simulations (parallel or sequential)
    reports = worker.map(
        Task(fn=run_simulation, num_gpus=num_gpus, num_cpus=num_cpus),
        runners,
        exit_on_failure,
        verbose=verbose
    )

    # 2. Store reports in dict for async callback updates
    results = {
        (report.scenario_name, report.planner_name, report.log_name): report
        for report in reports
    }

    # 3. Await async callbacks (metrics, serialization)
    # Find all MetricCallback and SimulationLogCallback futures
    callback_futures_map = {
        future: (simulation.scenario.scenario_name, runner.planner.name(), simulation.scenario.log_name)
        for runner in runners if isinstance(runner, SimulationRunner)
        for callback in runner.simulation.callback.callbacks
        if isinstance(callback, (MetricCallback, SimulationLogCallback))
        for future in callback.futures
    }

    # Block on futures, update reports if any fail
    for future in concurrent.futures.as_completed(callback_futures_map.keys()):
        try:
            future.result()  # Blocks until done, raises if exception
        except Exception:
            error_message = traceback.format_exc()
            runner_report = results[callback_futures_map[future]]
            runner_report.error_message = error_message
            runner_report.succeeded = False  # Mark report as failed
            runner_report.end_time = time.perf_counter()

    # 4. Log summary
    successful = sum(1 for r in reports if r.succeeded)
    failed = len(reports) - successful
    logger.info(f"Number of successful simulations: {successful}")
    logger.info(f"Number of failed simulations: {failed}")

    return list(results.values())
```

**Why await callbacks?**: Callbacks like MetricCallback submit async work (metric computation) that continues after simulation completes. We must wait for these to finish and capture any errors.

**CRITICAL**: A simulation can show `succeeded=True` initially, then fail later due to callback errors!

---

## Architecture

### System Context Diagram
```
┌───────────────────────────────────────────────────────────────────┐
│                    Main Script (run_simulation.py)                │
│                                                                   │
│  1. build_simulations(cfg) ──> List[SimulationRunner]            │
│  2. execute_runners(runners, worker) ──> List[RunnerReport]      │
│  3. save_runner_reports(reports, output_dir)                     │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌───────────────────────────────────────────────────────────────────┐
│                   execute_runners() (executor.py)                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  WorkerPool.map(run_simulation, runners) ──> reports        │ │
│  │    │                                                         │ │
│  │    ├─> Sequential: for runner in runners                    │ │
│  │    ├─> Parallel: ThreadPoolExecutor                         │ │
│  │    └─> Ray: ray.remote() across nodes/GPUs                  │ │
│  │                                                              │ │
│  │  for future in callback.futures:                            │ │
│  │    future.result()  # Block on async callbacks              │ │
│  └─────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌───────────────────────────────────────────────────────────────────┐
│              run_simulation(SimulationRunner)                     │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  try:                                                        │ │
│  │    runner.run() ──> RunnerReport(succeeded=True)            │ │
│  │  except Exception as e:                                     │ │
│  │    log traceback                                            │ │
│  │    return RunnerReport(succeeded=False, error_message=...)  │ │
│  └─────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌───────────────────────────────────────────────────────────────────┐
│              SimulationRunner.run() (simulations_runner.py)       │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  1. callback.on_simulation_start()                          │ │
│  │  2. _initialize() ─> planner.initialize()                   │ │
│  │  3. while simulation.is_simulation_running():               │ │
│  │       - callback.on_step_start()                            │ │
│  │       - planner_input = simulation.get_planner_input()      │ │
│  │       - trajectory = planner.compute_trajectory()           │ │
│  │       - callback.on_planner_end(trajectory)                 │ │
│  │       - simulation.propagate(trajectory)                    │ │
│  │       - callback.on_step_end(history.last())                │ │
│  │  4. callback.on_simulation_end(history)                     │ │
│  │  5. return RunnerReport(planner_report, timing, metadata)   │ │
│  └─────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────┘
```

### Simulation Loop Flow (SimulationRunner)

```python
def run(self) -> RunnerReport:
    start_time = time.perf_counter()

    # Create report (end_time filled later)
    report = RunnerReport(
        succeeded=True, error_message=None,
        start_time=start_time, end_time=None,
        planner_report=None,
        scenario_name=self._simulation.scenario.scenario_name,
        planner_name=self.planner.name(),
        log_name=self._simulation.scenario.log_name,
    )

    # Hook: on_simulation_start
    self.simulation.callback.on_simulation_start(self.simulation.setup)

    # Initialize planner (once)
    self._initialize()  # Calls on_initialization_start/end hooks

    # Main simulation loop (200+ iterations typically)
    while self.simulation.is_simulation_running():
        # Hook: on_step_start
        self.simulation.callback.on_step_start(self.simulation.setup, self.planner)

        # Get current state
        planner_input = self._simulation.get_planner_input()
        logger.debug("Simulation iteration: %s" % planner_input.iteration.index)

        # Hook: on_planner_start (timing starts here)
        self._simulation.callback.on_planner_start(self.simulation.setup, self.planner)

        # CORE PLANNER CALL
        trajectory = self.planner.compute_trajectory(planner_input)

        # Hook: on_planner_end (timing stops here)
        self._simulation.callback.on_planner_end(self.simulation.setup, self.planner, trajectory)

        # Propagate simulation state
        self.simulation.propagate(trajectory)

        # Hook: on_step_end
        self.simulation.callback.on_step_end(
            self.simulation.setup, self.planner, self.simulation.history.last()
        )

        # Update report if simulation just finished
        if not self.simulation.is_simulation_running():
            report.end_time = time.perf_counter()

    # Hook: on_simulation_end (metrics, serialization)
    self.simulation.callback.on_simulation_end(
        self.simulation.setup, self.planner, self.simulation.history
    )

    # Collect planner performance stats
    planner_report = self.planner.generate_planner_report()
    report.planner_report = planner_report

    return report
```

---

## Dependencies

### Imports From
- **nuplan/planning/scenario_builder/**: AbstractScenario (scenario interface)
- **nuplan/planning/simulation/planner/**: AbstractPlanner, PlannerReport
- **nuplan/planning/simulation/simulation.py**: Simulation (main simulation loop)
- **nuplan/planning/simulation/simulation_log.py**: SimulationLog (for MetricRunner)
- **nuplan/planning/simulation/callback/**: MetricCallback, SimulationLogCallback
- **nuplan/planning/utils/multithreading/**: WorkerPool, Task (parallelization)

### Used By
- **nuplan/planning/script/run_simulation.py**: Main entry point for batch simulations
- **nuplan/planning/script/run_metric.py**: Main entry point for offline metric computation
- **nuplan/planning/script/utils.py**: save_runner_reports(), build_simulations()

---

## Common Usage Patterns

### Pattern 1: Basic Sequential Execution

```python
from nuplan.planning.simulation.runner.simulations_runner import SimulationRunner
from nuplan.planning.simulation.runner.executor import execute_runners
from nuplan.planning.utils.multithreading.worker_sequential import Sequential

# Build runners (from scenario builder)
runners = [SimulationRunner(simulation, planner) for simulation, planner in ...]

# Execute sequentially (debugging)
worker = Sequential()
reports = execute_runners(
    runners=runners,
    worker=worker,
    num_gpus=None,
    num_cpus=None,
    exit_on_failure=True,  # Crash on first error
    verbose=True
)

# Check results
for report in reports:
    if report.succeeded:
        print(f"✓ {report.scenario_name}: {report.end_time - report.start_time:.2f}s")
    else:
        print(f"✗ {report.scenario_name}: {report.error_message}")
```

---

### Pattern 2: Parallel Execution with Thread Pool

```python
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor

# Execute in parallel (4 threads)
worker = SingleMachineParallelExecutor(
    use_process_pool=False,  # ThreadPoolExecutor (good for I/O)
    max_workers=4
)

reports = execute_runners(
    runners=runners,
    worker=worker,
    num_gpus=None,
    num_cpus=1,
    exit_on_failure=False,  # Continue on errors
    verbose=False
)

# Analyze success rate
successful = sum(1 for r in reports if r.succeeded)
print(f"Success rate: {successful}/{len(reports)} ({100*successful/len(reports):.1f}%)")
```

---

### Pattern 3: Ray Distributed Execution (GPU)

```python
from nuplan.planning.utils.multithreading.worker_ray import RayDistributed

# CRITICAL: Must use .venv/bin/python for Ray scripts!
# See CLAUDE.md "Lessons Learned - Ray/uv Integration Issues"

# Execute with Ray (GPU allocation)
worker = RayDistributed(
    threads_per_node=4,  # Recommended: 4 workers for 64GB RAM
    number_of_nodes=1,
    number_of_gpus_per_node=1,
    number_of_cpus_per_node=12
)

reports = execute_runners(
    runners=runners,
    worker=worker,
    num_gpus=0.25,  # Fractional GPU sharing (4 workers per GPU)
    num_cpus=1,
    exit_on_failure=False,
    verbose=True
)

# Ray auto-handles OOM retries
# Workers killed due to memory → Ray retries → eventual success
```

**AIDEV-NOTE**: See "Gotchas" section for Ray memory management details!

---

### Pattern 4: Offline Metric Computation

```python
from nuplan.planning.simulation.runner.metric_runner import MetricRunner
from nuplan.planning.simulation.simulation_log import SimulationLog
from nuplan.planning.simulation.callback.metric_callback import MetricCallback

# Load saved simulation logs
simulation_logs = [
    SimulationLog.load_data(Path(log_file))
    for log_file in log_dir.glob("*.msgpack.xz")
]

# Build metric runners
metric_callback = MetricCallback(metric_engine=my_metric_engine)
metric_runners = [
    MetricRunner(simulation_log=log, metric_callback=metric_callback)
    for log in simulation_logs
]

# Execute metric computation (fast, ~5s per scenario)
reports = execute_runners(
    runners=metric_runners,
    worker=worker,
    num_gpus=None,
    num_cpus=1,
    exit_on_failure=False
)

# Metrics saved to metric_engine.main_save_path
```

---

## Gotchas & Pitfalls

### 1. **Ray/uv Integration: MUST Use .venv/bin/python**

**Problem**: Ray's uv integration creates minimal worker environments without extras like `torch-cuda11`

**Symptoms**:
- `ModuleNotFoundError: No module named 'torch'` in Ray workers
- Workers show "Installed 165 packages" (missing many dependencies)
- RuntimeError about 'pip' or 'uv' runtime environments

**Root cause**: When using `uv run`, Ray's runtime_env creates a fresh uv environment in each worker, which:
1. Only installs base dependencies (not extras)
2. Ignores the parent venv with full dependencies
3. Cannot be customized without triggering errors

**Solution**:
```bash
# ❌ FAILS with Ray
uv run python nuplan/planning/script/run_simulation.py worker=ray_distributed

# ✅ WORKS with Ray
.venv/bin/python nuplan/planning/script/run_simulation.py worker=ray_distributed
```

**Reference**: See CLAUDE.md "Lessons Learned from Production Usage (2025-11-14)"

**AIDEV-NOTE**: This is the #1 production gotcha - cost hours of debugging!

---

### 2. **Ray Memory Management: OOM Kills with Too Many Workers**

**Problem**: Ray workers can overwhelm system memory, leading to OOM kills

**Symptoms**:
- Ray kills workers with "Task was killed due to node running low on memory"
- Memory usage at 95%+ threshold
- Workers get infinite retries (successful eventually but inefficient)

**Root causes**:
1. Default worker count = CPU count (12 workers on 12-core system)
2. Each worker holds model in memory (~1-2GB)
3. Background apps (Jupyter, IDEs, browsers) consume significant RAM
4. Ray's 95% memory threshold is conservative

**Solutions**:
1. **Reduce parallelism** (most effective):
   ```yaml
   worker.threads_per_node=4  # Instead of 12
   ```
2. **Close background applications** before simulation:
   - Jupyter notebooks: 2-4GB
   - Julia REPL sessions: 6GB+
   - IDEs (CLion, VSCode): 2-3GB each
   - Browsers (Firefox/Chrome): 4-8GB
3. **Adjust Ray memory threshold** (last resort):
   ```bash
   RAY_memory_usage_threshold=0.98  # Default 0.95
   ```

**Best practice for 64GB RAM**:
- 4 workers: Comfortable for most scenarios
- 8 workers: If background apps closed
- 12 workers: Only on clean system or 128GB+ RAM

**AIDEV-NOTE**: Production validation showed 4 workers optimal for 64GB with typical background apps

---

### 3. **Ray Disk Space: /tmp Fills Up with Sessions**

**Problem**: Ray fills `/tmp` with session data, causing "No space left on device"

**Symptoms**:
- Ray sessions under `/tmp/ray/session-*`
- Each session: ~10-50GB
- Stale sessions not auto-cleaned
- Simulation fails when `/tmp` is 100% full

**Solutions**:
1. **Redirect Ray temp directory** (add to `.env`):
   ```bash
   export RAY_TMPDIR="$HOME/.tmp/ray"
   ```
2. **Periodic cleanup**:
   ```bash
   # Clean old Ray sessions
   find ~/.tmp/ray -type d -mtime +7 -exec rm -rf {} +

   # Or use Justfile
   just clean-tmp
   ```
3. **Monitor disk usage**:
   ```bash
   df -h /tmp
   du -sh ~/.tmp/ray
   ```

**CRITICAL**: Must use `.venv/bin/python` (NOT `uv run`) to propagate `RAY_TMPDIR` env var!

**AIDEV-NOTE**: See `.env` file for RAY_TMPDIR configuration

---

### 4. **SimulationRunner.run() Does NOT Catch Exceptions**

```python
# ❌ Assumption: runner.run() never raises
try:
    report = runner.run()
except Exception:
    # This WILL trigger if run() crashes!
```

**Reality**: `run()` raises exceptions - `run_simulation()` wrapper catches them!

**Fix**: Always use `execute_runners()` or `run_simulation()` wrapper:
```python
# ✓ Correct usage
report = run_simulation(runner, exit_on_failure=False)  # Returns failed report
# or
reports = execute_runners([runner], worker, ...)  # Batch execution
```

---

### 5. **Callback Futures Must Be Awaited Before Exit**

```python
# ❌ BAD: Script exits before metrics written
reports = execute_runners(runners, worker)
sys.exit(0)  # Terminates worker pool → metrics lost!

# ✅ GOOD: execute_runners() awaits futures internally
reports = execute_runners(runners, worker)
# Futures already awaited here → metrics complete
```

**Why this matters**: MetricCallback and SimulationLogCallback submit async work that continues after `runner.run()` returns. If you exit early, this work is lost.

**AIDEV-NOTE**: execute_runners() handles future awaiting automatically (lines 105-114 in executor.py)

---

### 6. **Reports Can Show succeeded=True Then Fail Later**

```python
# Timeline:
# 1. runner.run() succeeds → report.succeeded=True
# 2. MetricCallback.on_simulation_end() submits async work
# 3. Metric computation fails (e.g., division by zero)
# 4. execute_runners() catches future exception → report.succeeded=False

# Final report: succeeded=False, but simulation itself didn't crash!
```

**Check `error_message`** to distinguish:
- Simulation failure: Stack trace includes `simulation.propagate()`, `planner.compute_trajectory()`
- Callback failure: Stack trace includes `metric_engine.compute()`, `serialize_to_disk()`

---

### 7. **RunnerReport.end_time May Be None Temporarily**

```python
# During simulation:
report = RunnerReport(..., end_time=None)

# Only set when simulation finishes:
if not self.simulation.is_simulation_running():
    report.end_time = time.perf_counter()

# ❌ BAD: Assume end_time always exists
duration = report.end_time - report.start_time  # TypeError if end_time=None!

# ✅ GOOD: Check or use saved reports (always have end_time)
if report.end_time is not None:
    duration = report.end_time - report.start_time
```

---

### 8. **execute_runners() Modifies Reports In-Place**

```python
# Reports returned from execute_runners():
reports = execute_runners(runners, worker)

# Later, async callback fails:
# execute_runners() mutates report.succeeded and report.error_message!

# ❌ BAD: Assume reports are immutable
cached_reports = reports  # Shares reference!
execute_runners(new_runners, worker)  # May mutate cached_reports!

# ✅ GOOD: Copy reports if needed
import copy
cached_reports = [copy.deepcopy(r) for r in reports]
```

---

### 9. **exit_on_failure Only Crashes on run() Exceptions**

```python
execute_runners(..., exit_on_failure=True)

# Crashes on:
# - Simulation initialization errors
# - Planner compute_trajectory() errors
# - Propagation errors

# Does NOT crash on:
# - Async callback failures (futures awaited after all simulations)
```

**Fix**: Check `reports` for failed callbacks after execute_runners() returns.

---

### 10. **Worker Thread Count vs CPU Count Mismatch**

```python
# Config:
worker:
  threads_per_node: 12  # Ray will spawn 12 workers

# But system only has 8 cores!
# Result: Context switching overhead, slower than threads_per_node=8
```

**Rule of thumb**:
- `threads_per_node` ≤ `cpu_count()` for CPU-bound planners
- `threads_per_node` ≤ `2 * cpu_count()` for I/O-bound planners

**Check CPU count**:
```python
import multiprocessing
print(f"CPU count: {multiprocessing.cpu_count()}")
```

---

### 11. **planner_report Is None If Simulation Fails Early**

```python
# If simulation crashes during initialization:
report = RunnerReport(
    succeeded=False,
    planner_report=None,  # Planner never ran!
    ...
)

# ❌ BAD: Assume planner_report always exists
mean_time = report.planner_report.mean_step_time  # AttributeError!

# ✅ GOOD: Check for None
if report.planner_report is not None:
    mean_time = report.planner_report.mean_step_time
```

---

### 12. **MetricCallback Created Per-Simulation (Stateful)**

```python
# ❌ Assumption: Share MetricCallback across runners
metric_callback = MetricCallback(metric_engine)
runners = [
    SimulationRunner(sim, planner, callback=metric_callback)  # Same instance!
    for sim in simulations
]
# Result: Callback futures overlap, metrics corrupted!

# ✅ Reality: MetricCallback created per-simulation in build_simulations()
# See simulation_builder.py lines 120-125
# Each runner gets its own callback instance (separate futures list)
```

**Reason**: MetricCallback stores per-simulation `_futures` list (stateful).

---

### 13. **Scenario Distribution Requires Synchronization (Multi-Node)**

```python
# Multi-node setup:
# Node 0: scenarios 0-500
# Node 1: scenarios 501-1000

# If node 0 finishes first and exits, node 1 may still be running!
# distributed_sync() in utils.py ensures all nodes wait at barrier.
```

**AIDEV-NOTE**: DistributedScenarioFilter handles this automatically via `node_rank`/`num_nodes`.

---

### 14. **Simulation Failures Don't Print Full Traceback to Console**

```python
# Only prints summary:
logger.warning(f"Simulation failed with error:\n{e}")

# Full traceback only in report.error_message
print(report.error_message)  # Full stack trace
```

**Fix**: Check `simulation_reports.parquet` or individual report objects for full error details.

---

### 15. **Ray Worker Timeout Can Cause Silent Failures**

**Problem**: Ray workers may time out on long simulations, causing silent task loss

**Symptoms**:
- Fewer reports returned than runners submitted
- No error message in logs
- Ray dashboard shows "lost workers"

**Solutions**:
1. **Increase timeout** (add to Ray config):
   ```python
   ray.init(
       _temp_dir=os.environ.get('RAY_TMPDIR'),
       object_store_memory=10**9,
       _system_config={
           'task_retry_delay_ms': 5000,
           'timeout_ms': 3600000  # 1 hour
       }
   )
   ```
2. **Monitor Ray dashboard**: `http://127.0.0.1:8265`
3. **Check for hung workers**: `ray status`

**AIDEV-NOTE**: Default timeout is often too short for ML planner first-run (model loading)

---

## Cross-References

### Upstream Dependencies
- **nuplan/planning/simulation/simulation.py** - Simulation class (propagate, history)
- **nuplan/planning/simulation/planner/** - AbstractPlanner interface
- **nuplan/planning/simulation/callback/** - AbstractCallback, MetricCallback, SimulationLogCallback
- **nuplan/planning/scenario_builder/** - AbstractScenario, scenario filtering
- **nuplan/planning/utils/multithreading/** - WorkerPool, Task, Ray integration

### Downstream Consumers
- **nuplan/planning/script/run_simulation.py** - Main entry point for simulations
- **nuplan/planning/script/run_metric.py** - Main entry point for metric re-computation
- **nuplan/planning/script/utils.py** - save_runner_reports(), build_simulations()

### Related Documentation
- **CLAUDE.md (root)**: Production usage lessons (Ray/uv, memory management, disk space)
- **nuplan/planning/simulation/CLAUDE.md**: Simulation class details
- **nuplan/planning/simulation/callback/CLAUDE.md**: Callback lifecycle and integration
- **nuplan/planning/simulation/planner/CLAUDE.md**: Planner interface

---

## Quick Reference Commands

```bash
# Sequential execution (debugging)
uv run python nuplan/planning/script/run_simulation.py \
    planner=simple_planner \
    worker=sequential \
    exit_on_failure=true

# Parallel execution (4 threads)
uv run python nuplan/planning/script/run_simulation.py \
    planner=simple_planner \
    worker=single_machine_thread_pool \
    worker.max_workers=4

# Ray distributed (GPU) - MUST use .venv/bin/python!
.venv/bin/python nuplan/planning/script/run_simulation.py \
    planner=ml_planner \
    worker=ray_distributed \
    worker.threads_per_node=4 \
    number_of_gpus_allocated_per_simulation=0.25

# Offline metric computation
uv run python nuplan/planning/script/run_metric.py \
    simulation_log_dir=$NUPLAN_EXP_ROOT/simulation_logs

# Justfile shortcuts (see CLAUDE.md)
just simulate              # Simple planner, sequential
just simulate-ml          # ML planner, Ray distributed (4 workers)
just nuboard              # Visualize results
```

---

## Performance Characteristics

### Execution Times (Typical)

| Worker Type | Setup Overhead | Per-Scenario Time | 31 Scenarios Total | Memory per Worker |
|-------------|----------------|-------------------|-------------------|-------------------|
| Sequential | ~1s | 30-35s | 15-18 minutes | 2-3GB (single) |
| Thread Pool (4 workers) | ~2s | 30-35s | 4-5 minutes | 2-3GB each (8-12GB total) |
| Ray (4 workers, GPU) | ~5s | 25-30s | 3-4 minutes | 2-3GB each + driver (10-15GB total) |

### Resource Allocation Matrix

| Worker Type | CPU per Sim | GPU per Sim | Max Parallel | Memory Footprint |
|-------------|-------------|-------------|--------------|------------------|
| Sequential | N/A | N/A | 1 | Low (~2GB) |
| Thread Pool | 1 | N/A | 4-8 | Medium (~8-16GB) |
| Process Pool | 1 | N/A | 4-8 | High (~16-32GB) |
| Ray (CPU) | 1 | 0 | threads_per_node | Medium (~8-24GB) |
| Ray (GPU) | 1 | 0.25 | 4 per GPU | High (~10-20GB) |

**AIDEV-NOTE**: Memory footprint includes model loading, history buffers, and metric computation

---

## Changelog

- **2025-11-15**: Initial Tier 2 documentation (Phase 2C)
  - Comprehensive coverage of runner system (6 files)
  - 15+ production-validated gotchas (Ray/uv, memory, disk space)
  - Integration with CLAUDE.md "Lessons Learned"
  - Complete execution patterns (Sequential/Thread/Ray)
  - Performance characteristics and resource allocation

---

**End of Documentation - Runner System**
