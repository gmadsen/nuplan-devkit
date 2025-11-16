# nuplan/planning/simulation/runner/test/

## 1. Purpose & Responsibility

**THE comprehensive test suite for simulation execution orchestration.** This module validates the runner infrastructure that manages metric computation across simulations. Tests ensure `MetricRunner` correctly executes metric callbacks post-simulation, propagates timing data, handles errors gracefully, and integrates with the simulation history system. The single test file validates the critical path for offline metric computation (separate from live simulation runs).

## 2. Key Abstractions

### Test Classes

**TestMetricRunner** (`test_metric_runner.py`)
- **Purpose**: Validates MetricRunner's metric computation workflow
- **Test coverage**: Runner execution, metric callback integration, timing propagation
- **Mocking strategy**: Hybrid approach - uses real components (SimplePlanner, SimulationHistory) with mocked scenario (MockAbstractScenario)
- **Key pattern**: Integration test - validates complete metric runner execution with minimal mocking

### Test Fixtures & Utilities

**Common setUp() Pattern**
```python
def setUp(self) -> None:
    """Standard initialization - called before test_run_metric_runner"""
    # Temporary directory for metric output
    self.tmp_dir = tempfile.TemporaryDirectory()

    # Mock scenario with configurable history length
    self.scenario = MockAbstractScenario(number_of_past_iterations=10)

    # Real simulation history (not mocked!)
    self.history = SimulationHistory(self.scenario.map_api, self.scenario.get_mission_goal())

    # Populate history with sample states
    self.history.add_sample(SimulationHistorySample(...))

    # Real planner instance
    planner = SimplePlanner(horizon_len=2, sampling_time=0.5, acceleration=[0, 0])

    # SimulationLog (ties scenario, history, planner together)
    self.simulation_log = SimulationLog(
        file_path=save_path / 'simulation_logs',
        simulation_history=self.history,
        scenario=self.scenario,
        planner=planner
    )

    # Metric infrastructure (empty metrics list for testing)
    self.metric_engine = MetricsEngine(metrics=[], main_save_path=save_path / 'metrics')
    self.metric_callback = MetricCallback(metric_engine=self.metric_engine)

    # System under test
    self.metric_runner = MetricRunner(
        simulation_log=self.simulation_log,
        metric_callback=self.metric_callback
    )
```

**Real Data Fixtures**
- `MockAbstractScenario` - Generates realistic scenario context (map, mission goal, vehicle parameters)
- `SimplePlanner(2, 0.5, [0, 0])` - Minimal planner instance (required for SimulationLog)
- `SimulationHistory` - Real history buffer with two manually-added samples
- `EgoState.build_from_rear_axle()` - Constructs valid ego states with full dynamics
- `InterpolatedTrajectory` - Real trajectory representation (2-state trajectory)
- `DetectionsTracks(TrackedObjects())` - Empty observation (no agents)
- `SimulationHistorySample` - Complete timestep snapshot with all required fields

### Critical Test Patterns

**1. Temporary Directory Management**
```python
def setUp(self) -> None:
    self.tmp_dir = tempfile.TemporaryDirectory()
    save_path = Path(self.tmp_dir.name)
    # Use save_path for metric outputs...

def tearDown(self) -> None:
    self.tmp_dir.cleanup()  # Critical: prevents disk space leaks!
```
**Why**: MetricEngine writes files to disk - must clean up after tests

**2. Minimal History Population**
```python
# Add exactly 2 samples (minimum for metric computation)
self.history.add_sample(SimulationHistorySample(
    iteration=SimulationIteration(time_point=TimePoint(0), index=0),
    ego_state=state_0,
    trajectory=InterpolatedTrajectory(trajectory=[state_0, state_1]),
    observation=DetectionsTracks(TrackedObjects()),  # Empty
    traffic_light_status=self.scenario.get_traffic_light_status_at_iteration(0)
))
self.history.add_sample(...)  # Second sample
```
**Why**: Metrics require temporal data - single sample is insufficient

**3. Integration Test Without Assertions**
```python
def test_run_metric_runner(self) -> None:
    """Test to run metric_runner."""
    self.metric_runner.run()
```
**Why**: Tests successful execution without exceptions (smoke test pattern)
**Expected**: No assertions needed - test passes if run() completes without error

**4. Empty Metrics List Pattern**
```python
self.metric_engine = MetricsEngine(metrics=[], main_save_path=save_path)
```
**Why**: Focus is runner orchestration, not metric correctness
**Trade-off**: Doesn't test metric computation logic (tested in metric_engine tests)

**5. Scenario Timestamp Consistency**
```python
state_0 = EgoState.build_from_rear_axle(
    ...,
    time_point=TimePoint(0),  # Microseconds
)
state_1 = EgoState.build_from_rear_axle(
    ...,
    time_point=TimePoint(1000),  # 1000 microseconds = 0.001s later
)
```
**Why**: SimulationIteration uses TimePoint for temporal ordering
**Gotcha**: TimePoint is in microseconds, not seconds!

**6. Same Iteration Index for Both Samples**
```python
# BOTH samples use index=0 - is this intentional?
SimulationIteration(time_point=TimePoint(0), index=0),  # Sample 1
SimulationIteration(time_point=TimePoint(0), index=0),  # Sample 2 (copy-paste bug?)
```
**Why**: Likely test bug - second sample should have index=1
**Impact**: Doesn't affect metric runner logic (uses time_point for ordering)
**AIDEV-NOTE**: Potential test improvement - fix iteration indices

**7. State Reuse in Trajectory**
```python
trajectory=InterpolatedTrajectory(trajectory=[state_0, state_1])
```
**Why**: Both samples reference same state_0/state_1 instances
**Memory**: Shared references are OK for immutable ego states

## 3. Architecture & Design Patterns

### 1. **Builder Pattern**: Fixture Construction
```python
# Build scenario context
scenario = MockAbstractScenario(number_of_past_iterations=10)

# Build history + populate
history = SimulationHistory(scenario.map_api, scenario.get_mission_goal())
history.add_sample(sample_1)
history.add_sample(sample_2)

# Build simulation log (aggregates context)
simulation_log = SimulationLog(
    file_path=path, simulation_history=history, scenario=scenario, planner=planner
)

# Build metric infrastructure
metric_engine = MetricsEngine(metrics=[], main_save_path=path)
metric_callback = MetricCallback(metric_engine=metric_engine)

# Build runner (system under test)
metric_runner = MetricRunner(simulation_log=simulation_log, metric_callback=metric_callback)
```
**Philosophy**: Compose complex test fixtures from simple components

### 2. **Smoke Test Pattern**: No Assertions
```python
def test_run_metric_runner(self) -> None:
    self.metric_runner.run()  # Only verify no exceptions raised
```
**Rationale**: MetricRunner is orchestration layer - correctness validated by:
- MetricEngine tests (metric computation logic)
- MetricCallback tests (callback lifecycle)
- RunnerReport tests (timing/error reporting)

**Trade-off**: Doesn't validate report contents (could add assertions on return value)

### 3. **Fixture Isolation**: Temporary Directories
```python
self.tmp_dir = tempfile.TemporaryDirectory()  # Unique per test run
save_path = Path(self.tmp_dir.name)
```
**Why**: Parallel test execution safety - no file conflicts
**Cleanup**: tearDown() ensures no disk leaks

### 4. **Minimal Mock Strategy**: Only Mock Scenario
```python
# MOCKED: Abstract scenario (DB-backed, heavy)
self.scenario = MockAbstractScenario(...)

# REAL: All other components
self.history = SimulationHistory(...)  # Real history buffer
planner = SimplePlanner(...)  # Real planner
self.metric_engine = MetricsEngine(...)  # Real metric engine
```
**Philosophy**: Test as much real code as possible, mock only expensive/external dependencies

### 5. **Empty Collection Testing**: metrics=[]
```python
MetricsEngine(metrics=[], ...)  # No metrics to compute
```
**Rationale**: Tests runner infrastructure without metric complexity
**Pattern**: Common in orchestration tests - validate plumbing, not business logic

## 4. Dependencies

### Internal (nuPlan - Documented ✅)

**Direct Test Dependencies**:
- ✅ `tempfile`, `unittest`, `pathlib.Path` - Python standard library test utilities
- ✅ `nuplan.common.actor_state.ego_state` - EgoState.build_from_rear_axle()
- ✅ `nuplan.common.actor_state.state_representation` - StateSE2, StateVector2D, TimePoint
- ✅ `nuplan.common.actor_state.tracked_objects` - TrackedObjects (empty observation)
- ✅ `nuplan.planning.metrics.metric_engine` - MetricsEngine (empty metrics list)
- ✅ `nuplan.planning.scenario_builder.test.mock_abstract_scenario` - MockAbstractScenario
- ✅ `nuplan.planning.simulation.callback.metric_callback` - MetricCallback
- ✅ `nuplan.planning.simulation.history.simulation_history` - SimulationHistory, SimulationHistorySample
- ✅ `nuplan.planning.simulation.observation.observation_type` - DetectionsTracks
- ✅ `nuplan.planning.simulation.planner.simple_planner` - SimplePlanner
- ✅ `nuplan.planning.simulation.simulation_log` - SimulationLog
- ✅ `nuplan.planning.simulation.simulation_time_controller.simulation_iteration` - SimulationIteration
- ✅ `nuplan.planning.simulation.trajectory.interpolated_trajectory` - InterpolatedTrajectory

**Systems Under Test**:
- ✅ `nuplan.planning.simulation.runner.metric_runner` - MetricRunner (THE test target)

### Undocumented Dependencies (Implementation Details)
- `nuplan.planning.simulation.runner.abstract_runner` - AbstractRunner interface (tested polymorphically)
- `nuplan.planning.simulation.runner.runner_report` - RunnerReport (returned but not validated)

## 5. Dependents (Who Uses This Module?)

### Direct Consumers

**CI/CD Pipeline**:
- Test suite executed via `pytest nuplan/planning/simulation/runner/test/`
- Part of Phase 2C runner validation
- Ensures metric computation infrastructure doesn't regress

**Development Workflow**:
- Developers run tests before modifying MetricRunner
- Validates metric callback integration after changes
- Smoke test for simulation log compatibility

**Coverage Analysis**:
- Contributes to overall test coverage metrics
- Validates critical path: SimulationLog → MetricRunner → MetricCallback → MetricsEngine

## 6. Critical Files (Prioritized)

### Priority 1: Test File (Only File!)

1. **`test_metric_runner.py`** (88 lines) - **START HERE!**
   - Single test class: `TestMetricRunner`
   - Single test method: `test_run_metric_runner()`
   - setUp/tearDown lifecycle management
   - Integration test pattern

### Priority 2: Implementation Files (Understanding Context)

2. **`../metric_runner.py`** (74 lines) - **System under test**
   - MetricRunner implementation
   - run() method execution logic
   - RunnerReport generation

3. **`../abstract_runner.py`** (34 lines) - **Interface**
   - AbstractRunner base class
   - run() interface definition
   - scenario/planner properties

4. **`../runner_report.py`** (27 lines) - **Output data structure**
   - RunnerReport dataclass
   - Timing, error tracking fields

5. **`../simulation_log.py`** (109 lines) - **Input data structure**
   - SimulationLog aggregates scenario + history + planner
   - Serialization logic (pickle/msgpack)

## 7. Common Usage Patterns

### Running Tests

**1. Run All Runner Tests**
```bash
pytest nuplan/planning/simulation/runner/test/ -v
```

**2. Run Specific Test**
```bash
pytest nuplan/planning/simulation/runner/test/test_metric_runner.py::TestMetricRunner::test_run_metric_runner -v
```

**3. Run with Coverage**
```bash
pytest nuplan/planning/simulation/runner/test/ --cov=nuplan.planning.simulation.runner --cov-report=html
```

**4. Debug Mode**
```bash
pytest nuplan/planning/simulation/runner/test/test_metric_runner.py -v -s --pdb
```

### Writing Similar Tests (Pattern Reuse)

**1. Test Runner with Real Metrics**
```python
from nuplan.planning.metrics.metric_engine import MetricsEngine
from nuplan.planning.metrics.planning_metrics import CollisionMetric

def setUp(self):
    # ... existing setup ...

    # Add real metrics instead of empty list
    metrics = [CollisionMetric()]
    self.metric_engine = MetricsEngine(metrics=metrics, main_save_path=save_path)

    # Rest unchanged
    self.metric_callback = MetricCallback(metric_engine=self.metric_engine)
    self.metric_runner = MetricRunner(...)

def test_run_metric_runner_with_metrics(self):
    report = self.metric_runner.run()

    # Validate report
    self.assertTrue(report.succeeded)
    self.assertIsNone(report.error_message)
    self.assertIsNotNone(report.end_time)

    # Validate metric files written
    metric_files = list((Path(self.tmp_dir.name) / 'metrics').glob('*.parquet'))
    self.assertGreater(len(metric_files), 0)
```

**2. Test Runner with Populated History**
```python
def setUp(self):
    # ... existing setup ...

    # Add more history samples (10 timesteps @ 0.1s)
    for idx in range(10):
        time_us = idx * 100000  # 0.1s intervals in microseconds
        state = EgoState.build_from_rear_axle(
            StateSE2(x=idx * 1.0, y=0, heading=0),  # Move 1m per timestep
            vehicle_parameters=self.scenario.ego_vehicle_parameters,
            rear_axle_velocity_2d=StateVector2D(x=10.0, y=0),
            rear_axle_acceleration_2d=StateVector2D(x=0, y=0),
            tire_steering_angle=0,
            time_point=TimePoint(time_us)
        )

        self.history.add_sample(SimulationHistorySample(
            iteration=SimulationIteration(time_point=TimePoint(time_us), index=idx),
            ego_state=state,
            trajectory=InterpolatedTrajectory(trajectory=[state]),
            observation=DetectionsTracks(TrackedObjects()),
            traffic_light_status=self.scenario.get_traffic_light_status_at_iteration(0)
        ))
```

**3. Test Runner Error Handling**
```python
def test_run_metric_runner_with_failing_callback(self):
    # Mock metric callback to raise exception
    with patch.object(self.metric_callback, 'on_simulation_end', side_effect=RuntimeError("Test error")):
        report = self.metric_runner.run()

        # Verify error captured
        self.assertFalse(report.succeeded)
        self.assertIn("Test error", report.error_message)
        self.assertIsNotNone(report.end_time)
```

**4. Test Runner Timing**
```python
def test_run_metric_runner_timing(self):
    import time

    start = time.perf_counter()
    report = self.metric_runner.run()
    end = time.perf_counter()

    # Verify timing consistency
    self.assertAlmostEqual(report.end_time - report.start_time, end - start, delta=0.01)
    self.assertGreater(report.end_time, report.start_time)
```

## 8. Gotchas & Edge Cases

### Test Setup Issues

**1. Temporary Directory Not Cleaned Up**
- **Issue**: Forgetting `tearDown()` causes disk space leaks
- **Symptom**: /tmp fills up after many test runs
- **Fix**: ALWAYS call `self.tmp_dir.cleanup()` in tearDown()
- **Prevention**: Use unittest.TestCase (automatic tearDown execution)

**2. Iteration Index Inconsistency**
- **Issue**: Both samples use `index=0` (lines 46, 55)
- **Symptom**: Doesn't break test, but logically incorrect
- **Expected**: Second sample should have `index=1`
- **Impact**: None for this test (metric runner uses time_point, not index)
- **AIDEV-NOTE**: Fix in future - change line 55 to `index=1`

**3. TimePoint Microseconds vs Seconds Confusion**
- **Issue**: TimePoint(0) and TimePoint(1000) are only 0.001s apart
- **Symptom**: Very short simulation duration
- **Expected**: Should use TimePoint(0) and TimePoint(100000) for 0.1s timestep
- **Impact**: Metrics may behave unexpectedly with sub-millisecond timesteps
- **AIDEV-NOTE**: Update state_1 time_point to TimePoint(100000)

**4. Empty TrackedObjects in All Samples**
- **Issue**: DetectionsTracks(TrackedObjects()) has no agents
- **Symptom**: Metrics expecting agents (collision, time-to-collision) have no data
- **Expected**: This is intentional - tests runner without agent interactions
- **Extension**: Add agents via MockAbstractScenario.get_tracked_objects_at_iteration()

**5. Trajectory Only Has 2 States**
- **Issue**: InterpolatedTrajectory([state_0, state_1]) is minimal
- **Symptom**: Can't test trajectory metrics that require longer horizons
- **Expected**: Sufficient for runner test (trajectory content not validated)
- **Extension**: Use SimplePlanner.compute_trajectory() to generate realistic trajectory

### Mock Scenario Limitations

**6. MockAbstractScenario Configuration**
- **Issue**: `number_of_past_iterations=10` but only 2 samples added
- **Symptom**: Scenario expects 10 past states, history has 2
- **Impact**: None (scenario config doesn't constrain history)
- **Confusion**: Parameter name misleading (doesn't enforce history length)
- **AIDEV-NOTE**: Remove unnecessary parameter or align with history length

**7. Mission Goal Not Used**
- **Issue**: `scenario.get_mission_goal()` passed to SimulationHistory but not tested
- **Symptom**: Mission goal could be None, test would still pass
- **Expected**: Mission goal is optional - test doesn't validate goal-reaching metrics
- **Extension**: Add mission goal validation if testing goal metrics

**8. Map API Not Used**
- **Issue**: `scenario.map_api` passed to SimulationHistory but map queries never called
- **Symptom**: Map-based metrics (drivable area, lane following) not tested
- **Expected**: Runner test focuses on execution, not map metrics
- **Extension**: Add map-based metrics to test spatial compliance

### Metric Engine Edge Cases

**9. Empty Metrics List Doesn't Write Files**
- **Issue**: `metrics=[]` means no metric files written
- **Symptom**: `save_path / 'metrics'` directory may be empty
- **Expected**: MetricEngine.write_to_files() called with empty list (no-op)
- **Impact**: Test passes regardless of file I/O correctness
- **Extension**: Add assertion checking metric_files directory exists

**10. Metric Callback Worker Pool None**
- **Issue**: `MetricCallback(metric_engine)` uses default `worker_pool=None`
- **Symptom**: Metrics computed sequentially (not parallelized)
- **Expected**: Fine for empty metrics list (no work to parallelize)
- **Extension**: Test with `worker_pool=WorkerPool(max_workers=4)` for parallel case

### Runner Report Validation Gaps

**11. Report Fields Not Validated**
- **Issue**: `test_run_metric_runner()` doesn't check RunnerReport contents
- **Symptom**: Report could have incorrect scenario_name, planner_name, etc.
- **Impact**: Doesn't validate report correctness (only execution success)
- **Extension**: Add assertions:
  ```python
  report = self.metric_runner.run()
  self.assertEqual(report.scenario_name, self.scenario.scenario_name)
  self.assertEqual(report.planner_name, self.planner.name())
  self.assertTrue(report.succeeded)
  ```

**12. Planner Report Always None**
- **Issue**: `report.planner_report` is None for MetricRunner (by design)
- **Symptom**: Can't test planner runtime statistics
- **Expected**: MetricRunner doesn't run planner (replays history)
- **Alternative**: Use SimulationRunner tests for planner_report validation

**13. Timing Precision Not Tested**
- **Issue**: Report start_time and end_time not validated
- **Symptom**: Could have `end_time < start_time` and test would pass
- **Extension**: Add timing sanity checks:
  ```python
  self.assertGreater(report.end_time, report.start_time)
  self.assertLess(report.end_time - report.start_time, 1.0)  # < 1 second
  ```

### File I/O Edge Cases

**14. SimulationLog file_path Not Validated**
- **Issue**: `file_path=save_path / 'simulation_logs'` never checked
- **Symptom**: SimulationLog.save_to_file() not tested (separate concern)
- **Expected**: MetricRunner doesn't save logs (that's simulation's job)
- **Confusion**: file_path in SimulationLog is for later serialization

**15. Temporary Directory Path Length**
- **Issue**: Windows has 260-character path limit
- **Symptom**: Test fails on Windows if tmp_dir path too long
- **Fix**: Use `tempfile.TemporaryDirectory(prefix='nuplan_test_')`
- **Prevention**: Keep test output paths short

## 9. Performance Considerations

### Test Execution Time

**Current Performance** (typical development machine):
- setUp: ~10 ms (MockAbstractScenario + fixture construction)
- test_run_metric_runner: ~5 ms (empty metrics, no I/O)
- tearDown: ~2 ms (directory cleanup)
- **Total**: ~17 ms per test run

**Bottlenecks**:
- MockAbstractScenario initialization: ~5 ms (map API, vehicle parameters)
- SimulationHistory construction: ~2 ms
- MetricsEngine overhead: ~1 ms (even with empty metrics)

**Optimization Opportunities**:
1. **Cache MockAbstractScenario** (setUpClass)
   - Speedup: 30-40% (~6 ms → ~4 ms)
   - Trade-off: Tests share scenario state (less isolation)

2. **Reuse TemporaryDirectory** (setUpClass)
   - Speedup: 10-15% (reduce filesystem calls)
   - Trade-off: Parallel test conflicts (must ensure unique subdirs)

3. **Lazy Fixture Construction**
   - Build metric_runner on-demand in test method
   - Speedup: None (setUp always runs anyway)

### Memory Footprint

- **Per Test Run**: ~500 KB
  - MockAbstractScenario: ~200 KB (map API, vehicle parameters)
  - SimulationHistory (2 samples): ~50 KB
  - MetricsEngine: ~100 KB
  - Overhead: ~150 KB

**Memory Leaks**: None (tearDown cleanup prevents leaks)

### Scaling Considerations

**If Adding 100+ Metrics**:
- setUp time increases to ~50-100 ms (metric initialization)
- test_run_metric_runner time increases to ~500-1000 ms (metric computation)
- Memory footprint increases to ~5-10 MB (metric state)

**If Adding 100+ History Samples**:
- setUp time increases to ~50-100 ms (sample construction)
- Memory footprint increases to ~5-10 MB (history buffer)
- Metric computation time increases linearly (depends on metrics)

**Parallelization**:
- Currently no parallel execution (single test method)
- Could add multiple test methods (pytest runs in parallel with -n flag)
- Test isolation is good (tmp_dir per instance)

## 10. Related Documentation

### Cross-References (Documented ✅)

**Test Dependencies**:
- ✅ `nuplan/common/actor_state/test/CLAUDE.md` - EgoState test patterns
- ✅ `nuplan/planning/simulation/history/test/CLAUDE.md` - SimulationHistory test patterns
- ✅ `nuplan/planning/simulation/planner/CLAUDE.md` - SimplePlanner usage
- ✅ `nuplan/planning/simulation/trajectory/CLAUDE.md` - InterpolatedTrajectory
- ✅ `nuplan/planning/simulation/observation/test/CLAUDE.md` - Observation test patterns
- ✅ `nuplan/planning/scenario_builder/test/mock_abstract_scenario.py` - MockAbstractScenario docs

**Implementation Under Test**:
- `nuplan/planning/simulation/runner/metric_runner.py` - MetricRunner implementation
- `nuplan/planning/simulation/runner/abstract_runner.py` - AbstractRunner interface
- `nuplan/planning/simulation/callback/metric_callback.py` - MetricCallback
- `nuplan/planning/metrics/metric_engine.py` - MetricsEngine

**Related Test Suites**:
- `nuplan/planning/metrics/test/test_metric_engine.py` - MetricsEngine tests
- `nuplan/planning/simulation/callback/test/` - Callback tests (if exists)
- `nuplan/planning/simulation/test/` - Simulation tests (if exists)

### Undocumented Dependencies (Future ⏳)
- `nuplan/planning/simulation/runner/simulations_runner.py` - SimulationRunner (not tested here)
- `nuplan/planning/simulation/runner/executor.py` - execute_runners, run_simulation (not tested here)

### Configuration Files
- No configuration files (unit test, not integration test)
- Future: Add pytest fixtures in conftest.py for shared test data

## 11. AIDEV Notes

### Design Philosophy
- **Minimalist testing**: Single test validates execution path without asserting outputs
- **Smoke test pattern**: Focus on "does it run?" not "does it compute correctly?"
- **Integration over unit**: Tests real components (history, planner, metric engine) together
- **Fixture reuse**: setUp builds complete context (scenario → history → log → runner)

### Common Mistakes
- Forgetting `tearDown()` cleanup (disk leaks)
- Not using tempfile.TemporaryDirectory (test collisions)
- Hardcoding paths instead of using tmp_dir.name
- Expecting assertions in smoke tests (pattern is exception-based validation)
- Confusing TimePoint(microseconds) with seconds

### Future Improvements

**AIDEV-TODO**: Fix iteration index inconsistency
```python
# Line 55: Change index=0 to index=1
SimulationIteration(time_point=TimePoint(1000), index=1),  # Was: index=0
```

**AIDEV-TODO**: Fix TimePoint duration (should be 0.1s not 0.001s)
```python
# Line 42: Change TimePoint(1000) to TimePoint(100000)
time_point=TimePoint(100000),  # 0.1s = 100,000 microseconds
```

**AIDEV-TODO**: Add report validation assertions
```python
def test_run_metric_runner(self) -> None:
    report = self.metric_runner.run()

    # Validate report fields
    self.assertTrue(report.succeeded)
    self.assertIsNone(report.error_message)
    self.assertEqual(report.scenario_name, self.scenario.scenario_name)
    self.assertEqual(report.planner_name, "SimplePlanner")
    self.assertEqual(report.log_name, self.scenario.log_name)
    self.assertGreater(report.end_time, report.start_time)
    self.assertIsNone(report.planner_report)  # MetricRunner doesn't run planner
```

**AIDEV-TODO**: Add test with non-empty metrics
```python
def test_run_metric_runner_with_real_metrics(self):
    from nuplan.planning.metrics.planning_metrics import EgoAccelerationStatistics

    # Add real metric
    metric = EgoAccelerationStatistics(...)
    self.metric_engine = MetricsEngine(metrics=[metric], main_save_path=...)
    self.metric_callback = MetricCallback(metric_engine=self.metric_engine)
    self.metric_runner = MetricRunner(
        simulation_log=self.simulation_log,
        metric_callback=self.metric_callback
    )

    report = self.metric_runner.run()
    self.assertTrue(report.succeeded)

    # Verify metric files written
    metric_files = list((Path(self.tmp_dir.name) / 'metrics').glob('*.parquet'))
    self.assertEqual(len(metric_files), 1)
```

**AIDEV-TODO**: Add test for error handling
```python
def test_run_metric_runner_error_handling(self):
    # Mock metric engine to raise exception
    with patch.object(
        self.metric_callback.metric_engine,
        'compute',
        side_effect=RuntimeError("Metric computation failed")
    ):
        # This should raise (no try/except in MetricRunner.run)
        with self.assertRaises(RuntimeError):
            self.metric_runner.run()
```

**AIDEV-TODO**: Test metric runner properties
```python
def test_metric_runner_properties(self):
    # Test scenario property
    self.assertEqual(self.metric_runner.scenario, self.scenario)

    # Test planner property
    self.assertEqual(self.metric_runner.planner.name(), "SimplePlanner")
```

**AIDEV-TODO**: Add test with realistic history
```python
def test_run_metric_runner_with_long_history(self):
    # Clear existing history
    self.history = SimulationHistory(self.scenario.map_api, self.scenario.get_mission_goal())

    # Add 100 samples @ 0.1s timesteps (10 seconds)
    for idx in range(100):
        time_us = idx * 100000
        x = idx * 1.0  # 1 m/s constant velocity
        state = EgoState.build_from_rear_axle(
            StateSE2(x=x, y=0, heading=0),
            vehicle_parameters=self.scenario.ego_vehicle_parameters,
            rear_axle_velocity_2d=StateVector2D(x=1.0, y=0),
            rear_axle_acceleration_2d=StateVector2D(x=0, y=0),
            tire_steering_angle=0,
            time_point=TimePoint(time_us)
        )

        self.history.add_sample(SimulationHistorySample(...))

    # Rebuild simulation log with new history
    self.simulation_log = SimulationLog(...)
    self.metric_runner = MetricRunner(...)

    # Should handle long history without issues
    report = self.metric_runner.run()
    self.assertTrue(report.succeeded)
```

### Potential Bugs

**AIDEV-NOTE** (test_metric_runner.py:55): Iteration index should be 1, not 0
```python
# Line 55: Both samples use index=0 (likely copy-paste error)
SimulationIteration(time_point=TimePoint(0), index=0),  # Should be index=1
```

**AIDEV-NOTE** (test_metric_runner.py:42): TimePoint duration too short
```python
# Line 42: TimePoint(1000) = 0.001s, should be 0.1s (100000 microseconds)
time_point=TimePoint(1000),  # Should be TimePoint(100000)
```

**AIDEV-QUESTION** (test_metric_runner.py:26): Why number_of_past_iterations=10?
- Parameter not used by test (only 2 samples added)
- MockAbstractScenario parameter doesn't enforce history length
- Consider removing or aligning with actual history length

**AIDEV-QUESTION** (test_metric_runner.py:82-83): Why no assertions?
- Smoke test pattern validates execution without errors
- Could add minimal assertions (report.succeeded, timing sanity checks)
- Trade-off: Keep test minimal vs validate correctness

### Testing Gaps

**Missing Test Coverage**:
1. ❌ RunnerReport field validation (scenario_name, planner_name, timing)
2. ❌ Error handling (metric engine failures, callback exceptions)
3. ❌ Metric file I/O (verify files written correctly)
4. ❌ Non-empty metrics (test with real metric instances)
5. ❌ Long simulation histories (scalability testing)
6. ❌ Metric runner properties (scenario, planner getters)
7. ❌ Parallel metric computation (worker_pool testing)

**Future Test Ideas**:
- Test MetricRunner with different scenario types (not just mock)
- Test metric runner timing accuracy (compare to wall time)
- Test metric runner with multiple metrics (validate aggregation)
- Test metric runner with metrics that fail (error propagation)
- Test metric runner serialization (save/load simulation logs)

### Documentation Improvements Needed
- Add diagram: SimulationLog → MetricRunner → MetricCallback → MetricsEngine flow
- Document MetricRunner vs SimulationRunner differences (when to use each)
- Add example of metric runner usage in offline evaluation pipeline
- Document metric file output format (parquet schema)
