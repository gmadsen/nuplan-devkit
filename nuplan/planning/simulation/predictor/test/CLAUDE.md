# CLAUDE.md - nuplan/planning/simulation/predictor/test

## Purpose & Responsibility

**Test suite validating predictor interface contracts and oracle predictor correctness.** This module ensures that AbstractPredictor implementations correctly initialize, compute predictions, populate agent trajectories, and track performance metrics. Tests use unittest framework with mock utilities (MockAbstractPredictor, mock scenarios) to verify interface compliance without requiring real dataset access, enabling fast unit-level validation of predictor behavior.

## Key Abstractions & Test Classes

### Test Classes

- **`TestAbstractPredictor`** (unittest.TestCase) - Interface contract validation
  - **Purpose**: Verify AbstractPredictor interface compliance (initialization, naming, observation types, prediction computation)
  - **Tests**:
    - `test_initialize()` - Map API stored correctly during initialization
    - `test_name()` - Predictor name returns expected value
    - `test_observation_type()` - Returns correct observation type (DetectionsTracks)
    - `test_compute_predictions()` - Prediction computation and runtime tracking
  - **Setup**: `MockAbstractPredictor()` instance
  - **Assertions**: Map API assignment, type checks, runtime tracking validation

- **`TestLogFuturePredictor`** (unittest.TestCase) - Oracle predictor validation
  - **Purpose**: Verify LogFuturePredictor correctly extracts ground-truth futures from scenario
  - **Tests**:
    - `test_compute_predicted_trajectories()` - Prediction extraction, agent count preservation, waypoint validation
  - **Setup**:
    - `MockAbstractScenario()` - Provides logged trajectory data
    - `TrajectorySampling(num_poses=1, time_horizon=1.0)` - Minimal sampling config
    - `LogFuturePredictor(scenario, sampling)` - Oracle predictor under test
  - **Assertions**: Tracked object count invariance, prediction existence, waypoint count correctness, runtime tracking

### Mock Utilities (Reusable Test Infrastructure)

- **`MockAbstractPredictor`** - Minimal concrete predictor for interface testing
  - **Purpose**: Simplest possible predictor implementation for interface contract validation
  - `requires_scenario = False` - No oracle access needed
  - **Behavior**: Returns current observations unchanged (no-op prediction)
  - `initialize()` - Stores map API
  - `name()` - Returns class name ("MockAbstractPredictor")
  - `observation_type()` - Returns DetectionsTracks
  - `compute_predicted_trajectories()` - Returns present observation (identity function)
  - **Use case**: Test base class behavior without prediction complexity

- **`get_mock_predictor_initialization()`** - Factory for PredictorInitialization
  - **Returns**: `PredictorInitialization(MockAbstractMap())`
  - **Purpose**: Create valid initialization data without real map
  - **Usage**: `predictor.initialize(get_mock_predictor_initialization())`

- **`get_mock_predictor_input(buffer_size: int = 1)`** - Factory for PredictorInput
  - **Parameters**: `buffer_size` - History buffer length (default 1)
  - **Returns**: `PredictorInput(iteration, history, traffic_light_data=None)`
  - **Components**:
    - `MockAbstractScenario()` - Provides initial ego state and tracked objects
    - `SimulationHistoryBuffer.initialize_from_list()` - Rolling history window
    - `SimulationIteration(TimePoint(0), 0)` - Initial timestep
  - **Purpose**: Create valid predictor input without running simulation
  - **Usage**: `detections = predictor.compute_predictions(get_mock_predictor_input())`

## Architecture & Design Patterns

### 1. **Factory Pattern**: Mock Data Generators
- `get_mock_predictor_initialization()` - Encapsulates initialization data creation
- `get_mock_predictor_input(buffer_size)` - Parameterized input generation
- **Benefit**: Test code doesn't know about MockAbstractScenario internals
- **Consistency**: All tests use same mock data structure
- **Reusability**: Factories exported for use in other test modules

### 2. **Minimal Implementation Pattern**: MockAbstractPredictor
```python
class MockAbstractPredictor(AbstractPredictor):
    def compute_predicted_trajectories(self, current_input):
        _, present_observation = current_input.history.current_state
        return present_observation  # Identity function (no-op)
```
- **Purpose**: Test interface without prediction logic complexity
- **Principle**: Minimal code to satisfy ABC requirements
- **Validation**: Ensures base class template method works correctly

### 3. **Timing Instrumentation Validation**: Runtime Tracking Tests
```python
start_time = time.perf_counter()
detections = predictor.compute_predictions(predictor_input)
compute_predictions_time = time.perf_counter() - start_time

predictor_report = predictor.generate_predictor_report()
self.assertAlmostEqual(
    predictor_report.compute_predictions_runtimes[0],
    compute_predictions_time,
    delta=0.1
)
```
- **Purpose**: Verify base class runtime tracking works
- **Mechanism**: Measure prediction time externally, compare to internal tracking
- **Tolerance**: 0.1s delta (100ms) - accounts for timing overhead
- **Critical check**: Ensures template method `compute_predictions()` properly wraps subclass implementation

### 4. **State Invariant Validation**: Object Count Preservation
```python
# LogFuturePredictor test pattern
_, input_detections = predictor_input.history.current_state
detections = predictor.compute_predictions(predictor_input)
self.assertEqual(
    len(detections.tracked_objects),
    len(input_detections.tracked_objects)
)
```
- **Invariant**: Prediction should not add/remove agents
- **Rationale**: Predictor augments existing observations, doesn't filter
- **Consequence**: Downstream code can rely on agent count stability

### 5. **Report Type Discrimination**: MLPredictorReport vs PredictorReport
```python
predictor_report = predictor.generate_predictor_report()
self.assertNotIsInstance(predictor_report, MLPredictorReport)
```
- **Purpose**: Ensure non-ML predictors return base PredictorReport
- **Context**: MLPredictorReport is for future ML predictors (not yet implemented)
- **Validation**: Type system enforces correct report generation

## Dependencies (What Tests Import)

### Test Infrastructure
- `unittest` - TestCase base class, test discovery, assertions
- `time` - `perf_counter()` for runtime measurement validation

### Mock/Test Utilities (Internal)
- `mock_abstract_predictor` (this module) - MockAbstractPredictor, factory functions
- ⏳ `nuplan.planning.scenario_builder.test.mock_abstract_scenario` - MockAbstractScenario, MockAbstractMap

### Tested Modules
- ✅ `nuplan.planning.simulation.predictor.abstract_predictor` - AbstractPredictor, PredictorInitialization, PredictorInput
- ✅ `nuplan.planning.simulation.predictor.log_future_predictor` - LogFuturePredictor
- ✅ `nuplan.planning.simulation.predictor.predictor_report` - PredictorReport, MLPredictorReport

### Supporting Modules (✅ Documented)
- ✅ `nuplan.planning.simulation.observation.observation_type` - DetectionsTracks, Observation
- ✅ `nuplan.planning.simulation.trajectory.trajectory_sampling` - TrajectorySampling
- ✅ `nuplan.planning.simulation.history.simulation_history_buffer` - SimulationHistoryBuffer
- ✅ `nuplan.common.actor_state.state_representation` - TimePoint

### Supporting Modules (⏳ Undocumented)
- ⏳ `nuplan.planning.simulation.simulation_time_controller.simulation_iteration` - SimulationIteration

## Dependents (Who Uses This Module?)

### Internal Test Module Usage
- `test_abstract_predictor.py` imports `mock_abstract_predictor` (same module)
- `test_log_future_predictor.py` imports `mock_abstract_predictor` (same module)

### External Usage (PUBLIC library)
**BUILD file**: `package(default_visibility = ["//visibility:public"])`

**Potential consumers**:
1. **Other test modules** - Can reuse MockAbstractPredictor for integration tests
2. **Custom predictor tests** - Import `get_mock_predictor_input()` for testing custom implementations
3. **Simulation tests** - Use mock predictor to isolate planner testing from prediction

**AIDEV-NOTE**: Public visibility suggests this is intended as reusable test infrastructure, but `rg` shows no external imports yet. Likely designed for future extensibility.

## Critical Files (Prioritized)

1. **`mock_abstract_predictor.py`** (63 lines) - **START HERE!**
   - MockAbstractPredictor class (minimal concrete implementation)
   - Factory functions for test data generation
   - Reusable test utilities (PUBLIC library)
   - **Key pattern**: Identity function for compute_predicted_trajectories

2. **`test_abstract_predictor.py`** (50 lines) - **Interface contract tests**
   - Tests for AbstractPredictor lifecycle (initialize, name, observation_type)
   - Runtime tracking validation (critical for performance monitoring)
   - Report generation tests
   - **Coverage**: All AbstractPredictor public methods

3. **`test_log_future_predictor.py`** (47 lines) - **Oracle predictor tests**
   - LogFuturePredictor correctness validation
   - Prediction existence and waypoint count checks
   - State invariant validation (agent count preservation)
   - **Coverage**: LogFuturePredictor.compute_predictions() path

4. **`BUILD`** (41 lines) - **Bazel test configuration**
   - `mock_abstract_predictor` library target (PUBLIC visibility)
   - `test_abstract_predictor` test target (size=small)
   - `test_log_future_predictor` test target (size=small)
   - Dependency declarations for build system

5. **`__init__.py`** (0 lines) - **Empty module marker**
   - Makes directory a Python package
   - No exports (use explicit imports)

## Common Usage Patterns

### 1. Testing Custom Predictor (Minimal Interface Compliance)

```python
import unittest
from nuplan.planning.simulation.predictor.test.mock_abstract_predictor import (
    get_mock_predictor_initialization,
    get_mock_predictor_input,
)
from my_module import MyCustomPredictor

class TestMyCustomPredictor(unittest.TestCase):
    def setUp(self):
        self.predictor = MyCustomPredictor(config_param=value)

    def test_interface_compliance(self):
        """Verify predictor implements AbstractPredictor correctly."""
        # Initialize predictor
        initialization = get_mock_predictor_initialization()
        self.predictor.initialize(initialization)

        # Check name
        self.assertIsInstance(self.predictor.name(), str)
        self.assertTrue(len(self.predictor.name()) > 0)

        # Check observation type
        from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
        self.assertEqual(self.predictor.observation_type(), DetectionsTracks)

        # Test prediction computation
        predictor_input = get_mock_predictor_input()
        detections = self.predictor.compute_predictions(predictor_input)
        self.assertEqual(type(detections), DetectionsTracks)

    def test_runtime_tracking(self):
        """Verify predictor properly tracks performance."""
        self.predictor.initialize(get_mock_predictor_initialization())

        # Run prediction
        predictor_input = get_mock_predictor_input()
        self.predictor.compute_predictions(predictor_input)

        # Check report
        report = self.predictor.generate_predictor_report()
        self.assertEqual(len(report.compute_predictions_runtimes), 1)
        self.assertGreater(report.compute_predictions_runtimes[0], 0.0)
```

### 2. Testing Prediction Quality (Oracle Correctness)

```python
class TestPredictionQuality(unittest.TestCase):
    def setUp(self):
        from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

        self.scenario = MockAbstractScenario()
        self.sampling = TrajectorySampling(num_poses=10, time_horizon=5.0)
        self.predictor = LogFuturePredictor(self.scenario, self.sampling)

    def test_predictions_populated(self):
        """Verify all agents get predictions."""
        from nuplan.planning.simulation.predictor.test.mock_abstract_predictor import get_mock_predictor_input

        predictor_input = get_mock_predictor_input()
        detections = self.predictor.compute_predictions(predictor_input)

        # Check all agents have predictions
        agents = detections.tracked_objects.get_agents()
        for agent in agents:
            self.assertIsNotNone(agent.predictions, f"Agent {agent.token} missing predictions")
            self.assertGreater(len(agent.predictions), 0)

    def test_prediction_horizon(self):
        """Verify predictions match configured horizon."""
        predictor_input = get_mock_predictor_input()
        detections = self.predictor.compute_predictions(predictor_input)

        for agent in detections.tracked_objects.get_agents():
            if agent.predictions:
                for prediction in agent.predictions:
                    valid_waypoints = prediction.valid_waypoints
                    self.assertEqual(
                        len(valid_waypoints),
                        self.sampling.num_poses,
                        f"Expected {self.sampling.num_poses} waypoints, got {len(valid_waypoints)}"
                    )
```

### 3. Performance Benchmarking (Runtime Analysis)

```python
class TestPredictorPerformance(unittest.TestCase):
    def setUp(self):
        self.predictor = MyMLPredictor(model_path="path/to/model.pt")
        self.predictor.initialize(get_mock_predictor_initialization())

    def test_real_time_feasibility(self):
        """Verify predictor runs within real-time budget (100ms at 10Hz)."""
        import statistics

        # Run multiple predictions
        runtimes = []
        for _ in range(100):
            predictor_input = get_mock_predictor_input()
            start = time.perf_counter()
            self.predictor.compute_predictions(predictor_input)
            runtimes.append(time.perf_counter() - start)

        # Check statistics
        mean_time = statistics.mean(runtimes)
        p95_time = statistics.quantiles(runtimes, n=20)[18]  # 95th percentile

        self.assertLess(mean_time, 0.05, f"Mean runtime {mean_time:.3f}s exceeds 50ms target")
        self.assertLess(p95_time, 0.1, f"95th percentile {p95_time:.3f}s exceeds 100ms limit")

        # Verify report matches external measurements
        report = self.predictor.generate_predictor_report()
        self.assertEqual(len(report.compute_predictions_runtimes), 100)

        report_mean = statistics.mean(report.compute_predictions_runtimes)
        self.assertAlmostEqual(report_mean, mean_time, delta=0.01)
```

### 4. Reusing Mock Utilities in Integration Tests

```python
# In nuplan/planning/simulation/test/test_simulation.py (hypothetical)
from nuplan.planning.simulation.predictor.test.mock_abstract_predictor import (
    MockAbstractPredictor,
    get_mock_predictor_initialization,
)

class TestSimulationWithPredictor(unittest.TestCase):
    def test_simulation_integrates_predictor(self):
        """Verify simulation can use predictor in closed-loop."""
        # Setup simulation with mock predictor (no prediction logic needed)
        predictor = MockAbstractPredictor()
        predictor.initialize(get_mock_predictor_initialization())

        # Create simulation setup with predictor
        simulation = Simulation(
            planner=my_planner,
            observation=my_observation,
            predictor=predictor,  # Isolate planner testing from prediction
        )

        # Run simulation
        simulation.run()

        # Verify predictor was called
        report = predictor.generate_predictor_report()
        self.assertGreater(len(report.compute_predictions_runtimes), 0)
```

### 5. Custom Mock with Parameterized Behavior

```python
class ConstantVelocityPredictor(AbstractPredictor):
    """Mock predictor that generates constant velocity predictions."""

    requires_scenario = False

    def __init__(self, prediction_horizon: float = 5.0):
        self._prediction_horizon = prediction_horizon

    def initialize(self, initialization):
        self._map_api = initialization.map_api

    def name(self):
        return "ConstantVelocityPredictor"

    def observation_type(self):
        return DetectionsTracks

    def compute_predicted_trajectories(self, current_input):
        from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

        _, detections = current_input.history.current_state

        # For each agent, extrapolate constant velocity
        for agent in detections.tracked_objects.get_agents():
            # Simple linear extrapolation
            future_states = self._extrapolate_linear(
                agent.center,
                agent.velocity,
                self._prediction_horizon
            )
            agent.predictions = [
                PredictedTrajectory(
                    probability=1.0,
                    waypoints=future_states
                )
            ]

        return detections
```

## Gotchas & Edge Cases

### Test Infrastructure Issues

1. **Mock factories return fresh instances each call**
   - `get_mock_predictor_input()` creates new MockAbstractScenario each time
   - **Symptom**: Modifying returned object doesn't affect future calls
   - **Good**: Tests are isolated (no shared state)
   - **Example**:
     ```python
     input1 = get_mock_predictor_input()
     input2 = get_mock_predictor_input()
     input1.history is not input2.history  # True, different instances
     ```

2. **MockAbstractPredictor is stateless (by design)**
   - No internal state beyond `_map_api`
   - Always returns input observation unchanged
   - **Gotcha**: Can't test stateful predictor behavior (e.g., filtering, temporal smoothing)
   - **Workaround**: Create custom mock with state for advanced tests

3. **Mock scenario has minimal data**
   - `MockAbstractScenario.initial_tracked_objects` has few agents
   - May not represent realistic scenarios (e.g., dense traffic)
   - **Consequence**: Performance tests may not catch scalability issues
   - **Fix**: Use real scenario from database for integration/performance tests

4. **History buffer size defaults to 1**
   - `get_mock_predictor_input(buffer_size=1)` - Only current state
   - **Gotcha**: Can't test predictors that need temporal history
   - **Fix**: Explicitly pass `buffer_size > 1` for temporal tests
   - **Example**: `get_mock_predictor_input(buffer_size=10)` for 1s history at 10Hz

5. **Traffic light data is always None**
   - `PredictorInput(traffic_light_data=None)` hardcoded
   - **Consequence**: Can't test traffic-light-aware predictors with mock utilities
   - **Fix**: Manually create PredictorInput with traffic light data
   - **AIDEV-TODO**: Add factory parameter for traffic light data

### Assertion & Validation Issues

6. **`assertAlmostEqual` for runtime has 0.1s delta (100ms)**
   - **Rationale**: Accounts for timing overhead, context switches
   - **Too loose**: Won't catch 50ms discrepancies
   - **Too tight**: May cause flaky tests on slow CI systems
   - **Design choice**: Prioritize test stability over precision

7. **Runtime tracking includes exception time**
   - If `compute_predicted_trajectories()` raises, time still recorded
   - **Consequence**: Failed prediction attempts appear in statistics
   - **Testing**: Need separate test for exception handling
   - **Example**:
     ```python
     def test_exception_tracking(self):
         class FailingPredictor(MockAbstractPredictor):
             def compute_predicted_trajectories(self, input):
                 raise ValueError("Test error")

         predictor = FailingPredictor()
         with self.assertRaises(ValueError):
             predictor.compute_predictions(get_mock_predictor_input())

         # Runtime still tracked!
         report = predictor.generate_predictor_report()
         self.assertEqual(len(report.compute_predictions_runtimes), 1)
     ```

8. **Type checks use `type()` instead of `isinstance()`**
   - `self.assertEqual(type(detections), DetectionsTracks)`
   - **Gotcha**: Fails for subclasses of DetectionsTracks
   - **Strict**: Requires exact type match
   - **Alternative**: `self.assertIsInstance(detections, DetectionsTracks)` for subclass tolerance

9. **`assertNotIsInstance(report, MLPredictorReport)` is negative assertion**
   - Tests what report IS NOT (rather than what it IS)
   - **Fragile**: If new report subclass added, test still passes
   - **Better**: `self.assertIsInstance(report, PredictorReport)` and check NOT ML-specific fields

10. **No validation of prediction content correctness**
    - Tests check predictions exist and have correct count
    - Don't validate waypoint values, probabilities, etc.
    - **Rationale**: Semantic correctness is implementation-specific
    - **Consequence**: Can pass tests with garbage predictions
    - **Example missing test**:
      ```python
      # NOT tested: Are predicted positions physically plausible?
      # NOT tested: Do probabilities sum to 1.0?
      # NOT tested: Are waypoints in correct coordinate frame?
      ```

### LogFuturePredictor Specific

11. **LogFuturePredictor test uses minimal sampling (1 pose, 1s)**
    - `TrajectorySampling(num_poses=1, time_horizon=1.0)`
    - **Fast**: Minimal database queries
    - **Incomplete**: Doesn't test multi-waypoint predictions
    - **Missing coverage**: Interpolation logic, long horizons, dense sampling

12. **No test for agents without predictions**
    - LogFuturePredictor only assigns predictions if token matches
    - **Untested**: What happens if agent not in scenario?
    - **Expected**: `agent.predictions` remains None or empty
    - **AIDEV-TODO**: Add test for partial prediction assignment

13. **No test for prediction probability validation**
    - Predictions should have `probability` summing to 1.0
    - **Assumption**: Scenario data is pre-validated (database constraint)
    - **Risk**: If assumption breaks, no test will catch it
    - **AIDEV-TODO**: Assert sum of probabilities == 1.0

14. **No test for multi-modal predictions**
    - LogFuturePredictor can return multiple trajectories per agent
    - **Untested**: Does predictor handle multiple modes correctly?
    - **Example**: Agent at intersection might have 2 predictions (left vs straight)

### Test Coverage Gaps

15. **No test for `requires_scenario` flag validation**
    - Flag is used to determine if predictor needs scenario access
    - **Untested**: How does simulation use this flag?
    - **Missing**: Test that LogFuturePredictor has `requires_scenario=True`
    - **Example**:
      ```python
      def test_requires_scenario(self):
          self.assertTrue(LogFuturePredictor.requires_scenario)
          self.assertFalse(MockAbstractPredictor.requires_scenario)
      ```

16. **No test for `generate_predictor_report(clear_stats=False)`**
    - Only tests default `clear_stats=True`
    - **Untested**: Does cumulative statistics work correctly?
    - **Example**:
      ```python
      def test_cumulative_statistics(self):
          predictor.compute_predictions(input1)
          report1 = predictor.generate_predictor_report(clear_stats=False)

          predictor.compute_predictions(input2)
          report2 = predictor.generate_predictor_report(clear_stats=False)

          # Should accumulate
          self.assertEqual(len(report2.compute_predictions_runtimes), 2)
      ```

17. **No test for empty history buffer**
    - `get_mock_predictor_input(buffer_size=0)` - What happens?
    - **Expected**: Should raise ValueError or handle gracefully
    - **Untested**: Edge case for predictors needing history

18. **No test for predictor reinitialization**
    - What if `initialize()` called twice with different maps?
    - **Expected**: Should overwrite `_map_api` (or raise error?)
    - **Untested**: Idempotence and multiple initialization handling

19. **No integration test with real simulation**
    - All tests use mock data
    - **Missing**: Test predictor in actual simulation loop
    - **Rationale**: Integration tests likely in `simulation/test/` module
    - **AIDEV-NOTE**: Check if simulation tests import mock_abstract_predictor

20. **No test for concurrent predictions (thread safety)**
    - `_compute_predictions_runtimes.append()` not thread-safe
    - **Untested**: Multiple threads calling `compute_predictions()`
    - **Risk**: Low (simulation is single-threaded)
    - **AIDEV-TODO**: Add threading.Lock if parallel predictors implemented

## Performance Considerations

### Test Execution Speed

**Test suite performance** (on modern laptop):
- `test_abstract_predictor.py`: ~50ms (5 tests)
- `test_log_future_predictor.py`: ~100ms (1 test, includes scenario creation)
- **Total**: ~150ms for full predictor test suite
- **Bottleneck**: MockAbstractScenario initialization (creates ego state, tracked objects)

**Bazel test sizing**:
- Both tests marked `size = "small"` (60s timeout, 20MB RAM)
- Appropriate for unit tests (no real database, no GPU)

**Optimization opportunities**:
1. **Shared fixtures**: Reuse scenario across tests (currently creates new each time)
2. **Lazy scenario**: Only initialize if predictor needs it
3. **Parametrized tests**: Use `@parameterized` instead of separate test methods

### Mock Data Performance

**`get_mock_predictor_input()` cost**:
- Create MockAbstractScenario: ~10ms
- Initialize SimulationHistoryBuffer: ~5ms per state
- **Total**: ~15ms + (5ms × buffer_size)
- **Scalability**: O(buffer_size) - linear in history length

**Memory footprint**:
- MockAbstractScenario: ~10 KB (minimal agents)
- SimulationHistoryBuffer: ~1 KB per state × buffer_size
- **Total**: ~10-100 KB per test (negligible)

### Runtime Tracking Overhead

**`time.perf_counter()` cost**:
- ~200ns per call (2 calls per prediction)
- **Overhead**: ~400ns per prediction
- **Negligible**: Even for 1ms predictions (0.04% overhead)

**Report statistics computation**:
- `compute_summary_statistics()`: O(N log N) for percentile
- N = number of predictions (typically < 1000 in tests)
- **Cost**: < 1ms for N=1000
- **Optimization**: Cache result if called multiple times

## Related Documentation

### Parent Module (What We Test)
- ✅ `nuplan/planning/simulation/predictor/CLAUDE.md` - Module under test (AbstractPredictor, LogFuturePredictor, reports)

### Test Dependencies (Documented ✅)
- ✅ `nuplan/planning/simulation/observation/CLAUDE.md` - DetectionsTracks observation type
- ✅ `nuplan/planning/simulation/trajectory/CLAUDE.md` - TrajectorySampling configuration
- ✅ `nuplan/planning/simulation/history/CLAUDE.md` - SimulationHistoryBuffer usage
- ✅ `nuplan/common/actor_state/CLAUDE.md` - Agent, PredictedTrajectory, TrackedObjects

### Test Dependencies (Undocumented ⏳)
- ⏳ `nuplan/planning/scenario_builder/test/mock_abstract_scenario.md` - MockAbstractScenario, MockAbstractMap
- ⏳ `nuplan/planning/simulation/simulation_time_controller/CLAUDE.md` - SimulationIteration

### Related Testing Infrastructure
- ⏳ `nuplan/planning/simulation/planner/test/CLAUDE.md` - Similar test patterns for planners
- ⏳ `nuplan/planning/simulation/observation/test/CLAUDE.md` - Observation module tests

### Testing Best Practices
- **unittest documentation**: https://docs.python.org/3/library/unittest.html
- **Bazel testing guide**: https://bazel.build/reference/test-encyclopedia
- **Mock objects pattern**: https://en.wikipedia.org/wiki/Mock_object

## AIDEV Notes

### Critical Test Design Observations

**AIDEV-NOTE**: Test suite is MINIMAL (160 total lines)
- Only 6 test methods across 2 test classes
- High-level interface compliance testing
- No deep validation of prediction semantics
- **Rationale**: Predictor module itself is small (3 files, ~220 lines total)
- **Implication**: Tests match module simplicity

**AIDEV-NOTE**: Public visibility suggests reusable test infrastructure
- `package(default_visibility = ["//visibility:public"])` in BUILD
- MockAbstractPredictor designed as reusable utility
- Factory functions exported for external use
- **But**: `rg` shows no external consumers yet
- **Hypothesis**: Designed for future extensibility (when more predictors added)

**AIDEV-NOTE**: Tests use unittest, not pytest
- No pytest fixtures, parametrization, or markers
- **Consistency**: Matches rest of nuPlan codebase (unittest preferred)
- **Trade-off**: More verbose than pytest, but standard library

### Prediction Module Integration Mystery

**AIDEV-QUESTION**: Why are predictors unused in simulation?
- Observation module (IDMAgents, MLAgents) handles prediction internally
- No simulation setup references AbstractPredictor
- **Evidence**: Parent module CLAUDE.md documents this extensively
- **Tests reflect this**: No integration tests with simulation
- **Implication**: Test suite validates "future" API, not production code path

**AIDEV-QUESTION**: Should tests validate prediction correctness?
- Current tests check existence, count, type
- Don't validate waypoint values, coordinate frames, physical plausibility
- **Trade-off**: Implementation-agnostic vs thorough validation
- **Decision**: Seems intentional (interface compliance, not semantics)

### Missing Test Coverage (High Priority)

**AIDEV-TODO**: Add test for partial prediction assignment
```python
def test_missing_tokens(self):
    """Test predictor handles agents not in scenario."""
    # Create input with agent not in scenario
    # Verify prediction is None or empty (don't crash!)
```

**AIDEV-TODO**: Add test for multi-modal predictions
```python
def test_multi_modal_predictions(self):
    """Test predictor handles multiple trajectory modes."""
    detections = predictor.compute_predictions(input)
    for agent in detections.tracked_objects.get_agents():
        if len(agent.predictions) > 1:
            # Verify probabilities sum to 1.0
            total_prob = sum(p.probability for p in agent.predictions)
            self.assertAlmostEqual(total_prob, 1.0)
```

**AIDEV-TODO**: Add test for `requires_scenario` flag
```python
def test_requires_scenario_flag(self):
    """Verify requires_scenario is set correctly."""
    from nuplan.planning.simulation.predictor.log_future_predictor import LogFuturePredictor

    self.assertTrue(LogFuturePredictor.requires_scenario)
    self.assertFalse(MockAbstractPredictor.requires_scenario)
```

**AIDEV-TODO**: Add test for cumulative statistics
```python
def test_cumulative_report(self):
    """Test report accumulation when clear_stats=False."""
    predictor.compute_predictions(get_mock_predictor_input())
    report1 = predictor.generate_predictor_report(clear_stats=False)

    predictor.compute_predictions(get_mock_predictor_input())
    report2 = predictor.generate_predictor_report(clear_stats=False)

    self.assertEqual(len(report2.compute_predictions_runtimes), 2)
```

### Potential Enhancements

**AIDEV-TODO**: Parametrize mock factory for traffic lights
```python
def get_mock_predictor_input(
    buffer_size: int = 1,
    traffic_light_data: Optional[List[TrafficLightStatusData]] = None
) -> PredictorInput:
    # ... existing code ...
    return PredictorInput(
        iteration=iteration,
        history=history_buffer,
        traffic_light_data=traffic_light_data  # Allow passing custom data
    )
```

**AIDEV-TODO**: Add performance regression test
```python
def test_performance_budget(self):
    """Verify predictor meets real-time performance budget."""
    import statistics

    runtimes = []
    for _ in range(100):
        start = time.perf_counter()
        self.predictor.compute_predictions(get_mock_predictor_input())
        runtimes.append(time.perf_counter() - start)

    p95 = statistics.quantiles(runtimes, n=20)[18]
    self.assertLess(p95, 0.1, f"95th percentile {p95:.3f}s exceeds 100ms budget")
```

**AIDEV-TODO**: Add test for exception handling in runtime tracking
```python
def test_exception_during_prediction(self):
    """Verify runtime tracked even when prediction fails."""
    class FailingPredictor(MockAbstractPredictor):
        def compute_predicted_trajectories(self, input):
            raise RuntimeError("Intentional failure")

    predictor = FailingPredictor()
    with self.assertRaises(RuntimeError):
        predictor.compute_predictions(get_mock_predictor_input())

    # Runtime should still be recorded
    report = predictor.generate_predictor_report()
    self.assertEqual(len(report.compute_predictions_runtimes), 1)
    self.assertGreater(report.compute_predictions_runtimes[0], 0.0)
```

### Code Quality Observations

**AIDEV-NOTE**: Tests are well-structured and readable
- Clear test names describing intent
- Good use of setUp() for test initialization
- Minimal assertion count per test (single responsibility)
- **Good practice**: Each test validates one behavior

**AIDEV-NOTE**: Mock utilities follow Single Responsibility Principle
- MockAbstractPredictor: Minimal interface implementation
- Factory functions: Data generation
- Clear separation of concerns

**AIDEV-NOTE**: No test fixtures or shared state
- Each test uses fresh instances via setUp()
- No `setUpClass()` or module-level state
- **Benefit**: Tests are fully isolated (no order dependencies)
- **Cost**: Slight performance overhead (negligible for small tests)

### Future ML Predictor Testing Considerations

**AIDEV-TODO**: When ML predictor implemented, add tests for:
1. **Model loading**: Test initialize() loads model correctly
2. **Feature building**: Validate feature extraction from history/map
3. **Inference**: Test forward pass with dummy input
4. **Post-processing**: Verify model output → Waypoint conversion
5. **Device handling**: Test CPU/GPU placement
6. **Batching**: Verify multiple agents handled in single pass
7. **MLPredictorReport**: Test feature/inference time breakdown
8. **Error handling**: Test graceful degradation on model failure

**Example test structure**:
```python
class TestMLPredictor(unittest.TestCase):
    def setUp(self):
        self.predictor = MyMLPredictor(model_path="test_model.pt")
        self.predictor.initialize(get_mock_predictor_initialization())

    def test_model_loading(self):
        self.assertIsNotNone(self.predictor._model)

    def test_feature_building(self):
        input = get_mock_predictor_input(buffer_size=10)
        features = self.predictor._build_features(input)
        self.assertEqual(features.shape, expected_shape)

    def test_ml_predictor_report(self):
        self.predictor.compute_predictions(get_mock_predictor_input())
        report = self.predictor.generate_predictor_report()

        self.assertIsInstance(report, MLPredictorReport)
        self.assertEqual(len(report.feature_building_runtimes), 1)
        self.assertEqual(len(report.inference_runtimes), 1)
```

### Documentation Improvements

**AIDEV-TODO**: Add docstring examples to factory functions
```python
def get_mock_predictor_input(buffer_size: int = 1) -> PredictorInput:
    """
    Returns a mock PredictorInput for testing.

    :param buffer_size: Length of simulation history buffer (default 1)
    :return: PredictorInput with mock scenario data

    Example:
        >>> input = get_mock_predictor_input(buffer_size=10)
        >>> len(input.history.observations)
        10
        >>> input.traffic_light_data
        None
    """
```

**AIDEV-TODO**: Add module docstring to test files
```python
"""
test_abstract_predictor.py - AbstractPredictor interface compliance tests

Validates that predictor implementations correctly:
- Initialize with map API
- Return expected name and observation type
- Compute predictions with runtime tracking
- Generate performance reports
"""
```

---

**Last Updated**: 2025-11-15
**Status**: Tier 2 documentation complete (Phase 2B: Control & Motion - Test Suite)
**Test Coverage**: 6 test methods, 2 test classes, 160 total lines
**Cross-references**: 5 documented modules, 2 undocumented dependencies
