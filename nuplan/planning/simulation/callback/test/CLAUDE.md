# nuplan/planning/simulation/callback/test/

## 1. Purpose & Responsibility

This module provides **comprehensive test coverage for the simulation callback system**, validating that callbacks correctly instrument simulation lifecycle events, serialize/log data, compute metrics, track timing, and trigger visualizations. Tests use extensive mocking to isolate callback behavior from simulation infrastructure, verify callback APIs match `AbstractCallback` contracts, and ensure serialized outputs are valid across multiple formats (JSON, pickle, msgpack). Every callback implementation has corresponding tests that validate initialization, step-level hooks, simulation-level hooks, and error handling.

**Test Philosophy**: Mock everything except the callback under test. Validate callbacks integrate correctly with TensorBoard, file I/O, metric engines, and visualization backends without running full simulations.

## 2. Test Structure & Organization

### Test Files by Callback Type

**test_timing_callback.py** (220 lines)
- **Tests**: `TimingCallback` - Simulation performance instrumentation
- **Coverage**: TensorBoard logging, timing capture lifecycle, assertions on uninitialized state
- **Mocks**: `SummaryWriter`, simulation setup, planner, trajectory, history
- **Key test cases**:
  - Constructor state initialization (all None/empty)
  - `on_planner_start()` / `on_planner_end()` - Planner timing capture
  - `on_simulation_start()` / `on_simulation_end()` - Full simulation timing + TensorBoard writes
  - `on_step_start()` / `on_step_end()` - Per-step timing accumulation
  - Assertions for uninitialized state (throws if start time not set)
  - `_get_time()` wraps `time.perf_counter()`
- **Timing metrics validated**:
  - `simulation_elapsed_time` - Total simulation duration
  - `mean_step_time` / `max_step_time` - Step timing statistics
  - `mean_planner_step_time` / `max_planner_step_time` - Planner timing statistics
- **TensorBoard integration**: Validates `add_scalar()` calls with correct metric names, values, global steps

**test_metric_callback.py** (70 lines)
- **Tests**: `MetricCallback` - Metrics computation at simulation end
- **Coverage**: MetricsEngine integration, logging statements, file I/O
- **Mocks**: `MetricsEngine`, simulation setup, planner, history
- **Key test cases**:
  - Constructor validates `_metric_engine` assignment
  - `on_simulation_end()` orchestration:
    1. Calls `metric_engine.compute(history, scenario, planner_name)`
    2. Calls `metric_engine.write_to_files(result)`
    3. Logs debug statements at 4 lifecycle points
- **Logger validation**: Checks debug log calls in sequence:
  - "Starting metrics computation..."
  - "Finished metrics computation!"
  - "Saving metric statistics!"
  - "Saved metrics!"
- **Planner name extraction**: Validates `planner.name()` called and passed to metric engine

**test_simulation_log_callback.py** (234 lines)
- **Tests**: `SimulationLogCallback` - Full simulation history serialization to msgpack
- **Coverage**: Directory creation, serialization, deserialization, equality validation
- **Mocks**: `AbstractSimulationTimeController`, `AbstractObservation`, `AbstractEgoController`
- **Key test cases**:
  - Directory path construction: `output_dir/simulation_log/PlannerName/scenario_type/log_name/scenario_name`
  - `on_initialization_start()` - Directory creation
  - `on_step_end()` - Step data accumulation (mocked in test via manual history building)
  - `on_simulation_end()` - Serialize full history to `.msgpack.xz` file
  - Deserialization round-trip: `SimulationLog.load_data()` preserves history
- **Equality utilities**:
  - `objects_are_equal(a, b)` - Recursive deep equality for complex objects
  - `callable_name_matches(a, b)` - Callable equality via `__name__` or repr parsing
  - `iterator_is_equal(a, b)` - Iterable content equality
- **Mock scenario setup**: Uses `MockAbstractScenario` with manually constructed `SimulationHistory`
- **Validation**: Tests serialized file exists and deserializes to equal `SimulationHistory`

**test_visualization_callback.py** (95 lines)
- **Tests**: `VisualizationCallback` - Rendering ego state, observations, trajectories
- **Coverage**: Visualization backend integration, step-by-step rendering, final frame rendering
- **Mocks**: `AbstractVisualization`, simulation setup, planner, history, history sample
- **Key test cases**:
  - Constructor assigns `_visualization`
  - `on_initialization_start()` calls `visualization.render_scenario(scenario, True)`
  - `on_step_end()` rendering sequence:
    1. `render_ego_state(sample.ego_state)`
    2. `render_observations(sample.observation)`
    3. `render_trajectory(sampled_trajectory)` - After calling `trajectory.get_sampled_trajectory()`
    4. `render(iteration)`
  - `on_simulation_end()` delegates to `on_step_end()` with final history sample
- **Trajectory sampling**: Validates `trajectory.get_sampled_trajectory()` called before rendering
- **Mock structure**: Uses `PropertyMock` for patching `_visualization` attribute

**test_serialization_callback_*.py** (3 files: json, pickle, msgpack)
- **Tests**: `SerializationCallback` - Scene serialization to multiple formats
- **Pattern**: Thin wrappers around `SkeletonTestSerializationCallback` base class
- **Coverage**: Format-specific serialization (each file tests one format)
- **Format configurations**:
  - `test_serialization_callback_json.py` - JSON serialization (`.json`)
  - `test_serialization_callback_pickle.py` - Pickle + lzma (`.pkl.xz`)
  - `test_serialization_callback_msgpack.py` - Msgpack + lzma (`.msgpack.xz`)
- **Test flow**: Each sets `_serialization_type` attribute, calls `_setUp()` and `_dump_test_scenario()`

**skeleton_test_serialization_callback.py** (184 lines)
- **Tests**: `SkeletonTestSerializationCallback` - Base class for serialization tests
- **Coverage**: Full serialization lifecycle, hypothesis property testing for edge cases
- **Mocks**: `AbstractSimulationTimeController`, `AbstractObservation`, `AbstractEgoController`
- **Key test cases**:
  - Constructor with `serialization_type`, `output_directory`, `folder_name`, `serialize_into_single_file`
  - Directory path validation (matches expected structure)
  - `on_initialization_start()` - Directory creation
  - `on_step_end()` - Step data capture (in single file mode)
  - `on_simulation_end()` - Trigger serialization to file
  - Deserialization and validation:
    1. Load file based on serialization type
    2. Validate scene structure keys: `"world"`, `"ego"`, `"trajectories"`, `"map"`
    3. Validate traffic light data serialization (timestamp integrity)
- **Hypothesis testing**: Property-based testing for timestamp overflow
  - Tests with `timestamp=0` (lower bound)
  - Tests with `timestamp=1627066061949808` (realistic large value)
  - Tests with `timestamp=18446744073709551615` (uint64 max - msgpack limit)
  - Validates large integers serialize/deserialize without corruption
- **Mock traffic light data**: Uses `mock_get_traffic_light_status_at_iteration()` to inject test timestamps
- **Extension mapping**: `.json`, `.pkl.xz`, `.msgpack.xz` based on serialization type
- **Validation strategy**: Deserialize and check structure (not full equality like simulation_log test)

**test_simulation_log.py** (53 lines)
- **Tests**: `SimulationLog` - File type detection utility
- **Coverage**: `simulation_log_type()` static method
- **Key test cases**:
  - **Msgpack detection**: Paths ending in `.msgpack.xz` → "msgpack"
  - **Pickle detection**: Paths ending in `.pkl.xz` → "pickle"
  - **Ambiguous cases**: `.msgpack.pkl.xz` treated as pickle (last extension wins)
  - **Failing cases**: Invalid extensions raise `ValueError`
    - Missing `.xz` compression (`.pkl`, `.msgpack`)
    - Unsupported formats (`.json.xz`)
    - No recognized extension
- **Real-world path handling**: Tests with long directory paths (production log paths)
- **Edge case validation**: Double extensions, nested directories, numeric path components

## 3. Key Testing Patterns & Strategies

### Pattern 1: Lifecycle Hook Validation
All callback tests validate the full lifecycle hook sequence:
1. **Initialization**: `on_initialization_start()` → `on_initialization_end()`
2. **Simulation start**: `on_simulation_start()`
3. **Step loop**: `on_step_start()` → `on_planner_start()` → `on_planner_end()` → `on_step_end()`
4. **Simulation end**: `on_simulation_end()`

**Example** (from test_timing_callback.py):
```python
# Validate start time is captured
self.tc.on_planner_start(self.setup, self.planner)
assert self.tc._planner_start == START_TIME

# Validate duration is computed and appended
self.tc.on_planner_end(self.setup, self.planner, self.trajectory)
assert self.tc._planner_step_duration[-1] == END_TIME - START_TIME
```

### Pattern 2: Mock Isolation Strategy
Tests mock ALL dependencies except the callback under test:
- **Simulation infrastructure**: `SimulationSetup`, `SimulationHistory`, `SimulationHistorySample`
- **Planner**: `AbstractPlanner` (with `name()` method)
- **External services**: `SummaryWriter` (TensorBoard), `MetricsEngine`, `AbstractVisualization`
- **I/O**: File system via `tempfile.TemporaryDirectory()`

**Gotcha**: Must mock `spec=` correctly to avoid AttributeError on method calls
```python
self.mock_planner = Mock(spec=AbstractPlanner)
self.mock_planner.name = Mock(return_value=PLANNER_NAME)  # Explicit method mock
```

### Pattern 3: Assertion-Based Error Handling Tests
Callbacks assert preconditions (e.g., start time set before end time):
```python
def test_on_planner_end_throws_if_no_start_time_set(self) -> None:
    """Tests if on_planner_end throws an exception if the planner_start time is not set."""
    with self.assertRaises(AssertionError):
        self.tc.on_planner_end(self.setup, self.planner, self.trajectory)
```

**Coverage**: Every callback with timing/state dependencies has corresponding assertion tests

### Pattern 4: Serialization Round-Trip Validation
Serialization tests validate data integrity via round-trip:
1. Serialize data to file (JSON/pickle/msgpack)
2. Deserialize from file
3. Validate structure/equality

**Two approaches**:
- **Full equality** (test_simulation_log_callback.py): Deep recursive equality check
- **Structure validation** (skeleton_test_serialization_callback.py): Check keys exist

**Why different approaches?**
- `SimulationLog`: High-level API, deserializes to exact `SimulationHistory` (full equality expected)
- `SerializationCallback`: Low-level scene serialization, validates schema (structure sufficient)

### Pattern 5: Property-Based Testing with Hypothesis
`skeleton_test_serialization_callback.py` uses Hypothesis for edge case generation:
```python
@settings(deadline=None)  # Disable timeout for complex serialization
@given(
    mock_timestamp=st.one_of(
        st.just(0),  # Lower bound
        st.integers(min_value=1627066061949808, max_value=18446744073709551615)  # Large values
    )
)
def _dump_test_scenario(self, mock_timestamp: int) -> None:
    # Test serialization with various timestamp values
```

**Why Hypothesis?**
- Msgpack has uint64 limit (`18446744073709551615`)
- Timestamps can overflow if not serialized correctly
- Property-based testing catches edge cases (e.g., year 2584 timestamps)

### Pattern 6: Custom Equality for Complex Objects
`test_simulation_log_callback.py` implements deep equality checking:
```python
def objects_are_equal(a: object, b: object) -> bool:
    """Recursively checks if two objects are equal by value."""
    # Handles: callables, numpy arrays, iterables, nested objects with __dict__
```

**Supported types**:
- Built-in types (int, float, str, bool)
- Callable objects (via `callable_name_matches()`)
- Numpy arrays (via `np.allclose()`)
- Nested objects (via `__dict__` recursion)
- Iterables (via `iterator_is_equal()`)

**Gotcha**: Type comparison uses `type() == type()` (strict, not isinstance)

## 4. Dependencies & Mocks

### External Dependencies
- **unittest** - Test framework (TestCase, Mock, MagicMock, patch, call)
- **tempfile** - Temporary directories for file I/O tests
- **pathlib** - Path manipulation and validation
- **lzma** - Compression for pickle/msgpack
- **msgpack** - Msgpack serialization
- **ujson** - Fast JSON serialization
- **pickle** - Python object serialization
- **numpy** - Array equality (via `np.allclose()`)
- **hypothesis** - Property-based testing

### Internal nuPlan Imports
**Callback implementations (under test)**:
- `nuplan.planning.simulation.callback.timing_callback.TimingCallback`
- `nuplan.planning.simulation.callback.metric_callback.MetricCallback`
- `nuplan.planning.simulation.callback.simulation_log_callback.SimulationLogCallback`
- `nuplan.planning.simulation.callback.visualization_callback.VisualizationCallback`
- `nuplan.planning.simulation.callback.serialization_callback.SerializationCallback`

**Supporting infrastructure**:
- `nuplan.planning.simulation.callback.abstract_callback.AbstractCallback` - Interface definition
- `nuplan.planning.simulation.simulation_setup.SimulationSetup` - Simulation configuration
- `nuplan.planning.simulation.simulation_log.SimulationLog` - Log file I/O
- `nuplan.planning.simulation.history.simulation_history` - SimulationHistory, SimulationHistorySample
- `nuplan.planning.simulation.planner.abstract_planner.AbstractPlanner` - Planner interface
- `nuplan.planning.simulation.planner.simple_planner.SimplePlanner` - Concrete planner (test_simulation_log_callback)
- `nuplan.planning.simulation.trajectory.abstract_trajectory.AbstractTrajectory` - Trajectory interface
- `nuplan.planning.simulation.trajectory.interpolated_trajectory.InterpolatedTrajectory` - Concrete trajectory

**Test utilities**:
- `nuplan.planning.scenario_builder.test.mock_abstract_scenario.MockAbstractScenario` - Test scenario
- `nuplan.planning.metrics.metric_engine.MetricsEngine` - Metrics computation
- `nuplan.planning.simulation.observation.observation_type.DetectionsTracks` - Observation wrapper
- `nuplan.planning.simulation.visualization.abstract_visualization.AbstractVisualization` - Viz backend

**Mocked (never imported directly in tests)**:
- `torch.utils.tensorboard.SummaryWriter` - TensorBoard integration
- `nuplan.planning.simulation.controller.abstract_controller.AbstractEgoController` - Controller
- `nuplan.planning.simulation.observation.abstract_observation.AbstractObservation` - Observation
- `nuplan.planning.simulation.simulation_time_controller.abstract_simulation_time_controller.AbstractSimulationTimeController` - Time controller

## 5. Testing Best Practices Demonstrated

### 1. Separation of Concerns
Each test file tests exactly one callback implementation, with clear setUp/tearDown:
```python
def setUp(self) -> None:
    self.writer = Mock(spec=SummaryWriter)
    self.tc = TimingCallback(self.writer)
    return super().setUp()
```

### 2. Descriptive Test Names
Test names document exact behavior being validated:
- `test_on_planner_end_throws_if_no_start_time_set` - Clear failure condition
- `test_on_simulation_end` - Covers happy path
- `test_serialization_callback` - Format-specific serialization

### 3. Test Documentation
Every test has docstrings explaining:
- What is being tested
- Expected behavior
- Validation strategy

Example:
```python
def test_on_simulation_end(self, get_time: MagicMock) -> None:
    """
    Tests if the get_time method is called and the elapsed time is set accordingly.
    Tests if the timings are calculated properly and writer is called with the correct values.
    Tests if the timings are stored in the scenarios_captured under the right token.
    Tests if the step_duration and planner_step_duration are cleared.
    """
```

### 4. Arrange-Act-Assert Pattern
Tests follow clear structure:
```python
# Arrange (setup in setUp() or test-specific)
get_time.return_value = END_TIME
self.tc._simulation_start = START_TIME

# Act
self.tc.on_simulation_end(self.setup, self.planner, self.history)

# Assert
get_time.assert_called_once()
self.writer.add_scalar.assert_has_calls([...])
```

### 5. Mock Verification
Tests validate mocks were called with correct arguments:
```python
self.writer.add_scalar.assert_has_calls([
    call("simulation_elapsed_time", END_TIME - START_TIME, 7),
    call('mean_step_time', 452, 7),
    # ... more calls
])
```

**Gotcha**: `assert_has_calls()` validates call sequence, not just call presence

### 6. Temporary File Cleanup
File I/O tests use `tempfile.TemporaryDirectory()` with proper cleanup:
```python
def setUp(self) -> None:
    self.output_folder = tempfile.TemporaryDirectory()
    # ... use self.output_folder.name

def tearDown(self) -> None:
    self.output_folder.cleanup()
```

### 7. Edge Case Coverage via Hypothesis
Property-based testing covers edge cases developers might miss:
- Timestamp overflow (uint64 limits)
- Large integer serialization
- Boundary conditions (0, max values)

## 6. Common Test Utilities & Helpers

### Equality Checking Utilities (test_simulation_log_callback.py)

**`objects_are_equal(a: object, b: object) -> bool`**
- Recursive deep equality for complex objects
- Handles nested dataclasses, numpy arrays, iterables, callables
- Used to validate `SimulationHistory` round-trip equality

**`callable_name_matches(a: Callable, b: Callable) -> bool`**
- Compares callables by `__name__` attribute or repr parsing
- Handles scipy interpolation objects and other callables without `__name__`

**`iterator_is_equal(a: Iterable, b: Iterable) -> bool`**
- Element-wise iterable comparison
- Validates both content and length match

### Mock Scenario Construction

**Pattern**: Build minimal `SimulationHistory` with mock states
```python
history = SimulationHistory(scenario.map_api, scenario.get_mission_goal())
state_0 = EgoState.build_from_rear_axle(
    StateSE2(0, 0, 0),
    vehicle_parameters=scenario.ego_vehicle_parameters,
    rear_axle_velocity_2d=StateVector2D(x=0, y=0),
    rear_axle_acceleration_2d=StateVector2D(x=0, y=0),
    tire_steering_angle=0,
    time_point=TimePoint(0),
)
history.add_sample(SimulationHistorySample(...))
```

**Used in**: test_simulation_log_callback.py, skeleton_test_serialization_callback.py

### Directory Path Validation

**Pattern**: Validate callback constructs correct output paths
```python
directory = self.callback._get_scenario_folder(planner.name(), scenario)
self.assertEqual(
    str(directory),
    f"{self.output_folder.name}/sim/DummyPlanner/mock_scenario_type/mock_log_name/mock_scenario_name"
)
```

**Structure**: `output_dir/folder_name/planner_name/scenario_type/log_name/scenario_name`

## 7. Key Abstractions Under Test

### AbstractCallback Lifecycle Hooks
All callbacks must implement 8 lifecycle hooks:
1. `on_initialization_start(setup, planner)` - Before scenario initialization
2. `on_initialization_end(setup, planner)` - After scenario initialization
3. `on_simulation_start(setup)` - Before simulation loop
4. `on_step_start(setup, planner)` - Before each simulation step
5. `on_planner_start(setup, planner)` - Before planner computation
6. `on_planner_end(setup, planner, trajectory)` - After planner computation
7. `on_step_end(setup, planner, sample)` - After each simulation step
8. `on_simulation_end(setup, planner, history)` - After simulation loop

**Test coverage**: Every callback implementation has tests for relevant hooks

### Callback State Management
Callbacks maintain internal state between hooks:
- **TimingCallback**: `_step_start`, `_simulation_start`, `_planner_start`, `_step_duration`, `_planner_step_duration`
- **SerializationCallback**: Accumulated scene data (when `serialize_into_single_file=True`)
- **VisualizationCallback**: Stateless (delegates to visualization backend)

**Tests validate**: State is correctly initialized, updated, and cleared across simulation lifecycle

### Serialization Formats
Three supported serialization formats:
1. **JSON** (`.json`) - Human-readable, no compression, slower
2. **Pickle** (`.pkl.xz`) - Python-specific, compressed, fast
3. **Msgpack** (`.msgpack.xz`) - Language-agnostic, compressed, compact

**Tests validate**: Each format produces valid output, deserializes correctly, handles edge cases (large integers)

## 8. Running the Tests

### Run All Callback Tests
```bash
# From repository root
uv run pytest nuplan/planning/simulation/callback/test/

# With coverage
uv run pytest --cov=nuplan.planning.simulation.callback nuplan/planning/simulation/callback/test/

# Verbose output
uv run pytest -v nuplan/planning/simulation/callback/test/
```

### Run Specific Test File
```bash
# Timing callback tests
uv run pytest nuplan/planning/simulation/callback/test/test_timing_callback.py

# Serialization tests (all formats)
uv run pytest nuplan/planning/simulation/callback/test/test_serialization_callback_*.py

# Hypothesis tests (property-based)
uv run pytest nuplan/planning/simulation/callback/test/skeleton_test_serialization_callback.py
```

### Run Specific Test Case
```bash
# Single test method
uv run pytest nuplan/planning/simulation/callback/test/test_timing_callback.py::TestTimingCallback::test_on_simulation_end

# All tests matching pattern
uv run pytest -k "serialization" nuplan/planning/simulation/callback/test/
```

### Debug Test Failures
```bash
# Show print statements and full diffs
uv run pytest -vv -s nuplan/planning/simulation/callback/test/test_timing_callback.py

# Stop on first failure
uv run pytest -x nuplan/planning/simulation/callback/test/

# Enter debugger on failure
uv run pytest --pdb nuplan/planning/simulation/callback/test/
```

## 9. Common Test Failures & Debugging

### Failure: Mock Attribute Not Found
**Symptom**: `AttributeError: Mock object has no attribute 'name'`
**Cause**: Mock created without `spec=` or method not explicitly mocked
**Fix**:
```python
# Wrong
self.mock_planner = Mock()

# Correct
self.mock_planner = Mock(spec=AbstractPlanner)
self.mock_planner.name = Mock(return_value=PLANNER_NAME)
```

### Failure: Assertion Order Mismatch
**Symptom**: `AssertionError: Calls not found` in `assert_has_calls()`
**Cause**: Mock calls happened in different order than expected
**Fix**: Check actual call order with `mock.call_args_list` or use `assert_any_call()` for unordered

### Failure: Temporary Directory Not Cleaned
**Symptom**: `/tmp` fills up after repeated test runs
**Cause**: `tearDown()` not called (test crashed) or not implemented
**Fix**: Always implement `tearDown()` with `self.output_folder.cleanup()`

### Failure: Hypothesis Deadline Exceeded
**Symptom**: `hypothesis.errors.DeadlineExceeded: Test took too long`
**Cause**: Serialization tests take variable time (compression, I/O)
**Fix**: Add `@settings(deadline=None)` to disable timeout

### Failure: File Already Exists
**Symptom**: `FileExistsError: [Errno 17] File exists`
**Cause**: Test reuses output directory without cleaning
**Fix**: Use `tempfile.TemporaryDirectory()` or `mkdir(exist_ok=True, parents=True)`

### Failure: Path Type Mismatch
**Symptom**: `TypeError: expected str, bytes or os.PathLike object, not Mock`
**Cause**: Mock object used where Path/str expected
**Fix**: Mock scenario attributes correctly:
```python
scenario = Mock()
scenario.scenario_type = "mock_scenario_type"  # str, not Mock
scenario.log_name = "mock_log_name"
scenario.scenario_name = "mock_scenario_name"
```

## 10. Test Coverage Gaps & Future Work

### Current Coverage Strengths
- ✅ All callback lifecycle hooks tested
- ✅ Error handling (assertions on uninitialized state)
- ✅ TensorBoard integration (TimingCallback)
- ✅ Metrics engine integration (MetricCallback)
- ✅ All serialization formats (JSON, pickle, msgpack)
- ✅ File I/O and directory creation
- ✅ Round-trip serialization validation
- ✅ Property-based testing for edge cases

### Coverage Gaps (Not Tested)
- ⚠️ **MultiCallback** - No tests for callback composition (wrapper that calls multiple callbacks)
- ⚠️ **Concurrent callbacks** - No tests for thread-safety or parallel simulation
- ⚠️ **Callback exceptions** - No tests for callback failures during simulation (should not crash simulation)
- ⚠️ **Large histories** - No tests for memory usage with 1000+ timestep simulations
- ⚠️ **Disk space exhaustion** - No tests for I/O failures (disk full, permission denied)
- ⚠️ **Malformed data** - No tests for corrupted serialized files (deserialization error handling)
- ⚠️ **Visualization backend failures** - VisualizationCallback doesn't test backend exceptions
- ⚠️ **MetricsEngine failures** - MetricCallback doesn't test metric computation exceptions

### Future Test Additions (Recommended)
1. **Add MultiCallback tests** - Validate callback composition, ordering, exception isolation
2. **Add integration tests** - Test callbacks in actual simulation loop (not just mocks)
3. **Add stress tests** - Large histories (10k+ samples), disk space limits, concurrent access
4. **Add deserialization error tests** - Validate graceful handling of corrupted files
5. **Add callback exception tests** - Validate simulation continues if callback raises exception

## 11. Critical Gotchas & Warnings

### Gotcha 1: Mock Spec Must Match Interface
**Problem**: Mocking without `spec=` allows calling non-existent methods (fails late)
**Impact**: Tests pass but don't reflect real behavior
**Solution**: Always use `Mock(spec=InterfaceClass)` for interface mocks
```python
# Bad - allows any method call
self.planner = Mock()

# Good - raises AttributeError for invalid methods
self.planner = Mock(spec=AbstractPlanner)
```

### Gotcha 2: Hypothesis Requires Deadline=None for I/O Tests
**Problem**: Hypothesis default timeout (200ms) too short for file I/O + compression
**Impact**: Tests fail intermittently on slow I/O
**Solution**: Add `@settings(deadline=None)` to serialization tests
```python
@settings(deadline=None)  # Disable timeout
@given(mock_timestamp=st.integers(...))
def test_serialization(self, mock_timestamp: int):
    # ... file I/O tests
```

### Gotcha 3: TempDirectory Cleanup Requires Explicit tearDown
**Problem**: Python doesn't guarantee `__del__` is called (circular references)
**Impact**: `/tmp` fills with test artifacts
**Solution**: Always implement `tearDown()` with explicit cleanup
```python
def tearDown(self) -> None:
    self.output_folder.cleanup()  # Explicit cleanup
```

### Gotcha 4: Mock Call Order Matters with assert_has_calls
**Problem**: `assert_has_calls()` validates call sequence, not just presence
**Impact**: Tests fail if calls happen in different (but valid) order
**Solution**: Use `assert_any_call()` for unordered checks or adjust expected order
```python
# Fails if call order differs
mock.assert_has_calls([call(1), call(2), call(3)])

# Succeeds if call exists (order-independent)
mock.assert_any_call(2)
```

### Gotcha 5: PropertyMock for Attribute Patching
**Problem**: Regular `Mock()` doesn't work for patching instance attributes
**Impact**: `patch.object(obj, 'attr')` doesn't intercept attribute access
**Solution**: Use `PropertyMock` for attribute-level patching
```python
# Wrong - doesn't intercept attribute access
@patch.object(VisualizationCallback, '_visualization')
def test(self, viz_mock):
    # viz_mock never called

# Correct - intercepts attribute access
@patch.object(VisualizationCallback, '_visualization', create=True, new_callable=PropertyMock)
def test(self, viz_mock):
    # viz_mock.return_value called when accessing self._visualization
```

### Gotcha 6: Msgpack Uint64 Limit for Timestamps
**Problem**: Msgpack can't serialize integers > `2^64 - 1` (18446744073709551615)
**Impact**: Future timestamps (year 2584+) cause serialization failures
**Solution**: Tests validate timestamp range, production code should use uint64 timestamps
```python
# Hypothesis tests this edge case
@given(mock_timestamp=st.integers(max_value=18446744073709551615))
def test_large_timestamps(self, mock_timestamp):
    # Validates msgpack handles max uint64
```

### Gotcha 7: Callable Equality is Fragile
**Problem**: Callables don't have reliable equality (`lambda != lambda` even if identical)
**Impact**: Deep equality checks fail for scipy interpolators, closures
**Solution**: Use `callable_name_matches()` to compare by name/repr instead of identity
```python
# Fails - callables have different identities
assert scipy.interpolate.interp1d(...) == scipy.interpolate.interp1d(...)

# Works - compares qualified name from repr
assert callable_name_matches(func1, func2)
```

### Gotcha 8: SimulationHistory Equality Needs Custom Logic
**Problem**: `SimulationHistory` contains numpy arrays, callables, nested objects
**Impact**: Standard `==` fails (numpy array truth value ambiguous)
**Solution**: Use `objects_are_equal()` recursive equality checker
```python
# Fails - numpy array equality is element-wise (not bool)
assert history1 == history2

# Works - deep recursive equality
assert objects_are_equal(history1, history2)
```

### Gotcha 9: Test Scenario Mocks Must Set String Attributes
**Problem**: Mock objects default to returning Mocks for attribute access
**Impact**: Path construction fails when concatenating Mock objects
**Solution**: Explicitly set string attributes on scenario mocks
```python
# Wrong - scenario.scenario_type is a Mock
scenario = Mock(spec=AbstractScenario)

# Correct - scenario.scenario_type is a str
scenario = Mock(spec=AbstractScenario)
scenario.scenario_type = "mock_scenario_type"
scenario.log_name = "mock_log_name"
scenario.scenario_name = "mock_scenario_name"
```

### Gotcha 10: Serialization Tests Must Match Extension to Format
**Problem**: Each serialization format has specific file extension (`.json`, `.pkl.xz`, `.msgpack.xz`)
**Impact**: Tests fail if checking wrong extension for format
**Solution**: Use `_serialization_type_to_extension_map` in skeleton test
```python
self._serialization_type_to_extension_map = {
    "json": ".json",
    "pickle": ".pkl.xz",
    "msgpack": ".msgpack.xz",
}
filename = "scenario" + self._serialization_type_to_extension_map[self._serialization_type]
```

### Gotcha 11: Timing Tests Must Mock time.perf_counter
**Problem**: Real time measurements are non-deterministic and affect test reproducibility
**Impact**: Tests fail intermittently due to timing variations
**Solution**: Patch `time.perf_counter` with fixed return values
```python
@patch('nuplan.planning.simulation.callback.timing_callback.time.perf_counter')
def test_timing(self, perf_counter: MagicMock):
    perf_counter.return_value = 12345.0  # Fixed timestamp
    # ... test timing logic
```

### Gotcha 12: Logger Tests Require Patching Logger Instance
**Problem**: Logger calls must be validated with correct patch path
**Impact**: Logger assertions fail if patching wrong module
**Solution**: Patch logger at callback module level, not logging module
```python
# Wrong - patches root logging module
@patch('logging.logger')

# Correct - patches callback's logger instance
@patch('nuplan.planning.simulation.callback.metric_callback.logger')
def test_logging(self, logger: MagicMock):
    logger.debug.assert_called_with("Expected message")
```

## 12. Integration with Callback Implementations

### Callback Test Coverage Map

| Callback | Test File | Lines | Lifecycle Hooks | I/O | External Services |
|----------|-----------|-------|-----------------|-----|-------------------|
| `TimingCallback` | test_timing_callback.py | 220 | ✅ All hooks | ❌ | ✅ TensorBoard |
| `MetricCallback` | test_metric_callback.py | 70 | ✅ Simulation end only | ❌ | ✅ MetricsEngine |
| `SimulationLogCallback` | test_simulation_log_callback.py | 234 | ✅ Init + end | ✅ Msgpack | ❌ |
| `VisualizationCallback` | test_visualization_callback.py | 95 | ✅ Init + step + end | ❌ | ✅ Visualization |
| `SerializationCallback` | skeleton_test_serialization_callback.py | 184 | ✅ Init + end | ✅ JSON/Pickle/Msgpack | ❌ |
| `SimulationLog` (utility) | test_simulation_log.py | 53 | N/A | ✅ Type detection | ❌ |

### Cross-References
- **Callback implementations**: `nuplan/planning/simulation/callback/` (parent directory)
- **AbstractCallback interface**: `nuplan/planning/simulation/callback/abstract_callback.py`
- **Simulation loop**: `nuplan/planning/simulation/simulation.py` (calls callback hooks)
- **Test utilities**: `nuplan/planning/scenario_builder/test/` (MockAbstractScenario)
- **Metrics system**: `nuplan/planning/metrics/` (MetricsEngine)
- **Visualization**: `nuplan/planning/simulation/visualization/` (AbstractVisualization)

## 13. Performance Characteristics

### Test Execution Time (Approximate)
- **test_timing_callback.py**: ~0.5s (fast, all mocked)
- **test_metric_callback.py**: ~0.2s (fast, all mocked)
- **test_visualization_callback.py**: ~0.3s (fast, all mocked)
- **test_simulation_log_callback.py**: ~2-5s (slow, real file I/O + compression)
- **skeleton_test_serialization_callback.py**: ~5-15s (slow, hypothesis + I/O + 3 formats)
- **test_simulation_log.py**: ~0.1s (fast, no I/O)

**Total suite**: ~10-25 seconds (mostly I/O and compression)

### Optimization Opportunities
1. **Reduce hypothesis examples** - Default 100 examples, could reduce to 10 for faster CI
2. **Mock lzma compression** - Could mock compression to test logic without I/O overhead
3. **Parallel test execution** - Tests are independent, could run with `pytest-xdist`
4. **Skip I/O tests in quick mode** - Add markers for slow tests (`@pytest.mark.slow`)

### Memory Usage
- **Peak memory**: ~500 MB (hypothesis generates many test cases, file I/O buffers)
- **Per-test memory**: ~50 MB (SimulationHistory with multiple samples)
- **Temporary files**: ~10-50 MB per serialization test (cleaned up in tearDown)

## 14. Summary & Key Takeaways

### What This Test Suite Validates
✅ **Callback lifecycle compliance** - All hooks called with correct arguments
✅ **State management** - Internal state initialized, updated, cleared correctly
✅ **External service integration** - TensorBoard, MetricsEngine, Visualization mocked correctly
✅ **File I/O correctness** - Serialization/deserialization round-trips preserve data
✅ **Error handling** - Assertions catch uninitialized state, invalid inputs
✅ **Format compatibility** - JSON, pickle, msgpack all tested
✅ **Edge cases** - Hypothesis tests large timestamps, boundary conditions

### What Developers Should Know
1. **All callbacks must implement AbstractCallback** - 8 lifecycle hooks required
2. **Tests use extensive mocking** - Isolate callback logic from simulation infrastructure
3. **Serialization is format-specific** - JSON/pickle/msgpack have different characteristics
4. **Timing callbacks use assertions** - Start time must be set before end time (defensive programming)
5. **File I/O tests use temp directories** - Always cleanup in tearDown to avoid /tmp bloat
6. **Hypothesis enables property-based testing** - Catches edge cases developers miss
7. **Deep equality is non-trivial** - Custom utilities needed for complex objects (numpy, callables)

### When to Update Tests
- **New callback implementation** → Add new test file (follow existing patterns)
- **New lifecycle hook** → Update all callback tests (add hook coverage)
- **New serialization format** → Add test_serialization_callback_*.py file
- **New external service** → Add mock validation (TensorBoard, MetricsEngine pattern)
- **Callback state changes** → Update state validation tests (constructor, lifecycle)

### Related Documentation
- **Phase 2C**: Callback system architecture (`nuplan/planning/simulation/callback/CLAUDE.md`)
- **Phase 2B**: Controller, predictor testing (adjacent test suites)
- **Phase 1**: Core abstractions (actor_state, geometry, maps tested separately)
