# nuplan/planning/simulation/controller/test/

## 1. Purpose & Responsibility

**THE comprehensive test suite for ego vehicle controllers and motion models.** This module validates all controller implementations (PerfectTracking, LogPlayback, TwoStage) and their underlying components (motion models, trackers, utilities). Tests ensure controllers correctly propagate ego state, track trajectories, apply control delays, enforce physical constraints, and integrate with the simulation loop. Coverage includes unit tests, integration tests, and edge case validation.

## 2. Key Abstractions

### Test Classes

**TestLogPlaybackController** (`test_log_playback.py`)
- **Purpose**: Validates LogPlaybackController oracle replay behavior
- **Test coverage**: Constructor initialization, state retrieval, iteration index updates
- **Mocking strategy**: Heavy use of `unittest.mock` - all dependencies mocked (scenario, iterations, ego state)
- **Key pattern**: PropertyMock for testing property assignment in constructor

**TestPerfectTracking** (`test_perfect_tracking.py`)
- **Purpose**: Validates PerfectTrackingController trajectory tracking
- **Test coverage**: Basic tracking accuracy over multiple timesteps
- **Mocking strategy**: Uses `MockAbstractScenario` for realistic scenario data
- **Key pattern**: Integration test - validates state propagation matches expert trajectory

**TestUtils** (`test_utils.py`)
- **Purpose**: Validates forward Euler integration utility
- **Test coverage**: Integration accuracy across random parameter space
- **Mocking strategy**: No mocks - pure numerical validation
- **Key pattern**: Randomized testing (100 test cases with random inputs)

**TestKinematicMotionModel** (`motion_model/test/test_kinematic_motion_model.py`)
- **Purpose**: Comprehensive validation of kinematic bicycle physics
- **Test coverage**: State derivatives, state propagation, steering limits, control delays
- **Mocking strategy**: Minimal - uses real EgoState construction via test utilities
- **Key pattern**: Physics validation - checks equations of motion, constraint enforcement

### Test Fixtures & Utilities

**Common setUp() Pattern**
```python
def setUp(self) -> None:
    """Standard initialization - called before each test method"""
    self.vehicle = get_pacifica_parameters()
    self.ego_state = get_sample_ego_state()  # or Mock/MagicMock
    self.scenario = MockAbstractScenario(...)  # or MagicMock(spec=...)
```

**Mock Hierarchy** (test_log_playback.py)
- `MagicMock(spec=AbstractScenario)` - Scenario with mocked methods
- `Mock(spec=SimulationIteration)` - Iteration with PropertyMock for index
- `Mock(spec=EgoState)` - Ego state (unused, just for signature)
- `Mock(spec=AbstractTrajectory)` - Trajectory (unused, LogPlayback ignores it)

**Real Data Fixtures** (test_perfect_tracking.py, test_kinematic_motion_model.py)
- `MockAbstractScenario` - Generates realistic expert trajectory
- `get_sample_ego_state()` - Creates valid EgoState with Pacifica parameters
- `TimePoint(microseconds)` - Explicit time representations

### Critical Test Patterns

1. **Property Mocking for Constructor Tests**
```python
@patch.object(LogPlaybackController, 'current_iteration', create=True, new_callable=PropertyMock)
@patch.object(LogPlaybackController, 'scenario', create=True, new_callable=PropertyMock)
def test_constructor(self, scenario_mock, iteration_mock):
    controller = LogPlaybackController(self.scenario)
    # Verify property setters called with correct values
    scenario_mock.assert_called_once_with(self.scenario)
    iteration_mock.assert_called_once_with(0)
```

2. **Context Manager for Temporary Mocking**
```python
with patch.object(LogPlaybackController, 'current_iteration', new_callable=PropertyMock) as mock:
    mock.return_value = CURRENT_ITERATION
    result = controller.get_state()
    # Assertions...
```

3. **Integration Test with Expert Trajectory**
```python
scenario = MockAbstractScenario(initial_time_us=initial_time_point)
trajectory = InterpolatedTrajectory(list(scenario.get_expert_ego_trajectory()))
controller = PerfectTrackingController(scenario)

# Propagate and compare to ground truth
controller.update_state(iteration, next_iteration, ego_state, trajectory)
actual_state = controller.get_state()
expected_state = scenario.get_ego_state_at_iteration(2)

self.assertAlmostEqual(actual_state.rear_axle.x, expected_state.rear_axle.x)
```

4. **Randomized Numerical Validation**
```python
def setUp(self):
    np.random.seed(0)  # Reproducibility
    self.test_params = np.random.rand(100)  # 100 random test cases

def test_integration(self):
    for param in self.test_params:
        result = function_under_test(param)
        expected = analytical_solution(param)
        self.assertAlmostEqual(result, expected)
```

5. **Physics Equation Validation**
```python
# Compute expected values from kinematic equations
wheel_base = vehicle.wheel_base
velocity = ego_state.dynamic_car_state.rear_axle_velocity_2d.x
x_dot = velocity * np.cos(heading)
y_dot = velocity * np.sin(heading)
yaw_dot = velocity * np.tan(steering_angle) / wheel_base

# Test motion model matches equations
state_dot = motion_model.get_state_dot(ego_state)
self.assertEqual(state_dot.rear_axle, StateSE2(x_dot, y_dot, yaw_dot))
```

6. **Constraint Enforcement Testing**
```python
# Create state near steering limit
ego_state = EgoState(..., tire_steering_angle=max_steering_angle - 1e-4)

# Command large steering rate (should saturate)
dynamic_state = DynamicCarState(..., tire_steering_rate=10.0)

# Propagate and verify clipping
next_state = motion_model.propagate_state(ego_state, dynamic_state, dt)
self.assertEqual(next_state.tire_steering_angle, max_steering_angle)
```

7. **Control Delay Validation**
```python
# Test 1: Zero delay → instantaneous response
no_delay_model = KinematicBicycleModel(vehicle, accel_time_constant=0, steering_angle_time_constant=0)
state = no_delay_model._update_commands(current_state, ideal_state, dt)
self.assertAlmostEqual(state.rear_axle_acceleration_2d.x, ideal_state.rear_axle_acceleration_2d.x)

# Test 2: Nonzero delay → filtered response (smaller magnitude)
delayed_model = KinematicBicycleModel(vehicle, accel_time_constant=0.2, ...)
state = delayed_model._update_commands(current_state, ideal_state, dt)
self.assertTrue(state.rear_axle_acceleration_2d.x < ideal_state.rear_axle_acceleration_2d.x)
```

## 3. Architecture & Design Patterns

### 1. **Separation of Concerns**: Different test files for different components
- `test_log_playback.py` - LogPlaybackController only
- `test_perfect_tracking.py` - PerfectTrackingController only
- `test_utils.py` - Utility functions only
- `motion_model/test/test_kinematic_motion_model.py` - Motion model only
- `tracker/test/test_lqr_tracker.py` - LQR tracker only (separate from this module)

### 2. **Mocking Strategy Spectrum**: From heavy mocking to integration tests
- **Heavy mocking** (test_log_playback.py): All dependencies mocked, tests controller logic in isolation
- **Partial mocking** (test_kinematic_motion_model.py): Real data structures, mocked scenario
- **Integration testing** (test_perfect_tracking.py): Real scenario, real trajectory, real controller
- **Pure unit testing** (test_utils.py): No mocks, just numerical validation

### 3. **PropertyMock Pattern**: Testing property assignments
- Python properties are tricky to mock - standard Mock doesn't track property sets
- `PropertyMock` with `@patch.object(..., create=True, new_callable=PropertyMock)` enables tracking
- Used in test_constructor to verify `self._scenario = scenario` and `self._current_iteration = 0`
- **AIDEV-NOTE**: `create=True` is critical - property doesn't exist until first assignment!

### 4. **Randomized Testing**: Statistical validation
- test_utils.py uses 100 random test cases to validate forward_integrate()
- Sets `np.random.seed(0)` for reproducibility
- Covers parameter space better than hand-picked test cases
- Pattern: Generate random inputs → compute both ways → compare

### 5. **Golden Reference Testing**: Compare to known-good data
- test_perfect_tracking.py: Controller output should match `scenario.get_ego_state_at_iteration()`
- test_kinematic_motion_model.py: State derivatives should match manually computed physics equations
- Pattern: Compute expected from first principles, compare to implementation

### 6. **Boundary Testing**: Validate constraints at limits
- test_kinematic_motion_model.py: `test_limit_steering_angle` tests saturation exactly at max
- Sets `tire_steering_angle = max_steering_angle - 1e-4` (just below limit)
- Commands `tire_steering_rate = 10.0` (large, should saturate)
- Verifies output equals `max_steering_angle` (clipped)

### 7. **Temporal Simulation Testing**: Multi-step propagation
- test_perfect_tracking.py: Tests controller over multiple timesteps
- Initial state → update_state() → get_state() → compare to iteration 2
- Validates state propagation logic, not just single-step computation

## 4. Dependencies (What We Import)

### Internal nuPlan (Documented ✅)
**Controller Module Under Test**:
- ✅ `nuplan.planning.simulation.controller.log_playback` - LogPlaybackController
- ✅ `nuplan.planning.simulation.controller.perfect_tracking` - PerfectTrackingController
- ✅ `nuplan.planning.simulation.controller.utils` - forward_integrate()
- ✅ `nuplan.planning.simulation.controller.motion_model.kinematic_bicycle` - KinematicBicycleModel

**Supporting Modules**:
- ✅ `nuplan.common.actor_state.ego_state` - EgoState
- ✅ `nuplan.common.actor_state.dynamic_car_state` - DynamicCarState
- ✅ `nuplan.common.actor_state.state_representation` - StateSE2, StateVector2D, TimePoint
- ✅ `nuplan.common.actor_state.vehicle_parameters` - get_pacifica_parameters()
- ✅ `nuplan.common.actor_state.car_footprint` - CarFootprint
- ✅ `nuplan.planning.simulation.trajectory.abstract_trajectory` - AbstractTrajectory
- ✅ `nuplan.planning.simulation.trajectory.interpolated_trajectory` - InterpolatedTrajectory
- ✅ `nuplan.planning.simulation.simulation_time_controller.simulation_iteration` - SimulationIteration

**Test Utilities**:
- `nuplan.planning.scenario_builder.test.mock_abstract_scenario` - MockAbstractScenario
- `nuplan.planning.scenario_builder.abstract_scenario` - AbstractScenario (for mocking spec)
- ✅ `nuplan.common.actor_state.test.test_utils` - get_sample_ego_state()

### External Dependencies
- **unittest** - Standard Python test framework (TestCase, main)
- **unittest.mock** - Mocking library (MagicMock, Mock, PropertyMock, patch)
- **numpy** - Numerical operations (cos, sin, tan, random, seed)

## 5. Dependents (Who Uses This Module?)

### Test Runners
- **pytest** - Discovers and runs tests via `pytest nuplan/planning/simulation/controller/test/`
- **unittest** - Can run directly via `python -m unittest discover`
- **Bazel** - Build system runs tests via BUILD file targets

### CI/CD Pipelines
- Pre-commit hooks may run subset of tests
- GitHub Actions / Jenkins run full test suite on PR
- Coverage reports track test coverage metrics

### Developers
- **G Money** - Runs tests during controller development and debugging
- **AI Assistants** - Reference tests to understand controller behavior and usage patterns
- **Contributors** - Add tests when implementing new controllers or fixing bugs

### Related Test Modules (Not covered in this file)
- `tracker/test/` - Tests for LQR, iLQR, and tracker utilities (separate test suite)
- `nuplan/planning/simulation/test/` - Integration tests for full simulation loop
- `nuplan/planning/script/test/` - End-to-end tests for training and simulation scripts

**AIDEV-NOTE**: This test module only covers controller base classes and motion models. Tracker tests are in `tracker/test/`, not here!

## 6. Critical Files (Prioritized)

### Priority 1: Controller Tests (Start Here!)

1. **`test_perfect_tracking.py`** (46 lines) - **SIMPLEST TEST - READ FIRST!**
   - Single integration test: `test_perfect_tracker()`
   - Uses `MockAbstractScenario` for realistic data
   - Validates tracking accuracy over 2 timesteps
   - Clean, minimal dependencies
   - **Perfect starting point** for understanding controller testing

2. **`test_log_playback.py`** (81 lines) - **MOCKING SHOWCASE**
   - 3 tests: `test_constructor`, `test_get_state`, `test_update_state`
   - Heavy mocking via `unittest.mock` - all dependencies isolated
   - PropertyMock pattern for constructor validation
   - Context manager pattern for temporary mocks
   - **Best reference** for learning Python mocking techniques

3. **`test_utils.py`** (32 lines) - **PURE UNIT TEST**
   - Single test: `test_forward_integrate()`
   - Randomized testing (100 cases)
   - No mocks, no dependencies, pure numerical validation
   - **Clearest example** of simple unit test structure

### Priority 2: Motion Model Tests (Physics Validation)

4. **`motion_model/test/test_kinematic_motion_model.py`** (167 lines) - **COMPREHENSIVE PHYSICS TEST**
   - 4 tests: state derivatives, propagation, steering limits, control delays
   - Validates kinematic bicycle equations
   - Tests constraint enforcement (steering saturation)
   - Tests control delay filters (low-pass filtering)
   - **Critical reference** for understanding vehicle physics in nuPlan

### Priority 3: Supporting Files

5. **`__init__.py`** (0 lines) - Empty module marker (controller/test/)
6. **`motion_model/test/__init__.py`** (0 lines) - Empty module marker (motion_model/test/)

**File Count**: 3 test files in `controller/test/`, 1 test file in `motion_model/test/`, 2 empty `__init__.py`

**Total Test Coverage**: ~325 lines of test code across all files

**Note**: Tracker tests (`tracker/test/test_lqr_tracker.py`, `test_ilqr_tracker.py`, `test_tracker_utils.py`) are NOT documented here - they are a separate subsystem with their own test suite.

## 7. Common Usage Patterns

### 1. Running Tests

```bash
# Run all controller tests (including motion_model/)
pytest nuplan/planning/simulation/controller/test/ -v

# Run specific test file
pytest nuplan/planning/simulation/controller/test/test_perfect_tracking.py -v

# Run specific test method
pytest nuplan/planning/simulation/controller/test/test_log_playback.py::TestLogPlaybackController::test_get_state -v

# Run with coverage
pytest nuplan/planning/simulation/controller/test/ --cov=nuplan.planning.simulation.controller --cov-report=html

# Run via unittest (alternative)
python -m unittest nuplan.planning.simulation.controller.test.test_utils
```

### 2. Creating Mock Scenario (test_log_playback.py pattern)

```python
from unittest.mock import MagicMock
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario

def setUp(self):
    # Create mock scenario with spec (enables autocomplete, catches typos)
    self.scenario = MagicMock(spec=AbstractScenario)

    # Configure return value for method
    self.scenario.get_ego_state_at_iteration.return_value = EXPECTED_STATE

    # Use in test
    controller = LogPlaybackController(self.scenario)
    state = controller.get_state()

    # Verify mock was called correctly
    self.scenario.get_ego_state_at_iteration.assert_called_once_with(0)
```

### 3. Using MockAbstractScenario (test_perfect_tracking.py pattern)

```python
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario
from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

# Create realistic scenario with expert trajectory
initial_time_point = TimePoint(0)
scenario = MockAbstractScenario(initial_time_us=initial_time_point)

# Extract expert trajectory (ground truth)
trajectory = InterpolatedTrajectory(list(scenario.get_expert_ego_trajectory()))

# Use for controller validation
controller = PerfectTrackingController(scenario)
controller.update_state(iteration, next_iteration, ego_state, trajectory)

# Compare to ground truth
actual_state = controller.get_state()
expected_state = scenario.get_ego_state_at_iteration(2)
assert actual_state.rear_axle.x ≈ expected_state.rear_axle.x
```

### 4. Testing Constructor with PropertyMock (test_log_playback.py pattern)

```python
from unittest.mock import patch, PropertyMock

@patch.object(LogPlaybackController, 'current_iteration', create=True, new_callable=PropertyMock)
@patch.object(LogPlaybackController, 'scenario', create=True, new_callable=PropertyMock)
def test_constructor(self, scenario_mock, iteration_mock):
    """Test constructor sets properties correctly"""
    controller = LogPlaybackController(self.scenario)

    # Verify property setters called (not regular attribute assignment!)
    scenario_mock.assert_called_once_with(self.scenario)
    iteration_mock.assert_called_once_with(0)
```

**CRITICAL**: Decorator order matters! Bottom decorator (`scenario`) is outermost argument, top decorator (`current_iteration`) is second argument.

### 5. Testing Motion Model Physics (test_kinematic_motion_model.py pattern)

```python
def test_get_state_dot(self):
    """Validate state derivatives match kinematic equations"""
    # Compute expected values from physics
    velocity = self.ego_state.dynamic_car_state.rear_axle_velocity_2d.x
    heading = self.ego_state.rear_axle.heading
    steering = self.ego_state.tire_steering_angle
    wheelbase = self.vehicle.wheel_base

    x_dot = velocity * np.cos(heading)
    y_dot = velocity * np.sin(heading)
    yaw_dot = velocity * np.tan(steering) / wheelbase

    # Test motion model
    state_dot = self.motion_model.get_state_dot(self.ego_state)

    # Compare
    self.assertEqual(state_dot.rear_axle, StateSE2(x_dot, y_dot, yaw_dot))
```

### 6. Testing Constraint Enforcement (test_kinematic_motion_model.py pattern)

```python
def test_limit_steering_angle(self):
    """Test steering angle saturation at max limit"""
    # Create state near steering limit
    dynamic_state = DynamicCarState.build_from_rear_axle(
        rear_axle_to_center=self.vehicle.rear_axle_to_center,
        rear_axle_velocity_2d=StateVector2D(0.0, 0.0),
        rear_axle_acceleration_2d=StateVector2D(0.0, 0.0),
        tire_steering_rate=10.0  # Large rate → should saturate
    )

    car_footprint = CarFootprint.build_from_rear_axle(
        rear_axle_pose=StateSE2(0.0, 0.0, 0.0),
        vehicle_parameters=self.vehicle
    )

    ego_state = EgoState(
        car_footprint,
        dynamic_state,
        tire_steering_angle=self.motion_model._max_steering_angle - 1e-4,  # Just below limit
        is_in_auto_mode=True,
        time_point=TimePoint(0)
    )

    # Propagate (should clip to max)
    next_state = self.motion_model.propagate_state(
        ego_state, dynamic_state, TimePoint(1000000)  # 1 second
    )

    # Verify saturation
    self.assertEqual(next_state.tire_steering_angle, self.motion_model._max_steering_angle)
```

### 7. Testing Control Delays (test_kinematic_motion_model.py pattern)

```python
def test_update_command(self):
    """Test control delay filtering behavior"""
    # Setup
    state = EgoState(...)  # Current state (zero velocity, zero accel)
    ideal_state = DynamicCarState.build_from_rear_axle(
        ...,
        rear_axle_acceleration_2d=StateVector2D(1.0, 0.0),  # Desired accel
        tire_steering_rate=0.5  # Desired steering rate
    )
    dt = TimePoint(1000000)  # 1 second

    # Test 1: Zero time constant → no delay
    no_delay_model = KinematicBicycleModel(
        self.vehicle,
        accel_time_constant=0.0,
        steering_angle_time_constant=0.0
    )
    result = no_delay_model._update_commands(state, ideal_state, dt)
    self.assertAlmostEqual(
        result.rear_axle_acceleration_2d.x,
        ideal_state.rear_axle_acceleration_2d.x,
        places=10
    )

    # Test 2: Nonzero time constant → filtering (smaller magnitude)
    delayed_model = KinematicBicycleModel(self.vehicle)  # Default delays
    result = delayed_model._update_commands(state, ideal_state, dt)
    self.assertLess(
        result.rear_axle_acceleration_2d.x,
        ideal_state.rear_axle_acceleration_2d.x
    )
```

### 8. Randomized Testing (test_utils.py pattern)

```python
def setUp(self):
    """Generate random test parameters"""
    np.random.seed(0)  # Reproducibility
    self.inits = np.random.rand(100)
    self.deltas = np.random.rand(100)
    self.sampling_times = np.random.randint(1000000, size=100)

def test_forward_integrate(self):
    """Test integration over random parameter space"""
    for init, delta, sampling_time in zip(self.inits, self.deltas, self.sampling_times):
        result = forward_integrate(init, delta, TimePoint(sampling_time))
        expected = init + delta * sampling_time * 1e-6  # Manual computation
        self.assertAlmostEqual(result, expected)
```

## 8. Gotchas & Edge Cases

### Test Execution Issues

1. **PropertyMock requires create=True for non-existent properties**
   - **Issue**: `@patch.object(Class, 'property_name', new_callable=PropertyMock)` fails if property doesn't exist yet
   - **Symptom**: `AttributeError: does not have the attribute 'property_name'`
   - **Fix**: Add `create=True` → `@patch.object(Class, 'property_name', create=True, new_callable=PropertyMock)`
   - **Why**: Properties are created via `@property` decorator, which doesn't exist before first use
   - **Example**: test_log_playback.py line 35-36

2. **Decorator order affects test method arguments**
   - **Issue**: Patch decorators are stacked, argument order is bottom-to-top
   - **Symptom**: Test receives arguments in unexpected order
   - **Rule**: Bottom decorator = first argument, top decorator = last argument
   - **Example**:
     ```python
     @patch.object(Class, 'attr1')  # Third argument
     @patch.object(Class, 'attr2')  # Second argument
     def test(self, attr2, attr1):  # Bottom-to-top!
     ```

3. **Mock.assert_called_once() vs assert_called_once_with()**
   - **Issue**: `assert_called_once()` checks call count only, not arguments
   - **Common mistake**: Using `assert_called_once()` when you need to verify arguments
   - **Fix**: Use `assert_called_once_with(expected_arg)` to check both
   - **Example**: test_log_playback.py line 45, 63

4. **MagicMock(spec=...) doesn't validate return types**
   - **Issue**: Mock returns Mock by default, even if spec says method returns int
   - **Symptom**: Type errors in code under test, but mock doesn't catch it
   - **Fix**: Explicitly set `mock.method.return_value = expected_value`
   - **Prevention**: Use `spec=True` + manual return value configuration
   - **Example**: test_log_playback.py line 29

5. **TimePoint microsecond vs seconds confusion**
   - **Issue**: TimePoint stores microseconds internally, but integration uses seconds
   - **Symptom**: Off-by-1e6 errors in test assertions
   - **Fix**: Use `TimePoint.time_s` for seconds, `TimePoint.time_us` for microseconds
   - **Example**: test_utils.py line 27 - manual computation uses `* 1e-6`

6. **Floating point comparison with assertEqual fails**
   - **Issue**: `assertEqual(0.30000000000000004, 0.3)` fails due to floating point error
   - **Symptom**: AssertionError on seemingly equal values
   - **Fix**: Use `assertAlmostEqual(actual, expected, places=7)` for floats
   - **Example**: test_perfect_tracking.py line 27, 28, 29

7. **np.random.seed() required for reproducible randomized tests**
   - **Issue**: Randomized tests pass/fail non-deterministically
   - **Symptom**: Test suite flakiness, CI failures
   - **Fix**: Call `np.random.seed(0)` in setUp() before generating random data
   - **Example**: test_utils.py line 16

8. **Mock scenario doesn't automatically update iteration index**
   - **Issue**: Mocked `next_iteration.index` returns same value every call
   - **Symptom**: Controller gets stuck, infinite loop, or assertion failure
   - **Fix**: Use `PropertyMock(return_value=...)` to set specific index value
   - **Example**: test_log_playback.py line 30-31

9. **Testing private methods (_update_commands) is brittle**
   - **Issue**: test_kinematic_motion_model.py tests `_update_commands()` directly (private method)
   - **Symptom**: Tests break on internal refactoring, even if public API unchanged
   - **Trade-off**: Necessary for unit testing control delays in isolation
   - **Best practice**: Prefer testing public API (`propagate_state`), use private method tests sparingly
   - **Example**: test_kinematic_motion_model.py line 145, 157

10. **MockAbstractScenario doesn't support all AbstractScenario methods**
    - **Issue**: Mock scenario has limited method implementations
    - **Symptom**: AttributeError or NotImplementedError when calling certain scenario methods
    - **Fix**: Check MockAbstractScenario source before using - may need MagicMock instead
    - **Example**: Only `get_expert_ego_trajectory()` and `get_ego_state_at_iteration()` used in tests

### Physics & Numerical Issues

11. **Steering angle near π/2 causes tan() to explode**
    - **Issue**: Kinematic bicycle model uses `tan(steering_angle)` in yaw rate
    - **Symptom**: Huge yaw rates, NaN propagation, test failures
    - **Prevention**: Test steering limits are enforced (test_limit_steering_angle)
    - **Real constraint**: max_steering_angle = π/3 (60°) prevents this
    - **Example**: test_kinematic_motion_model.py line 89-112

12. **Control delay filter approaches ideal value exponentially, never reaches it**
    - **Issue**: Low-pass filter: `y(t) = y(t-dt) + (dt/(dt+τ)) * (ideal - y(t-dt))`
    - **Symptom**: `assertAlmostEqual(filtered, ideal)` fails for small timesteps
    - **Expected**: Filtered value is always < ideal for step input
    - **Fix**: Test that filtered < ideal, not filtered == ideal
    - **Example**: test_kinematic_motion_model.py line 158-162

13. **Euler integration accumulates error over many timesteps**
    - **Issue**: test_perfect_tracking only tests 2 timesteps, not long horizons
    - **Missing coverage**: Long trajectory tracking (80+ steps) error accumulation
    - **Risk**: Integration drift not validated
    - **AIDEV-TODO**: Add test for long-horizon tracking accuracy

14. **Zero velocity causes division by zero in some formulas**
    - **Issue**: Curvature = yaw_rate / velocity → undefined at v=0
    - **Not tested**: test_kinematic_motion_model doesn't test v=0 edge case
    - **Real code**: KinematicBicycleModel handles this (yaw_rate = 0 when v=0)
    - **AIDEV-TODO**: Add test for zero velocity propagation

15. **Lateral velocity hardcoded to zero, not validated**
    - **Issue**: Kinematic assumption, but not explicitly tested
    - **Missing test**: Verify `state.rear_axle_velocity_2d.y == 0.0` always
    - **Example**: test_kinematic_motion_model line 70 - checks y velocity is 0, but only in one test
    - **AIDEV-TODO**: Add dedicated test for lateral velocity constraint

### Test Coverage Gaps

16. **No test for velocity > 50 m/s safety check in PerfectTrackingController**
    - **Issue**: PerfectTrackingController.update_state() has hardcoded 50 m/s check
    - **Missing**: Test that RuntimeError is raised for high-velocity trajectory
    - **Code**: perfect_tracking.py line 44-46
    - **AIDEV-TODO**: Add test_velocity_limit() to test_perfect_tracking.py

17. **No test for TwoStageController (most important controller!)**
    - **Issue**: TwoStageController is used in all realistic simulations, but no unit test
    - **Coverage gap**: Integration of tracker + motion model not tested
    - **Workaround**: Tested indirectly via tracker tests and simulation tests
    - **AIDEV-TODO**: Add test_two_stage_controller.py with tracker + motion model integration tests

18. **No test for trajectory shorter than next_iteration time**
    - **Issue**: PerfectTrackingController assumes trajectory covers next_iteration.time_point
    - **Missing**: Test that assertion fails if trajectory too short
    - **Code**: perfect_tracking.py line 41 - assertion checks this
    - **AIDEV-TODO**: Add test_trajectory_too_short() expecting AssertionError

19. **No test for controller.reset() behavior**
    - **Issue**: All controllers have reset(), but no test verifies state clearing
    - **Missing**: Test that reset() clears _current_state to None
    - **Workaround**: Integration tests implicitly test reset between scenarios
    - **AIDEV-TODO**: Add test_reset() to each controller test class

20. **No test for heading wrapping in motion model**
    - **Issue**: KinematicBicycleModel wraps heading to [-π, π], but not explicitly tested
    - **Missing**: Test propagation with heading near ±π boundary
    - **Code**: kinematic_bicycle.py line 123 - `principal_value()` wraps heading
    - **AIDEV-TODO**: Add test_heading_wrapping() with heading = π - 0.1, large yaw rate

21. **No test for angular acceleration computation**
    - **Issue**: Angular acceleration is computed via finite difference, but not validated
    - **Missing**: Test that `angular_accel = (angular_velocity - prev_angular_velocity) / dt`
    - **Code**: kinematic_bicycle.py line 136
    - **AIDEV-TODO**: Add test comparing angular_accel to expected finite difference

22. **Mock tests don't validate AbstractEgoController interface compliance**
    - **Issue**: test_log_playback uses mocks, doesn't verify controller implements AbstractEgoController
    - **Risk**: Controller could violate interface contract, mocks wouldn't catch it
    - **Fix**: Add `isinstance(controller, AbstractEgoController)` assertion
    - **Example**: test_log_playback.py doesn't check interface compliance

## 9. Performance Considerations

### Test Execution Speed
- **Fast**: All tests run in < 1 second total
  - test_utils.py: ~100ms (100 random cases)
  - test_log_playback.py: ~50ms (all mocked, no computation)
  - test_perfect_tracking.py: ~100ms (integration test, real scenario)
  - test_kinematic_motion_model.py: ~150ms (physics validation)

### Memory Footprint
- **Minimal**: All tests use < 10 MB RAM
  - Mocks are lightweight (no real scenario data loaded)
  - MockAbstractScenario generates small trajectories (~10 states)
  - No large dataset dependencies

### Parallelization
- **Fully parallelizable**: Tests are independent, no shared state
- **pytest -n auto**: Can run tests in parallel workers for speedup
- **Bazel**: Can parallelize test targets automatically

### Coverage Measurement Overhead
- **Moderate**: `pytest --cov` adds ~20-30% overhead
- **Negligible for small test suite**: Still < 2 seconds total
- **Recommendation**: Always run with coverage in CI/CD

### Randomized Test Considerations
- **Deterministic randomness**: `np.random.seed(0)` ensures reproducibility
- **No flakiness**: Randomized tests are stable (same random sequence every run)
- **Scalability**: 100 random cases is reasonable, 10,000 would slow tests significantly

## 10. Related Documentation

### Cross-References (Documented ✅)

**Module Under Test**:
- ✅ `nuplan/planning/simulation/controller/CLAUDE.md` - Controller implementations (PerfectTracking, LogPlayback, TwoStage)
- ✅ `nuplan/planning/simulation/controller/motion_model/CLAUDE.md` - KinematicBicycleModel physics

**Dependencies**:
- ✅ `nuplan/common/actor_state/CLAUDE.md` - EgoState, DynamicCarState, VehicleParameters
- ✅ `nuplan/planning/simulation/trajectory/CLAUDE.md` - InterpolatedTrajectory
- ✅ `nuplan/planning/simulation/simulation_time_controller/CLAUDE.md` - SimulationIteration

**Test Utilities**:
- `nuplan/planning/scenario_builder/test/mock_abstract_scenario.py` - MockAbstractScenario fixture
- ✅ `nuplan/common/actor_state/test/test_utils.py` - get_sample_ego_state() utility

**Related Test Suites**:
- `nuplan/planning/simulation/controller/tracker/test/` - LQR, iLQR, tracker utilities tests (separate module)
- `nuplan/planning/simulation/test/` - Full simulation loop integration tests
- `nuplan/planning/script/test/` - End-to-end simulation script tests

### Undocumented Dependencies (Future ⏳)
- `nuplan/planning/scenario_builder/test/mock_abstract_scenario.py` - Mock scenario implementation (not documented)

### Testing Resources
- **Python unittest docs**: https://docs.python.org/3/library/unittest.html
- **unittest.mock guide**: https://docs.python.org/3/library/unittest.mock.html
- **pytest documentation**: https://docs.pytest.org/
- **Mocking best practices**: https://realpython.com/python-mock-library/

## 11. AIDEV Notes

### Test Design Philosophy
- **Spectrum of testing**: From heavy mocking (isolation) to integration (realism)
- **Golden reference**: Compare to known-good data (expert trajectory, physics equations)
- **Boundary testing**: Validate constraints at limits (steering saturation, velocity limits)
- **Randomized testing**: Cover parameter space statistically (forward_integrate)

### Common Testing Mistakes
- **Forgetting create=True with PropertyMock** - Most common mock failure
- **Confusing decorator order** - Bottom decorator is first argument
- **Using assertEqual for floats** - Use assertAlmostEqual instead
- **Not setting random seed** - Leads to flaky tests
- **Testing private methods** - Brittle, breaks on refactoring

### Test Coverage Gaps (High Priority)
- **AIDEV-TODO**: Add test_two_stage_controller.py - THE most important controller, no unit test!
- **AIDEV-TODO**: Add velocity limit test to test_perfect_tracking.py (>50 m/s RuntimeError)
- **AIDEV-TODO**: Add controller.reset() tests to all controller test classes
- **AIDEV-TODO**: Add long-horizon tracking test (80+ timesteps, check error accumulation)
- **AIDEV-TODO**: Add heading wrapping test (near ±π boundary)
- **AIDEV-TODO**: Add zero velocity test to test_kinematic_motion_model.py

### Test Coverage Gaps (Medium Priority)
- **AIDEV-TODO**: Add test for trajectory too short (PerfectTracking assertion)
- **AIDEV-TODO**: Add test for angular acceleration computation (finite difference)
- **AIDEV-TODO**: Add lateral velocity constraint test (always zero)
- **AIDEV-TODO**: Add interface compliance tests (isinstance checks)

### Performance Optimization Opportunities
- **Not needed**: Tests run in < 1 second, parallelization works well
- **Coverage overhead acceptable**: ~20-30% slowdown for coverage measurement
- **Randomized tests scalable**: 100 cases is sweet spot (fast + good coverage)

### Mocking Strategy Recommendations
1. **Use MagicMock(spec=...) for type safety** - Catches attribute typos
2. **Use PropertyMock with create=True for properties** - Required for non-existent properties
3. **Use context managers for temporary mocks** - Cleaner than decorator stacking
4. **Use MockAbstractScenario for realistic data** - Better than full mocking for integration tests
5. **Mock at module boundary, not internal details** - More robust to refactoring

### Physics Testing Best Practices
1. **Compute expected values from first principles** - Don't copy implementation
2. **Test equations directly, not just end-to-end** - get_state_dot() validation
3. **Test constraints at boundaries** - Saturation, wrapping, limits
4. **Test control delays with zero time constant first** - Isolate filtering logic
5. **Use assertAlmostEqual for physics** - Floating point errors inevitable

### Future Test Improvements
- **Property-based testing**: Use Hypothesis for generative testing
- **Mutation testing**: Use mutmut to validate test quality
- **Snapshot testing**: For complex state comparisons
- **Performance regression tests**: Track test execution time over commits
- **Coverage targets**: Aim for 90%+ line coverage, 80%+ branch coverage

### Testing Patterns to Avoid
- ❌ **Testing implementation details** - Focus on public API
- ❌ **Overmocking** - Use real data when possible (MockAbstractScenario > MagicMock)
- ❌ **One giant test** - Split into focused tests per behavior
- ❌ **Magic numbers** - Use named constants (CURRENT_ITERATION, not 12)
- ❌ **No assertions** - Every test must have at least one assertion

### Useful Test Debugging Commands
```bash
# Run single test with output
pytest nuplan/planning/simulation/controller/test/test_perfect_tracking.py::TestPerfectTracking::test_perfect_tracker -v -s

# Run with debugger on failure
pytest nuplan/planning/simulation/controller/test/ --pdb

# Show local variables on failure
pytest nuplan/planning/simulation/controller/test/ -l

# Run only failed tests from last run
pytest nuplan/planning/simulation/controller/test/ --lf

# Show test durations (slowest first)
pytest nuplan/planning/simulation/controller/test/ --durations=10
```

### Test Maintenance Guidelines
- **Update tests when adding features** - Don't let coverage regress
- **Update tests when fixing bugs** - Add regression test for bug
- **Refactor tests when they become brittle** - High test maintenance = bad design
- **Document complex test setup** - Why are we mocking this way?
- **Keep tests fast** - Slow tests won't be run frequently

---

**CRITICAL REMINDER**: This test module ONLY covers controller base classes and motion models. Tracker tests (LQR, iLQR, tracker_utils) are in `tracker/test/` - a separate test suite with ~500+ lines of additional test code!
