# nuplan/planning/simulation/observation/test/

## 1. Purpose & Responsibility

This directory contains **unit and integration tests** for the observation layer, validating sensor simulation, abstract observation interfaces, IDM agents, and ML-based observation models. Tests ensure observation data (detections, tracks, traffic lights, IDM states) are correctly generated, transformed, and consumed by planners in both open-loop and closed-loop simulation contexts.

## 2. Key Abstractions

### Core Test Patterns

**AbstractObservation Testing**
- Validates observation type registration and discovery
- Tests observation data flow (raw sensors → processed observations → planner input)
- Ensures consistency across observation backends (oracle, IDM, ML)

**IDM Agent Testing**
- Unit tests for IDM physics (acceleration, spacing, lead agent selection)
- Integration tests for multi-agent IDM simulation
- Parameter sensitivity analysis (a, b, delta, s0, T, v0)

**ML Observation Model Testing**
- Tests for ML-based agent tracking and prediction
- Validation of observation feature extraction
- Integration with trained models (mocked and real)

**Sensor Simulation Testing**
- Tests for simulated camera, lidar, radar observations
- Noise injection and occlusion modeling validation
- Sensor fusion pipeline testing

### Key Test Files

```python
# Test abstract observation interface
test_abstract_observation.py
    - Test observation type registration
    - Test observation data structure validation
    - Test observation history management

# Test IDM observations
test_idm_agents.py
    - Test IDM physics formula correctness
    - Test lead agent identification logic
    - Test multi-agent IDM coordination
    - Test parameter boundary conditions

# Test ML observation models
test_ml_agents.py
    - Test ML feature extraction from observations
    - Test model inference integration
    - Test observation prediction accuracy
    - Test temporal consistency

# Test sensor simulation
test_sensors.py
    - Test camera image generation
    - Test lidar point cloud generation
    - Test sensor noise models
    - Test occlusion handling
```

## 3. Architecture & Design Patterns

### Test Organization Strategy

**Unit Tests**
- Isolated component testing (IDM formulas, observation types)
- Fast execution (no database, no heavy I/O)
- Parameterized tests for edge cases

**Integration Tests**
- End-to-end observation pipelines (sensors → observations → planners)
- Uses mock scenarios or mini dataset
- Validates data flow across module boundaries

**Fixture Patterns**
```python
# Common test fixtures
@pytest.fixture
def mock_ego_state() -> EgoState:
    """Standard ego state for observation tests"""
    return EgoState(...)

@pytest.fixture
def mock_detections() -> DetectionsTracks:
    """Standard detection data for planner input"""
    return DetectionsTracks(...)

@pytest.fixture
def idm_parameters() -> IDMParameters:
    """Standard IDM params for reproducibility"""
    return IDMParameters(a=1.0, b=1.5, delta=4, s0=2.0, T=1.5, v0=15.0)
```

### Test Data Management

**Mock Data Strategy**
- Lightweight mocks for unit tests (avoid database dependencies)
- Deterministic scenarios for reproducibility
- Parameterized inputs for boundary testing

**Real Data Integration**
- Mini dataset scenarios for integration tests
- Cached observations to speed up test suite
- Subset of scenario types for coverage

## 4. Dependencies

### Internal (nuPlan)

**Direct Dependencies**:
- ✅ `nuplan.planning.simulation.observation.abstract_observation` - Interface under test
- ✅ `nuplan.planning.simulation.observation.idm` - IDM implementation (if testing IDM)
- ✅ `nuplan.planning.simulation.observation.observation_type` - DetectionsTracks types
- ✅ `nuplan.common.actor_state.state_representation` - Mock state data
- ✅ `nuplan.database.nuplan_db` - Scenario loading for integration tests

**Test Infrastructure**:
- `pytest` - Test framework
- `pytest-mock` - Mocking utilities
- `numpy.testing` - Array assertions
- `unittest.mock` - Standard library mocking

### External

- `numpy` - Numerical assertions
- `pytest` - Test runner
- `pytest-xdist` - Parallel test execution
- `hypothesis` - Property-based testing (if used)

## 5. Dependents (Who Uses This Module?)

### Direct Consumers

**CI/CD Pipeline**:
- Pre-commit hooks run subset of tests
- GitHub Actions run full test suite
- Coverage reports track observation layer coverage

**Developers**:
- Run tests locally before commits
- Use tests as documentation (how to use observation APIs)
- Debug observation issues with failing tests

### Use Cases

1. **Validate New Observation Types**
   - Add tests when implementing custom observations
   - Ensure new observations conform to AbstractObservation interface
   - Test integration with simulation loop

2. **Regression Testing**
   - Detect breaking changes in observation APIs
   - Ensure IDM physics consistency across updates
   - Validate ML model observation compatibility

3. **Performance Benchmarking**
   - Profile observation generation overhead
   - Measure IDM computation time vs scenario complexity
   - Benchmark sensor simulation throughput

## 6. Critical Files (Prioritized)

### Priority 1: Core Interface Tests

1. **`test_abstract_observation.py`**
   - Tests for AbstractObservation base class
   - Validates observation type registration
   - Tests observation history management
   - **Key for**: Understanding observation API contracts

2. **`test_idm_agents.py`** (if exists)
   - Unit tests for IDM physics
   - Lead agent selection tests
   - Multi-agent coordination tests
   - **Key for**: IDM correctness and edge cases

### Priority 2: Integration Tests

3. **`test_observation_integration.py`** (if exists)
   - End-to-end observation pipeline tests
   - Tests observation → planner data flow
   - Validates observation consistency across backends
   - **Key for**: System-level validation

4. **`test_ml_agents.py`** (if exists)
   - Tests for ML-based observation models
   - Feature extraction validation
   - Model inference integration
   - **Key for**: ML observation correctness

### Priority 3: Utilities & Fixtures

5. **`conftest.py`**
   - Shared test fixtures
   - Mock data generators
   - Test configuration
   - **Key for**: Understanding test setup

6. **`test_sensors.py`** (if exists)
   - Sensor simulation tests
   - Noise model validation
   - Occlusion handling tests
   - **Key for**: Sensor simulation correctness

## 7. Common Usage Patterns

### Running Observation Tests

```bash
# Run all observation tests
pytest nuplan/planning/simulation/observation/test/

# Run specific test file
pytest nuplan/planning/simulation/observation/test/test_idm_agents.py

# Run with coverage
pytest --cov=nuplan.planning.simulation.observation \
       --cov-report=html \
       nuplan/planning/simulation/observation/test/

# Run in parallel (faster)
pytest -n auto nuplan/planning/simulation/observation/test/

# Run only fast tests (skip integration)
pytest -m "not integration" nuplan/planning/simulation/observation/test/
```

### Writing New Observation Tests

```python
import pytest
from nuplan.planning.simulation.observation.abstract_observation import AbstractObservation
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks

class TestCustomObservation:
    """Tests for custom observation implementation"""

    def test_observation_type(self, custom_observation: AbstractObservation):
        """Test observation returns correct type"""
        obs_type = custom_observation.observation_type()
        assert obs_type == DetectionsTracks

    def test_observation_data_structure(self, custom_observation: AbstractObservation):
        """Test observation data has required fields"""
        obs = custom_observation.get_observation()
        assert hasattr(obs, 'tracked_objects')
        assert len(obs.tracked_objects) >= 0

    @pytest.mark.parametrize("ego_speed", [0.0, 5.0, 15.0, 30.0])
    def test_observation_at_different_speeds(self, custom_observation, ego_speed):
        """Test observation consistency across ego speeds"""
        # Setup ego at different speeds
        custom_observation.set_ego_speed(ego_speed)
        obs = custom_observation.get_observation()
        # Validate observation invariants
        assert obs is not None
```

### Testing IDM Agents

```python
def test_idm_acceleration_free_flow():
    """Test IDM acceleration in free-flow (no lead agent)"""
    idm = IDMLeadAgentObservation(
        target_velocity=15.0,  # m/s
        min_gap_to_lead_agent=2.0,
        headway_time=1.5,
        accel_max=1.0,
        decel_max=1.5,
    )

    # Ego at 10 m/s, no lead agent
    ego_state = EgoState(velocity=10.0, ...)

    # Should accelerate toward target velocity
    acceleration = idm.compute_acceleration(ego_state, lead_agent=None)
    assert acceleration > 0  # Accelerating
    assert acceleration <= 1.0  # Within max acceleration

def test_idm_acceleration_following():
    """Test IDM acceleration when following lead agent"""
    idm = IDMLeadAgentObservation(...)

    # Ego at 15 m/s, lead agent at 10 m/s, 20m ahead
    ego_state = EgoState(velocity=15.0, ...)
    lead_agent = Agent(velocity=10.0, distance_ahead=20.0, ...)

    # Should decelerate (approaching slower vehicle)
    acceleration = idm.compute_acceleration(ego_state, lead_agent=lead_agent)
    assert acceleration < 0  # Decelerating
    assert acceleration >= -1.5  # Within comfortable deceleration
```

### Integration Testing with Scenarios

```python
@pytest.mark.integration
def test_observation_with_real_scenario(nuplan_scenario):
    """Test observation generation with real scenario data"""
    observation = CustomObservation()

    # Initialize with scenario
    observation.initialize(nuplan_scenario.get_expert_ego_trajectory())

    # Get observations at each timestep
    for iteration in nuplan_scenario:
        obs = observation.get_observation()

        # Validate observation properties
        assert obs.tracked_objects is not None
        assert all(obj.velocity >= 0 for obj in obs.tracked_objects)
        assert len(obs.tracked_objects) <= 100  # Reasonable limit
```

## 8. Gotchas & Edge Cases

### 1. **IDM Collision Detection**
- **Issue**: IDM can produce collisions if parameters poorly tuned
- **Symptom**: Agents crash in simulation despite IDM "safe following"
- **Fix**: Validate IDM parameters meet safety constraints (T ≥ 1.0, s0 ≥ 2.0)
- **Test**: `test_idm_collision_avoidance()` should catch this

### 2. **Observation Timestep Synchronization**
- **Issue**: Observations may lag/lead planner timestep
- **Symptom**: Planner sees stale or future observations
- **Fix**: Ensure observation timestamp matches current_input.iteration.time_point
- **Test**: `test_observation_timestamp_consistency()`

### 3. **Empty Observation Handling**
- **Issue**: Some observations may have zero tracked objects
- **Symptom**: Planner crashes on empty observation list
- **Fix**: Planners must handle `len(tracked_objects) == 0` gracefully
- **Test**: `test_empty_observation_handling()`

### 4. **ML Observation Model Loading**
- **Issue**: ML models may not load correctly in test environment
- **Symptom**: `FileNotFoundError` or `ModuleNotFoundError` for checkpoints
- **Fix**: Use mocked models in unit tests, real models only in integration tests
- **Test**: `test_ml_observation_with_mock_model()`

### 5. **IDM Parameter Boundary Cases**
- **Issue**: IDM unstable with extreme parameters (T=0, a=0, etc.)
- **Symptom**: Infinite acceleration, NaN values, crashes
- **Fix**: Validate parameters in IDM initialization
- **Test**: `test_idm_parameter_validation()` with `pytest.raises(ValueError)`

### 6. **Observation Coordinate Systems**
- **Issue**: Observations may be in different coordinate frames (global vs ego-centric)
- **Symptom**: Planner misinterprets object positions/velocities
- **Fix**: Document coordinate frame in observation API, test transformations
- **Test**: `test_observation_coordinate_frame()`

### 7. **Sensor Noise Reproducibility**
- **Issue**: Sensor noise may not be reproducible across test runs
- **Symptom**: Flaky tests that pass/fail randomly
- **Fix**: Seed RNG in tests (`np.random.seed(42)`)
- **Test**: Use deterministic noise in all sensor tests

### 8. **Observation History Length**
- **Issue**: Observation history may grow unbounded in long simulations
- **Symptom**: Memory leaks in extended simulations
- **Fix**: Implement fixed-size history buffer (e.g., last 10 observations)
- **Test**: `test_observation_history_memory_bounds()`

### 9. **Traffic Light Observation Timing**
- **Issue**: Traffic light states may change mid-observation
- **Symptom**: Inconsistent traffic light data in observations
- **Fix**: Sample traffic light state atomically at observation time
- **Test**: `test_traffic_light_observation_consistency()`

### 10. **Multi-Agent IDM Race Conditions**
- **Issue**: Multiple IDM agents may select same lead agent, causing conflicts
- **Symptom**: Unrealistic multi-agent behaviors
- **Fix**: IDM lead selection should be deterministic and conflict-free
- **Test**: `test_multi_agent_idm_lead_selection()`

### 11. **Observation Caching Invalidation**
- **Issue**: Cached observations may become stale if scenario changes
- **Symptom**: Planner sees outdated observation data
- **Fix**: Invalidate cache on scenario reset or parameter change
- **Test**: `test_observation_cache_invalidation()`

### 12. **Observation Performance Regression**
- **Issue**: Observation generation may become slow with large agent counts
- **Symptom**: Simulation runs slowly, observation bottleneck
- **Fix**: Profile observation code, optimize hot paths
- **Test**: `test_observation_performance_benchmark()` with time limits

## 9. Performance Considerations

**Test Suite Performance**:
- Unit tests: ~1-5 seconds total
- Integration tests: ~30-60 seconds (with database access)
- Full suite with coverage: ~2-5 minutes

**Optimization Strategies**:
- Use `pytest-xdist` for parallel execution (`-n auto`)
- Mark slow tests with `@pytest.mark.slow`, skip in CI
- Cache fixtures (scenario loading, model initialization)
- Mock heavy dependencies (database, ML models) in unit tests

**Benchmarking**:
- Use `pytest-benchmark` for performance regression tests
- Track observation generation time vs agent count
- Monitor memory usage in long-running tests

## 10. Related Documentation

### Cross-References
- ✅ `nuplan/planning/simulation/observation/CLAUDE.md` - Observation layer architecture
- ✅ `nuplan/planning/simulation/observation/idm/CLAUDE.md` - IDM implementation details
- ✅ `nuplan/common/actor_state/test/CLAUDE.md` - State representation test patterns
- `nuplan/planning/simulation/test/CLAUDE.md` - Simulation testing patterns (Phase 2C)
- `nuplan/planning/simulation/planner/test/CLAUDE.md` - Planner testing (Phase 1B)

### External Resources
- **pytest documentation**: https://docs.pytest.org/
- **unittest.mock guide**: https://docs.python.org/3/library/unittest.mock.html
- **IDM model**: Treiber & Kesting (2013) "Traffic Flow Dynamics"

## 11. AIDEV Notes

**Testing Philosophy**:
- Observation tests focus on **data correctness**, not planner logic
- Mock heavy dependencies (database, ML models) to keep tests fast
- Integration tests validate end-to-end pipelines, not individual functions

**Common Test Failures**:
- IDM collision tests fail if parameters too aggressive (reduce a, increase T)
- ML observation tests fail if model checkpoints missing (use mocked models)
- Integration tests slow if database not cached (use mini dataset subset)

**Debugging Tips**:
- Run single test with `-vv` for detailed output: `pytest -vv test_file.py::test_name`
- Use `--pdb` to drop into debugger on failure: `pytest --pdb`
- Check test logs in `$NUPLAN_EXP_ROOT/test_logs/`

**AIDEV-TODO**: Add property-based tests with Hypothesis for observation invariants
**AIDEV-NOTE**: Test suite should run in <5 minutes to maintain developer productivity
