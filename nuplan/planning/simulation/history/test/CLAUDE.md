# nuplan/planning/simulation/history/test/

## 1. Purpose & Responsibility

This directory contains **unit and integration tests** for the simulation history buffer, validating temporal state accumulation, thread safety, buffer overflow behavior, and snapshot serialization. Tests ensure the `SimulationHistoryBuffer` correctly manages rolling windows of ego states, observations, and trajectories for planner consumption.

## 2. Key Abstractions

### Core Test Patterns

**SimulationHistoryBuffer Testing**
- Validates FIFO buffer behavior (append, overflow, wraparound)
- Tests thread safety under concurrent access
- Verifies buffer size constraints
- Ensures correct current_state() semantics

**Snapshot Serialization Testing**
- Tests SimulationHistorySample dataclass construction
- Validates snapshot immutability
- Tests serialization for logging/replay

**Thread Safety Testing**
- Concurrent append from multiple threads
- Race condition detection (read during write)
- Lock acquisition/release validation

### Key Test Files

```python
# Test history buffer core functionality
test_simulation_history_buffer.py
    - Test append and FIFO behavior
    - Test buffer overflow and wraparound
    - Test current_state() access
    - Test thread-safe concurrent operations
    - Test buffer size limits

# Test snapshot creation
test_simulation_history.py
    - Test SimulationHistorySample construction
    - Test snapshot field validation
    - Test snapshot serialization
```

## 3. Architecture & Design Patterns

### Test Organization Strategy

**Unit Tests**
- Isolated buffer operations (append, access, overflow)
- Fast execution (no dependencies)
- Parameterized tests for buffer sizes

**Integration Tests**
- End-to-end simulation loops with history
- Validates history + planner integration
- Uses mock planners/observations

**Fixture Patterns**
```python
@pytest.fixture
def empty_history_buffer() -> SimulationHistoryBuffer:
    """Fresh buffer for each test"""
    return SimulationHistoryBuffer(buffer_size=10)

@pytest.fixture
def populated_history_buffer() -> SimulationHistoryBuffer:
    """Pre-filled buffer for testing access patterns"""
    buffer = SimulationHistoryBuffer(buffer_size=10)
    for i in range(5):
        buffer.append(mock_ego_state(i), mock_observation(i), mock_trajectory(i))
    return buffer

@pytest.fixture
def mock_simulation_sample() -> SimulationHistorySample:
    """Standard snapshot for testing"""
    return SimulationHistorySample(
        iteration=SimulationIteration(index=0, time_us=0),
        ego_state=mock_ego_state(),
        trajectory=mock_trajectory(),
        observation=mock_observation(),
        traffic_light_status=[],
    )
```

## 4. Dependencies

### Internal (nuPlan)

**Direct Dependencies**:
- ✅ `nuplan.planning.simulation.history.simulation_history_buffer` - Class under test
- ✅ `nuplan.planning.simulation.history.simulation_history` - Snapshot dataclass
- ✅ `nuplan.common.actor_state.ego_state` - Mock ego states
- ✅ `nuplan.planning.simulation.observation.observation_type` - Mock observations

**Test Infrastructure**:
- `pytest` - Test framework
- `pytest-mock` - Mocking utilities
- `threading` - Thread safety tests
- `unittest.mock` - Standard library mocking

### External

- `numpy` - Array assertions (if needed)
- `pytest` - Test runner
- `pytest-xdist` - Parallel test execution

## 5. Dependents (Who Uses This Module?)

### Direct Consumers

**CI/CD Pipeline**:
- Pre-commit hooks run fast unit tests
- GitHub Actions run full test suite
- Coverage reports track history module coverage

**Developers**:
- Run tests locally before commits
- Use tests as documentation (buffer usage patterns)
- Debug history issues with failing tests

## 6. Critical Files (Prioritized)

### Priority 1: Core Tests

1. **`test_simulation_history_buffer.py`**
   - Tests for SimulationHistoryBuffer class
   - FIFO behavior, overflow, thread safety
   - **Key for**: Understanding buffer semantics

2. **`test_simulation_history.py`** (if exists)
   - Tests for SimulationHistorySample dataclass
   - Snapshot creation and serialization
   - **Key for**: Snapshot structure validation

### Priority 2: Utilities

3. **`conftest.py`** (if exists)
   - Shared test fixtures
   - Mock data generators
   - Test configuration

## 7. Common Usage Patterns

### Running History Tests

```bash
# Run all history tests
pytest nuplan/planning/simulation/history/test/

# Run specific test file
pytest nuplan/planning/simulation/history/test/test_simulation_history_buffer.py

# Run with coverage
pytest --cov=nuplan.planning.simulation.history \
       --cov-report=html \
       nuplan/planning/simulation/history/test/

# Run thread safety tests only
pytest -k "thread" nuplan/planning/simulation/history/test/
```

### Testing Buffer Append and Access

```python
def test_history_buffer_append_and_access():
    """Test basic append and access operations"""
    buffer = SimulationHistoryBuffer(buffer_size=10)
    
    # Append first state
    ego1 = create_ego_state(x=0.0, y=0.0)
    obs1 = create_observation()
    traj1 = create_trajectory()
    
    buffer.append(ego1, obs1, traj1)
    
    # Verify buffer length
    assert len(buffer) == 1
    assert len(buffer.ego_states) == 1
    assert len(buffer.observations) == 1
    assert len(buffer.sample_trajectory) == 1
    
    # Verify current_state()
    current_ego, current_obs = buffer.current_state()
    assert current_ego == ego1
    assert current_obs == obs1

def test_history_buffer_overflow():
    """Test buffer behavior when exceeding buffer_size"""
    buffer = SimulationHistoryBuffer(buffer_size=5)
    
    # Append 10 states (exceeds buffer_size=5)
    for i in range(10):
        buffer.append(
            create_ego_state(x=float(i)),
            create_observation(),
            create_trajectory()
        )
    
    # Buffer should only contain last 5 states
    assert len(buffer) == 5
    
    # Oldest state should be at index 5, newest at index 9
    assert buffer.ego_states[0].center.x == 5.0
    assert buffer.ego_states[-1].center.x == 9.0
```

### Testing Thread Safety

```python
import threading
import time

def test_concurrent_append():
    """Test thread safety of concurrent appends"""
    buffer = SimulationHistoryBuffer(buffer_size=100)
    num_threads = 10
    appends_per_thread = 10
    
    def append_worker(thread_id):
        for i in range(appends_per_thread):
            buffer.append(
                create_ego_state(x=float(thread_id * 100 + i)),
                create_observation(),
                create_trajectory()
            )
            time.sleep(0.001)  # Simulate work
    
    # Launch threads
    threads = [
        threading.Thread(target=append_worker, args=(i,))
        for i in range(num_threads)
    ]
    
    for thread in threads:
        thread.start()
    
    for thread in threads:
        thread.join()
    
    # Verify all appends succeeded
    assert len(buffer) == num_threads * appends_per_thread
    
    # Verify no corrupted data (all ego states valid)
    for ego_state in buffer.ego_states:
        assert ego_state.center.x >= 0.0

def test_concurrent_read_write():
    """Test reading while appending from another thread"""
    buffer = SimulationHistoryBuffer(buffer_size=50)
    
    # Pre-fill buffer
    for i in range(10):
        buffer.append(create_ego_state(), create_observation(), create_trajectory())
    
    stop_flag = threading.Event()
    errors = []
    
    def writer():
        try:
            for i in range(100):
                buffer.append(create_ego_state(), create_observation(), create_trajectory())
                time.sleep(0.001)
        except Exception as e:
            errors.append(e)
        finally:
            stop_flag.set()
    
    def reader():
        try:
            while not stop_flag.is_set():
                ego, obs = buffer.current_state()
                assert ego is not None
                assert obs is not None
                time.sleep(0.001)
        except Exception as e:
            errors.append(e)
    
    writer_thread = threading.Thread(target=writer)
    reader_thread = threading.Thread(target=reader)
    
    writer_thread.start()
    reader_thread.start()
    
    writer_thread.join()
    reader_thread.join()
    
    # No exceptions should have occurred
    assert len(errors) == 0
```

### Testing Snapshot Creation

```python
def test_simulation_history_sample_creation():
    """Test SimulationHistorySample dataclass construction"""
    iteration = SimulationIteration(index=5, time_us=500000)
    ego_state = create_ego_state(x=10.0, y=20.0)
    trajectory = create_trajectory()
    observation = create_observation()
    traffic_lights = []
    
    sample = SimulationHistorySample(
        iteration=iteration,
        ego_state=ego_state,
        trajectory=trajectory,
        observation=observation,
        traffic_light_status=traffic_lights,
    )
    
    # Verify fields
    assert sample.iteration.index == 5
    assert sample.ego_state.center.x == 10.0
    assert sample.trajectory == trajectory
    assert sample.observation == observation

def test_snapshot_immutability():
    """Test that snapshot fields cannot be modified after creation"""
    sample = create_simulation_sample()
    
    # Dataclass is frozen if @dataclass(frozen=True)
    # This test validates immutability
    with pytest.raises(AttributeError):
        sample.iteration = SimulationIteration(index=999, time_us=0)
```

## 8. Gotchas & Edge Cases

### 1. **Buffer Overflow Silent Data Loss**
- **Issue**: Oldest states discarded when buffer wraps
- **Symptom**: Expected state missing from history
- **Test**: `test_history_buffer_overflow()` validates FIFO behavior

### 2. **Empty Buffer Access**
- **Issue**: Accessing `buffer.current_state()` on empty buffer
- **Symptom**: IndexError or None values
- **Test**: `test_empty_buffer_access()` should raise appropriate error

### 3. **Thread Safety Lock Deadlock**
- **Issue**: Lock not released on exception in append()
- **Symptom**: Simulation hangs on subsequent append
- **Test**: `test_exception_during_append()` validates lock release

### 4. **Buffer Size Zero or Negative**
- **Issue**: Invalid buffer_size parameter
- **Symptom**: Immediate overflow or crashes
- **Test**: `test_invalid_buffer_size()` validates parameter validation

### 5. **Mismatched List Lengths**
- **Issue**: ego_states and observations have different lengths
- **Symptom**: Misaligned history access
- **Test**: `test_list_length_consistency()` validates append integrity

### 6. **current_state() Race Condition**
- **Issue**: Lists mutate between ego_states[-1] and observations[-1]
- **Symptom**: Mismatched timesteps in returned tuple
- **Test**: `test_concurrent_current_state()` validates atomicity

### 7. **Memory Leak in Long Tests**
- **Issue**: Buffer retains references to large objects
- **Symptom**: Test suite OOM on long-running tests
- **Test**: Use `buffer_size=10` in tests, not 200

### 8. **Snapshot Serialization Failures**
- **Issue**: SimulationHistorySample contains non-serializable objects
- **Symptom**: pickle or JSON serialization fails
- **Test**: `test_snapshot_serialization()` validates JSON/pickle compatibility

## 9. Performance Considerations

**Test Suite Performance**:
- Unit tests: ~0.5-2 seconds total
- Thread safety tests: ~2-5 seconds (includes sleep for race conditions)
- Full suite: ~5-10 seconds

**Optimization Strategies**:
- Use small buffer_size (10) in tests instead of production size (200)
- Mock heavy objects (Trajectory, Observation) with minimal data
- Use `pytest-xdist` for parallel execution

## 10. Related Documentation

### Cross-References
- ✅ `nuplan/planning/simulation/history/CLAUDE.md` - History buffer architecture
- ✅ `nuplan/common/actor_state/test/CLAUDE.md` - State representation test patterns
- ✅ `nuplan/planning/simulation/observation/test/CLAUDE.md` - Observation test patterns
- `nuplan/planning/simulation/test/CLAUDE.md` - Simulation testing (Phase 2C)

### External Resources
- **pytest threading**: https://docs.pytest.org/en/stable/how-to/concurrency.html
- **Python threading**: https://docs.python.org/3/library/threading.html

## 11. AIDEV Notes

**Testing Philosophy**:
- Focus on thread safety - most bugs come from concurrent access
- Buffer overflow behavior is critical (FIFO semantics)
- Validate list consistency (ego_states, observations, trajectories always same length)

**Common Test Failures**:
- Thread safety tests are flaky without proper sleep/synchronization
- Use `time.sleep()` strategically to expose race conditions
- Increase iterations if race conditions don't manifest

**Debugging Tips**:
- Run thread safety tests with `--tb=long` for full stack traces
- Use `pytest-repeat` to run flaky tests multiple times: `pytest --count=100`
- Add logging in buffer methods to trace concurrent access patterns

**AIDEV-TODO**: Add property-based tests (Hypothesis) for buffer invariants
**AIDEV-NOTE**: Test suite should run in <10 seconds to maintain developer productivity
