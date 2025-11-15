# nuplan/planning/simulation/history/

## 1. Purpose & Responsibility

This module manages **temporal state history** during simulation, maintaining a rolling window of ego states, observations, and samples for planner consumption. The `SimulationHistoryBuffer` is THE central data structure that accumulates simulation state over time, enabling planners to access historical context (past ego trajectories, previous observations, traffic light history) when making decisions. Every simulation timestep appends new data to this buffer, which is then passed to the planner via `PlannerInput.history`.

## 2. Key Abstractions

### Core Concepts

**SimulationHistoryBuffer**
- **Purpose**: Thread-safe FIFO buffer storing simulation state sequences
- **Key fields**:
  - `ego_states: List[EgoState]` - Ego trajectory history
  - `observations: List[Observation]` - Sensor/perception history
  - `sample_trajectory: List[InterpolatedTrajectory]` - Sampled reference trajectories
  - `_buffer_size: int` - Maximum buffer length (default: 200 timesteps = 20s @ 0.1s)
- **Thread safety**: Uses threading.Lock to prevent race conditions during append/access
- **Circular behavior**: When full, oldest elements are removed (FIFO)

**SimulationHistorySample**
- **Purpose**: Single-timestep snapshot of simulation state
- **Fields**:
  - `iteration: SimulationIteration` - Timestep metadata
  - `ego_state: EgoState` - Ego pose, velocity, acceleration
  - `trajectory: InterpolatedTrajectory` - Planned/reference trajectory at this timestep
  - `observation: Observation` - Sensor data (DetectionsTracks, Sensors, etc.)
  - `traffic_light_status: List[TrafficLightStatusData]` - Traffic light states

**PlannerInput.history**
- The `SimulationHistoryBuffer` is accessed by planners via `PlannerInput.history`
- Planners use this to implement temporal reasoning (e.g., track agent velocities across frames)

### Key Classes

```python
class SimulationHistoryBuffer:
    """
    Rolling window buffer for simulation state history.
    Thread-safe append and access operations.
    """
    def __init__(self, buffer_size: int = 200):
        self._buffer_size = buffer_size
        self.ego_states: List[EgoState] = []
        self.observations: List[Observation] = []
        self.sample_trajectory: List[InterpolatedTrajectory] = []
        self._lock = threading.Lock()
    
    def append(self, ego_state: EgoState, observation: Observation, trajectory: InterpolatedTrajectory):
        """Add new timestep data, remove oldest if buffer full"""
    
    def current_state(self) -> Tuple[EgoState, Observation]:
        """Get most recent ego state and observation"""
    
    def __len__(self) -> int:
        """Number of timesteps in buffer"""

class SimulationHistorySample:
    """Single-timestep snapshot (dataclass)"""
    iteration: SimulationIteration
    ego_state: EgoState
    trajectory: InterpolatedTrajectory
    observation: Observation
    traffic_light_status: List[TrafficLightStatusData]
```

## 3. Architecture & Design Patterns

### Design Patterns

**Circular Buffer Pattern**
- Fixed-size FIFO buffer with automatic eviction
- Prevents unbounded memory growth in long simulations
- Trade-off: Older history is lost (configurable buffer_size)

**Thread-Safe Accumulator**
- `threading.Lock` protects concurrent access
- Critical for multi-threaded simulation (e.g., Ray-based workers)
- Lock acquired during `append()` and list access

**Snapshot Pattern**
- `SimulationHistorySample` is an immutable record of single timestep
- Enables serialization for logging, replay, debugging
- Used in scenario exports and simulation logs

### Relationships

```
Simulation Loop
    ├─ SimulationHistoryBuffer (accumulator)
    │   ├─ append(ego_state, observation, trajectory)  # Every timestep
    │   └─ current_state() → (latest ego, latest obs)
    │
    ├─ PlannerInput
    │   └─ history: SimulationHistoryBuffer  # Passed to planner
    │
    └─ AbstractPlanner.compute_planner_trajectory(current_input)
        └─ current_input.history.ego_states[-1]  # Access latest state
        └─ current_input.history.observations  # Access observation history
```

### Buffer Update Flow

```
Simulation.propagate():
  1. Controller.update_state(ego_trajectory) → new_ego_state
  2. Observation.update_observation(...) → new_observation
  3. history_buffer.append(new_ego_state, new_observation, ego_trajectory)
  4. PlannerInput(history=history_buffer) → planner
  5. Planner accesses history for temporal context
```

## 4. Dependencies

### Internal (nuPlan)

**Direct Dependencies**:
- ✅ `nuplan.common.actor_state.ego_state` - EgoState representation
- ✅ `nuplan.planning.simulation.observation.observation_type` - Observation types
- ✅ `nuplan.planning.simulation.trajectory.abstract_trajectory` - InterpolatedTrajectory
- ✅ `nuplan.planning.simulation.simulation_time_controller.simulation_iteration` - SimulationIteration
- `nuplan.planning.simulation.trajectory.interpolated_trajectory` - Concrete trajectory type

**Indirect Dependencies**:
- `nuplan.common.actor_state.state_representation` - StateSE2 (via EgoState)
- `nuplan.planning.simulation.observation.abstract_observation` - Observation interface

### External

- `threading` - Lock for thread safety
- `typing` - List, Tuple type hints
- `dataclasses` - SimulationHistorySample definition

### Dependency Notes

**AIDEV-NOTE**: History buffer is a "dumb" container - no business logic, just storage + thread safety. Complex temporal reasoning is in planners, not here.

## 5. Dependents (Who Uses This Module?)

### Direct Consumers

**Simulation Infrastructure**:
- `nuplan/planning/simulation/simulation.py` - Main simulation loop
  - Calls `history.append()` every timestep
  - Builds `PlannerInput(history=history_buffer)`
  
**Planners**:
- All planners access `current_input.history` to get temporal context
- Common usage:
  - `history.ego_states[-1]` - Latest ego state
  - `history.ego_states[-10:]` - Last 10 timesteps (1 second @ 0.1s)
  - `history.observations[-1]` - Latest observation

**Callbacks**:
- `nuplan/planning/simulation/callback/` - Logging, metrics callbacks
  - Access history for metrics computation (e.g., trajectory smoothness over time)

### Use Cases

1. **Temporal Filtering**
   - Track agent velocities over multiple frames
   - Smooth noisy observations
   - Detect stopped vs moving objects

2. **Trajectory History**
   - Compare planned vs executed trajectories
   - Compute tracking error over time
   - Visualize ego path in nuBoard

3. **Observation Consistency**
   - Track object IDs across frames (data association)
   - Validate observation quality (e.g., missing detections)

4. **Metrics Computation**
   - Jerk calculation (requires acceleration history)
   - Progress metrics (distance traveled over time)
   - Comfort metrics (requires velocity/acceleration sequences)

**AIDEV-NOTE**: Buffer size (200 timesteps = 20s) is tuned for planner context window. Longer planners may need larger buffers.

## 6. Critical Files (Prioritized)

### Priority 1: Core Implementation

1. **`simulation_history_buffer.py`**
   - SimulationHistoryBuffer class
   - Thread-safe append/access logic
   - Buffer size management
   - **Key for**: Understanding state accumulation

2. **`simulation_history.py`**
   - SimulationHistorySample dataclass
   - Snapshot serialization
   - **Key for**: Single-timestep state structure

### Priority 2: Utilities

3. **`__init__.py`**
   - Module exports
   - Public API surface

4. **Test files** (`test/` directory)
   - Thread safety tests
   - Buffer overflow tests
   - Snapshot serialization tests
   - **Key for**: Expected behavior and edge cases

**AIDEV-NOTE**: Module is small (~200-300 lines total) - start with simulation_history_buffer.py, it's self-contained.

## 7. Common Usage Patterns

### Accessing History in Planner

```python
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput

class MyPlanner(AbstractPlanner):
    def compute_planner_trajectory(self, current_input: PlannerInput):
        # Get latest ego state
        current_ego = current_input.history.ego_states[-1]
        
        # Get ego trajectory over last second (10 timesteps @ 0.1s)
        ego_history = current_input.history.ego_states[-10:]
        
        # Get latest observation
        latest_obs = current_input.history.observations[-1]
        
        # Track agent across frames
        if len(current_input.history.observations) >= 2:
            prev_obs = current_input.history.observations[-2]
            curr_obs = current_input.history.observations[-1]
            
            # Match tracked objects by token
            for curr_agent in curr_obs.tracked_objects:
                for prev_agent in prev_obs.tracked_objects:
                    if curr_agent.track_token == prev_agent.track_token:
                        # Compute velocity from position change
                        dt = 0.1  # Simulation timestep
                        velocity = (curr_agent.center - prev_agent.center) / dt
```

### Simulation Loop Integration

```python
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer

# Initialize history buffer
history = SimulationHistoryBuffer(buffer_size=200)  # 20 seconds @ 0.1s

# Simulation loop
for iteration in simulation_iterations:
    # Get current ego state (from controller)
    ego_state = controller.get_state()
    
    # Get current observation (from observation model)
    observation = observation_model.get_observation()
    
    # Append to history
    history.append(
        ego_state=ego_state,
        observation=observation,
        trajectory=planned_trajectory  # From previous planner call
    )
    
    # Build planner input with history
    planner_input = PlannerInput(
        iteration=iteration,
        history=history,
        traffic_light_data=traffic_lights,
    )
    
    # Planner uses history for temporal reasoning
    planned_trajectory = planner.compute_planner_trajectory(planner_input)
```

### Thread-Safe Access

```python
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer

# Multi-threaded simulation (e.g., Ray workers)
history = SimulationHistoryBuffer(buffer_size=200)

def simulation_worker():
    # Thread 1: Append new data
    history.append(ego_state, observation, trajectory)

def metrics_callback():
    # Thread 2: Read current state
    current_ego, current_obs = history.current_state()
    
    # Thread-safe access (Lock acquired internally)
    ego_sequence = history.ego_states[-10:]
```

### Creating Snapshots for Logging

```python
from nuplan.planning.simulation.history.simulation_history import SimulationHistorySample

# Create snapshot at specific timestep
snapshot = SimulationHistorySample(
    iteration=simulation_iteration,
    ego_state=ego_state,
    trajectory=planned_trajectory,
    observation=observation,
    traffic_light_status=traffic_light_data,
)

# Serialize for logging
snapshot_dict = {
    'iteration': snapshot.iteration.index,
    'time_s': snapshot.iteration.time_s,
    'ego_x': snapshot.ego_state.center.x,
    'ego_y': snapshot.ego_state.center.y,
    'ego_heading': snapshot.ego_state.center.heading,
    'ego_velocity': snapshot.ego_state.dynamic_car_state.rear_axle_velocity_2d.magnitude(),
}
```

### Computing Metrics from History

```python
def compute_jerk_metric(history: SimulationHistoryBuffer) -> float:
    """Compute average jerk from ego state history"""
    if len(history.ego_states) < 3:
        return 0.0
    
    jerks = []
    dt = 0.1  # Simulation timestep
    
    for i in range(2, len(history.ego_states)):
        # Get acceleration at t-2, t-1, t
        a_prev = history.ego_states[i-2].dynamic_car_state.rear_axle_acceleration_2d
        a_curr = history.ego_states[i-1].dynamic_car_state.rear_axle_acceleration_2d
        a_next = history.ego_states[i].dynamic_car_state.rear_axle_acceleration_2d
        
        # Compute jerk (da/dt)
        jerk = (a_next - a_curr).magnitude() / dt
        jerks.append(jerk)
    
    return sum(jerks) / len(jerks) if jerks else 0.0
```

## 8. Gotchas & Edge Cases

### 1. **Buffer Size vs Planner Context Window**
- **Issue**: Planner needs 5s history, buffer only stores 2s (20 timesteps)
- **Symptom**: IndexError when accessing `history.ego_states[-50]`
- **Fix**: Increase buffer_size in initialization: `SimulationHistoryBuffer(buffer_size=500)`
- **Trade-off**: Memory usage scales linearly with buffer size

### 2. **Empty History at Simulation Start**
- **Issue**: First planner call has `len(history.ego_states) == 1` (only initial state)
- **Symptom**: Planner crashes accessing `history.ego_states[-10:]` (expects 10 elements)
- **Fix**: Check history length before indexing: `if len(history.ego_states) >= 10:`

### 3. **Thread Safety Lock Contention**
- **Issue**: Multiple threads accessing history simultaneously → lock contention
- **Symptom**: Simulation slows down in multi-threaded mode
- **Fix**: Minimize locked critical sections, batch reads
- **AIDEV-NOTE**: Lock is coarse-grained (entire append/access), could use fine-grained locks

### 4. **Stale References After Buffer Wrap**
- **Issue**: Planner stores reference to `history.ego_states`, buffer wraps, list mutates
- **Symptom**: Unexpected state changes in planner's cached history
- **Fix**: Copy history lists if caching: `ego_history = list(history.ego_states)`

### 5. **Observation Type Changes Mid-Simulation**
- **Issue**: Observation switches from DetectionsTracks → Sensors between timesteps
- **Symptom**: Planner crashes when casting observation
- **Fix**: Validate observation type consistency in simulation setup
- **AIDEV-NOTE**: Current design doesn't prevent observation type changes

### 6. **Memory Leak from Trajectory References**
- **Issue**: `sample_trajectory` stores full InterpolatedTrajectory objects → large memory
- **Symptom**: Memory grows unbounded even with fixed buffer size
- **Fix**: Store only trajectory metadata (states at sample points) instead of full objects
- **AIDEV-TODO**: Consider compressing trajectory storage

### 7. **No Timestep Validation**
- **Issue**: Appending states out of chronological order
- **Symptom**: History has non-monotonic timestamps
- **Fix**: Add assertion in `append()`: `assert new_iteration.time_s >= prev_iteration.time_s`
- **AIDEV-NOTE**: Current implementation trusts caller to append in order

### 8. **Race Condition in current_state()**
- **Issue**: `current_state()` returns tuple, but list can mutate between access
- **Symptom**: `ego, obs = history.current_state()` → ego and obs from different timesteps
- **Fix**: Lock should cover entire tuple construction (currently does)

### 9. **Buffer Overflow Silent Data Loss**
- **Issue**: When buffer full, oldest data silently discarded
- **Symptom**: Planner expects data at t=0, but buffer only has t=100..300
- **Fix**: Log warning when buffer wraps: `logger.warning("History buffer full, discarding old data")`
- **AIDEV-TODO**: Add option for fixed-size with append rejection instead of wrap

### 10. **Observation History Length Mismatch**
- **Issue**: `len(ego_states) != len(observations)` if append calls inconsistent
- **Symptom**: Misaligned history (ego at t=5, obs at t=4)
- **Fix**: Assert equal lengths in `append()`: `assert len(self.ego_states) == len(self.observations)`

## 9. Performance Considerations

**Memory Usage**:
- EgoState: ~500 bytes per timestep
- Observation (DetectionsTracks): ~10-50 KB per timestep (depends on agent count)
- Trajectory: ~5-20 KB per timestep (depends on horizon/resolution)
- **Total per timestep**: ~20-70 KB
- **Buffer (200 timesteps)**: ~4-14 MB

**Scaling**:
- Buffer size linear with memory: 200 timesteps → ~10 MB, 2000 timesteps → ~100 MB
- Thread lock overhead: Negligible (<1% runtime) unless high contention

**Optimization Strategies**:
- Reduce buffer size if memory constrained
- Store only critical trajectory points (not full interpolation)
- Compress observations (e.g., only store deltas for static objects)
- Use memory-mapped storage for very long simulations

**AIDEV-NOTE**: For 10-minute simulation (6000 timesteps @ 0.1s), buffer stores only last 20s. This is intentional - longer history should be persisted to disk by callbacks.

## 10. Related Documentation

### Cross-References
- ✅ `nuplan/planning/simulation/planner/CLAUDE.md` - Planner interface (consumes history)
- ✅ `nuplan/common/actor_state/CLAUDE.md` - EgoState representation
- ✅ `nuplan/planning/simulation/observation/CLAUDE.md` - Observation types
- ✅ `nuplan/planning/simulation/trajectory/CLAUDE.md` - Trajectory types
- ✅ `nuplan/planning/simulation/simulation_time_controller/CLAUDE.md` - SimulationIteration
- `nuplan/planning/simulation/callback/CLAUDE.md` - Callbacks using history (Phase 2C)
- `nuplan/planning/metrics/CLAUDE.md` - Metrics computed from history (Phase 3)

### External Resources
- **Threading in Python**: https://docs.python.org/3/library/threading.html
- **Circular Buffers**: Design pattern for fixed-size FIFO storage

## 11. AIDEV Notes

**Design Philosophy**:
- History buffer is a "passive" component - no complex logic, just storage
- Thread safety is essential for multi-worker simulations (Ray)
- Buffer size is a trade-off between memory and temporal context

**Common Mistakes**:
- Assuming history is always full (check length before indexing!)
- Not copying lists before caching (stale references after buffer wrap)
- Forgetting thread safety when extending this module

**Future Improvements**:
- **AIDEV-TODO**: Add buffer overflow warning/logging
- **AIDEV-TODO**: Add timestep validation (monotonic timestamps)
- **AIDEV-TODO**: Compress trajectory storage (only sample points)
- **AIDEV-TODO**: Fine-grained locking (separate locks for ego_states, observations, trajectories)

**AIDEV-NOTE**: If you need history longer than 20s, consider:
1. Increase buffer_size (memory cost)
2. Persist to disk via callbacks (I/O cost)
3. Implement hierarchical buffer (recent high-res + older low-res)
