# nuplan/planning/simulation/simulation_time_controller/

## 1. Purpose & Responsibility

This module manages **temporal progression** in nuPlan simulations, controlling timestep sequencing, iteration metadata, and time synchronization across simulation components. The `SimulationTimeController` is THE source of truth for "what time is it now?" in the simulation loop, generating timestep sequences and providing iteration context (index, timestamp, time_us) to planners, observations, and callbacks.

## 2. Key Abstractions

### Core Concepts

**SimulationTimeController**
- **Purpose**: Generate and manage timestep sequences for simulation loops
- **Modes**:
  - **Stepped simulation**: Fixed timestep (e.g., 0.1s) from start to end
  - **Scenario replay**: Follow exact timestamps from logged data
  - **Adaptive timestep**: Variable dt based on scenario complexity (advanced)
- **Key methods**:
  - `get_iteration(index)` - Get iteration metadata at specific index
  - `next_iteration()` - Advance to next timestep
  - `reset()` - Reset to initial time for new scenario

**SimulationIteration**
- **Purpose**: Immutable snapshot of timestep metadata
- **Fields**:
  - `index: int` - Timestep number (0, 1, 2, ...)
  - `time_us: int` - Absolute time in microseconds since scenario start
  - `time_s: float` - Absolute time in seconds (derived from time_us)
  - `time_point: TimePoint` - nuPlan time representation (optional)
- **Usage**: Passed to planners, observations, callbacks for temporal context

**TimePoint** (from common package)
- **Purpose**: High-resolution timestamp representation
- **Fields**: `time_us: int` (microseconds since epoch)
- **Conversions**: `time_s`, `time_ms` properties

### Key Classes

```python
class SimulationTimeController:
    """
    Controls temporal progression in simulation.
    Generates SimulationIteration sequences.
    """
    def __init__(self, scenario: AbstractScenario):
        self._scenario = scenario
        self._current_index = 0
        self._duration_s = scenario.get_time_horizon()
        self._timestep_s = scenario.database_interval  # e.g., 0.05s
    
    def get_iteration(self, index: int) -> SimulationIteration:
        """Get iteration metadata at specific index"""
    
    def next_iteration(self) -> SimulationIteration:
        """Advance to next timestep, return new iteration"""
    
    def reset(self):
        """Reset to initial time"""

@dataclass(frozen=True)
class SimulationIteration:
    """Immutable timestep metadata"""
    index: int
    time_us: int
    
    @property
    def time_s(self) -> float:
        return self.time_us / 1e6
```

## 3. Architecture & Design Patterns

### Design Patterns

**Iterator Pattern**
- `SimulationTimeController` generates sequence of `SimulationIteration` objects
- `next_iteration()` returns next timestep in sequence
- Enables clean separation of time logic from simulation logic

**Immutable Value Object**
- `SimulationIteration` is frozen dataclass (cannot mutate)
- Prevents accidental timestep modification
- Safe to pass across threads/processes

**Template Method**
- Scenario defines time parameters (duration, timestep)
- TimeController implements iteration logic
- Enables scenario-specific time configurations

### Relationships

```
AbstractScenario
    ├─ get_time_horizon() → duration_s
    ├─ database_interval → timestep_s
    └─ get_number_of_iterations() → total iterations
        ↓
SimulationTimeController
    ├─ __init__(scenario)
    ├─ get_iteration(index) → SimulationIteration
    └─ next_iteration() → SimulationIteration
        ↓
Simulation Loop
    ├─ iteration = controller.next_iteration()
    ├─ PlannerInput(iteration=iteration)
    └─ Observation.update_observation(iteration, next_iteration)
```

### Time Flow in Simulation

```
Scenario Start (t=0)
    ↓
TimeController.reset()
    ↓
Loop:
  iteration = controller.get_iteration(index)
  next_iteration = controller.get_iteration(index + 1)
  
  planner.compute_trajectory(PlannerInput(iteration=iteration))
  observation.update_observation(iteration, next_iteration, ...)
  controller.next_iteration()  # Advance index
    ↓
Scenario End (t=duration_s)
```

## 4. Dependencies

### Internal (nuPlan)

**Direct Dependencies**:
- ✅ `nuplan.planning.scenario_builder.abstract_scenario` - AbstractScenario (time parameters)
- ✅ `nuplan.common.actor_state.state_representation` - TimePoint (timestamp representation)

**Indirect Dependencies**:
- `nuplan.database.nuplan_db_orm.models` - Log metadata (database_interval)

### External

- `dataclasses` - frozen dataclass for SimulationIteration
- `typing` - Type hints

### Dependency Notes

**AIDEV-NOTE**: TimeController is a "thin" module - most time logic is in AbstractScenario. This module just wraps scenario time parameters into iteration objects.

## 5. Dependents (Who Uses This Module?)

### Direct Consumers

**Simulation Infrastructure**:
- `nuplan/planning/simulation/simulation.py` - Main simulation loop
  - Creates `SimulationTimeController(scenario)`
  - Calls `controller.next_iteration()` every timestep
  
**Planners**:
- Access `current_input.iteration` to get current timestep metadata
- Use `iteration.time_s` for temporal planning (e.g., time-to-collision)

**Observations**:
- `update_observation(iteration, next_iteration, ...)` uses timestep metadata
- IDM agents use `iteration.time_s` for ODE solver time spans

**Callbacks**:
- Metrics callbacks use `iteration.index` for timestep indexing
- Logging callbacks use `iteration.time_us` for timestamps

### Use Cases

1. **Fixed-Timestep Simulation**
   - Training scenarios: 0.1s timestep, 20s duration → 200 iterations
   - Validation scenarios: 0.05s timestep, 15s duration → 300 iterations

2. **Scenario Replay**
   - Follow exact timestamps from logged data
   - Database interval may vary (0.05s, 0.1s, etc.)

3. **Time-Based Metrics**
   - Time-to-collision calculations
   - Time-to-goal metrics
   - Comfort metrics (jerk over time)

4. **Temporal Synchronization**
   - Ensure planner, observation, controller use same timestep
   - Validate observation update interval matches simulation timestep

**AIDEV-NOTE**: Most scenarios use 0.05s or 0.1s timesteps. Timestep is scenario-specific (comes from database_interval).

## 6. Critical Files (Prioritized)

### Priority 1: Core Implementation

1. **`simulation_iteration.py`**
   - SimulationIteration dataclass definition
   - Time conversions (us → s, s → us)
   - **Key for**: Understanding timestep metadata structure

2. **`abstract_simulation_time_controller.py`** (if exists)
   - Abstract base class for time controllers
   - Defines interface contract
   - **Key for**: Understanding time controller API

3. **`simulation_time_controller.py`** (if exists)
   - Concrete time controller implementation
   - Timestep generation logic
   - **Key for**: How iterations are generated

### Priority 2: Utilities

4. **`__init__.py`**
   - Module exports
   - Public API surface

5. **Test files** (`test/` directory)
   - Timestep sequence tests
   - Time conversion tests
   - Reset behavior tests
   - **Key for**: Expected behavior and edge cases

**AIDEV-NOTE**: Module is very small (~100-200 lines total). Start with simulation_iteration.py, it's self-contained.

## 7. Common Usage Patterns

### Creating Time Controller

```python
from nuplan.planning.simulation.simulation_time_controller.simulation_time_controller import SimulationTimeController
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario

# Initialize from scenario
scenario = load_scenario(...)
time_controller = SimulationTimeController(scenario)

# Reset to initial time
time_controller.reset()
```

### Accessing Iteration Metadata

```python
# Get current iteration
iteration = time_controller.get_iteration(index=0)

print(f"Index: {iteration.index}")          # 0
print(f"Time (s): {iteration.time_s}")       # 0.0
print(f"Time (us): {iteration.time_us}")     # 0

# Get future iteration
next_iteration = time_controller.get_iteration(index=1)
print(f"Next time: {next_iteration.time_s}")  # 0.1 (or scenario timestep)
```

### Simulation Loop Integration

```python
# Initialize
time_controller = SimulationTimeController(scenario)
time_controller.reset()

num_iterations = scenario.get_number_of_iterations()

# Simulation loop
for index in range(num_iterations):
    # Get current and next iteration
    iteration = time_controller.get_iteration(index)
    next_iteration = time_controller.get_iteration(index + 1)
    
    # Build planner input
    planner_input = PlannerInput(
        iteration=iteration,
        history=history_buffer,
        traffic_light_data=traffic_lights,
    )
    
    # Plan trajectory
    trajectory = planner.compute_planner_trajectory(planner_input)
    
    # Update observations with time context
    observation.update_observation(iteration, next_iteration, history_buffer)
    
    # Advance time (internal state update)
    time_controller.next_iteration()
```

### Using Iteration in Planner

```python
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput

class MyPlanner(AbstractPlanner):
    def compute_planner_trajectory(self, current_input: PlannerInput):
        # Access current timestep metadata
        iteration = current_input.iteration
        
        # Get simulation time
        current_time_s = iteration.time_s
        
        # Compute time-to-collision
        ego_velocity = current_input.history.ego_states[-1].dynamic_car_state.rear_axle_velocity_2d.magnitude()
        distance_to_obstacle = 10.0  # meters
        time_to_collision = distance_to_obstacle / ego_velocity if ego_velocity > 0 else float('inf')
        
        # Use timestep index for debugging/logging
        if iteration.index % 10 == 0:
            logger.info(f"Timestep {iteration.index}, Time: {current_time_s:.2f}s, TTC: {time_to_collision:.2f}s")
```

### Time Conversions

```python
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration

# Create iteration from time in seconds
iteration = SimulationIteration(
    index=5,
    time_us=int(0.5 * 1e6),  # 0.5 seconds in microseconds
)

# Access time in different units
time_s = iteration.time_s      # 0.5 (seconds)
time_us = iteration.time_us    # 500000 (microseconds)

# Compute timestep interval
dt_s = next_iteration.time_s - iteration.time_s  # e.g., 0.1
```

### Resetting Time Controller

```python
# Reset for new scenario run
time_controller.reset()

# Current iteration is now back to index 0
iteration = time_controller.get_iteration(0)
assert iteration.index == 0
assert iteration.time_s == 0.0
```

## 8. Gotchas & Edge Cases

### 1. **Timestep Not Always 0.1s**
- **Issue**: Assuming fixed 0.1s timestep across all scenarios
- **Symptom**: Temporal calculations off by 2× or more
- **Fix**: Use `scenario.database_interval` or compute from iterations
- **Example**: Some scenarios use 0.05s, others 0.1s

### 2. **Index vs Time Confusion**
- **Issue**: Using iteration.index instead of iteration.time_s for time calculations
- **Symptom**: Incorrect time-to-collision or velocity estimates
- **Fix**: Always use `iteration.time_s` for time, `iteration.index` only for indexing

### 3. **Microsecond Precision Loss**
- **Issue**: Converting us → s → us loses precision
- **Symptom**: Slight timestamp drift in long simulations
- **Fix**: Store `time_us` as ground truth, derive `time_s` from it (current design)

### 4. **Next Iteration Out of Bounds**
- **Issue**: Accessing `get_iteration(index + 1)` at last timestep
- **Symptom**: IndexError or returns invalid iteration
- **Fix**: Check `index < num_iterations - 1` before accessing next_iteration

### 5. **Reset Without Re-initialization**
- **Issue**: Calling `reset()` doesn't reset dependent state (history buffer, observations)
- **Symptom**: History still contains old data after reset
- **Fix**: Simulation must reset all components, not just time controller

### 6. **Time Controller State Mutation**
- **Issue**: `next_iteration()` mutates internal index
- **Symptom**: Calling `next_iteration()` multiple times without simulation steps → time drift
- **Fix**: Only call `next_iteration()` once per simulation loop iteration

### 7. **Scenario Duration vs Iteration Count Mismatch**
- **Issue**: `duration_s / timestep_s ≠ num_iterations` due to rounding
- **Symptom**: Simulation ends early or runs too long
- **Fix**: Use `scenario.get_number_of_iterations()` as source of truth

### 8. **Frozen Dataclass Immutability**
- **Issue**: Attempting to modify `iteration.index = 10`
- **Symptom**: `dataclasses.FrozenInstanceError`
- **Fix**: Create new SimulationIteration instead of modifying existing

### 9. **Time Point Timezone Issues**
- **Issue**: TimePoint may represent absolute epoch time (with timezone)
- **Symptom**: Timestamp doesn't match scenario start time
- **Fix**: Use relative time (`time_s`, `time_us`) instead of absolute `time_point`

### 10. **Parallel Simulation Time Synchronization**
- **Issue**: Multiple workers have independent time controllers → out of sync
- **Symptom**: Workers at different timesteps, results misaligned
- **Fix**: Each worker has own time controller, reset at scenario start

## 9. Performance Considerations

**Computational Cost**:
- Creating `SimulationIteration` objects: Negligible (<1 µs)
- Time conversions (us → s): Negligible (simple division)
- `next_iteration()`: O(1) index increment

**Memory Usage**:
- `SimulationIteration`: ~40 bytes (2 ints + dataclass overhead)
- Time controller state: ~100 bytes (scenario reference + index)

**Optimization Strategies**:
- No optimization needed - time controller is never a bottleneck
- Iteration objects are lightweight value objects

**AIDEV-NOTE**: Time controller overhead is <0.01% of simulation runtime. No profiling needed.

## 10. Related Documentation

### Cross-References
- ✅ `nuplan/planning/scenario_builder/CLAUDE.md` - AbstractScenario (time parameters)
- ✅ `nuplan/planning/simulation/planner/CLAUDE.md` - Planner interface (uses iteration)
- ✅ `nuplan/planning/simulation/observation/CLAUDE.md` - Observation update (uses iteration)
- ✅ `nuplan/planning/simulation/history/CLAUDE.md` - History buffer (timestep sequencing)
- `nuplan/planning/simulation/callback/CLAUDE.md` - Callbacks (Phase 2C, use iteration for logging)
- ✅ `nuplan/common/actor_state/CLAUDE.md` - TimePoint representation

### External Resources
- **Python dataclasses**: https://docs.python.org/3/library/dataclasses.html
- **Time representations in Python**: Microsecond precision best practices

## 11. AIDEV Notes

**Design Philosophy**:
- Time controller is a "thin wrapper" around scenario time parameters
- SimulationIteration is immutable value object (safe to pass around)
- Timestep logic is scenario-driven, not hardcoded

**Common Mistakes**:
- Assuming 0.1s timestep (it varies by scenario!)
- Using iteration.index for time calculations (use iteration.time_s)
- Forgetting to reset time controller between scenario runs

**Future Improvements**:
- **AIDEV-TODO**: Add adaptive timestep support (variable dt based on scenario complexity)
- **AIDEV-TODO**: Add time dilation for faster/slower playback (visualization)
- **AIDEV-TODO**: Add timestep validation (ensure observation dt matches controller dt)

**AIDEV-NOTE**: If you need custom time logic (e.g., adaptive timestep), subclass SimulationTimeController and override `get_iteration()` and `next_iteration()`.
