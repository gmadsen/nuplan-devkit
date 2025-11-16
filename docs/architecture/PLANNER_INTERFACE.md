# Planner Interface & Lifecycle Architecture

## Overview

The **planner interface** is the core contract between nuPlan's simulation framework and autonomous vehicle planning algorithms. All planners (rule-based, ML-based, hybrid) implement `AbstractPlanner`, which defines initialization, trajectory computation, and reporting requirements. This document details the planner API, lifecycle, threading model, and timing constraints.

## Purpose

Planners are **decision-making algorithms** that consume simulation state and produce trajectory plans. The planner interface enables:
1. **Diverse planning approaches** - Simple, ML, IDM, oracle, custom implementations
2. **Simulation integration** - Pluggable into evaluation framework via polymorphism
3. **Performance measurement** - Built-in timing instrumentation
4. **Observation flexibility** - Planners declare expected input types

## Architecture Overview

### Planner Lifecycle

```
┌────────────────────────────────────────────────────────────────┐
│ Script Layer (run_simulation.py)                              │
│  ├─ Parse Hydra config                                       │
│  ├─ Instantiate planner via Hydra factory                    │
│  └─ Pass to SimulationRunner                                 │
└─────────────────┬────────────────────────────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────────────────────────┐
│ SimulationRunner (runner/simulations_runner.py)              │
│  ├─ Create Simulation with SimulationSetup                   │
│  └─ Call sim.initialize()                                    │
└─────────────────┬────────────────────────────────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────────────────────────┐
│ PHASE 1: INITIALIZATION (once per scenario)                   │
│                                                                │
│  planner.initialize(PlannerInitialization)                   │
│  ├─ Cache route_roadblock_ids (path planning context)        │
│  ├─ Store map_api reference (for map queries)                │
│  └─ Store mission_goal (navigation context)                  │
│                                                                │
│  [Return from initialize()]                                   │
└─────────────────┬────────────────────────────────────────────┘
                  │
                  ▼ (repeat for each 0.1s simulation step)
┌────────────────────────────────────────────────────────────────┐
│ PHASE 2: TRAJECTORY COMPUTATION (called ~200x per scenario)   │
│                                                                │
│  FOR EACH TIMESTEP:                                           │
│    ├─ SimulationRunner calls:                                │
│    │  planner.compute_trajectory(PlannerInput)               │
│    │                                                          │
│    └─ Planner processes:                                      │
│       ├─ Current ego state (history[0])                      │
│       ├─ Past observations (history)                         │
│       ├─ Traffic lights (traffic_light_data)                 │
│       ├─ Map queries                                        │
│       └─ Return InterpolatedTrajectory                       │
└────────────────────────────────────────────────────────────────┘
                  │
                  ▼ (after all scenarios complete)
┌────────────────────────────────────────────────────────────────┐
│ PHASE 3: REPORTING (once per planner)                         │
│                                                                │
│  planner.generate_planner_report() → PlannerReport           │
│  ├─ compute_trajectory_runtimes: List[float]                │
│  └─ Returns timing statistics for all calls                  │
└────────────────────────────────────────────────────────────────┘
```

## Key Abstractions

### AbstractPlanner - The Core Interface

```python
class AbstractPlanner(abc.ABC):
    """Base class for all planners."""
    
    # Class attribute: Mark oracle planners (read scenario data)
    requires_scenario: bool = False
    
    def __new__(cls, *args, **kwargs):
        """Auto-initialize performance tracking."""
        instance = super().__new__(cls)
        instance._compute_trajectory_runtimes = []
        return instance

    @abstractmethod
    def name(self) -> str:
        """Return planner identifier (e.g., 'simple_planner')."""
        pass

    @abstractmethod
    def initialize(self, initialization: PlannerInitialization) -> None:
        """
        One-time setup called before simulation starts.
        
        Args:
            initialization: Contains route, map API, and mission goal
        
        Usage:
            self._map_api = initialization.map_api
            self._route = initialization.route_roadblock_ids
            self._goal = initialization.mission_goal
        """
        pass

    @abstractmethod
    def observation_type(self) -> Type[Observation]:
        """
        Declare expected observation format.
        
        Returns:
            DetectionsTracks, Sensors, or custom Observation subclass
        
        Note:
            Simulation validates this matches setup.observations.observation_type()
            Mismatch raises ValueError before planning starts
        """
        pass

    @abstractmethod
    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Core planning logic - called every simulation timestep.
        
        Args:
            current_input: Contains iteration, history buffer, traffic lights
        
        Returns:
            AbstractTrajectory (typically InterpolatedTrajectory)
        
        Timing constraints:
            - Must complete in ~0.1 seconds (simulation timestep)
            - Automatically timed by compute_trajectory() wrapper
            - Exceptions are caught and timing preserved
        """
        pass

    def compute_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Framework wrapper around compute_planner_trajectory().
        
        Responsibilities:
            - Time the planner execution
            - Record runtime in _compute_trajectory_runtimes
            - Catch exceptions and preserve timing
            - Rethrow exceptions to caller
        
        Note:
            User implements compute_planner_trajectory(),
            not this method!
        """
        start_time = time.time()
        try:
            trajectory = self.compute_planner_trajectory(current_input)
        finally:
            end_time = time.time()
            runtime = (end_time - start_time) * 1000  # Convert to ms
            self._compute_trajectory_runtimes.append(runtime)
        return trajectory

    def generate_planner_report(self) -> PlannerReport:
        """
        Extract performance statistics.
        
        Returns:
            PlannerReport with compute_trajectory_runtimes
        
        Usage:
            After simulation ends, extract timing data:
            report = planner.generate_planner_report()
            runtimes = report.compute_trajectory_runtimes
            mean_time = statistics.mean(runtimes)
            max_time = max(runtimes)
        """
        return PlannerReport(
            compute_trajectory_runtimes=self._compute_trajectory_runtimes
        )
```

### PlannerInitialization - One-time Setup Data

```python
@dataclass(frozen=True)
class PlannerInitialization:
    """Immutable initialization data passed to planner.initialize()."""
    
    route_roadblock_ids: List[str]      # Roadblock IDs comprising route to goal
    mission_goal: StateSE2              # Final destination pose
    map_api: AbstractMap                # Interface for map queries
```

**Key semantics**:
- `route_roadblock_ids`: GPS-like route guidance (sequence of road segments)
- `mission_goal`: Ultimate destination (may be beyond scenario end)
- `map_api`: Same map used for planner queries throughout simulation

**Immutability**: Frozen dataclass prevents accidental modification.

### PlannerInput - Per-Timestep Decision Context

```python
@dataclass(frozen=True)
class PlannerInput:
    """Immutable input snapshot for each planning decision."""
    
    iteration: SimulationIteration                      # Current timestep metadata
    history: SimulationHistoryBuffer                    # Rolling buffer of past states
    traffic_light_data: Optional[List[TrafficLightStatusData]] = None  # Signals
```

**Key fields**:
- `iteration.index`: Timestep number (0, 1, 2, ..., ~200)
- `iteration.time_s`: Simulation time in seconds
- `history.ego_states`: List of past ego poses (newest = index -1)
- `history.observations`: List of past sensor data (same length as ego_states)
- `history.current_state`: Shortcut to (ego_states[-1], observations[-1])

**Immutability**: Frozen dataclass - read-only input.

### AbstractTrajectory - Planner Output Format

```python
class AbstractTrajectory:
    """Base class for trajectory representations."""
    
    @property
    def start_time(self) -> TimePoint:
        """Trajectory start timestamp."""
        pass
    
    @property
    def end_time(self) -> TimePoint:
        """Trajectory end timestamp."""
        pass
    
    @property
    def duration(self) -> float:
        """Trajectory duration in seconds."""
        pass
    
    def get_state_at_time(self, time_point: TimePoint) -> InterpolatableState:
        """Query interpolated state at specific time."""
        pass
    
    def get_sampled_trajectory(self) -> List[InterpolatableState]:
        """Return original discrete waypoints."""
        pass
```

**Most common implementation**: `InterpolatedTrajectory`

```python
# Create trajectory from sequence of EgoState objects
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

future_states = [ego_t0, ego_t0_1, ego_t0_2, ..., ego_t8]  # ~80 states for 8s horizon
trajectory = InterpolatedTrajectory(future_states)

# Query at arbitrary time
state_at_1_5s = trajectory.get_state_at_time(time_point_1_5s)
```

## Design Patterns

### 1. Two-Phase Initialization

**Phase 1: Planner Construction** (`__init__`)
- Parameters: Learning rate, network size, algorithm hyperparameters
- Example:
  ```python
  planner = SimplePlanner(
      horizon_seconds=8.0,
      sampling_time=0.1,
      acceleration=1.5,
      max_velocity=10.0
  )
  ```

**Phase 2: Scenario Binding** (`initialize()`)
- Receives: Route, map, goal
- Enables: Scenario-specific precomputation, model loading
- Example:
  ```python
  planner.initialize(initialization)
  # Now planner has access to specific scenario's map and route
  ```

**Benefit**: Planner instances are reusable across scenarios without reconstruction.

### 2. Template Method Pattern

The `compute_trajectory()` wrapper provides consistent behavior across all planners:

```python
def compute_trajectory(self, current_input):
    # Framework responsibility: timing measurement
    start_time = time.time()
    try:
        # User responsibility: planning logic
        trajectory = self.compute_planner_trajectory(current_input)
    finally:
        # Framework responsibility: record timing
        self._compute_trajectory_runtimes.append(time.time() - start_time)
    return trajectory
```

**Benefit**: All planners get automatic performance profiling without duplicating timing code.

### 3. Type Declaration Pattern

Planners declare expected observation type:

```python
def observation_type(self) -> Type[Observation]:
    return DetectionsTracks  # This planner expects detection tracks
```

**Benefit**: Framework can validate planner-observation compatibility at startup, preventing runtime failures.

### 4. Strategy Pattern - Pluggable Planning Algorithms

All planner types implement same interface:

```python
# Planning strategy 1: Rule-based
planner = SimplePlanner()

# Planning strategy 2: Machine learning
planner = MLPlanner(checkpoint_path="model.ckpt")

# Planning strategy 3: Learning + rules hybrid
planner = HybridPlanner(ml_weight=0.7, rule_weight=0.3)

# All compatible with framework:
runner = SimulationRunner(simulation, planner)
```

**Benefit**: Planners are interchangeable - swap algorithms without changing framework.

## Concrete Implementations

### SimplePlanner - Rule-Based Baseline

```python
class SimplePlanner(AbstractPlanner):
    """Straight-line planner with constant acceleration."""
    
    def __init__(self, horizon_seconds=8.0, sampling_time=0.1, 
                 acceleration=1.5, max_velocity=10.0):
        super().__init__()
        self._horizon = horizon_seconds
        self._sampling_time = sampling_time
        self._acceleration = acceleration
        self._max_velocity = max_velocity
        self._map_api = None  # Set in initialize()
    
    def initialize(self, initialization: PlannerInitialization):
        self._map_api = initialization.map_api
    
    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks
    
    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """Plan trajectory: go forward with acceleration."""
        ego_state = current_input.history.current_state[0]
        
        # Generate sequence of future states
        future_states = [ego_state]
        for t in np.arange(self._sampling_time, self._horizon, self._sampling_time):
            # Simple motion model
            next_state = self._propagate_state(future_states[-1], self._acceleration)
            
            # Clamp velocity
            if next_state.velocity > self._max_velocity:
                next_state.velocity = self._max_velocity
            
            future_states.append(next_state)
        
        return InterpolatedTrajectory(future_states)
    
    def name(self) -> str:
        return "simple_planner"
```

### MLPlanner - Learned Behavior

```python
class MLPlanner(AbstractPlanner):
    """ML-based planner that learns from expert demonstrations."""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        super().__init__()
        self._checkpoint_path = checkpoint_path
        self._device = device
        self._model = None  # Loaded in initialize()
        self._map_api = None
    
    def initialize(self, initialization: PlannerInitialization):
        # Load PyTorch Lightning checkpoint
        self._model = load_model(self._checkpoint_path, self._device)
        self._map_api = initialization.map_api
    
    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks
    
    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """Plan using learned model."""
        # Extract features from observations
        features = self._extract_features(
            current_input.history,
            current_input.traffic_light_data
        )
        
        # Run inference
        with torch.no_grad():
            predictions = self._model(features)  # Shape: (batch, horizon, 3) for (x, y, heading)
        
        # Decode to trajectory
        ego_state = current_input.history.current_state[0]
        future_states = self._decode_predictions(ego_state, predictions)
        
        return InterpolatedTrajectory(future_states)
    
    def name(self) -> str:
        return "ml_planner"
```

## Timing & Performance Constraints

### Realtime Timing Budget

```
Simulation Timestep:     0.1 seconds (100 ms)
├─ get_planner_input():  ~1-2 ms (buffer access)
├─ compute_trajectory(): ~80 ms  (PLANNER MAIN WORK!)
└─ propagate():          ~5-10 ms (state update)
│
└─ TOTAL:               ~85-100 ms (at edge of realtime)
```

**Critical**: Planner must complete in ~80ms for 1.0x realtime.

### Performance Tracking

```python
# After simulation completes
report = planner.generate_planner_report()
runtimes_ms = report.compute_trajectory_runtimes

# Analyze
import statistics
print(f"Mean: {statistics.mean(runtimes_ms):.2f} ms")
print(f"Max:  {max(runtimes_ms):.2f} ms")
print(f"P95:  {statistics.quantiles(runtimes_ms, n=20)[18]:.2f} ms")
```

**Expected range for SimplePlanner**: 1-5 ms
**Expected range for MLPlanner**: 50-100 ms

## Threading Model

### Thread-Safety Properties

**Read-only operations (safe for concurrent threads)**:
- `observation_type()` - Returns immutable Type object
- `name()` - Returns immutable string
- `generate_planner_report()` - Reads immutable runtime list

**Initialization (not thread-safe, called once per scenario)**:
- `initialize()` - Modifies `self._map_api`, `self._route`, etc.
- Must be called from single thread before any compute calls

**Compute (varies by implementation)**:
- `compute_planner_trajectory()` - User code determines thread-safety
- Default assumption: **Stateless planners are thread-safe**
- Stateful planners (e.g., with temporal filtering) must synchronize

**Performance implication**:
- Ray workers run independent Simulation instances (no contention)
- Within single Simulation, planner is called sequentially (single-threaded per sim)

### Best Practice for Custom Planners

```python
class MyPlanner(AbstractPlanner):
    def initialize(self, initialization: PlannerInitialization):
        # ONE-TIME ONLY: set up route, map
        self._map_api = initialization.map_api
        self._route = initialization.route_roadblock_ids
        # CRITICAL: All state must be set here
    
    def compute_planner_trajectory(self, current_input: PlannerInput):
        # EVERY CALL: stateless or thread-safe modifications only
        # ✅ OK: Read from self._map_api (set in initialize)
        # ✅ OK: Create local variables for intermediate results
        # ❌ BAD: Modify self._state without locks
        # ❌ BAD: Use class-level mutable state
```

## Common Patterns & Recipes

### Pattern 1: Accessing Recent History

```python
def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
    # Current state
    ego_state, observation = current_input.history.current_state
    
    # Last N states (e.g., velocity estimation)
    ego_states = current_input.history.ego_states
    last_5_states = ego_states[-5:]
    
    # All observations
    observations = current_input.history.observations
    tracked_objects = observation.tracked_objects
    agents = [obj for obj in tracked_objects if obj.is_vehicle]
    
    return trajectory
```

### Pattern 2: Map Querying During Planning

```python
def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
    ego_state = current_input.history.current_state[0]
    
    # Get nearby lanes
    proximal_lanes = self._map_api.get_proximal_map_objects(
        point=ego_state.center,
        radius=50.0,  # meters
        layers=["lanes", "intersections"]
    )
    
    # Check if at intersection
    ego_roadblock = self._map_api.get_map_object(
        roadblock_id=self._route[0],
        layer=SemanticMapLayer.ROADBLOCK
    )
    
    return trajectory
```

### Pattern 3: Trajectory Horizon Selection

```python
def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
    ego_state = current_input.history.current_state[0]
    
    # Adaptive horizon based on speed
    if ego_state.velocity > 8.0:  # High speed
        horizon_seconds = 10.0  # Plan further ahead
    else:
        horizon_seconds = 6.0   # Shorter horizon when slow
    
    # Generate trajectory with variable length
    future_states = self._plan(horizon_seconds)
    return InterpolatedTrajectory(future_states)
```

### Pattern 4: Traffic Light Handling

```python
def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
    # Check traffic light status
    traffic_lights = current_input.traffic_light_data or []
    
    ego_state = current_input.history.current_state[0]
    
    # Find traffic lights on path
    for tl in traffic_lights:
        if tl.status == TrafficLightStatusType.RED:
            # Plan to stop before this light
            return self._plan_stop_before_light(tl.location)
        elif tl.status == TrafficLightStatusType.GREEN:
            # Plan to proceed through
            pass
    
    # No traffic light - normal planning
    return self._plan_normal()
```

## Gotchas & Anti-Patterns

### Gotcha 1: Modifying PlannerInput

```python
# ❌ WRONG - PlannerInput is frozen!
def compute_planner_trajectory(self, current_input: PlannerInput):
    current_input.history.ego_states.append(new_state)  # Fails or has side effects!

# ✅ CORRECT - Create new list
def compute_planner_trajectory(self, current_input: PlannerInput):
    states = list(current_input.history.ego_states)  # Copy
    states.append(new_state)  # Modify copy
    # Use states for planning
```

### Gotcha 2: Forgetting Trajectory Horizon

```python
# ❌ WRONG - Returns trajectory only 0.5s long
def compute_planner_trajectory(self, current_input: PlannerInput):
    for t in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:  # Only 5 steps!
        # ...
    return InterpolatedTrajectory(states)  # Too short

# ✅ CORRECT - Return 8-10 second horizon
def compute_planner_trajectory(self, current_input: PlannerInput):
    for t in np.arange(0, 8.0, 0.1):  # 80 steps = 8 seconds
        # ...
    return InterpolatedTrajectory(states)
```

### Gotcha 3: Observation Type Mismatch

```python
# ❌ WRONG - Observation types don't match
planner = MyPlanner()  # expects DetectionsTracks
observation_builder = LidarPcObservation()  # returns Sensors
setup = SimulationSetup(observations=observation_builder, ...)
validate_planner_setup(setup, planner)  # ValueError!

# ✅ CORRECT - Match types
planner = MyPlanner()  # expects DetectionsTracks
observation_builder = TracksObservation()  # returns DetectionsTracks
setup = SimulationSetup(observations=observation_builder, ...)
validate_planner_setup(setup, planner)  # Passes
```

### Gotcha 4: Accessing Uninitialized Map API

```python
# ❌ WRONG - initialize() not called yet
planner = MyPlanner()
# Planner tries to query map before initialize()
location = planner._map_api.get_proximal_map_objects(...)  # AttributeError!

# ✅ CORRECT - initialize() sets up map
planner = MyPlanner()
planner.initialize(initialization)
# Now map_api is available
location = planner._map_api.get_proximal_map_objects(...)
```

### Gotcha 5: Timing Constraint Violations

```python
# ❌ WRONG - Takes 200ms, exceeds 100ms timestep
def compute_planner_trajectory(self, current_input):
    # Feature extraction: 50ms
    # Monte Carlo rollout: 150ms
    # Total: 200ms = TIMEOUT!
    return trajectory

# ✅ CORRECT - Optimize to fit budget
def compute_planner_trajectory(self, current_input):
    # Feature extraction: 20ms (vectorized)
    # Deterministic planning: 60ms (single rollout)
    # Total: 80ms = OK
    return trajectory
```

### Gotcha 6: Memory Leaks in Stateful Planners

```python
# ❌ WRONG - Accumulates unbounded history
class StatefulPlanner(AbstractPlanner):
    def __init__(self):
        self._all_past_states = []  # Grows indefinitely!
    
    def compute_planner_trajectory(self, current_input):
        # Every call appends
        self._all_past_states.append(current_input.history.current_state)
        # After 200 calls: 200 states in memory!

# ✅ CORRECT - Bound memory with rolling window
class StatefulPlanner(AbstractPlanner):
    def __init__(self, memory_size=20):
        self._past_states = collections.deque(maxlen=memory_size)
    
    def compute_planner_trajectory(self, current_input):
        self._past_states.append(current_input.history.current_state)
        # Automatically evicts old states
```

## Cross-References

- **[SIMULATION_CORE.md](./SIMULATION_CORE.md)** - How planners fit into simulation loop
- **[planner/CLAUDE.md](../nuplan/planning/simulation/planner/CLAUDE.md)** - Planner implementation details
- **[ml_planner/CLAUDE.md](../nuplan/planning/simulation/planner/ml_planner/CLAUDE.md)** - ML planner specifics
- **[history/CLAUDE.md](../nuplan/planning/simulation/history/CLAUDE.md)** - History buffer API
- **[trajectory/CLAUDE.md](../nuplan/planning/simulation/trajectory/CLAUDE.md)** - Trajectory formats

---

**AIDEV-NOTE**: The planner interface is intentionally minimal (3 abstract methods + properties). Keep planning logic in `compute_planner_trajectory()`, not in auxiliary methods that are harder to profile.

