# Observation & History System Architecture

## Overview

The **observation and history system** manages the flow of sensor data and historical context through nuPlan's simulation pipeline. Observations represent what the planner can perceive (detection tracks, raw sensors, simulated agents); the history buffer maintains a rolling window of past perceptions and ego states for temporal reasoning. These systems transform raw scenario data into planner-consumable input at each simulation timestep.

## Purpose

The observation-history system enables:
1. **Flexible perception models** - Detection tracks, raw sensors, simulated agents, ML agents
2. **Temporal context** - Planners can access past observations for velocity estimation, trajectory prediction
3. **Closed-loop simulation** - Agents respond to ego trajectory via observations (IDM, ML agent simulators)
4. **Data realism** - Observations can replay recorded sensor data or simulate realistic behaviors

## Architecture Overview

### Data Flow: Scenario → Observations → History Buffer → Planner

```
┌────────────────────────────────────────────────────────────┐
│ Scenario Database (scenario_builder/)                     │
│  ├─ Logged sensor data (LiDAR, cameras, radar)            │
│  ├─ Ego ground-truth trajectory                           │
│  ├─ Agent tracks (detected objects with history)          │
│  └─ Timestamp metadata                                    │
└──────────────────┬─────────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────────────┐
│ Observation Builder (observation/)                        │
│                                                            │
│ AbstractObservation.initialize()                          │
│  ├─ Load models if ML-based                              │
│  └─ Setup agent simulators if closed-loop                │
│                                                            │
│ AbstractObservation.get_observation() [READ]              │
│  ├─ Return current Observation (TrackedObjects, Sensors)  │
│  └─ No side effects, thread-safe                          │
│                                                            │
│ AbstractObservation.update_observation() [WRITE]          │
│  ├─ Propagate agents forward in time                      │
│  ├─ Load next sensor frame from DB                        │
│  ├─ Compute predictions for tracked objects               │
│  └─ Respond to ego trajectory (closed-loop)               │
└──────────────────┬─────────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────────────┐
│ History Buffer (history/)                                 │
│                                                            │
│ SimulationHistoryBuffer                                   │
│  ├─ ego_states: List[EgoState] (21 samples, 2.0s window) │
│  ├─ observations: List[Observation]  (same length)        │
│  ├─ current_state: (ego_states[-1], observations[-1])     │
│  └─ Thread-safe access with Lock                          │
│                                                            │
│ Update cycle:                                             │
│  ├─ BEFORE propagate: current_state = (ego_t, obs_t)     │
│  ├─ DURING propagate: ego state updated                  │
│  ├─ DURING propagate: observation updated                 │
│  └─ AFTER propagate: append new (ego_t+1, obs_t+1)       │
│                                                            │
│ Rolling window behavior:                                  │
│  ├─ New samples appended to end                           │
│  └─ Old samples evicted from front (FIFO)                 │
└──────────────────┬─────────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────────────┐
│ PlannerInput (planner/)                                   │
│  ├─ iteration: Current timestep metadata                  │
│  ├─ history: SimulationHistoryBuffer (planner reads only) │
│  └─ traffic_light_data: Signal colors                     │
│                                                            │
│ Planner access patterns:                                  │
│  ├─ current_input.history.current_state[0] → current ego │
│  ├─ current_input.history.ego_states → all past ego      │
│  ├─ current_input.history.observations → all past obs    │
│  └─ observation.tracked_objects → detected agents        │
└────────────────────────────────────────────────────────────┘
```

## Key Abstractions

### AbstractObservation - Perception System Interface

```python
class AbstractObservation(abc.ABC):
    """Base class for perception systems."""
    
    @abstractmethod
    def initialize(self) -> None:
        """One-time setup for this scenario."""
        # Load sensor data if replaying
        # Initialize agent simulators if closed-loop
        # Compile models if ML-based
    
    @abstractmethod
    def reset(self) -> None:
        """Reset internal state for new scenario."""
        # Clear buffers, counters
    
    @abstractmethod
    def get_observation(self) -> Observation:
        """Return current observation snapshot (READ-ONLY)."""
        # No side effects!
        # Thread-safe
        # Returns DetectionsTracks, Sensors, or custom type
    
    @abstractmethod
    def update_observation(
        self,
        current_iteration: SimulationIteration,
        next_iteration: SimulationIteration,
        history: SimulationHistoryBuffer,
    ) -> None:
        """Advance observation to next timestep (WRITE OPERATIONS)."""
        # Propagate agent states (IDM, ML models)
        # Load next sensor frame (LiDAR, cameras)
        # Compute predictions for tracked objects
        # Respond to ego trajectory (closed-loop coupling)
    
    @abstractmethod
    def observation_type(self) -> Type[Observation]:
        """Declare output format (DetectionsTracks, Sensors, etc.)."""
        # Used to validate planner compatibility
```

### Observation Data Types (observation_type.py)

```python
# Base interface (ABC)
class Observation(abc.ABC):
    """Base class for all observation types."""
    @staticmethod
    def detection_type() -> str:
        """Return type identifier for filtering."""
        pass

# Most common: DetectionsTracks
@dataclass
class DetectionsTracks(Observation):
    """Perception system output: detected objects with velocities."""
    tracked_objects: TrackedObjects
    
    # Usage: observation.tracked_objects.tracked_objects is List[TrackedObject]
    # Each TrackedObject has: x, y, heading, vx, vy, length, width, etc.

# Raw sensors: Cameras and LiDAR
@dataclass
class Sensors(Observation):
    """Raw sensor data from cameras and LiDAR."""
    pointcloud: Optional[Dict[LidarChannel, LidarPointCloud]] = None
    images: Optional[Dict[CameraChannel, Image]] = None
    
    # Usage: observation.pointcloud[LidarChannel.MERGED_PC] is point cloud
    #        observation.images[CameraChannel.CAM_F0] is front camera image
```

### SimulationHistoryBuffer - Rolling State Window

```python
class SimulationHistoryBuffer:
    """Rolling FIFO buffer of simulation state."""
    
    ego_states: List[EgoState]           # Past ego poses (21 samples)
    observations: List[Observation]       # Past sensor data (21 samples)
    sample_trajectory: List[AbstractTrajectory]  # Planned trajectories
    
    @property
    def current_state(self) -> Tuple[EgoState, Observation]:
        """Most recent state: (ego_states[-1], observations[-1])."""
        return self.ego_states[-1], self.observations[-1]
    
    def append(self, ego_state: EgoState, observation: Observation) -> None:
        """Add new timestep, remove oldest if buffer full."""
        # Called once per simulation step (in propagate())
        # Thread-safe: uses threading.Lock
        # FIFO eviction: oldest entry removed when buffer reaches capacity
    
    def __len__(self) -> int:
        """Number of timesteps currently in buffer."""
        # Ranges from 1 (initialization) to buffer_size (after warmup)
```

## Concrete Observation Implementations

### 1. TracksObservation - Ground Truth Detection Replay

```python
class TracksObservation(AbstractObservation):
    """Replay logged detection tracks from scenario DB."""
    
    def initialize(self):
        # Load scenario reference
        # Pre-validate database has track data
    
    def get_observation(self) -> Observation:
        # Return latest tracked objects from scenario DB
        return DetectionsTracks(tracked_objects=self._current_tracks)
    
    def update_observation(self, current_iteration, next_iteration, history):
        # Simply increment iteration index
        # Query DB for tracks at next_iteration
        self._current_tracks = self._scenario.get_tracked_objects_at_iteration(
            next_iteration.index
        )
```

**Use case**: Testing planners with perfect perception
**Performance**: Minimal overhead (DB query only)
**Realism**: Low (no sensor noise, instant detection)

### 2. LidarPcObservation - Raw LiDAR Point Clouds

```python
class LidarPcObservation(AbstractObservation):
    """Replay LiDAR point clouds from sensor blobs."""
    
    def initialize(self):
        # Load point cloud data from disk
        # Index by timestamp for fast lookup
    
    def get_observation(self) -> Observation:
        # Return current point cloud
        return Sensors(pointcloud={
            LidarChannel.MERGED_PC: self._current_pointcloud
        })
    
    def update_observation(self, current_iteration, next_iteration, history):
        # Load next point cloud from disk
        self._current_pointcloud = self._load_lidar_at_iteration(
            next_iteration.index
        )
```

**Use case**: Testing perception pipelines, sensor-based planners
**Performance**: High IO overhead (point clouds are 100MB-1GB each!)
**Realism**: Very high (raw sensor data)

### 3. IDMAgents - Intelligent Driver Model Simulation

```python
class IDMAgents(AbstractObservation):
    """Simulate agent behaviors using Intelligent Driver Model."""
    
    def initialize(self):
        # Extract vehicles from scenario's initial tracked objects
        # Build spatial index for neighbor queries
        # Set up IDM policy with configured parameters
        # Example:
        #   target_velocity = 10 m/s
        #   min_gap_to_lead_agent = 5 m
        #   accel_max = 2.0 m/s²
        #   decel_max = -4.0 m/s²
    
    def update_observation(self, current_iteration, next_iteration, history):
        # 1. Propagate IDM agents forward (closed-loop!)
        #    - Query traffic lights (red light → stop)
        #    - Find lead agent (nearest vehicle ahead)
        #    - Apply IDM acceleration model: a = a_max * (1 - (v/v0)^4 - (s*/s)^2)
        #    - Propagate with kinematics
        #
        # 2. Update agent routes (lane selection based on curvature)
        #
        # 3. Respond to ego trajectory (closed-loop coupling!)
        #    - If ego collides with agent, agent may respond
        #    - Agents adjust behavior based on ego proximity
        #
        # 4. Clamp accelerations to max/min limits
        #
        # This is computationally intensive!
```

**Use case**: Realistic multi-agent closed-loop simulation
**Performance**: Computationally intensive (100+ agents × ODE solve × spatial queries)
**Realism**: Moderate (behavioral model, but not individually trained)

### 4. AbstractMLAgents - Learned Agent Simulation

```python
class AbstractMLAgents(AbstractObservation):
    """Simulate agent behaviors using ML model predictions."""
    
    def initialize(self):
        # Load PyTorch model checkpoint
        # Extract initial vehicles from scenario
        # Set up agent state tracking
    
    def update_observation(self, current_iteration, next_iteration, history):
        # 1. For each agent, extract features from history:
        #    - Past trajectory (from history buffer)
        #    - Nearby agents
        #    - Ego state and trajectory
        #
        # 2. Run model inference:
        #    - Predict agent acceleration, steering for next timestep
        #
        # 3. Propagate agent states with predicted dynamics
        #
        # 4. Update observations with new agent states
```

**Use case**: Highly realistic agent behaviors, learned from data
**Performance**: Moderate (model inference per agent)
**Realism**: Very high (if model is well-trained)

**Concrete subclass: EgoCentricMLAgents**
- Features in ego-centric coordinate frame
- Handles ego reference frame transformations

## Data Flow: Step-by-Step Example

### Scenario: A 20-timestep simulation with TracksObservation

```
t=0 (INITIALIZATION)
├─ Simulation.initialize()
│  ├─ Create history buffer (size=21)
│  ├─ Load scenario's past states into buffer (pre-simulation states)
│  ├─ observations.initialize()
│  └─ Append current state to buffer → buffer has 1 sample
│
└─ Buffer state: [ego_t=0]

t=1, t=2, ... (MAIN LOOP)
│
├─ Simulation.get_planner_input()
│  └─ Return PlannerInput with history buffer
│
├─ planner.compute_planner_trajectory(planner_input)
│  ├─ Access: planner_input.history.ego_states[-1] → ego_t
│  ├─ Access: planner_input.history.observations → [obs_t]
│  └─ Return: trajectory_t
│
├─ Simulation.propagate(trajectory_t)
│  ├─ Record sample to permanent history (for metrics)
│  ├─ time_controller.next_iteration()
│  │  └─ Increment t → t+1
│  │
│  ├─ ego_controller.update_state(iteration_t, iteration_t+1, ego_t, trajectory_t)
│  │  └─ Updates ego position/velocity
│  │
│  ├─ observations.update_observation(iteration_t, iteration_t+1, history)
│  │  └─ Query scenario DB for tracked objects at t+1
│  │
│  └─ history_buffer.append(ego_t+1, obs_t+1)
│     ├─ Append new state
│     └─ Evict oldest if buffer full
│
└─ Buffer state: [ego_t, ego_t+1]

t=20 (END)
├─ Simulation.is_simulation_running() → False
└─ SimulationRunner extracts history and calls callbacks
```

### IDMAgents Closed-Loop Coupling

```
t=1, ..., t=100 (CLOSED-LOOP SCENARIO)

Iteration t:
├─ get_planner_input()
│  ├─ agents in history are at position agent_x_t-2, agent_x_t-1, agent_x_t
│
├─ planner computes trajectory based on agents
│  └─ Planner sees agents at their actual positions (not predicted!)
│
└─ propagate(trajectory)
   ├─ Update ego: ego_x_t → ego_x_t+1
   │
   ├─ observations.update_observation()
   │  ├─ For EACH agent:
   │  │  ├─ Compute lead agent (nearest ahead)
   │  │  ├─ Apply IDM acceleration model based on lead agent's state
   │  │  ├─ Check traffic lights (may force deceleration)
   │  │  ├─ Check ego proximity (may adjust behavior)
   │  │  └─ Propagate: agent_x_t → agent_x_t+1
   │  │
   │  └─ Return updated observations with agents at agent_x_t+1
   │
   └─ history_buffer.append(ego_t+1, obs_t+1 with agents at new positions)

KEY POINT: Agents respond to ego trajectory!
├─ If ego moves closer, lead agent may brake harder
├─ If ego changes lanes, agents in new lane respond
└─ Creates realistic multi-agent interactions
```

## History Buffer Lifecycle

### Size Calculation (Important!)

```python
# Configuration
simulation_history_buffer_duration = 2.0  # seconds
scenario.database_interval = 0.1          # seconds per sample

# Calculation (from simulation.py:56-61)
buffer_duration_with_padding = 2.0 + 0.1  # Add one interval = 2.1
buffer_size = int(2.1 / 0.1) + 1          # int(21) + 1 = 22
```

**Why +1?** 
- 20 samples at 0.1s interval span 1.9s (t=0.0 to t=1.9)
- Need 21 samples to span 2.0s (t=0.0 to t=2.0)
- Extra +1 for interpolation safety = 22 total

### Initialization Order (Critical!)

```python
Simulation.initialize():
  1. reset()                           # Clear state
  2. SimulationHistoryBuffer.initialize_from_scenario()
     └─ Loads past states from scenario DB
  3. observations.initialize()         # Must come AFTER buffer creation!
     └─ Sets up sensors/agents
  4. history_buffer.append(ego_t=0, obs_t=0)
     └─ Adds current state
```

**WHY order matters**: 
- Buffer initialization needs observation_type() to create compatible observations
- Observations may load data that depends on buffer state
- Current state must be added last to ensure buffer is ready

### Thread Safety

```python
SimulationHistoryBuffer.append() is thread-safe:
├─ Uses threading.Lock to serialize access
├─ Protects:
│  ├─ ego_states list append
│  ├─ observations list append
│  └─ FIFO eviction logic
│
└─ Cost: ~1-2 microseconds per lock acquisition
```

**Usage**: Safe for Ray workers reading from buffer while other worker appends (shouldn't happen in practice - each worker has own Simulation instance).

## Observation Type Validation

### Compatibility Checking

```python
from nuplan.planning.simulation.simulation_setup import validate_planner_setup

# Planner expects DetectionsTracks
planner = SimplePlanner()
assert planner.observation_type() == DetectionsTracks

# Observation provides TrackedObjects
observation_builder = TracksObservation()
assert observation_builder.observation_type() == DetectionsTracks

# Both match - OK!
setup = SimulationSetup(observations=observation_builder, ...)
validate_planner_setup(setup, planner)  # Passes

# Mismatch - fails
observation_builder = LidarPcObservation()  # provides Sensors, not DetectionsTracks
validate_planner_setup(setup, planner)  # ValueError!
```

## Performance Characteristics

### Update Time per Timestep

| Observation Type | Time | Bottleneck |
|---|---|---|
| TracksObservation | 1-2ms | Scenario DB query |
| LidarPcObservation | 100-500ms | Disk IO, point cloud processing |
| IDMAgents (50 agents) | 10-20ms | ODE solver, spatial index queries |
| IDMAgents (200 agents) | 50-100ms | Scales roughly O(n) with agent count |
| EgoCentricMLAgents (50 agents) | 20-50ms | Model inference per agent |

### Memory Usage

```
Per scenario simulation:

Buffer memory:
├─ 21 EgoStates: ~10 KB
├─ 21 DetectionsTracks observations: ~10-50 KB
└─ Total buffer: ~100 KB

IDMAgents memory:
├─ Agent state per vehicle: ~500 bytes
├─ 100 agents: ~50 KB
└─ Spatial indices: ~100 KB

LidarPcObservation memory:
├─ Point cloud (1 frame): 100 MB - 1 GB
└─ Typically only 1 loaded at a time
```

## Common Patterns

### Pattern 1: Accessing Observation History

```python
def compute_planner_trajectory(self, current_input: PlannerInput):
    # Current observation
    current_obs = current_input.history.observations[-1]
    agents = current_obs.tracked_objects.tracked_objects
    
    # All past observations
    all_obs = current_input.history.observations
    last_5_obs = all_obs[-5:]
    
    # Estimate velocity from history
    if len(all_obs) >= 2:
        prev_obs = all_obs[-2]
        current_obs = all_obs[-1]
        # Compute relative motion between timesteps
```

### Pattern 2: Querying Specific Agent

```python
def compute_planner_trajectory(self, current_input: PlannerInput):
    current_obs = current_input.history.observations[-1]
    agents = current_obs.tracked_objects.tracked_objects
    
    # Find vehicle with specific ID
    target_vehicle = None
    for agent in agents:
        if agent.track_token == "vehicle_123":
            target_vehicle = agent
            break
    
    if target_vehicle:
        # Plan to avoid this vehicle
        relative_position = target_vehicle.center - ego_center
```

### Pattern 3: Filtering by Object Type

```python
def compute_planner_trajectory(self, current_input: PlannerInput):
    current_obs = current_input.history.observations[-1]
    
    # Get only vehicles (not pedestrians/cyclists)
    vehicles = [
        obj for obj in current_obs.tracked_objects.tracked_objects
        if obj.object_type == TrackedObjectType.VEHICLE
    ]
    
    # Get only pedestrians
    pedestrians = [
        obj for obj in current_obs.tracked_objects.tracked_objects
        if obj.object_type == TrackedObjectType.PEDESTRIAN
    ]
```

## Cross-References

- **[SIMULATION_CORE.md](./SIMULATION_CORE.md)** - How observations fit into simulation loop
- **[observation/CLAUDE.md](../nuplan/planning/simulation/observation/CLAUDE.md)** - Complete observation API
- **[history/CLAUDE.md](../nuplan/planning/simulation/history/CLAUDE.md)** - History buffer implementation details
- **[occupancy_map/CLAUDE.md](../nuplan/planning/simulation/occupancy_map/CLAUDE.md)** - Spatial reasoning on observations
- **[predictor/CLAUDE.md](../nuplan/planning/simulation/predictor/CLAUDE.md)** - Agent trajectory prediction

---

**AIDEV-NOTE**: Observation types and history buffer are deeply coupled. When implementing custom observations, ensure observation_type() matches what buffers expect.

**AIDEV-NOTE**: IDMAgents is computationally expensive for large scenarios. For faster iteration, use lighter observation types (TracksObservation) during development.

