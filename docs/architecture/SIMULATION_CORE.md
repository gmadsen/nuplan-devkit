# Core Simulation Loop Architecture

## Overview

The **core simulation loop** is the heart of nuPlan's closed-loop evaluation system. It orchestrates the interaction between planners, controllers, and observations in real-time simulation. This document explains how the `Simulation` class, `SimulationRunner`, and supporting components work together to execute a closed-loop simulation from start to finish.

## Purpose

The core simulation loop enables **closed-loop evaluation** of autonomous vehicle planners by:
1. Maintaining ego state and observation history as the simulation progresses
2. Repeatedly querying planners for trajectory decisions at each timestep
3. Propagating ego state forward based on planner output and vehicle dynamics
4. Computing metrics and recording simulation data for offline analysis

## Architecture Overview

### High-Level Simulation Flow

```
┌──────────────────────────────────────────────────────────────────┐
│ Execution Layer (runner/, script/)                              │
│  ├─ Hydra configuration resolution                             │
│  ├─ Scenario selection and filtering                           │
│  ├─ Planner instantiation                                      │
│  └─ Ray/Sequential worker orchestration                        │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│ SimulationRunner (runner/simulations_runner.py)                 │
│  ├─ One-time initialization (planner.initialize)              │
│  ├─ Lifecycle callback orchestration (8 hooks)                 │
│  └─ Metrics extraction and report generation                   │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│ Simulation (simulation.py)  ◄──── SimulationSetup              │
│                                                                 │
│  INITIALIZATION PHASE                                          │
│  ├─► reset()                                                   │
│  ├─► initialize()                                              │
│  │    ├─ Create SimulationHistoryBuffer (history/)            │
│  │    ├─ Pre-load past states from scenario DB                │
│  │    ├─ Initialize observations (observation/)               │
│  │    └─ Return PlannerInitialization                         │
│  │                                                              │
│  MAIN SIMULATION LOOP                                          │
│  └─► while is_simulation_running():                            │
│       ├─► get_planner_input()                                  │
│       │    ├─ Query current iteration (time_controller/)       │
│       │    ├─ Get traffic light status (scenario_builder/)     │
│       │    └─ Return PlannerInput with history buffer          │
│       │                                                         │
│       ├─► [Planner.compute_trajectory()]  ◄─── (in runner)    │
│       │    ├─ Process observations                             │
│       │    ├─ Query map (path/, occupancy_map/)                │
│       │    └─ Predict agents (predictor/)                      │
│       │                                                         │
│       └─► propagate(trajectory)                                │
│            ├─ Record to SimulationHistory (for metrics)        │
│            ├─ Update ego state (controller/)                   │
│            ├─ Update observations (observation/)               │
│            ├─ Advance time (time_controller/)                  │
│            └─ Append new state to history buffer               │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│ Callbacks (callback/)                                           │
│  ├─ on_step_end(): Compute metrics, serialize, visualize      │
│  ├─ on_simulation_end(): Aggregate results                     │
│  └─ on_*_*: 6 more hooks throughout lifecycle                  │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│ MainCallback (main_callback/)                                   │
│  ├─ Aggregate metrics across all scenarios                    │
│  ├─ Upload to S3 (if configured)                              │
│  └─ Generate parquet summary files                            │
└──────────────────────────────────────────────────────────────────┘
```

### Step-by-Step Execution Detail

#### 1. **Initialization Phase** (Called by SimulationRunner)

```python
# Create simulation with configuration
sim = Simulation(simulation_setup=setup, callback=callback)

# One-time initialization
initialization = sim.initialize()
```

**What happens inside `initialize()`:**

1. **Reset state** - Clear history from any previous runs
   ```python
   self.reset()
   ```

2. **Create history buffer** - Rolling window of past states
   ```python
   self._history_buffer = SimulationHistoryBuffer.initialize_from_scenario(
       self._history_buffer_size,  # Default: 21 samples = 2.0s @ 0.1s dt
       self._scenario,             # Scenario interface
       self._observations.observation_type()
   )
   ```
   - **Key detail**: Buffer is pre-loaded with past states from scenario DB
   - Duration = 2.0s by default + 1 extra sample for interpolation
   - Size calculation: `int(duration / db_interval) + 1`

3. **Initialize observations** - Set up perception system
   ```python
   self._observations.initialize()
   ```
   - Load sensor blobs (LiDAR, cameras) if replaying sensor data
   - Initialize agent simulators (IDM, ML models) if closed-loop agents
   - Prepare tracking data if using detection tracks

4. **Append current state to buffer**
   ```python
   self._history_buffer.append(
       self._ego_controller.get_state(),
       self._observations.get_observation()
   )
   ```
   - Adds t=0 ego state and current observations to buffer

5. **Return initialization data to planner**
   ```python
   return PlannerInitialization(
       route_roadblock_ids=self._scenario.get_route_roadblock_ids(),
       mission_goal=self._scenario.get_mission_goal(),
       map_api=self._scenario.map_api
   )
   ```

#### 2. **Main Simulation Loop** (Stepped execution)

```python
while sim.is_simulation_running():
    # Step 1: Get current state and traffic lights
    planner_input = sim.get_planner_input()
    
    # Step 2: Planner decides trajectory [~80ms typically]
    trajectory = planner.compute_planner_trajectory(planner_input)
    
    # Step 3: Propagate simulation state forward
    sim.propagate(trajectory)
    
    # Step 4: Run callbacks (metrics, serialization, etc.)
    # [Handled by SimulationRunner]
```

**Detailed breakdown of each step:**

##### Step 1: `get_planner_input()` - Assemble current observation snapshot

```python
def get_planner_input(self) -> PlannerInput:
    # Get current timestep metadata (index, timestamp)
    iteration = self._time_controller.get_iteration()
    
    # Query traffic light status at this iteration
    traffic_light_data = list(
        self._scenario.get_traffic_light_status_at_iteration(iteration.index)
    )
    
    # Return packed input
    return PlannerInput(
        iteration=iteration,
        history=self._history_buffer,      # Rolling buffer of past states
        traffic_light_data=traffic_light_data
    )
```

**Threading model**: `get_planner_input()` is SAFE to call from planner threads - it only reads state, doesn't modify.

**Data accessed**:
- `history_buffer.ego_states` - List of past ego poses [0.0s to -2.0s]
- `history_buffer.observations` - List of past sensor observations
- `history_buffer.current_state` - Latest ego state and observation
- `traffic_light_data` - Signal colors at specific IDs

##### Step 2: `compute_planner_trajectory()` - Planner decision point

This is **user code** (planner implementation), not part of core simulation.

**Contract**:
- **Input**: PlannerInput with history buffer and traffic lights
- **Output**: AbstractTrajectory (typically InterpolatedTrajectory)
- **Time budget**: Typically 0.1 seconds (must finish before next simulation step)
- **Timing**: Automatically tracked in `compute_trajectory()` wrapper

**What planners typically do**:
1. Extract current ego state: `current_input.history.current_state[0]`
2. Access history: `current_input.history.ego_states[-10:]` (last 10 timesteps)
3. Process observations: `observation.tracked_objects` (detected vehicles)
4. Query map: `self._map_api.get_proximal_map_objects(...)`
5. Make decision: Compute future waypoints
6. Return trajectory: `InterpolatedTrajectory(ego_states_t0_to_t8)`

##### Step 3: `propagate(trajectory)` - Advance simulation state

```python
def propagate(self, trajectory: AbstractTrajectory) -> None:
    # Get current state before step
    iteration = self._time_controller.get_iteration()
    ego_state, observation = self._history_buffer.current_state
    traffic_light_status = list(
        self._scenario.get_traffic_light_status_at_iteration(iteration.index)
    )
    
    # Record this step in permanent history (for metrics)
    self._history.add_sample(
        SimulationHistorySample(
            iteration=iteration,
            ego_state=ego_state,
            trajectory=trajectory,
            observation=observation,
            traffic_light_status=traffic_light_status
        )
    )
    
    # Advance to next timestep
    next_iteration = self._time_controller.next_iteration()
    
    if next_iteration:
        # Update ego controller: execute trajectory
        # (controller tracks trajectory using PerfectTracking or LQR)
        self._ego_controller.update_state(
            current_iteration=iteration,
            next_iteration=next_iteration,
            ego_state=ego_state,
            trajectory=trajectory
        )
        
        # Update observations: agents move, sensors update
        self._observations.update_observation(
            current_iteration=iteration,
            next_iteration=next_iteration,
            history=self._history_buffer
        )
        
        # Append new state to rolling history buffer
        self._history_buffer.append(
            self._ego_controller.get_state(),
            self._observations.get_observation()
        )
    else:
        # Scenario end reached
        self._is_simulation_running = False
```

**Key operations during propagate**:

1. **Recording** - Store sample for metrics computation
2. **Time stepping** - Advance iteration counter
3. **Ego control** - Update ego position/velocity based on trajectory
4. **Observation update** - Move agents, refresh sensor data
5. **History buffering** - Maintain rolling window for next planner input

## Key Data Structures

### SimulationSetup - Configuration Container

```python
@dataclass
class SimulationSetup:
    time_controller: AbstractSimulationTimeController  # Time stepping logic
    observations: AbstractObservation                   # Perception system
    ego_controller: AbstractEgoController              # Motion control
    scenario: AbstractScenario                          # Scenario definition
```

**Pattern**: Builder/Configuration - all components injected at construction time.

### PlannerInput - Planner's view of world state

```python
@dataclass(frozen=True)
class PlannerInput:
    iteration: SimulationIteration                      # Current time metadata
    history: SimulationHistoryBuffer                    # Rolling window of past states
    traffic_light_data: Optional[List[TrafficLightStatusData]]  # Signal colors
```

**Immutability**: Frozen dataclass prevents accidental modification during planning.

### SimulationHistoryBuffer - Rolling state window

```python
class SimulationHistoryBuffer:
    ego_states: List[EgoState]          # Past ego poses
    observations: List[Observation]      # Past sensor data
    
    @property
    def current_state(self) -> Tuple[EgoState, Observation]:
        """Most recent ego state and observation"""
        return self.ego_states[-1], self.observations[-1]
```

**Buffer management**:
- Fixed size (default: 21 samples = 2.0s @ 0.1s)
- Oldest entries automatically removed when full (FIFO)
- Thread-safe: Uses `threading.Lock` for concurrent access

### SimulationHistory - Permanent step records

```python
class SimulationHistory:
    samples: List[SimulationHistorySample]  # All recorded steps
    
    def add_sample(self, sample: SimulationHistorySample):
        """Record one simulation step (called in propagate)"""
```

**Purpose**: Full history for post-simulation analysis (metrics, visualization).

## Control Flow Details

### State Machine: is_simulation_running()

```python
def is_simulation_running(self) -> bool:
    return (not self._time_controller.reached_end() and
            self._is_simulation_running)
```

**Two stop conditions**:
1. **Time limit reached**: `time_controller.reached_end()` returns True
2. **Manual stop**: `_is_simulation_running` set to False

**When manual stop occurs**: In `propagate()` when `next_iteration()` returns None.

### Boundary Conditions

#### Buffer Size Calculation - The "+1" Guard

```python
# Problem: 20 samples at 0.1s interval = 1.9s duration, not 2.0s
# Solution: Add 1 extra sample

buffer_duration = 2.0 + scenario.database_interval  # Pad by 1 interval
buffer_size = int(buffer_duration / scenario.database_interval) + 1
# For scenario.database_interval = 0.1s:
# buffer_size = int(2.1 / 0.1) + 1 = 21 + 1 = 22 samples (!)
```

**Why**: Ensures buffer always covers minimum duration despite discrete sampling.

#### History Buffer Initialization Order

```
1. Create buffer with scenario's past data
   └─► Scenario DB is queried for historical states
       
2. Initialize observations
   └─► Sensors/agents configured, models loaded
       
3. Append current state (t=0)
   └─► Adds live ego state + current observation to buffer
```

**Critical**: Observations must be initialized AFTER buffer creation, because buffer uses observation type to initialize.

### Threading Model

**Simulation class is thread-safe for read operations:**
- ✅ Multiple threads can call `get_planner_input()` in parallel
- ✅ History buffer uses `threading.Lock` for concurrent access
- ❌ Do NOT call `propagate()` from multiple threads (not thread-safe)

**Usage pattern for parallel simulations**:
```python
# ✅ OK: Ray workers run independent Simulation instances
with Ray:
    futures = [
        worker.run_simulation.remote(sim1),
        worker.run_simulation.remote(sim2),
        worker.run_simulation.remote(sim3),
    ]
```

## Performance Characteristics

### Typical Timing (per 0.1s timestep)

| Operation | Time | Notes |
|-----------|------|-------|
| `get_planner_input()` | ~1-2ms | Buffer access, traffic light query |
| `planner.compute_trajectory()` | ~80ms | User code (varies by planner) |
| `propagate()` | ~5-10ms | Controller update, observation update |
| **Total per step** | **~85-100ms** | Should be < 100ms for realtime (0.1s step) |

### Memory Usage

```
Per simulation instance:
├─ History buffer (21 samples):
│  ├─ Ego states: 21 × ~500 bytes = 10 KB
│  └─ Observations: 21 × (depends on type)
│     ├─ DetectionsTracks: ~10-50 KB per sample
│     ├─ Sensors (LiDAR): ~1-10 MB per sample (!!)
│     └─ IDMAgents: ~50-100 KB per sample
│
├─ Scenario reference:
│  ├─ Map API: ~100-500 MB (shared across instances)
│  └─ Scenario DB: ~10 MB per scenario
│
└─ Total: ~100 MB to 1 GB per simulation (depends on observation type)
```

### Bottleneck Analysis

For 0.51x realtime (target: 1.0x):

**Hypothesis**: 80ms planner + 5-10ms propagation = 85ms per 100ms step = 85% utilization

**Optimization opportunities**:
1. **Reduce planner time** (biggest lever)
   - Profile planner: What's taking 80ms?
   - Reduce feature extraction time
   - Optimize model inference
   - Cache map queries

2. **Parallelize planner computation** (multi-threading)
   - Feature extraction on thread 1
   - Model inference on thread 2
   - Won't help if GIL-bound

3. **Reduce observation update time**
   - IDMAgents: Agent propagation via ODE solver
   - ML agents: Model inference per agent
   - Switch to lightweight observation type if possible

4. **Reduce propagate() overhead**
   - History buffer append: Check if creating copies
   - Controller update: Check LQR computation time
   - Observation reset: Check data structure operations

## Cross-References

- **[PLANNER_INTERFACE.md](./PLANNER_INTERFACE.md)** - Planner API contract and initialization
- **[OBSERVATION_HISTORY.md](./OBSERVATION_HISTORY.md)** - History buffer, observation update flow
- **[CONTROLLER.md](./CONTROLLER.md)** - Controller trajectory tracking details
- **[simulation_time_controller/CLAUDE.md](../nuplan/planning/simulation/simulation_time_controller/CLAUDE.md)** - Time stepping logic
- **[runner/CLAUDE.md](../nuplan/planning/simulation/runner/CLAUDE.md)** - SimulationRunner lifecycle
- **[callback/CLAUDE.md](../nuplan/planning/simulation/callback/CLAUDE.md)** - Callback hooks

## Commonly Modified Components

When optimizing the simulation loop, focus on these files:

1. **Planner performance** → Your planner implementation
2. **Observation updates** → `observation/{observation_type}.py` update_observation()
3. **History buffer** → `history/simulation_history_buffer.py` append()
4. **Controller tracking** → `controller/{controller_type}.py` update_state()
5. **Time control** → `simulation_time_controller/step_simulation_time_controller.py` next_iteration()

## Common Issues

### RuntimeError: "Simulation was not initialized!"
**Cause**: Called `get_planner_input()` before `initialize()`
**Fix**: Always call `sim.initialize()` first

### RuntimeError: "simulation_history_buffer_duration too small"
**Cause**: Buffer duration < scenario.database_interval
**Fix**: Use minimum 0.2s buffer duration

### Trajectory too short for controller
**Cause**: Planner returned trajectory ending before next iteration
**Fix**: Extend trajectory to 8-10 seconds horizon

### History buffer has wrong observation type
**Cause**: Observation type mismatch between planner and setup
**Fix**: Call `validate_planner_setup(setup, planner)` before simulation

---

**AIDEV-NOTE**: The simulation loop is deterministic and reproducible. Same scenario + planner + random seed = identical results. Use this for debugging!

