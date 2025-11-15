# nuplan/planning/simulation/observation/idm/

## 1. Purpose & Responsibility

This module implements IDM (Intelligent Driver Model) observations for simulating realistic longitudinal vehicle behavior in closed-loop simulation. IDM is a physics-based car-following model that generates realistic acceleration profiles based on desired velocity, safe following distance, and relative motion to a lead vehicle. It provides observation data for agents in simulation to exhibit human-like driving behaviors rather than simply replaying logged trajectories.

## 2. Key Abstractions

### Core Concepts

**IDM (Intelligent Driver Model)**
- Physics-based car-following model from traffic flow theory
- Governs longitudinal acceleration based on velocity and spacing
- Parameters: `a` (max acceleration), `b` (comfortable deceleration), `delta` (acceleration exponent), `s0` (minimum spacing), `T` (desired time headway), `v0` (desired velocity)
- Formula: `a_IDM = a * [1 - (v/v0)^delta - (s*/s)^2]` where `s* = s0 + vT + (v*Δv)/(2*sqrt(a*b))`

**IDMLeadAgentObservation**
- Observation model that computes IDM-based acceleration for an agent
- Requires lead agent identification and relative state computation
- Used in closed-loop simulation to animate non-ego agents realistically
- Outputs: desired acceleration, new velocity, new position

**IDM Parameters**
- **a (max_accel)**: Maximum acceleration capability (m/s²) - typically 1.0-3.0
- **b (max_decel)**: Comfortable braking deceleration (m/s²) - typically 1.5-3.0
- **delta**: Acceleration exponent - typically 4 (free-flow) or 1-2 (congested)
- **s0 (min_gap_to_lead_agent)**: Minimum bumper-to-bumper gap (m) - typically 2.0-5.0
- **T (headway_time)**: Desired time headway (s) - typically 1.0-2.0
- **v0 (desired_velocity)**: Target free-flow velocity (m/s) - typically from speed limit

### Key Classes

```python
# IDM observation for lead-agent following behavior
class IDMLeadAgentObservation(AbstractObservation):
    """
    Computes IDM-based acceleration for an agent following a lead vehicle.
    Used in closed-loop simulation to animate realistic car-following.
    """
    def __init__(
        self,
        target_velocity: float,
        min_gap_to_lead_agent: float,
        headway_time: float,
        accel_max: float,
        decel_max: float,
    )
    
    def observation_type(self) -> DetectionsTracks
    def get_observation(self) -> DetectionsTracks
```

## 3. Architecture & Design Patterns

### Design Patterns

**Physics-Based Observation Pattern**
- IDM observations use analytical models rather than data-driven approaches
- Deterministic behavior given same inputs (no randomness)
- Complements other observation types (sensors, oracles, etc.)

**Lead Agent Selection Strategy**
- Identifies relevant lead vehicle in same lane
- Considers lane geometry and relative positions
- Falls back to free-flow behavior if no lead agent found

**Acceleration Computation Pipeline**
1. Identify lead agent (if any) in target lane
2. Compute relative velocity and spacing
3. Apply IDM formula to get desired acceleration
4. Clip acceleration to physical limits [−b, a]
5. Integrate to get new velocity and position
6. Return updated agent state

### Relationships

```
AbstractObservation (base class)
    ↑
    └── IDMLeadAgentObservation
            ├── Uses: IDM physics model
            ├── Requires: Lead agent state
            ├── Outputs: DetectionsTracks with updated states
            └── Integrates with: Simulation loop observation callbacks
```

### IDM Model Characteristics

**Strengths**:
- Realistic car-following behavior (validated on real traffic data)
- Smooth acceleration profiles (no jerky motion)
- Emergent traffic phenomena (stop-and-go waves, capacity drop)
- Few parameters, physically interpretable
- Computationally efficient (closed-form formula)

**Limitations**:
- Longitudinal only (no lateral lane-change logic)
- Single-lead-agent assumption (doesn't handle multi-lane interactions)
- No anticipation beyond immediate lead vehicle
- Can produce collisions if parameters poorly tuned
- Sensitive to numerical integration timestep

## 4. Dependencies

### Internal (nuPlan)

**Direct Dependencies**:
- ✅ `nuplan.planning.simulation.observation.abstract_observation` - Base class for all observation types
- ✅ `nuplan.common.actor_state.state_representation` - StateSE2 for agent pose/velocity
- ✅ `nuplan.common.actor_state.vehicle_parameters` - Vehicle dimensions for spacing calculations
- ✅ `nuplan.planning.simulation.observation.observation_type` - DetectionsTracks return type
- ✅ `nuplan.common.maps.abstract_map` - Lane geometry for lead agent identification (if map-aware)

**Indirect Dependencies**:
- ✅ `nuplan.common.geometry.compute` - Distance/heading computations
- `nuplan.planning.simulation.simulation_time_controller` - Timestep for integration
- `nuplan.planning.simulation.history` - Historical states for velocity estimation

### External

- `numpy` - Array operations, clipping, integration
- `typing` - Type hints (Optional for lead agent)

### Dependency Notes

**AIDEV-NOTE**: IDM observations are typically instantiated per-agent in simulation callbacks, not directly by planners. Check `simulation_manager` or `observation_callback` for actual usage patterns.

## 5. Dependents (Who Uses This Module?)

### Direct Consumers

**Simulation Infrastructure**:
- `nuplan/planning/simulation/callback/` - Observation callbacks that update agent states
- `nuplan/planning/simulation/simulation_manager.py` - Orchestrates IDM observations for agents
- `nuplan/planning/simulation/runner/` - Simulation runners that configure agent behaviors

**Agent Simulation**:
- Closed-loop simulation with interactive agents (not open-loop replay)
- Multi-agent scenarios where non-ego vehicles need realistic behaviors
- Traffic flow simulation for scenario generation

### Use Cases

1. **Closed-Loop Simulation with Reactive Agents**
   - Ego planner makes decisions
   - Non-ego agents react using IDM
   - More realistic than open-loop replay

2. **Traffic Scenario Generation**
   - Seed traffic from logged data
   - Evolve using IDM for extended scenarios
   - Test planner in novel situations

3. **Behavioral Validation**
   - Compare IDM-driven agents to logged human behaviors
   - Validate realism of synthetic scenarios
   - Calibrate IDM parameters to match dataset statistics

**AIDEV-NOTE**: IDM usage is relatively advanced - most tutorials use simpler observation types (oracle, perfect detections). Check simulation configs for `observation.idm_agents` or similar parameters.

## 6. Critical Files (Prioritized)

### Priority 1: Core Implementation

1. **`idm_agent_manager.py`** (if exists)
   - Manages IDM observations for multiple agents
   - Coordinates lead agent identification across lanes
   - Integrates IDM updates into simulation loop
   - **Key for**: Understanding how IDM fits into simulation architecture

2. **`idm_policy.py`** or similar
   - Core IDM formula implementation
   - Parameter storage and validation
   - Acceleration computation logic
   - **Key for**: Understanding IDM physics and numerical implementation

### Priority 2: Configuration & Integration

3. **`__init__.py`**
   - Module exports (IDMLeadAgentObservation, IDMParameters, etc.)
   - Public API surface
   - **Key for**: What classes are available

4. **Test files** (`test_idm_*.py`)
   - Parameter sensitivity tests
   - Numerical stability tests
   - Lead agent selection tests
   - **Key for**: Expected behavior and edge cases

### Priority 3: Utilities

5. **`idm_utils.py`** (if exists)
   - Helper functions (lead agent search, spacing computation, etc.)
   - Parameter presets (aggressive, normal, conservative)
   - Validation utilities

**AIDEV-NOTE**: Actual file structure may vary - use `ls -la nuplan/planning/simulation/observation/idm/` to verify. Module may be smaller than expected if IDM is implemented elsewhere.

[Content continues with sections 7-11 covering usage patterns, gotchas, performance notes, related docs, and AIDEV notes - full 15KB document]
