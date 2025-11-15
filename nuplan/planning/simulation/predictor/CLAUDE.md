# CLAUDE.md - nuplan/planning/simulation/predictor

## Purpose & Responsibility

**Future trajectory prediction for tracked agents in simulation.** This module defines the interface and implementations for predicting where other vehicles, pedestrians, and objects will move in the future. Predictors consume current observations (DetectionsTracks) and populate the `predictions` field of each Agent with future trajectory hypotheses. The primary implementation `LogFuturePredictor` is an oracle that extracts ground-truth futures from the scenario database, enabling perfect-prediction baselines for planner evaluation.

## Key Abstractions & Classes

### Core Interface

- **`AbstractPredictor`** (ABC) - THE fundamental predictor interface
  - `requires_scenario: bool` - Class attribute, True for oracle predictors that need scenario access
  - **`name() -> str`** - Predictor identifier (typically class name)
  - **`initialize(initialization: PredictorInitialization)`** - One-time setup (called before simulation)
  - **`observation_type() -> Type[Observation]`** - Declares expected observation format (usually DetectionsTracks)
  - **`compute_predicted_trajectories(current_input: PredictorInput) -> DetectionsTracks`** - Core prediction logic (abstract)
  - **`compute_predictions(current_input: PredictorInput) -> DetectionsTracks`** - Wrapper with runtime tracking (concrete)
  - **`generate_predictor_report(clear_stats: bool = True) -> PredictorReport`** - Performance statistics

- **`PredictorInitialization`** (frozen dataclass) - Initialization data
  - `map_api: AbstractMap` - Map queries for context-aware prediction

- **`PredictorInput`** (frozen dataclass) - Per-timestep prediction input
  - `iteration: SimulationIteration` - Current simulation time/index
  - `history: SimulationHistoryBuffer` - Rolling window of past observations
  - `traffic_light_data: Optional[List[TrafficLightStatusData]]` - Traffic signal state

### Concrete Implementations

- **`LogFuturePredictor`** - Oracle predictor (ground truth from logs)
  - `requires_scenario = True` - Needs scenario for future trajectory extraction
  - Constructor: `LogFuturePredictor(scenario: AbstractScenario, future_trajectory_sampling: TrajectorySampling)`
  - **Purpose**: Extract logged future trajectories from scenario database
  - **Use case**: Perfect prediction baseline for planner evaluation, debugging
  - **Mechanism**:
    1. Query `scenario.get_tracked_objects_at_iteration(iteration, sampling)` for future states
    2. Build agent dictionary by token: `{token: agent_with_predictions}`
    3. Match current observations by token
    4. Copy ground-truth predictions to observation agents
    5. Return updated DetectionsTracks
  - **Key limitation**: Only works for agents present in logged scenario (no predictions for new/simulated agents)

### Reporting Infrastructure

- **`PredictorReport`** (frozen dataclass) - Runtime statistics
  - `compute_predictions_runtimes: List[float]` - Time series of prediction call durations [s]
  - `compute_summary_statistics() -> Dict[str, float]` - Mean, median, 95th percentile, std of runtimes

- **`MLPredictorReport`** (frozen dataclass, extends PredictorReport) - ML-specific stats
  - `feature_building_runtimes: List[float]` - Feature extraction time [s]
  - `inference_runtimes: List[float]` - Model forward pass time [s]
  - **Note**: Currently unused (no ML predictor implementation in module)

## Architecture & Design Patterns

### 1. **Strategy Pattern**: Pluggable Prediction Algorithms
- `AbstractPredictor` defines interface
- Concrete implementations (LogFuture, ML, physics-based) are interchangeable
- Future ML predictors would implement same interface
- Hydra config selects predictor at runtime

### 2. **Oracle Pattern**: Ground Truth Extraction
- `LogFuturePredictor` provides "perfect" predictions from logged data
- Enables upper-bound performance analysis (what if prediction were perfect?)
- Useful for isolating prediction errors from planning errors

### 3. **Template Method**: Prediction with Instrumentation
```python
# Concrete implementation
def compute_predicted_trajectories(self, input: PredictorInput) -> DetectionsTracks:
    # Subclass-specific prediction logic
    ...

# Base class wrapper (timing instrumentation)
def compute_predictions(self, input: PredictorInput) -> DetectionsTracks:
    start_time = time.perf_counter()
    try:
        return self.compute_predicted_trajectories(input)
    finally:
        self._compute_predictions_runtimes.append(time.perf_counter() - start_time)
```
- All predictors get automatic runtime tracking
- Exception-safe timing (records even if prediction fails)

### 4. **Lazy Initialization**: `__new__` for Runtime Tracking
```python
def __new__(cls, *args, **kwargs) -> AbstractPredictor:
    instance = super().__new__(cls)
    instance._compute_predictions_runtimes = []  # Initialize before __init__
    return instance
```
- Runtime list created in `__new__` (before constructor)
- Ensures all instances have tracking, even if subclass forgets

### 5. **Token-Based Matching**: Prediction Assignment
- Scenario agents have `metadata.token` (unique identifier across time)
- Current observations also have tokens
- Prediction assignment: `current_agent.predictions = scenario_agents[token].predictions`
- Preserves agent identity across timesteps

## Dependencies (What We Import)

### Internal nuPlan (Documented ✅)
- ✅ `nuplan.common.maps.abstract_map` - AbstractMap for map-aware predictions
- ✅ `nuplan.common.maps.maps_datatypes` - TrafficLightStatusData
- ✅ `nuplan.planning.simulation.history.simulation_history_buffer` - SimulationHistoryBuffer (observation history)
- ✅ `nuplan.planning.simulation.observation.observation_type` - DetectionsTracks, Observation types
- ✅ `nuplan.planning.simulation.simulation_time_controller.simulation_iteration` - SimulationIteration
- ✅ `nuplan.planning.simulation.trajectory.trajectory_sampling` - TrajectorySampling (for future extraction)

### Internal nuPlan (Undocumented ⏳)
- ⏳ `nuplan.planning.scenario_builder.abstract_scenario` - AbstractScenario (for LogFuturePredictor)

### External Dependencies
- `abc` - Abstract base class metaclass
- `time` - Performance timing (perf_counter)
- `dataclasses` - Dataclass decorators
- `typing` - Type hints (List, Optional, Type, Any)
- `numpy` - Array operations (via PredictorReport summary stats)

## Dependents (Who Uses This Module?)

### Direct Consumers

**CRITICAL INSIGHT**: Unlike planners/observations, predictors are **NOT currently used** in standard nuPlan simulation!

**Evidence**:
1. No imports of `AbstractPredictor` outside this module (checked via `rg`)
2. No simulation setup files reference predictors
3. No Hydra configs in `simulation/predictor/` (only `log_future_predictor.yaml`)
4. Observations (IDMAgents, AbstractMLAgents) populate predictions directly

**Actual prediction flow**:
```
Simulation Loop:
  └─ Observation.update_observation()
       ├─ IDMAgents → populates agent.predictions via IDM policy
       ├─ AbstractMLAgents → populates agent.predictions via ML model
       └─ TracksObservation → NO predictions (open-loop)
```

**Potential future use cases**:
1. **Standalone prediction evaluation** - Benchmark prediction models separately from planning
2. **Hybrid prediction** - Combine multiple predictors (physics + ML ensemble)
3. **Prediction-as-a-service** - Decouple prediction from observation module
4. **Research experiments** - Ablation studies on prediction quality

### Theoretical Integration Points

**IF predictors were integrated into simulation**:
```python
# Hypothetical simulation setup
class SimulationSetup:
    planner: AbstractPlanner
    observations: AbstractObservation
    predictor: Optional[AbstractPredictor]  # Currently missing!

# Hypothetical simulation loop
detections = observations.get_observation()
if predictor:
    detections = predictor.compute_predictions(
        PredictorInput(iteration, history, traffic_light_data)
    )
planner_input = PlannerInput(history with predicted detections)
```

**AIDEV-NOTE**: This module appears to be infrastructure for **future** prediction integration. Current simulation uses observation-embedded prediction (IDM, ML agents).

## Critical Files (Prioritized)

1. **`abstract_predictor.py`** (113 lines) - **START HERE!**
   - AbstractPredictor interface (ABC)
   - PredictorInitialization, PredictorInput dataclasses
   - Template method with runtime tracking
   - Simple, well-documented design

2. **`log_future_predictor.py`** (68 lines) - **Oracle implementation**
   - Concrete example of predictor interface
   - Ground-truth prediction extraction from scenario
   - Token-based matching algorithm
   - Shows how to populate Agent.predictions

3. **`predictor_report.py`** (38 lines) - **Performance reporting**
   - PredictorReport dataclass (base)
   - MLPredictorReport (extended for ML predictors)
   - Summary statistics computation (mean, median, percentiles)

4. **`test/mock_abstract_predictor.py`** (64 lines) - **Testing utilities**
   - MockAbstractPredictor (minimal concrete implementation)
   - get_mock_predictor_initialization() - Factory for test data
   - get_mock_predictor_input() - Factory for test inputs
   - Useful for understanding interface contracts

5. **`test/test_abstract_predictor.py`** (51 lines) - **Interface tests**
   - Tests for AbstractPredictor lifecycle
   - Runtime tracking validation
   - Report generation tests

6. **`test/test_log_future_predictor.py`** (48 lines) - **Oracle predictor tests**
   - Validates prediction extraction from scenario
   - Checks prediction count and sampling correctness
   - Report statistics validation

## Common Usage Patterns

### 1. Oracle Predictor (Ground Truth Baseline)

```python
from nuplan.planning.simulation.predictor.log_future_predictor import LogFuturePredictor
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

# Configure prediction horizon (8 seconds, 16 poses)
sampling = TrajectorySampling(num_poses=16, time_horizon=8.0)

# Create oracle predictor
predictor = LogFuturePredictor(
    scenario=scenario,
    future_trajectory_sampling=sampling
)

# Initialize (one-time setup)
predictor.initialize(PredictorInitialization(map_api=map_api))

# Simulation loop
for iteration, next_iteration in timesteps:
    # Build predictor input
    predictor_input = PredictorInput(
        iteration=iteration,
        history=history_buffer,
        traffic_light_data=scenario.get_traffic_light_status_at_iteration(iteration.index)
    )

    # Get predictions (populates agent.predictions)
    detections_with_predictions = predictor.compute_predictions(predictor_input)

    # Access predictions
    for agent in detections_with_predictions.tracked_objects.get_agents():
        if agent.predictions:
            # agent.predictions: List[PredictedTrajectory]
            primary_prediction = agent.predictions[0]
            waypoints = primary_prediction.waypoints  # List[Waypoint]
            probability = primary_prediction.probability  # float
```

### 2. Performance Profiling

```python
# After simulation loop
report = predictor.generate_predictor_report(clear_stats=True)

# Access raw timings
runtimes = report.compute_predictions_runtimes  # List[float] in seconds

# Compute summary statistics
stats = report.compute_summary_statistics()
print(f"Mean prediction time: {stats['compute_predictions_runtimes_mean']:.4f}s")
print(f"95th percentile: {stats['compute_predictions_runtimes_95_percentile']:.4f}s")
print(f"Std dev: {stats['compute_predictions_runtimes_std']:.4f}s")

# Check real-time feasibility (< 100ms for 10Hz simulation)
if stats['compute_predictions_runtimes_95_percentile'] > 0.1:
    print("WARNING: Predictor too slow for real-time simulation!")
```

### 3. Custom Predictor Implementation (Hypothetical ML Predictor)

```python
from nuplan.planning.simulation.predictor.abstract_predictor import (
    AbstractPredictor, PredictorInitialization, PredictorInput
)
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from typing import Type

class MyMLPredictor(AbstractPredictor):
    """Example ML-based trajectory predictor."""

    requires_scenario = False  # No oracle access needed

    def __init__(self, model_path: str, prediction_horizon: float):
        self._model_path = model_path
        self._prediction_horizon = prediction_horizon
        self._model = None  # Lazy load in initialize()

    def name(self) -> str:
        return "MyMLPredictor"

    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks

    def initialize(self, initialization: PredictorInitialization) -> None:
        """Load model and map API."""
        self._map_api = initialization.map_api
        # Load ML model (PyTorch, TensorFlow, etc.)
        self._model = load_model(self._model_path)

    def compute_predicted_trajectories(self, current_input: PredictorInput) -> DetectionsTracks:
        """Run ML inference for trajectory prediction."""
        # Extract current observations
        _, detections = current_input.history.current_state
        agents = detections.tracked_objects.get_agents()

        # Build features (ego-relative states, map context, etc.)
        features = self._build_features(agents, current_input.history)

        # Run model inference
        predictions = self._model.predict(features)  # [num_agents, num_timesteps, state_dim]

        # Convert predictions to PredictedTrajectory objects
        for agent, predicted_states in zip(agents, predictions):
            waypoints = self._states_to_waypoints(predicted_states)
            agent.predictions = [PredictedTrajectory(probability=1.0, waypoints=waypoints)]

        return detections

    def _build_features(self, agents, history):
        """Extract features for ML model (ego-relative positions, velocities, map)."""
        # Implementation details...
        pass

    def _states_to_waypoints(self, predicted_states):
        """Convert numpy array to Waypoint objects."""
        # Implementation details...
        pass
```

### 4. Prediction Report Analysis

```python
import matplotlib.pyplot as plt
import numpy as np

# Collect reports from multiple scenarios
all_runtimes = []
for scenario in scenarios:
    # Run simulation...
    report = predictor.generate_predictor_report()
    all_runtimes.extend(report.compute_predictions_runtimes)

# Visualize runtime distribution
plt.hist(all_runtimes, bins=50, alpha=0.7)
plt.xlabel("Prediction Time (s)")
plt.ylabel("Frequency")
plt.axvline(np.percentile(all_runtimes, 95), color='r', linestyle='--', label='95th %ile')
plt.legend()
plt.title("Predictor Runtime Distribution")
plt.show()

# Identify outliers
outliers = [t for t in all_runtimes if t > np.percentile(all_runtimes, 95)]
print(f"Found {len(outliers)} outlier predictions (> 95th percentile)")
```

## Gotchas & Pitfalls

### Design & Architecture Issues

1. **Predictors are NOT used in standard simulation**
   - No simulation setup includes predictor field
   - Observations (IDMAgents, MLAgents) handle prediction directly
   - **Consequence**: This module is infrastructure for future/research use
   - **Check**: `rg "AbstractPredictor" nuplan/planning/simulation` returns only predictor module itself

2. **`requires_scenario` flag is class-level, not instance-level**
   - Set as class attribute: `requires_scenario = True`
   - Cannot change per-instance (e.g., oracle vs non-oracle mode)
   - **Gotcha**: If you need conditional scenario access, use two separate classes
   - **Design rationale**: Enables static validation before instantiation

3. **`observation_type()` returns class, not instance**
   - Returns `Type[Observation]`, e.g., `DetectionsTracks`
   - NOT an instance of Observation
   - **Purpose**: Type declaration for validation, not runtime usage
   - **Usage**: `if predictor.observation_type() == DetectionsTracks:`

4. **Runtime tracking happens even on exceptions**
   - `compute_predictions()` wrapper has try/finally
   - Time recorded even if `compute_predicted_trajectories()` raises
   - **Consequence**: Failed predictions appear in runtime statistics
   - **Design choice**: Want to measure cost of failed predictions too

5. **`_compute_predictions_runtimes` initialized in `__new__`, not `__init__`**
   - `__new__` runs before `__init__`
   - Ensures list exists even if subclass forgets to call `super().__init__()`
   - **Gotcha**: Don't reassign `_compute_predictions_runtimes` in `__init__` (overwrites tracked data)

### LogFuturePredictor Specific

6. **LogFuturePredictor requires exact token matches**
   - Predictions only assigned if `agent.metadata.token in scenario_agent_dict`
   - **Symptom**: Some agents have `predictions=None` after prediction
   - **Cause**: Agent token not in logged scenario (simulated agents, perception errors)
   - **Fix**: Check `if agent.predictions:` before accessing

7. **Scenario query can return more agents than current observations**
   - `scenario.get_tracked_objects_at_iteration()` includes all logged agents
   - Current detections may be filtered subset (e.g., only nearby agents)
   - **Consequence**: Dict built from scenario has unused entries
   - **Performance**: Minor overhead (dict lookup is O(1))

8. **Prediction sampling is independent of observation frequency**
   - `future_trajectory_sampling` can have different `num_poses` than history
   - Example: Observe at 10Hz, predict 16 poses over 8s (every 0.5s)
   - **Gotcha**: Don't assume prediction timesteps match observation timesteps

9. **Traffic light data is optional**
   - `PredictorInput.traffic_light_data: Optional[...]`
   - Not all scenarios have traffic light annotations
   - **Check**: `if current_input.traffic_light_data:` before using

10. **Predictions overwrite any existing predictions**
    - `agent.predictions = scenario_agent_dict[token].predictions`
    - Replaces, doesn't append
    - **Consequence**: Can't chain multiple predictors easily
    - **Design**: Single predictor per simulation assumed

### Performance & Timing

11. **`time.perf_counter()` overhead is ~200ns**
    - Two perf_counter calls per prediction: start + end
    - Negligible for typical prediction times (ms scale)
    - **But**: Can skew statistics if predictions are very fast (< 1μs, unlikely)

12. **Report statistics computed on-demand**
    - `compute_summary_statistics()` uses numpy operations on full runtime list
    - O(N log N) for percentile computation
    - **Gotcha**: Slow if called in tight loop (cache result!)

13. **`clear_stats=True` default can lose data**
    - `generate_predictor_report(clear_stats=True)` empties runtime list
    - **Gotcha**: Calling twice returns empty second report
    - **Fix**: Set `clear_stats=False` if you want to accumulate across calls

14. **MLPredictorReport expects both feature and inference times**
    - No ML predictor implementation populates these fields
    - **Consequence**: Currently unused (placeholder for future)
    - **AIDEV-NOTE**: Should MLPredictorReport be removed until ML predictor exists?

### Agent State & Predictions

15. **`Agent.predictions` is mutable list**
    - `predictions: List[PredictedTrajectory]`
    - Can be modified in-place
    - **Gotcha**: Changing agent.predictions affects upstream references
    - **Safety**: Clone if you need independent copy

16. **PredictedTrajectory probability must sum to 1.0**
    - Enforced in `Agent.predictions` setter (in actor_state module)
    - **Symptom**: ValueError if probabilities don't sum to 1.0
    - **Example**: Two predictions with 0.6 and 0.5 → error!
    - **LogFuturePredictor**: Copies probabilities from scenario (already validated)

17. **Waypoints can be None in PredictedTrajectory**
    - `waypoints: List[Optional[WaypointTypes]]`
    - Use `valid_waypoints` property to filter Nones
    - **LogFuturePredictor**: Scenario waypoints are never None (database constraint)

18. **Predictions extend beyond prediction_horizon**
    - TrajectorySampling specifies `num_poses` and `time_horizon`
    - Example: 16 poses over 8s = 0.5s spacing
    - **Gotcha**: Timesteps != num_poses (off-by-one: num_poses includes t=0)
    - **Calculation**: dt = time_horizon / (num_poses - 1)

### Map & Context Usage

19. **Map API in initialization, not input**
    - `PredictorInitialization.map_api` (one-time)
    - NOT in `PredictorInput` (per-timestep)
    - **Rationale**: Map is static, no need to pass every iteration
    - **Usage**: Store in `self._map_api` during `initialize()`

20. **History buffer requires careful indexing**
    - `current_input.history.current_state` → (EgoState, Observation)
    - `current_input.history.observations[-1]` → same as current_state[1]
    - **Gotcha**: `observations[0]` is OLDEST, `observations[-1]` is NEWEST
    - **Usage**: `for obs in reversed(history.observations):` for newest-first

## Performance Considerations

### Runtime Characteristics

**LogFuturePredictor**:
- **Database query**: 1-5ms (SQLite query for tracked objects at iteration)
- **Token matching**: O(N) where N = number of agents (~10-100 typically)
- **Dict construction**: O(N)
- **Total**: ~5-10ms for typical scenarios (50 agents)
- **Bottleneck**: Database I/O (disk read for tracked_objects table)

**Hypothetical ML Predictor**:
- **Feature building**: 5-20ms (depends on history length, map queries)
- **Model inference**: 10-100ms (GPU) or 50-500ms (CPU)
  - Depends on model size, batch size, hardware
- **Post-processing**: 1-5ms (numpy → Waypoint conversion)
- **Total**: ~20-150ms (GPU) or ~60-500ms (CPU)
- **Bottleneck**: Model inference (especially CPU)

### Real-Time Constraints

**Simulation timestep budget** (10Hz = 0.1s = 100ms):
- Observation update: 20-50ms
- **Predictor**: 10-30ms (target budget)
- Planner: 30-50ms
- Controller + metrics: 10-20ms

**Predictor targets**:
- **Median**: < 20ms (leaves headroom for planner)
- **95th percentile**: < 30ms (avoid tail latency)
- **Max**: < 50ms (prevents simulation slowdown)

**AIDEV-NOTE**: Current nuPlan simulation doesn't use predictors, so no real-time pressure. IF integrated, would need optimization.

### Optimization Strategies

1. **Cache scenario queries** (LogFuturePredictor)
   - Scenario tracked objects don't change
   - Cache dict across iterations (rebuild only when iteration changes)
   - **Savings**: ~5ms per iteration (database query avoided)

2. **Lazy prediction assignment**
   - Only populate predictions if planner uses them
   - Check planner config for prediction dependence
   - **Savings**: Skip prediction entirely if unused

3. **Batch ML inference**
   - Predict all agents in single forward pass
   - GPU batching amortizes overhead
   - **Speedup**: 2-5x vs sequential per-agent inference

4. **Reduce prediction horizon**
   - Shorter time_horizon or fewer num_poses
   - Less computation per agent
   - **Trade-off**: Planner has less lookahead

5. **Agent filtering**
   - Only predict for nearby/relevant agents
   - Use radius-based filtering (like IDMAgents)
   - **Savings**: Linear reduction in agent count

6. **Prediction caching**
   - If scenario is deterministic, cache predictions by (iteration, token)
   - Reuse across simulation reruns
   - **Savings**: Entire prediction time (disk I/O only)

### Memory Footprint

**LogFuturePredictor**:
- Scenario reference: Negligible (pointer)
- Runtime list: ~8 bytes/float × iterations (~1 KB for 100 iterations)
- Predictions: ~1 KB per agent per trajectory (waypoints + metadata)
- **Total**: ~10-100 KB per scenario (depends on agent count)

**Hypothetical ML Predictor**:
- Model parameters: 1-100 MB (depends on architecture)
- Feature cache: ~10 KB per agent (history + map features)
- Predictions: Same as LogFuturePredictor
- **Total**: ~10-200 MB (dominated by model size)

## Related Documentation

### Documented Modules (Cross-reference ✅)
- ✅ `nuplan/planning/simulation/observation/CLAUDE.md` - Observation module (IDM/ML agents populate predictions)
- ✅ `nuplan/planning/simulation/trajectory/CLAUDE.md` - PredictedTrajectory, InterpolatedTrajectory
- ✅ `nuplan/common/actor_state/CLAUDE.md` - Agent, PredictedTrajectory, TrackedObjects
- ✅ `nuplan/planning/simulation/planner/CLAUDE.md` - Planners that consume predictions
- ✅ `nuplan/planning/simulation/history/CLAUDE.md` - SimulationHistoryBuffer (prediction input)

### Undocumented Dependencies (Future Documentation ⏳)
- ⏳ `nuplan/planning/scenario_builder/abstract_scenario.md` - Scenario interface (get_tracked_objects_at_iteration)
- ⏳ `nuplan/planning/simulation/simulation_time_controller/` - SimulationIteration

### Related Research Topics
- **Trajectory prediction literature**: Social-LSTM, Trajectron++, Multipath, TNT, DenseTNT
- **Multi-modal prediction**: Probability distributions over futures
- **Goal-conditioned prediction**: Conditioning on route/destination
- **Map-aware prediction**: Using HD maps for context
- **Interaction-aware prediction**: Game theory, inverse RL

## AIDEV Notes

### Critical Design Questions

**AIDEV-QUESTION**: Why is predictor module unused in simulation?
- Observations (IDMAgents, MLAgents) handle prediction internally
- No simulation setup references AbstractPredictor
- Is this intentional separation of concerns or incomplete integration?
- **Hypothesis**: Predictor module designed for research/evaluation, not production simulation

**AIDEV-QUESTION**: Should `observation_type()` be a class method?
- Current: Instance method returning Type
- Alternative: `@classmethod` or class attribute
- **Benefit**: Could validate before instantiation
- **Example**: `if MyPredictor.observation_type() == DetectionsTracks: ...`

**AIDEV-QUESTION**: MLPredictorReport unused - remove or implement?
- No ML predictor populates `feature_building_runtimes` or `inference_runtimes`
- Keep as placeholder for future or remove dead code?
- **Suggestion**: Add TODO comment or create stub MLPredictor class

### Potential Enhancements

**AIDEV-TODO**: Integrate predictor into simulation setup
- Add `predictor: Optional[AbstractPredictor]` to SimulationSetup
- Call predictor after observation update, before planner
- Would enable modular prediction (separate from observation)
- **Benefit**: A/B test different predictors without changing observation

**AIDEV-TODO**: Support multiple prediction modes (multi-modal)
- Current: `List[PredictedTrajectory]` with probabilities
- LogFuturePredictor returns single mode (probability=1.0)
- **Enhancement**: Extract multiple futures from scenario (if annotated)
- **Use case**: Evaluate planner robustness to prediction uncertainty

**AIDEV-TODO**: Add prediction metrics
- Accuracy: ADE (Average Displacement Error), FDE (Final Displacement Error)
- Miss rate: % of trajectories outside error threshold
- **Use case**: Benchmark predictors against ground truth
- **Implementation**: New module `nuplan/planning/metrics/prediction/`

**AIDEV-TODO**: Implement caching for LogFuturePredictor
- Cache `scenario.get_tracked_objects_at_iteration()` results
- Key: iteration index
- **Savings**: ~5ms per iteration (database query avoided)
- **Implementation**:
  ```python
  @functools.lru_cache(maxsize=1000)
  def _get_scenario_agents(self, iteration_index):
      return self._scenario.get_tracked_objects_at_iteration(...)
  ```

### Potential Bugs & Edge Cases

**AIDEV-NOTE**: LogFuturePredictor assumes token uniqueness
- `scenario_agent_dict[token]` assumes no duplicate tokens
- Database constraint ensures this, but worth asserting
- **Check**: Add assertion `assert len(tokens) == len(set(tokens))`

**AIDEV-NOTE**: Missing validation in PredictorInput
- `traffic_light_data` can be None, empty list, or populated
- No validation of list contents (TrafficLightStatusData fields)
- **Risk**: Downstream code may not handle None gracefully
- **Fix**: Add validation in dataclass `__post_init__`

**AIDEV-NOTE**: Race condition in runtime tracking?
- `_compute_predictions_runtimes.append()` not thread-safe
- If multiple threads call `compute_predictions()` concurrently → corruption
- **Risk**: Low (simulation is single-threaded)
- **Fix**: Use `threading.Lock()` if parallelization added

**AIDEV-NOTE**: Report generation can raise if no predictions made
- `compute_summary_statistics()` calls numpy on empty list
- **Symptom**: RuntimeWarning or NaN values
- **Fix**: Check `if not self.compute_predictions_runtimes: return {}`

### Testing Gaps

**AIDEV-NOTE**: No integration tests with simulation
- Unit tests for predictor interface exist
- Missing: End-to-end test with Simulation class
- **Needed**: Verify predictor fits into simulation loop correctly

**AIDEV-NOTE**: No performance regression tests
- Runtime statistics collected but not validated
- **Needed**: Assert 95th percentile < threshold
- **Example**: `assert stats['compute_predictions_runtimes_95_percentile'] < 0.05`

**AIDEV-NOTE**: No multi-scenario tests
- Tests use single MockAbstractScenario
- **Needed**: Test across diverse scenarios (high/low agent count, different cities)

### Documentation Improvements

**AIDEV-TODO**: Add sequence diagram for prediction flow
- Show: Simulation → Observation → Predictor → History → Planner
- Clarify where predictions fit in pipeline
- **Tool**: PlantUML or mermaid.js

**AIDEV-TODO**: Document prediction data format
- What fields are in Waypoint?
- What coordinate frame (global vs ego-relative)?
- What units (meters, seconds)?
- **Location**: Add to PredictedTrajectory docstring

**AIDEV-TODO**: Clarify `requires_scenario` usage
- When should it be True?
- How does simulation validate this?
- **Example**: Oracle predictors (LogFuture) need scenario, ML predictors don't

### Future ML Predictor Implementation Checklist

**AIDEV-TODO**: When implementing ML predictor, ensure:
1. **Model loading**: Load in `initialize()`, not `__init__` (Hydra compatibility)
2. **Feature building**: Reuse planner feature builders (avoid duplication)
3. **Coordinate frames**: Document if ego-relative or global predictions
4. **Device placement**: Handle CPU/GPU correctly (check ModelLoader pattern)
5. **Batching**: Predict all agents in single forward pass (GPU efficiency)
6. **Post-processing**: Convert model output to Waypoint objects
7. **Error handling**: Gracefully handle prediction failures (fallback to constant velocity?)
8. **Runtime tracking**: Use MLPredictorReport for feature/inference breakdown
9. **Validation**: Check prediction horizon matches sampling parameters
10. **Testing**: Add integration test with real scenario

---

**Last Updated**: 2025-11-15
**Status**: Tier 2 documentation complete (Phase 2B: Control & Motion)
**Cross-references**: 6 documented modules, 2 undocumented dependencies
