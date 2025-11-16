# CLAUDE.md - Training Metrics Module (Tier 3)

## ABOUTME
This directory contains training-time metrics for ML planner development. These are **NOT** the same as simulation metrics - they evaluate imitation learning quality during training, not closed-loop safety/performance.

## Directory Purpose

**What it does**: Computes loss-like metrics to monitor ML training quality in real-time

**When it runs**: During PyTorch Lightning training loop (logged to TensorBoard)

**Key distinction**:
- **Training metrics** (here): Open-loop imitation quality (ADE, FDE, heading errors)
- **Simulation metrics** (nuplan/planning/metrics): Closed-loop safety/performance (collisions, comfort, progress)

## Files (4 total)

```
metrics/
├── abstract_training_metric.py    # Base interface for all training metrics
├── planning_metrics.py            # Ego trajectory imitation metrics (4 metrics)
├── agents_imitation_metrics.py    # Multi-agent prediction metrics (4 metrics)
└── __init__.py                    # Empty module init
```

## Core Abstractions

### AbstractTrainingMetric (Base Class)

**Interface contract**:
```python
def compute(predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
    """Returns scalar metric value (already averaged over batch)"""

def get_list_of_required_target_types() -> List[str]:
    """Returns ["trajectory"] or ["agents_trajectory"]"""

def name() -> str:
    """Metric name for TensorBoard logging"""
```

**Key design**: Metrics operate on **feature dictionaries** (TargetsType), not raw tensors

## Metric Implementations

### Planning Metrics (Ego Trajectory)

**Required target**: `"trajectory"` (Trajectory feature)

1. **AverageDisplacementError (ADE)**
   - Average L2 error across all trajectory waypoints
   - Formula: `mean(||pred_xy - target_xy||₂)`
   - Typical range: 0.5-5.0 meters (lower = better imitation)

2. **FinalDisplacementError (FDE)**
   - L2 error at trajectory endpoint only
   - Formula: `mean(||pred_terminal - target_terminal||₂)`
   - Critical for long-horizon planning (8 seconds)

3. **AverageHeadingError**
   - Average angular error (wrapped to [-π, π])
   - Formula: `mean(atan2(sin(Δθ), cos(Δθ)))`
   - Wrapping prevents discontinuity at ±π

4. **FinalHeadingError**
   - Angular error at endpoint only
   - Important for parking/stop scenarios

### Agents Imitation Metrics (Multi-Agent Prediction)

**Required target**: `"agents_trajectory"` (AgentsTrajectories feature)

1. **AgentsAverageDisplacementError**
   - ADE averaged over **all agents** in scene
   - Per-sample loop due to variable agent counts
   - Inefficient but flexible for ragged batches

2. **AgentsFinalDisplacementError**
   - FDE averaged over all agents
   - Useful for interactive prediction models

3. **AgentsAverageHeadingError**
   - Average heading error for all agents
   - Same wrapping logic as ego metrics

4. **AgentsFinalHeadingError**
   - Final heading error for all agents

## Key Dependencies

**Internal (nuPlan)**:
- `nuplan.planning.training.modeling.types.TargetsType` - Feature dictionary type alias
- `nuplan.planning.training.preprocessing.features.trajectory.Trajectory` - Ego trajectory feature
- `nuplan.planning.training.preprocessing.features.agents_trajectories.AgentsTrajectories` - Multi-agent feature

**External (PyTorch)**:
- `torch.Tensor` - Return type (scalar)
- `torch.norm` - L2 distance computation
- `torch.atan2`, `torch.sin`, `torch.cos` - Angle wrapping

## Data Flow

```
Training Loop (PyTorch Lightning)
    ↓
Model Forward Pass
    ↓ predictions: TargetsType
    ↓ targets: TargetsType
    ↓
Metric.compute(predictions, targets)
    ↓ Extract features by key
    ↓ Trajectory.xy, .heading, .terminal_position
    ↓ Compute error (L2, angular)
    ↓
Return: torch.Tensor (scalar)
    ↓
Logger → TensorBoard
```

## Usage Patterns

### Registering Metrics in Lightning Module

```python
from nuplan.planning.training.modeling.metrics.planning_metrics import (
    AverageDisplacementError,
    FinalDisplacementError,
)

class RasterModel(LightningModule):
    def __init__(self):
        self.metrics = [
            AverageDisplacementError(),
            FinalDisplacementError(),
        ]

    def validation_step(self, batch, batch_idx):
        predictions = self.forward(features)
        targets = batch["targets"]

        for metric in self.metrics:
            value = metric.compute(predictions, targets)
            self.log(metric.name(), value)
```

### Hydra Configuration

Metrics are typically configured in `config/training/default_training.yaml`:
```yaml
objective:
  aggregate_trajectory_metrics:
    - avg_displacement_error
    - final_displacement_error
```

## Enhanced Hydra Configuration Patterns

### Metric Configuration Structure

**Individual metric configs** (`config/training/training_metric/*.yaml`):
```yaml
# config/training/training_metric/avg_displacement_error.yaml
avg_displacement_error:
  _target_: nuplan.planning.training.modeling.metrics.planning_metrics.AverageDisplacementError
  _convert_: 'all'
```

**Key patterns**:
- `_target_`: Full Python path to metric class (Hydra instantiates this)
- `_convert_`: Set to `'all'` to convert config DictConfig → native Python types
- Config name (e.g., `avg_displacement_error`) must match file name for discovery

### Metric Composition in Experiments

**Experiment configs** (`experiments/training/training_raster_model.yaml`):
```yaml
# @package _global_
job_name: raster_model
py_func: train
objective_aggregate_mode: mean

defaults:
  - override /training_metric:
      - avg_displacement_error
      - avg_heading_error
      - final_displacement_error
      - final_heading_error
```

**Composition logic**:
1. Hydra loads each named metric config from `training_metric/` directory
2. Instantiates classes using `_target_` path
3. Passes list to `LightningModuleWrapper.__init__(metrics=[...])`
4. Each metric logged separately to TensorBoard

### CLI Override Patterns

**Selecting specific metrics** at runtime:
```bash
# Override entire metric list
uv run python nuplan/planning/script/run_training.py \
    experiment=training_raster_model \
    training_metric='[avg_displacement_error, final_displacement_error]'

# Add single metric to existing list
uv run python nuplan/planning/script/run_training.py \
    experiment=training_raster_model \
    +training_metric=agents_avg_displacement_error

# Disable all metrics (empty list)
uv run python nuplan/planning/script/run_training.py \
    experiment=training_raster_model \
    training_metric='[]'
```

**Common use cases**:
- **Debugging**: Disable metrics to isolate loss issues (`training_metric='[]'`)
- **Fast iteration**: Only use cheap metrics during development
- **Agent models**: Add `agents_*` metrics for multi-agent prediction
- **Ablation studies**: Test impact of different error signals

### Config Discovery Paths

Hydra searches for metrics in:
1. `nuplan/planning/script/config/training/training_metric/` (default)
2. Experiment searchpath: `script/experiments/` (custom experiments)
3. Common configs: `script/config/common/` (shared across training/simulation)

**AIDEV-NOTE**: Config file name must match metric name string for Hydra discovery

## Config Internals & Hydra Instantiation

### The `_convert_: 'all'` Pattern

**What it does**: Converts Hydra's `DictConfig` objects to native Python types before instantiation

**Without `_convert_`**:
```python
# Hydra passes DictConfig wrapper objects
avg_displacement_error:
  _target_: nuplan.planning.training.modeling.metrics.planning_metrics.AverageDisplacementError
  some_param: 10  # This would be DictConfig({'some_param': 10})
```

**With `_convert_: 'all'`**:
```python
avg_displacement_error:
  _target_: nuplan.planning.training.modeling.metrics.planning_metrics.AverageDisplacementError
  _convert_: 'all'
  some_param: 10  # Converted to native int: 10
```

**Why this matters**:
- Metrics expect native Python types (int, float, str, list)
- DictConfig objects have different behavior (lazy evaluation, interpolation)
- `_convert_: 'all'` ensures clean type boundaries

**Current metrics**: No constructor parameters, so `_convert_` is defensive (future-proof)

### Hydra Instantiation Mechanism

**Step-by-step process**:
```python
# 1. Hydra reads YAML config
config = {
    '_target_': 'nuplan.planning.training.modeling.metrics.planning_metrics.AverageDisplacementError',
    '_convert_': 'all'
}

# 2. Hydra imports the class
from nuplan.planning.training.modeling.metrics.planning_metrics import AverageDisplacementError

# 3. Hydra extracts constructor args (none for current metrics)
constructor_args = {}  # No params in config beyond _target_ and _convert_

# 4. Hydra instantiates the class
metric_instance = AverageDisplacementError(**constructor_args)

# 5. Instance added to metrics list
metrics.append(metric_instance)
```

**AIDEV-NOTE**: `_target_` and `_convert_` are reserved Hydra keywords, never passed to constructor

### Config Dataclass Mapping (Future Pattern)

**If metrics had parameters** (hypothetical example):
```yaml
# config/training/training_metric/weighted_ade.yaml
weighted_ade:
  _target_: custom.metrics.WeightedADE
  _convert_: 'all'
  temporal_weights: [1.0, 1.2, 1.5, 2.0]  # Weight later timesteps more
  normalize: true
```

**Corresponding Python class**:
```python
class WeightedADE(AbstractTrainingMetric):
    def __init__(self, temporal_weights: List[float], normalize: bool = True):
        self.temporal_weights = torch.tensor(temporal_weights)
        self.normalize = normalize

    def compute(self, predictions, targets):
        errors = compute_l2_errors(predictions, targets)
        weighted = errors * self.temporal_weights
        return weighted.mean()
```

**Mapping rules**:
- YAML keys → Python `__init__` parameter names (exact match required)
- YAML types → Python types (int → int, list → List, etc.)
- Missing params → Use Python defaults if provided
- Extra YAML params → Error (unexpected keyword argument)

### Custom Metric Registration

**To add a new custom metric to Hydra**:

1. **Implement the metric class**:
```python
# custom_metrics/my_metric.py
from nuplan.planning.training.modeling.metrics.planning_metrics import AbstractTrainingMetric

class MyCustomMetric(AbstractTrainingMetric):
    def name(self) -> str:
        return "my_custom_metric"

    def get_list_of_required_target_types(self) -> List[str]:
        return ["trajectory"]

    def compute(self, predictions, targets):
        # Your custom logic here
        return torch.tensor(0.0)
```

2. **Create Hydra config**:
```yaml
# config/training/training_metric/my_custom_metric.yaml
my_custom_metric:
  _target_: custom_metrics.my_metric.MyCustomMetric
  _convert_: 'all'
```

3. **Use in experiment**:
```yaml
# experiments/training/my_experiment.yaml
defaults:
  - override /training_metric:
      - avg_displacement_error
      - my_custom_metric  # Your custom metric!
```

**AIDEV-NOTE**: Config file name (`my_custom_metric.yaml`) must match metric list entry

### Hydra Debug Commands

**Inspect resolved config** before training:
```bash
# See final composed config
uv run python nuplan/planning/script/run_training.py \
    experiment=training_raster_model \
    --cfg job --resolve

# See metric instantiation config
uv run python nuplan/planning/script/run_training.py \
    experiment=training_raster_model \
    --cfg job --resolve | grep -A 10 "training_metric"
```

**Common config errors**:
- `MissingMandatoryValue`: Missing required parameter
- `ConfigCompositionException`: Typo in metric name or wrong path
- `InstantiationException`: Constructor parameter mismatch
- `ImportError`: `_target_` path incorrect

## Metric Validation System

### LightningModuleWrapper Initialization Validation

**Validation logic** (`lightning_module_wrapper.py:60-67`):
```python
# Validate metrics objectives and model
model_targets = {builder.get_feature_unique_name() for builder in model.get_list_of_computed_target()}
for objective in self.objectives:
    for feature in objective.get_list_of_required_target_types():
        assert feature in model_targets, f"Objective target: \"{feature}\" is not in model computed targets!"
for metric in self.metrics:
    for feature in metric.get_list_of_required_target_types():
        assert feature in model_targets, f"Metric target: \"{feature}\" is not in model computed targets!"
```

**What this checks**:
1. **Model compatibility**: Model must produce features that metrics require
2. **Early failure**: Catches config mismatches before training starts
3. **Clear errors**: Assertion message shows exact missing feature

### Validation Failure Examples

**Example 1: Missing trajectory output**
```python
# Model only outputs agents_trajectory
model.get_list_of_computed_target() = [AgentsTrajectoriesBuilder()]

# But metric requires trajectory
metrics = [AverageDisplacementError()]  # Requires "trajectory"

# Error at initialization:
AssertionError: Metric target: "trajectory" is not in model computed targets!
```

**Example 2: Agent metric without agent predictions**
```python
# Raster model only outputs ego trajectory
model_targets = {"trajectory"}

# But you added agent metric
metrics = [AgentsAverageDisplacementError()]  # Requires "agents_trajectory"

# Error at initialization:
AssertionError: Metric target: "agents_trajectory" is not in model computed targets!
```

### Metric Name Uniqueness

**No explicit uniqueness check** in current implementation!

**Potential collision scenario**:
```python
metrics = [
    AverageDisplacementError(),  # name() = "avg_displacement_error"
    AverageDisplacementError(),  # name() = "avg_displacement_error" (duplicate!)
]
```

**What happens**:
- Both metrics compute independently
- Both log to TensorBoard with same key
- **TensorBoard overwrites** the first metric's value with the second
- No error, just silent data loss in logs

**Workaround**: Override `name()` method for custom metric instances:
```python
class CustomADE(AverageDisplacementError):
    def name(self) -> str:
        return "custom_avg_displacement_error"
```

### Feature Type Requirements Table

| Metric Class | Required Target Type | Feature Class | Model Must Output |
|--------------|---------------------|---------------|-------------------|
| AverageDisplacementError | `"trajectory"` | Trajectory | TrajectoryBuilder |
| FinalDisplacementError | `"trajectory"` | Trajectory | TrajectoryBuilder |
| AverageHeadingError | `"trajectory"` | Trajectory | TrajectoryBuilder |
| FinalHeadingError | `"trajectory"` | Trajectory | TrajectoryBuilder |
| AgentsAverageDisplacementError | `"agents_trajectory"` | AgentsTrajectories | AgentsTrajectoriesBuilder |
| AgentsFinalDisplacementError | `"agents_trajectory"` | AgentsTrajectories | AgentsTrajectoriesBuilder |
| AgentsAverageHeadingError | `"agents_trajectory"` | AgentsTrajectories | AgentsTrajectoriesBuilder |
| AgentsFinalHeadingError | `"agents_trajectory"` | AgentsTrajectories | AgentsTrajectoriesBuilder |

**AIDEV-NOTE**: Validation happens before any training data is loaded - fails fast!

## 10+ Critical Gotchas

### 1. Training vs Simulation Metrics Confusion
**Problem**: New users expect training metrics to correlate perfectly with simulation performance
**Reality**: Low ADE ≠ good closed-loop planner (distribution shift, compounding errors)
**Fix**: Always validate with closed-loop simulation (`just simulate-ml`)

### 2. Angle Wrapping is Essential
**Problem**: Direct subtraction `pred_heading - target_heading` breaks at ±π boundary
**Example**: pred=179°, target=-179° → error=358° instead of 2°
**Fix**: Always use `atan2(sin(Δθ), cos(Δθ))` for angular errors (lines 119, 143, 157, 188)

### 3. Feature Dictionary Key Mismatches
**Problem**: Metrics hardcode `"trajectory"` or `"agents_trajectory"` keys
**Error**: `KeyError: 'trajectory'` if model doesn't output this feature
**Fix**: Ensure model's `get_list_of_computed_target_types()` matches metric requirements

### 4. Per-Sample Loop Inefficiency in Agent Metrics
**Problem**: Lines 45-52 (agents_imitation_metrics.py) loop over batch instead of vectorizing
**Impact**: 5-10x slower than necessary for large batches
**Reason**: Variable agent counts create ragged tensors (hard to vectorize)
**Workaround**: Pad to max agents, use masking (not implemented)

### 5. Tensor Device Mismatch
**Problem**: Metrics don't explicitly move tensors to same device
**Error**: `RuntimeError: Expected all tensors to be on the same device`
**Cause**: predictions on GPU, targets on CPU (rare but possible)
**Fix**: Features should already be on correct device from data loader

### 6. Missing `.mean()` Over Batch Dimension
**Problem**: Forgetting batch aggregation returns tensor of shape [B] not scalar
**Impact**: Logger expects scalar, gets vector
**Fix**: All metrics correctly call `.mean()` at end (lines 44, 81, 119, 145)

### 7. Terminal Position Property Assumption
**Problem**: `Trajectory.terminal_position` assumes property exists
**Fragility**: If Trajectory class changes, metric breaks
**Current**: Property correctly implemented in Trajectory feature
**Watch**: Refactors to Trajectory class

### 8. No NaN/Inf Handling
**Problem**: Metrics don't check for invalid values (NaN, Inf) in predictions
**Impact**: One bad prediction can corrupt entire metric
**Cause**: Unstable training (exploding gradients, bad initialization)
**Fix**: Add `.nan_to_num()` or explicit checks if training unstable

### 9. Metric Name Collisions
**Problem**: Default names like `'avg_displacement_error'` can collide if multiple metrics registered
**Example**: Both ego and agents metrics use similar names
**Fix**: Use descriptive names: `'ego_ade'` vs `'agents_ade'` (override in __init__)

### 10. No Temporal Weighting
**Problem**: ADE treats all timesteps equally (errors at t=0 same weight as t=8s)
**Reality**: Long-horizon errors matter more (harder to predict)
**Improvement**: Add exponential weighting by timestep (not implemented)
**Workaround**: Use FDE to emphasize endpoint

### 11. Heading Error Units (Radians vs Degrees)
**Problem**: `torch.atan2` returns radians, but logs/plots may expect degrees
**Confusion**: Error of 0.1 looks tiny (but it's 5.7° - significant!)
**Fix**: Document units clearly, consider logging both radians and degrees

### 12. Batch Size Zero Edge Case
**Problem**: Empty batch (batch_size=0) causes division by zero in `.mean()`
**Likelihood**: Rare, but possible with filtering or small validation sets
**Error**: `RuntimeError: mean(): input tensor must have at least one element`
**Fix**: Add guard `if batch_size == 0: return torch.tensor(0.0)`

### 13. Feature Type Assumptions
**Problem**: No runtime type checking - assumes dict values are correct feature classes
**Error**: Cryptic AttributeError if wrong feature type passed
**Example**: `predictions["trajectory"]` is a Tensor, not Trajectory object
**Fix**: Add `isinstance()` checks or better type hints

## Performance Characteristics

**Ego metrics** (planning_metrics.py):
- Fully vectorized over batch dimension
- Fast: O(B × T) where B=batch_size, T=trajectory_length
- Typical: <1ms for B=32, T=16

**Agent metrics** (agents_imitation_metrics.py):
- Per-sample loop (inefficient)
- Slow: O(B × N × T) where N=avg_agents_per_scene
- Typical: 5-20ms for B=32, N=20, T=16
- **AIDEV-NOTE**: Vectorization opportunity for speedup

## Validation Logic

**Required target types check**:
```python
# Lightning module should verify before calling compute
for metric in self.metrics:
    required = metric.get_list_of_required_target_types()
    for target_type in required:
        assert target_type in targets, f"Missing {target_type} for {metric.name()}"
```

**Current**: This check is **implicit** (assumes correct feature builders configured)

## Testing Considerations

**Unit test coverage**: Minimal (no dedicated test file in repo)

**What to test**:
1. Angle wrapping correctness (±π boundary)
2. Zero/one-element batch edge cases
3. Perfect predictions → metric = 0.0
4. Known geometry → known metric value
5. Device consistency (CPU/GPU)

**Example test**:
```python
def test_angle_wrapping():
    pred = Trajectory(heading=torch.tensor([3.14]))  # ~π
    target = Trajectory(heading=torch.tensor([-3.14]))  # ~-π

    metric = AverageHeadingError()
    error = metric.compute({"trajectory": pred}, {"trajectory": target})

    assert error < 0.01  # Should be ~0, not ~2π
```

## Related Systems

**Upstream** (produce inputs):
- `nuplan/planning/training/modeling/models/` - Generate predictions
- `nuplan/planning/training/preprocessing/features/` - Define Trajectory, AgentsTrajectories classes
- `nuplan/planning/training/data_loader/` - Batch targets from dataset

**Downstream** (consume outputs):
- `nuplan/planning/training/callbacks/` - Log metrics to TensorBoard
- PyTorch Lightning - Aggregate metrics across epochs

**Parallel** (similar but different):
- `nuplan/planning/metrics/` - Closed-loop simulation metrics (NOT imitation metrics)

## Feature Builder Integration

### How Metrics Interface with Builders

**The dependency chain**:
```
Feature Builders (preprocessing/)
    ↓ Define target types
    ↓
Model (TorchModuleWrapper)
    ↓ Declares computed targets
    ↓
Metrics (this module)
    ↓ Declare required targets
    ↓
Validation (LightningModuleWrapper)
    ↓ Ensures compatibility
```

### `get_list_of_required_target_types()` Usage

**In metric class** (planning_metrics.py):
```python
class AverageDisplacementError(AbstractTrainingMetric):
    @staticmethod
    def get_list_of_required_target_types() -> List[str]:
        """Returns feature types needed from targets dict"""
        return ["trajectory"]  # Must match builder's unique name
```

**In validation logic** (lightning_module_wrapper.py:60-67):
```python
model_targets = {builder.get_feature_unique_name() for builder in model.get_list_of_computed_target()}
# Example: model_targets = {"trajectory", "generic_agents"}

for metric in self.metrics:
    for feature in metric.get_list_of_required_target_types():
        # Check: "trajectory" in {"trajectory", "generic_agents"} → True ✓
        assert feature in model_targets, f"Metric target: \"{feature}\" is not in model computed targets!"
```

**Why static method?**: Called before instantiation during config validation

### Target Type Requirements by Metric

**Ego trajectory metrics** (require `"trajectory"`):
```python
# These 4 metrics all require the same builder output
required_builders = [TrajectoryBuilder()]

metrics = [
    AverageDisplacementError(),       # Needs trajectory.xy, trajectory.data
    FinalDisplacementError(),         # Needs trajectory.terminal_position
    AverageHeadingError(),            # Needs trajectory.heading
    FinalHeadingError(),              # Needs trajectory.heading
]
```

**Agent prediction metrics** (require `"agents_trajectory"`):
```python
# These 4 metrics all require multi-agent builder output
required_builders = [AgentsTrajectoriesBuilder()]

metrics = [
    AgentsAverageDisplacementError(),
    AgentsFinalDisplacementError(),
    AgentsAverageHeadingError(),
    AgentsFinalHeadingError(),
]
```

### Feature Builder Compatibility Matrix

| Builder Class | Unique Name | Compatible Metrics |
|--------------|-------------|-------------------|
| TrajectoryBuilder | `"trajectory"` | AverageDisplacementError<br>FinalDisplacementError<br>AverageHeadingError<br>FinalHeadingError |
| AgentsTrajectoriesBuilder | `"agents_trajectory"` | AgentsAverageDisplacementError<br>AgentsFinalDisplacementError<br>AgentsAverageHeadingError<br>AgentsFinalHeadingError |

**AIDEV-NOTE**: String matching is case-sensitive and exact - typos cause validation failure!

### Builder Output Structure

**TrajectoryBuilder output** (for ego metrics):
```python
trajectory_feature = Trajectory(
    data=torch.tensor([...]),  # Shape: [B, T, state_dim]
)

# Properties available to metrics:
trajectory_feature.xy              # [B, T, 2] - positions
trajectory_feature.heading         # [B, T] - orientations
trajectory_feature.terminal_position  # [B, 2] - final position
```

**AgentsTrajectoriesBuilder output** (for agent metrics):
```python
agents_feature = AgentsTrajectories(
    data=[
        # Per-sample list (ragged)
        {
            'agent_1': torch.tensor([...]),  # [T, state_dim]
            'agent_2': torch.tensor([...]),
        },
        # Next sample...
    ]
)

# Access pattern in metrics:
for sample in agents_feature.data:
    for agent_id, agent_trajectory in sample.items():
        # Compute per-agent error
```

### Custom Builder Requirements Example

**Hypothetical: Velocity error metric**:
```python
class VelocityError(AbstractTrainingMetric):
    @staticmethod
    def get_list_of_required_target_types() -> List[str]:
        return ["trajectory"]  # Reuse existing trajectory builder

    def compute(self, predictions, targets):
        pred_traj = predictions["trajectory"]
        tgt_traj = targets["trajectory"]

        # Compute velocity from positions
        pred_vel = torch.diff(pred_traj.xy, dim=1)
        tgt_vel = torch.diff(tgt_traj.xy, dim=1)

        return torch.norm(pred_vel - tgt_vel, dim=-1).mean()
```

**Key insight**: Most metrics can reuse existing builders by accessing different properties!

### Debugging Builder Mismatches

**Common error**:
```
AssertionError: Metric target: "trajectory" is not in model computed targets!
```

**Debugging steps**:
```python
# 1. Check what model actually outputs
print([b.get_feature_unique_name() for b in model.get_list_of_computed_target()])
# Example output: ['generic_agents', 'vector_set_map']

# 2. Check what metric needs
print(AverageDisplacementError.get_list_of_required_target_types())
# Output: ['trajectory']

# 3. Fix: Add TrajectoryBuilder to model
model._computed_target_builders.append(TrajectoryBuilder())
```

**Or fix in model config**:
```yaml
# config/model/raster_model.yaml
model:
  _target_: nuplan.planning.training.modeling.models.RasterModel
  feature_builders:
    - trajectory_builder  # This creates TrajectoryBuilder
    - generic_agents_builder
```

### Multi-Builder Metrics (Future Pattern)

**Hypothetical: Metric requiring multiple features**:
```python
class InteractionMetric(AbstractTrainingMetric):
    @staticmethod
    def get_list_of_required_target_types() -> List[str]:
        return ["trajectory", "agents_trajectory"]  # Needs BOTH!

    def compute(self, predictions, targets):
        ego = predictions["trajectory"]
        agents = predictions["agents_trajectory"]
        # Compute interaction-aware error
        return interaction_error(ego, agents)
```

**Validation**: All required types must be in model outputs

## Migration Notes

**Why separate from simulation metrics**:
- Training metrics = fast, approximate, open-loop
- Simulation metrics = slow, comprehensive, closed-loop
- Different use cases require different implementations

**Historical**: Originally mixed together, separated for clarity

## Improvement Opportunities

**High priority**:
1. Vectorize agent metrics (5-10x speedup)
2. Add temporal weighting to ADE
3. Add NaN/Inf guards for robustness

**Medium priority**:
4. Per-timestep error breakdown (early vs late trajectory)
5. Weighted metrics by scenario difficulty
6. Covariance-aware metrics (if probabilistic predictions)

**Low priority**:
7. GPU kernel fusion for heading wrapping
8. FP16 support for faster logging

## Configuration Integration

**Hydra path**: `config/training/objective/`

**Example objective config**:
```yaml
# config/training/objective/imitation_objective.yaml
aggregate_trajectory_metrics:
  - avg_displacement_error
  - final_displacement_error
  - avg_heading_error

agents_imitation_metrics:  # Optional
  - agents_avg_displacement_error
```

**Registration**: Metrics instantiated by name string → must match `name()` method

## Debugging Tips

**Metric always zero**:
- Check predictions == targets (not updating model?)
- Check feature extraction (wrong dictionary key?)

**Metric always NaN**:
- Gradient explosion (check loss)
- Invalid predictions (Inf positions)
- Device mismatch (rare)

**Metric unexpectedly high**:
- Units confusion (meters vs pixels, radians vs degrees)
- Forgot angle wrapping (heading errors)
- Coordinate frame mismatch (global vs ego)

**Monitor with TensorBoard**:
```bash
just tensorboard  # localhost:6010
# Look for: train/*, val/*
# Expect: Decreasing trend over epochs
```

## TensorBoard Visualization Details

### Metric Logging Hierarchy

**TensorBoard structure**:
```
Scalars/
├── train/
│   ├── loss                           # Aggregate objective loss
│   ├── avg_displacement_error         # Ego ADE
│   ├── final_displacement_error       # Ego FDE
│   ├── avg_heading_error              # Ego heading ADE
│   └── final_heading_error            # Ego heading FDE
├── val/
│   ├── loss
│   ├── avg_displacement_error
│   ├── final_displacement_error
│   ├── avg_heading_error
│   └── final_heading_error
└── learning_rate                      # LR scheduler tracking
```

**Logging logic** (`lightning_module_wrapper.py:_log_step`):
- Prefix `train/` for training steps
- Prefix `val/` for validation steps
- Metric name comes from `metric.name()` method
- Values logged every batch (smoothed in TensorBoard UI)

### Typical Metric Curves (Visual Description)

**Healthy training pattern**:
```
ADE (meters)
  5.0 ┤╮
      │ ╰╮
  3.0 ┤   ╰╮                          Training curve (noisy)
      │     ╰─╮
  1.0 ┤        ╰────────────          Validation curve (smooth)
      │
  0.5 ┤
      └────────────────────────> Epochs
      0    5         10        20
```

**What to expect**:
1. **Initial drop**: Steep decrease in first 1-3 epochs (model learns basic patterns)
2. **Plateau**: Gradual improvement after epoch 5-10 (refinement phase)
3. **Noise**: Training metrics jitter (batch variance), validation is smooth
4. **Train/val gap**: Training slightly better than validation (expected)

**Red flags**:
- **Divergence**: Train ↓, val ↑ after epoch 5 = overfitting
- **Stagnation**: No change after epoch 3 = learning rate too low or bad initialization
- **Explosion**: Metrics → Inf/NaN = gradient explosion or bad data
- **Negative values**: Impossible for L2 metrics = bug in computation

### Example TensorBoard Session

**Terminal output**:
```bash
$ just tensorboard
# TensorBoard 2.11.0 at http://localhost:6010/ (Press CTRL+C to quit)
```

**UI navigation**:
1. **Scalars tab** → Default view for metrics
2. **Left panel** → Filter by regex: `avg_displacement.*` to see only ADE metrics
3. **Smoothing slider** → Adjust to reduce noise (default 0.6)
4. **Download data** → CSV export for custom plotting

**Multi-run comparison**:
- Different experiment runs appear as separate curves
- Color-coded by run name (e.g., `raster_model_run1`, `raster_model_run2`)
- Useful for hyperparameter sweeps (batch size, learning rate, etc.)

### Metric Units in TensorBoard

| Metric | Unit | Typical Range | Good Value |
|--------|------|---------------|------------|
| avg_displacement_error | meters | 0.5 - 5.0 | < 1.0 |
| final_displacement_error | meters | 1.0 - 10.0 | < 2.0 |
| avg_heading_error | radians | 0.05 - 0.5 | < 0.1 (5.7°) |
| final_heading_error | radians | 0.1 - 1.0 | < 0.2 (11.5°) |

**AIDEV-NOTE**: Heading errors in radians look small numerically - convert to degrees for intuition!

### Custom TensorBoard Logging

**Adding custom metric visualizations**:
```python
# In your custom metric class
def compute(self, predictions, targets):
    error = self._compute_error(predictions, targets)

    # Log additional debugging info (not returned as metric value)
    if hasattr(self, 'logger'):
        self.logger.experiment.add_histogram(
            f"{self.name()}_distribution",
            error,  # Full tensor before .mean()
            global_step=self.logger.global_step
        )

    return error.mean()  # Return scalar for main metric
```

**Useful additions**:
- Histograms: Error distribution across batch
- Images: Trajectory visualizations (predicted vs ground truth)
- Text: Failure case scenario IDs

## Summary

This module provides **8 lightweight imitation metrics** for monitoring ML training quality:
- **4 ego metrics**: Position + heading, average + final
- **4 agent metrics**: Same structure, multi-agent prediction

**Critical points**:
- Metrics are **NOT** closed-loop safety metrics (common confusion)
- Angle wrapping is **essential** for heading errors
- Agent metrics have **vectorization opportunity** (current: per-sample loop)
- All metrics return **scalars** (batch-averaged)

**When to use**:
- Real-time training monitoring (TensorBoard)
- Hyperparameter tuning (early stopping)
- Sanity checks (metrics → 0 with perfect imitation)

**When NOT to use**:
- Final model evaluation (use closed-loop simulation instead)
- Safety validation (use collision metrics)
- Comfort assessment (use jerk/acceleration metrics)
