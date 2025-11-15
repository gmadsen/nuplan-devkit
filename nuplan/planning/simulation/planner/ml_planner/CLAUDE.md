# ML Planner Module

## Purpose & Key Abstractions

**THE bridge between trained PyTorch Lightning models and nuPlan's simulation framework.** This module implements `MLPlanner` - the concrete planner that wraps neural network models for learned autonomous driving. It handles the complete inference pipeline: extracting features from observations, running model inference, and decoding predicted trajectories. This is how you deploy trained ML models into closed-loop simulation for evaluation.

**Key abstractions:**
- **MLPlanner** - Wraps `TorchModuleWrapper` models to implement `AbstractPlanner` interface
- **ModelLoader** - Manages PyTorch device setup, feature extraction, and model inference
- **transform_utils** - Converts relative trajectory predictions to absolute `EgoState` sequences with proper velocities/accelerations

## Architecture & Design Patterns

### 1. Adapter Pattern: Model → Planner Bridge
`MLPlanner` adapts PyTorch Lightning models (training interface) to `AbstractPlanner` (simulation interface):

```
Training World                    Simulation World
───────────────                  ─────────────────
TorchModuleWrapper  ─────────>   AbstractPlanner
  ├─ forward()                     ├─ compute_planner_trajectory()
  ├─ feature_builders              ├─ observation_type()
  └─ target_builders               └─ initialize()
       │
       └─────────> MLPlanner (bridges both worlds)
                    ├─ ModelLoader (feature extraction)
                    └─ transform_utils (trajectory decoding)
```

### 2. Two-Phase Feature Pipeline
**Phase 1: Feature Building** (ModelLoader.build_features)
```
PlannerInput → feature_builders → dict[str, AbstractModelFeature]
                                           ↓
                                   .to_feature_tensor()
                                           ↓
                                   .to_device(cuda/cpu)
                                           ↓
                                   .collate([feature])  # Add batch dimension
                                           ↓
                                   FeaturesType (ready for model)
```

**Phase 2: Model Inference** (ModelLoader.infer)
```
FeaturesType → model.forward() → TargetsType (dict[str, AbstractModelFeature])
                                        ↓
                                  Extract 'trajectory' key
                                        ↓
                                  Trajectory.data (torch.Tensor)
                                        ↓
                                  .cpu().detach().numpy()[0]  # Remove batch dim
                                        ↓
                                  numpy array [num_poses, 3] (x, y, heading)
```

### 3. Coordinate Frame Transformations
Models predict **relative poses** (ego-centric frame), but simulation requires **absolute states** (global frame):

```
Model Output                Transform Pipeline                  Simulation Input
────────────                ───────────────────                 ─────────────────
Relative poses    →    relative_to_absolute_poses()    →    Absolute StateSE2 poses
[Δx, Δy, Δθ]                      ↓                           [x_global, y_global, θ_global]
                     _get_velocity_and_acceleration()
                     (numerical differentiation)
                                  ↓
                          EgoState construction                List[EgoState]
                     (pose + velocity + accel)          →     InterpolatedTrajectory
```

### 4. Lazy Initialization Pattern
```python
# Construction: Model loaded but not initialized
planner = MLPlanner(model)  # model_loader._initialized = False

# Simulation calls initialize() before first step
planner.initialize(initialization)
    ↓
model_loader.initialize()
    ↓ _initialize_torch()
    torch.set_grad_enabled(False)  # Inference only
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ↓ _initialize_model()
    self._model.eval()
    self._model = self._model.to(self.device)
```

### 5. Performance Monitoring Decorator Pattern
`MLPlanner` extends `AbstractPlanner`'s timing with feature-building and inference breakdowns:
```python
# AbstractPlanner tracks total time
compute_trajectory()  # Wrapped with timing
    ↓
compute_planner_trajectory()  # User implementation
    ├─ Feature building (timed separately)
    └─ Model inference (timed separately)
```

Result: `MLPlannerReport` provides 3 time series:
- `compute_trajectory_runtimes` - Total planning time (inherited)
- `feature_building_runtimes` - Feature extraction time
- `inference_runtimes` - Model forward pass time

## Dependencies

### Internal nuPlan (Documented ✅)
- `nuplan.planning.simulation.planner.abstract_planner` - Base planner interface
  - `AbstractPlanner`, `PlannerInput`, `PlannerInitialization`
- `nuplan.planning.simulation.trajectory.interpolated_trajectory` - Output trajectory format
- `nuplan.common.actor_state.ego_state` - EgoState representation
- `nuplan.common.actor_state.state_representation` - StateSE2, StateVector2D, TimePoint
- `nuplan.common.geometry.convert` - `relative_to_absolute_poses()` coordinate transform

### Internal nuPlan (Undocumented - Session 5-6 ⏳)
- `nuplan.planning.training.modeling.torch_module_wrapper` - Base model wrapper
  - Provides `future_trajectory_sampling`, `feature_builders`, `target_builders`
- `nuplan.planning.training.modeling.types` - Type definitions
  - `FeaturesType = Dict[str, AbstractModelFeature]`
  - `TargetsType = Dict[str, AbstractModelFeature]`
- `nuplan.planning.training.preprocessing.features.trajectory` - Trajectory feature
  - Wraps model output tensor [batch, num_poses, 3]
- `nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder`
  - `AbstractFeatureBuilder` - Feature extraction interface
  - Used to build features from `PlannerInput` + `PlannerInitialization`
- `nuplan.planning.training.modeling.lightning_module_wrapper`
  - `LightningModuleWrapper.load_from_checkpoint()` - Checkpoint loading
  - See `planner_builder.py:26-28` for usage

### External Dependencies
- **PyTorch** - Model inference, device management, tensor operations
- **NumPy** - Trajectory array manipulation, type hints (`npt.NDArray`)
- **SciPy** - `scipy.interpolate.interp1d` for velocity/acceleration estimation
- **PyTorch Lightning** - Checkpoint loading (implicit via LightningModuleWrapper)

## Dependents (Who Uses Us)

- `nuplan.planning.script.builders.planner_builder` - Instantiates MLPlanner from config
  - Loads checkpoint via `LightningModuleWrapper.load_from_checkpoint()`
  - Extracts model and passes to MLPlanner constructor
- `nuplan.planning.script.run_simulation.py` - Main simulation script
- Hydra config: `nuplan/planning/script/config/simulation/planner/ml_planner.yaml`

## Critical Files (Prioritized)

1. **`ml_planner.py`** (115 lines) - **MUST READ FIRST!**
   - MLPlanner class implementing AbstractPlanner
   - Feature extraction → inference → trajectory decoding pipeline
   - Performance tracking (feature building + inference)

2. **`model_loader.py`** (75 lines) - **Core inference engine**
   - PyTorch initialization (device, eval mode, no_grad)
   - Feature building from PlannerInput via feature_builders
   - Model forward pass wrapper

3. **`transform_utils.py`** (201 lines) - **Coordinate transformation math**
   - `transform_predictions_to_states()` - Main entry point
   - Relative → absolute pose conversion
   - Velocity/acceleration numerical differentiation
   - Global XY → ego-centric DS projection

4. **`planner_report.py`** (37 lines) - Runtime statistics dataclass

5. **`test/test_ml_planner.py`** (202 lines) - **Best usage examples!**
   - Shows how to construct MLPlanner with RasterModel, VectorMapSimpleMLP, UrbanDriverOpenLoopModel
   - Full simulation pipeline test

6. **`test/test_transform_utils.py`** (72 lines) - Transform edge cases

## Common Usage Patterns

### 1. Loading Trained Model for Simulation (via Hydra)

**Config file:** `nuplan/planning/script/config/simulation/planner/ml_planner.yaml`
```yaml
ml_planner:
  _target_: nuplan.planning.simulation.planner.ml_planner.ml_planner.MLPlanner
  model_config: ???  # Dict with model architecture (e.g. raster_model)
  checkpoint_path: ???  # Path to .ckpt file
```

**Hydra command override:**
```bash
python nuplan/planning/script/run_simulation.py \
    planner=ml_planner \
    planner.model_config=raster_model \
    planner.checkpoint_path=/path/to/epoch=9-step=409.ckpt
```

**What happens internally** (`planner_builder.py:23-36`):
```python
# Build model architecture (feature builders + target builders)
torch_module_wrapper = build_torch_module_wrapper(planner_cfg.model_config)

# Load trained weights from checkpoint
model = LightningModuleWrapper.load_from_checkpoint(
    planner_cfg.checkpoint_path,
    model=torch_module_wrapper  # Architecture to load weights into
).model

# Create planner with loaded model
planner = MLPlanner(model=model)
```

### 2. Manual Construction (For Testing)

**Example: Raster Model MLPlanner** (from `test_ml_planner.py:44-75`)
```python
from nuplan.planning.simulation.planner.ml_planner.ml_planner import MLPlanner
from nuplan.planning.training.modeling.models.raster_model import RasterModel
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.feature_builders.raster_feature_builder import RasterFeatureBuilder
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import EgoTrajectoryTargetBuilder

# Define trajectory sampling
future_trajectory_sampling = TrajectorySampling(time_horizon=6.0, num_poses=12)

# Create model with feature and target builders
model = RasterModel(
    model_name="resnet50",
    pretrained=True,
    num_input_channels=4,
    num_features_per_pose=3,  # x, y, heading
    future_trajectory_sampling=future_trajectory_sampling,
    feature_builders=[
        RasterFeatureBuilder(
            map_features={'LANE': 1.0, 'INTERSECTION': 1.0, 'STOP_LINE': 0.5, 'CROSSWALK': 0.5},
            num_input_channels=4,
            target_width=224,
            target_height=224,
            target_pixel_size=0.5,
            ego_width=2.297,
            ego_front_length=4.049,
            ego_rear_length=1.127,
            ego_longitudinal_offset=0.0,
            baseline_path_thickness=1,
        )
    ],
    target_builders=[
        EgoTrajectoryTargetBuilder(future_trajectory_sampling=future_trajectory_sampling)
    ],
)

# Create MLPlanner
planner = MLPlanner(model=model)
```

### 3. Using MLPlanner in Simulation

**Full simulation example** (from `test_ml_planner.py:170-197`):
```python
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration

# Get scenario
scenario = get_test_nuplan_scenario()

# Initialize history buffer (2 seconds = 21 samples @ 0.1s)
buffer_size = int(2.0 / scenario.database_interval + 1)
history = SimulationHistoryBuffer.initialize_from_scenario(
    buffer_size=buffer_size,
    scenario=scenario,
    observation_type=DetectionsTracks
)

# Initialize planner with scenario-specific data
initialization = PlannerInitialization(
    route_roadblock_ids=scenario.get_route_roadblock_ids(),
    mission_goal=scenario.get_mission_goal(),
    map_api=scenario.map_api,
)
planner.initialize(initialization)

# Compute trajectory (typically called every 0.1s in simulation loop)
trajectory = planner.compute_trajectory(
    PlannerInput(
        iteration=SimulationIteration(index=0, time_point=scenario.start_time),
        history=history,
        traffic_light_data=list(scenario.get_traffic_light_status_at_iteration(0)),
    )
)

# Trajectory includes current state + predicted states
assert len(trajectory.get_sampled_trajectory()) == planner._num_output_dim + 1
```

### 4. Extracting Performance Metrics

```python
# After running simulation
report = planner.generate_planner_report(clear_stats=True)

# Access time series
feature_times = report.feature_building_runtimes  # List[float] in seconds
inference_times = report.inference_runtimes       # List[float] in seconds
total_times = report.compute_trajectory_runtimes  # List[float] in seconds

# Compute statistics
summary = report.compute_summary_statistics()
print(f"Mean inference time: {summary['inference_runtimes_mean']:.4f}s")
print(f"95th percentile: {np.percentile(inference_times, 95):.4f}s")

# Ensure real-time constraint (< 0.1s typically)
max_time = max(total_times)
assert max_time < 0.1, f"Planning too slow: {max_time:.4f}s > 0.1s"
```

## Gotchas & Pitfalls

### 1. **Model Must Match Checkpoint Architecture** ⚠️
**Problem:** Loading checkpoint with mismatched architecture crashes.

```python
# ❌ FAILS - Model architecture doesn't match checkpoint
checkpoint_model = RasterModel(num_input_channels=8, ...)  # Trained with 8 channels
current_model = RasterModel(num_input_channels=4, ...)     # Trying to load with 4
LightningModuleWrapper.load_from_checkpoint(path, model=current_model)
# RuntimeError: size mismatch for conv1.weight: [64, 8, ...] vs [64, 4, ...]

# ✅ CORRECT - Exactly match training architecture
checkpoint_model = build_torch_module_wrapper(cfg.model_config)  # From same config as training
loaded_model = LightningModuleWrapper.load_from_checkpoint(path, model=checkpoint_model).model
```

**Solution:** Always use `build_torch_module_wrapper()` with the same `model_config` as training.

### 2. **Checkpoint Path Special Characters** ⚠️
**Problem:** Hydra config parser fails on paths with `=` in filename (common in PyTorch Lightning checkpoints).

```bash
# ❌ FAILS - Hydra parses = as config delimiter
planner.checkpoint_path=/path/epoch=9-step=409.ckpt
# Error: mismatched input '=' expecting <EOF>

# ✅ WORKAROUND - Create symlink without special chars
ln -s epoch=9-step=409.ckpt best_model.ckpt
planner.checkpoint_path=/path/best_model.ckpt

# ✅ BETTER - Use Justfile helper (auto-detects latest)
just simulate-ml  # Finds checkpoint automatically
```

**Root cause:** See CLAUDE.md Session 2 "Lessons Learned" - known Hydra limitation.

### 3. **Feature Builder Compatibility** ⚠️
**Problem:** Model's feature builders must support `get_features_from_simulation()`.

```python
# Feature builders used in TRAINING
builder.get_features_from_scenario(scenario, iteration)

# Feature builders used in SIMULATION (MLPlanner)
builder.get_features_from_simulation(current_input, initialization)
```

**Not all feature builders implement both methods!** Check before training.

**Solution:** Verify feature builders inherit from `AbstractFeatureBuilder` and implement both methods. Most standard builders (RasterFeatureBuilder, VectorSetMapFeatureBuilder, etc.) support both.

### 4. **Batch Dimension Mismatch** ⚠️
**Problem:** Models expect batched inputs, but simulation provides single samples.

```python
# Model trained with batches
features['raster'].data.shape  # [batch_size, channels, height, width]

# Simulation provides single sample
# ModelLoader.build_features() handles this (line 63):
features = {name: feature.collate([feature]) for name, feature in features.items()}
# Adds batch dimension: [1, channels, height, width]

# But extraction must remove batch dimension (ml_planner.py:60)!
trajectory_tensor = trajectory_predicted.data  # [1, num_poses, 3]
trajectory = trajectory_tensor.cpu().detach().numpy()[0]  # Remove batch [0]
```

**CRITICAL:** Always index `[0]` when extracting single prediction from batched output!

### 5. **Device Management: CPU vs GPU** ⚠️
**Problem:** Model may be on CUDA but features on CPU (or vice versa).

```python
# ModelLoader handles this automatically (model_loader.py:31-38):
self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
self._model = self._model.to(self.device)
# AND moves features to device (line 62):
features = {name: feature.to_device(self.device) for name, feature in features.items()}
```

**But beware manual feature construction:**
```python
# ❌ FAILS if feature not on model device
custom_feature = MyFeature(data=torch.randn(1, 10))  # Defaults to CPU
predictions = model.forward({'custom': custom_feature})  # Model on CUDA
# RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same

# ✅ CORRECT - Let ModelLoader handle it
features = model_loader.build_features(current_input, initialization)  # Auto-device
```

**Solution:** Always use `ModelLoader.build_features()` - don't construct features manually!

### 6. **Precision Handling (FP16/FP32)** ⚠️
**Problem:** Models trained with mixed precision may expect FP16 inputs.

```python
# Training often uses automatic mixed precision (AMP)
trainer = pl.Trainer(precision=16)

# But ModelLoader defaults to FP32 inference
# PyTorch Lightning's .eval() should handle this, but verify:
model.half()  # If trained with FP16
```

**AIDEV-NOTE:** Currently MLPlanner doesn't explicitly handle precision. If you train with FP16, you may need to modify ModelLoader to cast features appropriately.

**Workaround:** Train with FP32 for simulation, or explicitly cast in model forward:
```python
def forward(self, features):
    # Explicitly handle precision
    if self.dtype == torch.float16:
        features = {k: v.half() for k, v in features.items()}
    ...
```

### 7. **Gradient Computation Disabled** ⚠️
**Problem:** Forgetting to disable gradients wastes memory and slows inference.

```python
# ✅ ModelLoader handles this (model_loader.py:28)
torch.set_grad_enabled(False)  # Called in initialize()

# ❌ But beware external model manipulation
model.train()  # Re-enables gradients!
# Always use:
model.eval()   # Sets to evaluation mode
```

**Note:** `torch.set_grad_enabled(False)` is global! If you need gradients elsewhere (e.g., adversarial examples), use context manager:
```python
with torch.no_grad():
    predictions = model.forward(features)
```

### 8. **Trajectory Horizon Mismatches** ⚠️
**Problem:** Model may predict different horizon than expected by controller.

```python
# Model trained with 6 second horizon, 12 poses
future_trajectory_sampling = TrajectorySampling(time_horizon=6.0, num_poses=12)

# Controller expects 8 seconds
# MLPlanner just returns what model predicts - no automatic extension!

# Prediction: 12 states @ 0.5s intervals = 6 seconds
# Controller needs: 80 states @ 0.1s intervals = 8 seconds
```

**Solution:** Either:
1. Retrain model with longer horizon
2. Interpolate predicted trajectory to controller's desired sampling
3. Use `InterpolatedTrajectory` (already handles resampling on query)

**AIDEV-NOTE:** `InterpolatedTrajectory.get_state_at_time()` handles interpolation, but extrapolation beyond horizon may give poor results.

### 9. **Relative vs Absolute Pose Confusion** ⚠️
**Problem:** Model outputs are **ego-centric relative poses**, not global absolute poses!

```python
# ❌ WRONG - Treating relative poses as absolute
predictions = model.forward(features)  # [Δx, Δy, Δθ] relative to current ego
ego_state = current_input.history.current_state[0]
# Can't use predictions directly as global coordinates!

# ✅ CORRECT - Use transform_predictions_to_states (ml_planner.py:96-98)
states = transform_predictions_to_states(
    predictions,
    ego_history=current_input.history.ego_states,
    future_horizon=self._future_horizon,
    step_interval=self._step_interval
)
# Now states are absolute EgoState objects in global frame
```

**Why relative?** Training uses ego-centric frame for translation invariance. Simulation needs global coordinates for map queries and collision checking.

### 10. **Velocity/Acceleration Approximation Artifacts** ⚠️
**Problem:** Numerical differentiation for velocity/acceleration can introduce noise.

**How it works** (`transform_utils.py:74-150`):
1. Combine past history + predicted poses
2. Interpolate to uniform timesteps
3. Take numerical derivatives (1st for velocity, 2nd for acceleration)
4. Project from global XY to ego-centric DS frame
5. Interpolate back to desired timesteps

**Potential issues:**
- **Jagged trajectories** → High jerk (derivative of acceleration)
- **Interpolation artifacts** near trajectory start/end
- **Coordinate frame discontinuities** during large heading changes

**Solution:** Apply trajectory smoothing before or after conversion:
```python
from scipy.signal import savgol_filter

# Smooth predicted poses before transformation
smoothed_predictions = savgol_filter(predictions, window_length=5, polyorder=2, axis=0)
states = transform_predictions_to_states(smoothed_predictions, ...)
```

### 11. **Model Warmup Not Handled** ⚠️
**Problem:** First inference call may be slow (CUDA initialization, JIT compilation).

```python
# First call: ~0.5-1.0s (CUDA warmup, cache population)
trajectory1 = planner.compute_trajectory(input1)

# Subsequent calls: ~0.01-0.05s
trajectory2 = planner.compute_trajectory(input2)
```

**AIDEV-TODO:** Add warmup to `ModelLoader.initialize()`:
```python
def initialize(self):
    self._initialize_torch()
    self._initialize_model()

    # Warmup with dummy input
    dummy_features = self._build_dummy_features()
    _ = self.infer(dummy_features)

    self._initialized = True
```

**Workaround:** Ignore first trajectory's timing, or run dummy inference before simulation starts.

### 12. **Missing History Buffer** ⚠️
**Problem:** Feature builders often need past states (e.g., agent tracks, ego trajectory history).

```python
# ❌ FAILS - History buffer not initialized
history = SimulationHistoryBuffer(buffer_size=1)  # Only stores current state!
features = model_loader.build_features(current_input, initialization)
# Feature builder expects 2 seconds of history (21 samples @ 0.1s)
# RuntimeError: Insufficient history

# ✅ CORRECT - Size buffer to match feature requirements
buffer_duration = 2.0  # seconds
buffer_size = int(buffer_duration / scenario.database_interval + 1)
history = SimulationHistoryBuffer.initialize_from_scenario(
    buffer_size=buffer_size,
    scenario=scenario,
    observation_type=DetectionsTracks
)
```

**Rule of thumb:** Buffer size ≥ `max(feature_builder.history_duration) / dt + 1`

### 13. **Trajectory Decoding Edge Cases** ⚠️
**Problem:** `transform_predictions_to_states()` can fail on degenerate inputs.

**Edge cases:**
- **Zero-length trajectory:** `predictions.shape[0] == 0`
- **Insufficient history:** Less than 2 ego states (can't compute derivatives)
- **NaN/Inf in predictions:** Garbage model output
- **Extremely large headings:** Heading wraparound issues

**Defensive coding:**
```python
# Check prediction validity before decoding (ml_planner.py:92)
predictions = self._infer_model(features)

# Add validation:
if predictions.shape[0] == 0:
    raise ValueError("Model returned empty trajectory")
if np.any(~np.isfinite(predictions)):
    raise ValueError(f"Model returned invalid predictions: {predictions}")

states = transform_predictions_to_states(predictions, ...)
```

### 14. **Model Expects Different Observation Type** ⚠️
**Problem:** Model trained with one observation type, but simulation uses another.

```python
# Model trained with DetectionsTracks (agent bounding boxes)
class RasterFeatureBuilder:
    def get_features_from_simulation(self, current_input, initialization):
        observation = current_input.history.observations[-1]
        # Expects DetectionsTracks!
        agents = observation.tracked_objects.get_agents()
        ...

# But simulation provides Sensors (raw sensor data)
# AttributeError: 'Sensors' object has no attribute 'tracked_objects'
```

**Solution:** Match `observation_type()` to feature builder requirements:
```python
def observation_type(self) -> Type[Observation]:
    # Must match what feature builders expect!
    return DetectionsTracks  # Most common for ML planners
```

### 15. **Inference Runtime Exceeds Real-Time Constraint** ⚠️
**Problem:** Model too slow for 10 Hz simulation (0.1s budget).

**Typical breakdown:**
- Feature building: 20-40 ms
- Model inference: 10-80 ms (depending on model complexity)
- Trajectory decoding: 5-10 ms
- **Total:** 35-130 ms

**If > 100 ms, you're in trouble!**

**Solutions:**
1. **Reduce model complexity** - Smaller ResNet, fewer GCN layers
2. **Enable TorchScript JIT** - Compile model for faster inference
3. **Use GPU** - Even modest GPU is 5-10x faster than CPU
4. **Optimize feature builders** - Cache map queries, reduce rasterization resolution
5. **Decrease prediction horizon** - 6s instead of 8s (fewer poses to predict)

**Monitor with:**
```python
report = planner.generate_planner_report()
print(f"Mean total time: {np.mean(report.compute_trajectory_runtimes)*1000:.1f} ms")
print(f"Feature building: {np.mean(report.feature_building_runtimes)*1000:.1f} ms")
print(f"Inference: {np.mean(report.inference_runtimes)*1000:.1f} ms")
```

## Test Coverage Notes

Test directory: `nuplan/planning/simulation/planner/ml_planner/test/`

**test_ml_planner.py:**
- Tests 3 model architectures: VectorMapSimpleMLP, RasterModel, UrbanDriverOpenLoopModel
- Validates trajectory length: `len(trajectory) == num_output_dim + 1` (includes current state)
- Shows construction pattern for each model type

**test_transform_utils.py:**
- Unit test for `transform_predictions_to_states()`
- Validates:
  - Current ego state preservation
  - Relative → absolute pose conversion
  - Velocity computation (expects [1.0, 0.0] m/s for constant motion)
  - Acceleration computation (expects [0.0, 0.0] for constant velocity)
  - Time progression

**Coverage gaps** (AIDEV-TODO):
- No test for checkpoint loading via `LightningModuleWrapper.load_from_checkpoint()`
- No test for device management (CPU vs CUDA)
- No test for edge cases: empty predictions, NaN/Inf, insufficient history
- No test for performance monitoring (MLPlannerReport)

## Related Documentation

### Parent Module
- `nuplan/planning/simulation/planner/CLAUDE.md` - AbstractPlanner interface (Session 1 ✅)

### Critical Dependencies (Session 1 ✅)
- `nuplan/common/actor_state/CLAUDE.md` - EgoState, StateSE2 (trajectory representations)
- `nuplan/common/geometry/CLAUDE.md` - `relative_to_absolute_poses()` coordinate transforms
- `nuplan/planning/simulation/trajectory/CLAUDE.md` - InterpolatedTrajectory output format

### Undocumented Dependencies (Future Sessions ⏳)
- `nuplan/planning/training/modeling/CLAUDE.md` - TorchModuleWrapper, model architectures (Session 5)
- `nuplan/planning/training/preprocessing/CLAUDE.md` - Feature builders, target builders (Session 5)
- `nuplan/planning/training/CLAUDE.md` - Training pipeline, checkpointing (Session 6)

### Siblings
- `nuplan/planning/simulation/planner/simple_planner.py` - Non-ML baseline for comparison
- `nuplan/planning/simulation/planner/idm_planner.py` - Rule-based alternative

### Dependents
- `nuplan/planning/script/builders/planner_builder.py` - Instantiation logic
- `nuplan/planning/script/run_simulation.py` - Main simulation entry point
- `nuplan/planning/metrics/CLAUDE.md` - Evaluation of ML planner performance

---

**AIDEV-NOTE:** MLPlanner is THE way to deploy trained models in nuPlan. Understanding this module is critical for evaluating ML-based autonomous driving systems. Start with test_ml_planner.py to see concrete examples.

**AIDEV-NOTE:** The feature building pipeline (lines 48-64 in model_loader.py) is subtle: get_features → to_feature_tensor → to_device → collate. Each step is required! Missing any step causes shape/device/type errors.

**AIDEV-NOTE:** transform_utils.py implements sophisticated coordinate math. The velocity/acceleration computation (lines 74-150) is non-trivial: interpolation → differentiation → frame projection → interpolation back. Understand this to debug trajectory issues.

**AIDEV-TODO:** Add model warmup to ModelLoader.initialize() to avoid first-call latency spikes.

**AIDEV-TODO:** Add precision handling (FP16/FP32) configuration to support mixed-precision trained models.

**AIDEV-TODO:** Consider adding trajectory validation/sanitization in _infer_model() to catch bad model outputs early.

**AIDEV-QUESTION:** Should MLPlanner support ensemble models (multiple checkpoints, average predictions)? Current design assumes single model.

**AIDEV-QUESTION:** Why does transform_predictions_to_states default to including current ego state (include_ego_state=True)? InterpolatedTrajectory expects future states only, but code adds current. Is this for controller compatibility?
