# Hydra Configuration System Guide

## Overview

nuPlan uses **Hydra 1.1.0rc1** (pinned RC version) for declarative configuration management. Configuration controls every aspect of training and simulation: dataset loading, parallelization, metrics, callbacks, and experiment hyperparameters. The system enables rapid hypothesis testing by composing configurations from YAML files and CLI overrides without code changes.

**Key principle**: All configuration is declarative YAML with hierarchical composition and interpolation. This enables reproducible experiments and clean separation of concerns.

## Directory Structure

Configuration is organized into two primary roots under `nuplan/planning/script/config/`:

```
config/
├── common/                           # Shared across training and simulation
│   ├── default_experiment.yaml       # Output paths, logging, seed, profiling
│   ├── default_common.yaml           # Default dataset, worker, model selection
│   ├── default_submission.yaml       # Submission metadata (contestant_id, submission_id)
│   ├── scenario_builder/             # Dataset sources (nuplan_mini, nuplan, nuplan_challenge)
│   │   ├── nuplan_mini.yaml          # Mini dataset config
│   │   ├── nuplan.yaml               # Full dataset config
│   │   ├── nuplan_challenge.yaml     # Challenge dataset config
│   │   ├── scenario_mapping/         # Scenario extraction info (duration, offset, subsampling)
│   │   └── vehicle_parameters/       # Vehicle params (Pacifica, etc.)
│   ├── scenario_filter/              # Scenario filtering logic
│   │   ├── one_continuous_log.yaml   # Single log filter
│   │   ├── one_of_each_scenario_type.yaml  # One scenario per type
│   │   ├── training_scenarios.yaml   # 2000 scenarios from training set
│   │   ├── all_scenarios.yaml        # No filtering
│   │   ├── few_training_scenarios.yaml  # Minimal dataset (prototyping)
│   │   └── nuplan_challenge_scenarios.yaml  # Challenge scenarios
│   ├── simulation_metric/            # Metric specifications
│   │   ├── common_metrics.yaml       # Default metrics
│   │   └── high_level/               # Specific metrics (comfort, progress, etc.)
│   ├── worker/                       # Parallelization strategy
│   │   ├── ray_distributed.yaml      # Ray parallel execution
│   │   ├── sequential.yaml           # Single-threaded execution
│   │   └── single_machine_thread_pool.yaml  # Thread pool on single machine
│   ├── model/                        # ML model configs
│   │   ├── raster_model.yaml         # Raster representation model
│   │   ├── vector_model.yaml         # Vector representation model
│   │   └── simple_vector_model.yaml  # Simplified vector model
│   └── splitter/                     # Train/val/test split strategy
│       └── nuplan.yaml               # nuPlan standard splits
│
├── simulation/                       # Simulation-specific configs
│   ├── default_simulation.yaml       # Simulation root config (calls defaults)
│   ├── default_run_metric_aggregator.yaml
│   ├── default_submission_planner.yaml
│   ├── planner/                      # Available planners
│   │   ├── simple_planner.yaml       # Hand-crafted baseline planner
│   │   ├── ml_planner.yaml           # ML-based planner (requires checkpoint)
│   │   ├── idm_planner.yaml          # Intelligent driver model
│   │   ├── log_future_planner.yaml   # Oracle planner (uses expert data)
│   │   └── remote_planner.yaml       # External planner service
│   ├── observation/                  # Ego observation types
│   │   ├── box_observation.yaml      # Bounding box representation
│   │   ├── ego_centric_ml_agents_observation.yaml
│   │   ├── idm_agents_observation.yaml
│   │   └── lidar_pc_observation.yaml # Point cloud representation
│   ├── ego_controller/               # Ego motion execution
│   │   ├── two_stage_controller.yaml # Reference tracking + stabilization
│   │   ├── log_play_back_controller.yaml  # Replay expert actions
│   │   ├── motion_model/             # Motion models (kinematic, dynamic)
│   │   └── ...
│   ├── predictor/                    # Agent motion prediction
│   │   └── log_future_predictor.yaml # Use expert future trajectories
│   ├── simulation_time_controller/   # Timestep advancement
│   │   └── step_simulation_time_controller.yaml  # Fixed timesteps
│   ├── callback/                     # Simulation callbacks
│   │   ├── simulation_log_callback.yaml  # Save simulation data
│   │   ├── serialization_callback.yaml
│   │   ├── timing_callback.yaml      # Measure performance
│   │   └── streaming_viz_callback.yaml  # Real-time visualization
│   ├── main_callback/                # Post-scenario aggregation
│   │   ├── time_callback.yaml        # Timing summary
│   │   ├── metric_file_callback.yaml # Save metric files
│   │   ├── metric_aggregator_callback.yaml  # Aggregate metrics
│   │   ├── metric_summary_callback.yaml    # Summary statistics
│   │   ├── publisher_callback.yaml   # Publish results
│   │   ├── completion_callback.yaml  # Mark completion
│   │   └── validation_callback.yaml  # Validate outputs
│   └── metric_aggregator/            # Metric aggregation strategy
│       └── default_weighted_average.yaml
│
└── training/                         # Training-specific configs
    ├── default_training.yaml         # Training root config
    ├── lightning/                    # PyTorch Lightning trainer
    │   └── default_lightning.yaml
    ├── callbacks/                    # Training callbacks
    │   └── default_callbacks.yaml
    ├── data_loader/                  # DataLoader config
    │   └── default_data_loader.yaml
    ├── objective/                    # Loss functions
    │   └── raster_model_objective.yaml
    ├── optimizer/                    # Optimizers
    │   ├── adam.yaml
    │   └── adamw.yaml
    ├── lr_scheduler/                 # Learning rate scheduling
    │   └── one_cycle_lr.yaml
    ├── warm_up_lr_scheduler/         # Warmup strategies
    │   ├── linear_warm_up.yaml
    │   └── constant_warm_up.yaml
    ├── training_metric/              # Metrics tracked during training
    │   └── default_training_metrics.yaml
    ├── scenario_type_weights/        # Per-type training weights
    │   └── default_scenario_type_weights.yaml
    ├── data_augmentation/            # Augmentation strategies
    │   └── default_data_augmentation.yaml
    └── data_augmentation_scheduler/  # Augmentation scheduling
        └── default_augmentation_schedulers.yaml
```

## Configuration Composition Pattern

Hydra uses a **defaults list** to compose configurations hierarchically. Each config file has a `defaults:` section specifying which groups to include and in what order.

### Example: default_simulation.yaml (Simulation Root)

```yaml
# Step 1: Include base configs (order matters!)
defaults:
  - default_experiment        # Experiment metadata (output dirs, seed)
  - default_common            # Common defaults (dataset, worker)
  - default_submission        # Submission info

  # Step 2: Include optional group configs (with subgroup selection)
  - simulation_metric:
      - default_metrics      # Which metrics to compute
  - callback:
      - simulation_log_callback  # Record simulation data
  - main_callback:           # Aggregation callbacks (list of items)
      - time_callback
      - metric_file_callback
      - metric_aggregator_callback
  - main_callback:
      - metric_summary_callback
  - splitter: nuplan         # Train/val/test split

  # Step 3: Mark which groups are mandatory (must be specified)
  - observation: null        # MUST be provided (planner observation model)
  - ego_controller: null     # MUST be provided (motion execution)
  - planner: null            # MUST be provided (planning algorithm)
  - simulation_time_controller: step_simulation_time_controller
  - metric_aggregator:
      - default_weighted_average

  # Step 4: Override Hydra's own behavior
  - override hydra/job_logging: none  # Disable Hydra's logging
  - override hydra/hydra_logging: none

# Step 5: Define parameters specific to this config
experiment_name: 'simulation'
enable_simulation_progress_bar: False
simulation_history_buffer_duration: 2.0
number_of_gpus_allocated_per_simulation: 0
number_of_cpus_allocated_per_simulation: 1
run_metric: true
exit_on_failure: false
max_callback_workers: 4
```

### Example: default_common.yaml (Common Defaults)

```yaml
defaults:
  - scenario_builder: nuplan_mini        # Dataset source
  - scenario_filter: one_continuous_log  # Filtering strategy
  - model: null                          # ML model (if used)
  - worker: ray_distributed              # Parallelization

distribute_by_scenario: true
distributed_timeout_seconds: 7200
verbose: false
```

## Config Group Structure

### 1. Scenario Builder Groups

**Purpose**: Specify which dataset to use.

**Available configs**:
- `nuplan_mini.yaml` - Mini dataset (~8GB DB + sensors)
- `nuplan.yaml` - Full training set
- `nuplan_challenge.yaml` - Challenge set

**Key parameters**:
```yaml
_target_: nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder.NuPlanScenarioBuilder
data_root: ${oc.env:NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/mini
map_root: ${oc.env:NUPLAN_MAPS_ROOT}
sensor_root: ${oc.env:NUPLAN_DATA_ROOT}/nuplan-v1.1/sensor_blobs
db_files: null  # if provided, overrides data_root
map_version: nuplan-maps-v1.0
include_cameras: false  # Camera data (slow!)
max_workers: null       # Parallel DB loading
```

### 2. Scenario Filter Groups

**Purpose**: Select subset of scenarios from builder.

**Available configs**:
- `one_continuous_log.yaml` - Single log file (~50-200 scenarios)
- `one_of_each_scenario_type.yaml` - 1 scenario per type (~14 scenarios)
- `few_training_scenarios.yaml` - Minimal set (prototyping)
- `training_scenarios.yaml` - 2000 scenarios for training
- `all_scenarios.yaml` - No filtering (expensive!)
- `nuplan_challenge_scenarios.yaml` - Challenge specific scenarios

**Key parameters**:
```yaml
_target_: nuplan.planning.scenario_builder.scenario_filter.ScenarioFilter
scenario_types: null           # List of scenario types to include
scenario_tokens: null          # List of specific scenario tokens
log_names: null                # Filter by log name
map_names: null                # Filter by city/map
num_scenarios_per_type: null   # Limit per type (applied first)
limit_total_scenarios: null    # Hard limit (float=fraction, int=count)
timestamp_threshold_s: null    # Temporal spacing between scenarios
ego_displacement_minimum_m: null  # Ego must move X meters
ego_start_speed_threshold: null   # Ego must accelerate above X m/s
ego_stop_speed_threshold: null    # Ego must decelerate below X m/s
ego_route_radius: null        # Route must be near ego (map queries!)
token_set_path: null          # Pre-computed token set file
remove_invalid_goals: true    # Skip scenarios without mission goal
shuffle: false                # Randomize selection
expand_scenarios: false       # Split multi-sample scenarios
```

**Filter application order** (important!):
1. `num_scenarios_per_type` - Type-level limit (cheap)
2. `limit_total_scenarios` - Total limit (cheap)
3. `timestamp_threshold_s` - Temporal filter (medium)
4. `ego_displacement_minimum_m` - Motion filter (expensive)
5. `ego_start/stop_speed_threshold` - Speed filters (expensive)
6. `token_set_threshold` - Set operations (medium)
7. `ego_route_radius` - Map queries (very expensive!)

**AIDEV-NOTE**: Filters are applied sequentially. Putting expensive filters early impacts performance. Cheap filters should come first to reduce dataset size before expensive operations.

### 3. Worker Groups

**Purpose**: Parallelization strategy.

**Available configs**:

#### ray_distributed.yaml (Default)
```yaml
_target_: nuplan.planning.utils.multithreading.worker_ray.RayDistributed
threads_per_node: null        # CPU threads (null = all available)
debug_mode: false             # Execute serially for debugging
log_to_driver: true           # Print Ray worker output
master_node_ip: null          # For cluster execution
use_distributed: false        # Ray's distributed mode
```

**Performance tuning**:
- `threads_per_node=4` - Safe for 16GB RAM, 4-8 workers total
- `threads_per_node=1` - CPU-only, no parallelism
- `threads_per_node=null` - Use all CPUs (can OOM!)

#### sequential.yaml
```yaml
_target_: nuplan.planning.utils.multithreading.worker_sequential.Sequential
# Single-threaded, no parallelism. Used for debugging or CPU-only machines.
```

**Performance tuning**:
- Disable callback parallelization in default_simulation.yaml: `disable_callback_parallelization: true`
- No Ray overhead, but slower scenario execution

### 4. Planner Groups

**Purpose**: Select planning algorithm.

**Available configs**:
- `simple_planner.yaml` - Baseline hand-crafted planner
- `ml_planner.yaml` - Neural network planner (requires checkpoint)
- `idm_planner.yaml` - Intelligent driver model
- `log_future_planner.yaml` - Oracle (uses expert trajectories)
- `remote_planner.yaml` - External service planner

**Simple planner example**:
```yaml
simple_planner:
  _target_: nuplan.planning.simulation.planner.simple_planner.SimplePlanner
  horizon_seconds: 10.0    # Planning horizon
  sampling_time: 0.25      # Trajectory sample time
  acceleration: [0.0, 0.0] # [longitudinal, lateral]
  max_velocity: 5.0        # m/s
  steering_angle: 0.0      # radians
```

**ML planner example**:
```yaml
ml_planner:
  _target_: nuplan.planning.simulation.planner.ml_planner.ml_planner.MLPlanner
  model_config: raster_model      # Which model config to use
  checkpoint_path: ???             # Path to trained checkpoint
```

### 5. Callback Groups

**Purpose**: Hooks executed during simulation.

**Available callbacks**:
- `simulation_log_callback.yaml` - Save scenario simulation logs
- `serialization_callback.yaml` - Serialize objects
- `timing_callback.yaml` - Measure step timing
- `streaming_viz_callback.yaml` - Real-time visualization (HTTP POST)

**Post-scenario callbacks** (main_callback/):
- `time_callback.yaml` - Total time summary
- `metric_file_callback.yaml` - Write metric files to disk
- `metric_aggregator_callback.yaml` - Aggregate metrics
- `metric_summary_callback.yaml` - Summary statistics
- `publisher_callback.yaml` - Publish to external service
- `completion_callback.yaml` - Mark job complete
- `validation_callback.yaml` - Validate output schema

**Performance tuning**:
```yaml
# Disable all callbacks (fastest, metrics disabled)
callback: []

# Keep only essential callbacks
callback:
  - simulation_log_callback  # Needed for post-analysis

# Disable parallelization of callbacks (faster for small jobs)
disable_callback_parallelization: true
max_callback_workers: 1
```

### 6. Metric Groups

**Purpose**: Specify which metrics to compute during simulation.

**Available metrics** (high_level/):
- `ego_is_comfortable_statistics.yaml` - Comfort (jerk, lateral accel)
- `ego_is_making_progress_statistics.yaml` - Goal progress
- `drivable_area_compliance_statistics.yaml` - Lane/area adherence
- `collision_statistics.yaml` - Collision detection
- `ego_expert_final_l2_error_within_bound_statistics.yaml` - L2 distance to expert

**Performance impact**:
- **Low**: `ego_expert_l2_error_statistics` (trajectory comparison)
- **Medium**: `drivable_area_compliance_statistics` (map queries)
- **High**: Reactive agent metrics (prediction + interaction modeling)

**AIDEV-NOTE**: Reactive agent metrics require predicting other agents' futures, which is expensive. Disable for performance investigation: `simulation_metric: []`

## Override Syntax

### CLI Overrides

Hydra allows overriding config values from command line without editing files.

**Basic syntax**:
```bash
# Override simple value
python script.py key=value

# Override nested value (dot notation)
python script.py worker.threads_per_node=4

# Override in a config group
python script.py planner=simple_planner

# Override with list
python script.py scenario_filter.scenario_types=[starting_left_turn,near_multiple_vehicles]

# Add new parameter (not in config)
python script.py +new_param=value

# Interpolation
python script.py experiment_name=my_exp output_dir=/tmp/${experiment_name}
```

**Real examples for testing**:
```bash
# Minimal config: disable callbacks, reduce scenarios
uv run python nuplan/planning/script/run_simulation.py \
  planner=simple_planner \
  observation=box_observation \
  ego_controller=two_stage_controller \
  callback=[] \
  scenario_filter.limit_total_scenarios=10 \
  worker.threads_per_node=1

# Fast iteration: sequential worker, no parallelism
uv run python nuplan/planning/script/run_simulation.py \
  worker=sequential \
  callback=[] \
  disable_callback_parallelization=true \
  scenario_filter.num_scenarios_per_type=1

# ML planner with checkpoint
uv run python nuplan/planning/script/run_simulation.py \
  planner=ml_planner \
  planner.ml_planner.model_config=raster_model \
  planner.ml_planner.checkpoint_path=/path/to/epoch=9-step=409.ckpt \
  scenario_filter.limit_total_scenarios=5
```

**Common override patterns**:

1. **Disable expensive features**:
   ```bash
   callback=[]                           # No callbacks
   simulation_metric=[]                  # No metrics (CRITICAL for performance!)
   disable_callback_parallelization=true # No callback workers
   max_callback_workers=1                # Minimal callbacks
   worker.threads_per_node=1             # Single-threaded
   ```

2. **Reduce dataset size**:
   ```bash
   scenario_filter.limit_total_scenarios=10      # 10 scenarios total
   scenario_filter.num_scenarios_per_type=1      # 1 per type
   scenario_filter=one_of_each_scenario_type     # ~14 scenarios
   ```

3. **Change filtering**:
   ```bash
   scenario_filter.scenario_types=[starting_left_turn]      # Single type
   scenario_filter.map_names=[us-nv-las-vegas-strip]        # Single city
   scenario_filter.log_names=[2021.07.16.20.45.29_veh-35_01095_01486]  # Single log
   scenario_filter.limit_total_scenarios=0.1     # 10% of dataset (fraction)
   ```

4. **Investigate database I/O**:
   ```bash
   callback=[]                          # Remove callbacks (they fetch metrics)
   simulation_metric=[]                 # Remove metrics (they query DB)
   observation=box_observation          # Minimal observation
   worker.threads_per_node=1            # Single thread to profile cleanly
   ```

## Debugging Configurations

### View Resolved Config

```bash
# Show the final composed config (with interpolations resolved)
python script.py --cfg job --resolve

# Show what config groups are being used
python script.py --cfg job

# Show composition order
python script.py --cfg defaults
```

### Common Config Errors

**Error**: `ConfigCompositionException: Could not load 'planner/my_planner.yaml'`
- **Cause**: Config file doesn't exist in right directory
- **Fix**: Check `config/simulation/planner/my_planner.yaml` exists

**Error**: `MissingMandatoryValue: config is missing mandatory value for key 'planner'`
- **Cause**: Planner not specified in defaults or CLI
- **Fix**: Add `planner=simple_planner` to CLI or config defaults

**Error**: `InterpolationKeyError: Key 'oc.env:MISSING_VAR' not found`
- **Cause**: Environment variable not set (e.g., NUPLAN_DATA_ROOT)
- **Fix**: `export NUPLAN_DATA_ROOT=/path/to/data`

**Error**: `mismatched input '=' expecting <EOF>`
- **Cause**: Checkpoint path contains `=` character, Hydra parses as override
- **Fix**: Use symlink without special chars, or use Justfile auto-detection

## Performance-Relevant Configurations

### Configuration Impact on 95ms Slowdown Investigation

**CRITICAL: Metrics compute database queries per-scenario**

During simulation, each scenario triggers:
1. **Observation queries**: `get_tracked_objects_at_iteration()` → DB queries per step
2. **Metric queries**: `get_ego_future_trajectory()`, `get_traffic_light_status_at_iteration()` → DB queries per metric

**To isolate database I/O**:
```bash
# Disable all metrics (removes metric DB queries)
simulation_metric: []

# Disable callbacks (removes serialization, logging overhead)
callback: []

# Single-threaded (cleaner profiling)
worker.threads_per_node: 1

# Minimal scenario set
scenario_filter.limit_total_scenarios: 5
```

**Expected improvement**: 95ms → 70-80ms per step if metrics were the bottleneck.

### Configuration Impact on Memory Usage

**Memory scaling**:
- Base: ~1GB (Hydra, Python runtime)
- Per Ray worker: ~500MB (PyTorch, model, cache)
- Per scenario in memory: ~50-200MB depending on sensors

**To reduce memory**:
```bash
# Reduce parallelism
worker.threads_per_node: 2              # 2 workers instead of 12

# Disable camera loading
scenario_builder:
  include_cameras: false                # Skip camera data

# Disable callback workers
disable_callback_parallelization: true
max_callback_workers: 1

# Pre-process scenarios
cache.cache_path: ${oc.env:NUPLAN_EXP_ROOT}/cache
cache.use_cache_without_dataset: true   # Load from cache only
```

### Configuration for Quick Prototyping

```yaml
# Fast iteration config (change only what's needed)
scenario_filter.limit_total_scenarios: 10
scenario_filter.shuffle: true           # Randomize to test different scenarios
callback: []
simulation_metric:
  - low_level:
      - ego_mean_speed_statistics       # Fast metric
worker.threads_per_node: 2
disable_callback_parallelization: true
```

## Key Configuration Patterns

### Pattern 1: Minimal Testing Config

```bash
# Run single scenario, no overhead
uv run python nuplan/planning/script/run_simulation.py \
  scenario_filter.limit_total_scenarios=1 \
  callback=[] \
  simulation_metric=[] \
  worker=sequential \
  planner=simple_planner \
  observation=box_observation \
  ego_controller=two_stage_controller
```

### Pattern 2: Full Metrics Config

```bash
# Comprehensive metrics, parallelize aggressively
uv run python nuplan/planning/script/run_simulation.py \
  scenario_filter.limit_total_scenarios=500 \
  worker.threads_per_node=8 \
  planner=ml_planner \
  planner.ml_planner.checkpoint_path=/path/to/model.ckpt
  # Uses default callbacks and metrics from config
```

### Pattern 3: Training Config

```bash
# Training-specific overrides
uv run python nuplan/planning/script/run_training.py \
  data_loader.params.batch_size=32 \
  objective=raster_model_objective \
  scenario_filter.limit_total_scenarios=500 \
  cache.cache_path=${NUPLAN_EXP_ROOT}/cache
```

## Interpolation & Environment Variables

### Environment Variable Interpolation

```yaml
# In config files, use ${oc.env:VARIABLE_NAME}
data_root: ${oc.env:NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/mini
output_dir: ${oc.env:NUPLAN_EXP_ROOT}/exp/${experiment}
```

### Config Value Interpolation

```yaml
# Reference other config values
experiment_name: simulation
experiment: ${experiment_name}/${job_name}/${experiment_uid}
output_dir: ${group}/${experiment}
```

### Function Interpolation

```yaml
# Hydra resolver functions
experiment_uid: ${now:${date_format}}  # Current timestamp
date_format: '%Y.%m.%d.%H.%M.%S'
```

## Configuration Validation & Hydra Features

### Struct Mode

By default, Hydra prevents adding unknown keys to configs (strict mode):

```python
# In code, disable for dynamic configs:
from omegaconf import OmegaConf
OmegaConf.set_struct(cfg, False)
cfg.new_key = value  # Now allowed
```

### Type Hints

Configs support type annotations (limited support in RC1):

```yaml
# myconfig.yaml
params:
  _target_: my.module.MyClass
  learning_rate: 0.001  # float
  num_epochs: 10        # int
  batch_size: 32        # int
```

## Key Takeaways for Testing

1. **Disable metrics & callbacks** for performance investigation: `callback=[]` + `simulation_metric=[]`
2. **Reduce dataset** before profiling: `scenario_filter.limit_total_scenarios=5`
3. **Single-thread** for clean profiling: `worker=sequential` or `worker.threads_per_node=1`
4. **Override via CLI** instead of editing configs: preserves version control
5. **Use interpolation** for reproducibility: `${oc.env:VARIABLE}`
6. **Check resolved config**: `--cfg job --resolve` to debug composition issues

## References

- **Hydra documentation**: https://hydra.cc/docs/intro/
- **nuPlan config examples**: `config/` directory
- **Entry points**: `run_simulation.py`, `run_training.py`
- **Justfile recipes**: Common command patterns

---

**AIDEV-NOTE**: This configuration system is the highest-leverage tool for rapid hypothesis testing. Most performance bottlenecks can be isolated by selectively disabling components via config overrides without touching code.
