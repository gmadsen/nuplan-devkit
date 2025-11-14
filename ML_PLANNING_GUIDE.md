# ML Planning Workflow Guide 🚗🧭

**A complete guide to training, evaluating, and iterating on ML-based autonomous vehicle planners using nuPlan**

This guide captures the complete workflow from zero to trained ML planner, with practical insights and a roadmap for continuous improvement.

---

## Table of Contents

1. [Quick Start - Your First ML Planner](#quick-start---your-first-ml-planner)
2. [The Complete Workflow](#the-complete-workflow)
3. [Understanding Your Results](#understanding-your-results)
4. [Iteration & Improvement Roadmap](#iteration--improvement-roadmap)
5. [Common Issues & Solutions](#common-issues--solutions)
6. [Advanced Topics](#advanced-topics)

---

## Quick Start - Your First ML Planner

**Goal**: Train a raster-based ML planner and evaluate it in simulation (30-45 minutes)

### Prerequisites
```bash
# 1. Environment setup
just setup
source .env

# 2. Verify CUDA (optional but recommended)
just check-cuda

# 3. Download mini dataset (~50GB, one-time)
just cli download --mini
```

### Complete Pipeline in 3 Commands

```bash
# 1. Train the model (20-30 minutes)
just train

# 2. Run simulation with trained model (10-20 minutes)
just simulate-ml

# 3. Visualize results in nuBoard
just nuboard
```

That's it! Your ML planner is trained, evaluated, and ready to analyze. 🎉

---

## The Complete Workflow

### Phase 1: Understanding the Dataset

Before training, spend time exploring the data:

```bash
# Launch Jupyter Lab
just tutorial

# Work through these notebooks in order:
# 1. nuplan_framework.ipynb - Architecture overview
# 2. nuplan_scenario_visualization.ipynb - Explore scenarios
```

**Key concepts to understand:**
- **Scenarios**: 20-second slices of real driving (lane changes, turns, intersections)
- **Scenario types**: `near_multiple_vehicles`, `starting_left_turn`, `on_pickup_dropoff`, etc.
- **Map data**: Lane boundaries, intersections, traffic lights
- **Agents**: Other vehicles, pedestrians in the scene
- **Expert trajectory**: The human driver's actual behavior (your training target)

**Time investment**: 1-2 hours
**Why it matters**: Understanding the data prevents garbage-in-garbage-out

---

### Phase 2: Training Your First Model

#### What Gets Trained?

The default **raster model** architecture:
- **Input**: Bird's-eye view rasterized images (128x128 pixels)
  - Ego vehicle position
  - Other agents (vehicles, pedestrians)
  - Map features (lanes, intersections)
  - Historical trajectories (past 2 seconds)
- **Output**: Future trajectory (8 seconds, 80 timesteps at 0.1s intervals)
- **Model**: CNN encoder → MLP → trajectory decoder
- **Loss**: L2 distance between predicted and expert trajectory

#### Training Commands

```bash
# Full training (recommended for first model)
just train
# → 10 epochs, 500 scenarios
# → ~20-30 minutes on GPU
# → Output: /tmp/tutorial_nuplan_framework/training_raster_experiment/

# Quick training (for testing/iteration)
just train-quick
# → 3 epochs, 100 scenarios
# → ~5-10 minutes
# → Use for rapid experimentation
```

#### Monitoring Training

```bash
# Terminal 1: Watch logs in real-time
just train-monitor

# Terminal 2: Launch TensorBoard
just tensorboard
# → Open http://localhost:6010
# → Watch loss curves, learning rate, validation metrics
```

#### What to Watch During Training

**Healthy training looks like:**
- 📉 **Loss decreases steadily**: 400+ → 30-50 over 10 epochs
- 📊 **Validation loss follows training loss**: No huge gap (overfitting)
- ⚡ **GPU utilization**: 80-95% (check with `nvidia-smi`)
- 💾 **Checkpoints saved**: `epoch=X-step=Y.ckpt` files created

**Red flags:**
- 🚩 Loss plateaus immediately (learning rate too low)
- 🚩 Loss explodes/NaN (learning rate too high, bad normalization)
- 🚩 Validation loss increases while training decreases (overfitting)
- 🚩 Very slow training (<0.5 it/s on GPU)

#### Post-Training

```bash
# Find your trained checkpoints
just checkpoints

# Typical output:
# [89M] /tmp/.../best_model/epoch=9-step=409.ckpt  ← Use this one!
# [89M] /tmp/.../epoch=5-step=204.ckpt
# [89M] /tmp/.../last.ckpt
```

**Best model selection**: The `best_model` checkpoint is automatically selected based on validation loss during training.

---

### Phase 3: Simulation & Evaluation

Now test your trained planner in closed-loop simulation:

```bash
# Run simulation (auto-detects latest checkpoint)
just simulate-ml

# Or specify a checkpoint explicitly:
just simulate-ml /path/to/checkpoint.ckpt
```

#### What Happens During Simulation?

1. **Scenario loading**: 31 scenarios loaded (4 types × 10 scenarios + 1 extra)
   - `near_multiple_vehicles`: Crowded driving
   - `on_pickup_dropoff`: Drop-off zones with unpredictable behavior
   - `starting_unprotected_cross_turn`: Complex intersections
   - `high_magnitude_jerk`: Requires smooth control

2. **Closed-loop rollout**: For each scenario:
   - Initialize ego vehicle at scenario start
   - At each timestep (0.1s):
     - Observe: ego state, nearby agents, map
     - Plan: ML model predicts 8-second trajectory
     - Execute: Follow first 0.5s of prediction
     - Repeat for 20 seconds

3. **Metric computation**: After each scenario:
   - Safety: Collisions, drivable area violations
   - Comfort: Acceleration, jerk within human limits
   - Progress: Route following, time efficiency

4. **Aggregation**: Combine metrics across all scenarios
   - Pass/fail rates per scenario type
   - Weighted averages
   - Histograms of metric distributions

#### Simulation Performance

**Memory usage**: ~5-6GB main process + 1-2GB per worker
- Default: 4 workers (safe for 16GB RAM)
- Adjust in Justfile: `worker.threads_per_node=2` for 8GB RAM systems

**Runtime**: ~10-20 minutes for 31 scenarios
- Depends on: number of workers, scenario complexity
- Expect: 20-60 seconds per scenario

**Output location**:
```
/tmp/tutorial_nuplan_framework/
  └── simulation_raster_experiment/
      └── open_loop_boxes/
          └── 2025.XX.XX.XX.XX.XX/
              ├── runner_report.parquet  ← Summary
              ├── metric_aggregator.parquet ← Aggregated metrics
              └── scenario_XXXXX/  ← Per-scenario results
```

---

### Phase 4: Analysis & Interpretation

#### Launch nuBoard Dashboard

```bash
# Open visualization dashboard
just nuboard

# Or specify a specific simulation run:
just nuboard "/tmp/tutorial_nuplan_framework/simulation_raster_experiment/open_loop_boxes/2025.XX.XX.XX.XX.XX"
```

**Access**: Open http://localhost:5006 in your browser

#### Dashboard Navigation

**1. Overview Tab** (Top-level summary)
- Total scenarios: 31
- Pass/fail breakdown
- Planner comparison (if multiple planners simulated)
- Aggregate metrics

**2. Scenario List** (Left sidebar)
- Sort by score, scenario type, pass/fail
- Filter by metric violations
- Click to visualize individual scenarios

**3. Scenario Viewer** (Main panel)
- **Bird's-eye view**:
  - Blue box = Your ego vehicle
  - Colored boxes = Other agents
  - Green lines = Lane boundaries
  - Yellow = Intersections
- **Trajectory overlay**:
  - Green trajectory = Safe prediction
  - Red trajectory = Collision/violation
  - Gray = Expert (human) trajectory for reference
- **Timeline scrubber**: Replay scenario frame-by-frame

**4. Metrics Panel** (Right sidebar)
- Detailed metric breakdown
- Per-timestep violations
- Comparison to thresholds

#### Key Metrics Explained

##### Planning Metrics (Safety) 🚨
| Metric | What It Means | Good Target |
|--------|---------------|-------------|
| `ego_at_fault_collisions` | Did your planner cause a crash? | 0 collisions |
| `drivable_area_compliance` | Stayed in legal driving areas? | >95% compliance |
| `driving_direction_compliance` | Correct direction in lane? | >95% compliance |
| `no_ego_at_fault_collisions` | Overall collision safety | 100% pass rate |

**Why it matters**: These determine if your planner is *safe* to deploy. Even 5% collision rate is unacceptable for real AVs.

##### Dynamics Metrics (Comfort) 🛋️
| Metric | What It Means | Good Target | Units |
|--------|---------------|-------------|-------|
| `ego_lat_acceleration` | Sideways acceleration | <4 m/s² | m/s² |
| `ego_lon_acceleration` | Forward/back acceleration | -4 to +3 m/s² | m/s² |
| `ego_lon_jerk` | Jerkiness of acceleration | <4 m/s³ | m/s³ |
| `ego_yaw_acceleration` | Turning smoothness | <2 rad/s² | rad/s² |

**Why it matters**: Passengers experience these as G-forces. High jerk = uncomfortable ride = car sickness.

##### Progress Metrics (Efficiency) 📍
| Metric | What It Means | Good Target |
|--------|---------------|-------------|
| `time_to_collision_within_bound` | Safety margin to others | >3 seconds |
| `ego_progress_along_expert_route` | Following the intended route | >90% |

**Why it matters**: A planner that stops and refuses to move is "safe" but useless. Need balance of safety + progress.

#### Interpreting Your First Results

**Typical first-run performance (10 epochs, 500 scenarios):**

| Metric Category | Expected Pass Rate | Notes |
|-----------------|-------------------|-------|
| Safety (no collisions) | 85-95% | Some collisions expected in complex scenarios |
| Drivable area compliance | 90-98% | May clip lane boundaries in tight turns |
| Comfort (jerk/accel) | 80-90% | Model may be jerky without smoothing |
| Progress | 75-85% | May be overly conservative |

**🎯 Baseline to beat**: Simple rule-based planner typically gets:
- 90% safety
- 85% comfort
- 70% progress

**🏆 Expert-level**: Human expert demonstrations:
- 98% safety
- 95% comfort
- 95% progress

#### Common Failure Patterns

**Pattern 1: Collision with crossing traffic**
- **What**: Planner fails to predict other vehicle will proceed through intersection
- **Why**: Model hasn't learned interaction dynamics
- **Fix**: More training data with multi-agent interactions

**Pattern 2: Drivable area violations at turns**
- **What**: Cuts corners or overshoots lanes
- **Why**: Insufficient map awareness or poor trajectory curvature
- **Fix**: Increase map feature resolution, add lane boundary loss

**Pattern 3: High jerk / jerky motion**
- **What**: Trajectory has sudden direction changes
- **Why**: Model outputs independent waypoints without smoothness constraint
- **Fix**: Add trajectory smoothing, temporal consistency loss

**Pattern 4: Too conservative / gets stuck**
- **What**: Stops and won't proceed in ambiguous situations
- **Why**: Overfitting to cautious demonstrations
- **Fix**: Balance dataset (include assertive behaviors), adjust imitation loss

---

## Understanding Your Results

### Scenario Type Breakdown

#### `near_multiple_vehicles` (10 scenarios)
**Challenge**: Dense traffic, need to predict multiple agent interactions
**Common failures**:
- Collision with merging vehicles
- Cutting off other drivers
**What good looks like**: Smooth lane changes, appropriate gaps, no close calls

#### `on_pickup_dropoff` (10 scenarios)
**Challenge**: Unpredictable pedestrian and vehicle behavior near curb
**Common failures**:
- Getting too close to stopped vehicles
- Not anticipating vehicle pulling out
**What good looks like**: Maintains safe distance, anticipates door openings

#### `starting_unprotected_cross_turn` (10 scenarios)
**Challenge**: Complex intersection with conflicting traffic signals
**Common failures**:
- Proceeding when not safe
- Getting stuck at intersection
**What good looks like**: Assertive but safe gap acceptance, smooth turns

#### `high_magnitude_jerk` (1 scenario)
**Challenge**: Requires very smooth, comfortable control
**Common failures**:
- Jerky acceleration
- Sudden steering changes
**What good looks like**: Sub-threshold jerk throughout scenario

### Benchmark Your Performance

Use this table to gauge progress:

| Stage | Safety (Collision-free) | Comfort (Jerk < limit) | Progress (Route following) | Overall Pass |
|-------|------------------------|------------------------|----------------------------|--------------|
| **Random policy** | ~20% | ~40% | ~10% | ~5% |
| **Simple planner** | 90% | 85% | 70% | 65% |
| **Your first ML (10 epochs)** | 85-95% | 80-90% | 75-85% | 70-80% |
| **Refined ML (30 epochs)** | 92-97% | 90-95% | 85-90% | 82-88% |
| **SOTA research** | 95-98% | 93-97% | 90-95% | 88-93% |
| **Human expert** | 98%+ | 95%+ | 95%+ | 93%+ |

---

## Iteration & Improvement Roadmap

### Stage 1: First Working Model ✅ **(You are here!)**

**Goals:**
- [x] Complete training pipeline working
- [x] Model runs in simulation
- [x] Baseline metrics established

**Next steps**: Analyze failure modes, identify biggest issues

---

### Stage 2: Quick Wins (1-2 sessions)

**Focus**: Low-hanging fruit improvements

#### 2A. More Training (Easiest)
```bash
# Train for 30 epochs instead of 10
# Edit Justfile: lightning.trainer.params.max_epochs=30
just train
```

**Expected improvement**: +3-5% across all metrics
**Why it works**: Model hasn't converged yet, more optimization helps
**Diminishing returns after**: ~50 epochs

#### 2B. More Training Data
```bash
# Increase from 500 to 2000 scenarios
# Edit Justfile: scenario_filter.limit_total_scenarios=2000
just train
```

**Expected improvement**: +5-8% especially on rare scenarios
**Why it works**: More diversity in training data
**Watch out for**: Longer training time (4x), may need larger cache directory

#### 2C. Trajectory Smoothing
Add post-processing to smooth outputs:

```python
# nuplan/planning/simulation/planner/ml_planner/ml_planner.py
def compute_planner_trajectory(self, current_input):
    raw_trajectory = self._model(features)
    smooth_trajectory = self._smooth(raw_trajectory)  # ← Add this
    return smooth_trajectory

def _smooth(self, trajectory, window=5):
    """Moving average smoothing"""
    return scipy.signal.savgol_filter(trajectory, window, polyorder=3)
```

**Expected improvement**: +10-15% comfort metrics
**Why it works**: Removes high-frequency jitter from neural network outputs
**Trade-off**: Slightly less responsive to sudden events

---

### Stage 3: Architecture Improvements (2-4 sessions)

**Focus**: Enhance model capacity and features

#### 3A. Upgrade to Vector Model
```bash
# Train vector-based model (operates on agent states directly)
just cli run_training +training=training_vector_model
```

**Advantages**:
- More precise agent representations
- Better geometry understanding
- Lower memory footprint

**Expected improvement**: +5-10% overall
**Complexity**: Medium (need to understand vector features)

#### 3B. Add Temporal Attention
Modify model to attend over historical states:

```python
class ImprovedRasterModel(nn.Module):
    def __init__(self):
        self.temporal_encoder = nn.TransformerEncoder(...)
        self.spatial_encoder = nn.Conv2d(...)

    def forward(self, x):
        # x shape: [batch, time, channels, H, W]
        temporal_features = self.temporal_encoder(x)
        spatial_features = self.spatial_encoder(x)
        return self.decoder(temporal_features + spatial_features)
```

**Expected improvement**: +8-12% especially in dynamic scenarios
**Why it works**: Better modeling of agent motion patterns over time
**Complexity**: High (need to modify training pipeline)

#### 3C. Multi-Modal Prediction
Instead of single trajectory, predict multiple possible futures:

```python
class MultiModalPlanner(AbstractPlanner):
    def compute_planner_trajectory(self, current_input):
        # Predict K=5 possible trajectories with probabilities
        trajectories, probs = self._model(features)  # [K, T, 3], [K]

        # Select highest probability safe trajectory
        safe_trajs = filter_unsafe(trajectories)
        return safe_trajs[probs.argmax()]
```

**Expected improvement**: +10-15% especially in ambiguous scenarios
**Why it works**: Real driving has multiple valid solutions (e.g., merge left vs stay)
**Complexity**: High (need multi-modal loss function)

---

### Stage 4: Advanced Training Techniques (4-6 sessions)

**Focus**: Optimization and loss engineering

#### 4A. Imitation + Reinforcement Hybrid
```python
# Combine IL (imitation) with RL (reinforcement)
loss = imitation_loss + λ * reinforcement_loss

imitation_loss = L2(predicted_traj, expert_traj)
reinforcement_loss = -reward(collision_free, progress, comfort)
```

**Expected improvement**: +10-20% overall, huge progress gains
**Why it works**: RL optimizes for actual driving metrics, not just mimicking expert
**Complexity**: Very high (need reward engineering, stable RL training)

#### 4B. Adversarial Training
Train discriminator to distinguish expert vs. model trajectories:

```python
generator_loss = L2_loss + adversarial_loss
adversarial_loss = -log(discriminator(model_trajectory))
```

**Expected improvement**: +5-10% in realism, fewer artifacts
**Why it works**: Forces model to match expert distribution, not just average
**Complexity**: High (GAN training instability)

#### 4C. Curriculum Learning
Gradually increase scenario difficulty:

```python
# Week 1: Train on easy scenarios (straight roads)
# Week 2: Add moderate scenarios (simple turns)
# Week 3: Add hard scenarios (complex intersections)
# Week 4: Train on mixed difficulty
```

**Expected improvement**: +5-8% especially on hard scenarios
**Why it works**: Prevents model from being overwhelmed early in training
**Complexity**: Medium (need to categorize scenario difficulty)

---

### Stage 5: Production Readiness (Ongoing)

**Focus**: Robustness, edge cases, deployment

#### 5A. Out-of-Distribution Detection
```python
def compute_planner_trajectory(self, current_input):
    if self._is_ood(current_input):
        return fallback_safe_planner(current_input)
    return self._model(current_input)
```

**Goal**: Detect unusual scenarios and fallback to safe baseline
**Techniques**: Uncertainty estimation, ensemble disagreement, novelty detection

#### 5B. Closed-Loop Fine-Tuning (DAgger)
```python
# 1. Train model on expert demonstrations
# 2. Run model in simulation, collect on-policy data
# 3. Ask expert to correct model mistakes
# 4. Retrain on corrected trajectories
# 5. Repeat
```

**Expected improvement**: +15-25% (eliminates distribution shift)
**Why it works**: Model learns to recover from its own mistakes
**Complexity**: Very high (need interactive expert labeling)

#### 5C. Real-World Validation
- Shadow mode testing on test vehicle
- A/B testing against baseline planner
- Edge case stress testing
- Safety validation (SOTIF ISO 21448)

---

## Common Issues & Solutions

### Issue: Training Loss Not Decreasing

**Symptoms:**
- Loss stays at ~400 across all epochs
- Validation loss equals training loss

**Possible causes:**
1. **Learning rate too low**
   - Fix: Increase LR from 1e-4 to 1e-3
   - Edit: `config/training/training_raster_model.yaml`

2. **Data not loading**
   - Fix: Check cache path is valid
   - Verify: `ls $NUPLAN_DATA_ROOT/nuplan.db`

3. **Model architecture bug**
   - Fix: Check model output shape matches target
   - Debug: Add print statements in forward pass

### Issue: OOM (Out of Memory) During Training

**Symptoms:**
- CUDA OOM error
- Process killed by kernel

**Solutions:**
```yaml
# Reduce batch size (default 8 → 4)
data_loader:
  params:
    batch_size: 4

# Reduce number of workers
data_loader:
  params:
    num_workers: 4  # down from 8

# Enable gradient checkpointing
model:
  use_gradient_checkpointing: true
```

### Issue: OOM During Simulation

**Symptoms:**
- Ray killing workers
- "Running low on memory" errors

**Solutions:**
```bash
# Reduce parallelism (default 4 → 2)
# Edit Justfile: worker.threads_per_node=2

# Close background apps
# Jupyter, CLion, Firefox can use 10GB+ combined

# Use sequential mode for debugging
worker=sequential  # Only 1 scenario at a time
```

### Issue: Disk Space Full

**Symptoms:**
- "No space left on device"
- Training/simulation crashes

**Solutions:**
```bash
# Clean old cache
just clean-tmp

# Clean old experiments
rm -rf /tmp/tutorial_nuplan_framework/training_*/

# Check Ray temp dir
du -sh ~/.tmp/ray
```

### Issue: Hydra Configuration Errors

**Symptoms:**
- "Could not override config"
- "Key not found"

**Solutions:**
```bash
# Debug config resolution
python script.py --cfg job --resolve

# Enable full error messages
export HYDRA_FULL_ERROR=1
```

### Issue: Slow Simulation

**Symptoms:**
- <1 scenario per minute
- Ray worker utilization low

**Solutions:**
```bash
# Increase parallelism (if you have RAM)
worker.threads_per_node=8

# Use GPU for inference
model:
  device: cuda

# Disable expensive metrics
metric:
  disable_heavy_metrics: true
```

---

## Advanced Topics

### Custom Planner Implementation

Create your own planner:

```python
# nuplan/planning/simulation/planner/my_planner.py
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner

class MyPlanner(AbstractPlanner):
    def name(self) -> str:
        return "my_planner"

    def initialize(self, initialization: PlannerInitialization):
        self._map_api = initialization.map_api
        self._model = self._load_model()

    def compute_planner_trajectory(
        self,
        current_input: PlannerInput
    ) -> AbstractTrajectory:
        # 1. Extract features
        ego_state = current_input.history.ego_states[-1]
        agents = current_input.history.observations[-1].tracked_objects

        # 2. Run model
        features = self._build_features(ego_state, agents)
        prediction = self._model(features)

        # 3. Convert to trajectory
        trajectory = self._to_trajectory(prediction, ego_state)

        return trajectory
```

Register in config:
```yaml
# config/planner/my_planner.yaml
_target_: nuplan.planning.simulation.planner.my_planner.MyPlanner
```

### Custom Metrics

Add domain-specific metrics:

```python
# nuplan/planning/metrics/my_metric.py
from nuplan.planning.metrics.abstract_metric import AbstractMetric

class MyMetric(AbstractMetric):
    def name(self) -> str:
        return "my_custom_metric"

    def compute(
        self,
        history: SimulationHistory,
        scenario: AbstractScenario
    ) -> MetricStatistics:
        # Extract data
        ego_trajectory = history.data[SimulationHistoryKey.EGO_STATE]

        # Compute metric
        score = self._compute_score(ego_trajectory)

        return MetricStatistics(
            name=self.name(),
            statistics={
                "score": score,
                "passed": score > threshold
            }
        )
```

### Experiment Tracking

Integrate with MLflow/Weights&Biases:

```python
# Add to training script
import mlflow

with mlflow.start_run():
    mlflow.log_params({
        "learning_rate": cfg.optimizer.lr,
        "batch_size": cfg.batch_size,
        "num_scenarios": cfg.scenario_filter.limit_total_scenarios
    })

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_one_epoch()
        val_loss = validate()

        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss
        }, step=epoch)

    # Log model
    mlflow.pytorch.log_model(model, "model")

    # Log metrics
    simulation_results = run_simulation()
    mlflow.log_metrics({
        "collision_rate": simulation_results.collision_rate,
        "comfort_score": simulation_results.comfort_score
    })
```

---

## Appendix: Quick Reference

### Essential Commands

```bash
# Training
just train              # Full training (10 epochs, 500 scenarios)
just train-quick        # Quick training (3 epochs, 100 scenarios)
just train-monitor      # Watch training logs
just tensorboard        # Launch TensorBoard

# Simulation
just simulate           # Simple planner baseline
just simulate-ml        # ML planner (auto-detects checkpoint)
just nuboard            # Visualization dashboard

# Checkpoints
just checkpoints        # List available checkpoints

# Utilities
just info               # System info
just check-cuda         # Verify GPU
just clean-tmp          # Clean temp files
```

### File Locations

```
Training outputs:
  /tmp/tutorial_nuplan_framework/training_raster_experiment/
    └── train_default_raster/
        └── YYYY.MM.DD.HH.MM.SS/
            ├── best_model/
            │   └── epoch=X-step=Y.ckpt  ← Use this
            └── log.txt

Simulation outputs:
  /tmp/tutorial_nuplan_framework/simulation_raster_experiment/
    └── open_loop_boxes/
        └── YYYY.MM.DD.HH.MM.SS/
            ├── runner_report.parquet
            └── metric_aggregator.parquet

Dataset:
  $NUPLAN_DATA_ROOT/
    ├── nuplan.db
    └── maps/
```

### Metric Thresholds

| Metric | Threshold | Units |
|--------|-----------|-------|
| Collision distance | >0 m | meters |
| Drivable area | >95% time | % |
| Lateral acceleration | <4 m/s² | m/s² |
| Longitudinal acceleration | -4 to +3 m/s² | m/s² |
| Jerk | <4 m/s³ | m/s³ |
| Yaw acceleration | <2 rad/s² | rad/s² |
| Time to collision | >3 s | seconds |

---

**Happy planning! 🚗💨**

For questions or issues, see [CLAUDE.md](CLAUDE.md) for AI assistant guidance or check [GitHub Issues](https://github.com/motional/nuplan-devkit/issues).
