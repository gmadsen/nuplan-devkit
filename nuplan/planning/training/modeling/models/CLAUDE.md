# CLAUDE.md - Training Models Directory (Tier 3)

**Directory**: `nuplan/planning/training/modeling/models/`
**Tier**: 3 (Component Implementation)
**Last Updated**: 2025-11-15

## Purpose

Pre-built neural network architectures for trajectory prediction. Contains 4 complete model implementations (raster, vector, LaneGCN, Urban Driver) plus reusable dynamics layers for physics-based trajectory generation.

## Quick Reference

**Model Selection Guide:**
- **RasterModel**: Image-based, pretrained backbone, easiest to start → `timm.create_model()`
- **VectorMapSimpleMLP**: Lightweight vector baseline, TorchScript-compatible → Fast inference
- **LaneGCN**: Graph neural network, lane topology + attention → Research sota circa 2020
- **UrbanDriverOpenLoopModel**: Transformer-based, configurable, production-ready → L5Kit origin

**When to use dynamics layers:**
- Convert control outputs (accel, steering) → trajectory states
- Enforce physics constraints during training
- Generate differentiable trajectory rollouts

## Architecture Overview

```
models/
├── Model Implementations (4 architectures)
│   ├── raster_model.py              # CNN-based (ResNet, EfficientNet via timm)
│   ├── simple_vector_map_model.py   # MLP baseline (TorchScript)
│   ├── lanegcn_model.py             # GNN with attention layers
│   └── urban_driver_open_loop_model.py  # Transformer (L5Kit adaptation)
│
├── Model Utilities
│   ├── lanegcn_utils.py             # Attention modules, LaneNet GCN
│   └── urban_driver_open_loop_model_utils.py  # PointNet layers, embeddings
│
└── dynamics_layers/                 # Physics-based trajectory generation
    ├── abstract_dynamics.py         # DynamicsLayer interface
    ├── kinematic_bicycle_layer_*.py # Bicycle model variants (rear/center)
    ├── kinematic_unicycle_layer_*.py # Unicycle model variants
    ├── deep_dynamical_system_layer.py # Multi-step rollout wrapper
    └── *_utils.py                   # State/input index enums
```

## Model Implementations

### 1. RasterModel (Simplest - Start Here!)

**File**: `raster_model.py` (85 lines)

**What it does**: Wraps any `timm` CNN model (ResNet, EfficientNet, etc.) to predict trajectories from rasterized BEV images.

**Key features:**
- Single forward pass: `raster.data` → CNN → MLP head → trajectory
- Pretrained ImageNet weights available
- Input: `Raster` feature (H×W×C image)
- Output: `Trajectory` (N poses × 3 features)

**Architecture:**
```python
timm_backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
# Replace classifier/fc layer:
timm_backbone.{classifier|fc} = Linear(backbone_features → num_poses * 3)
```

**Usage example:**
```python
model = RasterModel(
    feature_builders=[RasterFeatureBuilder(...)],
    target_builders=[EgoTrajectoryTargetBuilder(...)],
    model_name="resnet50",  # or efficientnet_b3, convnext_tiny, etc.
    pretrained=True,
    num_input_channels=3,   # ego + agents + map layers
    num_features_per_pose=3,  # (x, y, heading)
    future_trajectory_sampling=TrajectorySampling(...)
)
```

**Critical gotcha**: Must handle both `classifier` and `fc` output layer names (timm inconsistency).

---

### 2. VectorMapSimpleMLP (Lightweight Baseline)

**File**: `simple_vector_map_model.py` (233 lines)

**What it does**: 3-stream MLP (ego + agents + map) with max-pooling aggregation. TorchScript-compatible for deployment.

**Architecture flow:**
```
1. Extract features (per-batch due to variable sizes):
   - Ego trajectory → ego_mlp → [hidden_size]
   - Agent trajectories → agent_mlp → max_pool → [hidden_size]
   - Lane polylines → vectormap_mlp → max_pool → [hidden_size]

2. Concatenate [3 × hidden_size] → final_mlp → trajectory
```

**Key implementation details:**
- **Dual forward paths**: `forward()` for training, `scriptable_forward()` for TorchScript
- **Handles empty inputs**: Zero-padding when no agents/map features present
- **Distributed training workaround**: Always runs agent_mlp (even with 0 agents) to avoid DDP crashes

**Zero-feature handling:**
```python
if vectormap_coords.numel() == 0:
    vectormap_coords = torch.zeros((1, self._vector_map_flatten_lane_coord_dim), ...)
# Still run MLP, then mask output with agents_multiplier
```

---

### 3. LaneGCN (Research-Grade GNN)

**File**: `lanegcn_model.py` (235 lines)
**Utils**: `lanegcn_utils.py` (560 lines - attention modules)

**What it does**: Implements "Learning Lane Graph Representations for Motion Forecasting" (2020). Graph convolutions on lane topology + multi-stage attention (actor↔lane, actor↔actor).

**Architecture stages:**
```
1. Feature extraction:
   - Ego/agents → MLPs → actor_features [feature_dim]
   - Lane graph → LaneNet (multi-scale GCN) → lane_features [feature_dim]

2. Attention fusion (repeated num_attention_layers times):
   - Actor → Lane: Actors attend to nearby lanes (l2a_dist_threshold)
   - Lane → Actor: Lanes attend to nearby actors
   - Actor → Actor: Actors attend to each other (a2a_dist_threshold)

3. Regression:
   - ego_feature (post-attention) → MLP → trajectory
```

**Critical components** (from `lanegcn_utils.py`):
- **GraphAttention**: Distance-thresholded message passing (src → dst nodes)
- **LaneNet**: Multi-scale residual GCN (4 blocks, N scales)
- **Actor2LaneAttention / Lane2ActorAttention**: Cross-domain attention with traffic light + route metadata
- **LinearWithGroupNorm**: Custom layer used throughout

**Empty input handling:**
```python
if not vector_map_data.is_valid:
    # Create single dummy lane at (0, 0) to avoid crashes
    coords = torch.zeros((1, 2, 2), ...)
    connections = {scale: torch.zeros((1, 2), ...) for scale in self.connection_scales}
```

**Distance thresholds** (from paper):
- `l2a_dist_threshold`: 100m (lane-to-actor aggregation)
- `a2a_dist_threshold`: 30m (actor-to-actor interactions)

---

### 4. UrbanDriverOpenLoopModel (Production Transformer)

**File**: `urban_driver_open_loop_model.py` (487 lines)
**Utils**: `urban_driver_open_loop_model_utils.py` (434 lines)

**What it does**: Adapted from L5Kit's "Urban Driver" paper. PointNet-style local subgraphs + global Transformer attention. Production-ready with extensive configuration options.

**When to use:**
- **Use UrbanDriver when**: Need state-of-the-art performance, have sufficient compute, want configurable ablation studies (disable agents/map)
- **Use Raster when**: Quick prototyping, have pretrained CNN backbones, limited feature engineering time
- **Use SimpleMLP when**: Fast inference required, deployment constraints, baseline comparisons
- **Use LaneGCN when**: Lane topology critical, graph structure natural fit, research comparison needed

---

## Urban Driver Open Loop Model (UDOM) - Deep Dive

### Architecture (4-Stage Pipeline)

**Stage 1: Feature Extraction & Reverse Chronological Ordering**
```python
# Extract and pad to fixed sizes
ego_features = extract_ego_features(ego_state_history)  # [batch, total_max_points, ego_dimension]
agent_features = extract_agent_features(agents_history)  # [batch, max_agents, total_max_points, agent_dimension]
map_features = extract_map_features(map_data)           # [batch, sum(max_elements.values()), total_max_points, map_dimension]

# CRITICAL: Reverse to (t_0, t_-1, ..., t_-N) for positional encoding
ego_features = torch.flip(ego_features, dims=[1])
agent_features = torch.flip(agent_features, dims=[2])  # Note: dim 2, not 1!
map_features = torch.flip(map_features, dims=[2])
```

**Stage 2: Embedding (Feature + Positional + Type)**
```python
# Feature embedding: Raw features → local_embedding_size
embedded = self.feature_embedding(features)  # [*, total_max_points, local_embedding_size]

# Add sinusoidal positional encoding
pos_encoding = compute_positional_encoding(total_max_points, local_embedding_size)
embedded += pos_encoding

# Add learned type embeddings (EGO, VEHICLE, PEDESTRIAN, LANE, etc.)
type_emb = self.type_embedding_layer(type_ids)  # [*, local_embedding_size]
embedded += type_emb.unsqueeze(-2)  # Broadcast across points
```

**Stage 3: Local Subgraph Processing (PointNet-style)**
```python
# Repeated num_subgraph_layers times
for layer in self.local_subgraph_layers:
    embedded = layer(embedded, invalid_mask)  # [*, total_max_points, local_embedding_size]
    # Inside layer:
    #   1. MLP on each point
    #   2. Max-pooling across points (with masking)
    #   3. Residual connection
```

**Stage 4: Global Attention (Transformer)**
```python
# Prepare inputs
queries = ego_polyline  # [batch, 1, global_embedding_size]
keys = all_polylines    # [batch, num_polylines, global_embedding_size]
values = all_polylines

# Multi-head attention (ego queries all elements)
attended = self.global_attention(
    queries, keys, values,
    attn_mask=attention_mask  # Mask padding + optionally agents/map
)

# MLP head → trajectory
output = self.output_head(attended)  # [batch, num_output_features]
```

---

### Configuration Deep Dive

**1. UrbanDriverOpenLoopModelParams** (Architecture Hyperparameters)
```python
@dataclass
class UrbanDriverOpenLoopModelParams:
    local_embedding_size: int = 128           # Local subgraph feature size
    global_embedding_size: int = 256          # Global attention feature size
    num_subgraph_layers: int = 3              # PointNet depth
    global_head_dropout: float = 0.0          # Dropout AFTER attention (disabled by default!)
    disable_other_agents: bool = False        # Ablation: remove agents
    disable_map: bool = False                 # Ablation: remove map
    disable_lane_boundaries: bool = False     # Ablation: remove lane boundaries specifically

    def __post_init__(self):
        # Validation: local must divide evenly into global
        assert self.global_embedding_size % self.local_embedding_size == 0, \
            f"global ({self.global_embedding_size}) must be multiple of local ({self.local_embedding_size})"
```

**2. LocalSubGraphParams** (PointNet Configuration)
```python
@dataclass
class LocalSubGraphParams:
    num_layers: int = 3              # Depth of PointNet MLP
    num_pre_layers: int = 1          # Pre-aggregation layers
    residual_layer: int = 1          # Which layer gets residual connection
    aggregation_type: str = "max"    # Max-pooling (only option)
```

**3. GlobalAttentionParams** (Transformer Configuration)
```python
@dataclass
class GlobalAttentionParams:
    num_heads: int = 8                    # Multi-head attention heads
    dropout: float = 0.1                  # Attention dropout (active!)
    use_relative_position: bool = False   # Relative position encoding (unused)
```

**4. UrbanDriverOpenLoopModelFeatureParams** (Feature Extraction)
```python
@dataclass
class UrbanDriverOpenLoopModelFeatureParams:
    # Agent configuration
    agent_features: List[str] = field(default_factory=lambda: [
        "VEHICLE", "PEDESTRIAN", "BICYCLE", "GENERIC_OBJECT"
    ])
    max_agents: Dict[str, int] = field(default_factory=lambda: {
        "VEHICLE": 64, "PEDESTRIAN": 32, "BICYCLE": 16, "GENERIC_OBJECT": 16
    })
    past_trajectory_sampling: TrajectorySampling = ...  # History length

    # Map configuration
    map_features: List[str] = field(default_factory=lambda: [
        "LANE", "LEFT_BOUNDARY", "RIGHT_BOUNDARY", "CROSSWALK", "ROUTE_LANES"
    ])
    max_elements: Dict[str, int] = field(default_factory=lambda: {
        "LANE": 40, "LEFT_BOUNDARY": 20, "RIGHT_BOUNDARY": 20,
        "CROSSWALK": 10, "ROUTE_LANES": 10
    })
    vector_set_map_feature_radius: float = 35.0  # Meters around ego

    # Fixed tensor sizes (CRITICAL!)
    total_max_points: int = 20      # Points per polyline (time history or spatial samples)
    feature_dimension: int = 10     # Features per point (x, y, heading, speed, etc.)

    def __post_init__(self):
        # Validate agent types have max counts
        assert set(self.agent_features) == set(self.max_agents.keys())
        assert set(self.map_features) == set(self.max_elements.keys())
```

**5. UrbanDriverOpenLoopModelTargetParams** (Output Configuration)
```python
@dataclass
class UrbanDriverOpenLoopModelTargetParams:
    num_output_features: int = 3  # (x, y, heading) per pose
    future_trajectory_sampling: TrajectorySampling = ...

    def __post_init__(self):
        # Validate output size matches trajectory
        expected = self.future_trajectory_sampling.num_poses * self.num_output_features
        # This value must match model output head size
```

---

### Feature Requirements (VectorSetMap + GenericAgents)

**CRITICAL**: UrbanDriver requires specific feature builders, **not** the same as LaneGCN!

**Required feature builders:**
```python
from nuplan.planning.training.preprocessing.feature_builders.vector_set_map_feature_builder import VectorSetMapFeatureBuilder
from nuplan.planning.training.preprocessing.feature_builders.generic_agents_feature_builder import GenericAgentsFeatureBuilder

# VectorSetMap (NOT VectorMap!)
vector_map_builder = VectorSetMapFeatureBuilder(
    radius=35.0,  # Match model config!
    longitudinal_offset=0.0,
    map_features={
        "LANE": 40,
        "LEFT_BOUNDARY": 20,
        "RIGHT_BOUNDARY": 20,
        "CROSSWALK": 10,
        "ROUTE_LANES": 10
    }
)

# GenericAgents (NOT Agents!)
agents_builder = GenericAgentsFeatureBuilder(
    trajectory_sampling=past_trajectory_sampling,
    agent_features=["VEHICLE", "PEDESTRIAN", "BICYCLE", "GENERIC_OBJECT"],
    max_agents={
        "VEHICLE": 64,
        "PEDESTRIAN": 32,
        "BICYCLE": 16,
        "GENERIC_OBJECT": 16
    }
)
```

**Why VectorSetMap vs VectorMap?**
- **VectorSetMap**: Returns polylines grouped by type (`Dict[str, torch.Tensor]`)
- **VectorMap**: Returns flat lane graph with connections
- UrbanDriver expects type-separated features for type embeddings

**Reverse chronological ordering option:**
```python
# In feature builder config
past_trajectory_sampling = TrajectorySampling(
    num_poses=10,
    time_horizon=2.0,
    interval_length=0.2,
    reverse_chronological=True  # CRITICAL for UrbanDriver!
)
```

---

### Gotchas & Pitfalls (14+ Issues)

**1. Reverse Chronological Ordering** (CRITICAL!)
```python
# Features expected in order: (t_0, t_-1, t_-2, ..., t_-N)
# But some feature builders return: (t_-N, ..., t_-2, t_-1, t_0)
# Model explicitly flips them:
sample_ego_feature = torch.flip(sample_ego_feature, dims=[1])

# GOTCHA: If feature builder has reverse_chronological=True, DON'T flip again!
# Check feature builder config before modifying model code
```

**2. Feature Dimension Padding Cascades** (3 levels!)
```python
# Level 1: Per-point padding (e.g., 7 features → 10)
point_features = F.pad(point_features, (0, feature_dimension - 7))

# Level 2: Per-polyline padding (e.g., 15 points → 20)
polyline = F.pad(polyline, (0, 0, 0, total_max_points - 15))

# Level 3: Per-batch padding (e.g., 50 agents → 64)
agents = F.pad(agents, (0, 0, 0, 0, 0, max_agents - 50))

# GOTCHA: Mismatch at ANY level → silent truncation or NaN padding
# Validate: ego_dimension, agent_dimension, map_dimension ALL match feature_dimension after extraction
```

**3. Type Embedding Dict Requirements**
```python
# TypeEmbedding class requires these in feature_types dict:
feature_types = {
    "NONE": 0,        # REQUIRED! For padding
    "EGO": 1,         # REQUIRED! For ego trajectory
    "VEHICLE": 2,     # Agent types
    "PEDESTRIAN": 3,
    "LANE": 4,        # Map types
    "CROSSWALK": 5,
    # ...
}

# GOTCHA: Missing "NONE" or "EGO" → AssertionError in __post_init__
# Error message: "feature_types must include NONE and EGO"
```

**4. Empty Agent Handling in DDP** (Same as SimpleMLP)
```python
# When batch has 0 agents of a type, must still run embedding layer
# Otherwise DDP "expected reduction operation" crash

# BAD:
if num_agents > 0:
    agent_features = self.agent_embedding(agents)
else:
    agent_features = torch.zeros(...)  # DDP crash!

# GOOD:
agent_features = self.agent_embedding(agents)  # Always run
agent_features *= float(min(num_agents, 1))    # Mask output
```

**5. VectorSetMap vs VectorMap Incompatibility**
```python
# UrbanDriver expects VectorSetMap:
# - Returns: Dict[str, torch.Tensor] (grouped by type)
# - Keys: "LANE", "LEFT_BOUNDARY", etc.

# VectorMap returns:
# - Flat tensor of all lanes
# - Connection graph (adj matrix)

# GOTCHA: Using VectorMap → KeyError when accessing feature types
# Solution: Check feature builder class, not just config name
```

**6. Availability Mask Logic Inversion**
```python
# Model uses "invalid_mask" (True = invalid/padding)
invalid_mask = (features == 0).all(dim=-1)  # True for padding

# But some code uses "avails" (True = valid)
avails = (features != 0).any(dim=-1)

# GOTCHA: Inverting logic → attention on padding, ignoring valid data!
# Symptom: Model performs terribly despite converging (NaN gradients eventually)
```

**7. Multi-Scale Positional Encoding**
```python
# Combines TWO positional encodings:
# 1. Sinusoidal (temporal position in polyline)
pos_encoding = compute_sinusoidal_encoding(total_max_points, embedding_size)

# 2. Type embedding (EGO vs VEHICLE vs LANE, etc.)
type_emb = self.type_embedding(type_id)

# Final: feature_emb + pos_encoding + type_emb
# GOTCHA: Type embedding dimension must match local_embedding_size exactly
# Off-by-one → broadcasting error
```

**8. Global Head Dropout Disabled by Default**
```python
# global_head_dropout: float = 0.0  # Note: 0.0!
# Attention dropout is enabled (0.1), but output dropout disabled

# GOTCHA: Expecting regularization from output dropout → overfitting
# Solution: Set global_head_dropout > 0 for small datasets
```

**9. Ego Always Enabled in Attention Mask**
```python
# Even with disable_other_agents=True or disable_map=True
# Ego polyline ALWAYS participates in attention

attention_mask[0] = False  # Ego never masked

# GOTCHA: Can't ablate ego for sanity checks
# Workaround: Zero out ego features before forward pass
```

**10. num_output_features Calculation Validation**
```python
# Must match exactly:
num_output_features = future_trajectory_sampling.num_poses * Trajectory.state_size()

# Example: 12 poses × 3 features/pose = 36
# Model output head: Linear(global_embedding_size, 36)

# GOTCHA: Mismatch → shape error when converting to Trajectory
# Symptom: "RuntimeError: shape [batch, 36] cannot be reshaped to [batch, 12, 3]"
```

**11. Config Parameter Validation in __post_init__**
```python
# All dataclasses have validation logic:
def __post_init__(self):
    assert self.global_embedding_size % self.local_embedding_size == 0
    assert set(self.agent_features) == set(self.max_agents.keys())
    # ...

# GOTCHA: AssertionError messages sometimes unclear
# Example: "assertion failed" (which one?)
# Solution: Add explicit error messages to asserts
```

**12. Fixed Tensor Sizes - total_max_points, max_agents CRITICAL**
```python
# These determine memory allocation:
total_max_points = 20  # MUST exceed all polyline lengths
max_agents = 64        # MUST exceed agent count in scenarios

# GOTCHA: Scenario with 21 points → silent truncation (last point dropped)
# GOTCHA: Scenario with 65 agents → crashes or drops agents randomly
# Solution: Profile dataset first, add 20% buffer
```

**13. L5Kit API Differences from nuPlan**
```python
# L5Kit original uses:
# - Zarr datasets (not SQLite)
# - Different coordinate frames
# - Rasterizer-based features

# nuPlan adaptation changes:
# - VectorSetMapFeatureBuilder (new)
# - GenericAgentsFeatureBuilder (new)
# - PyTorch Lightning training loop

# GOTCHA: L5Kit examples don't work directly
# Solution: Use nuPlan tutorials, not L5Kit docs
```

**14. TorchScript Incompatibility - TypeEmbedding Dict[str, int]**
```python
# TypeEmbedding class:
def __init__(self, feature_types: Dict[str, int], embedding_dim: int):
    self.feature_types = feature_types  # Dict doesn't serialize!

# GOTCHA: torch.jit.script(model) → error
# Error: "Dict[str, int] not supported in TorchScript"

# Workaround: Convert to Enum before init, store as int keys
feature_type_enum = IntEnum("FeatureType", feature_types)
```

---

### Comparison with Other Models

| Feature | RasterModel | SimpleMLP | LaneGCN | UrbanDriver |
|---------|-------------|-----------|---------|-------------|
| **Input Type** | Raster (BEV image) | Vector (ego/agents/map) | Vector (lane graph) | Vector (polylines) |
| **Architecture** | CNN (ResNet/EfficientNet) | 3-stream MLP | GNN + Attention | PointNet + Transformer |
| **Pretraining** | ✅ ImageNet | ❌ None | ❌ None | ❌ None |
| **TorchScript** | ❌ No | ✅ Yes (dual forward) | ❌ No | ❌ No (Dict[str, int]) |
| **Empty Input Handling** | N/A (always BEV) | ✅ Zero padding | ⚠️ Dummy lane | ✅ Availability masks |
| **Agent/Map Ablation** | ❌ No (baked into raster) | ❌ No | ❌ No | ✅ Yes (config flags) |
| **Computational Cost** | High (CNN) | Low (fastest) | Medium (GNN) | High (Transformer) |
| **Memory Usage** | ~8 GB | ~4 GB | ~12 GB | ~16 GB |
| **Training Time** | ~20 min/epoch | ~5 min/epoch | ~30 min/epoch | ~40 min/epoch |
| **Performance** | Good (pretrain helps) | Baseline | Research sota 2020 | SOTA 2021+ |
| **Configuration Complexity** | Low (5 params) | Low (3 params) | Medium (10 params) | High (20+ params) |
| **Best For** | Quick prototyping | Fast inference | Lane-centric scenarios | Production + research |

---

### Usage Examples

**Example 1: Basic Training Configuration**
```python
from nuplan.planning.training.modeling.models.urban_driver_open_loop_model import (
    UrbanDriverOpenLoopModel,
    UrbanDriverOpenLoopModelParams,
    UrbanDriverOpenLoopModelFeatureParams,
    UrbanDriverOpenLoopModelTargetParams,
)

# Define sampling
past_sampling = TrajectorySampling(num_poses=10, time_horizon=2.0, interval_length=0.2)
future_sampling = TrajectorySampling(num_poses=12, time_horizon=6.0, interval_length=0.5)

# Architecture params
model_params = UrbanDriverOpenLoopModelParams(
    local_embedding_size=128,
    global_embedding_size=256,
    num_subgraph_layers=3,
    global_head_dropout=0.1,  # Enable dropout for regularization
    disable_other_agents=False,
    disable_map=False,
)

# Feature params
feature_params = UrbanDriverOpenLoopModelFeatureParams(
    agent_features=["VEHICLE", "PEDESTRIAN"],
    max_agents={"VEHICLE": 64, "PEDESTRIAN": 32},
    map_features=["LANE", "ROUTE_LANES"],
    max_elements={"LANE": 40, "ROUTE_LANES": 10},
    total_max_points=20,
    feature_dimension=10,
    past_trajectory_sampling=past_sampling,
    vector_set_map_feature_radius=35.0,
)

# Target params
target_params = UrbanDriverOpenLoopModelTargetParams(
    num_output_features=3,  # (x, y, heading)
    future_trajectory_sampling=future_sampling,
)

# Instantiate model
model = UrbanDriverOpenLoopModel(
    model_params=model_params,
    feature_params=feature_params,
    target_params=target_params,
    feature_builders=[vector_map_builder, agents_builder],
    target_builders=[trajectory_target_builder],
)
```

**Example 2: Custom Feature Configuration**
```python
# Scenario: High-speed highway (fewer pedestrians, more map detail)
feature_params_highway = UrbanDriverOpenLoopModelFeatureParams(
    agent_features=["VEHICLE"],  # Only vehicles
    max_agents={"VEHICLE": 128},  # Double capacity
    map_features=["LANE", "LEFT_BOUNDARY", "RIGHT_BOUNDARY", "ROUTE_LANES"],
    max_elements={
        "LANE": 60,              # More lanes for complex interchanges
        "LEFT_BOUNDARY": 30,
        "RIGHT_BOUNDARY": 30,
        "ROUTE_LANES": 20,
    },
    total_max_points=30,         # Longer history for high-speed
    feature_dimension=12,        # Extra features (speed, accel)
    vector_set_map_feature_radius=50.0,  # Larger radius for speed
)
```

**Example 3: Ablation Studies**
```python
# Ablation 1: No agents (ego + map only)
model_no_agents = UrbanDriverOpenLoopModel(
    model_params=UrbanDriverOpenLoopModelParams(disable_other_agents=True),
    feature_params=feature_params,
    target_params=target_params,
    # ...
)

# Ablation 2: No map (ego + agents only)
model_no_map = UrbanDriverOpenLoopModel(
    model_params=UrbanDriverOpenLoopModelParams(disable_map=True),
    feature_params=feature_params,
    target_params=target_params,
    # ...
)

# Ablation 3: Ego only (baseline)
model_ego_only = UrbanDriverOpenLoopModel(
    model_params=UrbanDriverOpenLoopModelParams(
        disable_other_agents=True,
        disable_map=True
    ),
    feature_params=feature_params,
    target_params=target_params,
    # ...
)

# Compare performance:
# - Ego-only: Baseline (inertial prediction)
# - Ego+map: Tests map understanding
# - Ego+agents: Tests interaction modeling
# - Full: Tests integration of all modalities
```

---

### Cross-References

**Feature Builders (CRITICAL DEPENDENCIES):**
- `nuplan/planning/training/preprocessing/feature_builders/vector_set_map_feature_builder.py` - See CLAUDE.md for configuration
- `nuplan/planning/training/preprocessing/feature_builders/generic_agents_feature_builder.py` - See CLAUDE.md for agent type mapping

**Features (Data Structures):**
- `nuplan/planning/training/preprocessing/features/vector_set_map.py` - VectorSetMap feature spec
- `nuplan/planning/training/preprocessing/features/generic_agents.py` - GenericAgents feature spec

**Related Models:**
- See RasterModel (above) for CNN-based alternative
- See SimpleMLP (above) for lightweight baseline
- See LaneGCN (above) for graph-based alternative

**Configuration Files (Hydra):**
- `config/training/model/urban_driver_open_loop_model.yaml` - Default model config
- `config/training/feature_builder/vector_set_map_feature_builder.yaml` - Map features
- `config/training/feature_builder/generic_agents_feature_builder.yaml` - Agent features

**Research Papers:**
- Original L5Kit paper: "Urban Driver: Learning to Drive from Real-world Demonstrations Using Policy Gradients"
- PointNet: "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation"
- Transformer: "Attention Is All You Need"

---

## Dynamics Layers (Physics-Based Trajectory Generation)

**Purpose**: Convert control sequences → state trajectories with differentiable physics models.

### Abstract Interface

**File**: `abstract_dynamics.py` (53 lines)

```python
class DynamicsLayer(ABC, nn.Module):
    @abstractmethod
    def forward(
        initial_state: torch.FloatTensor,  # [..., state_dim()]
        controls: torch.FloatTensor,        # [..., input_dim()]
        timestep: float,
        vehicle_parameters: torch.FloatTensor  # [..., 1 or 2] (length, width)
    ) -> torch.FloatTensor:  # [..., state_dim()]
```

**Flexible batch dimensions**: Supports single/multi-batch, single/multi-vehicle (ellipsis notation).

### Kinematic Bicycle Model

**File**: `kinematic_bicycle_layer_rear_axle.py` (89 lines)

**State**: (x, y, yaw, vel_x, vel_y, yaw_rate) - 6D
**Control**: (acceleration, steering_angle) - 2D

**Kinematics (Forward Euler):**
```python
vel = vel_init + accel * dt
yaw_rate = vel_init * tan(steering_angle) / wheelbase
yaw = yaw + yaw_rate * dt
x = x + vel_x * dt
y = y + vel_y * dt
```

**Critical note**: Forward Euler discretization means inputs at t=0 affect position at t=2 (one-step lag).

**Variants:**
- `kinematic_bicycle_layer_rear_axle.py`: Reference point at rear axle
- `kinematic_bicycle_layer_geometric_center.py`: Reference at vehicle center (different kinematics)

### Kinematic Unicycle Model

**Files**: `kinematic_unicycle_layer_rear_axle.py`

**State**: (x, y, yaw, vel) - 4D (simpler than bicycle)
**Control**: (acceleration, yaw_rate) - 2D

**Use case**: Simplified dynamics when steering angle not needed.

### Multi-Step Rollout

**File**: `deep_dynamical_system_layer.py` (60 lines)

**What it does**: Wraps any `DynamicsLayer` to roll out k timesteps.

**Usage:**
```python
dynamics = KinematicBicycleLayerRearAxle()
rollout_layer = DeepDynamicalSystemLayer(dynamics)

trajectory = rollout_layer(
    initial_state,   # [..., 6]
    controls,        # [..., k, 2]
    timestep=0.1,
    vehicle_params
)  # → [..., k, 6]
```

**Implementation**: Simple for-loop with state propagation:
```python
for i in range(k):
    initial_state = self.dynamics(initial_state, controls[..., i, :], dt, params)
    xout[..., i, :] = initial_state
```

---

## Common Patterns

### 1. Prediction → Trajectory Conversion

All models use this helper (slight variations):
```python
def convert_predictions_to_trajectory(predictions: torch.Tensor) -> torch.Tensor:
    """predictions: [batch, num_poses * state_size] → [batch, num_poses, state_size]"""
    num_batches = predictions.shape[0]
    return predictions.view(num_batches, -1, Trajectory.state_size())
```

### 2. Per-Sample Feature Extraction

Variable-size inputs (agents, map) require batch-loop:
```python
for sample_idx in range(batch_size):
    sample_ego_feature = self.ego_mlp(ego[sample_idx].view(1, -1))
    sample_agents = process_agents(agents[sample_idx])  # Handle variable count
    sample_map = process_map(map_coords[sample_idx])    # Handle variable lanes
    features.append(torch.cat([sample_ego_feature, ...]))
```

### 3. Empty Input Handling (Critical!)

**Why**: Scenarios may have no agents or minimal map features. Must avoid:
- Division by zero in max-pooling
- Empty tensor operations
- NaN gradients

**Strategies:**
- **Zero tensor replacement**: Create single dummy element at origin
- **Multiplier masking**: Run network but multiply output by 0
- **Availability tensors**: Track valid vs padded elements

**Example (LaneGCN):**
```python
if not vector_map_data.is_valid:
    coords = torch.zeros((1, 2, 2), device=..., dtype=...)  # Single dummy lane
    connections = {scale: torch.zeros((1, 2), ...) for scale in scales}
else:
    coords = vector_map_data.coords[sample_idx]
```

**Example (SimpleMLP):**
```python
agents_multiplier = float(min(agents.shape[1], 1))  # 0 if no agents, 1 otherwise
agent_feature = self.agent_mlp(agents)
agent_feature *= agents_multiplier  # Zero out if no agents
```

### 4. TorchScript Compatibility (SimpleVectorMapMLP)

**Pattern**: Dual forward methods
```python
@torch.jit.unused
def forward(self, features: FeaturesType) -> TargetsType:
    # Unpack features from nuPlan types
    tensor_inputs = {...}
    list_tensor_inputs = {...}

    # Call scriptable core
    output_tensors, _, _ = self.scriptable_forward(tensor_inputs, list_tensor_inputs, {})

    # Repack to nuPlan types
    return {"trajectory": Trajectory(data=output_tensors["trajectory"])}

@torch.jit.export
def scriptable_forward(
    self,
    tensor_data: Dict[str, torch.Tensor],
    list_tensor_data: Dict[str, List[torch.Tensor]],
    list_list_tensor_data: Dict[str, List[List[torch.Tensor]]]
) -> Tuple[...]:
    # Pure tensor operations (TorchScript-compatible)
    ...
```

---

## Critical Gotchas & Pitfalls (14+)

### Model Architecture

1. **timm output layer inconsistency** (RasterModel)
   - Some models use `classifier`, others use `fc` for final layer
   - Must check `hasattr()` and handle both: `if hasattr(model, 'classifier'): ...`
   - Missing check → AttributeError at runtime

2. **Trajectory state_size hardcoded assumptions**
   - Most models assume `Trajectory.state_size() = 3` (x, y, heading)
   - Dynamics layers assume 6 (x, y, yaw, vel_x, vel_y, yaw_rate)
   - Mismatch → shape errors in prediction reshaping

3. **Forward Euler discretization lag** (Kinematic layers)
   - Input at t=0 affects state at t=2 (one-step delay)
   - Can cause off-by-one errors when matching ground truth
   - Mentioned in docstring but easy to miss

4. **Reverse chronological ordering** (UrbanDriver)
   - Features flipped: `(t_0, t_-1, ..., t_-N)` for positional encoding
   - Easy to forget when preprocessing custom features
   - No runtime check → silent performance degradation

### Empty Input Handling

5. **Empty map features crash LaneGCN**
   - `vector_map_data.is_valid=False` when no lanes extracted
   - Must create dummy lane at (0,0) with fake connections
   - Missing check → IndexError in graph operations

6. **Zero agents in distributed training** (SimpleMLP)
   - Skipping agent_mlp when no agents → DDP "expected reduction" crash
   - **Solution**: Always run MLP, use multiplier mask: `feature *= float(min(count, 1))`
   - Only occurs in multi-GPU training, hard to debug

7. **Max-pooling over empty tensors**
   - `torch.max(empty_tensor, dim=0)` → RuntimeError
   - Must check `.numel() == 0` and substitute zeros
   - Affects all models with variable-size inputs

8. **Availability mask inconsistencies**
   - UrbanDriver uses `invalid_mask` (True = invalid)
   - Some code uses `avails` (True = valid)
   - Easy to invert logic → model attends to padding!

### Configuration & Feature Extraction

9. **Feature dimension padding mismatches** (UrbanDriver)
   - `ego_dimension`, `agent_dimension`, `feature_dimension` must align
   - Padding applied at multiple stages (per-point, per-polyline, per-batch)
   - Off-by-one in any dimension → silent truncation or NaN padding

10. **Multi-scale connections indexing** (LaneGCN)
    - `conns` dict keyed by scale: `conns[scale]` → `[num_connections, 2]`
    - Scale can be 0-indexed or 1-indexed depending on config
    - Wrong scale → KeyError or wrong graph topology

11. **Type embedding feature_types dict** (UrbanDriver)
    - Must include `"NONE"` and `"EGO"` or AssertionError at init
    - Agent/map feature names must match exactly (case-sensitive)
    - Post-init validation catches this, but error message unclear

12. **Distance threshold units** (LaneGCN attention)
    - Thresholds in **meters**, but coordinates might be in different units
    - Default 100m for l2a, 30m for a2a (from paper)
    - Too small → no edges, model can't learn
    - Too large → O(N²) memory explosion

### Training & Deployment

13. **TorchScript Dict[str, int] serialization** (UrbanDriver TypeEmbedding)
    - `feature_types: Dict[str, int]` doesn't serialize well
    - Workaround: Convert enum → dict at init
    - Breaks if trying to use Enum directly

14. **Batch size 1 edge case** (GroupNorm in LaneGCN)
    - `nn.GroupNorm(num_groups, num_channels)` with `num_groups > num_channels` → error
    - Fixed via `gcd(num_groups, num_channels)` in `LinearWithGroupNorm`
    - Only manifests with very small feature dims

15. **Dynamics layer vehicle_parameters shape**
    - Can be `[..., 1]` (length only) or `[..., 2]` (length + width)
    - Width unused in kinematic models but expected in shape checks
    - Pass `vehicle_params[..., :1]` or `[..., 0]` depending on implementation

---

## Dependency Reference (External/Internal Mapping)

**External libraries:**
- `torch` / `torch.nn` - Core PyTorch
- `timm` - Model zoo for RasterModel (ResNet, EfficientNet, etc.)
- `torch.nn.functional` - F.relu, F.normalize

**Internal nuPlan dependencies** (listed, not read):
- `nuplan.planning.simulation.trajectory.trajectory_sampling.TrajectorySampling`
- `nuplan.planning.training.modeling.torch_module_wrapper.TorchModuleWrapper`
- `nuplan.planning.training.modeling.scriptable_torch_module_wrapper.ScriptableTorchModuleWrapper`
- `nuplan.planning.training.modeling.types.FeaturesType`, `.TargetsType`
- `nuplan.planning.training.preprocessing.feature_builders.*` (Raster, Agents, VectorMap, VectorSetMap, GenericAgents)
- `nuplan.planning.training.preprocessing.features.*` (Raster, Agents, Trajectory, VectorMap, VectorSetMap, GenericAgents)
- `nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder.EgoTrajectoryTargetBuilder`
- `nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils.LaneOnRouteStatusData`, `.LaneSegmentTrafficLightData`

---

## Usage Examples

### Training RasterModel (Quickstart)

```python
from nuplan.planning.training.modeling.models.raster_model import RasterModel
from nuplan.planning.training.preprocessing.feature_builders.raster_feature_builder import RasterFeatureBuilder
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import EgoTrajectoryTargetBuilder

model = RasterModel(
    feature_builders=[RasterFeatureBuilder(...)],
    target_builders=[EgoTrajectoryTargetBuilder(future_trajectory_sampling)],
    model_name="resnet50",
    pretrained=True,
    num_input_channels=3,
    num_features_per_pose=3,
    future_trajectory_sampling=TrajectorySampling(num_poses=12, time_horizon=6.0)
)

# Training loop
for batch in dataloader:
    predictions = model(batch["features"])  # {"trajectory": Trajectory}
    loss = criterion(predictions["trajectory"].data, batch["targets"]["trajectory"].data)
```

### Using Dynamics Layer for Control-Based Planning

```python
from nuplan.planning.training.modeling.models.dynamics_layers.kinematic_bicycle_layer_rear_axle import KinematicBicycleLayerRearAxle
from nuplan.planning.training.modeling.models.dynamics_layers.deep_dynamical_system_layer import DeepDynamicalSystemLayer

# Single-step prediction
dynamics = KinematicBicycleLayerRearAxle()
next_state = dynamics(
    initial_state=ego_state,  # [batch, 6]
    controls=control_input,   # [batch, 2]
    timestep=0.1,
    vehicle_parameters=torch.tensor([[4.5]])  # wheelbase
)

# Multi-step rollout
rollout = DeepDynamicalSystemLayer(dynamics)
trajectory = rollout(
    initial_state=ego_state,     # [batch, 6]
    controls=control_sequence,   # [batch, 12, 2] (12 timesteps)
    timestep=0.1,
    vehicle_parameters=torch.tensor([[4.5]])
)  # → [batch, 12, 6]
```

### Custom Model Integration Pattern

```python
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper

class MyCustomModel(TorchModuleWrapper):
    def __init__(self, feature_builders, target_builders, future_trajectory_sampling, **kwargs):
        super().__init__(
            feature_builders=feature_builders,
            target_builders=target_builders,
            future_trajectory_sampling=future_trajectory_sampling
        )
        # Your architecture here
        self._encoder = ...
        self._decoder = ...

    def forward(self, features: FeaturesType) -> TargetsType:
        # Extract features (handle variable sizes per sample!)
        encoded = self._encoder(features)
        predictions = self._decoder(encoded)

        return {"trajectory": Trajectory(data=convert_predictions_to_trajectory(predictions))}
```

---

## Testing & Validation Notes

**Test coverage**: Each model has corresponding test file in `test/` subdirectory.

**Key test scenarios:**
- Empty agents/map inputs (must not crash)
- Batch size 1 edge case (GroupNorm)
- Variable-size inputs across batch samples
- TorchScript serialization (SimpleVectorMapMLP)
- Gradient flow through dynamics layers

**Manual checks when adding new model:**
1. Handle empty inputs gracefully
2. Check output shape: `[batch, num_poses, 3]`
3. Verify gradient backprop through all paths
4. Test with batch_size=1 and batch_size=32
5. Profile memory usage with max agents/map features

---

## Performance Characteristics

**Model complexity** (relative):
- RasterModel: High (CNN backbone), but benefits from ImageNet pretraining
- SimpleMLP: Low (fastest inference, smallest memory)
- LaneGCN: Medium-High (graph ops can be slow, attention is O(N²))
- UrbanDriver: High (Transformer attention, large embedding dims)

**Typical training times** (on 1x A100, mini dataset):
- RasterModel: ~20 min/epoch
- SimpleMLP: ~5 min/epoch
- LaneGCN: ~30 min/epoch
- UrbanDriver: ~40 min/epoch

**Memory usage** (approximate peak):
- RasterModel: ~8 GB (depends on image size)
- SimpleMLP: ~4 GB
- LaneGCN: ~12 GB (graph connectivity)
- UrbanDriver: ~16 GB (large embeddings + attention)

---

## Migration & Compatibility Notes

**L5Kit origin** (UrbanDriver):
- Adapted from `l5kit.planning.vectorized.open_loop_model`
- Changes: nuPlan features, PyTorch Lightning integration
- Original paper: "Urban Driver: Learning to Drive from Real-world Demonstrations Using Policy Gradients"

**LaneGCN origin**:
- Based on "Learning Lane Graph Representations for Motion Forecasting" (2020)
- Implementation differences: Some LayerNorm operations omitted vs. original paper

**TorchScript support**:
- SimpleMLP: Full support (dual forward pattern)
- Others: Limited (use standard torch.jit.script with caution)

---

## Cross-References

**Related documentation:**
- See `../CLAUDE.md` (parent modeling directory) for training loop integration
- See `../../preprocessing/feature_builders/CLAUDE.md` for feature extraction
- See `../../preprocessing/features/CLAUDE.md` for feature tensor specs

**Used by:**
- `nuplan/planning/script/run_training.py` - Main training entry point
- `nuplan/planning/training/experiments/` - Experiment configs via Hydra

**Configuration files** (Hydra):
- `config/training/model/raster_model.yaml`
- `config/training/model/simple_vector_map_model.yaml`
- `config/training/model/lanegcn_model.yaml`
- `config/training/model/urban_driver_open_loop_model.yaml`

---

**Navigator 🧭 says**: Start with RasterModel for quick experiments (pretrained backbones!), graduate to SimpleMLP for fast iteration, then explore LaneGCN/UrbanDriver when you need sota performance. The dynamics layers are gold for control-based planners - differentiable physics FTW! 🚗⚡
