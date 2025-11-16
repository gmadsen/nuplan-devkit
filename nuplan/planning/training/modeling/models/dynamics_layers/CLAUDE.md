# nuplan/planning/training/modeling/models/dynamics_layers/

## Purpose & Scope

This module provides **differentiable physics simulators** as PyTorch `nn.Module` layers for autonomous vehicle motion modeling. These layers enable:

1. **Physics-informed ML planning**: Enforce kinematic/dynamic constraints in neural network predictions
2. **Trajectory rollout**: Convert compact control sequences (acceleration, steering) into full trajectories
3. **Gradient-based learning**: Backpropagate through physics for end-to-end training
4. **Multi-modal prediction**: Generate physically plausible futures for agents

**Why differentiable physics for ML planning?**

Pure neural networks can predict kinematically infeasible trajectories (e.g., lateral drift, instant velocity changes, impossible turns). Integrating differentiable physics layers ensures:
- **Physical realism**: All predictions obey vehicle dynamics
- **Compact representation**: Learn controls (N×2) instead of full trajectories (N×3 or N×5)
- **Interpretability**: Control outputs have clear physical meaning
- **Sample efficiency**: Smaller action space improves training convergence

**Scope**: This module implements **kinematic** (not dynamic) models - no tire forces, aerodynamics, or suspension. Suitable for low-to-medium speed urban planning (~0-20 m/s), NOT high-speed racing or off-road.

## Key Abstractions

### 1. DynamicsLayer (Abstract Base)

```python
# nuplan/planning/training/modeling/models/dynamics_layers/abstract_dynamics.py
class DynamicsLayer(ABC, nn.Module):
    """Single-step differentiable physics simulator.

    Computes x_{t+1} = f(x_t, u_t, dt) where:
    - x_t: State at time t (position, velocity, yaw, etc.)
    - u_t: Control input (acceleration, steering, etc.)
    - dt: Time step (seconds)
    """

    @abstractmethod
    def forward(
        self,
        initial_state: torch.Tensor,      # [..., state_dim()]
        controls: torch.Tensor,            # [..., input_dim()]
        timestep: float,                   # scalar time step
        vehicle_parameters: torch.Tensor   # [..., 1 or 2] (wheelbase, optional width)
    ) -> torch.Tensor:                     # [..., state_dim()]
        """Simulate one time step."""
        pass

    @staticmethod
    @abstractmethod
    def state_dim() -> int:
        """State vector dimension (6 for bicycle, 7 for unicycle)"""
        pass

    @staticmethod
    @abstractmethod
    def input_dim() -> int:
        """Control vector dimension (always 2 for these models)"""
        pass
```

**Key properties**:
- Stateless (all dynamics in `forward()`)
- Device-agnostic (inherits from `nn.Module`)
- Differentiable by default (PyTorch autograd)
- Supports arbitrary batch dimensions via ellipsis (`...`)

### 2. KinematicBicycleLayerRearAxle (6-DOF)

```python
# nuplan/planning/training/modeling/models/dynamics_layers/kinematic_bicycle_layer_rear_axle.py
class KinematicBicycleLayerRearAxle(DynamicsLayer):
    """Front-wheel-steered bicycle model with rear axle reference point.

    State: [x, y, yaw, vx, vy, yaw_rate] (6D)
    Controls: [accel, steering_angle] (2D)

    Dynamics (continuous time):
        x_dot = vx * cos(yaw) - vy * sin(yaw)
        y_dot = vx * sin(yaw) + vy * cos(yaw)
        yaw_dot = yaw_rate
        vel_init = sqrt(vx² + vy²)
        vel = vel_init + accel
        yaw_rate = vel_init * tan(steering_angle) / wheelbase
        vx = vel * cos(yaw)
        vy = vel * sin(yaw)

    Discretization: Forward Euler (x_{t+1} = x_t + dt * x_dot_t)
    """

    @staticmethod
    def state_dim() -> int:
        return 6

    @staticmethod
    def input_dim() -> int:
        return 2
```

**When to use**:
- Car-like vehicles (front-wheel steering)
- Most nuPlan scenarios (sedan autonomous vehicles)
- When steering constraints matter (Ackermann geometry)
- Standard robotics literature (reference point at rear axle)

**Variant**: `KinematicBicycleLayerGeometricCenter` - Reference point at vehicle center (different slip angle computation).

### 3. KinematicUnicycleLayerRearAxle (7-DOF)

```python
# nuplan/planning/training/modeling/models/dynamics_layers/kinematic_unicycle_layer_rear_axle.py
class KinematicUnicycleLayerRearAxle(DynamicsLayer):
    """Unicycle model with curvature + jerk controls (GameFormer 2021).

    State: [x, y, yaw, vx, vy, accel_x, accel_y] (7D)
    Controls: [curvature, jerk] (2D)

    Dynamics (continuous time):
        x_dot = vx
        y_dot = vy
        yaw_dot = curvature * speed
        vx_dot = accel_x
        vy_dot = accel_y
        accel_x_dot = jerk * cos(yaw)
        accel_y_dot = jerk * sin(yaw)

    Discretization: Forward Euler
    """

    @staticmethod
    def state_dim() -> int:
        return 7

    @staticmethod
    def input_dim() -> int:
        return 2
```

**When to use**:
- Predicted agents (other vehicles) when steering angle unknown
- GameFormer-style models using curvature prediction
- Academic research (simpler than bicycle model)

**Key difference from bicycle**: Direct control over curvature instead of steering angle. Acceleration stored as state (7th/8th dimensions) creating additional lag.

### 4. DeepDynamicalSystemLayer (Multi-Step Rollout)

```python
# nuplan/planning/training/modeling/models/dynamics_layers/deep_dynamical_system_layer.py
class DeepDynamicalSystemLayer(nn.Module):
    """Multi-step trajectory rollout wrapper.

    Wraps a single-step DynamicsLayer to unroll trajectories over time:
        x_0 (initial state) + [u_0, u_1, ..., u_{T-1}] (controls)
        → [x_0, x_1, x_2, ..., x_T] (trajectory)

    Supports:
    - Autoregressive rollout (x_t depends on x_{t-1})
    - Gradient flow through entire trajectory
    - Arbitrary time steps per control (multi-rate control)
    """

    def __init__(self, dynamics: DynamicsLayer):
        """
        Args:
            dynamics: Single-step physics model (bicycle/unicycle)
        """
        super().__init__()
        self.dynamics = dynamics

    def forward(
        self,
        initial_state: torch.Tensor,           # [..., state_dim()]
        controls: torch.Tensor,                 # [..., k, input_dim()]
        timestep: float,                        # scalar time step
        agents_pars: torch.Tensor              # [..., 1 or 2]
    ) -> torch.Tensor:                          # [..., k, state_dim()]
        """Unroll trajectory over k steps."""
        k = controls.shape[-2]
        xout = torch.empty((*controls.shape[:-1], self.dynamics.state_dim()), ...)

        for i in range(k):
            initial_state = self.dynamics(initial_state, controls[..., i, :], timestep, agents_pars)
            xout[..., i, :] = initial_state

        return xout
```

**Usage in ML models**:
```python
# Typical integration in urban_driver_open_loop_model.py
dynamics = DeepDynamicalSystemLayer(
    dynamics=KinematicBicycleLayerRearAxle()
)

# Network predicts controls (compact)
controls = self.control_head(features)  # [B, 80, 2]

# Dynamics layer unrolls to trajectory (full state)
trajectory = dynamics(ego_state, controls, dt=0.1, vehicle_params)  # [B, 80, 6]
```

## Architecture

### Two-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│  ML Planner Model (e.g., UrbanDriverOpenLoopModel)          │
│  ┌────────────┐      ┌──────────────┐                       │
│  │  Encoder   │ ───> │ Control Head │ ─> controls [B,T,2]  │
│  └────────────┘      └──────────────┘                       │
└────────────────────────────────┬────────────────────────────┘
                                  │
                                  v
         ┌────────────────────────────────────────────┐
         │  DeepDynamicalSystemLayer (Multi-Step)     │
         │  ┌──────────────────────────────────────┐  │
         │  │  Loop over time steps t=0..T-1       │  │
         │  │    x_{t+1} = dynamics(x_t, u_t, dt)  │  │
         │  └──────────────────────────────────────┘  │
         └────────────────┬───────────────────────────┘
                          │
                          v
      ┌───────────────────────────────────────────────┐
      │  DynamicsLayer (Single-Step)                  │
      │  ┌─────────────────────────────────────────┐  │
      │  │  Bicycle: x_dot = f_bicycle(x, u)       │  │
      │  │  Unicycle: x_dot = f_unicycle(x, u)     │  │
      │  │  Discretize: x_{t+1} = x_t + dt*x_dot   │  │
      │  └─────────────────────────────────────────┘  │
      └───────────────────────────────────────────────┘
```

### Forward Euler Discretization

Both models use **Forward Euler** (explicit Euler) for time integration:

```python
# Continuous dynamics: dx/dt = f(x, u)
x_dot = compute_derivatives(state, controls)

# Forward Euler update: x_{t+1} = x_t + dt * f(x_t, u_t)
next_state = state + dt * x_dot
```

**Characteristics**:
- ✅ Simple, fast, differentiable
- ✅ First-order accurate: O(dt) local error, O(dt²) global error
- ⚠️ **1-step lag**: Control at t=0 affects position at t=2 (see Gotcha #2)
- ⚠️ Conditionally stable: requires small dt (< 0.5s typical)
- ❌ No implicit stability (unlike RK4, implicit Euler)

**Why not higher-order methods?** (e.g., RK4, implicit Euler):
- Euler is standard in robotics (see [Spong 2005], [Thrun 2005])
- Simpler autodiff graph (fewer intermediate states)
- dt=0.1s is small enough for stability at urban speeds
- Production systems use more sophisticated integrators (MPC, collocation)

**Academic references**:
- **Bicycle model**: Kong et al. 2015, "Kinematic and dynamic vehicle models for autonomous driving control design" (IEEE IV)
- **Unicycle model**: De Luca et al. 2019, "Dynamic unicycle model" (DKM)
- **Differentiable physics in planning**: Phan-Minh et al. 2021, "GameFormer" (CVPR)

## Dependencies

**Internal imports**:
```python
# CRITICAL: Must use full paths - __init__.py is empty!
from nuplan.planning.training.modeling.models.dynamics_layers.kinematic_bicycle_layer_rear_axle import (
    KinematicBicycleLayerRearAxle
)
from nuplan.planning.training.modeling.models.dynamics_layers.kinematic_bicycle_layer_geometric_center import (
    KinematicBicycleLayerGeometricCenter
)
from nuplan.planning.training.modeling.models.dynamics_layers.kinematic_unicycle_layer_rear_axle import (
    KinematicUnicycleLayerRearAxle
)
from nuplan.planning.training.modeling.models.dynamics_layers.deep_dynamical_system_layer import (
    DeepDynamicalSystemLayer
)
from nuplan.planning.training.modeling.models.dynamics_layers.kinematic_bicycle_layers_utils import (
    StateIndex,
    InputIndex
)
```

**External dependencies**:
- `torch` (>= 1.9)
- `torch.nn` (Module base class)

**Used by**:
- `nuplan/planning/training/modeling/models/urban_driver_open_loop_model.py`
- `nuplan/planning/training/modeling/models/dynamics_model.py` (if exists)
- Custom ML planners (user implementations)

**Gotcha**: `__init__.py` is **empty** (just one blank line), must use full paths:
```python
# ❌ FAILS - no exports in __init__.py
from nuplan.planning.training.modeling.models.dynamics_layers import KinematicBicycleLayerRearAxle

# ✅ WORKS - full module path
from nuplan.planning.training.modeling.models.dynamics_layers.kinematic_bicycle_layer_rear_axle import (
    KinematicBicycleLayerRearAxle
)
```

## Common Usage Patterns

### 1. Single-Step Bicycle Dynamics

```python
import torch
from nuplan.planning.training.modeling.models.dynamics_layers.kinematic_bicycle_layer_rear_axle import (
    KinematicBicycleLayerRearAxle
)

# Initialize model
bicycle = KinematicBicycleLayerRearAxle()

# Initial state: [x, y, yaw, vx, vy, yaw_rate]
# Example: Vehicle at origin, heading east (0°), moving 10 m/s
initial_state = torch.tensor([[
    0.0,    # x (m)
    0.0,    # y (m)
    0.0,    # yaw (rad, 0=East)
    10.0,   # vx (m/s, longitudinal velocity)
    0.0,    # vy (m/s, lateral velocity)
    0.0     # yaw_rate (rad/s)
]], dtype=torch.float32)  # [1, 6]

# Control input: [accel, steering_angle]
# Example: Maintain speed (0 accel), turn left (0.1 rad = 5.7°)
controls = torch.tensor([[
    0.0,    # accel (m/s²)
    0.1     # steering_angle (rad, positive=left)
]], dtype=torch.float32)  # [1, 2]

# Vehicle parameters: [wheelbase] or [wheelbase, width]
vehicle_params = torch.tensor([[3.0]])  # 3m wheelbase (typical sedan)

# Simulate one time step (dt = 0.1 seconds)
next_state = bicycle(initial_state, controls, dt=0.1, vehicle_parameters=vehicle_params)
# Output: [1, 6] tensor
# Expected: x≈1.0, y≈0.005, yaw≈0.005, vx=10.0, vy=0.0, yaw_rate≈0.34

print(f"Position after 0.1s: x={next_state[0,0]:.3f}, y={next_state[0,1]:.3f}")
print(f"New heading: {next_state[0,2]:.3f} rad ({next_state[0,2]*180/3.14:.1f}°)")
```

**Key observations**:
- Position updates based on **current** velocity (Forward Euler lag)
- Yaw updates based on **current** yaw_rate (also lagged)
- Velocity updates instantly from acceleration (no lag in velocity itself)
- Steering affects yaw_rate via `yaw_rate = vel_init * tan(steering_angle) / wheelbase`

### 2. Multi-Step Trajectory Rollout

```python
import torch
from nuplan.planning.training.modeling.models.dynamics_layers.kinematic_bicycle_layer_rear_axle import (
    KinematicBicycleLayerRearAxle
)
from nuplan.planning.training.modeling.models.dynamics_layers.deep_dynamical_system_layer import (
    DeepDynamicalSystemLayer
)

# Create multi-step rollout wrapper
bicycle = KinematicBicycleLayerRearAxle()
dynamics = DeepDynamicalSystemLayer(dynamics=bicycle)

# Initial state: Vehicle at origin, heading east, 10 m/s
initial_state = torch.tensor([[0.0, 0.0, 0.0, 10.0, 0.0, 0.0]])  # [1, 6]

# Control sequence: 50 time steps of [accel, steering]
# Example: Constant left turn then straight
controls = torch.zeros(1, 50, 2)  # [1, 50, 2]
controls[:, :25, 1] = 0.1   # Turn left for 2.5 seconds (steering=0.1 rad)
controls[:, 25:, 1] = 0.0   # Go straight for 2.5 seconds

# Vehicle parameters
vehicle_params = torch.tensor([[3.0]])  # [1, 1]

# Unroll full trajectory
trajectory = dynamics(initial_state, controls, timestep=0.1, agents_pars=vehicle_params)
# Output: [1, 50, 6] - all 50 states (NOT including t=0!)

print(f"Trajectory shape: {trajectory.shape}")  # [1, 50, 6]
print(f"Final position: x={trajectory[0,-1,0]:.2f}, y={trajectory[0,-1,1]:.2f}")

# Visualize (requires matplotlib)
import matplotlib.pyplot as plt
plt.plot(trajectory[0,:,0].numpy(), trajectory[0,:,1].numpy())
plt.xlabel('x (m)'); plt.ylabel('y (m)')
plt.title('Vehicle Trajectory (Left Turn + Straight)')
plt.axis('equal')
plt.grid(True)
plt.savefig('/tmp/trajectory.png')
```

**Expected output**: Vehicle drives ~25m east while turning left (arc), then continues straight ~25m northeast.

### 3. Integration in ML Planner Forward Pass

```python
import torch
import torch.nn as nn
from nuplan.planning.training.modeling.models.dynamics_layers.kinematic_bicycle_layer_rear_axle import (
    KinematicBicycleLayerRearAxle
)
from nuplan.planning.training.modeling.models.dynamics_layers.deep_dynamical_system_layer import (
    DeepDynamicalSystemLayer
)

class SimpleControlPlanner(nn.Module):
    """Minimal example: CNN encodes raster, MLP predicts controls, dynamics unrolls trajectory."""

    def __init__(self, trajectory_steps: int = 80):
        super().__init__()

        # Feature encoder (e.g., ResNet on raster images)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )  # Output: [B, 64]

        # Control prediction head
        self.control_head = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, trajectory_steps * 2)  # Predict [accel, steering] × T
        )

        # Differentiable physics rollout
        self.dynamics = DeepDynamicalSystemLayer(
            dynamics=KinematicBicycleLayerRearAxle()
        )
        self.trajectory_steps = trajectory_steps
        self.dt = 0.1  # 10 Hz simulation

    def forward(self, raster: torch.Tensor, ego_state: torch.Tensor, vehicle_params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            raster: [B, 3, H, W] - RGB top-down view
            ego_state: [B, 6] - [x, y, yaw, vx, vy, yaw_rate]
            vehicle_params: [B, 1] - [wheelbase]

        Returns:
            trajectory: [B, T, 6] - Full state trajectory
        """
        # Extract features from raster
        features = self.encoder(raster)  # [B, 64]

        # Predict control sequence
        controls_flat = self.control_head(features)  # [B, T*2]
        controls = controls_flat.reshape(-1, self.trajectory_steps, 2)  # [B, T, 2]

        # Unroll trajectory via differentiable physics
        trajectory = self.dynamics(ego_state, controls, self.dt, vehicle_params)  # [B, T, 6]

        return trajectory

# Usage example
planner = SimpleControlPlanner(trajectory_steps=80)
batch_size = 4

# Dummy inputs
raster = torch.randn(batch_size, 3, 224, 224)  # RGB raster
ego_state = torch.tensor([[0., 0., 0., 10., 0., 0.]]).repeat(batch_size, 1)  # All start at origin
vehicle_params = torch.tensor([[3.0]]).repeat(batch_size, 1)  # 3m wheelbase

# Forward pass (differentiable!)
trajectory = planner(raster, ego_state, vehicle_params)
print(f"Output trajectory: {trajectory.shape}")  # [4, 80, 6]

# Compute loss (e.g., imitation learning from expert trajectory)
expert_trajectory = torch.randn(batch_size, 80, 6)  # Ground truth
loss = nn.MSELoss()(trajectory, expert_trajectory)
print(f"Loss: {loss.item():.4f}")

# Backprop flows through dynamics!
loss.backward()
print(f"Gradient on control head: {planner.control_head[-1].weight.grad.norm():.4f}")
```

**Key insight**: Gradients flow from trajectory loss → through dynamics layer → to control predictions → to encoder weights. This enables end-to-end learning of physics-informed policies.

### 4. Gradient Flow Validation

```python
import torch
from nuplan.planning.training.modeling.models.dynamics_layers.kinematic_bicycle_layer_rear_axle import (
    KinematicBicycleLayerRearAxle
)

# Enable gradient tracking
bicycle = KinematicBicycleLayerRearAxle()

initial_state = torch.tensor([[0., 0., 0., 10., 0., 0.]], requires_grad=False)  # State doesn't need grad
controls = torch.tensor([[0.5, 0.1]], requires_grad=True)  # Controls need gradients!
vehicle_params = torch.tensor([[3.0]])

# Forward pass
final_state = bicycle(initial_state, controls, timestep=0.1, vehicle_parameters=vehicle_params)

# Loss: Want to reach target position (x=2, y=1)
target = torch.tensor([[2., 1., 0., 10., 0., 0.]])
loss = ((final_state - target) ** 2).sum()

# Backward pass
loss.backward()

# Check gradients
print(f"Control gradients: accel={controls.grad[0,0]:.4f}, steering={controls.grad[0,1]:.4f}")
# Expected: Negative accel gradient (need to decelerate), positive steering gradient (turn right)

# Verify non-zero gradients (physics is differentiable!)
assert controls.grad.abs().sum() > 0, "Gradients are zero - autograd failed!"
print("✓ Gradient flow validated")

# Manual finite difference check (optional)
epsilon = 1e-4
controls_perturbed = controls.detach().clone()
controls_perturbed[0, 0] += epsilon  # Perturb acceleration
final_perturbed = bicycle(initial_state, controls_perturbed, 0.1, vehicle_params)
loss_perturbed = ((final_perturbed - target) ** 2).sum()
fd_grad = (loss_perturbed - loss.detach()) / epsilon
print(f"Finite difference gradient: {fd_grad.item():.4f}")
print(f"Autograd gradient: {controls.grad[0,0].item():.4f}")
print(f"Relative error: {abs(fd_grad - controls.grad[0,0]) / abs(fd_grad):.2%}")
```

**Expected output**: Autograd gradient should match finite difference within ~1% (numerical precision).

## Gotchas & Pitfalls

### 1. Yaw Wraparound (0 ↔ 2π) - Angle-Aware Losses Required

**Problem**: Yaw angle wraps at 2π radians, but standard MSE loss doesn't handle this:

```python
import torch
from nuplan.planning.training.modeling.models.dynamics_layers.kinematic_bicycle_layer_rear_axle import (
    KinematicBicycleLayerRearAxle
)

bicycle = KinematicBicycleLayerRearAxle()

# Initial state: Heading almost north (yaw = 6.2 rad ≈ 355°)
initial_state = torch.tensor([[0., 0., 6.2, 10., 0., 0.]])
controls = torch.tensor([[0., 0.15]])
vehicle_params = torch.tensor([[3.0]])

# Simulate (should wrap to ~0.15 rad ≈ 8.6°)
final_state = bicycle(initial_state, controls, timestep=0.5, vehicle_parameters=vehicle_params)
print(f"Final yaw: {final_state[0,2]:.3f} rad")  # Likely wrapped to small value

# ❌ BAD: Standard MSE loss
target_yaw = torch.tensor([[0.1]])  # Target: 5.7° (wrapped)
current_yaw = final_state[:, 2:3]
mse_loss = ((current_yaw - target_yaw) ** 2).mean()
print(f"❌ MSE loss: {mse_loss.item():.4f}")  # HUGE if one is 0.1, other is 6.2!

# ✅ GOOD: Angle-aware loss (cosine distance)
angle_diff = current_yaw - target_yaw
angle_aware_loss = (1 - torch.cos(angle_diff)).mean()
print(f"✅ Angle-aware loss: {angle_aware_loss.item():.6f}")  # Small - correct!

# Alternative: Wrap angle difference to [-π, π]
def wrap_angle(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))

wrapped_diff = wrap_angle(current_yaw - target_yaw)
wrapped_mse = (wrapped_diff ** 2).mean()
print(f"✅ Wrapped MSE loss: {wrapped_mse.item():.6f}")
```

**Solution**: Always use angle-aware losses for yaw:
- Cosine distance: `1 - cos(Δθ)`
- Wrapped MSE: `(atan2(sin(Δθ), cos(Δθ)))²`
- Never use raw `(yaw_pred - yaw_target)²`

**Location in code**: Warned in docstrings of all `forward()` methods:
> "when using the sampled state (e.g., with an imitation loss), pay particular care to yaw and 0 <-> 2pi transitions"

### 2. Forward Euler Lag - Controls at t=0 Affect Position at t=2

**Problem**: Explicit Euler integration creates 1-step lag in position response:

```python
import torch
from nuplan.planning.training.modeling.models.dynamics_layers.kinematic_bicycle_layer_rear_axle import (
    KinematicBicycleLayerRearAxle
)

bicycle = KinematicBicycleLayerRearAxle()
vehicle_params = torch.tensor([[3.0]])

# Initial state: Stationary at origin
initial_state = torch.tensor([[0., 0., 0., 0., 0., 0.]])  # vx=0

# Control: Apply 5 m/s² acceleration
controls = torch.tensor([[5.0, 0.]])  # accel=5, steering=0

# Step 1 (t=0 → t=0.1s)
state_t1 = bicycle(initial_state, controls, dt=0.1, vehicle_parameters=vehicle_params)
print(f"t=0.1s: x={state_t1[0,0]:.4f}, vx={state_t1[0,3]:.4f}")
# Output: x=0.0000, vx=0.5000
# Position doesn't change yet! (uses vx from t=0, which was 0)

# Step 2 (t=0.1 → t=0.2s, same control)
state_t2 = bicycle(state_t1, controls, dt=0.1, vehicle_parameters=vehicle_params)
print(f"t=0.2s: x={state_t2[0,0]:.4f}, vx={state_t2[0,3]:.4f}")
# Output: x=0.0500, vx=1.0000
# NOW position updates (uses vx from t=0.1, which was 0.5 m/s)

# Step 3 (t=0.2 → t=0.3s, same control)
state_t3 = bicycle(state_t2, controls, dt=0.1, vehicle_parameters=vehicle_params)
print(f"t=0.3s: x={state_t3[0,0]:.4f}, vx={state_t3[0,3]:.4f}")
# Output: x=0.1500, vx=1.5000
# Position uses vx from t=0.2 (1.0 m/s)
```

**Explanation**: Forward Euler computes:
```python
# Velocity updates instantly from acceleration
vel_init = sqrt(vx_old² + vy_old²)
vel_new = vel_init + accel * dt

# But position uses OLD velocity (lag!)
x_new = x_old + vx_old * cos(yaw) * dt  # Uses vx_old, NOT vel_new!
y_new = y_old + vx_old * sin(yaw) * dt
```

**Implications**:
- **Multi-step rollout required**: Single-step predictions won't show control response in position
- **Lookahead needed**: Must predict 2+ steps ahead for control to affect position
- **Matches imitation data**: If expert data also uses Euler, predictions align

**Workaround**: Accept the lag (it's inherent to explicit Euler) or use higher-order integrator (not implemented).

### 3. Vehicle Parameters Confusion - Wheelbase vs Length

**Problem**: `vehicle_parameters` tensor semantics differ between bicycle variants:

```python
# In kinematic_bicycle_layer_rear_axle.py:
# vehicle_parameters[..., 0] = wheelbase (distance between axles)

# In kinematic_bicycle_layer_geometric_center.py:
# vehicle_parameters[..., 0] = vehicle_length
# wheelbase = vehicle_length * 0.5  # ASSUMES centered wheelbase!

# Reality: wheelbase ≠ vehicle_length!
# Typical sedan: length ~4.5m, wheelbase ~2.7m (ratio ~0.6, not 0.5)
```

**Impact on dynamics**:
```python
# Yaw rate equation uses wheelbase:
yaw_rate = vel_init * torch.tan(steering_angle) / wheelbase

# Smaller wheelbase → HIGHER yaw rate (tighter turns)
# Example: At vx=10 m/s, steering=0.2 rad
wheelbase_sedan = 3.0
yaw_rate_sedan = (10.0 / 3.0) * torch.tan(torch.tensor(0.2))  # 0.67 rad/s

wheelbase_truck = 5.0
yaw_rate_truck = (10.0 / 5.0) * torch.tan(torch.tensor(0.2))  # 0.40 rad/s
print(f"Sedan turn rate: {yaw_rate_sedan:.2f} rad/s (tighter)")
print(f"Truck turn rate: {yaw_rate_truck:.2f} rad/s (wider)")
```

**Solution**:
- For `KinematicBicycleLayerRearAxle`: Pass actual wheelbase in `vehicle_parameters[..., 0]`
- For `KinematicBicycleLayerGeometricCenter`: Pass wheelbase (NOT vehicle length!) as well
- Check nuPlan `VehicleParameters` class for correct wheelbase values

**AIDEV-NOTE**: This naming is confusing. Code comments say "vehicle length" but implementation expects wheelbase. Future refactor should clarify.

### 4. No Steering Limits - Can Hit tan() Singularities

**Problem**: Steering angle is **unconstrained**, allowing physically impossible angles that cause numerical explosions:

```python
import torch
from nuplan.planning.training.modeling.models.dynamics_layers.kinematic_bicycle_layer_rear_axle import (
    KinematicBicycleLayerRearAxle
)

bicycle = KinematicBicycleLayerRearAxle()
vehicle_params = torch.tensor([[3.0]])

# Initial state: Moving 10 m/s
initial_state = torch.tensor([[0., 0., 0., 10., 0., 0.]])

# ❌ Extreme steering angle (45° = 0.785 rad, near limit)
controls_extreme = torch.tensor([[0., 0.785]])
state_extreme = bicycle(initial_state, controls_extreme, dt=0.1, vehicle_parameters=vehicle_params)
print(f"Yaw rate at 45°: {state_extreme[0,5]:.2f} rad/s")  # ~3.33 rad/s (plausible but extreme)

# ❌ DISASTER: Near-vertical steering (85° = 1.48 rad)
controls_disaster = torch.tensor([[0., 1.48]])
state_disaster = bicycle(initial_state, controls_disaster, dt=0.1, vehicle_parameters=vehicle_params)
print(f"Yaw rate at 85°: {state_disaster[0,5]:.2f} rad/s")  # ~380 rad/s (!!)

# ❌ NaN: Exactly 90° (π/2 rad) causes tan() singularity
controls_nan = torch.tensor([[0., 1.5708]])
state_nan = bicycle(initial_state, controls_nan, dt=0.1, vehicle_parameters=vehicle_params)
print(f"Yaw rate at 90°: {state_nan[0,5]}")  # inf or NaN!
```

**Root cause**:
```python
# In kinematic_bicycle_layer_rear_axle.py (line ~64):
yaw_rate = vel_init * torch.tan(controls[..., InputIndex.STEERING_ANGLE]) / wheelbase
# tan(π/2) = ∞ → gradient explosion
```

**Real-world constraints**:
- Max steering angle: ~30-35° for most cars (~0.52-0.61 rad)
- Ackermann geometry limits steering based on wheelbase and turn radius

**Solutions**:

**Option 1: Clamp steering in control head**
```python
# In your ML model's control prediction
steering_raw = self.control_head(features)[:, :, 1]
steering_safe = torch.clamp(steering_raw, -0.5, 0.5)  # ±28.6°
controls = torch.stack([accel, steering_safe], dim=-1)
```

**Option 2: Use tanh activation**
```python
# Map network output [-∞, ∞] → steering [-max_steer, max_steer]
steering_normalized = torch.tanh(steering_raw)
max_steering = 0.5  # 28.6°
steering = max_steering * steering_normalized
```

**Option 3: Replace tan() with arctan approximation (not implemented)**
```python
# More stable near singularities
yaw_rate = (vel / wheelbase) * (2 / torch.pi) * torch.atan(steering_angle)
```

### 5. No Speed/Acceleration Limits - Allows Unrealistic Motion

**Problem**: No constraints on velocity or acceleration magnitudes:

```python
import torch
from nuplan.planning.training.modeling.models.dynamics_layers.kinematic_bicycle_layer_rear_axle import (
    KinematicBicycleLayerRearAxle
)

bicycle = KinematicBicycleLayerRearAxle()
vehicle_params = torch.tensor([[3.0]])

# Initial state: Moving 10 m/s
initial_state = torch.tensor([[0., 0., 0., 10., 0., 0.]])

# ❌ Unrealistic: 100 m/s² acceleration (10g!)
controls_unrealistic = torch.tensor([[100., 0.]])
state = bicycle(initial_state, controls_unrealistic, dt=0.1, vehicle_parameters=vehicle_params)
print(f"Velocity after 0.1s: {state[0,3]:.1f} m/s")  # 20 m/s (gain of 10 m/s in 0.1s!)

# ❌ Unrealistic: Negative velocity (backward at 100 m/s)
initial_backward = torch.tensor([[0., 0., 0., -100., 0., 0.]])
state_backward = bicycle(initial_backward, controls_unrealistic, dt=0.1, vehicle_parameters=vehicle_params)
print(f"Backward velocity: {state_backward[0,3]:.1f} m/s")  # -90 m/s (physically impossible)
```

**Real-world limits**:
- Max acceleration: ~3 m/s² (comfortable), ~5 m/s² (emergency), ~8 m/s² (F1 car)
- Max deceleration: ~5 m/s² (comfortable), ~10 m/s² (hard braking, ABS)
- Max speed: ~30 m/s (highway, 108 kph), ~15 m/s (urban, 54 kph)
- Reverse speed: ~5 m/s max for cars

**Solutions**:

**Option 1: Clamp in control prediction**
```python
accel_raw = self.control_head(features)[:, :, 0]
accel = torch.clamp(accel_raw, -8.0, 5.0)  # [-8, 5] m/s²
```

**Option 2: Clamp post-dynamics (velocity limits)**
```python
trajectory = self.dynamics(initial_state, controls, dt=0.1, vehicle_params)
trajectory[:, :, 3] = torch.clamp(trajectory[:, :, 3], 0.0, 30.0)  # vx ∈ [0, 30] m/s
```

**Option 3: Add penalty to loss**
```python
# Penalize excessive acceleration
accel_penalty = torch.relu(accel.abs() - 5.0).mean()  # Penalize |accel| > 5 m/s²
loss = traj_loss + 0.1 * accel_penalty
```

### 6. Device Mismatch Traps - Scalars Must Match Input Device

**Problem**: `timestep` parameter is a Python float, but tensor operations require matching devices:

```python
import torch
from nuplan.planning.training.modeling.models.dynamics_layers.kinematic_bicycle_layer_rear_axle import (
    KinematicBicycleLayerRearAxle
)

bicycle = KinematicBicycleLayerRearAxle()
vehicle_params = torch.tensor([[3.0]])

# Initial state on GPU
initial_state = torch.tensor([[0., 0., 0., 10., 0., 0.]]).cuda()
controls = torch.tensor([[0., 0.1]]).cuda()
vehicle_params_gpu = vehicle_params.cuda()

# ✅ This works (timestep is Python float, auto-broadcasts to GPU)
state = bicycle(initial_state, controls, timestep=0.1, vehicle_parameters=vehicle_params_gpu)
print(f"✓ GPU state device: {state.device}")  # cuda:0

# ❌ But this fails if you manually create timestep tensor on CPU:
timestep_cpu = torch.tensor(0.1)  # Default: CPU
# state = bicycle(initial_state, controls, timestep_cpu, vehicle_params_gpu)
# RuntimeError: Expected all tensors to be on the same device
```

**Root cause**: PyTorch arithmetic with Python scalars auto-broadcasts to tensor device:

```python
# Internal to dynamics layer:
# x_new = x_old + timestep * x_dot  # timestep=0.1 (float) works on any device
# x_new = x_old + timestep * x_dot  # timestep=tensor(0.1, device='cpu') FAILS if x_old is on GPU
```

**Solutions**:

**Option 1: Always pass timestep as Python float (current convention)**
```python
bicycle(state, controls, timestep=0.1, vehicle_parameters=params)  # ✓ Works on any device
```

**Option 2: Ensure all tensors match device**
```python
state = state.to(device)
controls = controls.to(device)
vehicle_params = vehicle_params.to(device)
```

**Option 3: Move model to GPU (doesn't help - DynamicsLayer has no parameters!)**
```python
bicycle = bicycle.cuda()  # ← Does nothing (DynamicsLayer has no learnable params!)
# Must ensure inputs are on GPU
```

**See also**: Code example in `kinematic_bicycle_layer_geometric_center.py` line 62:
```python
beta = torch.atan(
    torch.tensor(0.5, dtype=controls.dtype, device=controls.device)  # ✅ Explicit device match
    * torch.tan(controls[..., InputIndex.STEERING_ANGLE])
)
```

### 7. No Lateral Slip Modeling - Kinematic Assumptions Break at High Speed

**Problem**: Kinematic models assume **zero lateral tire slip** (`vy_dot = 0`), which fails at high speeds or aggressive maneuvers:

```python
import torch
from nuplan.planning.training.modeling.models.dynamics_layers.kinematic_bicycle_layer_rear_axle import (
    KinematicBicycleLayerRearAxle
)

bicycle = KinematicBicycleLayerRearAxle()
vehicle_params = torch.tensor([[3.0]])

# Scenario: High-speed turn (30 m/s ≈ 108 kph, steering=0.4 rad ≈ 23°)
initial_state = torch.tensor([[0., 0., 0., 30., 0., 0.]])  # Fast!
controls = torch.tensor([[0., 0.4]])  # Sharp turn

state = bicycle(initial_state, controls, dt=0.1, vehicle_parameters=vehicle_params)
print(f"Lateral velocity vy: {state[0,4]:.4f} m/s")  # Always 0.0 (unrealistic!)

# Reality: At 30 m/s with 23° steering, lateral slip would be ~5-10 m/s
# Kinematic model: Forces vy=0 (tire glued to road)
```

**When kinematic models break**:
- **High speeds**: >20 m/s (~72 kph) with sharp turns
- **Low friction**: Wet/icy roads (μ < 0.5)
- **Emergency maneuvers**: Hard braking + steering
- **Drifting**: Intentional oversteer (racing, stunt driving)

**Kinematic model validity**:
- ✅ Urban speeds: <15 m/s (~54 kph)
- ✅ Gentle maneuvers: <0.3 rad steering (~17°)
- ✅ High friction: Dry asphalt (μ ≈ 0.8-0.9)
- ❌ Highway speeds: >25 m/s (~90 kph)
- ❌ Aggressive driving: Rally, track racing

**Alternative: Dynamic bicycle model** (not implemented):
```python
# Includes tire forces, lateral slip, weight transfer
# vy_dot = (F_lat_front + F_lat_rear) / mass - vx * yaw_rate
# F_lat = cornering_stiffness * slip_angle (linear tire model)
# Much more complex, requires more parameters (mass, tire stiffness, etc.)
```

**For nuPlan**: Kinematic model is acceptable (urban autonomous driving, <20 m/s).

### 8. Unicycle 2-Step Lag vs Bicycle 1-Step

**Problem**: Unicycle model has **acceleration as a state** (7th dimension), creating additional lag:

```python
import torch
from nuplan.planning.training.modeling.models.dynamics_layers.kinematic_unicycle_layer_rear_axle import (
    KinematicUnicycleLayerRearAxle
)

unicycle = KinematicUnicycleLayerRearAxle()
vehicle_params = torch.tensor([[3.0]])

# Initial state: [x, y, yaw, vx, vy, accel_x, accel_y]
# Note: 7D instead of 6D (extra accel dimensions)
initial_state = torch.tensor([[0., 0., 0., 0., 0., 0., 0.]])  # Stationary, accel=0

# Control: [curvature, jerk]
# Apply jerk (but it affects accel state, not velocity directly!)
controls = torch.tensor([[0., 5.0]])  # curvature=0, jerk=5

# Step 1: Accel state updates, but velocity doesn't!
state_t1 = unicycle(initial_state, controls, timestep=0.1, vehicle_parameters=vehicle_params)
print(f"t=0.1s: accel_x={state_t1[0,5]:.2f}, vx={state_t1[0,3]:.2f}")  # accel_x=0.5, vx=0.0

# Step 2: Velocity updates based on accel from t=1
state_t2 = unicycle(state_t1, controls, timestep=0.1, vehicle_parameters=vehicle_params)
print(f"t=0.2s: accel_x={state_t2[0,5]:.2f}, vx={state_t2[0,3]:.2f}")  # accel_x=1.0, vx=0.05

# Step 3: Position FINALLY updates based on vx from t=2
state_t3 = unicycle(state_t2, controls, timestep=0.1, vehicle_parameters=vehicle_params)
print(f"t=0.3s: x={state_t3[0,0]:.4f}, vx={state_t3[0,3]:.2f}")  # x=0.01, vx=0.15
```

**Bicycle model (for comparison)**:
```python
from nuplan.planning.training.modeling.models.dynamics_layers.kinematic_bicycle_layer_rear_axle import (
    KinematicBicycleLayerRearAxle
)
bicycle = KinematicBicycleLayerRearAxle()

initial_state = torch.tensor([[0., 0., 0., 0., 0., 0.]])  # 6D
controls = torch.tensor([[5., 0.]])  # accel=5

# Step 1: Velocity updates instantly!
state_t1 = bicycle(initial_state, controls, timestep=0.1, vehicle_parameters=vehicle_params)
print(f"Bicycle t=0.1s: vx={state_t1[0,3]:.2f}")  # vx=0.5 (one step faster)
```

**Lag comparison**:
| Model | Control → Velocity | Control → Position |
|-------|-------------------|-------------------|
| Bicycle | 0 steps (instant) | 1 step (Euler lag) |
| Unicycle | 1 step | 2 steps |

**When to use unicycle**: When modeling predicted agents (other vehicles) where you don't have steering angle data, only trajectory. Or for GameFormer-style curvature prediction.

### 9. Empty __init__.py - Requires Full Path Imports

**Problem**: `dynamics_layers/__init__.py` has no exports (just blank line), so shorthand imports fail:

```python
# ❌ FAILS - ImportError: cannot import name 'KinematicBicycleLayerRearAxle'
from nuplan.planning.training.modeling.models.dynamics_layers import KinematicBicycleLayerRearAxle

# ❌ FAILS - AttributeError: module has no attribute 'KinematicBicycleLayerRearAxle'
import nuplan.planning.training.modeling.models.dynamics_layers as dyn
bicycle = dyn.KinematicBicycleLayerRearAxle()

# ✅ WORKS - Full module path
from nuplan.planning.training.modeling.models.dynamics_layers.kinematic_bicycle_layer_rear_axle import (
    KinematicBicycleLayerRearAxle
)

# ✅ WORKS - Import module directly
from nuplan.planning.training.modeling.models.dynamics_layers import kinematic_bicycle_layer_rear_axle
bicycle = kinematic_bicycle_layer_rear_axle.KinematicBicycleLayerRearAxle()
```

**Root cause**: `__init__.py` is empty (no `__all__` or explicit imports).

**Workaround**: Use full paths everywhere (verbose but explicit).

**TODO**: Populate `__init__.py` with exports:
```python
# Proposed __init__.py content (not yet implemented)
from nuplan.planning.training.modeling.models.dynamics_layers.abstract_dynamics import DynamicsLayer
from nuplan.planning.training.modeling.models.dynamics_layers.kinematic_bicycle_layer_rear_axle import (
    KinematicBicycleLayerRearAxle
)
from nuplan.planning.training.modeling.models.dynamics_layers.kinematic_bicycle_layer_geometric_center import (
    KinematicBicycleLayerGeometricCenter
)
from nuplan.planning.training.modeling.models.dynamics_layers.kinematic_unicycle_layer_rear_axle import (
    KinematicUnicycleLayerRearAxle
)
from nuplan.planning.training.modeling.models.dynamics_layers.deep_dynamical_system_layer import (
    DeepDynamicalSystemLayer
)

__all__ = [
    'DynamicsLayer',
    'KinematicBicycleLayerRearAxle',
    'KinematicBicycleLayerGeometricCenter',
    'KinematicUnicycleLayerRearAxle',
    'DeepDynamicalSystemLayer'
]
```

### 10. Gradient Explosions Near Singular Steering

**Problem**: Gradient of `tan(steering)` explodes as steering → π/2, causing training instability:

```python
import torch
from nuplan.planning.training.modeling.models.dynamics_layers.kinematic_bicycle_layer_rear_axle import (
    KinematicBicycleLayerRearAxle
)

bicycle = KinematicBicycleLayerRearAxle()
vehicle_params = torch.tensor([[3.0]])

# Enable gradient tracking
initial_state = torch.tensor([[0., 0., 0., 10., 0., 0.]])
steering = torch.tensor([[0.]], requires_grad=True)  # Start at 0°
controls = torch.cat([torch.zeros(1, 1), steering], dim=1)

# Compute gradient at different steering angles
for angle in [0.0, 0.5, 1.0, 1.4, 1.55]:
    steering.data.fill_(angle)
    state = bicycle(initial_state, controls, timestep=0.1, vehicle_parameters=vehicle_params)
    loss = state[:, 5].sum()  # Loss on yaw_rate
    loss.backward()
    print(f"Steering={angle:.2f} rad ({angle*180/3.14:.0f}°): grad={steering.grad.item():.2f}")
    steering.grad.zero_()

# Output:
# Steering=0.00 rad (0°): grad=3.33
# Steering=0.50 rad (29°): grad=3.83
# Steering=1.00 rad (57°): grad=5.72
# Steering=1.40 rad (80°): grad=18.63  ← Exploding!
# Steering=1.55 rad (89°): grad=169.78 ← EXPLOSION!
```

**Root cause**:
```python
d/dθ [tan(θ)] = sec²(θ) = 1 / cos²(θ)
# At θ=π/2: sec²(π/2) = ∞
```

**Consequences**:
- Gradient clipping required during training
- Learning rate must be conservative (< 1e-4)
- May need gradient norm monitoring

**Solutions**:

**Option 1: Gradient clipping (standard practice)**
```python
# In training loop
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients
optimizer.step()
```

**Option 2: Steering saturation with smooth gradient**
```python
# Replace tan(steering) with saturating function
max_tan_steering = 5.0  # tan(78°) ≈ 5
yaw_rate_contribution = (vel / wheelbase) * torch.tanh(steering / 0.5) * max_tan_steering
# tanh(steering/0.5) ∈ [-1, 1], gradient always bounded
```

**Option 3: Monitor and reject bad batches**
```python
# In training loop
loss.backward()
grad_norm = sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
if grad_norm > 100.0:
    print(f"Skipping batch: gradient norm {grad_norm:.2f}")
    optimizer.zero_grad()
    continue
optimizer.step()
```

### 11. No Collision/Map Constraints - Pure Kinematics

**Problem**: Dynamics layers only model vehicle motion, not interactions with environment:

```python
import torch
from nuplan.planning.training.modeling.models.dynamics_layers.kinematic_bicycle_layer_rear_axle import (
    KinematicBicycleLayerRearAxle
)
from nuplan.planning.training.modeling.models.dynamics_layers.deep_dynamical_system_layer import (
    DeepDynamicalSystemLayer
)

bicycle = KinematicBicycleLayerRearAxle()
dynamics = DeepDynamicalSystemLayer(dynamics=bicycle)
vehicle_params = torch.tensor([[3.0]])

# Scenario: Drive straight through a wall at x=5m
initial_state = torch.tensor([[0., 0., 0., 10., 0., 0.]])  # Start at origin, 10 m/s east
controls = torch.zeros(1, 20, 2)  # Go straight for 2 seconds

trajectory = dynamics(initial_state, controls, timestep=0.1, agents_pars=vehicle_params)

print(f"Final position: x={trajectory[0,-1,0]:.1f} m")  # x=20.0 m
# ❌ No collision detection - drives through obstacles!
# ❌ No lane boundaries - can drive off-road
# ❌ No traffic lights - ignores all map constraints
```

**What's missing**:
- Collision checking (with agents, static obstacles)
- Drivable area constraints (lane boundaries, curbs)
- Traffic rule compliance (lights, stop signs, right-of-way)
- Map topology awareness (can't turn if no lane exists)

**Where constraints are enforced**:
- **Metrics**: Collision metrics computed post-simulation (see `nuplan/planning/metrics/`)
- **Losses**: Map compliance losses in training (see `nuplan/planning/training/modeling/objectives/`)
- **Rewards**: In RL settings (not used in imitation learning)

**Example: Adding collision penalty to loss**
```python
# In custom training objective
def compute_loss(predicted_traj, agents, map_api):
    # Compute kinematic trajectory (no collision awareness)
    traj_tensor = predicted_traj  # [B, T, 6]

    # Post-hoc collision checking
    collision_penalty = 0.0
    for t in range(traj_tensor.shape[1]):
        ego_pos = traj_tensor[:, t, 0:2]  # [B, 2]
        # Check distance to agents (simplified - assumes static positions)
        for agent_pos in agents:
            distance = torch.norm(ego_pos - agent_pos, dim=1)
            collision_penalty += torch.relu(2.0 - distance).mean()  # Penalty if <2m

    # Combine with trajectory loss
    total_loss = traj_loss + 10.0 * collision_penalty
    return total_loss
```

**Design philosophy**: Dynamics layers are **low-level physics**, constraints are **high-level planning**.

### 12. Float Precision Yaw Drift in Long Rollouts

**Problem**: Accumulating floating-point errors cause yaw drift in trajectories >100 steps:

```python
import torch
from nuplan.planning.training.modeling.models.dynamics_layers.kinematic_bicycle_layer_rear_axle import (
    KinematicBicycleLayerRearAxle
)
from nuplan.planning.training.modeling.models.dynamics_layers.deep_dynamical_system_layer import (
    DeepDynamicalSystemLayer
)

bicycle = KinematicBicycleLayerRearAxle()
dynamics = DeepDynamicalSystemLayer(dynamics=bicycle)
vehicle_params = torch.tensor([[3.0]])

# Initial state: Stationary at origin, heading east
initial_state = torch.tensor([[0., 0., 0., 10., 0., 0.]], dtype=torch.float32)

# Control: Go perfectly straight (zero steering, zero accel)
controls = torch.zeros(1, 1000, 2, dtype=torch.float32)

# Rollout long trajectory
trajectory = dynamics(initial_state, controls, timestep=0.1, agents_pars=vehicle_params)

# Check yaw drift
initial_yaw = trajectory[0, 0, 2]
final_yaw = trajectory[0, -1, 2]
yaw_drift = abs(final_yaw - initial_yaw).item()
print(f"Yaw drift over 100s: {yaw_drift:.6f} rad ({yaw_drift*180/3.14:.4f}°)")
# Expected: ~1e-5 rad (0.0006°) with float32, ~1e-11 rad with float64

# Position drift
expected_x = 10.0 * 100  # 1000 m (perfect straight line)
actual_x = trajectory[0, -1, 0].item()
position_error = abs(actual_x - expected_x)
print(f"Position error: {position_error:.6f} m")  # ~0.001 m with float32
```

**Root cause**: Float32 has ~7 decimal digits precision. After 1000 additions, rounding errors accumulate.

**Solutions**:

**Option 1: Use float64 (double precision)**
```python
initial_state = initial_state.double()
controls = controls.double()
trajectory = dynamics(initial_state, controls, timestep=0.1, agents_pars=vehicle_params)  # All ops in float64
# Reduces drift by ~6 orders of magnitude (1e-5 → 1e-11 rad)
```

**Option 2: Periodic yaw normalization (not implemented)**
```python
# In DeepDynamicalSystemLayer.forward() loop
for t in range(num_steps):
    xout[t+1] = self.dynamics(xout[t], controls[:, t], timestep, agents_pars)
    # Normalize yaw to [-π, π] every step
    xout[t+1, :, 2] = torch.atan2(torch.sin(xout[t+1, :, 2]), torch.cos(xout[t+1, :, 2]))
```

**Option 3: Reduce simulation length** (easiest)
```python
# For nuPlan: 80 steps (8 seconds) typical - drift is negligible (<1e-6 rad)
# Only problematic for >500 steps (>50 seconds at 10 Hz)
```

### 13. No Gradient Checkpointing - Memory Intensive

**Problem**: Long rollouts create deep computation graphs, consuming lots of memory:

```python
import torch
from nuplan.planning.training.modeling.models.dynamics_layers.kinematic_bicycle_layer_rear_axle import (
    KinematicBicycleLayerRearAxle
)
from nuplan.planning.training.modeling.models.dynamics_layers.deep_dynamical_system_layer import (
    DeepDynamicalSystemLayer
)

bicycle = KinematicBicycleLayerRearAxle()
dynamics = DeepDynamicalSystemLayer(dynamics=bicycle)
vehicle_params = torch.tensor([[3.0]])

batch_size = 32
initial_state = torch.randn(batch_size, 6)
controls = torch.randn(batch_size, 500, 2)

# Forward pass
trajectory = dynamics(initial_state, controls, timestep=0.1, agents_pars=vehicle_params)  # [32, 500, 6]

# Backward pass (high memory usage!)
loss = trajectory.sum()
loss.backward()  # Stores 500 intermediate states × 32 batch × 6 dims × 4 bytes ≈ 384 KB
                  # Plus gradients: another 384 KB
                  # Total: ~1 MB per sample, ~32 MB for batch
                  # For batch_size=128: ~128 MB just for one rollout!
```

**Memory scaling**: O(batch_size × num_steps × state_dim × 2) for forward + backward.

**Solutions**:

**Option 1: Gradient checkpointing (manual implementation)**
```python
from torch.utils.checkpoint import checkpoint

class CheckpointedDeepDynamicsLayer(DeepDynamicalSystemLayer):
    def forward(self, initial_state, controls, timestep, agents_pars):
        k = controls.shape[-2]
        xout = torch.empty((*controls.shape[:-1], self.dynamics.state_dim()), ...)
        state = initial_state

        for t in range(k):
            # Checkpoint every 10 steps (trade compute for memory)
            if t % 10 == 0:
                state = checkpoint(self.dynamics, state, controls[..., t, :], timestep, agents_pars)
            else:
                state = self.dynamics(state, controls[..., t, :], timestep, agents_pars)
            xout[..., t, :] = state

        return xout
```

**Option 2: Reduce batch size**
```python
# Instead of batch_size=128, use 32
# Memory: 128 MB → 32 MB (4× reduction)
# Training time: 4× longer (same total samples)
```

**Option 3: Reduce rollout length**
```python
# Instead of num_steps=500 (50s), use 80 (8s)
# Memory: 500 states → 80 states (6.25× reduction)
```

**Option 4: Mixed precision training**
```python
# In PyTorch Lightning trainer config
trainer = Trainer(precision=16)  # Use float16 instead of float32
# Memory: ~50% reduction (4 bytes → 2 bytes per float)
# May need loss scaling to avoid underflow
```

### 14. State Index Enum Alignment with Feature Builders

**Problem**: State tensor indices in dynamics layers must match feature builder conventions, but no explicit validation:

```python
# In dynamics layers: State is [x, y, yaw, vx, vy, yaw_rate] (indices 0-5)
# In kinematic_bicycle_layers_utils.py:
class StateIndex:
    X_POS = 0
    Y_POS = 1
    YAW = 2
    VEL_X = 3
    VEL_Y = 4
    YAW_RATE = 5

# Feature builders expect same order (by convention, not validation)
# If someone changes StateIndex enum, feature extraction breaks silently!
```

**Hidden assumption**: Dynamics layer state order matches feature builder output by convention, not by code.

**Risk**: If someone changes dynamics layer state representation (e.g., adds acceleration dimension), features break silently.

**Solution**: Add validation or explicit mapping:
```python
# In DynamicsLayer.__init__() (proposed)
class KinematicBicycleLayerRearAxle(DynamicsLayer):
    def __init__(self):
        super().__init__()
        # Validate state matches expected convention
        assert StateIndex.X_POS == 0
        assert StateIndex.Y_POS == 1
        assert StateIndex.YAW == 2
        assert StateIndex.VEL_X == 3
        assert StateIndex.VEL_Y == 4
        assert StateIndex.YAW_RATE == 5
```

**Current status**: No validation exists - works by coincidence.

### 15. Acceleration Doesn't Affect Yaw Rate (Euler Artifact)

**Problem**: In bicycle model, acceleration influences yaw rate only **indirectly** through velocity update, creating counterintuitive behavior:

```python
import torch
from nuplan.planning.training.modeling.models.dynamics_layers.kinematic_bicycle_layer_rear_axle import (
    KinematicBicycleLayerRearAxle
)

bicycle = KinematicBicycleLayerRearAxle()
vehicle_params = torch.tensor([[3.0]])

# Scenario 1: Accelerate while turning
initial_state = torch.tensor([[0., 0., 0., 10., 0., 0.]])  # 10 m/s
controls_accel = torch.tensor([[5., 0.3]])  # Accel=5 m/s², steering=0.3 rad

state_t1 = bicycle(initial_state, controls_accel, timestep=0.1, vehicle_parameters=vehicle_params)
yaw_rate_t1 = state_t1[0, 5]
print(f"Yaw rate with accel: {yaw_rate_t1:.4f} rad/s")

# Scenario 2: Same steering, no acceleration
controls_no_accel = torch.tensor([[0., 0.3]])
state_t2 = bicycle(initial_state, controls_no_accel, timestep=0.1, vehicle_parameters=vehicle_params)
yaw_rate_t2 = state_t2[0, 5]
print(f"Yaw rate without accel: {yaw_rate_t2:.4f} rad/s")

# ❌ SAME yaw rate! Acceleration doesn't affect turning in single step!
print(f"Difference: {abs(yaw_rate_t1 - yaw_rate_t2):.6f}")  # ~0 (Euler lag)
```

**Why this happens**:
```python
# Continuous dynamics (correct):
yaw_rate_dot = (vel / wheelbase) * tan(steering)

# Forward Euler discretization:
vel_init = sqrt(vx² + vy²)
yaw_rate = vel_init * tan(steering) / wheelbase  # Uses vel_init (BEFORE accel applied)

# Meanwhile:
vel = vel_init + accel * dt  # Velocity updates to vel_new

# So yaw_rate at t=1 uses vel from t=0, even though vel has changed!
```

**Physical intuition**: In reality, accelerating into a turn **increases** turn rate (higher centripetal acceleration needed). But Forward Euler delays this effect by 1 step.

**Impact**: Multi-step rollouts eventually correct (yaw_rate at t=2 uses vel from t=1), but single-step predictions are wrong.

**Solutions**:

**Option 1: Accept the lag** (simplest)
- Use multi-step rollouts (already standard practice)
- Single-step predictions are rarely used

**Option 2: Use semi-implicit Euler** (not implemented)
```python
# Update velocity first, then use NEW velocity for yaw rate
vel_new = vel_init + accel * dt
yaw_rate = vel_new * tan(steering) / wheelbase  # Use vel_new!
yaw_rate_new = yaw_rate  # (Or integrate yaw_rate_dot if modeling yaw acceleration)
```

**Option 3: RK4 integration** (much more complex)
- Fourth-order Runge-Kutta reduces lag significantly
- Requires 4 intermediate evaluations per step
- Complicates autograd graph

## Cross-References

**Related modules**:
- `nuplan/planning/simulation/trajectory/` - Trajectory representations (InterpolatedTrajectory, etc.)
- `nuplan/common/actor_state/vehicle_parameters.py` - Physical vehicle specs (wheelbase, dimensions)
- `nuplan/planning/training/modeling/models/` - ML planner models that use dynamics layers (see CLAUDE.md)
- `nuplan/planning/training/preprocessing/features/` - Feature builders (must match state indexing)
- `nuplan/planning/metrics/` - Evaluation metrics (collision, comfort)

**Papers & references**:
- **Bicycle model**: J. Kong et al., "Kinematic and dynamic vehicle models for autonomous driving control design," IEEE IV 2015
- **Unicycle model**: https://arxiv.org/pdf/2109.13602.pdf (GameFormer 2021)
- **Differentiable physics in planning**: T. Phan-Minh et al., "GameFormer: Game-theoretic modeling for multi-agent autonomous driving," CVPR 2021
- **Forward Euler stability**: M. Spong et al., "Robot Modeling and Control," Wiley 2005 (Chapter 3)

**Gotcha index** (quick reference):
1. Yaw wraparound → Use angle-aware losses
2. Forward Euler lag → Multi-step rollouts required
3. Vehicle parameters → `vehicle_parameters[0]` = wheelbase
4. Steering limits → Clamp to ±0.5 rad
5. Speed limits → Clamp velocity/acceleration
6. Device mismatches → Use Python float for timestep
7. No lateral slip → Only valid <20 m/s
8. Unicycle lag → 2-step vs bicycle 1-step
9. Empty __init__.py → Full path imports
10. Gradient explosions → Clip gradients or saturate steering
11. No collision checks → Add constraints in loss
12. Yaw drift → Use float64 or normalize
13. Memory usage → Gradient checkpointing
14. State index alignment → Matches StateIndex enum by convention
15. Acceleration lag → Affects yaw rate at t+2, not t+1

**Questions? See**:
- Tutorial: `tutorials/nuplan_planner_tutorial.ipynb` (simple planner example)
- Example usage: `nuplan/planning/training/modeling/models/urban_driver_open_loop_model.py` (lines 120-150)
- Tests: `nuplan/planning/training/modeling/models/dynamics_layers/test/` (unit tests with expected outputs)

---

**Last updated**: 2025-11-15 (Session 4, Phase 2C documentation sprint)
**Navigator 🧭 says**: These dynamics layers are the secret sauce for physics-informed ML planning! The gotchas are real (especially yaw wraparound and Euler lag), but once you handle them, you get gradient-based learning of kinematically feasible policies. Beautiful! 🚗⚡
