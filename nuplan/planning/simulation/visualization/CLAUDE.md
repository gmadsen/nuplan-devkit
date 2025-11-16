# nuplan/planning/simulation/visualization/

ABOUTME: Abstract interface for rendering simulation scenarios, ego states, and trajectories.
ABOUTME: Thin abstraction layer used by VisualizationCallback for bird's-eye view rendering.

## Purpose & Scope

**What is this module?**

The visualization module provides an abstract interface (`AbstractVisualization`) for rendering simulation state during closed-loop execution. It defines the contract for displaying:
- Map geometry and road network
- Ego vehicle state and trajectory
- Agent observations (tracked objects, IDM agents, etc.)
- Goal locations and route information

**Why an abstract interface?**

Different visualization backends (matplotlib, OpenGL, web-based dashboards like nuBoard) have different rendering requirements. This interface abstracts the rendering mechanism, allowing VisualizationCallback to invoke rendering without knowing the concrete implementation.

**Who implements this?**

Currently, nuBoard (Phase 4B) provides the primary implementation. This is a **lightweight, execution-time visualization** (real-time frame rendering), separate from nuBoard's post-hoc replay functionality.

**When is it used?**

- **During simulation**: VisualizationCallback invokes methods at each timestep for real-time rendering
- **Interactive debugging**: Quick visual feedback while testing planners
- **Benchmark verification**: Ensure trajectory looks reasonable before computing metrics

**Critical constraint**: Rendering is **extremely slow** (~0.5-2s per frame). Never use in production evaluations - use for debugging only!

## Key Abstractions

### AbstractVisualization Interface

The interface defines 5 rendering methods:

```python
from abc import ABCMeta, abstractmethod
from nuplan.planning.simulation.visualization.abstract_visualization import AbstractVisualization

class AbstractVisualization(metaclass=ABCMeta):
    """Generic visualization interface for rendering simulation state."""

    @abstractmethod
    def render_scenario(self, scenario: AbstractScenario, render_goal: bool) -> None:
        """
        Render map, road network, and lane markings.

        :param scenario: Current scenario with map_api for spatial queries
        :param render_goal: If True, also render target goal location

        Purpose: Initialize frame with map geometry (static once per scenario)
        Typical calls: 1x per scenario (in on_initialization_start)
        """

    @abstractmethod
    def render_ego_state(self, state_center: EgoState) -> None:
        """
        Render ego vehicle as rectangle/circle at current position.

        :param state_center: Current ego state (position, heading, velocity, etc.)

        Purpose: Draw ego vehicle on map
        Typical calls: ~200x per scenario (every timestep)
        Performance: Critical - must be < 0.1s for real-time
        """

    @abstractmethod
    def render_polygon_trajectory(self, trajectory: List[StateSE2]) -> None:
        """
        Render trajectory as filled polygon (area, not path).

        :param trajectory: List of SE2 states (x, y, heading) along path

        Purpose: Show ego "footprint" over time (occupancy envelope)
        Typical calls: ~0-10x per scenario (optional visualization)
        Use case: Collision detection, drivable area violations
        """

    @abstractmethod
    def render_trajectory(self, trajectory: List[StateSE2]) -> None:
        """
        Render trajectory as path (connected line, not filled area).

        :param trajectory: List of SE2 states along planned path

        Purpose: Display ego planned trajectory vs actual trajectory
        Typical calls: ~200x per scenario (every timestep)
        Difference from polygon: Shows path outline only (faster)
        """

    @abstractmethod
    def render_observations(self, observations: Any) -> None:
        """
        Render tracked agents (vehicles, pedestrians, cyclists).

        :param observations: Agent data (TracksObservation, IDMAgents, Sensors, etc.)

        Purpose: Draw other vehicles, prediction trajectories
        Typical calls: ~200x per scenario (every timestep)
        Performance: Variable (depends on # agents, ~5-20 agents typical)
        """

    @abstractmethod
    def render(self, iteration: SimulationIteration) -> None:
        """
        Trigger actual rendering to display (save frame, send to display, etc.).

        :param iteration: Current simulation iteration (time, step number)

        Purpose: Finalize frame and output (save PNG, write video frame, etc.)
        Typical calls: ~200x per scenario
        Note: Called AFTER all other render_* calls for a timestep
        """
```

## Architecture

### Render Call Sequence (Per Timestep)

The rendering pipeline follows a strict order:

```
Timestep T:
│
├─► render_scenario(scenario, render_goal=True)  [1x per scenario]
│   └─ Initialize map, lanes, goals
│
└─► FOR EACH STEP T in simulation:
    │
    ├─► render_ego_state(ego_state)       [Draw ego vehicle]
    │   └─ Position: ego_state.center (x, y)
    │   └─ Heading: ego_state.heading (radians)
    │   └─ Size: ~4.5m x 2m (typical car)
    │
    ├─► render_trajectory(planned_trajectory)  [Draw plan]
    │   └─ List of SE2 states (x, y, heading)
    │   └─ Color: Usually green (plan) vs blue (actual)
    │
    ├─► render_polygon_trajectory(occupancy)   [Optional]
    │   └─ Footprint envelope of trajectory
    │   └─ Shows collision risk areas
    │
    ├─► render_observations(agents)            [Draw other vehicles]
    │   └─ Agent positions, headings, velocities
    │   └─ Prediction trajectories (optional)
    │
    └─► render(iteration)                      [Finalize frame]
        └─ Save frame, update display, encode video
```

**Enforced from VisualizationCallback** (`callback/visualization_callback.py`):

```python
def on_step_end(self, setup, planner, sample):
    # Order is CRITICAL - implementation may depend on it
    self._visualization.render_ego_state(sample.ego_state)
    self._visualization.render_observations(sample.observation)
    self._visualization.render_trajectory(sample.trajectory.get_sampled_trajectory())
    self._visualization.render(sample.iteration)
```

**AIDEV-NOTE**: Some implementations (OpenGL) batch renders and only flush on `render()` call - respect order!

### Parameter Types & Imports

**From callback arguments**:
- `AbstractScenario` - Map API, scenario name, mission goal (location → [Lanes])
- `EgoState` - Position (x,y), heading (radians), velocity, acceleration
- `SimulationIteration` - Timestep, time (seconds since t=0), step number
- `Any` (observations) - Usually `TracksObservation` (list of Agent with position, heading, velocity) or `Sensors` (raw sensor data)

**Geometric types**:
- `StateSE2` - (x, y, heading) tuples, immutable, hashable

**AIDEV-NOTE**: Observations type varies by planner - some expect TracksObservation, others Sensors!

## Dependencies & Implementation Locations

### What It Imports

**Core simulation types** (thin imports):
```python
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
```

**AIDEV-NOTE**: Minimal imports by design - interface is decoupled from planner/controller layers!

### Who Uses It

**Direct consumer**:
- `callback/visualization_callback.py` - Invokes all 5 methods per timestep during simulation

**Indirect consumer**:
- `runner/` - Executes VisualizationCallback in main simulation loop
- `nuBoard` (Phase 4B) - Provides implementation for real-time and post-hoc visualization

### No Submodules

This is a leaf module - **only 3 files**:
- `abstract_visualization.py` (64 lines) - Interface definition
- `__init__.py` (empty)
- `BUILD` (Bazel config)

No test directory, no implementations within simulation package.

## Usage Patterns

### Pattern 1: Implementing a Concrete Renderer (matplotlib)

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from nuplan.planning.simulation.visualization.abstract_visualization import AbstractVisualization

class MatplotlibRenderer(AbstractVisualization):
    """Simple matplotlib-based renderer for debugging."""

    def __init__(self, output_dir: str = '/tmp/frames'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.fig, self.ax = None, None
        self._frame_count = 0

    def render_scenario(self, scenario, render_goal: bool) -> None:
        """Initialize figure with map."""
        self.fig, self.ax = plt.subplots(figsize=(12, 12))

        # Draw lane centerlines
        map_api = scenario.map_api
        for lane in map_api.get_all_lanes():
            coords = lane.baseline_path.discrete_path
            xs, ys = zip(*coords)
            self.ax.plot(xs, ys, 'k-', linewidth=1, alpha=0.5)

        # Draw goal if requested
        if render_goal and scenario.goal_location:
            goal = scenario.goal_location
            self.ax.plot(goal.x, goal.y, 'g*', markersize=15, label='Goal')

        self.ax.set_aspect('equal')
        self.ax.legend()

    def render_ego_state(self, state_center: EgoState) -> None:
        """Draw ego as rectangle."""
        # Ego dimensions (typical nuPlan vehicle)
        width, length = 2.0, 4.5

        # Create rotated rectangle
        from matplotlib.transforms import Affine2D
        x, y = state_center.center.x, state_center.center.y
        heading = state_center.heading  # radians

        # Rectangle centered at (x, y) with given heading
        rect = patches.Rectangle(
            (-length/2, -width/2), length, width,
            linewidth=2, edgecolor='red', facecolor='red', alpha=0.7
        )

        # Apply rotation and translation
        t = Affine2D().translate(x, y).rotate(heading) + self.ax.transData
        rect.set_transform(t)
        self.ax.add_patch(rect)

    def render_polygon_trajectory(self, trajectory: List[StateSE2]) -> None:
        """Draw trajectory envelope as polygon."""
        if not trajectory:
            return

        # Simple: Draw bounding box around trajectory
        xs, ys = [s.x for s in trajectory], [s.y for s in trajectory]
        polygon = patches.Polygon(
            list(zip(xs, ys)),
            closed=True, facecolor='yellow', alpha=0.2, edgecolor='yellow'
        )
        self.ax.add_patch(polygon)

    def render_trajectory(self, trajectory: List[StateSE2]) -> None:
        """Draw trajectory as line."""
        if not trajectory:
            return

        xs, ys = [s.x for s in trajectory], [s.y for s in trajectory]
        self.ax.plot(xs, ys, 'g--', linewidth=2, label='Planned trajectory')

    def render_observations(self, observations) -> None:
        """Draw observed agents."""
        if not observations:
            return

        # Assuming TracksObservation with tracked_objects
        for agent in observations.tracked_objects:
            # Draw agent as circle
            circle = patches.Circle(
                (agent.center.x, agent.center.y), radius=1.0,
                edgecolor='blue', facecolor='blue', alpha=0.5
            )
            self.ax.add_patch(circle)

            # Draw velocity vector
            vx = agent.velocity.x
            vy = agent.velocity.y
            self.ax.arrow(
                agent.center.x, agent.center.y, vx*0.5, vy*0.5,
                head_width=0.3, head_length=0.2, fc='blue', ec='blue'
            )

    def render(self, iteration: SimulationIteration) -> None:
        """Save frame to disk."""
        self.ax.set_title(f'Time: {iteration.time:.2f}s, Step: {iteration.index}')

        # Save frame
        output_path = self.output_dir / f'frame_{self._frame_count:05d}.png'
        self.fig.savefig(output_path, dpi=50)
        self._frame_count += 1

        # Clear for next frame
        self.ax.clear()
```

**Usage in simulation**:
```python
from nuplan.planning.simulation.callback.visualization_callback import VisualizationCallback

renderer = MatplotlibRenderer(output_dir='/tmp/simulation_frames')
callback = VisualizationCallback(renderer=renderer)

# Add to simulation
sim = Simulation(setup, callback=callback)
```

### Pattern 2: Registering Renderer via Hydra Config

**Create renderer config**: `config/visualization/matplotlib_renderer.yaml`

```yaml
_target_: my_project.visualization.MatplotlibRenderer

output_dir: ${oc.env:NUPLAN_EXP_ROOT}/frames
```

**Create callback config**: `config/callback/viz_callback.yaml`

```yaml
_target_: nuplan.planning.simulation.callback.visualization_callback.VisualizationCallback

renderer: ???  # Must override

# Usage: callback=viz_callback hydra.defaults+=[visualization=matplotlib_renderer]
```

**Run simulation with visualization**:
```bash
uv run python nuplan/planning/script/run_simulation.py \
    planner=simple_planner \
    callback=viz_callback \
    callback.renderer@_global_=visualization/matplotlib_renderer
```

### Pattern 3: Conditional Rendering Based on Scenario Type

```python
from nuplan.planning.simulation.visualization.abstract_visualization import AbstractVisualization

class ConditionalRenderer(AbstractVisualization):
    """Only render "interesting" scenarios."""

    def __init__(self, base_renderer: AbstractVisualization,
                 interesting_types: List[str] = None):
        self.base_renderer = base_renderer
        self.interesting_types = interesting_types or [
            'starting_left_turn',
            'starting_right_turn',
            'near_multiple_vehicles'
        ]
        self._should_render = False

    def render_scenario(self, scenario, render_goal: bool) -> None:
        # Check if scenario is "interesting"
        self._should_render = any(
            t in scenario.scenario_type for t in self.interesting_types
        )

        if self._should_render:
            self.base_renderer.render_scenario(scenario, render_goal)

    def render_ego_state(self, state_center) -> None:
        if self._should_render:
            self.base_renderer.render_ego_state(state_center)

    # ... other methods ...
```

## Gotchas & Pitfalls

### Gotcha 1: Rendering Blocks Simulation Loop

**Problem**: matplotlib.savefig() takes 0.5-2s per frame. Simulation loop stalls.

```python
# With 200 timesteps per scenario:
# 200 * 0.5s = 100s per scenario (50x slower than real-time!)

def render(self, iteration):
    plt.savefig(...)  # BLOCKS for 0.5-2 seconds
```

**Solution**: Only render if debugging. Use offline rendering for large batches.

```bash
# ❌ SLOW: Real-time rendering blocks simulation
just simulate callback=viz_callback  # Takes 100s per scenario!

# ✅ FAST: Disable visualization, run 10s per scenario
just simulate callback=no_viz  # Takes 10s per scenario

# ✅ FAST: Render offline from saved history (parallel)
uv run render_frames.py history_dir=/tmp/results output_dir=/tmp/frames num_workers=8
```

**AIDEV-NOTE**: VisualizationCallback overhead: 80-90% of simulation runtime!

### Gotcha 2: Observations Type Varies by Planner

**Problem**: `render_observations(observations: Any)` can receive different types.

```python
def render_observations(self, observations: Any) -> None:
    # FAILS if observations is Sensors instead of TracksObservation
    for agent in observations.tracked_objects:  # AttributeError!
        self.ax.plot(agent.center.x, agent.center.y, 'b.')
```

**Solution**: Type-check observations and handle gracefully.

```python
def render_observations(self, observations: Any) -> None:
    if observations is None:
        return

    # Handle TracksObservation
    if hasattr(observations, 'tracked_objects'):
        for agent in observations.tracked_objects:
            self.ax.plot(agent.center.x, agent.center.y, 'b.')

    # Handle Sensors (raw data, no agents)
    elif hasattr(observations, 'camera'):
        # Just skip - can't render raw camera data
        return
    else:
        logger.warning(f"Unknown observation type: {type(observations)}")
```

**AIDEV-NOTE**: Check planner's `observation_type()` before implementing render_observations()!

### Gotcha 3: StateSE2 Immutability Issues

**Problem**: StateSE2 objects are immutable. Mutations fail silently or throw.

```python
# ❌ WRONG: Can't modify trajectory
trajectory[0].x = 100.0  # TypeError or silently fails

# ✅ CORRECT: Create new StateSE2 if modification needed
modified = StateSE2(x=100.0, y=trajectory[0].y, heading=trajectory[0].heading)
```

**Common mistake**: Forgetting trajectory is a **list of immutable objects**.

### Gotcha 4: Coordinate System Mismatch

**Problem**: Different modules use different coordinate conventions.

```python
# nuPlan uses UTM (Universal Transverse Mercator)
# But local rendering often wants x=East, y=North (standard Cartesian)

# Ego heading: 0 = East, π/2 = North (standard)
# Some systems use: 0 = North, π/2 = East (compass bearing)

def render_ego_state(self, state):
    # WRONG: Assumes heading = angle from x-axis
    angle = state.heading  # May be compass bearing instead!

    # RIGHT: Always verify convention in nuPlan
    # nuPlan uses: 0 rad = East (positive x), π/2 rad = North (positive y)
    angle = state.heading  # Correct for UTM
```

**AIDEV-NOTE**: Always check `nuplan.common.geometry` for coordinate transformations!

### Gotcha 5: Goal Rendering Without Null Check

**Problem**: Goal location may be None or absent in some scenarios.

```python
# ❌ CRASH: scenario.goal_location could be None
goal = scenario.goal_location
self.ax.plot(goal.x, goal.y, 'g*')  # AttributeError if None!

# ✅ SAFE: Check before accessing
if scenario.goal_location:
    goal = scenario.goal_location
    self.ax.plot(goal.x, goal.y, 'g*')
```

**AIDEV-NOTE**: Some scenario types (lane following) have no explicit goal!

### Gotcha 6: Trajectory Empty List Errors

**Problem**: Planned trajectory can be empty at initialization.

```python
# ❌ CRASH: trajectory may be empty
def render_trajectory(self, trajectory):
    xs, ys = [s.x for s in trajectory], [s.y for s in trajectory]
    self.ax.plot(xs, ys, 'g-')  # matplotlib.ValueError if empty!

# ✅ SAFE: Guard against empty
def render_trajectory(self, trajectory):
    if not trajectory:
        return
    xs, ys = [s.x for s in trajectory], [s.y for s in trajectory]
    self.ax.plot(xs, ys, 'g-')
```

### Gotcha 7: Figure/Axis State Persistence

**Problem**: Matplotlib figures accumulate if not cleared properly.

```python
# ❌ Memory leak: 200 figures + axes in memory
def render(self, iteration):
    fig, ax = plt.subplots()  # Create new figure EVERY frame!
    # ... plot ...
    fig.savefig(...)  # Never cleared!

# ✅ CORRECT: Reuse or clear
def __init__(self):
    self.fig, self.ax = plt.subplots()

def render(self, iteration):
    # ... plot on self.ax ...
    self.fig.savefig(...)
    self.ax.clear()  # Clear for next frame
```

**Memory impact**: 200 frames * 10MB per figure = 2GB RAM leak per scenario!

### Gotcha 8: DPI Trade-offs in savefig()

**Problem**: High DPI produces huge files, low DPI looks pixelated.

```python
# ❌ TOO HIGH: 300 DPI → 50MB per frame, 10GB per scenario
fig.savefig('frame.png', dpi=300)

# ❌ TOO LOW: 50 DPI → Unreadable text, vehicle outlines blocky
fig.savefig('frame.png', dpi=50)

# ✅ BALANCED: 100 DPI → ~5-10MB per frame, reasonable quality
fig.savefig('frame.png', dpi=100)
```

**Recommendation**: Use DPI=100 for debugging, DPI=50 for videos.

### Gotcha 9: Forgetting to Call render() Finalizes Frame

**Problem**: Implementation may batch renders and only flush on `render()` call.

```python
# ❌ WRONG: OpenGL renderer batches, nothing drawn until render()
def on_step_end(self, setup, planner, sample):
    self._visualization.render_ego_state(sample.ego_state)
    self._visualization.render_trajectory(sample.trajectory.get_sampled_trajectory())
    # Frame not displayed yet! Missing render() call.

# ✅ CORRECT: Finalize with render()
def on_step_end(self, setup, planner, sample):
    self._visualization.render_ego_state(sample.ego_state)
    self._visualization.render_trajectory(sample.trajectory.get_sampled_trajectory())
    self._visualization.render(sample.iteration)  # Flushes batched commands
```

**AIDEV-NOTE**: VisualizationCallback enforces this order - respect it in implementations!

### Gotcha 10: Agent Velocity Vector Scaling

**Problem**: Velocity vectors may be too small/large to visualize without scaling.

```python
# ❌ TOO SMALL: Velocity in m/s, but scale too aggressive
for agent in observations.tracked_objects:
    vx, vy = agent.velocity.x, agent.velocity.y
    ax.arrow(agent.center.x, agent.center.y, vx, vy)  # Invisible!

# ❌ TOO LARGE: Scale too loose, arrows go off-screen
ax.arrow(agent.center.x, agent.center.y, vx*10, vy*10)  # Overshoots!

# ✅ REASONABLE: Scale velocity to visible length (~0.1-0.5s lookahead)
lookahead_time = 0.2  # 200ms
vx_scaled = agent.velocity.x * lookahead_time
vy_scaled = agent.velocity.y * lookahead_time
ax.arrow(agent.center.x, agent.center.y, vx_scaled, vy_scaled)
```

**AIDEV-NOTE**: Speed ranges 0-25 m/s in nuPlan - scale for 0.1-0.5s lookahead!

### Gotcha 11: Scenario Map API Queries Are Expensive

**Problem**: `scenario.map_api.get_all_lanes()` queries database on every render_scenario().

```python
# ❌ SLOW: Map query happens per scenario
def render_scenario(self, scenario, render_goal):
    for lane in scenario.map_api.get_all_lanes():  # DB query!
        # ... render lane ...
    # 200 lanes * 0.5s query = 100s per scenario!

# ✅ FASTER: Cache map or lazy-load
def __init__(self):
    self._map_cache = {}

def render_scenario(self, scenario, render_goal):
    if scenario.map_name not in self._map_cache:
        lanes = scenario.map_api.get_all_lanes()  # Query once
        self._map_cache[scenario.map_name] = lanes

    for lane in self._map_cache[scenario.map_name]:
        # ... render lane ...
```

**AIDEV-NOTE**: Map rendering dominates render_scenario() time (80-90% overhead)!

### Gotcha 12: MultiCallback Order Matters

**Problem**: Visualization before metrics = metrics missing from visualization.

```python
# ❌ WRONG: Visualization runs first, metrics not computed yet
callbacks = MultiCallback([
    VisualizationCallback(...),
    MetricCallback(...)
])

# ✅ CORRECT: Metrics before visualization (if annotating with metrics)
callbacks = MultiCallback([
    MetricCallback(...),
    VisualizationCallback(...)  # Can now annotate frames with metrics
])
```

**AIDEV-NOTE**: Callback order in MultiCallback is execution order!

## Cross-References

### Related Modules

**Direct consumer**:
- `callback/visualization_callback.py` - Wraps AbstractVisualization, invokes per-step
- `callback/test/test_visualization_callback.py` - Test patterns for mocking renderer

**Execution layer**:
- `runner/` - Executes simulation with VisualizationCallback
- `simulation.py` - Passes SimulationHistorySample to callbacks

**Related abstractions**:
- `history/` - SimulationHistory, SimulationHistorySample (state passed to renderer)
- `trajectory/` - AbstractTrajectory with `get_sampled_trajectory()` method
- `observation/` - Observation types (TracksObservation, Sensors, etc.)

**Scenario data**:
- `nuplan/planning/scenario_builder/abstract_scenario.py` - map_api, goal_location
- `nuplan/common/actor_state/` - EgoState, StateSE2 types

### Implementation References

**Primary implementation** (Phase 4B):
- nuBoard visualization dashboard - Full-featured rendering, interactive playback
- Web-based rendering backend for real-time and post-hoc visualization

**Test patterns**:
- `callback/test/test_visualization_callback.py` - Mock pattern for AbstractVisualization

### Related Documentation

- **Callback system**: `callback/CLAUDE.md` - Lifecycle hooks, execution context
- **Scenario builder**: `nuplan/planning/scenario_builder/CLAUDE.md` - Map API usage
- **Simulation module**: `CLAUDE.md` (root) - Integration with SimulationSetup

## Quick Reference

### Minimal Implementation Template

```python
from nuplan.planning.simulation.visualization.abstract_visualization import AbstractVisualization

class MinimalRenderer(AbstractVisualization):
    def render_scenario(self, scenario, render_goal: bool) -> None:
        """Initialize visualization."""
        pass

    def render_ego_state(self, state_center) -> None:
        """Draw ego vehicle."""
        pass

    def render_polygon_trajectory(self, trajectory) -> None:
        """Optional: Draw trajectory envelope."""
        pass

    def render_trajectory(self, trajectory) -> None:
        """Draw planned trajectory."""
        pass

    def render_observations(self, observations) -> None:
        """Draw observed agents."""
        pass

    def render(self, iteration) -> None:
        """Finalize and output frame."""
        pass
```

### Enable/Disable Visualization

```bash
# ❌ SLOW: Real-time rendering (100s per scenario)
just simulate callback=viz_callback

# ✅ FAST: No visualization (10s per scenario)
just simulate callback=no_viz

# ✅ ALTERNATIVE: Render offline from saved history
just simulate callback=no_viz  # First: Run without visualization
uv run render_from_history.py  # Second: Render from saved data
```

## Lessons Learned

- **Performance**: Visualization adds 80-90% overhead - debug only
- **Type variance**: Observations can be TracksObservation or Sensors - type-check
- **Immutability**: StateSE2 objects immutable, trajectory is list of them
- **Null safety**: Goal location may be None, empty trajectories possible
- **Caching**: Map queries expensive - cache by map_name
- **Batching**: OpenGL/GPU renderers batch until `render()` call - respect order

## Summary

The visualization module is a **thin, purpose-built interface** for real-time rendering during simulation. It abstracts rendering implementation details, allowing VisualizationCallback to invoke rendering without coupling to specific backends (matplotlib, OpenGL, web).

**Key design principles**:
- **Simple interface**: 5 methods covering 99% of rendering needs
- **Implementation-agnostic**: Works with matplotlib, OpenGL, web dashboards
- **Non-blocking contract**: Allows async/batched implementations
- **Observer pattern**: Cannot modify simulation, only observe state

**Remember**: Use for debugging only. Disable in production evals - visualization adds 50-100x overhead to simulation runtime!

