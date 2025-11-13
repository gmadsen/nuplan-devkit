# CLAUDE.md - nuPlan Development Assistant Guide

## Project Identity
- **Claude Project Name**: Navigator 🧭 (autonomous vehicle planning research assistant)
- **User Project Name**: G Money / GareBear
- **Project Theme**: Exploring autonomous vehicle planning with the nuPlan dataset and simulation devkit
- **Commit Signature**: All commits signed as "Navigator 🧭" with appropriate co-author attribution

## Project Overview

**nuPlan** is Motional's large-scale autonomous vehicle planning dataset and development kit. It contains ~1300 hours of driving data from real autonomous vehicles operating in 4 cities (Las Vegas, Singapore, Pittsburgh, Boston), along with a sophisticated simulation environment and planning benchmarks.

This project is G Money's experimental playground for:
- Learning autonomous vehicle planning concepts through tutorials
- Running experiments with planning algorithms
- Developing and testing custom planners
- Understanding the nuPlan dataset structure and metrics

### Repository Migration Status
This project has been **migrated from conda to uv** for faster, more reliable dependency management. The hybrid approach provides:
- **Native uv environment**: Primary development environment for fast iteration and IDE integration
- **Docker fallback**: Available when CUDA/driver conflicts arise with the Titan RTX GPU

## Quick Start for AI Assistants

### Essential Commands
```bash
# Setup environment (first time)
just setup                    # Full dev environment with CUDA
just setup-env               # Create .env file from template
source .env                  # Load dataset paths

# Daily workflow
just tutorial                # Launch Jupyter Lab for tutorials
just notebook <name>         # Open specific tutorial
just test                    # Run test suite
just lint                    # Check code quality
just format                  # Auto-format code

# CUDA verification
just check-cuda              # Verify GPU setup
just info                    # Show environment details

# CLI access
uv run nuplan_cli --help     # Main CLI interface
just cli <args>              # Shortcut for nuplan_cli
```

### Environment Variables (CRITICAL!)
Before running any nuPlan code, ensure these are set:
```bash
export NUPLAN_DATA_ROOT="/path/to/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/path/to/nuplan/maps"
export NUPLAN_EXP_ROOT="/path/to/experiments/output"
```
See `.env.example` for full details. **Always check these are set when debugging issues!**

## Project Structure

```
nuplan-devkit/
├── nuplan/                      # Main package
│   ├── cli/                     # CLI tools (nuplan_cli)
│   ├── common/                  # Shared utilities, maps, geometry
│   ├── database/                # SQLAlchemy ORM for dataset access
│   │   └── nuplan_db/           # Database schema and models
│   ├── planning/                # Core planning algorithms & simulation
│   │   ├── metrics/             # Evaluation metrics (comfort, progress, etc.)
│   │   ├── nuboard/             # Web dashboard for visualization
│   │   ├── scenario_builder/   # Scenario creation from logs
│   │   ├── script/              # Main entry points (training, simulation)
│   │   ├── simulation/          # Simulation loop, callbacks, planner API
│   │   │   ├── planner/         # Base planner interface
│   │   │   ├── observation/     # Sensor data processing
│   │   │   └── trajectory/      # Trajectory representations
│   │   ├── training/            # ML training infrastructure
│   │   │   ├── modeling/        # PyTorch Lightning models
│   │   │   └── preprocessing/   # Feature builders, caching
│   │   └── utils/               # Planning-specific utilities
│   └── submission/              # Competition submission code
│
├── tutorials/                   # Jupyter notebooks (START HERE!)
│   ├── nuplan_framework.ipynb   # Overview and architecture
│   ├── nuplan_planner_tutorial.ipynb  # Building custom planners
│   ├── nuplan_scenario_visualization.ipynb  # Exploring scenarios
│   ├── nuplan_simulation.ipynb  # Running simulations
│   └── nuplan_training.ipynb    # Training ML planners
│
├── docs/                        # Sphinx documentation
├── pyproject.toml              # Modern uv-based dependencies
├── uv.lock                     # Locked dependencies
├── Justfile                    # Common commands (run `just`)
└── .env.example                # Environment variable template
```

## Tutorial Workflow (Recommended for Learning)

The tutorials are designed to be completed in order:

1. **nuplan_framework.ipynb** - Understand the overall architecture
   - Dataset structure (logs, scenarios, maps)
   - Key abstractions (AbstractPlanner, PlannerInput, Trajectory)
   - Coordinate systems and map APIs

2. **nuplan_scenario_visualization.ipynb** - Explore the dataset
   - Load scenarios from the database
   - Visualize agent behaviors, map topology
   - Understand scenario types and filters

3. **nuplan_planner_tutorial.ipynb** - Build your first planner
   - Implement SimplePlanner interface
   - Process observations (ego state, agents, map)
   - Generate trajectories

4. **nuplan_simulation.ipynb** - Test planners in simulation
   - Set up simulation config (Hydra)
   - Run closed-loop simulation
   - Evaluate with metrics

5. **nuplan_training.ipynb** - Train ML-based planners
   - Feature caching pipeline
   - PyTorch Lightning training
   - Model evaluation

### Tutorial Tips for AI Assistants
- **Always run cells sequentially** - state depends on previous cells
- Check environment variables before running (NUPLAN_DATA_ROOT etc.)
- Dataset download required for most tutorials (mini dataset = ~50GB)
- Visualizations may require X11 forwarding or notebook inline mode
- Cache directory can grow large - monitor `$NUPLAN_EXP_ROOT/cache/`

## Key Concepts & Architecture

### 1. Hydra Configuration System
nuPlan uses **Hydra 1.1.0rc1** for configuration management. This is a pinned RC version!

**Important patterns:**
```python
# Main config composition
@hydra.main(config_path='config', config_name='default_simulation')
def main(cfg: DictConfig):
    # Access nested configs
    planner_cfg = cfg.planner
    scenario_cfg = cfg.scenario_builder
```

**Config overrides via CLI:**
```bash
uv run python nuplan/planning/script/run_simulation.py \
    planner=simple_planner \
    scenario_filter.scenario_types=[starting_unprotected_cross_turn] \
    +experiment_name=my_experiment
```

**AIDEV-NOTE**: Hydra RC version is intentionally pinned - don't update without testing!

### 2. Planner Interface
All planners implement `AbstractPlanner`:
```python
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner

class MyPlanner(AbstractPlanner):
    def initialize(self, initialization: PlannerInitialization):
        """Called once before simulation"""
        self._map_api = initialization.map_api
        self._route_roadblock_ids = initialization.route_roadblock_ids

    def name(self) -> str:
        return "my_planner"

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """Called every simulation timestep (0.1s typically)"""
        # Access ego state
        ego_state = current_input.history.ego_states[-1]

        # Access observations
        agents = current_input.history.observations[-1].tracked_objects
        traffic_lights = current_input.traffic_light_data

        # Query map
        nearby_lanes = self._map_api.get_proximal_map_objects(
            ego_state.center, radius=50.0
        )

        # Return trajectory (InterpolatedTrajectory or other)
        return self._plan(ego_state, agents, nearby_lanes)
```

**AIDEV-NOTE**: Planners must be registered in configs to be discovered by Hydra

### 3. Scenario System
**Scenarios** are extracted from real driving logs:
- Each scenario is a temporal slice of a log (~20 seconds)
- Classified by maneuver type (lane following, turns, lane changes, etc.)
- Stored in SQLite database (nuplan.db)
- Accessed via `ScenarioBuilder`

**Common scenario types:**
- `starting_left_turn`, `starting_right_turn`
- `near_multiple_vehicles`, `high_lateral_acceleration`
- `traversing_pickup_dropoff`, `on_intersection`

**AIDEV-TODO**: Document custom scenario filters when implementing experiments

### 4. Metrics & Evaluation
Closed-loop simulation produces metrics in categories:
- **Planning metrics**: collision avoidance, drivable area compliance
- **Dynamics metrics**: comfort (jerk, lat/lon acceleration limits)
- **Progress metrics**: time to goal, progress along route
- **Map compliance**: lane following, intersection handling

Metrics are configured in `metric/` configs and computed via callbacks.

## Development Patterns

### Running Experiments
```bash
# 1. Create experiment config (Hydra composition)
# Edit config/experiment/my_experiment.yaml

# 2. Run simulation with custom config
uv run python nuplan/planning/script/run_simulation.py \
    experiment=my_experiment \
    worker=sequential \
    +experiment_name=test_run

# 3. View results in nuBoard
uv run python nuplan/planning/script/run_nuboard.py \
    simulation_path=$NUPLAN_EXP_ROOT/exp/test_run
```

### Adding a New Planner
1. Implement `AbstractPlanner` in `nuplan/planning/simulation/planner/`
2. Register in `config/planner/` (create YAML config)
3. Test with simulation script
4. Add unit tests in `test/` directory

**AIDEV-NOTE**: See simple_planner.py as reference implementation

### Debugging Tips

**CUDA issues:**
```bash
# Check if PyTorch sees GPU
uv run python -c "import torch; print(torch.cuda.is_available())"
just check-cuda

# If driver mismatch, use Docker fallback
just docker-build && just docker-run
```

**Dataset loading errors:**
```bash
# Verify paths are set
just dataset-info

# Check database integrity
sqlite3 $NUPLAN_DATA_ROOT/nuplan.db "SELECT COUNT(*) FROM scenario;"
```

**Hydra config errors:**
```bash
# Debug config composition
uv run python nuplan/planning/script/run_simulation.py --cfg job --resolve
export HYDRA_FULL_ERROR=1  # More detailed errors
```

**Memory issues during training:**
- Reduce batch size in training config
- Enable cache reuse: `cache.use_cache_without_dataset=True`
- Monitor with: `just info` or `htop`

### Testing Patterns
```bash
# Run all tests
just test

# Run specific module tests
just test-path nuplan/planning/simulation/planner/test

# Test with coverage
just test-coverage

# Run notebook tests
uv run pytest --nbmake tutorials/
```

**AIDEV-NOTE**: Pre-commit hooks enforce code quality - don't bypass with --no-verify!

## Common Gotchas & Solutions

### Issue: "No module named 'nuplan'"
**Solution**: Ensure uv environment is active and package is installed
```bash
just setup  # Re-run setup
uv run python -c "import nuplan; print(nuplan.__file__)"  # Verify install
```

### Issue: Hydra ConfigCompositionException
**Solution**: Check config paths and composition
- Verify config file exists in `config/` directory
- Use `--cfg job --resolve` to debug composition
- Check for typos in config overrides

### Issue: Scenario database not found
**Solution**: Set NUPLAN_DATA_ROOT correctly
```bash
# Check if database exists
ls -lh $NUPLAN_DATA_ROOT/nuplan.db

# If missing, download dataset
just cli download --mini  # Mini dataset for testing
```

### Issue: OOM (Out of Memory) during training
**Solution**: Reduce memory footprint
- Decrease `data_loader.params.batch_size`
- Reduce `data_loader.params.num_workers`
- Enable cache: `cache.cache_path=$NUPLAN_EXP_ROOT/cache`
- Clear old cache: `rm -rf $NUPLAN_EXP_ROOT/cache/*`

### Issue: Slow simulation runs
**Solution**: Profile and optimize
- Use `worker.threads_per_node` for parallelism
- Enable caching for scenario building
- Profile with: `uv run python -m cProfile ...`

## Dataset Management

### Dataset Download
```bash
# Full dataset (~10TB) - requires registration at nuscenes.org/nuplan
uv run nuplan_cli download --version v1.1 --data_root $NUPLAN_DATA_ROOT

# Mini dataset (~50GB) - good for tutorials
uv run nuplan_cli download --mini --data_root $NUPLAN_DATA_ROOT

# Maps only
uv run nuplan_cli download --maps_only --data_root $NUPLAN_MAPS_ROOT
```

### Dataset Structure
```
$NUPLAN_DATA_ROOT/
├── nuplan.db                    # Main SQLite database (scenarios, logs)
├── maps/                        # Map data (if not in NUPLAN_MAPS_ROOT)
│   ├── nuplan-maps-v1.0.json   # Map metadata
│   ├── sg-one-north/           # Singapore
│   ├── us-nv-las-vegas/        # Las Vegas
│   ├── us-pa-pittsburgh/       # Pittsburgh
│   └── us-ma-boston/           # Boston
└── sensor_blobs/                # Sensor data (images, lidar)
    └── 2021.XX.XX.XX.XX.XX/    # Timestamped logs
```

### Cache Management
```bash
# Check cache size
du -sh $NUPLAN_EXP_ROOT/cache/

# Clear old cache (free up space)
find $NUPLAN_EXP_ROOT/cache -mtime +30 -delete  # Older than 30 days

# Selective cache clear
rm -rf $NUPLAN_EXP_ROOT/cache/training_cache/*
```

## Advanced Topics

### Custom Metrics
Implement `AbstractMetric` and register in `config/metric/`:
```python
from nuplan.planning.metrics.abstract_metric import AbstractMetric

class MyMetric(AbstractMetric):
    def compute(self, scenario: AbstractScenario, ...) -> MetricStatistics:
        # Compute metric from scenario history
        ...
```

### Custom Observation Models
Extend `AbstractObservation` to add new sensor modalities or processing:
```python
from nuplan.planning.simulation.observation.abstract_observation import AbstractObservation
```

### Distributed Training
Use Ray for multi-GPU training:
```yaml
# config/training/my_training.yaml
trainer:
  params:
    gpus: 4
    strategy: ddp
```

## Coordinate Systems & Units

- **Position**: meters (x, y) in global frame
- **Heading**: radians, 0 = East, counter-clockwise
- **Velocity**: m/s
- **Acceleration**: m/s²
- **Time**: seconds (simulation dt = 0.1s typically)
- **Map**: UTM coordinates for each city

**AIDEV-NOTE**: Be careful with heading conversions - use `nuplan.common.geometry` utilities!

## Resources & Documentation

- **Official Docs**: https://nuplan-devkit.readthedocs.io/
- **Dataset Website**: https://www.nuscenes.org/nuplan
- **GitHub**: https://github.com/motional/nuplan-devkit
- **Paper**: https://arxiv.org/abs/2106.11810
- **Challenge**: https://eval.ai/web/challenges/challenge-page/1856

## Notes for AI Assistants

1. **Always verify environment setup first** - most issues stem from missing env vars or dataset
2. **Respect pinned versions** - especially hydra-core==1.1.0rc1, don't suggest updates
3. **Use Justfile commands** - they handle uv context automatically
4. **CUDA is optional** - code should work CPU-only (slower for training)
5. **Check AIDEV-NOTE and AIDEV-TODO comments** in code for context
6. **Tutorials first** - don't jump into custom development without understanding the framework
7. **Hydra configs** - always check existing configs before creating new ones
8. **Pre-commit hooks** - enforce quality, don't bypass
9. **Dataset size** - be mindful when suggesting full dataset operations (10TB!)
10. **G Money prefers learning by doing** - provide runnable examples, not just explanations

## Migration Notes (Conda → uv)

**What changed:**
- ✅ Faster installs (uv is 10-100x faster)
- ✅ Better dependency resolution
- ✅ Smaller Docker images (~70% reduction)
- ✅ Modern Python packaging (pyproject.toml)
- ✅ Justfile for common commands
- ⚠️ `nb_conda_kernels` removed (use ipykernel instead)
- ⚠️ Pre-release support enabled for Hydra RC

**Backwards compatibility:**
- Old environment.yml preserved for reference
- All pinned versions maintained (phase 1: stability)
- Pre-commit hooks unchanged
- Existing scripts and notebooks work as-is

**Future modernization** (phase 2):
- Upgrade Hydra to stable 1.3+
- Update PyTorch to 2.x
- Modernize other dependencies
- Only after G Money verifies stability!

---

**Remember**: This is G Money's learning environment - prioritize educational value and experimentation over production polish. Make it easy to run experiments, break things, and learn from the dataset! 🧭
