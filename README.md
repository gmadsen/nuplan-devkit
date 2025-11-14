# nuPlan DevKit

The official devkit of the nuPlan dataset (www.nuPlan.org).

nuPlan is the world's first large-scale planning benchmark for autonomous vehicles, featuring:
- 📊 **1,300+ hours** of real-world driving data
- 🌆 **4 cities**: Las Vegas, Singapore, Pittsburgh, Boston
- 🎯 **Diverse scenarios**: Unprotected turns, lane changes, parking, and more
- 🤖 **ML-ready**: PyTorch Lightning training pipeline
- 📈 **Rich metrics**: Safety, comfort, progress evaluation

## ✨ Enhanced Dataset Management Tools

This fork includes **comprehensive CLI tools** for managing the massive nuPlan dataset (144 sensor blob zips across train/val/test!):

### 🔍 Explore the Complete Dataset
```bash
just explore                    # View all 13 database splits + 144 sensor blob sets
just explore-sensors train      # See what's in train_set (43 camera + 43 lidar)
```

### 📦 Inventory - Know What You Have
```bash
just inventory                  # Beautiful table: local vs remote comparison
just inventory-logs             # Which logs have sensor data locally
```

### 🗺️ Smart Mapping - Find What You Need
```bash
# Map a specific log to required sensor zips
just map-log 2021.05.12.22.00.38_veh-35_01008_01518

# Map all mini DBs to show which sensor sets are needed
just map-db /path/to/splits/mini/*.db --summary
```

### 📥 Selective Downloads - Save Bandwidth & Storage
```bash
# Generate download scripts for tutorial (camera_0 + lidar_0 = ~410GB)
just download-tutorial

# Custom sensor sets (download only what you need!)
just download-sensors camera="0,1,2" lidar="0,1"
```

**Why use these tools?**
- ✅ **Full dataset visibility** - Understand all 144 sensor blob zips (not just mini's 18)
- ✅ **Selective downloads** - Download specific sensor sets instead of all 15TB
- ✅ **Smart mapping** - Know exactly which zips contain your scenario's sensor data
- ✅ **Beautiful output** - Rich CLI with tables, colors, progress indicators

See [CLAUDE.md](CLAUDE.md#dataset-management) for complete documentation.

## Quick Start

### Prerequisites

- **Python 3.9** (required)
- **CUDA 11.1+** (optional, for GPU acceleration)
- **~10TB storage** (full dataset) or **~50GB** (mini dataset)
- **uv** package manager (or Docker)

### Installation (Native with uv - Recommended)

**1. Install uv** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. Clone the repository**:
```bash
git clone https://github.com/motional/nuplan-devkit.git
cd nuplan-devkit
```

**3. Set up environment**:
```bash
# Full development environment with CUDA support
just setup

# Or manually:
uv sync --all-extras

# For CPU-only (no CUDA):
just setup-cpu
```

**4. Configure dataset paths**:
```bash
# Create .env file from template
cp .env.example .env

# Edit .env and set your paths:
# export NUPLAN_DATA_ROOT="/path/to/dataset"
# export NUPLAN_MAPS_ROOT="/path/to/maps"
# export NUPLAN_EXP_ROOT="/path/to/experiments"

# Load environment
source .env
```

**5. Download dataset** (requires registration at nuscenes.org/nuplan):
```bash
# Mini dataset for tutorials (~50GB)
uv run nuplan_cli download --mini

# Or use just command
just cli download --mini
```

**6. Run tutorials**:
```bash
just tutorial
# Opens Jupyter Lab with tutorial notebooks
```

### Installation (Docker - CUDA Conflict Fallback)

If you encounter CUDA/driver compatibility issues:

```bash
# Build Docker image
just docker-build

# Run container
just docker-run
```

### Installation (Legacy Conda)

The project was migrated from conda to uv for better performance. If you need the old conda setup, check `environment.yml` (preserved for reference).

## Development Commands

This project uses [Just](https://github.com/casey/just) for common commands. Run `just` or `just --list` to see all available commands:

```bash
# Setup & Installation
just setup              # Full dev environment (CUDA)
just setup-cpu          # CPU-only environment
just install            # Core dependencies only

# Dataset Management (NEW!)
just explore            # Show complete dataset structure (144 sensor zips)
just inventory          # Check what's downloaded locally
just map-log <log>      # Find which sensor zips contain a log
just download-tutorial  # Generate download scripts for tutorial

# Development
just tutorial           # Launch Jupyter Lab
just notebook <name>    # Open specific tutorial
just test               # Run test suite
just lint               # Check code quality
just format             # Auto-format code

# Information
just info               # Show environment info
just check-cuda         # Verify GPU setup
just dataset-info       # Show dataset paths

# Utilities
just clean              # Clean build artifacts
just clean-all          # Deep clean (includes venv)
```

## Project Structure

```
nuplan-devkit/
├── nuplan/                      # Main package
│   ├── cli/                     # CLI tools
│   ├── common/                  # Utilities, maps, geometry
│   ├── database/                # Dataset access layer
│   ├── planning/                # Planning algorithms & simulation
│   │   ├── metrics/             # Evaluation metrics
│   │   ├── nuboard/             # Visualization dashboard
│   │   ├── scenario_builder/   # Scenario extraction
│   │   ├── simulation/          # Simulation loop
│   │   └── training/            # ML training pipeline
│   └── submission/              # Competition code
├── tutorials/                   # Jupyter notebooks (start here!)
├── docs/                        # Sphinx documentation
└── CLAUDE.md                    # AI assistant guide
```

## Tutorials

The `tutorials/` directory contains Jupyter notebooks that walk through the key concepts:

1. **nuplan_framework.ipynb** - Architecture overview
2. **nuplan_scenario_visualization.ipynb** - Exploring scenarios
3. **nuplan_planner_tutorial.ipynb** - Building planners
4. **nuplan_simulation.ipynb** - Running simulations
5. **nuplan_training.ipynb** - Training ML models

**Start with the tutorials to learn the framework!**

## Usage Examples

### Running Simulations

```bash
# Run simulation with simple planner
uv run python nuplan/planning/script/run_simulation.py \
    planner=simple_planner \
    scenario_filter=one_of_each_scenario_type \
    +experiment_name=my_test

# Use just shortcut
just cli run_simulation planner=simple_planner
```

### Training a Model

```bash
# Cache training features
uv run python nuplan/planning/script/run_training.py \
    experiment=training_simple_model \
    py_func=cache

# Train model
uv run python nuplan/planning/script/run_training.py \
    experiment=training_simple_model \
    py_func=train
```

### Visualizing Results

```bash
# Launch nuBoard dashboard
uv run python nuplan/planning/script/run_nuboard.py \
    simulation_path=$NUPLAN_EXP_ROOT/exp/my_test
```

## Configuration

nuPlan uses [Hydra](https://hydra.cc/) for configuration management. Configs are in `config/`:

```
config/
├── simulation/          # Simulation settings
├── planner/            # Planner configs
├── scenario_filter/    # Scenario selection
├── metric/             # Evaluation metrics
└── experiment/         # Composed experiments
```

**Override configs via CLI**:
```bash
planner=my_planner scenario_filter.limit_total_scenarios=10
```

## Development

### Running Tests

```bash
# All tests
just test

# Specific test
just test-path nuplan/planning/simulation/test

# With coverage
just test-coverage
```

### Code Quality

```bash
# Format code
just format

# Run linters
just lint

# Type checking
just typecheck
```

### Pre-commit Hooks

Install pre-commit hooks to enforce code quality:

```bash
just install-hooks
```

Hooks run automatically on commit:
- Black (formatting)
- isort (import sorting)
- flake8 (linting)
- mypy (type checking)

## System Requirements

### Minimum
- **CPU**: 8 cores
- **RAM**: 16GB
- **Storage**: 100GB (mini dataset + cache)
- **Python**: 3.9

### Recommended
- **CPU**: 16+ cores
- **RAM**: 32GB+
- **GPU**: NVIDIA GPU with 8GB+ VRAM (Titan RTX, RTX 3080, etc.)
- **Storage**: 500GB SSD (for experiments) + 10TB HDD (full dataset)
- **Python**: 3.9
- **CUDA**: 11.1+

## Migration from Conda

This repository was migrated from conda to **uv** for faster, more reliable dependency management.

### Benefits of uv
- ⚡ **10-100x faster** than conda/pip
- 📦 **70% smaller** Docker images
- 🔒 **Better dependency resolution** with comprehensive lock files
- 🛠️ **Modern tooling** with better IDE integration

### For Existing Users

If you have an existing conda environment:

```bash
# Deactivate conda
conda deactivate

# Remove old environment (optional)
conda env remove -n nuplan

# Set up with uv
just setup
```

### Backwards Compatibility

- ✅ All pinned versions preserved (phase 1: stability)
- ✅ Pre-commit hooks unchanged
- ✅ Existing scripts and notebooks work as-is
- ✅ Docker still available for CUDA conflicts
- ⚠️ `nb_conda_kernels` replaced with `ipykernel`

## Troubleshooting

### CUDA/GPU Issues

```bash
# Check CUDA setup
just check-cuda

# If driver mismatch or conflicts
just docker-build && just docker-run
```

### Dataset Loading Errors

```bash
# Verify environment variables
just dataset-info

# Check database
sqlite3 $NUPLAN_DATA_ROOT/nuplan.db "SELECT COUNT(*) FROM scenario;"
```

### Memory Issues

- Reduce batch size in training config
- Clear cache: `rm -rf $NUPLAN_EXP_ROOT/cache/*`
- Monitor usage: `htop` or `just info`

### Import Errors

```bash
# Verify installation
uv run python -c "import nuplan; print(nuplan.__file__)"

# Reinstall
just setup
```

## Contributing

Contributions are welcome! Please:

1. Follow code quality standards (pre-commit hooks)
2. Add tests for new features
3. Update documentation
4. Use descriptive commit messages

## Citation

If you use nuPlan in your research, please cite:

```bibtex
@inproceedings{nuplan2021,
  title={nuPlan: A Closed-loop ML-based Planning Benchmark for Autonomous Vehicles},
  author={...},
  booktitle={...},
  year={2021}
}
```

## License

apache-2.0 - Free for non-commercial use. See [LICENSE.txt](LICENSE.txt) for details.

## Resources

- 📖 **Documentation**: https://nuplan-devkit.readthedocs.io/
- 🌐 **Dataset Website**: https://www.nuscenes.org/nuplan
- 💻 **GitHub**: https://github.com/motional/nuplan-devkit
- 📝 **Paper**: https://arxiv.org/abs/2106.11810
- 🏆 **Challenge**: https://eval.ai/web/challenges/challenge-page/1856

## Support

For issues, questions, or contributions:
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/motional/nuplan-devkit/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/motional/nuplan-devkit/discussions)
- 📧 **Email**: nuscenes@motional.com

---

**Happy Planning! 🚗💨**
