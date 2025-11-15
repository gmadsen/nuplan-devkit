# ABOUTME: Just command runner for common nuPlan development tasks
# ABOUTME: Run `just` or `just --list` to see all available commands

# Default recipe - show help
default:
    @just --list

# Setup full development environment with CUDA support
setup:
    @echo "📦 Setting up nuPlan development environment with CUDA..."
    uv sync --all-extras
    @echo "✓ Environment ready! Activate with: source .venv/bin/activate"

# Setup CPU-only environment (no CUDA)
setup-cpu:
    @echo "📦 Setting up nuPlan (CPU-only)..."
    uv sync --extra torch-cpu --extra dev --extra tutorials
    @echo "✓ CPU environment ready!"

# Install just the core dependencies (no dev tools)
install:
    @echo "📦 Installing nuPlan core dependencies..."
    uv sync --extra torch-cuda11
    @echo "✓ Core installation complete!"

# Update all dependencies to latest compatible versions
update:
    @echo "🔄 Updating dependencies..."
    uv lock --upgrade
    uv sync
    @echo "✓ Dependencies updated!"

# Run Jupyter Lab for tutorials
tutorial:
    @echo "🎓 Launching Jupyter Lab for nuPlan tutorials..."
    uv run jupyter lab tutorials/

# Run specific tutorial notebook
notebook name:
    @echo "🎓 Opening tutorial: {{name}}"
    uv run jupyter lab tutorials/{{name}}.ipynb

# Run all tests
test:
    @echo "🧪 Running test suite..."
    uv run pytest -v

# Run tests with coverage
test-coverage:
    @echo "🧪 Running tests with coverage..."
    uv run pytest --cov=nuplan --cov-report=html --cov-report=term
    @echo "📊 Coverage report: htmlcov/index.html"

# Run specific test file or directory
test-path path:
    @echo "🧪 Running tests in: {{path}}"
    uv run pytest {{path}} -v

# Lint code with all pre-commit hooks
lint:
    @echo "🔍 Running linters..."
    uv run pre-commit run --all-files --hook-stage manual

# Format code with black and isort
format:
    @echo "✨ Formatting code..."
    uv run black nuplan tutorials --line-length=120 --skip-string-normalization
    uv run isort nuplan tutorials --line-length=120 --profile=black
    @echo "✓ Code formatted!"

# Type check with mypy
typecheck:
    @echo "🔍 Type checking..."
    uv run mypy nuplan --config-file=.mypy.ini --ignore-missing-imports

# Run nuplan CLI command
cli *args:
    uv run nuplan_cli {{args}}

# Build Docker image with uv
docker-build:
    @echo "🐳 Building Docker image..."
    docker build -t nuplan-devkit:latest .
    @echo "✓ Docker image built!"

# Run Docker container
docker-run:
    @echo "🐳 Starting Docker container..."
    docker-compose up

# Clean generated files and caches
clean:
    @echo "🧹 Cleaning build artifacts..."
    rm -rf build/ dist/ *.egg-info .pytest_cache/ .coverage htmlcov/
    find . -type d -name __pycache__ -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete
    @echo "✓ Clean complete!"

# Deep clean including uv cache and venv
clean-all: clean
    @echo "🧹 Deep cleaning (including venv)..."
    rm -rf .venv/
    uv cache clean
    @echo "✓ Deep clean complete!"

# Show environment info
info:
    @echo "📊 nuPlan Environment Info:"
    @echo "----------------------------"
    @echo "Python version:"
    @uv run python --version
    @echo "\nPyTorch version:"
    @uv run python -c "import torch; print(f'  {torch.__version__}')" 2>/dev/null || echo "  Not installed"
    @echo "\nCUDA available:"
    @uv run python -c "import torch; print(f'  {torch.cuda.is_available()}')" 2>/dev/null || echo "  N/A"
    @echo "\nCUDA devices:"
    @uv run python -c "import torch; print(f'  {torch.cuda.device_count()} device(s)')" 2>/dev/null || echo "  N/A"
    @echo "\nEnvironment variables:"
    @env | grep NUPLAN || echo "  No NUPLAN_* vars set"
    @echo "\nuv version:"
    @uv --version

# Check CUDA setup
check-cuda:
    @echo "🔍 Checking CUDA setup..."
    @uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}' if torch.cuda.is_available() else 'No CUDA'); print(f'Device Count: {torch.cuda.device_count()}'); [print(f'Device {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

# Install pre-commit hooks
install-hooks:
    @echo "🪝 Installing pre-commit hooks..."
    uv run pre-commit install
    @echo "✓ Pre-commit hooks installed!"

# Download nuPlan datasets (unified flexible command)
# Examples:
#   just download --all-db              # All database files (test, val, train)
#   just download --test --val          # Specific datasets
#   just download --interactive         # Interactive mode
#   just download --all-db --extract    # Download and extract
download *args:
    @echo "📥 nuPlan Dataset Download Manager"
    @echo "⚠️  Ensure NUPLAN_DATA_ROOT is set in your environment"
    uv run nuplan_cli download datasets {{args}}

# Set up environment variables template
setup-env:
    @if [ ! -f .env ]; then \
        cp .env.example .env; \
        echo "📝 Created .env file - please edit with your paths"; \
    else \
        echo "⚠️  .env already exists, not overwriting"; \
    fi

# Quick smoke test - verify installation works
smoke-test:
    @echo "💨 Running smoke test..."
    @uv run python -c "import nuplan; print('✓ nuPlan imports successfully')"
    @uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('✓ CUDA is available')"
    @echo "✓ Smoke test passed!"

# Comprehensive health check of the environment
health:
    @echo "🏥 Running comprehensive health check..."
    uv run python scripts/health_check.py

# Show dataset statistics (if dataset is available)
dataset-info:
    @echo "📊 Dataset Information:"
    @uv run python -c "import os; root=os.getenv('NUPLAN_DATA_ROOT', 'Not set'); print(f'NUPLAN_DATA_ROOT: {root}')"
    @uv run python -c "import os; root=os.getenv('NUPLAN_MAPS_ROOT', 'Not set'); print(f'NUPLAN_MAPS_ROOT: {root}')"
    @uv run python -c "import os; root=os.getenv('NUPLAN_EXP_ROOT', 'Not set'); print(f'NUPLAN_EXP_ROOT: {root}')"

# Explore full nuPlan dataset structure (all train/val/test splits)
explore:
    @echo "🔍 Exploring nuPlan dataset structure..."
    uv run nuplan_cli explore datasets

# Explore sensor blob zips available in S3 (scrape actual file list)
explore-sensors set="mini":
    @echo "🔍 Exploring sensor blob zips for {{set}}_set..."
    uv run nuplan_cli explore sensors --sensor-set={{set}}

# Show dataset inventory (local vs remote)
inventory *args:
    @echo "📦 Dataset Inventory:"
    uv run nuplan_cli inventory main {{args}}

# Show which logs have sensor blobs
inventory-logs *args:
    @echo "📦 Sensor Blob Status:"
    uv run nuplan_cli inventory logs {{args}}

# Map log or DB file to required sensor blob zips
map-log log_name:
    @echo "🗺️  Mapping log to sensor blob zips:"
    uv run nuplan_cli map log {{log_name}}

# Map DB files to required sensor blob zips
map-db *db_files:
    @echo "🗺️  Mapping DB files to sensor blob zips:"
    uv run nuplan_cli map db {{db_files}}

# Download tutorial setup (mini dataset + camera_0 + lidar_0)
download-tutorial:
    @echo "📥 Generating download commands for tutorial setup..."
    @echo "   (mini DB + camera_0 + lidar_0 = ~418 GB)"
    uv run nuplan_cli download datasets --mini --camera=0 --lidar=0

# Download all database files (no sensors) - what most researchers need
download-db:
    @echo "📥 Generating download commands for all database files..."
    @echo "   (test + val + all train splits, no sensors)"
    uv run nuplan_cli download datasets --all-db

# Interactive download mode - shows table and prompts for selection
download-interactive:
    @echo "📥 Starting interactive download mode..."
    uv run nuplan_cli download datasets --interactive

# Interactive Python shell with nuPlan loaded
shell:
    @echo "🐍 Starting Python shell with nuPlan..."
    uv run ipython

# Generate documentation
docs:
    @echo "📚 Generating documentation..."
    cd docs && uv run make html
    @echo "📖 Docs: docs/_build/html/index.html"

# Serve documentation locally
docs-serve:
    @echo "📚 Serving documentation at http://localhost:8000"
    cd docs/_build/html && python -m http.server 8000

# Train a raster model (full tutorial training)
train:
    @echo "🚗 Training raster model (10 epochs, 500 scenarios)..."
    @echo "   Output: $${NUPLAN_EXP_ROOT:-$$HOME/nuplan/exp}/tutorial_nuplan_framework/training_raster_experiment/"
    @echo "   Monitor: tail -f $${NUPLAN_EXP_ROOT:-$$HOME/nuplan/exp}/tutorial_nuplan_framework/training_raster_experiment/train_default_raster/*/log.txt"
    uv run python nuplan/planning/script/run_training.py \
        group=$${NUPLAN_EXP_ROOT:-$$HOME/nuplan/exp}/tutorial_nuplan_framework \
        cache.cache_path=$${NUPLAN_EXP_ROOT:-$$HOME/nuplan/exp}/cache \
        experiment_name=training_raster_experiment \
        job_name=train_default_raster \
        py_func=train \
        +training=training_raster_model \
        scenario_builder=nuplan_mini \
        scenario_filter.limit_total_scenarios=500 \
        lightning.trainer.params.accelerator=ddp \
        lightning.trainer.params.max_epochs=10 \
        data_loader.params.batch_size=8 \
        data_loader.params.num_workers=8

# Quick training run (reduced epochs/scenarios for testing)
train-quick:
    @echo "🚗 Quick training (3 epochs, 100 scenarios)..."
    uv run python nuplan/planning/script/run_training.py \
        group=$${NUPLAN_EXP_ROOT:-$$HOME/nuplan/exp}/tutorial_nuplan_framework \
        cache.cache_path=$${NUPLAN_EXP_ROOT:-$$HOME/nuplan/exp}/cache \
        experiment_name=training_raster_quick \
        job_name=train_quick \
        py_func=train \
        +training=training_raster_model \
        scenario_builder=nuplan_mini \
        scenario_filter.limit_total_scenarios=100 \
        lightning.trainer.params.accelerator=ddp \
        lightning.trainer.params.max_epochs=3 \
        data_loader.params.batch_size=8 \
        data_loader.params.num_workers=8

# Run simulation with simple planner
simulate:
    @echo "🎮 Running simulation with simple planner..."
    uv run python nuplan/planning/script/run_simulation.py \
        experiment_name=simulation_simple_experiment \
        group=$${NUPLAN_EXP_ROOT:-$$HOME/nuplan/exp}/tutorial_nuplan_framework \
        planner=simple_planner \
        +simulation=open_loop_boxes \
        scenario_builder=nuplan_mini \
        scenario_filter=all_scenarios \
        scenario_filter.scenario_types=[near_multiple_vehicles,on_pickup_dropoff,starting_unprotected_cross_turn,high_magnitude_jerk] \
        scenario_filter.num_scenarios_per_type=10

# Run simulation with trained ML planner (auto-detects latest checkpoint if not specified)
simulate-ml checkpoint="":
    #!/usr/bin/env bash
    set -euo pipefail

    EXP_ROOT="${NUPLAN_EXP_ROOT:-$HOME/nuplan/exp}"

    if [ -z "{{checkpoint}}" ]; then
        echo "🔍 No checkpoint specified, finding most recent..."
        CHECKPOINT=`find "$EXP_ROOT/tutorial_nuplan_framework/training_raster_experiment" -name "*.ckpt" -type f 2>/dev/null | xargs ls -t 2>/dev/null | head -1`
        if [ -z "$CHECKPOINT" ]; then
            echo "❌ No checkpoints found in $EXP_ROOT/tutorial_nuplan_framework/training_raster_experiment/"
            echo "   Run 'just train' first to create a checkpoint"
            exit 1
        fi
        echo "   Using: $CHECKPOINT"
    else
        CHECKPOINT="{{checkpoint}}"
        echo "🎮 Running simulation with ML planner..."
        echo "   Using checkpoint: $CHECKPOINT"
    fi

    # AIDEV-NOTE: Use direct python instead of uv run to avoid Ray uv integration issues
    # Ray's uv integration creates minimal venvs without extras like torch-cuda11
    .venv/bin/python nuplan/planning/script/run_simulation.py \
        experiment_name=simulation_raster_experiment \
        group="$EXP_ROOT/tutorial_nuplan_framework" \
        planner=ml_planner \
        model=raster_model \
        planner.ml_planner.model_config=\${model} \
        planner.ml_planner.checkpoint_path="$CHECKPOINT" \
        +simulation=open_loop_boxes \
        scenario_builder=nuplan_mini \
        scenario_filter=all_scenarios \
        scenario_filter.scenario_types=[near_multiple_vehicles,on_pickup_dropoff,starting_unprotected_cross_turn,high_magnitude_jerk] \
        scenario_filter.num_scenarios_per_type=10 \
        worker.threads_per_node=4

# Launch nuBoard visualization dashboard
nuboard *paths:
    @echo "📊 Launching nuBoard dashboard..."
    @if [ -z "{{paths}}" ]; then \
        echo "   No simulation paths provided - you can load them in the UI"; \
        uv run python nuplan/planning/script/run_nuboard.py scenario_builder=nuplan_mini; \
    else \
        echo "   Loading: {{paths}}"; \
        uv run python nuplan/planning/script/run_nuboard.py scenario_builder=nuplan_mini simulation_path='[{{paths}}]'; \
    fi

# Monitor training progress (follows log file)
train-monitor:
    @echo "👀 Monitoring training logs..."
    @EXP_ROOT="$${NUPLAN_EXP_ROOT:-$$HOME/nuplan/exp}"; \
    LOG_FILE=$$(ls -t "$$EXP_ROOT/tutorial_nuplan_framework/training_raster_experiment/train_*/*/log.txt" 2>/dev/null | head -1); \
    if [ -z "$$LOG_FILE" ]; then \
        echo "❌ No training logs found in $$EXP_ROOT/tutorial_nuplan_framework/training_raster_experiment/"; \
        exit 1; \
    else \
        echo "   Following: $$LOG_FILE"; \
        tail -f "$$LOG_FILE"; \
    fi

# Launch TensorBoard for training visualization
tensorboard:
    @echo "📈 Launching TensorBoard..."
    @EXP_ROOT="$${NUPLAN_EXP_ROOT:-$$HOME/nuplan/exp}"; \
    LATEST_RUN=$$(ls -td "$$EXP_ROOT/tutorial_nuplan_framework/training_raster_experiment/train_default_raster/"*/ 2>/dev/null | head -1); \
    if [ -z "$$LATEST_RUN" ]; then \
        echo "❌ No training runs found"; \
        exit 1; \
    else \
        echo "   Viewing: $$LATEST_RUN"; \
        echo "   URL: http://localhost:6010"; \
        uv run tensorboard --logdir "$$LATEST_RUN" --port 6010; \
    fi

# List available model checkpoints
checkpoints:
    #!/usr/bin/env bash
    echo "📦 Available model checkpoints:"
    EXP_ROOT="${NUPLAN_EXP_ROOT:-$HOME/nuplan/exp}"
    find "$EXP_ROOT/tutorial_nuplan_framework/training_raster_experiment" -name "*.ckpt" -type f 2>/dev/null | \
        while read ckpt; do \
            size=$(du -h "$ckpt" | cut -f1); \
            echo "   [$size] $ckpt"; \
        done || echo "   No checkpoints found"

# Clean up temp directories (Ray sessions, stale cache)
clean-tmp:
    @echo "🧹 Cleaning up temp directories..."
    @EXP_ROOT="$${NUPLAN_EXP_ROOT:-$$HOME/nuplan/exp}"; \
    echo "📊 Before cleanup:"; \
    du -sh ~/.tmp 2>/dev/null || echo "   ~/.tmp: not found"; \
    du -sh /tmp/ray 2>/dev/null || echo "   /tmp/ray: not found"; \
    du -sh "$$EXP_ROOT/cache" 2>/dev/null || echo "   $$EXP_ROOT/cache: not found"; \
    echo ""; \
    echo "🗑️  Removing Ray sessions..."; \
    rm -rf ~/.tmp/ray/session-* 2>/dev/null || true; \
    rm -rf /tmp/ray/session-* 2>/dev/null || true; \
    echo "🗑️  Removing old nuPlan cache (>30 days)..."; \
    find "$$EXP_ROOT/cache" -mtime +30 -type f -delete 2>/dev/null || true; \
    echo ""; \
    echo "📊 After cleanup:"; \
    du -sh ~/.tmp 2>/dev/null || echo "   ~/.tmp: not found"; \
    du -sh /tmp/ray 2>/dev/null || echo "   /tmp/ray: not found"; \
    du -sh "$$EXP_ROOT/cache" 2>/dev/null || echo "   $$EXP_ROOT/cache: not found"; \
    echo "✓ Cleanup complete!"
