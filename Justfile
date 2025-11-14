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

# Download nuPlan dataset (requires credentials)
download-dataset:
    @echo "📥 Downloading nuPlan dataset..."
    @echo "⚠️  Ensure NUPLAN_DATA_ROOT is set in your environment"
    uv run nuplan_cli download

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
