# Installation Guide

## Prerequisites

- Python 3.10, 3.11, or 3.12
- Poetry (for dependency management)
- CUDA-capable GPU (recommended but not required)

## Option 1: Poetry (Recommended)

Poetry handles all dependencies including the robotron2084gym submodule automatically.

### Install Poetry

If you don't have Poetry installed:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Or on macOS/Linux with brew:

```bash
brew install poetry
```

### Install Project

```bash
# Clone repository
git clone <your-repo-url>
cd robotron-rl

# Initialize git submodule
git submodule init
git submodule update

# Install all dependencies (this installs the gym environment from the submodule)
poetry install

# Activate the virtual environment
poetry shell
```

That's it! Poetry automatically:
- Creates a virtual environment
- Installs the robotron2084gym environment from the submodule in editable mode
- Installs all training dependencies (stable-baselines3, wandb, torch, etc.)

### Usage with Poetry

```bash
# Activate the virtual environment
poetry shell

# Run training
python train_improved.py --model ppo --config curriculum_config.yaml

# Or run without activating shell
poetry run python train_improved.py --model ppo --config curriculum_config.yaml
```

## Option 2: pip + venv (Alternative)

If you prefer pip:

```bash
# Clone repository
git clone <your-repo-url>
cd robotron-rl

# Initialize git submodule
git submodule init
git submodule update

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the gym environment from submodule
pip install -e robotron2084gym/

# Install training dependencies
pip install --upgrade -r requirements.txt
```

## Important Version Notes

**NumPy:** This project uses NumPy 1.26.x (not 2.x) for compatibility with most RL libraries. Stable Baselines3 and many other ML packages are not yet compatible with NumPy 2.x.

**Gymnasium:** Uses version 1.x (the submodule requires ^1.0.0). Make sure you don't have the older `gym` package installed.

## Verify Installation

Test that everything is installed correctly:

```bash
# Check robotron package imports
python -c "from robotron import RobotronEnv; print('✓ Robotron environment installed')"

# Check stable-baselines3
python -c "from stable_baselines3 import PPO; print('✓ Stable Baselines 3 installed')"

# Check wandb
python -c "import wandb; print('✓ WandB installed')"

# Run a quick test
python -c "
from robotron import RobotronEnv
env = RobotronEnv(headless=True)
obs, info = env.reset(seed=42)
print('✓ Environment works!')
print(f'  Observation shape: {obs.shape}')
print(f'  Action space: {env.action_space}')
env.close()
"
```

## CUDA Setup (GPU Training)

Poetry/pip will install PyTorch with CUDA support automatically on Linux. Verify:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

If CUDA is not available but you have an NVIDIA GPU:

```bash
# Uninstall CPU-only torch
poetry remove torch
# Or: pip uninstall torch

# Install CUDA version (example for CUDA 12.1)
poetry add torch --source https://download.pytorch.org/whl/cu121
# Or: pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Updating Dependencies

### With Poetry:

```bash
# Update all dependencies
poetry update

# Update specific package
poetry update stable-baselines3

# Update the gym environment (pull latest submodule changes)
cd robotron2084gym
git pull origin main
cd ..
poetry install  # Reinstall in editable mode
```

### With pip:

```bash
# Update all dependencies
pip install --upgrade -r requirements.txt

# Update the gym environment
cd robotron2084gym
git pull origin main
cd ..
pip install -e robotron2084gym/
```

## Development Setup

If you want to develop the gym environment itself:

```bash
# Install with dev dependencies
cd robotron2084gym
poetry install --with dev

# Run tests
poetry run pytest

# Format code
poetry run black .

# Lint
poetry run ruff check .
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'robotron'"

The gym environment isn't installed. Make sure you:
1. Initialized the git submodule (`git submodule update --init`)
2. Installed the submodule package:
   - Poetry: `poetry install`
   - pip: `pip install -e robotron2084gym/`

### "ImportError: cannot import name 'RobotronEnv'"

Your Python path might be wrong. From the robotron-rl directory:

```bash
# Check what's importable
python -c "import sys; print('\n'.join(sys.path))"

# Should show the virtual environment site-packages
# and robotron2084gym should be in editable mode
```

### Poetry lock file conflicts

```bash
# Regenerate lock file
poetry lock --no-update

# Or force reinstall
rm poetry.lock
poetry install
```

### Submodule is empty

```bash
# Initialize submodule
git submodule init
git submodule update

# Or in one command
git submodule update --init --recursive
```

### Out of memory during training

Reduce parallel environments:

```bash
python train_improved.py --model ppo --num-envs 4  # Instead of 8
```

## Next Steps

Once installed, see:
- **[QUICK_START.md](QUICK_START.md)** - Start training immediately
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Comprehensive training guide
- **[CLAUDE.md](CLAUDE.md)** - Architecture documentation
