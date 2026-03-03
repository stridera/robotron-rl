# robotron-rl

Robotron 2084 RL Gym Models

https://wandb.ai/stridera/robotron

## ⚠️ Important - All Setup Issues Fixed!

**✅ SETUP COMPLETE!** All version conflicts, import errors, and bugs are fixed.

See **[SETUP_COMPLETE.md](SETUP_COMPLETE.md)** for the complete summary of what was fixed and how to get started.

## Quick Start

```bash
# Installation with Poetry (recommended)
git submodule update --init
poetry install
poetry shell

# Or with pip (alternative)
git submodule update --init
python3 -m venv .venv
source .venv/bin/activate
pip install -e robotron2084gym/
pip install --upgrade -r requirements.txt

# Train (USE THIS - not train.py!)
python train_improved.py --model ppo --config curriculum_config.yaml --num-envs 8

# Monitor on WandB: https://wandb.ai/stridera/robotron
# Evaluate when done
python evaluate.py --model models/{run_id}/best/best_model.zip --render
```

## Documentation

### Start Here
- **[SETUP_COMPLETE.md](SETUP_COMPLETE.md)** ⭐ **START HERE** - Everything that was fixed!

### Installation & Setup
- **[INSTALL.md](INSTALL.md)** - Detailed installation (Poetry & pip)
- **[VERSION_FIX.md](VERSION_FIX.md)** - NumPy/Gymnasium version fixes
- **[GYMNASIUM_FIXES.md](GYMNASIUM_FIXES.md)** - Wrapper compatibility fixes
- **[MIGRATION_SUMMARY.md](MIGRATION_SUMMARY.md)** - Poetry migration details

### Training
- **[QUICK_START.md](QUICK_START.md)** - What was wrong with training
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Comprehensive training guide

### Development
- **[CLAUDE.md](CLAUDE.md)** - Architecture and development guide
- **[robotron2084gym/README.md](robotron2084gym/README.md)** - Gymnasium environment docs
