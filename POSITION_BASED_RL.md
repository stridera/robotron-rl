# Position-Based RL Implementation

## Overview

We've implemented a **two-stage learning approach** to test whether PPO's failure is due to the vision problem or the control problem.

### The Hypothesis

PPO has been failing to learn Robotron from pixels. The question is **why**:
- **Vision problem**: PPO struggles to learn "what are sprites and where are they?" from 28k-dim pixels
- **Control problem**: PPO fundamentally can't learn the navigation/shooting strategy

### The Test

Train PPO on **ground truth sprite positions** instead of pixels:
- ✅ If it works → Vision was the bottleneck → Build sprite detector (Phase 2)
- ❌ If it fails → PPO can't solve this task → Try different algorithm

---

## What We Built

### 1. GroundTruthPositionWrapper (`position_wrapper.py`)

Converts pixel observations to position-based features:

**Input**: 84x84x4 pixels (28,224 dimensions)

**Output**: Position vector (222 dimensions)
- Player position: `[x, y]` normalized to [-1, 1]
- For 10 nearest sprites:
  - Type: one-hot encoded (17 sprite types)
  - Relative position: `[dx, dy]` from player
  - Distance: normalized to [0, 1]
  - Angle: angle from player to sprite
  - Valid flag: 1 if sprite exists, 0 if padding

**Data source**: Uses `env.unwrapped.engine.get_sprite_data()` for perfect ground truth positions.

**Key features**:
- Handles variable numbers of sprites (sorts by distance, pads to fixed size)
- Normalizes all values for better learning
- No occlusion issues (uses ground truth)

### 2. train_positions.py

Training script for position-based RL:

**Key differences from pixel-based training**:
- `MlpPolicy` instead of `CnnPolicy` (no CNN needed!)
- 222-dim observations instead of 28k-dim pixels
- Deeper MLP: 2x256 layers
- Longer rollouts: 2048 steps (not memory-constrained like CNN)
- Standard MLP learning rate: 3e-4

**Expected benefits**:
- **10-100x faster learning** (smaller obs space)
- **Much lower memory usage** (no image replay buffer)
- **Clearer learning signal** (no vision noise)

---

## How to Use

### Start Position-Based Training

```bash
poetry run python train_positions.py \
  --config ultra_simple_curriculum.yaml \
  --num-envs 8 \
  --device cuda:0 \
  --timesteps 1000000
```

**Monitor training**:
```bash
# Find run ID
ls -t runs/ | head -1

# Check TensorBoard
tensorboard --logdir runs/

# Watch for:
# - robotron/episode_score trending upward
# - robotron/episode_kills > 0 (agent killing grunts)
```

### Evaluate at Checkpoints

```bash
# After 100k steps
poetry run python check_dense_rewards.py models/{run_id}/checkpoints/ppo_positions_checkpoint_100000_steps.zip

# Look for:
# ✅ Action diversity > 4 for movement/shooting
# ✅ Episode scores > 100 (1 grunt = 100 points)
# ✅ Consistent kills
```

### Compare to Pixel-Based Training

| Metric | Pixel-Based (Failed) | Position-Based (?) |
|--------|---------------------|-------------------|
| Observation dims | 28,224 | 222 |
| Policy type | CNN | MLP |
| Memory usage | ~8GB | ~2GB |
| Training speed | Slow | Fast |
| Expected time to learn | >1M steps (failed) | 100k-300k steps (?) |

---

## Success Criteria

### ✅ **Phase 1 Success** (Position-based RL works)

If after 200k-300k steps:
- Agent consistently kills 1 grunt (score > 100)
- Action diversity > 4 for both movement and shooting
- Deterministic policy not collapsed
- Evaluation scores trending upward

**Conclusion**: PPO CAN solve the task! Vision was the bottleneck.

**Next step**: Phase 2 - Build sprite detector
- Train CNN to detect sprites from pixels
- Replace GroundTruthPositionWrapper with DetectorWrapper
- Deploy to Xbox with detector

### ❌ **Phase 1 Failure** (Position-based RL still fails)

If after 500k steps:
- Policy still collapsed or random
- No improvement in scores
- Can't kill even 1 grunt with perfect position info

**Conclusion**: PPO fundamentally can't solve this task.

**Next steps**:
1. Try **Rainbow DQN** (off-policy, better for sparse rewards)
2. Try **DreamerV3** (world model, better for visual RL)
3. Try **SAC** (maximum entropy, better exploration)
4. Consider **hybrid approach** (handcrafted FSM + learned refinements)

---

## Technical Details

### Observation Space Design

**Player features (2 dims)**:
- Normalized position in play area

**Per-sprite features (21 dims each × 10 sprites = 210 dims)**:
- **Type (17 dims)**: One-hot encoding of sprite type
  - Player, Grunt, Electrode, Hulk, Sphereoid, Quark, Brain, Enforcer, Tank
  - Mommy, Daddy, Mikey, Prog, Cruise, PlayerBullet, EnforcerBullet, TankShell
- **Relative position (2 dims)**: `[dx, dy]` from player, normalized
- **Distance (1 dim)**: Euclidean distance, normalized by max distance
- **Angle (1 dim)**: `arctan2(dy, dx)` from player to sprite
- **Valid flag (1 dim)**: 1 if real sprite, 0 if padding

**Total: 2 + 10 × 21 = 222 dims**

### Why This Should Work

**From RL theory**:
- Observation space reduced by **127x** (28,224 → 222)
- Sample complexity scales with `O(|S|)` where `|S|` is state space size
- Approximate speedup: `√(28224/222) ≈ 11x` faster learning

**From practice**:
- Position-based RL is standard for robotics (no vision)
- MLP policies train much faster than CNN policies
- Ground truth eliminates perceptual aliasing

**The key test**: If PPO can't learn with 222-dim perfect info, it definitely can't learn with 28k-dim noisy pixels.

---

## Current Status

- ✅ GroundTruthPositionWrapper implemented and tested
- ✅ train_positions.py created
- ⏳ Waiting for run dj1fct3y (pixel-based with 1 grunt) to complete
- 📅 Ready to start position-based training

**Parallel runs**:
- `dj1fct3y`: Pixel-based PPO with 1 grunt (10x stronger rewards)
- Next: Position-based PPO with 1 grunt (perfect position info)

**Timeline**:
- If dj1fct3y succeeds: PPO works, just needed simpler curriculum
- If dj1fct3y fails: Start position-based training immediately
- Expected: Position-based should learn in 1-3 hours (~200k-500k steps)

---

## Fallback Plan

If position-based PPO fails, we've proven it's an algorithm issue. Next steps:

### Option 1: Rainbow DQN
- Better for sparse rewards (off-policy learning)
- Prioritized replay (focus on rare kills)
- Available in SB3-Contrib
- Implementation: 1-2 hours

### Option 2: DreamerV3
- State-of-the-art for visual RL
- Learns world model, plans in latent space
- Best sample efficiency
- Implementation: 2-3 days (TensorFlow, complex)

### Option 3: Hybrid Approach
- Start with handcrafted rules (simplified FSM)
- Use RL to refine/improve specific behaviors
- Guaranteed baseline performance
- Implementation: 1-2 days

---

## Files Created

1. **position_wrapper.py**: GroundTruthPositionWrapper implementation
2. **train_positions.py**: Training script for position-based RL
3. **POSITION_BASED_RL.md**: This document

**Ready to run**: `poetry run python train_positions.py`
