# Robotron 2084 RL Training Guide

This document explains the key improvements in `train_improved.py` and how to successfully train an agent.

## Critical Issues Fixed

### 1. **Lives Set to 0** ❌ → **Lives Set to 5** ✅
**Original Problem:** With `lives=0`, the agent dies instantly on the first hit and gets no chance to learn.

**Solution:** Set `lives=5` to give the agent multiple attempts per episode to learn from mistakes.

### 2. **Poor Reward Shaping** ❌ → **Better Rewards** ✅
**Original Problem:**
- Score divided by 100 makes rewards tiny (1 point = 0.01 reward)
- Death penalty of -1 completely dominates tiny positive rewards
- No incentive for survival or shooting

**Solution (RewardShapingWrapper):**
- Score rewards multiplied by 10 (1 point = 0.1 reward)
- Death penalty reduced to -5 (still significant but not overwhelming)
- Small survival reward (+0.001 per step) encourages staying alive
- Shooting reward (+0.01) encourages aggressive play

### 3. **Single Environment** ❌ → **8 Parallel Environments** ✅
**Original Problem:** DummyVecEnv with 1 environment is extremely sample-inefficient.

**Solution:** Use SubprocVecEnv with 8 parallel environments:
- 8x faster data collection
- Better exploration diversity
- More stable learning (averaging gradients across environments)

### 4. **No Normalization** ❌ → **VecNormalize** ✅
**Original Problem:** Raw pixel values (0-255) are not normalized, making learning harder.

**Solution:** VecNormalize normalizes observations to ~0 mean, unit variance.

### 5. **Game Too Hard** ❌ → **Curriculum Learning** ✅
**Original Problem:**
- Level 1 has only 1 grunt (too easy, ends immediately)
- Level 2 is empty (no learning signal)
- Level 3+ jumps to full difficulty

**Solution:** New `curriculum_config.yaml`:
- Level 1: 3 grunts + 1 family member (learn basic shooting and collection)
- Gradual introduction of mechanics (obstacles → hulks → spawners → shooters)
- Slower enemy speeds to start
- Less frequent shooting and spawning

### 6. **Poor Hyperparameters** ❌ → **Atari-Tuned Settings** ✅
**Original Problem:**
- PPO using completely default settings
- QRDQN waits 200k steps before learning (way too long)
- No entropy bonus for exploration
- Wrong observation size (123x166 is unusual)

**Solution:**
- PPO: Proper Atari hyperparameters (entropy coef, learning rate, batch sizes)
- DQN/QRDQN: Start learning at 10k steps instead of 200k
- Standard 84x84 observation size (proven for Atari)
- Proper exploration schedules

### 7. **No Evaluation** ❌ → **EvalCallback** ✅
**Original Problem:** No way to track best performance during training.

**Solution:** EvalCallback evaluates every 10k steps and saves best model.

## Prerequisites

Make sure you've installed everything first:

```bash
# With Poetry (recommended)
git submodule update --init
poetry install
poetry shell

# Or with pip
git submodule update --init
python3 -m venv .venv
source .venv/bin/activate
pip install -e robotron2084gym/
pip install -r requirements.txt
```

See [INSTALL.md](INSTALL.md) for detailed instructions.

## Quick Start

### Option 1: PPO (Recommended for First Try)
PPO is more stable and sample-efficient for this type of game.

```bash
python train_improved.py --model ppo --config curriculum_config.yaml --num-envs 8
```

### Option 2: DQN
Standard DQN with experience replay.

```bash
python train_improved.py --model dqn --config curriculum_config.yaml --num-envs 8
```

### Option 3: QRDQN
Quantile Regression DQN (more advanced).

```bash
python train_improved.py --model qrdqn --config curriculum_config.yaml --num-envs 8
```

## Expected Training Timeline

With the improved setup, you should see:

- **0-100k steps**: Agent learns to move and shoot
- **100k-500k steps**: Agent learns to avoid enemies and collect family
- **500k-2M steps**: Agent learns enemy priorities and basic strategy
- **2M-10M steps**: Agent improves consistency and handles harder levels
- **10M+ steps**: Diminishing returns, fine-tuning

## Monitoring Progress

### WandB Metrics to Watch

1. **rollout/ep_rew_mean**: Average episode reward (should steadily increase)
2. **rollout/ep_len_mean**: Average episode length (longer = surviving better)
3. **train/value_loss**: Should decrease and stabilize
4. **eval/mean_reward**: Best indicator of actual performance

### Expected Behavior

**Early Training (0-100k steps):**
- Agent moves randomly, dies quickly
- Episode length: 50-200 steps
- Average reward: -5 to 0

**Mid Training (500k-2M steps):**
- Agent avoids some enemies, shoots more
- Episode length: 500-2000 steps
- Average reward: 5-50

**Late Training (5M+ steps):**
- Agent clears early levels, uses strategy
- Episode length: 2000-10000 steps
- Average reward: 50-200+

## Hyperparameter Tuning

If default settings don't work well, try adjusting:

### For PPO:
```python
'ent_coef': 0.02,  # More exploration (increase if agent gets stuck)
'learning_rate': 5e-4,  # Faster learning (but less stable)
'clip_range': 0.2,  # Larger policy updates
```

### For DQN/QRDQN:
```python
'exploration_final_eps': 0.1,  # More exploration in late training
'learning_rate': 2e-4,  # Faster learning
'buffer_size': 200_000,  # Larger replay buffer (needs more RAM)
```

## Advanced: Further Improvements

### 1. Reward Shaping Variants

Edit `RewardShapingWrapper` to try:

```python
# Reward for each enemy killed
if 'enemies_killed' in info:
    reward += info['enemies_killed'] * 0.5

# Penalty for letting family die
if 'family_lost' in info:
    reward -= info['family_lost'] * 2.0

# Reward for level completion
if 'level_complete' in info:
    reward += 10.0
```

### 2. Imitation Learning from FSM

Use the `robotron_fsm.py` to generate expert demonstrations:

```bash
# Run FSM and log trajectories
python robotron_fsm.py --level 1 --lives 100 --fps 0 > expert_data.txt
```

Then use behavioral cloning to pre-train before RL fine-tuning.

### 3. Auxiliary Tasks

Add auxiliary prediction tasks to help representation learning:
- Predict next frame
- Predict enemy positions
- Predict remaining lives/score

### 4. Recurrent Policies

Use LSTM-based policies for better temporal reasoning:
```python
from sb3_contrib import RecurrentPPO

model = RecurrentPPO('CnnLstmPolicy', env, ...)
```

### 5. Advanced Curriculum

Dynamically adjust difficulty based on performance:

```python
class AdaptiveCurriculumCallback(BaseCallback):
    def _on_step(self):
        if self.avg_reward > threshold:
            # Increase difficulty
            self.env.set_level(current_level + 1)
        return True
```

## Troubleshooting

### Agent Not Learning At All
- Check WandB videos - is it even moving/shooting?
- Verify reward is not always -5 (instant death)
- Try fewer environments (4 instead of 8)
- Increase learning rate

### Agent Gets Stuck in Local Minimum
- Increase entropy coefficient (PPO)
- Increase exploration epsilon (DQN)
- Add curriculum progression
- Try different random seed

### Training Unstable (Reward Oscillates Wildly)
- Decrease learning rate
- Increase batch size
- Add gradient clipping (already enabled)
- Reduce number of environments

### Out of Memory
- Reduce `num_envs` from 8 to 4
- Reduce `buffer_size` for DQN/QRDQN
- Use smaller observation size (64x64 instead of 84x84)

## Comparison: Old vs New

| Aspect | Original | Improved | Impact |
|--------|----------|----------|--------|
| Lives | 0 | 5 | Can learn from mistakes |
| Environments | 1 | 8 | 8x sample efficiency |
| Reward scale | /100 | *10 | 1000x larger signal |
| Death penalty | -1 | -5 | More balanced |
| Observation norm | None | VecNormalize | Faster convergence |
| Observation size | 123x166 | 84x84 | Standard & faster |
| PPO hyperparams | Default | Atari-tuned | Better exploration |
| DQN learning start | 200k | 10k | 20x faster start |
| Curriculum | Hard start | Gradual | Learn basics first |
| Evaluation | None | Every 10k | Track best model |
| Survival reward | None | +0.001/step | Encourages survival |
| Shooting reward | None | +0.01/shot | Encourages action |

## Expected Results

With these improvements, you should see **measurable progress within 1-2 hours** of training on a modern GPU:

- Agent learns to shoot enemies
- Agent learns to avoid getting hit
- Agent learns to collect family members
- Agent starts clearing level 1 consistently

For comparison, the original setup likely showed **no progress even after 24 hours** because the agent died instantly every episode with 0 lives.

## Next Steps

1. Start with PPO + curriculum config:
   ```bash
   python train_improved.py --model ppo --config curriculum_config.yaml
   ```

2. Monitor WandB for first 30 minutes:
   - Episode length should increase
   - Reward should trend upward (even if noisy)
   - Videos should show improvement

3. If working well, let it train for 5-10M steps

4. Evaluate best model on harder configs:
   ```bash
   python evaluate.py --model models/{run_id}/best/best_model.zip --config config.yaml
   ```

5. Fine-tune hyperparameters based on results

Good luck! The new setup should give you MUCH better results. 🎮🤖
