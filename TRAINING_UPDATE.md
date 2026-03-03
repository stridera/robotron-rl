# Training Update - Stability Fixes

## Run xz08295c Analysis (400k steps)

### Problem Identified: Catastrophic Forgetting

The training showed unstable learning with clear catastrophic forgetting:

**Quartile Analysis:**
- Q1 (0-100k):   Score = 11,532
- Q2 (100-200k): Score = 11,402 (-1.1%)
- Q3 (200-300k): Score = 13,291 (+16.6%) ✓ **Learning happened!**
- Q4 (300-400k): Score = 11,293 (-15.0%) ❌ **Then forgot it!**

**Recent Trend:**
- Previous 50 episodes: 12,736
- Recent 50 episodes:  11,073
- Change: -13.1% (getting worse)

### Root Causes

**1. Massive Reward Variance (22x variation)**
- Scores ranged from 1,900 to 42,850
- Kill reward: 38 to 856 points per episode
- High variance confuses value network (value loss was exploding)
- Makes learning unstable

**2. Learning Rate Too High (2.5e-4)**
- Combined with high reward variance causes instability
- Agent learns something in Q3, then overwrites it in Q4
- Classic catastrophic forgetting pattern

**3. Survival Bonus Too Weak (0.1)**
- Only ~90 points per episode vs 40-800 from kills
- Movement head getting minimal consistent signal
- Kill rewards dominate (>90% of total reward)

## Changes Applied

### 1. Reduced Reward Scaling (50 → 100)
```python
# Before: reward = score_delta / 50.0
# After:  reward = score_delta / 100.0
```

**Effect:**
- Reduces reward variance by 2x
- 1 grunt kill: 2.0 → 1.0 reward
- Level clear (30k): 600 → 300 reward
- Death penalty stays at -10

**Why this helps:**
- Smaller reward magnitudes = more stable value estimates
- Value network can learn more consistent Q-values
- PPO policy updates are less volatile

### 2. Reduced Learning Rate (2.5e-4 → 1.0e-4)
```python
'learning_rate': 1.0e-4  # Was 2.5e-4
```

**Effect:**
- 2.5x slower policy updates
- More gradual learning = less catastrophic forgetting
- Takes longer to converge BUT converges more stably

**Why this helps:**
- High LR + high variance = unstable learning
- Lower LR allows agent to smooth out variance
- Prevents overwriting previously learned behavior

### 3. Fixed typo (reI ward → reward)
Minor bug fix in evaluation callback.

## Expected Outcomes

### Short Term (0-200k steps)
- **Slower initial learning** - Lower LR means more gradual improvements
- **More stable metrics** - Less wild swings in scores
- **Consistent value loss** - Should stay relatively flat, not exploding
- **Gradual entropy decrease** - Steady exploration decay

### Medium Term (200-500k steps)
- **No Q3→Q4 crash** - Should maintain improvements instead of forgetting
- **Linear or superlinear improvement** - Scores should trend upward consistently
- **Best model improving** - Evaluation scores should beat previous best

### Long Term (500k-1M+ steps)
- **Continued improvement** - Should keep learning without plateau
- **Higher peak performance** - More stable learning = better final performance
- **Transferable knowledge** - Can fine-tune on harder configs

## How to Monitor

Check TensorBoard for these signals:

**Good signs (learning is stable):**
- `robotron/episode_score` trending upward over 100k+ step windows
- `train/value_loss` staying relatively constant or slowly decreasing
- `rollout/ep_rew_mean` increasing steadily
- `rollout/ep_len_mean` increasing (agent surviving longer)
- `eval_raw/mean_score` beating previous bests

**Warning signs (still unstable):**
- Scores spike then crash (like Q3→Q4 in xz08295c)
- Value loss increasing or oscillating wildly
- Entropy dropping too fast (<50% in first 100k steps)
- Recent scores worse than earlier scores

## Next Run Command

```bash
poetry run python train_improved.py \
  --model ppo \
  --config curriculum_config.yaml \
  --num-envs 8 \
  --device cuda:0
```

## Fallback Plans

**If still unstable after 400k steps:**
1. Further reduce LR: 1.0e-4 → 5.0e-5
2. Increase value function coefficient: 0.5 → 1.0 (help value network learn faster)
3. Add reward clipping: clip(reward, -10, 10)
4. Try simpler baseline: Just `reward = score_delta / 100` (no death penalty)

**If learning too slow:**
1. Increase entropy coefficient: 0.02 → 0.05 (more exploration)
2. Decrease n_steps: 64 → 32 (faster updates, less stable)
3. Verify curriculum_config.yaml is actually being loaded (check early game logs)
