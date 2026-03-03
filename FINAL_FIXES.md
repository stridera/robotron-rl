# Final Training Fixes - Run 3

## Analysis of Run 6nqhjgz2 (400k steps)

### The Problem: Learning to Survive, Not to Score

**Training metrics:**
- Episode lengths: +76.5% (474→836 steps) ✓ Learned survival
- Episode scores: -6.1% (12,258→10,906) ✗ Scores declined!
- Episode rewards: Flat (109.9→104.1) ✗ No improvement

**Evaluation metrics (deterministic policy):**
- Scores wildly inconsistent: 180 to 4200
- No learning trend over 400k steps
- Episode lengths barely changed (+3.6%)

### Root Cause: Death Penalty Too Strong

The death penalty (-10) was ~10% of total episode reward (~110), making the agent optimize for "don't die" instead of "score points by killing enemies".

**Typical episode reward breakdown:**
- Kill reward: 11,000 / 100 = 110 points
- Death penalty: -10 points (9% of total!)
- Agent learns: "Survive = avoid -10 penalty" (risk-averse)

**What we want:**
- Agent learns: "Kill enemies = +reward now"
- Death = loss of future kill rewards (through gamma=0.99 discount)
- No explicit death penalty needed!

## Changes Applied

### 1. **REMOVED Death Penalty** (train_improved.py:73-77)

**Before:**
```python
reward = score_delta / 100.0
if terminated:
    reward -= self.death_penalty  # -10 penalty
```

**After:**
```python
reward = score_delta / 100.0
# NO death penalty - let PPO learn survival naturally through gamma discount
# Death means losing future rewards, which is penalty enough
```

**Why this works:**
- PPO with gamma=0.99 naturally values survival through discounted future rewards
- Killing enemy = +1 reward immediately
- Dying = lose ability to earn future +1 rewards
- Agent learns: "Kill more = survive longer = more total reward"

### 2. **Checkpoints Every 100k** (was 500k)

```python
CheckpointCallback(
    save_freq=100_000 // num_envs,  # Was 500_000
    ...
)
```

**Benefit:** Can evaluate earlier to catch problems faster

### 3. **Videos Every 100k by Default** (was disabled)

```python
video_freq: int = 100_000,  # Was 0 (disabled)
```

**Benefit:** Visual confirmation of what agent is learning

## Expected Outcomes

### What Should Happen Now:

**0-100k steps:**
- Random exploration, low scores
- Agent discovers that shooting enemies = reward
- Scores should start climbing

**100-300k steps:**
- Agent learns to aim and shoot effectively
- Learns to avoid enemies while shooting
- Scores should increase steadily

**300-500k steps:**
- More sophisticated strategies emerge
- Better positioning, target prioritization
- Scores continue improving

**Key indicators of success:**
- Scores INCREASING over time (not flat or declining)
- Evaluation scores showing consistent upward trend
- Agent taking risks to kill enemies (not just surviving)

**Key indicators of failure:**
- Scores flat or declining after 200k steps
- Agent still running away without shooting
- Evaluation scores wildly inconsistent

## Command to Restart Training

```bash
# Kill current run
pkill -f train_improved

# Start new run with fixes
poetry run python train_improved.py \
  --model ppo \
  --config curriculum_config.yaml \
  --num-envs 8 \
  --device cuda:0
```

Videos will be saved to `wandb/{run_id}/media/videos/` and logged to TensorBoard.

## Monitoring

### TensorBoard:
```bash
tensorboard --logdir runs/
```

Watch these metrics:
- `robotron/episode_score` - Should trend UPWARD
- `eval_raw/mean_score` - Deterministic evaluation scores
- `rollout/ep_len_mean` - Episode length (survival)
- `train/value_loss` - Should stay stable or decrease
- Videos tab - Visual confirmation of behavior

### What Good Videos Look Like:
- Agent moving toward enemies
- Actively shooting
- Getting kills (score increasing)
- Clearing levels

### What Bad Videos Look Like:
- Agent running away from enemies
- Not shooting or shooting randomly
- Dying quickly without fighting
- Just surviving without scoring

## Fallback Plan

If after 200k steps scores are still not improving:

1. **Increase reward scaling** - Change `/100` to `/50` (make kills more rewarding)
2. **Increase entropy** - Change `ent_coef: 0.02 → 0.05` (more exploration)
3. **Simpler environment** - Temporarily use even simpler config (1 grunt only)
4. **Verify frame skip** - Check if 4 frames is too much (try 2)

## Theory

PPO should naturally learn:
- V(alive with N enemies) = immediate_kill_reward + gamma * V(alive with N-1 enemies)
- V(dead) = 0

Therefore:
- Killing enemy increases value by: kill_reward + gamma * (future_value_alive - 0)
- Agent learns killing is good because it leads to being alive to kill more
- No explicit death penalty needed!

The death penalty was INTERFERING by making the agent value survival over scoring.
