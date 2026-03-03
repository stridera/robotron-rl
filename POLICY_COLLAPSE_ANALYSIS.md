# Policy Collapse Analysis - Run fjmk0ac9 @ 900k steps

## Problem

Even after removing the death penalty, the agent's deterministic policy collapsed to constant actions:
- **Movement**: ALWAYS action 7 (LEFT) - 100% of the time
- **Shooting**: ALWAYS action 6 (DOWN-LEFT) - 100% of the time
- **Behavior**: Agent runs to upper-left corner and shoots left until it dies

User observation: "It basically does the same thing as before. Now it runs upper-left and shoots to the left. If it doesn't die before the corner, it just runs into the corner until it dies."

## Diagnostic Results

### ✅ What's NOT the problem:

1. **VecNormalize is working correctly**:
   - Observation variance: 4.3 to 249.9 (healthy range)
   - Temporal variation: 0.225 (observations changing properly)
   - No collapsed variance pixels

2. **Observations are informative**:
   - Agent CAN see the game state
   - Observations vary significantly over time
   - No signal destruction

3. **Policy entropy is HIGH (4.06)**:
   - Policy distribution is actually quite flat/diverse
   - Stochastic mode uses all 8 actions for both movement and shooting
   - This is CONFUSING at first...

### 🚨 The ACTUAL problem:

**The reward signal is EXTREMELY SPARSE**:
- Only **7 out of 100 steps** have non-zero rewards (7% reward density)
- Mean reward: 0.065 per step
- Max reward: 1.0 (killing a 100-point enemy)
- With `score_delta / 100.0`, rewards only come from kills

## Why This Causes Policy Collapse

The policy distribution has high entropy (all actions have similar probabilities). But they're ALL nearly equal because the agent rarely gets feedback:

```
Action values might look like:
  Movement: [2.01, 2.00, 1.99, 2.02, 1.98, 2.01, 2.03, 2.00]
  Shooting: [1.99, 2.01, 2.00, 1.98, 2.02, 2.03, 2.00, 2.01]
```

When sampling (stochastic mode), all actions are used. But when taking argmax (deterministic mode), it ALWAYS picks the highest value - action 7 for movement, action 6 for shooting.

Since rewards are so sparse (93% of steps give zero reward), the value estimates barely differ. The agent hasn't learned meaningful Q-values because it doesn't get enough feedback to distinguish good actions from bad.

## Why Removing Death Penalty Didn't Help

Removing the death penalty was correct - it prevented the agent from learning a "hide in corner" strategy. But it wasn't enough because:

1. **Sparse rewards** - Agent gets feedback only when killing enemies
2. **No guidance** - Between kills, agent gets zero information about whether it's doing the right thing
3. **Flat value landscape** - All actions seem equally good (or equally bad) due to lack of feedback

The agent needs **dense feedback** to learn which directions to move and shoot.

## The Solution: Dense Reward Shaping

Created `train_dense_rewards.py` with three types of rewards:

### 1. **Stronger Score Rewards** (main signal)
```python
score_reward = score_delta / 50.0  # Was /100.0
```
- Killing grunt: +2.0 reward (was +1.0)
- 2x stronger signal

### 2. **Distance-Based Rewards** (dense movement signal)
```python
distance_delta = last_min_distance - current_min_distance
distance_reward = distance_delta * 0.001
```
- Moving closer to enemies: +0.001 per pixel
- Moving away from enemies: -0.001 per pixel
- Gives feedback EVERY STEP about movement direction

### 3. **Wall Proximity Penalty** (anti-camping)
```python
if wall_dist < 100:
    normalized_dist = wall_dist / 100.0  # 0 to 1
    wall_penalty = -(1.0 - normalized_dist) * 0.01
```
- At wall (dist=0): -0.01 penalty
- At distance 50: -0.005 penalty
- At distance 100+: ~0 penalty
- Prevents corner-camping behavior

### 4. **Lower Gamma** (better for sparse rewards)
```python
gamma = 0.95  # Was 0.99
```
- Shorter time horizon helps with sparse reward problems
- Agent focuses on immediate kills rather than long-term survival

## Expected Outcomes

With dense rewards, the agent should:

1. **Learn to approach enemies** (distance reward)
2. **Learn to avoid walls** (wall penalty)
3. **Get stronger feedback for kills** (score/50 instead of score/100)
4. **Develop diverse behaviors** (constant feedback prevents value collapse)

### What to watch for:

**Good signs:**
- Deterministic policy shows varied actions (not constant)
- Agent moves toward enemies, not away
- Agent stays in center of play area
- Evaluation scores improving over time

**Bad signs:**
- Still collapses to constant actions after 200k steps
- Agent still corner-camping
- Scores not improving

## How to Train

```bash
# Kill old training
pkill -f train_improved

# Start new training with dense rewards
poetry run python train_dense_rewards.py \
  --model ppo \
  --config curriculum_config.yaml \
  --num-envs 8 \
  --device cuda:0
```

## Monitor Progress

Check diagnostics at 100k steps:
```bash
# After 100k steps, evaluate the checkpoint
poetry run python diagnose_900k.py  # (modify to use 100k checkpoint)
```

Look for:
- Deterministic action diversity > 1 for both movement and shooting
- Higher reward density (>20% non-zero steps)
- Policy engaging enemies instead of fleeing

## Theory

PPO needs dense feedback to learn value functions. With 93% of steps giving zero reward, the value estimates become flat and meaningless. By adding dense rewards:

1. **Every step** now provides feedback (distance + wall penalties)
2. **Value function** can learn meaningful distinctions between actions
3. **Policy** learns which directions are good (toward enemies, away from walls)
4. **Exploration** is guided by shaped rewards, not just random

The shaped rewards act as "training wheels" to help the agent discover that engaging enemies is good and corner-camping is bad.
