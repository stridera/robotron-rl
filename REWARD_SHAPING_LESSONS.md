# Reward Shaping Lessons Learned

## Failed Approaches

### 1. Death Penalty Removal (Run fjmk0ac9)
**Problem**: Policy collapsed to constant actions [7, 6] (left + down-left) even after 900k steps.

**Diagnosis**:
- Rewards too sparse: only 7% of steps had non-zero rewards
- High policy entropy (4.06) but deterministic policy collapsed
- Agent couldn't distinguish between actions due to flat value landscape

### 2. Weak Dense Rewards (Run cddtkl23)
**Problem**: Policy still collapsed to constant actions after 100k steps.

**Why it failed**:
- Distance rewards: 0.001/pixel = ~0.01 per step
- Wall penalties: 0.01 per step
- Score rewards: 2.0 per kill
- Dense signals were **200x weaker** than kills → got drowned out

### 3. Strong Dense Rewards (Run 3odwp1jj)
**Problem**: Agent learned to pace back and forth in center, not killing enemies.

**Why it failed - REWARD HACKING**:
- Distance rewards: 0.1/pixel × 100 pixels = ±10 reward per step
- Agent discovered: "I can get +10/step by moving near enemies without killing them!"
- Optimized the shaped rewards instead of the actual objective
- Classic reward shaping failure

## Core Issue

**PPO with visual inputs is struggling with sparse rewards.** After multiple runs totaling ~3M steps:
- Sparse rewards → policy collapse
- Weak dense rewards → policy collapse (drowned out)
- Strong dense rewards → reward hacking

The fundamental problem: killing enemies from visual inputs is HARD, and reward signal is too weak.

## New Approach: Simple + Strong

Created `train_simple_strong.py`:

### Strategy
1. **NO reward shaping** - no auxiliary rewards, no tricks
2. **ONLY score rewards** - but scaled VERY aggressively: `score/10` instead of `score/100`
3. **Much stronger signal**: Grunt kill = 10.0 reward (was 1.0)
4. **High entropy**: ent_coef=0.05 (was 0.01) for more exploration
5. **Ultra-simple curriculum**: Start with literally 1 grunt

### Theory
The agent needs an absolutely massive primary reward signal to learn from pixels. Kill rewards need to be so strong that:
- Value function can clearly distinguish good from bad actions
- Even with sparse rewards, the signal is strong enough to shape behavior
- Agent is incentivized to take risks to get those big rewards

### Expected Outcomes

**If this works:**
- Scores should start increasing after 100k-200k steps
- Agent should learn to approach and shoot the single grunt
- Deterministic policy should show action diversity (not collapsed)

**If this still fails:**
- PPO might not be the right algorithm for this task
- Might need imitation learning (bootstrap from FSM player)
- Might need curiosity-driven learning (intrinsic motivation)
- Might need to simplify observation space (not just pixels)

## Running the New Approach

```bash
# Kill current training
pkill -f train

# Start simple+strong training with ultra-simple curriculum
poetry run python train_simple_strong.py \
  --config ultra_simple_curriculum.yaml \
  --num-envs 8 \
  --device cuda:0
```

**Check progress at 100k steps:**
```bash
# Find the run ID
ls -t runs/ | head -1

# Check policy behavior
poetry run python check_dense_rewards.py models/{run_id}/checkpoints/ppo_checkpoint_100000_steps.zip
```

**What to look for:**
- Action diversity > 2 for both movement and shooting
- Episode scores > 100 (agent killing the 1 grunt)
- Evaluation scores trending upward
- Video shows agent moving toward and shooting grunt

## If This Fails Too

Then we need to consider:

1. **Different algorithm**: Try SAC, DreamerV3, or other algorithms designed for visual inputs
2. **Imitation learning**: Use FSM player to bootstrap, then fine-tune with RL
3. **Curiosity-driven**: Add intrinsic motivation (ICM, RND, NGU)
4. **Simpler observations**: Add hand-crafted features (enemy positions, distances) alongside pixels
5. **Different architecture**: Try transformer-based policies or world models

## Key Lesson

**Reward shaping is dangerous.** It's very easy to create auxiliary rewards that the agent optimizes instead of the actual objective. When in doubt, use simpler, stronger primary rewards rather than complex shaped rewards.
