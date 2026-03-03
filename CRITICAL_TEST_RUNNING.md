# CRITICAL TEST IN PROGRESS: Position-Based RL

## What Just Happened

### Pixel-Based PPO Failed Completely (Run dj1fct3y)

After **700,000 steps** with:
- Just 1 grunt (easiest possible scenario)
- 10x stronger rewards (grunt kill = 10.0 reward)
- Ultra-simple curriculum
- High entropy (0.05) for exploration

**Result**: No improvement in scores. Policy still random/collapsed.

**This is definitive**: PPO cannot learn Robotron from raw pixels, even in the absolute simplest scenario.

---

## The Critical Test (Run ot9m8v4i - NOW RUNNING)

### What's Different

Training PPO with **perfect ground truth positions** instead of pixels:

| Metric | Pixel-Based (Failed) | Position-Based (Testing) |
|--------|---------------------|--------------------------|
| **Observation** | 84x84x4 pixels = 28,224 dims | Sprite positions = 222 dims |
| **Policy** | CNN (deep network) | MLP (2x256 layers) |
| **Vision** | Must learn from scratch | Perfect position info |
| **Learning** | Failed after 700k steps | Testing now... |

### What This Tests

**If position-based PPO works (scores improve within 200k steps):**
- ✅ Vision was the bottleneck!
- ✅ PPO CAN solve the control problem
- ✅ Next step: Build sprite detector (Phase 2)
- ✅ Path forward: Train detector, deploy to Xbox

**If position-based PPO fails (no improvement after 500k steps):**
- ❌ PPO fundamentally can't solve this task
- ❌ Not a vision problem, it's an algorithm problem
- ❌ Next step: Try different algorithm (Rainbow DQN, DreamerV3)
- ❌ Or accept pure RL may not work

---

## Monitoring Progress

### Run Details
- **Run ID**: `ot9m8v4i`
- **Config**: ultra_simple_curriculum.yaml (1 grunt)
- **Observations**: 222-dim position vectors
- **Policy**: MlpPolicy (not CNN!)
- **Total steps**: 1,000,000

### Check Progress

```bash
# Monitor training logs
tail -f position_training.log

# Check TensorBoard
tensorboard --logdir runs/ot9m8v4i

# Watch for key metrics:
# - robotron/episode_score (should trend upward)
# - robotron/episode_kills (should be > 0)
```

### Evaluation Checkpoints

```bash
# At 100k steps (early indicator)
poetry run python check_dense_rewards.py models/ot9m8v4i/checkpoints/ppo_positions_checkpoint_100000_steps.zip

# At 200k steps (should be learning)
poetry run python check_dense_rewards.py models/ot9m8v4i/checkpoints/ppo_positions_checkpoint_200000_steps.zip

# At 300k steps (should be good)
poetry run python check_dense_rewards.py models/ot9m8v4i/checkpoints/ppo_positions_checkpoint_300000_steps.zip
```

---

## Expected Timeline

### Optimistic Case (Position-based works)

**0-50k steps**: Random exploration
- Scores: 0-50 (random)
- Learning curve: Flat

**50k-150k steps**: Discovery phase
- Agent learns shooting in direction of enemy helps
- First kills happen
- Scores: 50-100+
- Learning curve: Starting to rise

**150k-300k steps**: Mastery
- Consistent kills of 1 grunt
- Scores: 100+ regularly
- Learning curve: Steep rise, then plateau

**300k+ steps**: Ready for curriculum
- Mastered 1 grunt
- Can add 2nd grunt, then 3rd
- Continue scaling difficulty

### Pessimistic Case (Position-based fails)

**0-500k steps**: No learning
- Scores remain random
- No improvement in kill rate
- Policy collapsed or random

**Conclusion**: PPO can't solve this, need different approach

---

## What We've Learned So Far

### Failed Approaches (All with Pixels)

1. **Death penalty removal** (Run fjmk0ac9, 900k steps)
   - Policy collapsed to constant [7, 6]
   - Rewards too sparse (7% non-zero)

2. **Weak dense rewards** (Run cddtkl23, 100k steps)
   - Policy still collapsed
   - Dense signals drowned out by sparse kills

3. **Strong dense rewards** (Run 3odwp1jj, 1.7M steps)
   - Reward hacking: agent paced back/forth for distance bonuses
   - Optimized shaped rewards instead of kills

4. **Simple strong rewards** (Run dj1fct3y, 700k steps)
   - Just 1 grunt, 10x rewards
   - Still no learning
   - **Final confirmation pixel-based PPO doesn't work**

### Key Insights

**The vision problem is massive:**
- 28,224-dim observation space
- CNN must learn "what is a grunt?" from sparse rewards
- Kills only happen 7% of steps
- Signal-to-noise ratio too low

**Reward shaping is dangerous:**
- Auxiliary rewards often become the objective
- Agent optimizes shaped rewards, not actual goal
- Simpler is better

**PPO has limitations:**
- Great for continuous control with dense rewards
- Struggles with sparse rewards and high-dim observations
- May not be suitable for pixel-based Robotron

---

## Next Steps (Based on Results)

### If Position-Based Succeeds

**Phase 2: Build Sprite Detector**

1. **Collect dataset** (1-2 hours)
   - Run random policy for 1000 episodes
   - Save (frame, sprite_data) pairs
   - ~500k training examples

2. **Train detector** (2-3 hours)
   - Grid-based CNN detector
   - Input: 84x84x4 pixels
   - Output: sprite positions
   - Target: >90% accuracy

3. **Add temporal tracking** (1-2 hours)
   - Track sprites frame-to-frame
   - Handle occlusions with prediction
   - Smooth detections

4. **Test detector + RL** (2-3 hours)
   - Replace ground truth with detector
   - Train PPO on detected positions
   - Should perform similarly to ground truth

5. **Deploy to Xbox** (future)
   - Collect Xbox gameplay footage
   - Fine-tune detector on Xbox frames
   - Deploy full pipeline

### If Position-Based Fails

**Try Different Algorithms**

1. **Rainbow DQN** (1-2 days)
   - Off-policy learning (better for sparse rewards)
   - Prioritized replay (focus on rare kills)
   - Available in SB3-Contrib

2. **DreamerV3** (3-5 days)
   - State-of-the-art for visual RL
   - World model + planning
   - Best sample efficiency
   - Complex implementation (TensorFlow)

3. **Hybrid Approach** (2-3 days)
   - Handcrafted rules for basics
   - RL for refinements
   - Guaranteed baseline performance

4. **Accept Limitations**
   - Pure RL may not be suitable
   - Consider alternative approaches
   - Focus on detector + rules-based system

---

## Current Status

- ✅ Position wrapper implemented
- ✅ Training script created
- 🔄 **Position-based training RUNNING** (run ot9m8v4i)
- ⏳ Will check at 100k, 200k, 300k steps
- ⏳ Expected results in 2-6 hours

**This is the decisive test.** In a few hours, we'll know definitively whether:
1. Vision is the bottleneck (build detector)
2. PPO can't solve this (try different algorithm)

---

## Files Created

1. `position_wrapper.py` - Ground truth position extraction
2. `train_positions.py` - Position-based training script
3. `POSITION_BASED_RL.md` - Full documentation
4. `CRITICAL_TEST_RUNNING.md` - This file

**Log file**: `position_training.log`
**Run ID**: `ot9m8v4i`
**Monitor**: `tensorboard --logdir runs/ot9m8v4i`
