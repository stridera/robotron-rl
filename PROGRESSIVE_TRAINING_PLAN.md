# Progressive Curriculum Training Plan

## Overview

Testing if position-based RL scales from simple (1 grunt) to full game complexity (85 sprites).

**Phase 1 Result:** ✅ Position-based RL solved 1 grunt in 100k steps (48 kills/episode at 1M)

**Phase 1.5 Goal:** Prove position-based RL scales to realistic game scenarios with 10-50 sprites.

---

## Training Configuration

### Environment Setup
- **Config:** `progressive_curriculum.yaml`
- **Observation:** Position vectors (442 dims)
  - Player position: 2 dims
  - 20 nearest sprites × 22 features each: 440 dims
  - Features per sprite: type (17), rel_pos (2), dist (1), angle (1), valid (1)
- **Max sprites:** 20 (handles levels with up to 20 simultaneous sprites)
- **Reward:** `score_delta / 10.0` (simple, strong signal)

### Model Architecture
- **Policy:** MlpPolicy (no CNN needed!)
- **Network:** 2-layer MLP, 512 units each (larger than Phase 1's 256)
- **Parameters:**
  - Learning rate: 3e-4
  - Batch size: 64
  - n_steps: 2048
  - Entropy: 0.01
  - Parallel envs: 8

### Training Plan
- **Total steps:** 3,000,000
- **Estimated time:** ~12-18 hours on GPU
- **Checkpoints:** Every 100k steps
- **VecNormalize:** Saved every 100k steps

---

## Curriculum Stages

### Stage 1: Levels 1-3 (Sprites: 2-4)
**Goal:** Warm-up with proven scenario

- Level 1: 1 grunt
- Level 2: 2 grunts
- Level 3: 3 grunts

**Expected:** Agent already masters this from Phase 1

### Stage 2: Levels 4-6 (Sprites: 6-11)
**Goal:** Learn to navigate obstacles and collect family

- Level 4: 3 grunts + 1 electrode + 1 family
- Level 5: 5 grunts + 2 electrodes + 1 family
- Level 6: 5 grunts + 3 electrodes + 1 family + 1 daddy

**New challenges:**
- Obstacles that must be avoided or destroyed
- Family members that give bonus points
- Multiple sprite types

### Stage 3: Levels 7-9 (Sprites: 12-20)
**Goal:** Learn hulks are invincible

- Level 7: 5 grunts + 3 electrodes + 1 hulk + 1 family + 1 daddy
- Level 8: 8 grunts + 4 electrodes + 1 hulk + 2 family + 1 mikey + 1 daddy
- Level 9: 8 grunts + 5 electrodes + 2 hulks + 2 family + 1 mikey + 1 daddy

**New challenges:**
- Hulks can't be killed, must avoid
- Higher sprite density (approaching max_sprites=20)

### Stage 4: Levels 10-12 (Sprites: 22-33) ⚠️
**Goal:** Handle realistic grunt counts (10-15)

- Level 10: 10 grunts + 5 electrodes + 2 hulks + family
- Level 11: 12 grunts + 6 electrodes + 2 hulks + family
- Level 12: 15 grunts + 8 electrodes + 3 hulks + family

**Critical test:**
- Sprite count exceeds max_sprites=20
- Agent only sees 20 nearest sprites (sorted by distance)
- Must still perform well with incomplete information

### Stage 5: Levels 13-15 (Sprites: 34-41) ⚠️
**Goal:** Handle spawners (dynamic enemy counts)

- Level 13: Add 1 brain (spawns cruise missiles)
- Level 14: Add 1 sphereoid (spawns enforcers)
- Level 15: Multiple spawners

**New challenges:**
- Enemies spawn dynamically during level
- Need to prioritize killing spawners
- Sprite count changes over time

### Stage 6: Levels 16-18 (Sprites: 42-52) ⚠️⚠️
**Goal:** Handle quarks (spawn tanks that shoot)

- Level 16: Add 1 quark (spawns tanks)
- Level 17: More quarks and spawners
- Level 18: Dense spawner scenario

**New challenges:**
- Tanks shoot projectiles
- Must dodge or destroy projectiles
- High sprite density (50+ at peak)

### Stage 7: Levels 19-21 (Sprites: 62-85) ⚠️⚠️⚠️
**Goal:** Master full game difficulty

- Level 19: 25 grunts + full enemy roster
- Level 20: 30 grunts + many spawners
- Level 21: 35 grunts + maximum difficulty

**Final test:**
- 85 sprites at spawn (agent only sees nearest 20)
- All enemy types present
- Realistic full-game scenario

---

## Success Criteria

### Minimum Success (Good Enough for Phase 2)
- ✅ Master Stages 1-3 (1-9 sprites): 100% success
- ✅ Master Stage 4 (10-15 grunts): 10+ kills consistently
- ✅ Survive Stage 5-6 (spawners): 20+ kills on average
- ✅ Maintain action diversity (use all 8 movement/shooting actions)

**Outcome:** Position-based RL scales to realistic scenarios → Build detector with confidence

### Ideal Success (Phase 2 Will Be Easy)
- ✅ Master Stages 1-4: Near-perfect performance
- ✅ Master Stages 5-6: 30+ kills consistently
- ✅ Compete on Stage 7: 40+ kills on full difficulty
- ✅ Action diversity maintained throughout

**Outcome:** Position-based RL is robust → Detector only needs >80% accuracy

### Partial Success (Still Valuable)
- ✅ Master Stages 1-3
- ⚠️ Struggles with Stage 4 (can't handle 10+ grunts)
- ❌ Fails on Stages 5-7

**Outcome:** Need to improve position-based approach before building detector:
- Try larger network (1024x1024)
- Try different feature engineering (velocities, sprite history)
- Try longer training (5M+ steps)

---

## Evaluation Checkpoints

### 500k steps (Expected: Master Stages 1-3)
```bash
poetry run python check_position_model.py models/{run_id}/checkpoints/ppo_progressive_checkpoint_500000_steps.zip
```

**Look for:**
- Kills on Level 1-3: Should be 100% success
- Kills on Level 4-6: Starting to learn
- Action diversity: All 8 actions used

### 1M steps (Expected: Master Stage 4)
```bash
poetry run python check_position_model.py models/{run_id}/checkpoints/ppo_progressive_checkpoint_1000000_steps.zip
```

**Look for:**
- Kills on Level 10-12: 10+ kills consistently
- Navigates obstacles effectively
- Avoids hulks

### 2M steps (Expected: Master Stages 5-6)
```bash
poetry run python check_position_model.py models/{run_id}/checkpoints/ppo_progressive_checkpoint_2000000_steps.zip
```

**Look for:**
- Kills on Level 13-18: 20-30 kills
- Prioritizes spawners
- Dodges projectiles

### 3M steps (Final: Should handle Stage 7)
```bash
poetry run python check_position_model.py models/{run_id}/checkpoints/ppo_progressive_checkpoint_3000000_steps.zip
```

**Look for:**
- Kills on Level 19-21: 30+ kills (target)
- Stable performance on full difficulty
- No policy collapse

---

## Key Risks and Mitigations

### Risk 1: Truncation at max_sprites=20 breaks learning
**Problem:** Agent only sees nearest 20 sprites, may miss important distant enemies

**Mitigation:**
- Sprites sorted by distance (nearest first)
- Most dangerous enemies (shooting, moving fast) are usually near player
- If fails, increase max_sprites to 30 or 40

**Evidence:** Will show in Stage 4+ (levels 10+)

### Risk 2: Network capacity insufficient for complex scenarios
**Problem:** 512x512 MLP can't learn effective policy for 40+ sprites

**Mitigation:**
- If fails, increase to 1024x1024 or 3-layer network
- Phase 1 used 256x256 for 10 sprites; 512x512 should handle 20

**Evidence:** Will show as plateauing performance after Stage 4

### Risk 3: 3M steps not enough for full curriculum
**Problem:** Agent doesn't reach Stage 7 or plateaus early

**Mitigation:**
- Continue training to 5M steps
- Adjust curriculum (more gradual progression)
- May need 10M+ steps for full mastery (still much faster than pixel-based)

**Evidence:** Check highest_level metric - should reach 19+ by 3M steps

### Risk 4: Dynamic spawning confuses agent
**Problem:** Sprite count changes mid-level, agent's policy breaks

**Mitigation:**
- Valid flags in observation help agent handle variable sprite counts
- VecNormalize should adapt to changing input distributions
- If fails, may need recurrent policy (LSTM)

**Evidence:** Will show in Stage 5+ (spawners)

---

## Decision Tree After Training

```
Training Complete (3M steps)
│
├─ Success: 30+ kills on Level 19-21
│  └─ ✅ Position-based RL scales!
│     └─ Proceed to Phase 2 (sprite detector)
│        - Target: Detector must match 90%+ of position-based performance
│        - Collect 500k training frames
│        - Train grid-based CNN detector
│        - Test detector + RL pipeline
│
├─ Partial: 20-29 kills on Level 19-21
│  └─ ⚠️ Position-based RL mostly scales
│     └─ Options:
│        1. Continue training to 5M steps
│        2. Increase network size (1024x1024)
│        3. Proceed to Phase 2 with lower target (80% detector accuracy)
│
├─ Minimal: <20 kills on Level 19-21 but >10 on Level 10-12
│  └─ ⚠️ Position-based RL scales to 10 grunts but not full game
│     └─ Options:
│        1. Adjust curriculum (slower progression)
│        2. Try different architecture (LSTM for temporal)
│        3. Improve feature engineering (add velocities)
│        4. Proceed to Phase 2 but target simpler scenarios (10 grunts max)
│
└─ Failure: <10 kills on Level 10-12
   └─ ❌ Position-based RL doesn't scale
      └─ Options:
         1. Debug: Check for policy collapse, reward hacking
         2. Try different algorithm (DQN, SAC)
         3. Reconsider approach (maybe hybrid: rules + RL refinement)
```

---

## Command to Start Training

```bash
# Full 3M step training (~12-18 hours)
poetry run python train_progressive.py

# With custom settings
poetry run python train_progressive.py \
    --config progressive_curriculum.yaml \
    --device cuda:0 \
    --num-envs 8 \
    --timesteps 3000000 \
    --max-sprites 20
```

---

## Monitoring During Training

### WandB Metrics to Watch

1. **robotron/episode_score** - Should steadily increase
2. **robotron/episode_kills** - Target: 30+ by 3M steps
3. **robotron/episode_level** - Should reach Level 19+ by 3M steps
4. **robotron/highest_score** - Should show consistent improvement

### Signs of Success
- Episode scores increasing over time
- Highest level reached increases (1 → 5 → 10 → 15 → 19+)
- No sudden drops in performance (no catastrophic forgetting)
- Action diversity maintained (check with evaluation script)

### Signs of Trouble
- Scores plateau early (<1M steps)
- Highest level stuck below 10
- Explained variance drops to near zero
- Evaluation shows policy collapse

---

## Files Created

1. **progressive_curriculum.yaml** - 8-stage curriculum (1 → 85 sprites)
2. **train_progressive.py** - Training script for progressive curriculum
3. **test_progressive_curriculum.py** - Validation script for curriculum
4. **PROGRESSIVE_TRAINING_PLAN.md** - This document

---

## Next Steps After This Training

### If Successful:
**Phase 2: Build Sprite Detector**
1. Collect 500k training frames with ground truth positions
2. Train CNN detector (grid-based or YOLO)
3. Add temporal tracking for occlusions
4. Create DetectorWrapper to replace GroundTruthPositionWrapper
5. Test detector + position-based policy pipeline
6. Target: Match 90% of ground truth performance

**Estimated time:** 10-14 hours

### If Partially Successful:
**Iterate on Position-Based RL**
1. Analyze failure modes (which stages failed?)
2. Adjust curriculum or network architecture
3. Retrain with improvements
4. Once confident, proceed to Phase 2

**Estimated time:** 1-3 days

### If Failed:
**Reassess Approach**
1. Debug thoroughly (policy collapse? reward hacking?)
2. Consider different RL algorithm (DQN, SAC, DreamerV3)
3. Consider hybrid approach (rules + RL refinement)
4. May need to reconsider detector approach

---

**Ready to train!** 🚀

The test run showed promising results (149 kills, level 13 in just 1000 steps).
Let's see if position-based RL can master the full game!
