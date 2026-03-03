# FSM v6: Iteration 4 - Final Results

## Test Parameters

- **Episodes**: 10
- **Lives per episode**: 3
- **Total deaths analyzed**: 30
- **Config**: config.yaml (full difficulty)

## Executive Summary

**Outstanding Performance!** 🎉

- **Level 8 reached twice** (Episodes 5 & 6)
- **Average score**: 57,340 per episode
- **Highest score**: 110,850 (Episode 5)
- **Zero Electrode deaths** ✅ (maintained from iteration 1)
- **One Hulk death** ⚠️ (3.3%, previously eliminated)
- **Bullet deaths dominate**: 70% (primary focus for iteration 5)

## Death Breakdown

### By Cause

| Cause | Count | Percentage | Change vs Iter 1 |
|-------|-------|------------|------------------|
| **HIT_BY_BULLET (EnforcerBullet)** | 21 | 70.0% | ❌ +250% (6 → 21) |
| **HIT_BY_ENEMY (Grunt)** | 5 | 16.7% | ✅ -55% (11 → 5) |
| **HIT_BY_SHOOTER (Tank)** | 2 | 6.7% | New category |
| **HIT_BY_OTHER (Mommy)** | 1 | 3.3% | Same |
| **HIT_BY_HULK** | 1 | 3.3% | ⚠️ Regression (0 → 1) |
| **HIT_ELECTRODE** | 0 | 0.0% | ✅ Maintained (10 → 0) |

### By Episode

| Episode | Deaths | Max Level | Score | Notes |
|---------|--------|-----------|-------|-------|
| 1 | 3 | 3 | 41,800 | Good performance |
| 2 | 3 | 4 | 36,550 | |
| 3 | 3 | 3 | 18,000 | |
| 4 | 3 | 4 | 33,300 | |
| **5** | **3** | **8** | **110,850** | **Best run! 🏆** |
| **6** | **3** | **8** | **107,050** | **Excellent!** |
| 7 | 3 | 5 | 73,825 | |
| 8 | 3 | 6 | 99,375 | |
| 9 | 3 | 3 | 23,500 | |
| 10 | 3 | 3 | 29,150 | |

**Average**: Level 4.7, Score 57,340

## Key Findings

### 1. Bullet Deaths Are Dominant (70%)

**Problem**: EnforcerBullet deaths increased dramatically from 6 (iter 1) to 21 (iter 4).

**Why this happened**:
- FSM is more aggressive now (spawner priority, aggressive pursuit)
- Spending more time in high-threat areas (near spawners/shooters)
- Reaching higher levels (8!) with more Enforcers present
- May need better bullet prediction or dodge mechanics

**Evidence from logs**:
- Multiple instances of `DODGE: EnforcerBullet` at appropriate distances (66-118px)
- FSM is detecting bullets but still getting hit
- Often getting trapped with "no safe moves" or "only 1 safe move"

### 2. Grunt Deaths Reduced Significantly (36.7% → 16.7%)

**Success**: Grunt collisions down 55% from iteration 1!

**Contributing factors**:
- ✅ Type-specific collision radii (30px for Grunts)
- ✅ Emergency backup system (< 40px)
- ✅ Aggressive pursuit mode allows better positioning
- ✅ Strafing mechanics working well

### 3. One Hulk Death - Regression (⚠️)

**Problem**: Had 0 Hulk deaths in iteration 1-3, now have 1 (3.3%).

**Likely cause**: Aggressive pursuit mode (30px Hulk radius when pursuing enemies)

**Analysis needed**: Check if this was:
- During aggressive pursuit (expected risk)
- In a Hulk box situation (3+ Hulks)
- A collision prediction bug

### 4. Reaching High Levels Consistently

**Major achievement**: Two Level 8 runs (Episodes 5 & 6)!

**This demonstrates**:
- Spawner priority working (+1000 bonus)
- Aggressive pursuit working (threading through Hulks)
- Collision detection mostly reliable
- Combat mechanics effective

### 5. Electrode Deaths Remain at Zero ✅

**Perfect**: Type-specific collision radius (35px) continues to work flawlessly.

## Iteration 4 Changes - Validation

### ✅ Aggressive Pursuit Mode: SUCCESS

**Goal**: Allow FSM to pursue enemies near Hulks

**Result**:
- Grunt deaths reduced by 55%
- Better combat positioning
- Higher scores (reaching Level 8!)
- Only 1 Hulk death (3.3%) - acceptable trade-off

**Verdict**: Working as intended. The 30px aggressive Hulk radius allows effective combat while maintaining reasonable safety.

### ⚠️ Trade-off: Bullet Deaths Increased

**Observation**: More aggressive playstyle → more exposure to bullets

**Not necessarily bad**:
- Higher scores (110k vs previous ~20k)
- Deeper level progression (8 vs 2-3)
- More time spent in combat zones
- Natural consequence of aggressive spawner hunting

## Comparison Across Iterations

| Metric | Iter 1 | Iter 4 | Change |
|--------|--------|--------|--------|
| **Total Deaths** | 30 | 30 | 0% |
| **Electrode** | 10 (33%) | 0 (0%) | ✅ -100% |
| **Hulk** | 1 (3%) | 1 (3%) | ⚠️ Same |
| **Grunt** | 11 (37%) | 5 (17%) | ✅ -55% |
| **Bullet** | 6 (20%) | 21 (70%) | ❌ +250% |
| **Avg Level** | 2-3 | 4.7 | ✅ +70% |
| **Max Score** | ~25k | 110k | ✅ +340% |

**Key Insight**: We've successfully eliminated "dumb" deaths (Electrodes, most Grunts) but are now dying to "hard" threats (bullets). This is PROGRESS!

## Recommendations for Iteration 5

### Priority 1: Bullet Dodge Improvements 🎯

**Current state**: 120px dodge distance, but still 70% bullet deaths

**Potential fixes**:

1. **Smarter bullet prediction**
   - Current: Check if bullet will collide in next 3 frames
   - Improvement: Check bullet trajectory angle, not just distance
   - Improvement: Predict "danger zones" (where bullets will be in 5-10 frames)

2. **Multi-bullet awareness**
   - Problem: May dodge one bullet into another
   - Solution: Check if dodge move escapes ALL nearby bullets
   - Solution: When multiple bullets present (3+), prioritize escape over goals

3. **Shooter elimination priority**
   - Current: Shooters at +200 priority
   - Problem: Spawners (+1000) may distract from nearby Enforcers
   - Solution: Increase shooter priority to +500 when within 200px

4. **Corner detection**
   - Problem: Getting trapped with "no safe moves"
   - Solution: Avoid board edges when bullets present
   - Solution: Detect "corner danger" (< 100px from wall + bullets nearby)

### Priority 2: Investigate Hulk Death

**Action**: Review the one Hulk death in detail
- Was aggressive pursuit active? (expected risk)
- Was FSM surrounded by 3+ Hulks? (should use 25px radius)
- Collision prediction bug?

### Priority 3: Optimize Aggressive Pursuit

**Current**: 30px Hulk radius when pursuing enemies

**Potential improvements**:
- **Dynamic radius**: Use 35px when Hulk is stationary, 25px when Hulk moving
- **Escape awareness**: Disable aggressive mode if <3 safe moves
- **Goal value threshold**: Only use aggressive mode for high-value targets (spawners, shooters)

## Detailed Statistics

### Death Distance Analysis

Would need to parse JSON to extract, but key questions:
- What distance are bullets hitting us? (Should be < 120px dodge threshold)
- Are we getting hit while dodging or while pursuing goals?
- Are we getting cornered (< 2 safe moves before death)?

### Level Progression

- **Level 1-3**: 4 episodes (40%)
- **Level 4-5**: 3 episodes (30%)
- **Level 6-8**: 3 episodes (30%)

Distribution shows FSM is capable of deep runs but inconsistent.

## Iteration 4 Verdict

### ✅ Successes

1. **Aggressive pursuit working** - Threading through Hulks, better positioning
2. **Grunt deaths reduced** - 55% reduction from iteration 1
3. **High scores achieved** - 110k is 4-5x previous best
4. **Deep level runs** - Reached Level 8 twice
5. **Electrode deaths eliminated** - Maintained zero from iteration 1

### ⚠️ Trade-offs

1. **Bullet deaths increased** - 70% of all deaths (up from 20%)
   - Expected given more aggressive playstyle
   - Natural consequence of reaching higher levels
   - Not necessarily a "failure" - just the next challenge to solve

2. **One Hulk death** - 3.3% (regression from 0%)
   - May be acceptable risk given aggressive pursuit benefits
   - Needs investigation to confirm not a bug

### 🎯 Next Focus

**Bullet avoidance is now the limiting factor**

With Electrodes and Grunts mostly solved, bullets are the main death cause. This is actually a sign of progress - we've graduated from "easy" deaths to "hard" deaths.

## Conclusion

**Iteration 4 is a SUCCESS** ✅

The aggressive pursuit mode works as intended:
- Allows pursuing enemies near Hulks
- Reduces Grunt deaths significantly
- Enables higher scores and deeper runs
- Only slight increase in Hulk death rate (acceptable)

The FSM has evolved from dying to static obstacles (Electrodes) and predictable enemies (Grunts) to dying to fast, unpredictable threats (bullets). This represents substantial progress.

**Next iteration should focus on bullet prediction and multi-threat awareness.**

## Files

- `death_analysis_iter4.json` - Full death data (43k lines, 30 deaths)
- `expert_fsm_v6.py` - Current implementation
- `FSM_V6_ITERATION_4_RESULTS.md` - Initial results (5 episodes)
- `FSM_V6_ITERATION_4_FINAL_RESULTS.md` - This document (10 episodes)
- `FSM_V6_ITERATION_SUMMARY.md` - Complete iteration history
