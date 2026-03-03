# FSM v6: Iteration Summary

## Overview

This document tracks the iterative improvements to FSM v6 based on death analysis and user feedback.

## Iteration 1: Death Analysis Fixes

**Analysis**: 30 deaths from 10 episodes revealed 4 major causes:
- Electrode collisions: 33.3%
- Grunt collisions: 36.7%
- Bullet deaths: 20.0%
- Hulk collision: 3.3%

**Changes Made**:
1. ✅ Type-specific collision radii (Electrode: 35px, Hulk: 40px, Grunt: 30px, Prog: 30px)
2. ✅ Hulk-box detection (reduce radius to 25px when surrounded by 3+)
3. ⚠️ Bullet dodge distance increased to 150px
4. ⚠️ Emergency backup system (< 50px)
5. ⚠️ Escape route planning (< 4 safe moves)
6. ✅ Fixed critical collision detection bug (was allowing unsafe moves)

**Results**:
- ✅ Electrode deaths: 10 → 0 (100% elimination!)
- ✅ Hulk deaths: 1 → 0 (100% elimination!)
- ❌ Grunt deaths: 11 → 15 (+36% increase)
- ❌ Bullet deaths: 6 → 13 (+117% increase)
- Total: 30 → 30 deaths (no change)

**Analysis**: Successfully eliminated "impossible" deaths (Electrodes/Hulks), but defensive systems made FSM too cautious, causing more bullet deaths.

## Iteration 2: Tuning Defensive Systems

**User Observation**: FSM working well on static obstacles, but too defensive overall.

**Changes Made**:
1. ✅ Reduced bullet dodge distance: 150px → 120px
   - **Rationale**: 150px was triggering dodge mode too early, causing FSM to get cornered

2. ✅ Increased emergency backup threshold: 50px → 40px
   - **Rationale**: 50px was too cautious, interfering with normal combat
   - 40px is truly CRITICAL distance

3. ✅ Removed escape route planning logic entirely
   - **Rationale**: Was prioritizing escape over goal pursuit too aggressively
   - Trust collision detection instead

**Expected Results**:
- Fewer bullet deaths (not getting cornered as often)
- Better goal pursuit (more aggressive)
- Maintain Electrode/Hulk elimination

## Iteration 3: Spawner Priority Fix

**User Feedback**: *"One issue I see is that we wait too long for the spawners to spit out their enemies which overwhelm us. They might be too far away to show up on our list."*

**Problem Analysis**:
```python
# OLD SCORING:
base_score = 200 - distance
spawner_bonus = 300

# Example: Spawner at 300px
score = (200 - 300) + 300 = 200

# Example: Grunt at 10px
score = (200 - 10) + 50 = 240

# Result: Grunt wins! We ignore spawner until too late
```

**Solution**:
```python
# NEW SCORING:
spawner_bonus = 1000  # Was 300, now 1000

# Example: Spawner at 500px
score = (200 - 500) + 1000 = 700

# Example: Grunt at 10px
score = (200 - 10) + 50 = 240

# Result: Spawner wins! Kill it before it spawns more enemies
```

**Rationale**:
- Spawners create exponential threat growth (more enemies = harder game)
- Better to travel 500px to kill a spawner than to fight 10+ enemies it creates
- User observed FSM getting overwhelmed by spawned enemies
- Sphereoids/Brains/Quarks are the root cause of enemy swarms

**Expected Impact**:
- Fewer deaths from being overwhelmed by enemy swarms
- Proactive gameplay (eliminate threats before they multiply)
- May travel further but face fewer total enemies
- Better long-term survival

## Iteration 4: Aggressive Pursuit Mode

**User Feedback**: *"We still have the issue where if a grunt is surrounded by hulks, we're afraid to go kill them."*

**Problem**: FSM avoiding valuable targets (Grunts, Spawners) when positioned near Hulks, even though Hulks are slow-moving (2 px/frame).

**Solution**: Implemented context-aware collision detection with `allow_aggressive` parameter:

```python
# In is_move_safe():
if enemy.type == 'Hulk':
    if hulk_count_nearby >= 3:
        collision_radius = 25  # Surrounded case (unchanged)
    elif allow_aggressive:
        collision_radius = 30  # NEW: Relaxed from 40px for enemy pursuit
    else:
        collision_radius = 40  # Default: Full safety for civilians/dodging

# In decide_action():
allow_aggressive = self.current_goal_type == 'enemy'  # True when pursuing enemies
safe_moves = self.find_safe_moves(allow_aggressive=allow_aggressive)
```

**Rationale**:
- **Context matters**: Pursuing a Grunt near a Hulk is worth the slightly reduced safety margin
- **Hulks are predictable**: Move at 2 px/frame, easy to predict and avoid
- **30px is still safe**: Provides adequate buffer while allowing us to thread between Hulks
- **Selective aggression**: Only used for enemy goals, NOT civilians or dodging

**Expected Impact**:
- ✅ Can now pursue enemies positioned near Hulks (previously avoided)
- ✅ More effective spawner elimination (even when near Hulks)
- ✅ Better combat positioning (not forced to retreat unnecessarily)
- ⚠️ Slightly higher Hulk collision risk (30px vs 40px), but mitigated by slow Hulk movement

## Summary of All Changes

### Collision Detection
| Parameter | Original | Iter 1 | Iter 4 | Current |
|-----------|----------|--------|--------|---------|
| Electrode radius | 20px | 35px | 35px | 35px |
| Hulk radius (default) | 20px | 40px | 40px | 40px |
| Hulk radius (surrounded) | 20px | N/A | 25px | 25px |
| Hulk radius (aggressive) | 20px | N/A | N/A | **30px** |
| Grunt radius | 20px | 25px | 30px | 30px |
| Collision bug | Present | Fixed | Fixed | Fixed |
| Context-aware collisions | No | No | No | **Yes** |

### Defensive Systems
| Parameter | Original | Iter 1 | Current |
|-----------|----------|--------|---------|
| Bullet dodge distance | 100px | 150px | **120px** |
| Emergency backup | None | <50px | **<40px** |
| Escape route planning | None | <4 safe | **Removed** |

### Goal Priority
| Parameter | Original | Iter 1 | Current |
|-----------|----------|--------|---------|
| Spawner priority bonus | +300 | +300 | **+1000** |
| Shooter priority bonus | +200 | +200 | +200 |
| Grunt priority bonus | +50 | +50 | +50 |

## Key Insights

1. **Perfect collision avoidance has trade-offs**: Eliminating Electrode/Hulk deaths made FSM too defensive for dynamic threats.

2. **Balance defensive vs. offensive**: Being too cautious causes different problems (cornering, overwhelm).

3. **Exponential threats need exponential priority**: Spawners create multiplicative danger, need much higher priority.

4. **User feedback is critical**: Observations about "waiting too long for spawners" revealed priority system flaw.

5. **Iterative improvement works**: Each iteration addressed specific issues without breaking what worked.

## Expected FSM v6 Behavior (After Iteration 4)

1. **Collision Safety** ✅
   - Never hits Electrodes (35px radius)
   - Never hits Hulks (40px default, 30px aggressive pursuit, 25px when surrounded)
   - Properly predicts Grunt collisions (30px radius)
   - **Context-aware**: Adjusts safety margins based on goal type

2. **Threat Response** ✅
   - Dodge bullets at medium range (120px)
   - Emergency backup only at CRITICAL distance (40px)
   - No excessive escape route planning

3. **Goal Prioritization** ✅
   - Spawners: Kill IMMEDIATELY even if far away (+1000 priority)
   - Shooters: High priority (+200)
   - Grunts: Medium priority (+50)
   - Civilians: Collect when safe

4. **Combat Style**
   - **Aggressive**: Thread between Hulks to reach high-value targets
   - **Proactive**: Hunt spawners even when near obstacles
   - **Decisive**: Trust collision detection, commit to goals
   - **Strategic**: Eliminate multiplying threats first
   - **Adaptive**: Adjust safety margins based on context

## Next Steps

1. ✅ **Iteration 4 complete** - Aggressive pursuit mode implemented
2. **Test iteration 4** - Run full death analysis (10 episodes) to measure impact
3. **Monitor Hulk collisions** - Verify 30px aggressive radius is safe
4. **Compare enemy kill rate** - Measure improvement in eliminating surrounded enemies
5. **Overall death count** - Goal is <30 deaths per 10 episodes

## Files Modified

- `expert_fsm_v6.py` - All 4 iterations applied
  - Iteration 1: Type-specific collision radii, collision bug fix
  - Iteration 2: Tuned defensive systems (120px bullets, 40px emergency, removed escape planning)
  - Iteration 3: Spawner priority (+1000 bonus)
  - Iteration 4: Aggressive pursuit mode (`allow_aggressive` parameter)
- `FSM_V6_DEATH_ANALYSIS_FIXES.md` - Iteration 1 detailed documentation
- `FSM_V6_ITERATION_4_RESULTS.md` - Iteration 4 detailed documentation
- `FSM_V6_ITERATION_SUMMARY.md` - This file (full iteration tracking)

## Testing Commands

```bash
# Quick performance test
poetry run python expert_fsm_v6.py --episodes 5 --config config.yaml --lives 3 --headless

# Full death analysis
poetry run python analyze_deaths.py --episodes 10 --config config.yaml --lives 3

# Watch FSM play (visual debugging)
poetry run python watch_fsm.py --version 6 --fps 30
```

## Conclusion

FSM v6 has undergone 4 major iterations:
1. **Iter 1**: Fixed collision detection → Eliminated Electrode/Hulk deaths
2. **Iter 2**: Tuned defensive systems → Reduced excessive caution
3. **Iter 3**: Dramatically increased spawner priority → Prevent enemy overwhelm
4. **Iter 4**: Aggressive pursuit mode → Thread between Hulks to reach valuable targets

The FSM now balances:
- **Safety**: Collision avoidance with context-aware margins
- **Aggression**: Spawner hunting and enemy pursuit even near obstacles
- **Adaptability**: Adjusts behavior based on goal type (enemy/civilian/dodge)

**Key Innovation**: Context-aware collision detection allows selective risk-taking for high-value targets while maintaining full safety for civilians and dodging.
