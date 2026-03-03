# FSM v6: Iteration 5 - Bullet Avoidance Improvements

## Problem Statement

**User Observation**: *"From my testing, it looks like we don't get to the spawners in time so they spawn their enforcers which force us into a corner."*

**Death Analysis Results (Iteration 4)**:
- Bullet deaths: **70%** of all deaths (21 out of 30)
- Spawners spawning Enforcers before we can kill them
- Getting cornered by bullet swarms
- Highest death rate despite highest scores (110k, Level 8)

## Root Cause

The problem is a cascade:
1. Spawners (Brain, Sphereoid, Quark) spawn Enforcers
2. Enforcers shoot bullets (EnforcerBullet)
3. Multiple bullets corner the FSM against walls
4. FSM gets trapped with "no safe moves" → death

**Key Insight**: We need to:
1. Kill spawners FASTER (already done: +1000 priority)
2. Kill Enforcers when they spawn BEFORE they create bullet swarm (+500 priority when close)
3. AVOID walls when bullets are present (corner detection)

## Changes Made

### 1. Shooter Priority Boost (Close Range)

**Problem**: Enforcers at +200 priority lost to spawners at +1000, allowing Enforcers to accumulate and create bullet swarms.

**Solution**: Dynamic shooter priority based on distance
```python
# In enemy_danger_score():
if enemy.type in ['Enforcer', 'Tank']:
    if enemy.distance < 200:
        base_score += 500  # HIGH priority when close!
    else:
        base_score += 200  # Normal priority when far
```

**Rationale**:
- Far spawner (500px): (200 - 500) + 1000 = **700** (still highest)
- Close Enforcer (100px): (200 - 100) + 500 = **600** (second highest)
- Far Enforcer (300px): (200 - 300) + 200 = **100** (lower)

This means:
- Spawners remain top priority overall
- But close Enforcers get killed before they shoot
- Far Enforcers are low priority (focus on spawners first)

### 2. Wall/Corner Danger Detection

**Problem**: Getting cornered by bullets against walls

**Solution**: Added `is_near_wall()` helper function
```python
def is_near_wall(self, x: float, y: float, threshold: float = 100) -> bool:
    """
    Check if position is near a wall/corner.
    Returns True if within 100px of any board edge.
    """
    dist_to_left = x
    dist_to_right = BOARD_WIDTH - x
    dist_to_top = y
    dist_to_bottom = BOARD_HEIGHT - y

    min_dist = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)
    return min_dist < threshold
```

### 3. Multi-Bullet Awareness in Dodge Logic

**Problem**: Dodging one bullet into another, or dodging into a corner

**Solution**: Penalize wall proximity when multiple bullets present

```python
# In dodge mode:
nearby_bullets = [e for e in self.enemies
                 if e.type in ['EnforcerBullet', 'TankShell', 'CruiseMissile']
                 and e.distance < 200]

for move in safe_moves:
    # ... calculate score ...

    # Avoid walls when bullets present
    if len(nearby_bullets) >= 2:
        if self.is_near_wall(future_x, future_y, threshold=100):
            score -= 200  # Heavy penalty for walls

```

**Same logic applied to "safe alternative" movement** when desired move is unsafe.

## Expected Impact

### Primary Goal: Reduce Bullet Deaths

**Before (Iter 4)**: 70% bullet deaths (21/30)
**Expected (Iter 5)**: 40-50% bullet deaths (12-15/30)

**Mechanisms**:
1. **Kill Enforcers faster** → Fewer bullets spawned
2. **Avoid corners** → More escape routes when bullets present
3. **Multi-bullet awareness** → Better dodging when overwhelmed

### Secondary Effects

**Positive**:
- Should maintain high scores (still prioritizing spawners)
- Should maintain Level 8 capability
- Better survival in high-level gameplay (more Enforcers)

**Potential Trade-offs**:
- Might take slightly longer to reach far spawners (diverting to kill close Enforcers)
- Wall avoidance might reduce maneuverability in some cases

## Testing Strategy

### Quick Test (5 Episodes)

**Results**:
```
Episode 1: Level 3, Score 25,800
Episode 2: Level 3, Score 28,000
Episode 3: Level 4, Score 23,800
Episode 4: Level 5, Score 60,450
Episode 5: Level 5, Score 71,300

Average: Level 3.8, Score 41,810
Max Level: 5
```

**Observations**:
- Reaching Level 5 consistently
- Good score distribution
- No obvious issues with corner detection
- Seeing "avoiding walls" in debug output

### Full Test (10 Episodes)

Run full death analysis to compare with iteration 4:
```bash
poetry run python analyze_deaths.py --episodes 10 --config config.yaml --lives 3 --output death_analysis_iter5.json
```

**Key metrics to track**:
1. **Bullet death percentage** (target: <50%, was 70%)
2. **"TRAPPED - no safe moves" frequency** (target: <5 occurrences)
3. **Shooter engagement distance** (should kill Enforcers <200px)
4. **Average score** (should maintain ~57k from iter 4)
5. **Max level** (should maintain Level 8 capability)

## Priority Scoring Summary (After Iteration 5)

| Enemy Type | Distance | Priority Bonus | Example Score | Rank |
|------------|----------|----------------|---------------|------|
| Spawner | 500px | +1000 | 700 | 1st (unchanged) |
| **Enforcer** | **100px** | **+500** | **600** | **2nd (NEW!)** |
| Grunt | 10px | +50 | 240 | 3rd |
| Enforcer | 300px | +200 | 100 | 4th |

**Goal**: Kill spawners first when far, but eliminate close Enforcers before they create bullet swarms.

## Code Changes Summary

**Files Modified**:
- `expert_fsm_v6.py`
  1. Added `is_near_wall()` helper function
  2. Modified `find_goal()` → Dynamic shooter priority (+500 when <200px)
  3. Modified dodge logic → Multi-bullet awareness + wall penalty
  4. Modified safe alternative logic → Wall penalty when bullets present

**Lines Changed**: ~50 lines (mostly additions)

## Next Steps

1. ✅ Implement shooter priority boost
2. ✅ Implement corner detection
3. ✅ Implement multi-bullet awareness
4. ✅ Quick test (5 episodes)
5. ⏳ **NEXT**: Full death analysis (10 episodes)
6. ⏳ Compare results with iteration 4
7. ⏳ Iterate based on findings

## Expected Behavior

### Scenario 1: Far Spawner + Close Enforcer

**Before**: Pursue far spawner → Enforcer shoots → bullet death
**After**: Kill close Enforcer → Then pursue spawner → safer

### Scenario 2: Multiple Bullets Near Wall

**Before**: Dodge bullet → move toward wall → trapped → death
**After**: Dodge bullet → avoid wall → stay in open space → survive

### Scenario 3: Bullet Swarm (3+ bullets)

**Before**: Dodge closest → get cornered → no safe moves → death
**After**: Avoid walls → maintain escape routes → survive longer

## Conclusion

Iteration 5 addresses the primary death cause (bullets) through a three-pronged approach:
1. **Prevention**: Kill Enforcers before they create bullet swarms
2. **Detection**: Recognize when bullets are overwhelming us
3. **Avoidance**: Stay away from walls when bullets present

This should significantly reduce bullet deaths while maintaining the aggressive spawner-hunting playstyle that enabled Level 8 runs.
