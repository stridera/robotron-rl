# FSM v3 Analysis - Shooting Alignment Fix

## Critical Bug Fixed

**Problem Identified**: The FSM was shooting at enemies without positioning itself to hit them on the 8-way grid.

In Robotron, bullets only travel in 8 directions (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°). If an enemy is at 15° angle, shooting at UP_RIGHT (45°) will miss.

**Solution**: FSM v3 implements "shooting alignment":
1. Check if target is on one of the 8 firing lines (`is_aligned_with_target()`)
2. If aligned → shoot
3. If not aligned → move perpendicular to get target on firing line (`get_alignment_move()`)

## Performance Improvements

### v3 vs v1 (Baseline) - 10 episodes each:

| Metric | v1 (Baseline) | v3 (Alignment) | Improvement |
|--------|---------------|----------------|-------------|
| Avg Level | 1.0 | 1.2 | **+20%** |
| Max Level | 1 | 2 | **First time reaching level 2** |
| Avg Kills | 20.9 | 32.3 | **+54.5%** |
| Avg Survival | 1028 steps | 1265 steps | **+23.1%** |

### v3 Extended Test - 20 episodes:

| Metric | Value |
|--------|-------|
| Avg Level | 1.0 |
| Max Level | 1 |
| Avg Kills | 33.4 ± 15.5 |
| Avg Score | 3350 ± 1556 |
| Avg Survival | 1067 steps |

**Key Finding**: High variance in performance (15.5 kills std dev). FSM sometimes gets 59 kills, sometimes only 2 kills. This suggests inconsistent behavior based on spawn patterns.

## Remaining Issues

### Issue 1: Still Not Reaching Level 2 Consistently

- In 20 episodes, max level was only 1
- In earlier 10 episodes, reached level 2 once
- Need to clear level 1 more reliably

**Level 1 Requirements**:
- Kill all 15 grunts
- Kill 5 electrodes (obstacles)
- Collect 1 family member
- Avoid getting killed

FSM is killing 33 enemies on average but still dying on level 1. This suggests:
- Dying after clearing most enemies
- Not collecting family efficiently
- Getting trapped by remaining enemies

### Issue 2: Alignment Logic May Be Too Aggressive

The `get_alignment_move()` function moves perpendicular to align, but this might:
- Move INTO other enemies while trying to align
- Move away from edges (dangerous)
- Take too long to align (enemy escapes or more enemies arrive)

**Possible fix**:
- Check if alignment move is safe (won't move into enemies)
- If not safe, retreat to edge instead
- Only try to align if clear space

### Issue 3: Not Prioritizing Level Completion

FSM doesn't have a "level completion mode" where it actively hunts down the last few enemies to advance.

Currently it's reactive - waits for enemies to come close. But to clear a level:
- Need to hunt down distant grunts
- Need to collect family member
- Need to be aggressive when only a few enemies left

### Issue 4: High Performance Variance

2-59 kills range suggests FSM behavior depends heavily on:
- Initial spawn positions
- Whether enemies cluster or spread out
- Random projectile timing

A robust FSM should handle all spawn patterns.

## Next Improvements Needed

### Priority 1: Family Collection (CRITICAL)

FSM is not collecting family members reliably. In 20 episodes, it should collect 20 family members if working properly.

**Current issue**: Family collection is Priority 3 (after retreat and dodge). By the time FSM tries to collect, family may be:
- Converted to Prog (brainwashed)
- Too dangerous to approach
- FSM dies before collecting

**Fix needed**:
```python
# Detect "safe window" for family collection
def is_safe_to_collect_family():
    # Early in wave, fewer enemies
    if enemy_count < 5:
        return True

    # Family very close, no enemies blocking
    if nearest_family.distance < 100 and no_enemies_between:
        return True

    return False
```

### Priority 2: Level Completion Mode

When close to clearing a level, switch to aggressive hunting:

```python
def should_enter_completion_mode():
    # Only a few enemies left
    if len(enemies) <= 3:
        return True

    # Killed most enemies
    if kills_this_level >= 12:  # Level 1 has ~15 grunts
        return True

    return False

def hunt_remaining_enemies():
    # Find furthest enemy
    furthest_enemy = max(enemies, key=lambda e: e.distance)

    # Move toward it (ignore edge bias)
    move_action = get_direction_to(furthest_enemy.x, furthest_enemy.y)

    # Align and shoot
    if is_aligned:
        fire_action = get_direction_to(furthest_enemy.x, furthest_enemy.y)
```

### Priority 3: Safe Alignment

Only try to align if it's safe:

```python
def get_safe_alignment_move(target):
    # Calculate alignment move
    alignment_dir = get_alignment_move(target)

    # Simulate where we'd move
    future_pos = simulate_move(alignment_dir)

    # Check if that position is safe
    enemies_at_future_pos = count_enemies_within(future_pos, radius=60)

    if enemies_at_future_pos > 2:
        # Not safe - retreat to edge instead
        return get_edge_direction()

    return alignment_dir
```

### Priority 4: Spawner Hunting (from v2)

v2 tried to add spawner hunting but it made performance worse. Need to re-implement more carefully:

```python
def should_hunt_spawner():
    spawners = [e for e in enemies if e.type in ['Brain', 'Sphereoid', 'Quark']]

    if not spawners:
        return None

    nearest = spawners[0]

    # Only hunt if:
    # 1. Spawner is reasonably close
    # 2. Path is mostly clear
    # 3. We're not overwhelmed by other enemies

    if nearest.distance > 250:
        return None  # Too far

    if len(close_enemies) > 5:
        return None  # Too many other threats

    enemies_between = count_enemies_between(player, nearest)
    if enemies_between > 3:
        return None  # Path blocked

    return nearest  # Safe to hunt
```

## Expected Improvements

If we implement these fixes:

**Family Collection**: Should reach level 2 in 50%+ of episodes (currently ~10%)

**Level Completion Mode**: Should clear level 1 faster, advance to level 2-3

**Safe Alignment**: Should reduce deaths while aligning (fewer "walked into enemy" deaths)

**Spawner Hunting**: Should prevent enemy count from growing uncontrollably

**Target Performance**:
- Avg Level: 2-3 (currently 1.0-1.2)
- Max Level: 5 (currently 2)
- Avg Kills: 50-80 (currently 33)
- Consistent level 2 clearing (currently rare)

## Implementation Plan

1. **Add family collection priority** (1-2 hours)
   - Detect safe windows
   - Interrupt retreat if family is very close
   - Track family collection success rate

2. **Add level completion mode** (1 hour)
   - Detect when close to clearing level
   - Hunt down remaining enemies
   - Aggressive family collection at end

3. **Add safe alignment checks** (1 hour)
   - Check if alignment move is safe
   - Fall back to retreat if unsafe
   - Prefer shooting from safe distance over risky alignment

4. **Re-implement spawner hunting carefully** (1-2 hours)
   - Only when safe
   - Clear path required
   - Not when overwhelmed

5. **Test and iterate** (2-3 hours)
   - Run 50-episode batches
   - Compare performance
   - Adjust thresholds

**Total time estimate**: 6-10 hours of work

**Target outcome**: FSM reaches level 5+ consistently, ready for imitation learning demonstrations.
