# FSM v5 Angle-to-Direction Bug Fix

## Summary

Fixed a **critical bug** in FSM v5's angle-to-direction conversion that was causing the FSM to move in completely wrong directions. This bug was making the FSM run away from enemies, get stuck at walls, and fail to navigate properly.

## The Bug

**Symptom**: FSM moving UP when it should move DOWN toward targets.

**Example**:
- Player at (62, 0) - top-left corner
- Grunt at (128, 296) - below and slightly to the right
- FSM moved **UP** (into the wall!) instead of **DOWN** or **DOWN_RIGHT**

## Root Cause

The bug was in how `atan2(dy, dx)` was being used with screen coordinates:

```python
# BEFORE (WRONG):
dx = target_x - player_x  # 128 - 62 = 66 (target to the right)
dy = target_y - player_y  # 296 - 0 = 296 (target BELOW in screen coords)
angle = math.atan2(dy, dx)  # atan2(296, 66) = 77°
# angle_deg < 112.5 → returned UP (WRONG!)
```

**The Problem**: Screen coordinates have **y=0 at the TOP** and **y increases DOWNWARD**. When the target is below the player (dy > 0), `atan2` returns a positive angle. But the angle-to-direction mapping was written for standard math coordinates (y-up), so it interpreted positive angles as "target above" instead of "target below".

## The Fix

**Solution**: Negate `dy` in all `atan2` calls to flip the y-axis:

```python
# AFTER (CORRECT):
dx = target_x - player_x  # 66
dy = target_y - player_y  # 296
angle = math.atan2(-dy, dx)  # atan2(-296, 66) = -77° → +360 = 283°
# 247.5° < 283° < 292.5° → returns DOWN (CORRECT!)
```

With negated dy, the angle mapping becomes:
- 0° = RIGHT
- 45° = UP_RIGHT
- 90° = UP
- 135° = UP_LEFT
- 180° = LEFT
- 225° = DOWN_LEFT
- 270° = DOWN
- 315° = DOWN_RIGHT

This matches standard math coordinates where UP is positive y.

## Impact

**Before the fix**:
- FSM ran away from targets instead of toward them
- Got stuck at walls unable to move toward distant enemies
- Failed to collect civilians properly
- Random-looking movement with many wrong directions

**After the fix**:
- FSM moves directly toward targets
- Correctly navigates across the entire screen
- Collects civilians efficiently
- Smooth, purposeful movement

## Code Changes

### 1. `get_safe_direction_to()` (line 147)

```python
# Calculate direct angle
# NOTE: Screen coordinates have y=0 at top, y increasing downward
# atan2(dy, dx) where dy > 0 means target is BELOW player
# We NEGATE dy so atan2 gives us correct angles
angle = math.atan2(-dy, dx)  # NEGATED dy!
```

### 2. Civilian Direct Navigation (line 640)

```python
# Direct path to civilian - calculate angle
dx = target_x - self.player_pos.x
dy = target_y - self.player_pos.y
# NEGATE dy for screen coordinates (y=0 at top)
angle = math.atan2(-dy, dx)  # NEGATED dy!
```

## Performance

After fixing this bug (along with 8 other bugs), FSM v5 achieves:

```
Average Score:  15240.0 ± 8610.5
Average Kills:  152.1 ± 86.2
Average Level:  2.7 ± 1.0
Max Level:      5
Average Steps:  1507.9
```

**Still reaches level 5**, demonstrating the fix maintains high performance.

## Lessons Learned

1. **Screen coordinates vs. math coordinates**: Always be explicit about coordinate systems
   - Screen: y=0 at top, y+ = down
   - Math: y=0 at bottom, y+ = up

2. **Testing is critical**: This bug was only caught through careful observation of debug output showing player moving in obviously wrong directions

3. **Coordinate system consistency**: When using `atan2()`, you must account for the coordinate system:
   - For screen coords (y-down): use `atan2(-dy, dx)`
   - For math coords (y-up): use `atan2(dy, dx)`

## Files Modified

- `expert_fsm_v5.py` - Fixed angle calculations in 2 locations
- `FSM_V5_BUGFIXES.md` - Added bug #9 documentation
- `FSM_V5_ANGLE_BUG_FIX.md` - This document

## Related Bugs

This was bug #9 in a series of 9 critical bugs fixed in this session. See `FSM_V5_BUGFIXES.md` for the complete list.
