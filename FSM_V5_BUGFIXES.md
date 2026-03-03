# FSM v5 Bug Fixes Session

## Summary

This session focused on fixing critical bugs discovered during testing of FSM v5.

## Bugs Fixed

### 1. Wall Redirect Commitment Bug
**Problem**: FSM getting stuck sliding along walls, moving left/right repeatedly without escaping.

**Root Cause**: Wall redirect changed `move_action` but didn't break movement commitment, so FSM kept executing old committed move.

**Fix**: Break commitment when wall redirect occurs:
```python
if x < margin and move_action in [LEFT, UP_LEFT, DOWN_LEFT]:
    move_action = UP  # Redirect
    # Break commitment so we actually move away from wall
    self.committed_move = move_action
    self.commit_frames_remaining = self.commit_duration
```

### 2. Civilian Collection Too Restrictive
**Problem**: FSM ignoring nearby civilians even when safe to collect.

**Root Cause**: Required exactly 1 enemy AND civilian closer AND no dangerous types - too strict.

**Fix**: Made collection more opportunistic:
- Collect if 0 enemies (always)
- Collect if 1-2 enemies AND civilian closer
- Collect if civilian < 80px AND no immediate threats (< 100px)

### 3. Grunt Misunderstanding (CRITICAL USER CORRECTION)
**Problem**: Initially thought Grunts were immortal like Hulks.

**User Correction**: "HULKS are the immortal threat. Grunts are regular. We need to kill grunts, which die to one hit."

**Fix**:
- Grunts are regular enemies with +30 priority bonus (they chase)
- Hulks are immortal obstacles (already filtered in parse_sprites)
- Removed special Grunt deprioritization

### 4. Projectile Priority
**Problem**: FSM not prioritizing bullets highly enough.

**Fix**: Added +200 priority bonus to projectiles (EnforcerBullet, TankShell, CruiseMissile) - highest priority after immediate threats.

### 5. Civilian Navigation Bug
**Problem**: FSM moving horizontally when should move vertically toward civilians (e.g., player at bottom, civilian at top, FSM moved LEFT/RIGHT instead of UP).

**Root Cause**: `get_safe_direction_to()` was applying center avoidance to civilians, causing wrong direction choices.

**Fix**: For civilians, use direct angle calculation without center avoidance:
```python
if self.current_target_type == 'civilian':
    # Direct path to civilian - calculate angle
    dx = target_x - self.player_pos.x
    dy = target_y - self.player_pos.y
    angle = math.atan2(dy, dx)
    # ... convert to 8-way direction
```

Also break commitment when switching to civilian target:
```python
if self.current_target_type != 'civilian':
    self.commit_frames_remaining = 0
```

### 6. Wall Avoidance Blocking Civilian Collection
**Problem**: Wall redirect preventing FSM from collecting civilians near walls (run at them, then back up).

**Fix**: Skip wall redirect when targeting civilians:
```python
if self.player_pos and self.current_target_type != 'civilian':
    # Wall avoidance logic...
```

### 7. Infinite Loop Bug (CRITICAL)
**Problem**: FSM stuck backing up from Electrodes forever.

**Root Cause**: Electrodes are obstacles, not moving enemies. FSM tried to back up from them, but since they don't move, it just backed up in place infinitely.

**Fix**: Don't apply backup behavior to obstacles:
```python
is_obstacle = target.type in ['Electrode', 'Hulk']

if self.current_target_type == 'enemy' and target.distance < 80 and not is_obstacle:
    # Backup logic only for real enemies...
```

## Priority System (Final)

1. **Immediate threats** (<80px, non-obstacles)
2. **Current target** (target persistence)
3. **Civilians** (if 0-2 enemies, or very close)
4. **Dangerous enemies** by score:
   - Projectiles: +200 (HIGHEST - instant death)
   - Spawners (Brain, Sphereoid, Quark): +150
   - Shooters (Enforcer, Tank): +100
   - Grunts: +30 (they chase)
   - Base: 100 - distance

## Movement Behaviors

- **Electrodes/Hulks**: Treated as obstacles, no backup behavior
- **Projectiles**: 80px backup distance
- **Regular enemies**: 70px backup distance
- **Civilians**: Direct path (no center avoidance), no wall redirect

## Key Insights

1. **Electrodes are not threats** - they're obstacles that should be shot but not fled from
2. **Grunts are regular enemies** - not immortal, kill them normally
3. **Projectiles are highest priority** - they kill instantly
4. **Civilians need special handling** - direct paths, no wall avoidance
5. **Target persistence is critical** - prevents jittering

## Testing Status

- Single episode tests: Working correctly
- Batch test (10 episodes): In progress

### 8. Wall Redirect Preventing Movement (CRITICAL)
**Problem**: FSM stuck at walls unable to chase distant enemies. Example: Player at right wall (x=627), Grunt at left side (x=176), FSM just moved up/down along the wall.

**Root Cause**: Wall redirect was TOO aggressive - it redirected ANY movement near a wall, even when trying to move AWAY from the wall toward a target.

**Fix**: Only redirect if moving INTO the wall AND target is not on the opposite side:
```python
if x > BOARD_WIDTH - margin and move_action in [RIGHT, UP_RIGHT, DOWN_RIGHT]:
    # At right wall AND moving into it
    if target and target.x < self.player_pos.x:
        # Target is to the left - move toward it (away from wall)
        # Don't redirect! Let the FSM move left toward target
        pass
    else:
        # Target is on same side as wall - move along wall
        # ... redirect logic
```

**Impact**: FSM can now leave walls to chase distant enemies.

### 9. Angle-to-Direction Conversion Bug (CRITICAL)
**Problem**: FSM moving UP when should move DOWN toward targets. Example: Player at (62, 0) top-left, Grunt at (128, 296) below, FSM moved UP instead of DOWN.

**Root Cause**: The `atan2(dy, dx)` function was using raw screen coordinates where y=0 is at the TOP and y increases DOWNWARD. When target is below player (dy > 0), atan2 returns a positive angle, but the angle-to-direction mapping incorrectly interpreted positive angles as "target above".

**Example of the bug**:
```python
# Player at (62, 0), Grunt at (128, 296)
dx = 128 - 62 = 66
dy = 296 - 0 = 296  # Positive (target BELOW in screen coords)
angle = atan2(296, 66) = 77°  # Positive angle
# Code returned UP (wrong!) because 67.5° < 77° < 112.5°
# Should return DOWN_RIGHT (target is down and to the right)
```

**Fix**: Negate dy in all atan2 calls to flip the y-axis:
```python
# Correct conversion for screen coordinates
angle = math.atan2(-dy, dx)  # Negate dy!
# Now: 0°=RIGHT, 90°=UP, 180°=LEFT, 270°=DOWN
# Player at (62, 0), Grunt at (128, 296)
# angle = atan2(-296, 66) = -77° → +360 = 283°
# 247.5° < 283° < 292.5° → DOWN (correct!)
```

**Impact**: FSM now moves in the correct direction toward all targets. This was causing the FSM to run away from enemies, get stuck at walls, and fail to collect civilians.

**Locations Fixed**:
- `get_safe_direction_to()` line 147: `math.atan2(-dy, dx)`
- Civilian direct navigation line 640: `math.atan2(-dy, dx)`

## Files Modified

- `expert_fsm_v5.py` - All bug fixes applied
- `FSM_V5_BUGFIXES.md` - This documentation
