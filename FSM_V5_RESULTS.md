# FSM v5 Results - Target Persistence & Wall Redirect

## Summary

FSM v5 implements a complete redesign with state-based logic matching human player mental model. Key improvements:

1. **Target Persistence** - Lock onto a target until killed/collected
2. **Opportunistic Shooting** - Always shoot at any enemy on firing line
3. **Wall Redirect** - Move along walls instead of stopping
4. **Detailed Debugging** - Clear explanations of all decisions

## Performance (10 Episodes)

```
Average Score:  8050.0 ± 3485.4
Average Kills:  80.2 ± 34.7
Average Level:  2.0 ± 0.4
Max Level:      3
Average Steps:  1364.1
```

## Comparison to Previous Versions

| Version | Avg Level | Max Level | Avg Kills | Key Feature |
|---------|-----------|-----------|-----------|-------------|
| v1 | 1.0 | 1 | 20.9 | Baseline FSM |
| v3 | 1.2 | 2 | 32.3 | Shooting alignment fix (+54% kills) |
| v4 | 1.0 | 2 | Variable | Family priority, level completion |
| **v5** | **2.0** | **3** | **80.2** | **Target persistence + wall redirect** |

**Improvement from v3**: +148% kills, +67% avg level, **first time reaching level 3**

## Key Fixes in v5

### 1. Target Persistence (SOLVED: Jittering)

**Problem**: FSM was constantly switching targets, vibrating back and forth.

**Solution**: Lock onto a target and pursue it until killed/collected.

```python
def find_target(self) -> Optional[Sprite]:
    # Priority 1: Immediate threats (<80px)
    # Priority 2: Keep current target if still valid
    # Priority 3: Civilians if only 1 enemy left
    # Priority 4: Dangerous enemies (spawners, shooters, then by distance)
    # Priority 5: Civilians if safe
```

**Impact**: FSM now smoothly pursues targets instead of oscillating.

### 2. Opportunistic Shooting (SOLVED: Missing Enemies)

**Problem**: FSM only shot at primary target, missed opportunities.

**Solution**: Always shoot at ANY enemy on a firing line, not just the target.

```python
def find_opportunistic_shot(self) -> Optional[int]:
    # Check all enemies within 300px
    for enemy in self.enemies:
        if enemy.distance < 300:
            fire_dir = self.can_shoot_at(enemy)
            if fire_dir is not None:
                return fire_dir
```

**Impact**: FSM shoots constantly while moving, killing more enemies.

### 3. Wall Redirect (SOLVED: Standing Still)

**Problem**: FSM would hit a wall and freeze with `move_action = STAY`.

**Solution**: Redirect movement along the wall toward target.

```python
# Before (v4):
if x < margin and move_action in [LEFT, UP_LEFT, DOWN_LEFT]:
    move_action = STAY  # FREEZES!

# After (v5):
if x < margin and move_action in [LEFT, UP_LEFT, DOWN_LEFT]:
    # Move along wall toward target
    if target and target.y < self.player_pos.y:
        move_action = UP
    else:
        move_action = DOWN
```

**Impact**: FSM never stops moving, no more getting stuck at walls.

### 4. Detailed Debugging

**Problem**: Hard to diagnose why FSM makes certain decisions.

**Solution**: Print clear explanations every 20 frames.

```
[FSM] ENEMY: Grunt (dist=171) | Closest: Electrode (dist=121)
[FSM] Movement: Moving toward enemy Grunt (dist=171)
[FSM] Shooting at Electrode (dist=121)

[FSM] ENEMY: Grunt (dist=77) | Closest: Grunt (dist=77)
[FSM] Movement: WALL REDIRECT (top edge -> RIGHT)
[FSM] Shooting at Grunt (dist=77)

[FSM] ENEMY: Grunt (dist=37) | Closest: Grunt (dist=37)
[FSM] Movement: BACKING UP from Grunt (dist=37, too close)
[FSM] Shooting at Grunt (dist=37)
```

**Impact**: Easy to identify and fix bugs by watching debug output.

## Remaining Issues

### Issue 1: Not Reaching Level 5 Consistently

Current best: Level 3 (once in 10 episodes)
Target: Level 5+ for imitation learning

**Analysis**:
- FSM is killing 80 enemies per game (good!)
- FSM is reaching level 2-3 consistently
- Need to improve survival in later levels with more enemy types

**Potential improvements**:
1. Better handling of spawners (Brain, Sphereoid, Quark)
2. More aggressive projectile avoidance (EnforcerBullet, TankShell)
3. Strategic retreating when overwhelmed (>5 close enemies)
4. Family collection timing (collect early before Progs convert them)

### Issue 2: High Performance Variance

Standard deviation: ±34.7 kills

Some episodes: 110+ kills
Other episodes: 40-50 kills

**Causes**:
- Random spawn patterns (enemies clustered vs. spread out)
- Early bad positioning leading to death
- Spawner RNG (some games have more spawners)

**Potential fixes**:
1. More robust initial positioning (start at edge, not center)
2. Better spawner priority (kill them ASAP)
3. Emergency retreat logic when surrounded

### Issue 3: Family Collection

FSM still struggles to collect all family members before they're converted to Progs.

**Current behavior**:
- Family is Priority 3 (after threats and current target)
- Only collected when 1 enemy left and safe

**Improvement needed**:
- Detect "safe windows" early in wave
- Interrupt target pursuit if family is very close (<100px)
- Track family conversion timing (they convert after ~10 seconds)

## Debug Output Examples

### Successful Target Lock

```
[FSM] ENEMY: Grunt (dist=120) | Closest: Grunt (dist=120)
[FSM] Movement: Moving toward enemy Grunt (dist=120)
[FSM] Shooting at Grunt (dist=120)

[FSM] ENEMY: Grunt (dist=71) | Closest: Grunt (dist=71)
[FSM] Movement: Committed move (6 frames left)  ← PERSISTENCE!
[FSM] Shooting at Grunt (dist=260)

[FSM] ENEMY: Grunt (dist=64) | Closest: Grunt (dist=64)
[FSM] Movement: Committed move (2 frames left)
[FSM] Shooting at Grunt (dist=64)
```

FSM locked onto Grunt and pursued it for 8 frames until killed.

### Wall Redirect Working

```
[FSM] ENEMY: Grunt (dist=244) | Closest: Grunt (dist=244)
[FSM] Movement: WALL REDIRECT (bottom edge -> LEFT)  ← NO FREEZE!
[FSM] Not shooting

[FSM] ENEMY: Grunt (dist=206) | Closest: Grunt (dist=206)
[FSM] Movement: WALL REDIRECT (bottom edge -> LEFT)
[FSM] Not shooting

[FSM] ENEMY: Grunt (dist=161) | Closest: Grunt (dist=161)
[FSM] Movement: WALL REDIRECT (bottom edge -> LEFT)
[FSM] Not shooting
```

FSM redirected movement along wall instead of freezing.

### Backing Up from Immediate Threat

```
[FSM] ENEMY: Grunt (dist=37) | Closest: Grunt (dist=37)
[FSM] Movement: BACKING UP from Grunt (dist=37, too close)  ← EMERGENCY!
[FSM] Shooting at Grunt (dist=37)

[FSM] ENEMY: Grunt (dist=29) | Closest: Grunt (dist=29)
[FSM] Movement: BACKING UP from Grunt (dist=29, too close)
[FSM] Shooting at Grunt (dist=29)
```

FSM correctly identified immediate threat and retreated while shooting.

## Next Steps

1. **Improve spawner hunting** - Kill Brains/Sphereoids before they spawn too many enemies
2. **Better projectile avoidance** - React faster to EnforcerBullets and TankShells
3. **Family collection strategy** - Collect early before conversion
4. **Emergency retreat logic** - Detect "overwhelmed" state (>5 close enemies)
5. **Test with harder config** - Try `config.yaml` (harder than curriculum)

## Code Files

- `expert_fsm_v5.py` - Main FSM implementation
- `watch_fsm.py` - Visual debugging tool (use `--version 5 --fps 30`)
- `FSM_V5_RESULTS.md` - This file

## How to Test

```bash
# Watch FSM play (visual debugging)
poetry run python watch_fsm.py --version 5 --fps 30

# Batch test (statistics)
poetry run python expert_fsm_v5.py --episodes 50 --config config.yaml --headless

# God mode (see how far it gets)
poetry run python watch_fsm.py --version 5 --fps 30 --godmode
```

## Conclusion

FSM v5 is a **major improvement** over previous versions:
- **+148% kills** compared to v3
- **Reached level 3** for first time
- **No jittering** - smooth movement
- **No wall freezing** - continuous motion
- **Better shooting** - opportunistic firing

Still need to reach **level 5+** for imitation learning, but v5 is a solid foundation. Next iteration should focus on spawner hunting and projectile avoidance to push performance higher.
