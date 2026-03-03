# FSM v6: Iteration 7 - Bullet Trajectory Prediction

## Problem Statement

**Level 9 Death Analysis Results**:
- **80% bullet deaths** (12/15) - WORSE at high levels!
- All bullets hitting at **16-28px** (average 23px)
- **40% trapped deaths** (6/15) - escape mode not activating early enough
- Dodging at 120-150px but bullets STILL catching us

## Root Cause Analysis

### Why Bullets Are Hitting Us

**Current dodge logic**:
1. Detect bullet at 120px distance
2. Move away from bullet
3. **Problem**: We move at 5 px/frame, bullet moves at 8-10 px/frame
4. Bullet catches up in ~10 frames
5. At frame 10: bullet at ~20px → **HIT!**

**The fundamental flaw**: We dodge based on **distance**, not **trajectory**!

### The Math

```
Bullet speed: ~8-10 px/frame
Player speed: 5 px/frame
Dodge distance: 120px

Time to close gap: 120px / (8-5) = 40 frames (if moving directly away)
But bullets are AIMED and we're not always moving directly away!
Effective time: ~10-15 frames
Distance after 10 frames: 120 - (10 * 8) + (10 * 5) = 90px still closing!
```

**We're running but bullets are faster!**

## Iteration 7 Solution: Trajectory-Based Dodging

### Change 1: Bullet Trajectory Prediction (NEW)

**Calculate if bullet will actually hit us**:

```python
def predict_bullet_collision(self, bullet: Sprite) -> Tuple[bool, int, Tuple[float, float]]:
    """
    Predict if bullet will collide with us.

    Returns:
        (will_collide, frames_until_collision, collision_point)
    """
    # Get bullet velocity
    bx, by = bullet.velocity_x, bullet.velocity_y

    # If bullet not moving, can't predict
    if abs(bx) < 0.1 and abs(by) < 0.1:
        # Use distance-based dodge as fallback
        return (bullet.distance < 120, int(bullet.distance / 8), (bullet.x, bullet.y))

    # Predict bullet path for next N frames
    for frame in range(1, 30):  # Check next 30 frames
        # Bullet future position
        future_bx = bullet.x + bx * frame
        future_by = bullet.y + by * frame

        # Player future position (if we continue current move)
        # Assume we'll move in current direction
        future_px = self.player_pos.x
        future_py = self.player_pos.y

        # Check collision
        dist = math.hypot(future_bx - future_px, future_by - future_py)
        if dist < 30:  # Collision radius
            return (True, frame, (future_bx, future_by))

    return (False, -1, (0, 0))
```

**Key improvement**: Check if bullet's trajectory will INTERSECT our position, not just if it's close!

### Change 2: Trajectory-Based Dodge Direction

**Instead of "move away from bullet", move PERPENDICULAR to bullet path**:

```python
def get_bullet_dodge_direction(self, bullet: Sprite) -> int:
    """
    Get best direction to dodge bullet based on its trajectory.
    Move PERPENDICULAR to bullet path, not just away!
    """
    bx, by = bullet.velocity_x, bullet.velocity_y

    if abs(bx) < 0.1 and abs(by) < 0.1:
        # Bullet not moving, move away
        return self.get_direction_away(bullet.x, bullet.y)

    # Calculate perpendicular directions to bullet trajectory
    # Bullet moving in direction (bx, by)
    # Perpendicular directions are (by, -bx) and (-by, bx)

    # Check both perpendicular directions
    perp1_x = by
    perp1_y = -bx
    perp2_x = -by
    perp2_y = bx

    # Calculate which perpendicular direction is safer
    # (away from walls, away from other threats)

    # Direction 1
    future1_x = self.player_pos.x + perp1_x * 3
    future1_y = self.player_pos.y + perp1_y * 3
    score1 = 0

    # Prefer directions away from walls
    wall_dist1 = min(future1_x, BOARD_WIDTH - future1_x,
                     future1_y, BOARD_HEIGHT - future1_y)
    score1 += wall_dist1

    # Direction 2
    future2_x = self.player_pos.x + perp2_x * 3
    future2_y = self.player_pos.y + perp2_y * 3
    score2 = 0

    wall_dist2 = min(future2_x, BOARD_WIDTH - future2_x,
                     future2_y, BOARD_HEIGHT - future2_y)
    score2 += wall_dist2

    # Choose better perpendicular direction
    if score1 > score2:
        return self.get_direction_to(future1_x, future1_y)
    else:
        return self.get_direction_to(future2_x, future2_y)
```

**Key improvement**: Move PERPENDICULAR to bullet path = bullet misses us!

### Change 3: Aggressive Bullet Shooting (CRITICAL!)

**User insight**: "We can shoot bullets!"

**Current priority**:
```python
# PRIORITY 1: Projectiles (bullets) - MUST kill these!
for enemy in self.enemies:
    if enemy.type in ['EnforcerBullet', 'TankShell', 'CruiseMissile']:
        if enemy.distance < 300:
            fire_dir = self.calculate_lead_shot(enemy)
```

**Problem**: We only shoot bullets within 300px and only if we can line up a shot.

**New approach**:
```python
# PRIORITY 1: INCOMING bullets that will hit us - DESTROY THEM!
for enemy in self.enemies:
    if enemy.type in ['EnforcerBullet', 'TankShell', 'CruiseMissile']:
        # Check if bullet is coming AT us
        will_hit, frames, collision_pt = self.predict_bullet_collision(enemy)

        if will_hit and frames < 20:  # Will hit us in next 20 frames
            # Try to shoot it!
            fire_dir = self.calculate_lead_shot(enemy)
            if fire_dir is not None:
                # CRITICAL: Shoot incoming bullets first!
                return fire_dir

        # Also shoot bullets within 300px as before
        elif enemy.distance < 300:
            fire_dir = self.calculate_lead_shot(enemy)
            if fire_dir is not None:
                return fire_dir
```

**Key improvement**: Prioritize bullets that are ACTUALLY COMING AT US!

### Change 4: Earlier Escape Mode Trigger

**Current**: Escape mode when < 4 safe moves

**Problem**: By the time we have < 4 moves, we're already boxed in (40% trapped deaths!)

**New**: Escape mode when < 5 safe moves OR when bullet collision predicted in < 10 frames

```python
# Check for escape mode triggers
escape_triggers = []

# Trigger 1: Low safe moves
if len(safe_moves) < 5:  # Changed from < 4
    escape_triggers.append(f"low_safe_moves={len(safe_moves)}")

# Trigger 2: Bullet will hit soon
for enemy in self.enemies:
    if enemy.type in ['EnforcerBullet', 'TankShell', 'CruiseMissile']:
        will_hit, frames, _ = self.predict_bullet_collision(enemy)
        if will_hit and frames < 10:
            escape_triggers.append(f"bullet_collision_in_{frames}f")
            break

if escape_triggers:
    # ESCAPE MODE: Find safest move
    # ... (existing escape mode logic)
```

**Key improvement**: Escape BEFORE trapped, not WHEN trapped!

## Expected Impact

### Primary Goals

**Reduce Bullet Deaths**: 80% → 40-50%
- Trajectory prediction: Dodge perpendicular to bullet path
- Bullet shooting: Destroy incoming bullets
- Early escape: Get out before trapped

**Reduce Trapped Deaths**: 40% → 15-20%
- Earlier escape trigger (< 5 moves vs < 4)
- Bullet collision prediction triggers escape

### Expected Results at Level 9

**Current (Iter 6)**:
- Total deaths: 15
- Bullet deaths: 12 (80%)
- Trapped deaths: 6 (40%)

**Target (Iter 7)**:
- Total deaths: 10-12 (-20% to -33%)
- Bullet deaths: 6-8 (50-60%, down from 80%)
- Trapped deaths: 2-3 (20%, down from 40%)

## Implementation Plan

### Step 1: Add Trajectory Prediction
1. `predict_bullet_collision()` method
2. Calculate if bullet path intersects player
3. Return (will_hit, frames_until, collision_point)

### Step 2: Trajectory-Based Dodge
1. `get_bullet_dodge_direction()` method
2. Move perpendicular to bullet trajectory
3. Choose safer perpendicular direction

### Step 3: Prioritize Incoming Bullets
1. In `find_opportunistic_shot()`:
2. Check which bullets will hit us
3. Shoot those FIRST (before other targets)

### Step 4: Earlier Escape Trigger
1. Change threshold: < 5 safe moves (was < 4)
2. Add bullet collision trigger
3. Escape when bullet will hit in < 10 frames

### Step 5: Update Dodge Logic
1. In `find_goal()`: Use trajectory prediction
2. Instead of distance < 120, check `will_hit` and `frames < 15`
3. Use trajectory-based dodge direction

## Testing Strategy

### Quick Test (3 Episodes at Level 9)
```bash
poetry run python expert_fsm_v6.py --episodes 3 --level 9 --config config.yaml --lives 3 --headless
```

**Look for**:
- Fewer bullet deaths
- "Shooting incoming bullet" messages
- Perpendicular dodge movements
- Earlier escape mode activation

### Full Test (5 Episodes at Level 9)
```bash
poetry run python analyze_deaths.py --episodes 5 --level 9 --config config.yaml --lives 3 --output death_analysis_iter7.json
```

**Success criteria**:
- Bullet deaths < 60% (down from 80%)
- Trapped deaths < 25% (down from 40%)
- Total deaths < 12 (down from 15)

## Risk Assessment

### Low Risk
- ✅ Bullet shooting priority (just reordering)
- ✅ Escape trigger threshold (incremental change)

### Medium Risk
- ⚠️ Trajectory prediction (new logic, may have bugs)
- ⚠️ Perpendicular dodge (may dodge into other threats)

### Mitigation
- Fall back to distance-based dodge if trajectory prediction fails
- Still check if perpendicular move is safe (collision detection)
- Test thoroughly at level 9 before deploying to level 1

## Key Innovations

### 1. Physics-Based Dodging
- Not "move away from threat"
- But "move perpendicular to threat trajectory"
- Bullets MISS instead of CATCH UP

### 2. Predictive Threat Assessment
- Not "is bullet close?"
- But "will bullet HIT us?"
- Only dodge bullets that are actually dangerous

### 3. Active Defense
- Not just dodge bullets
- But SHOOT bullets before they hit
- Turn defense into offense

## Expected Behavior Changes

### Before (Iteration 6):
1. Bullet at 120px → dodge away
2. Bullet chases us
3. Bullet catches up at 20px → death

### After (Iteration 7):
1. Bullet detected → predict trajectory
2. **If will hit**: Shoot it OR dodge perpendicular
3. Bullet misses us OR gets destroyed

**Fundamental shift**: From reactive (dodge when close) to predictive (dodge if will hit)

## Files to Modify

- `expert_fsm_v6.py`:
  1. Add `predict_bullet_collision()` method (~40 lines)
  2. Add `get_bullet_dodge_direction()` method (~50 lines)
  3. Modify `find_opportunistic_shot()` - prioritize incoming bullets (~20 lines)
  4. Modify `find_goal()` - use trajectory prediction (~15 lines)
  5. Update escape mode trigger (~10 lines)

**Total**: ~135 lines of new code

## Next Steps

1. ✅ **Design iteration 7** (this document)
2. ⏳ **Implement trajectory prediction**
3. ⏳ **Implement bullet shooting priority**
4. ⏳ **Test at level 9** (5 episodes)
5. ⏳ **Compare with iteration 6**
6. ⏳ **Iterate based on results**

## Conclusion

**The missing piece**: We've been dodging bullets like they're stationary objects!

Iteration 7 treats bullets as **projectiles with trajectories**:
- Predict where bullet WILL BE
- Dodge perpendicular to trajectory
- Shoot incoming bullets
- Escape earlier when collision predicted

This should reduce bullet deaths from 80% to 50-60% at level 9, making high-level play much more survivable.

**Core insight**: "We can shoot bullets!" - Use our offense as defense!
