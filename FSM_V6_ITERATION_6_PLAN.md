# FSM v6: Iteration 6 - Escape Corridor & Grunt Fix

## Problem Statement

**Iteration 5 Results**:
- Bullet deaths: 62% (19/29) - still dominant ⚠️
- Trapped deaths: 24% (7/29) - getting cornered despite wall avoidance
- Grunt deaths: 21% (6/29) - increased from 17% ⚠️
- Bullets hitting at 16-30px (should dodge at 120px)
- Grunts hitting at 30-39px (collision radius is 30px)

## Root Causes Identified

### 1. Bullet Dodging Insufficient
**Problem**: ALL 19 bullet deaths happened at < 30px distance
- Dodge threshold: 120px
- Actual hit distance: 16-30px (average 21px)
- **Gap**: Bullets closing 90-100px AFTER dodge triggered

**Why this happens**:
- Bullets move fast (5-10 px/frame)
- FSM dodges at 120px but bullet catches up
- Wall avoidance (100px threshold) insufficient
- Getting cornered with "no safe moves"

### 2. Getting Trapped (24% of deaths)
**Problem**: 7 deaths show "TRAPPED - no safe moves" or very limited moves

**Contributing factors**:
- Wall avoidance 100px threshold too small
- No proactive escape (waits until trapped)
- Pursuing goals even when escape routes closing
- Multiple bullets boxing FSM into corners

### 3. Grunt Collision Prediction Off
**Problem**: Grunts hitting at 30-39px (collision radius 30px)
- 4 deaths at < 35px (should be safe beyond 30px)
- Grunts fast-moving, prediction failing

## Iteration 6 Solution: Escape Corridor System

### Change 1: Escape Corridor Detection

**Concept**: Track "escape routes" - directions with open space

**Implementation**:
```python
def count_escape_corridors(self) -> int:
    """
    Count number of directions with open escape routes.

    An escape corridor is a direction where:
    - No walls within 150px
    - No bullets within 100px
    - No enemies within 80px (except family/obstacles)

    Returns:
        Number of safe escape directions (0-8)
    """
    escape_count = 0

    for move in [UP, UP_RIGHT, RIGHT, DOWN_RIGHT, DOWN, DOWN_LEFT, LEFT, UP_LEFT]:
        dx, dy = MOVE_DELTAS[move]

        # Check 150px in this direction
        check_x = self.player_pos.x + dx * 150
        check_y = self.player_pos.y + dy * 150

        # Wall check
        if not (0 < check_x < BOARD_WIDTH and 0 < check_y < BOARD_HEIGHT):
            continue  # This direction leads to wall

        # Threat check
        has_threat = False
        for enemy in self.enemies:
            if enemy.type in ['EnforcerBullet', 'TankShell', 'CruiseMissile']:
                # Check if bullet in this direction
                threat_angle = math.atan2(enemy.y - self.player_pos.y,
                                         enemy.x - self.player_pos.x)
                direction_angle = math.atan2(dy, dx)
                angle_diff = abs(threat_angle - direction_angle)

                if angle_diff < math.pi/4 and enemy.distance < 100:  # 45° cone, 100px
                    has_threat = True
                    break

        if not has_threat:
            escape_count += 1

    return escape_count
```

**Rationale**:
- Tracks available escape routes (not just "safe moves")
- Looks 150px ahead (not just 3 frames)
- Considers walls, bullets, and enemies
- Used to detect when FSM getting boxed in

### Change 2: Escape Mode State

**Trigger**: Enter "Escape Mode" when escape corridors < 3

**Escape Mode Behavior**:
```python
# In decide_action(), before goal pursuit:

escape_corridors = self.count_escape_corridors()

if escape_corridors < 3:
    # ESCAPE MODE: Prioritize survival over goals
    # Find move that maximizes escape corridors

    best_move = None
    best_corridor_count = -1

    for move in safe_moves:
        # Simulate this move
        dx, dy = MOVE_DELTAS[move]
        future_x = self.player_pos.x + dx * 3
        future_y = self.player_pos.y + dy * 3

        # Count escape corridors from future position
        # (would need to pass future position to count_escape_corridors)
        future_corridors = self.count_escape_corridors_from(future_x, future_y)

        if future_corridors > best_corridor_count:
            best_corridor_count = future_corridors
            best_move = move

    if best_move:
        move_action = best_move
        move_reason = f"ESCAPE MODE: {escape_corridors} corridors → {best_corridor_count}"
```

**Rationale**:
- Proactively escape BEFORE getting trapped
- Maximizes escape routes instead of pursuing goals
- Prevents the 24% "TRAPPED" deaths
- Only activates when truly needed (< 3 escape routes)

### Change 3: Increase Wall Avoidance Threshold

**Current**: Penalize walls when within 100px AND 2+ bullets present

**New**: Penalize walls when within 150px AND any bullets within 200px

```python
# In dodge mode and safe alternative logic:

nearby_bullets = [e for e in self.enemies
                 if e.type in ['EnforcerBullet', 'TankShell', 'CruiseMissile']
                 and e.distance < 200]

if len(nearby_bullets) >= 1:  # Changed from >= 2
    if self.is_near_wall(future_x, future_y, threshold=150):  # Changed from 100
        score -= 250  # Increased from 200
```

**Rationale**:
- More aggressive wall avoidance (150px vs 100px)
- Trigger with ANY bullet nearby (not just 2+)
- Higher penalty (250 vs 200)
- Should prevent getting cornered

### Change 4: Fix Grunt Collision Radius

**Current**: Grunt collision radius = 30px

**New**: Grunt collision radius = 35px

```python
# In is_move_safe():

if enemy.type == 'Grunt':
    collision_radius = 35  # Was 30px
```

**Rationale**:
- Simple fix for 30-39px Grunt collisions
- All 6 Grunt deaths were at 30-39px range
- 4 deaths < 35px suggest 30px insufficient
- 35px provides safety margin

### Change 5: Dynamic Dodge Distance Based on Bullet Count

**Current**: Dodge all bullets at 120px

**New**: Earlier dodging when multiple bullets present

```python
# In find_goal(), when checking for dodge threats:

nearby_bullet_count = sum(1 for e in self.enemies
                         if e.type in ['EnforcerBullet', 'TankShell', 'CruiseMissile']
                         and e.distance < 250)

if nearby_bullet_count >= 3:
    dodge_threshold = 150  # More aggressive with 3+ bullets
elif nearby_bullet_count >= 2:
    dodge_threshold = 135  # Slightly more aggressive with 2 bullets
else:
    dodge_threshold = 120  # Normal dodging with 1 bullet

# Check if bullet is within dynamic threshold
if bullet.distance < dodge_threshold:
    # Enter dodge mode
```

**Rationale**:
- Adapt dodge distance to threat level
- 3+ bullets: dodge at 150px (more time to react)
- 2 bullets: dodge at 135px
- 1 bullet: dodge at 120px (current)
- Should prevent getting overwhelmed by bullet swarms

## Expected Impact

### Primary Goals

**Reduce Trapped Deaths**: 24% → 10%
- Escape mode prevents getting boxed in
- Proactive escape when corridors < 3
- Better wall avoidance (150px threshold)

**Reduce Bullet Deaths**: 62% → 40-45%
- Dynamic dodge distance (up to 150px)
- Better wall avoidance
- Escape mode when overwhelmed

**Fix Grunt Collisions**: 21% → 15%
- Increase radius 30px → 35px
- Simple fix for prediction gap

### Expected Total Deaths

**Current**: 29 deaths per 10 episodes

**Target**: 22-24 deaths per 10 episodes (-20% to -25%)

**Breakdown**:
- Bullet deaths: 19 → 11-13 (reduce by ~35%)
- Trapped deaths: 7 → 3-4 (reduce by ~50%)
- Grunt deaths: 6 → 4-5 (reduce by ~20%)
- Other deaths: 4 → 4 (maintain)

### Secondary Effects

**Positive**:
- Should maintain high scores (still hunting spawners)
- Should maintain high levels (better survival)
- More strategic gameplay (escape when needed)

**Potential Trade-offs**:
- May take longer to reach goals (escaping instead)
- May miss some family members (escape > collect)
- Slightly more defensive playstyle

## Implementation Plan

### Step 1: Add Escape Corridor Detection
1. Implement `count_escape_corridors()` method
2. Implement `count_escape_corridors_from(x, y)` helper
3. Test with debug output

### Step 2: Add Escape Mode Logic
1. Add escape mode trigger (< 3 corridors)
2. Add move selection based on corridor maximization
3. Add debug output for escape mode activation

### Step 3: Increase Wall Avoidance
1. Change threshold from 100px → 150px
2. Change trigger from 2+ bullets → 1+ bullets
3. Increase penalty from 200 → 250

### Step 4: Fix Grunt Collision
1. Change Grunt radius from 30px → 35px
2. Simple one-line change

### Step 5: Dynamic Dodge Distance
1. Count nearby bullets (< 250px)
2. Set dodge threshold: 150px (3+), 135px (2), 120px (1)
3. Use dynamic threshold in dodge detection

## Testing Strategy

### Quick Test (5 Episodes)

**Goal**: Verify changes working without regressions

**Commands**:
```bash
poetry run python expert_fsm_v6.py --episodes 5 --config config.yaml --lives 3 --headless
```

**What to look for**:
1. "ESCAPE MODE" messages in output (should see when < 3 corridors)
2. Fewer "TRAPPED - no safe moves" in deaths
3. No Grunt deaths < 35px
4. Higher survival with similar scores

### Full Test (10 Episodes)

**Goal**: Measure iteration 6 effectiveness

**Commands**:
```bash
poetry run python analyze_deaths.py --episodes 10 --config config.yaml --lives 3 --output death_analysis_iter6.json
```

**Target Metrics**:
| Metric | Iter 5 | Iter 6 Target |
|--------|--------|---------------|
| Total deaths | 29 | 22-24 |
| Bullet deaths | 19 (62%) | 11-13 (40-45%) |
| Trapped deaths | 7 (24%) | 3-4 (10-15%) |
| Grunt deaths | 6 (21%) | 4-5 (15-18%) |
| Avg level | 6.7 | 7+ (maintain or improve) |
| Avg score | 104k | 100k+ (maintain) |

## Risk Assessment

### Low Risk Changes
- ✅ Grunt radius 30px → 35px (simple, well-understood)
- ✅ Wall threshold 100px → 150px (incremental improvement)

### Medium Risk Changes
- ⚠️ Dynamic dodge distance (may cause excessive dodging)
- ⚠️ Wall penalty trigger (1+ bullets vs 2+) (may be too defensive)

### High Risk Changes
- ⚠️⚠️ Escape mode system (NEW behavior, may interfere with goals)
- ⚠️⚠️ Escape corridor detection (complex logic, may have bugs)

### Mitigation Strategy

**If escape mode too defensive**:
- Increase trigger threshold (< 3 corridors → < 2 corridors)
- Only activate in high-threat situations (3+ bullets nearby)

**If dodge distance too aggressive**:
- Reduce 150px threshold to 140px
- Only use for 4+ bullets instead of 3+

**If scoring decreases significantly**:
- Disable escape mode for high-value goals (spawners with 1000 priority)
- Allow goal pursuit if goal priority > 800

## Code Changes Summary

**Files Modified**:
- `expert_fsm_v6.py`
  1. Add `count_escape_corridors()` method (~40 lines)
  2. Add `count_escape_corridors_from(x, y)` method (~30 lines)
  3. Add escape mode logic in `decide_action()` (~30 lines)
  4. Modify wall avoidance threshold and trigger (2 lines)
  5. Modify Grunt collision radius (1 line)
  6. Add dynamic dodge distance logic in `find_goal()` (~15 lines)

**Total Lines Changed**: ~120 lines (mostly additions)

## Comparison to Previous Approaches

### Iteration 5: Wall Avoidance + Shooter Priority
- **Focus**: Kill Enforcers faster, avoid walls when 2+ bullets
- **Result**: Bullet deaths 70% → 62% (progress but insufficient)
- **Limitation**: Reactive (avoids walls AFTER bullets present)

### Iteration 6: Escape Corridor System
- **Focus**: Proactive escape when corridors closing, better wall avoidance
- **Result**: Expected bullet deaths 62% → 40-45%
- **Innovation**: Proactive (escapes BEFORE getting trapped)

**Key Difference**: Iteration 5 was reactive (avoid walls when threatened), Iteration 6 is proactive (maintain escape routes at all times)

## Success Criteria

### Must Have (Minimum Viable)
1. ✅ Trapped deaths reduced by 50% (7 → 3-4)
2. ✅ Bullet deaths reduced by 25% (19 → 14-15)
3. ✅ Grunt deaths reduced by 15% (6 → 5)
4. ✅ No increase in other death types

### Should Have (Target)
1. ✅ Bullet deaths reduced by 35% (19 → 11-13)
2. ✅ Total deaths reduced by 20% (29 → 23-24)
3. ✅ Maintain average level 6.7+
4. ✅ Maintain average score 100k+

### Nice to Have (Stretch)
1. ✅ Bullet deaths < 40% (< 12 deaths)
2. ✅ Total deaths < 22
3. ✅ Average level 7+
4. ✅ Reach Level 15+ at least once

## Next Steps

1. ✅ **Design iteration 6** (this document)
2. ⏳ **Implement changes** in expert_fsm_v6.py
3. ⏳ **Quick test** (5 episodes)
4. ⏳ **Full test** (10 episodes death analysis)
5. ⏳ **Analyze results** and compare to iteration 5
6. ⏳ **Iterate** based on findings

## Conclusion

Iteration 6 introduces a **proactive escape system** to address the core issue: getting trapped by bullets.

**Key Innovations**:
1. **Escape corridors**: Track available escape routes (not just safe moves)
2. **Escape mode**: Prioritize survival when < 3 corridors
3. **Better wall avoidance**: 150px threshold, trigger with 1+ bullets
4. **Dynamic dodge distance**: 120-150px based on bullet count
5. **Grunt collision fix**: 30px → 35px radius

**Expected Outcome**: Reduce bullet deaths from 62% to 40-45%, reduce trapped deaths from 24% to 10%, while maintaining high scores and levels.

This approach shifts from reactive defense (iteration 5) to proactive escape planning (iteration 6).
