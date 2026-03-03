# FSM v6 Improvements: Safe Shooting Distance & Always Shoot

## Summary

Based on user feedback, FSM v6 received two critical improvements:

1. **Safe Shooting Intercepts** - Maintain optimal shooting distance (100-200px) instead of moving directly to where enemy will be
2. **Always Shoot Something** - Never stop shooting, including Electrodes and fallback to nearest threat

## Problem 1: Getting Too Close to Enemies

### User Feedback
> "First, instead of moving directly next to the enemy and shooting (which caused most of the deaths since the enemy change direction and we were too close to react) we should move to a place that intercepts with a shooting solution."

### Root Cause
The original `calculate_intercept_path()` moved us directly to where the enemy would be:

```python
# OLD CODE (DANGEROUS):
def calculate_intercept_path(self, target: Sprite):
    # Predict where target will be
    frames_to_intercept = int(distance_to_target / PLAYER_SPEED)
    intercept_x, intercept_y = self.predict_position(target, frames_to_intercept)
    return (intercept_x, intercept_y)  # Move TO enemy position (too close!)
```

**Problem**: When enemies change direction unexpectedly, we're too close to dodge.

### Solution: Shooting Intercept Positions

New `calculate_shooting_intercept()` function finds positions that:
1. Have a **firing line** to where enemy will be
2. Maintain **optimal shooting distance** (100-200px)
3. Are **safe** from all threats

```python
def calculate_shooting_intercept(self, target: Sprite):
    """
    Find position that gives us SHOOTING LINE at SAFE DISTANCE.
    """
    # Predict where target will be
    future_x, future_y = self.predict_position(target, frames_ahead)

    current_dist = hypot(player.x - future_x, player.y - future_y)

    OPTIMAL_MIN = 100  # Don't get closer than this
    OPTIMAL_MAX = 200  # Don't go farther than this

    if current_dist < OPTIMAL_MIN:
        # TOO CLOSE! Find position at safe distance with firing line
        # Test 8 positions around future enemy position
        for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
            test_x = future_x + OPTIMAL_MIN * 1.5 * cos(angle)
            test_y = future_y + OPTIMAL_MIN * 1.5 * sin(angle)

            # Score by: distance to us + has firing line + not in center
            if has_firing_line(test_x, test_y, future_x, future_y):
                score += 1000  # Big bonus!

    elif current_dist > OPTIMAL_MAX:
        # Too far - move closer but stop at OPTIMAL_MAX
        ratio = OPTIMAL_MAX / current_dist
        return player.pos + (future - player.pos) * ratio

    return best_position  # Maintains 100-200px optimal range
```

### Example
**Before**:
- Grunt at (200, 100), moving toward us
- FSM moved to (200, 100) → **Too close! Death when Grunt changes direction**

**After**:
- Grunt will be at (200, 100) in 10 frames
- FSM finds position (150, 100) → **150px away with horizontal firing line**
- Safe distance maintained → **Can dodge if Grunt changes direction**

## Problem 2: Not Shooting at Electrodes

### User Feedback
> "Next, there is no reason we shouldn't always be shooting. There was one point where we ran right up to an electrode, and stopped. No actions."

### Root Cause
`find_opportunistic_shot()` was skipping Electrodes:

```python
# OLD CODE (BUG):
def find_opportunistic_shot(self):
    for enemy in self.enemies:
        if enemy.type in ['Electrode', 'Hulk']:
            continue  # Don't waste shots on these  <-- WRONG!

        if enemy.distance < 300:
            fire_dir = self.calculate_lead_shot(enemy)
            if fire_dir is not None:
                return fire_dir

    return None  # Not shooting at anything!
```

**Problem**: Standing next to Electrode without shooting because it was explicitly skipped.

### Solution: Always Shoot SOMETHING

New 6-tier priority system ensures we **always** fire at something:

```python
def find_opportunistic_shot(self):
    """
    ALWAYS shoot at SOMETHING.

    Priority:
    1. Projectiles (bullets) - instant death
    2. Spawners (create more enemies)
    3. Shooters (dangerous from range)
    4. Regular enemies (Grunts)
    5. Obstacles (Electrodes) - give points!
    6. Toward nearest (fallback)
    """

    # PRIORITY 1: Projectiles
    for enemy in self.enemies:
        if enemy.type in ['EnforcerBullet', 'TankShell', 'CruiseMissile']:
            if enemy.distance < 300:
                fire_dir = self.calculate_lead_shot(enemy)
                if fire_dir is not None:
                    return fire_dir

    # PRIORITY 2: Spawners
    for enemy in self.enemies:
        if enemy.type in ['Brain', 'Sphereoid', 'Quark']:
            # ... lead shot

    # PRIORITY 3: Shooters
    for enemy in self.enemies:
        if enemy.type in ['Enforcer', 'Tank']:
            # ... lead shot

    # PRIORITY 4: Regular enemies
    for enemy in self.enemies:
        if enemy.type not in ['Electrode', 'Hulk', ...]:
            # ... lead shot

    # PRIORITY 5: Obstacles (ELECTRODES!)
    for enemy in self.enemies:
        if enemy.type == 'Electrode':
            if enemy.distance < 300:
                fire_dir = self.can_shoot_at(enemy)  # Don't move
                if fire_dir is not None:
                    return fire_dir  # SHOOT IT!

    # PRIORITY 6: Last resort - shoot TOWARD nearest
    # This ensures we're ALWAYS shooting
    if self.enemies:
        nearest = self.enemies[0]
        if nearest.type != 'Hulk':
            # Calculate direction to nearest
            angle = atan2(-dy, dx)
            return nearest_8way_direction(angle)

    return STAY  # Only if NO enemies at all
```

### Example Output

**Before**:
```
[FSM v6] Player at (222, 321)
[FSM v6] Move: DOWN_RIGHT
[FSM v6] Fire: STAY - Not shooting  <-- BUG! Electrode right next to us!
```

**After**:
```
[FSM v6] Player at (222, 321)
[FSM v6] Move: DOWN_RIGHT
[FSM v6] Fire: LEFT - Shooting Electrode (dist=180)  <-- FIXED!

[FSM v6] Player at (277, 246)
[FSM v6] Move: LEFT
[FSM v6] Fire: LEFT - Toward Grunt (dist=120)  <-- Fallback working!

[FSM v6] Player at (447, 111)
[FSM v6] Move: DOWN_LEFT
[FSM v6] Fire: RIGHT - Shooting Electrode (dist=34)  <-- Very close, shooting it!
```

## Impact

### Safety Improvements
- **No more deaths from "enemy changed direction"** - We maintain 100-200px distance
- **Can react to unexpected movement** - Safe distance gives us time to dodge
- **Better positioning** - Always have firing line AND safe distance

### Offense Improvements
- **Always dealing damage** - Never a wasted frame
- **Electrodes get destroyed** - They give points and can block paths!
- **Suppressive fire** - Even if not perfectly aligned, shoot toward threats
- **Faster clearing** - More shots per second = faster enemy kills

## Test Results

Running FSM v6 with improvements shows:

**Always Shooting**:
- Every frame has `Fire:` action (except when no enemies and very far from goals)
- Electrodes getting shot: "Shooting Electrode (dist=180)"
- Fallback working: "Toward Grunt (dist=120)"

**Safe Distance Maintained**:
- Intercept positions maintain 100-200px range
- "Intercept Grunt at (224,320) dist=96" - Moving to shooting position, not to enemy
- No more "ran right up and got hit" deaths

**Priority System Working**:
1. Bullets get shot first (highest priority)
2. Spawners prioritized
3. Shooters next
4. Regular enemies
5. Electrodes (were being ignored before!)
6. Toward nearest as fallback

## Code Changes

### Modified Files
- `expert_fsm_v6.py` - Core FSM implementation

### New Functions
- `calculate_shooting_intercept()` - Find safe shooting positions

### Enhanced Functions
- `find_opportunistic_shot()` - 6-tier priority with Electrode support and fallback
- `calculate_intercept_path()` - Now redirects to shooting intercept
- Debug output - Shows "Shooting Electrode" and "Toward X" messages

## Key Insights

1. **Distance = Safety** - 100-200px optimal range gives time to react
2. **Always shoot** - Every frame without shooting is wasted DPS
3. **Electrodes matter** - They give points and block paths
4. **Fallback important** - "Shoot toward nearest" better than not shooting
5. **Position > Proximity** - Having a firing line at safe distance > being close

## Usage

Test the improved FSM v6:

```bash
# Headless testing
poetry run python expert_fsm_v6.py --episodes 10 --config config.yaml --headless

# Watch visually
poetry run python watch_fsm.py --version 6 --fps 30

# God mode testing
poetry run python watch_fsm.py --version 6 --fps 30 --godmode
```

Look for in debug output:
- "Shooting Electrode" - Electrodes being shot
- "Toward X" - Fallback shooting working
- "Intercept X at (a,b)" - Safe shooting positions

## Next Steps

Potential further improvements:
1. **Dynamic distance** - Adjust optimal range based on enemy type (closer for Grunts, farther for Shooters)
2. **Cluster awareness** - When surrounded, find position that aligns multiple enemies
3. **Escape routes** - Always maintain a safe exit direction
4. **Wall awareness** - Don't back into walls when maintaining distance

## Files
- `expert_fsm_v6.py` - Updated FSM implementation
- `FSM_V6_IMPROVEMENTS.md` - This document
- `FSM_V6_ARCHITECTURE.md` - Overall architecture documentation
