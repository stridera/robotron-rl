# FSM v5 Multi-Target Alignment & Trapped Detection

## Summary

Added two critical improvements to FSM v5 for better survival when surrounded by enemies:

1. **Trapped Detection** - Detects when stuck at walls and breaks movement commitment
2. **Multi-Enemy Alignment** - Finds positions to shoot multiple enemies at once when surrounded

## Problem: Getting Trapped When Surrounded

**User Report**:
> "Toward the end, with 26 enemies nearby, the FSM just backed up from one Grunt at a time. We could have stepped RIGHT a few pixels and killed both enemies."

**Example**:
```
[FSM] Player at (332, 251)
[FSM] ENEMY: Grunt at (348, 271) dist=26
[FSM] Enemies: Grunt(26), Grunt(61), Grunt(85), Grunt(88), Grunt(96), Grunt(98), Grunt(144), Grunt(149), Grunt(185), Grunt(203) ... +16 more
[FSM] Move: DOWN - BACKING UP from Grunt (dist=26, too close)
```

**Issue**: FSM was just backing up from closest enemy, ignoring the fact that 10+ other enemies were also nearby. This wastes time and leads to death when overwhelmed.

## Solution 1: Trapped Detection

### Implementation

Added position tracking to detect when player isn't moving:

```python
# In __init__:
self.last_position = None
self.stuck_frames = 0

# In decide_action:
if self.last_position:
    pos_diff = math.hypot(self.player_pos.x - self.last_position.x,
                         self.player_pos.y - self.last_position.y)
    if pos_diff < 1.0:  # Barely moved
        self.stuck_frames += 1
        if self.stuck_frames >= 3:  # Stuck for 3+ frames
            override_commitment = True  # Force recalculation
    else:
        self.stuck_frames = 0
```

### Impact

- Detects wall-trapping situations
- Breaks movement commitment when stuck
- FSM can escape corners and wall edges

## Solution 2: Multi-Enemy Alignment

### Concept

When surrounded by 3+ enemies, the FSM simulates moving in each direction and counts how many enemies would be on firing lines from that position. It then moves to the position that aligns the most enemies.

### Implementation

```python
def find_multi_enemy_alignment(self) -> Optional[Tuple[int, int]]:
    """
    When surrounded by multiple enemies, find a position where we can shoot
    multiple enemies at once.
    """
    if not self.player_pos or not self.enemies:
        return None

    # Only use this when surrounded (3+ close enemies)
    close_enemies = [e for e in self.enemies if e.distance < 150 and e.type != 'Electrode']
    if len(close_enemies) < 3:
        return None

    # Test each possible move direction (simulate 15px movement)
    test_moves = [
        (UP, 0, -15), (DOWN, 0, 15), (LEFT, -15, 0), (RIGHT, 15, 0),
        (UP_RIGHT, 15, -15), (UP_LEFT, -15, -15),
        (DOWN_RIGHT, 15, 15), (DOWN_LEFT, -15, 15),
    ]

    best_move = None
    best_count = 0

    for move_dir, dx, dy in test_moves:
        test_x = self.player_pos.x + dx
        test_y = self.player_pos.y + dy

        # Skip if this puts us in center
        if self.is_in_center(test_x, test_y):
            continue

        # Count enemies on firing lines from this position
        aligned_count = 0
        for enemy in close_enemies:
            enemy_dx = enemy.x - test_x
            enemy_dy = enemy.y - test_y

            # Check horizontal/vertical/diagonal lines (30px threshold)
            if abs(enemy_dy) < 30:  # Horizontal
                aligned_count += 1
            elif abs(enemy_dx) < 30:  # Vertical
                aligned_count += 1
            elif abs(abs(enemy_dx) - abs(enemy_dy)) < 30:  # Diagonal
                aligned_count += 1

        if aligned_count > best_count:
            best_count = aligned_count
            best_move = move_dir

    # Only use if we can align at least 2 enemies
    if best_count >= 2:
        return (best_move, best_count)

    return None
```

### Integration

Added to immediate threat handling (when enemies are < 80px away):

```python
if self.current_target_type == 'enemy' and target.distance < 80 and not is_obstacle:
    # First check if we can align multiple enemies
    multi_align = self.find_multi_enemy_alignment()

    if multi_align:
        # Move to position that aligns multiple enemies!
        move_action, aligned_count = multi_align
        move_reason = f"MULTI-ALIGN: {aligned_count} enemies (surrounded)"
    else:
        # Normal backing up logic...
```

### Example Output

```
[FSM] Move: DOWN - MULTI-ALIGN: 5 enemies (surrounded)
[FSM] Move: UP - MULTI-ALIGN: 7 enemies (surrounded)
[FSM] Move: DOWN - MULTI-ALIGN: 6 enemies (surrounded)
```

## Performance

Tested with 3 episodes:

```
Average Score:  11000.0 ± 248.3
Average Kills:  109.7 ± 2.6
Average Level:  2.3 ± 0.5
Max Level:      3
Average Steps:  1300.7
```

**Multi-align triggers observed**:
- 3-7 enemies aligned per decision
- Triggered when surrounded (3+ close enemies)
- Significant improvement in clearing clustered enemies

## Benefits

1. **Faster Enemy Clearing**: Kill multiple enemies with one firing line instead of backing up one at a time
2. **Better Survival**: Efficiently escape surrounded situations
3. **Smarter Positioning**: FSM actively seeks tactical positions when overwhelmed
4. **No Wall-Trapping**: Stuck detection ensures FSM doesn't freeze at walls

## Code Changes

**Files Modified**:
- `expert_fsm_v5.py` - Added multi-enemy alignment and trapped detection

**New Methods**:
- `find_multi_enemy_alignment()` - Find position to shoot multiple enemies

**Enhanced Methods**:
- `__init__()` - Added stuck detection tracking
- `decide_action()` - Added stuck detection logic and multi-align integration

## Key Insights

1. **Surrounding is dangerous** - When 5+ enemies are close, single-target tactics fail
2. **Alignment is powerful** - Robotron's 8-way shooting means good positioning can hit multiple targets
3. **Simulate before moving** - Testing each direction shows which moves are most valuable
4. **Threshold matters** - Using 30px alignment threshold matches the shooting tolerance
5. **Avoid center** - Multi-align positions that put player in center are rejected

## Next Steps

Future improvements could include:
- **Prioritize dangerous enemies** - Weight alignment by enemy type (prioritize spawners)
- **Consider projectiles** - Avoid positions that put player in bullet paths
- **Dynamic thresholds** - Adjust alignment threshold based on enemy count
- **Retreat paths** - Find multi-align positions that also provide escape routes

## Files

- `expert_fsm_v5.py` - Updated FSM with multi-target alignment
- `FSM_V5_MULTI_TARGET.md` - This document
