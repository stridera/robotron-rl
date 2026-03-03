# FSM v6: Iteration 6 - Implementation Summary

## Changes Implemented

### 1. Escape Corridor Detection (NEW)

**Added two methods**:

```python
def count_escape_corridors_from(self, x: float, y: float) -> int:
    """Count escape routes from a given position"""
    # Checks 8 directions for:
    # - Walls within 150px
    # - Bullets within 100px in that direction
    # Returns count of safe directions (0-8)

def count_escape_corridors(self) -> int:
    """Count escape routes from current position"""
    # Wrapper for current player position
```

**Purpose**: Proactive escape planning to avoid "TRAPPED - no safe moves" deaths (24% of iteration 5 deaths)

### 2. Escape Mode Logic (NEW)

**Trigger**: `len(safe_moves) < 4`

**Behavior**: When safe moves limited, prioritize escape over goal pursuit
- Find move that maximizes future escape corridors
- Move toward open space instead of pursuing goals
- Prevents getting boxed in

**Code**:
```python
elif len(safe_moves) < 4 and len(safe_moves) > 0:
    # Find move that maximizes future safe moves
    for move in safe_moves:
        future_corridors = self.count_escape_corridors_from(future_x, future_y)
        # Choose move with most escape corridors
```

**Purpose**: Proactively escape BEFORE getting trapped (not reactively)

### 3. Grunt Collision Radius Fix

**Change**: `Grunt: 30px → 35px`, `Prog: 30px → 35px`

**Reason**: Death analysis showed Grunt deaths at 30-39px range (4 deaths < 35px)

**Expected Impact**: Reduce Grunt deaths from 21% to 15%

### 4. More Aggressive Wall Avoidance

**Changes in dodge mode and safe alternative logic**:
- **Trigger**: `>= 1` bullets (was `>= 2`)
- **Threshold**: `150px` (was `100px`)
- **Penalty**: `250` (was `200`) in dodge mode, `200` (was `150`) in safe alternative

**Rationale**: 24% of deaths show "TRAPPED", need MORE wall avoidance

### 5. Dynamic Dodge Distance

**Added bullet count-based dodge threshold**:

```python
nearby_bullet_count = sum(1 for e in self.enemies
                         if e.type in ['EnforcerBullet', 'TankShell', 'CruiseMissile']
                         and e.distance < 250)

if nearby_bullet_count >= 3:
    dodge_threshold = 150  # Very conservative
elif nearby_bullet_count >= 2:
    dodge_threshold = 135  # Slightly conservative
else:
    dodge_threshold = 120  # Normal
```

**Rationale**: More bullets = dodge earlier to avoid getting overwhelmed

## Summary of All Changes

| Change | Before | After | Reason |
|--------|--------|-------|--------|
| **Grunt radius** | 30px | 35px | Deaths at 30-39px |
| **Wall threshold (dodge)** | 100px | 150px | Getting trapped at walls |
| **Wall trigger (dodge)** | >=2 bullets | >=1 bullets | More aggressive avoidance |
| **Wall penalty (dodge)** | -200 | -250 | Stronger deterrent |
| **Wall penalty (alternative)** | -150 | -200 | Stronger deterrent |
| **Dodge distance (3+ bullets)** | 120px | 150px | Earlier dodge when overwhelmed |
| **Dodge distance (2 bullets)** | 120px | 135px | Slightly earlier |
| **Dodge distance (0-1 bullets)** | 120px | 120px | Unchanged |
| **Escape mode** | None | <4 safe moves | Proactive escape |
| **Escape corridors** | None | New system | Maintain escape routes |

## Expected Impact

### Primary Goals

**Reduce Trapped Deaths**: 24% → 10%
- Escape mode prevents getting boxed in
- Escape corridor system maintains open routes
- Proactive instead of reactive

**Reduce Bullet Deaths**: 62% → 40-45%
- Dynamic dodge distance (up to 150px)
- More aggressive wall avoidance
- Escape mode when overwhelmed

**Fix Grunt Collisions**: 21% → 15%
- 35px radius (was 30px)
- Simple fix for prediction gap

### Expected Total Deaths

**Current (Iter 5)**: 29 deaths per 10 episodes

**Target (Iter 6)**: 22-24 deaths per 10 episodes (-20% to -25%)

**Breakdown**:
- Bullet deaths: 19 → 11-13 (reduce by ~35%)
- Trapped deaths: 7 → 3-4 (reduce by ~50%)
- Grunt deaths: 6 → 4-5 (reduce by ~20%)
- Other deaths: 4 → 4 (maintain)

## Testing Plan

### Quick Test (5 Episodes) - NEXT

**Commands**:
```bash
poetry run python expert_fsm_v6.py --episodes 5 --config config.yaml --lives 3 --headless
```

**What to look for**:
1. "ESCAPE MODE" messages in output
2. Fewer "TRAPPED - no safe moves" deaths
3. No Grunt deaths < 35px
4. Similar or better scores

### Full Test (10 Episodes)

**Commands**:
```bash
poetry run python analyze_deaths.py --episodes 10 --config config.yaml --lives 3 --output death_analysis_iter6.json
```

**Success Criteria**:
- Total deaths < 25
- Bullet deaths < 45% (down from 62%)
- Trapped deaths < 15% (down from 24%)
- Grunt deaths < 18% (down from 21%)

## Key Innovations

### 1. Escape Corridor System
- Not just "safe moves" (3-frame lookahead)
- But "escape corridors" (150px lookahead)
- Detects when getting boxed in BEFORE trapped

### 2. Proactive vs Reactive
- **Iteration 5**: Reactive (avoid walls when bullets present)
- **Iteration 6**: Proactive (maintain escape routes always)

### 3. Threat-Based Thresholds
- Dynamic dodge distance based on bullet count
- More conservative when overwhelmed
- Normal behavior when few threats

## Files Modified

- `expert_fsm_v6.py`:
  1. Added `count_escape_corridors_from()` method (~65 lines)
  2. Added `count_escape_corridors()` wrapper (~8 lines)
  3. Added escape mode logic in `decide_action()` (~20 lines)
  4. Modified Grunt/Prog collision radius (2 lines)
  5. Modified wall avoidance in dodge mode (3 lines changed)
  6. Modified wall avoidance in safe alternative (3 lines changed)
  7. Added dynamic dodge distance in `find_goal()` (~15 lines)

**Total Lines Changed**: ~115 lines (mostly additions)

## Risk Assessment

### Low Risk Changes ✅
- Grunt radius 30px → 35px (well-tested approach)
- Wall threshold 100px → 150px (incremental)

### Medium Risk Changes ⚠️
- Dynamic dodge distance (may be too conservative)
- Wall trigger (1+ vs 2+ bullets) (may be too defensive)

### High Risk Changes ⚠️⚠️
- Escape mode system (NEW behavior)
- Escape corridor detection (complex logic)

### Mitigation
If escape mode too defensive:
- Increase trigger threshold (< 4 → < 3 safe moves)
- Only activate when bullets nearby

If scores decrease:
- Disable escape mode for high-priority goals
- Allow goal pursuit if goal priority > 800

## Next Steps

1. ✅ **Implement iteration 6 changes** (COMPLETE)
2. ⏳ **Quick test (5 episodes)** - NEXT
3. ⏳ **Full death analysis (10 episodes)**
4. ⏳ **Analyze results**
5. ⏳ **Iterate based on findings**

## Conclusion

Iteration 6 introduces **proactive escape planning** to address getting trapped by bullets (24% of deaths).

**Core approach**: Maintain escape routes BEFORE getting trapped, not react AFTER.

**Key changes**:
1. Escape corridor detection (150px lookahead)
2. Escape mode when < 4 safe moves
3. More aggressive wall avoidance
4. Dynamic dodge distance based on threat level
5. Fixed Grunt collision radius

**Expected outcome**: Reduce total deaths from 29 to 22-24 (-20% to -25%), with bullet deaths dropping from 62% to 40-45%.
