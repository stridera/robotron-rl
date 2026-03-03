# FSM Development Progress Summary

## What We've Accomplished

### 1. Diagnostic Infrastructure ✅
- Created `diagnose_fsm.py` - comprehensive FSM analysis tool
- Tracks 15+ metrics per frame (positioning, threats, decision quality)
- Identifies death causes and failure patterns
- Provides actionable recommendations

### 2. Expert FSM v1 ✅
- Created `expert_fsm.py` with human expert strategies:
  - Edge circling (avoid center)
  - Enemy prioritization (spawners > shooters > regular)
  - Projectile dodging
  - Family collection when safe
  - Kiting behavior
- Performance: **Level 0.9 average, Level 1 max**

### 3. Parameter Sweep ✅
- Created `test_fsm_variants.py` - automated A/B testing
- Tested 5 parameter configurations (50 episodes total)
- Found optimal: **v4_aggressive** (shorter distances, more killing)
- Results: Level 0.9 avg, 40.7% time in danger zone, 3485 avg score

### 4. Root Cause Identified ✅
**Primary failure mode: "Trapped in center" (80% of deaths)**

The FSM is:
- Getting pulled into center by enemies
- Not recognizing enemy clusters early enough
- Dying when surrounded (5+ enemies within 120px)
- Not eliminating spawners fast enough

---

## Current Performance

| Metric | Value |
|--------|-------|
| Average Level | 0.9 |
| Max Level | 1 |
| Average Score | 3485 |
| Average Survival | 1580 steps |
| % Time in Danger Zone | 40.7% |
| Primary Death Cause | Trapped in center (80%) |

**Comparison to Goal:**
- Current: Level 1
- RL baseline: Level 1.4
- Goal: Level 40
- Gap: 39 levels

---

## Why FSM Is Failing

### Problem 1: No Cluster Avoidance
FSM treats each enemy independently. Doesn't recognize when 10 enemies are forming a cluster in center. Walks into clusters trying to shoot individuals.

### Problem 2: Reactive, Not Predictive
Only reacts to current positions. Doesn't predict:
- Where enemies will converge
- Projectile trajectories
- Safe vs dangerous paths

### Problem 3: Poor Spawner Elimination
Spawners (Brain, Sphereoid, Quark) create endless enemies. FSM doesn't aggressively hunt them down. Result: enemy count grows until overwhelmed.

### Problem 4: No Safe Zone Management
Doesn't maintain knowledge of "safe corners" or escape routes. Gets boxed into corners with no exit.

### Problem 5: Indecisive Movement
Changes direction frequently (jittery). Gets caught between "retreat from enemy A" and "retreat from enemy B", ends up staying in place.

---

## Next Steps: Advanced Tactics

### Phase 2A: Cluster Avoidance (HIGH PRIORITY)

**Implementation:**
```python
def find_enemy_clusters(enemies, cluster_radius=80):
    """Group enemies within cluster_radius of each other."""
    clusters = []
    for enemy in enemies:
        # Find all enemies near this one
        nearby = [e for e in enemies if distance(enemy, e) < cluster_radius]
        if len(nearby) >= 3:  # 3+ enemies = cluster
            cluster_center = average_position(nearby)
            cluster_danger = len(nearby) * 10  # Danger score
            clusters.append((cluster_center, cluster_danger))
    return clusters

def avoid_clusters(player_pos, clusters):
    """Never path through clusters."""
    for cluster_center, danger in clusters:
        dist_to_cluster = distance(player_pos, cluster_center)
        if dist_to_cluster < 150:  # Approaching cluster
            return get_direction_away(cluster_center)
    return None
```

**Expected improvement:** Reduce "trapped in center" deaths from 80% to 40%

### Phase 2B: Spawner Hunting Mode (HIGH PRIORITY)

**Implementation:**
```python
def enter_spawner_hunt_mode(spawners, enemies):
    """Focus on eliminating spawners."""
    if not spawners:
        return None

    # Prioritize nearest spawner
    nearest_spawner = min(spawners, key=lambda s: s.distance)

    # Check if path is clear
    enemies_between = count_enemies_between(player, nearest_spawner)

    if enemies_between < 3:
        # Path is clearish - hunt the spawner
        return {
            'mode': 'hunt_spawner',
            'target': nearest_spawner,
            'move': toward(nearest_spawner),
            'fire': toward(nearest_spawner),
        }
    else:
        # Too many enemies - clear path first
        return {
            'mode': 'clear_path',
            'target': nearest_enemy_blocking_spawner,
            'move': kite_position(),
            'fire': toward(blocking_enemy),
        }
```

**Expected improvement:** Clear levels faster, prevent enemy swarms

### Phase 2C: Safe Zone Navigation (MEDIUM PRIORITY)

**Implementation:**
```python
SAFE_ZONES = [
    (50, 50),      # Top-left corner
    (615, 50),     # Top-right corner
    (50, 442),     # Bottom-left corner
    (615, 442),    # Bottom-right corner
]

def get_nearest_safe_zone():
    """Find safest corner."""
    safe_scores = []
    for zone in SAFE_ZONES:
        # Score = distance to enemies in that area
        enemies_near_zone = count_enemies_within(zone, radius=150)
        score = -enemies_near_zone  # Fewer enemies = higher score
        safe_scores.append((zone, score))

    return max(safe_scores, key=lambda x: x[1])[0]

def retreat_to_safe_zone():
    """Navigate to safest corner when overwhelmed."""
    safe_zone = get_nearest_safe_zone()
    return navigate_to(safe_zone, avoid_enemies=True)
```

**Expected improvement:** Better survival when overwhelmed

### Phase 2D: Predictive Movement (MEDIUM PRIORITY)

**Implementation:**
```python
def predict_enemy_position(enemy, timesteps=15):
    """Predict where enemy will be in N frames."""
    if enemy.type == 'Grunt':
        # Grunts move toward player at speed 7
        angle_to_player = get_angle(enemy, player)
        predicted_x = enemy.x + timesteps * 7 * cos(angle_to_player)
        predicted_y = enemy.y + timesteps * 7 * sin(angle_to_player)
        return (predicted_x, predicted_y)

    elif enemy.type == 'Enforcer':
        # Enforcers chase at max_speed 20
        # (Similar calculation)
        pass

    else:
        # Unknown movement - assume stays in place
        return (enemy.x, enemy.y)

def avoid_future_position(enemies):
    """Avoid where enemies will be, not just where they are."""
    future_positions = [predict_enemy_position(e) for e in enemies]

    # Check if current path intersects future positions
    for future_pos in future_positions:
        if will_collide_with(future_pos):
            return adjust_path_to_avoid(future_pos)

    return None
```

**Expected improvement:** Stop walking into enemies, better dodging

### Phase 2E: Decisiveness (LOW PRIORITY)

**Implementation:**
```python
def make_decisive_move(current_direction, new_direction):
    """Reduce jittery movement - commit to a direction for several frames."""

    # Hysteresis: Only change direction if significantly different
    angle_diff = abs(current_direction - new_direction)

    if angle_diff < 45:  # Less than 45 degrees
        return current_direction  # Keep current direction

    return new_direction  # Big change - commit to new direction
```

**Expected improvement:** Smoother movement, less getting stuck

---

## Implementation Priority

### Week 1 Goals (Reach Level 5-10)

**Day 1-2: Cluster Avoidance** (8 hours)
- Implement cluster detection
- Add "never path through clusters" rule
- Test: Should reduce center deaths significantly
- **Target: Level 3-5, avoid 60% of center deaths**

**Day 3-4: Spawner Hunting** (8 hours)
- Implement spawner hunting mode
- Test on levels with brains/sphereoids
- **Target: Level 5-8, faster level clears**

**Day 5: Safe Zone Navigation** (4 hours)
- Add safe zone retreat logic
- **Target: Level 8-10, better survival when overwhelmed**

**Day 6-7: Integration & Testing** (8 hours)
- Combine all tactics
- Run 100-episode test
- **Target: Level 10+ reached, ready for IL**

### Week 2 Goals (Imitation Learning)

**Day 8-9: Demonstration Collection** (4 hours)
- Run FSM for 1000 episodes
- Collect (observation, action) pairs
- **Target: 500k+ expert demonstrations**

**Day 10-12: Behavior Cloning** (12 hours)
- Train RL policy to imitate FSM
- **Target: RL reaches level 5-8 from BC alone**

**Day 13-14: RL Fine-Tuning** (10 hours)
- Train with PPO starting from BC policy
- **Target: RL reaches level 15-20**

---

## Success Metrics

### Milestone 1: FSM Reaches Level 5 (Week 1, Day 4)
- ✅ Demonstrates cluster avoidance working
- ✅ Spawner hunting functional
- ✅ Ready for initial demonstrations

### Milestone 2: FSM Reaches Level 10 (Week 1, Day 7)
- ✅ Strong expert baseline
- ✅ All advanced tactics integrated
- ✅ Ready for serious IL

### Milestone 3: RL Reaches Level 15 (Week 2, Day 14)
- ✅ IL + RL fine-tuning working
- ✅ Policy performs beyond FSM
- ✅ On track for level 40

### Final Goal: Level 40 (Week 3-4)
- Extended RL training (50M+ steps)
- Possible hierarchical RL
- Human demonstrations if needed

---

## Files Created

1. `expert_fsm.py` - Initial FSM with expert strategies
2. `diagnose_fsm.py` - Diagnostic tool for analyzing FSM failures
3. `test_fsm_variants.py` - Parameter sweep and A/B testing
4. `FSM_IMPROVEMENT_PLAN.md` - Complete roadmap
5. `FSM_PROGRESS_SUMMARY.md` - This document
6. `fsm_variants_results.txt` - Test results

---

## Next Action

**Implement cluster avoidance** (highest priority, biggest impact)

Create `expert_fsm_v2.py` with:
1. Cluster detection algorithm
2. "Never path through clusters" rule
3. Retreat to safe zones when clusters form

**Expected outcome after implementation:**
- Level 3-5 consistently
- "Trapped in center" deaths drop from 80% → 30%
- Ready for spawner hunting implementation

**Time estimate:** 2-4 hours to implement + test

---

## Why This Approach Will Work

1. **Incremental improvements** - Each tactic addresses specific failure
2. **Measurable progress** - Can A/B test each improvement
3. **Proven in humans** - These are tactics human players use
4. **Perfect information** - Simulator gives us everything we need
5. **Imitation learning boost** - Even level 10 FSM gives RL huge head start

The path is clear. Next session: implement cluster avoidance and watch performance jump!
