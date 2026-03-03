# FSM v6: Goal-Oriented Planning with Entity Tracking

## Summary

FSM v6 is a **complete redesign** of the Robotron FSM with a fundamental shift in philosophy:

**v5 Philosophy**: Reactive - commit to ACTIONS (directions) for several frames
**v6 Philosophy**: Proactive - commit to GOALS (targets) and plan how to achieve them

## Core Insight

> "Since the environment waits until we choose a move before the next frame is selected, **we should never get hit**, right?"

This is the key realization: Robotron 2084 is **turn-based with perfect information**. We can:
- See all entity positions
- Predict their movement
- Test moves before making them
- **Never get hit if we check safety first**

## Major Improvements from v5

### 1. Entity Tracking
**Problem (v5)**: FSM shot at current enemy position, chased current enemy position
**Solution (v6)**: Track position history, calculate velocity, predict future positions

```python
class EntityTracker:
    def get_velocity(self, sprite: Sprite) -> Tuple[float, float]:
        """Calculate velocity from last 3-5 frames."""

    def predict_position(self, sprite: Sprite, frames_ahead: int) -> Tuple[float, float]:
        """Where will sprite be in N frames?"""
```

**Benefits**:
- Lead shots (shoot where enemy will be)
- Intercept paths (move to where enemy will be)
- Bullet dodge prediction (know bullet trajectories)

### 2. Collision Prediction
**Problem (v5)**: Sometimes ran into Hulks or got hit by bullets
**Solution (v6)**: Check if every move is safe before making it

```python
def is_move_safe(self, move_action: int, look_ahead_frames: int = 3) -> bool:
    """Check if a move will get us hit."""

def will_collide(self, my_future_pos, threat, frames_ahead) -> bool:
    """Check collision at each interpolated frame."""
```

**Benefits**:
- **Never get hit by Hulks** (immortal obstacles)
- **Never get hit by bullets** (predict trajectories)
- **Never get stuck** (always have safe moves)

### 3. Goal-Oriented Planning
**Problem (v5)**: Committed to actions (e.g., "move LEFT for 8 frames"), leading to stuck situations
**Solution (v6)**: Commit to goals (e.g., "kill this Grunt"), calculate how to achieve them each frame

```python
def find_goal(self) -> Optional[Sprite]:
    """Choose a goal to pursue (spawner, shooter, civilian, etc.)."""

def calculate_intercept_path(self, target: Sprite) -> Tuple[float, float]:
    """Where should we move to intercept target?"""
```

**Benefits**:
- Move with intention (intercept, not chase)
- Adapt to changing situations (re-plan each frame)
- No more getting stuck at walls

### 4. Lead Shots
**Problem (v5)**: Shot at current position, missed moving enemies
**Solution (v6)**: Calculate bullet travel time, shoot where enemy will be

```python
def calculate_lead_shot(self, target: Sprite) -> Optional[int]:
    """Shoot where enemy WILL BE, not where they ARE."""
    distance = target.distance
    travel_time = distance / BULLET_SPEED
    future_x, future_y = self.predict_position(target, travel_time)
    return self.can_shoot_at_position(future_x, future_y)
```

**Benefits**:
- Hit moving enemies more reliably
- Kill enemies faster
- More efficient ammo usage

### 5. Intercept Paths
**Problem (v5)**: Chased current enemy position (inefficient)
**Solution (v6)**: Move to where enemy will be (intercept)

```python
def calculate_intercept_path(self, target: Sprite) -> Tuple[float, float]:
    """Move to where target WILL BE."""
    frames_to_intercept = int(target.distance / PLAYER_SPEED)
    return self.predict_position(target, frames_to_intercept)
```

**Benefits**:
- Faster enemy kills (shorter path)
- More efficient movement
- Better civilian collection

## Architecture

### Decision Hierarchy

FSM v6 uses a strict priority hierarchy:

```
1. SAFETY CHECK: Will this move get us hit?
   └─> If no safe moves: STAY (trapped)
   └─> Otherwise: Continue to step 2

2. SHOOT: Can we shoot at any enemy? (with lead shots)
   └─> Fire opportunistically while moving

3. EXECUTE PLAN: Move toward goal
   └─> If goal is DODGE: Move away from immediate threat
   └─> If goal is ENEMY/CIVILIAN: Use intercept path
   └─> If desired move unsafe: Find closest safe alternative
   └─> If no goal: Patrol edges
```

**Key Difference from v5**: Safety check comes FIRST, always. We never make an unsafe move.

### Goal Priority System

```
Priority 1: DODGE immediate threats
  - Bullets within 100px (instant death)
  - Hulks within 50px (immortal obstacles)

Priority 2: Keep current goal if still valid
  - Target persistence prevents jittering

Priority 3: Collect civilians if safe
  - No dangerous enemies within 150px
  - OR civilian < 80px

Priority 4: Hunt dangerous enemies
  - Spawners (Brain, Sphereoid, Quark): +300 priority
  - Shooters (Enforcer, Tank): +200 priority
  - Grunts (chase aggressively): +50 priority
  - Base: 200 - distance

Priority 5: Any remaining civilians
```

### Entity Movement Patterns

Different entities have different movement patterns that we can exploit:

**Grunts**: Chase player directly
→ Velocity = direction toward player at fixed speed
→ Predictable - easy to intercept

**Bullets**: Straight line, constant speed
→ VERY predictable - perfect for dodging
→ High speed (8 px/frame) - dodge early!

**Hulks**: Wander randomly
→ Less predictable - use recent velocity for short-term prediction
→ Immortal - MUST dodge, cannot kill

**Electrodes**: Stationary
→ Zero velocity - easy to avoid
→ Can shoot if blocking path

**Spawners**: Mostly stationary
→ Minimal movement - easy to target
→ HIGH priority (create more enemies)

## Key Features

### EntityTracker Class

Tracks position history for all sprites:

```python
class EntityTracker:
    def __init__(self, history_size: int = 5):
        self.position_history: Dict[str, deque] = ...

    def update(self, sprites: List[Sprite]):
        """Update position history for all sprites."""

    def get_sprite_id(self, sprite: Sprite) -> str:
        """Match sprites across frames using type + grid position."""

    def get_velocity(self, sprite: Sprite) -> Tuple[float, float]:
        """Calculate velocity from last 3 frames (smoothing)."""

    def predict_position(self, sprite: Sprite, frames_ahead: int):
        """Linear prediction: pos + velocity * frames."""
```

**Implementation Details**:
- Uses 5-frame rolling window
- Smooths velocity over last 3 frames
- Matches sprites by type + 20px grid position
- Clamps predictions to board boundaries

### Collision Prediction

Checks every possible move for safety:

```python
def is_move_safe(self, move_action: int, look_ahead_frames: int = 3) -> bool:
    """
    Simulate move 3 frames ahead, check collisions.

    Returns:
        True if safe (no collisions predicted)
        False if unsafe (will get hit)
    """
```

**Checks**:
- Electrodes: Static distance check (< 20px = collision)
- Hulks: Trajectory prediction (interpolate both player and Hulk)
- Bullets: Trajectory prediction (CRITICAL - bullets move fast!)
- Regular enemies: Only unsafe if very close (< 30px)

**Interpolation**: Checks collision at each frame, not just endpoint:
```python
for frame in range(1, look_ahead_frames + 1):
    t = frame / look_ahead_frames
    my_x = player.x + (future_x - player.x) * t
    threat_x = threat.x + (threat_future_x - threat.x) * t
    # Check collision at this intermediate position
```

### Lead Shot Calculation

```python
def calculate_lead_shot(self, target: Sprite) -> Optional[int]:
    """
    1. Calculate bullet travel time: distance / BULLET_SPEED
    2. Predict where target will be at that time
    3. Check if we can shoot at that future position
    """
```

**Example**:
- Grunt at (200, 100), distance = 150px
- Grunt velocity = (3, -2) px/frame (chasing player)
- Bullet speed = 8 px/frame
- Travel time = 150/8 = 18.75 frames ≈ 19 frames
- Future position = (200 + 3*19, 100 + -2*19) = (257, 62)
- Shoot at (257, 62) instead of (200, 100)

### Intercept Path Calculation

```python
def calculate_intercept_path(self, target: Sprite) -> Tuple[float, float]:
    """
    1. Calculate time to intercept: distance / PLAYER_SPEED
    2. Predict where target will be at that time
    3. Move toward that future position
    """
```

**Example**:
- Grunt at (200, 100), distance = 100px
- Grunt velocity = (3, -2) px/frame
- Player speed = 5 px/frame
- Intercept time = 100/5 = 20 frames
- Future position = (200 + 3*20, 100 + -2*20) = (260, 60)
- Move toward (260, 60) instead of chasing (200, 100)

**Benefit**: Shorter path = faster kill

## Performance Characteristics

### Expected Improvements Over v5:

1. **Never get hit by bullets** - Bullet dodge prediction
2. **Never get hit by Hulks** - Collision prediction
3. **Faster enemy kills** - Lead shots + intercept paths
4. **Better survival** - Safety-first decision making
5. **No wall trapping** - Goal-oriented (no action commitment)

### Potential Challenges:

1. **Prediction accuracy** - Linear prediction may fail for complex paths
2. **Trapped situations** - May get cornered if all moves unsafe
3. **Computational cost** - More calculations per frame (but turn-based = plenty of time)

## Usage

### Run FSM v6 (Headless Testing):

```bash
# Test 10 episodes
poetry run python expert_fsm_v6.py --episodes 10 --config config.yaml --headless

# Test 50 episodes for statistics
poetry run python expert_fsm_v6.py --episodes 50 --config config.yaml --headless
```

### Watch FSM v6 (Visual):

```bash
# Normal speed
poetry run python watch_fsm.py --version 6 --fps 30

# God mode (test behavior without dying)
poetry run python watch_fsm.py --version 6 --fps 30 --godmode

# Start at higher level
poetry run python watch_fsm.py --version 6 --fps 30 --level 3
```

### Debug Output

Every 20 frames, FSM v6 prints:

```
[FSM v6] Player at (267, 151)
[FSM v6] ENEMY: Grunt at (168, 112) dist=106 v=[3.0,1.5]
[FSM v6] Enemies: Electrode(84), Grunt(106), Electrode(181), ...
[FSM v6] Safe moves: STAY, UP, UP_RIGHT, RIGHT, DOWN_RIGHT, DOWN, DOWN_LEFT, LEFT, UP_LEFT
[FSM v6] Move: LEFT - Intercept Grunt at (231,144) dist=106 v=[3.0,1.5]
[FSM v6] Fire: LEFT - Lead shot at Grunt (dist=106, v=[3.0,1.5])
```

**Key Info**:
- `v=[3.0,1.5]` - Entity velocity from tracker
- `Safe moves: ...` - List of moves that won't get us hit
- `Intercept Grunt at (231,144)` - Moving to where Grunt will be (not current position)
- `Lead shot at Grunt` - Shooting where Grunt will be when bullet arrives

## Code Structure

### Main Components

**ExpertFSMv6 class** (`expert_fsm_v6.py`):

```python
class ExpertFSMv6:
    def __init__(self):
        self.tracker = EntityTracker()  # NEW
        self.current_goal = None        # Changed from current_target
        self.current_goal_type = None   # 'enemy', 'civilian', or 'dodge'
        # No movement commitment - re-plan each frame

    def parse_sprites(self, sprite_data):
        """Parse sprites and update entity tracker."""

    def is_move_safe(self, move_action):
        """NEW: Check if move will get us hit."""

    def find_safe_moves(self):
        """NEW: Get list of all safe moves."""

    def calculate_lead_shot(self, target):
        """NEW: Shoot where enemy will be."""

    def calculate_intercept_path(self, target):
        """NEW: Move to where enemy will be."""

    def find_goal(self):
        """Choose goal with priority system."""

    def decide_action(self, sprite_data):
        """Main decision loop with safety-first approach."""
```

### Constants

```python
COLLISION_RADIUS = 20   # Distance considered a collision
PLAYER_SPEED = 5        # Player movement speed
BULLET_SPEED = 8        # Bullet speed (fast!)
GRUNT_SPEED = 3         # Grunt chase speed
HULK_SPEED = 2          # Hulk wander speed
```

## Comparison: v5 vs v6

| Feature | FSM v5 | FSM v6 |
|---------|--------|--------|
| **Philosophy** | Reactive (respond to threats) | Proactive (plan ahead) |
| **Commitment** | Actions (directions for 8 frames) | Goals (targets until killed) |
| **Movement** | Chase current position | Intercept future position |
| **Shooting** | Aim at current position | Lead shots (predict) |
| **Safety** | Try to avoid (reactive) | Check before moving (proactive) |
| **Bullet Dodge** | Back up if close | Predict trajectory and dodge |
| **Hulk Handling** | Sometimes hit | Never hit (collision prediction) |
| **Wall Handling** | Redirect + commitment issues | No commitment issues |
| **Entity Tracking** | None | Full velocity tracking |
| **Collision Prediction** | None | Check all moves, all threats |

## Next Steps

### Potential Improvements:

1. **Path Planning** - Multi-step paths (A* algorithm)
2. **Group Prediction** - Predict when enemies will cluster
3. **Optimal Positioning** - Find positions that align multiple enemies
4. **Advanced Prediction** - Non-linear paths (enemies change direction)
5. **Escape Routes** - Always maintain a safe exit path
6. **Spawner Timing** - Kill spawners before they spawn

### Integration with RL:

FSM v6 can be used for:
- **Imitation Learning** - Collect expert demonstrations
- **Curriculum Learning** - Provide initial policy for RL
- **Verification** - Compare RL performance to expert FSM

## Files

- `expert_fsm_v6.py` - Main FSM v6 implementation
- `watch_fsm.py` - Visual debugging (updated for v6)
- `FSM_V6_ARCHITECTURE.md` - This document

## Conclusion

FSM v6 represents a **fundamental paradigm shift** from reactive to proactive play:

**v5**: "An enemy is close - back up!"
**v6**: "I see 5 enemies. This Grunt will be at (200,100) in 10 frames. I'll move to (180,100) to intercept and shoot him when he arrives. I've checked - this move won't get me hit by any bullets or Hulks."

The key insight: **Turn-based game with perfect information = We should never get hit**.

By checking safety FIRST and planning AHEAD, FSM v6 should achieve:
- Zero hits from bullets (perfect dodge)
- Zero hits from Hulks (perfect dodge)
- Faster enemy kills (lead shots + intercepts)
- Better survival overall

Ready for testing! 🚀
