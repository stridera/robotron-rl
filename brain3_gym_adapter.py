"""
Brain3GymAdapter — re-implements Brain3's decision strategy for the gym environment.

Brain3 (~/win/code/robotron/brain3.py) can't be imported directly on Linux because it
depends on Windows-only memory APIs (XeniaMemory, ReadProcessMemory). This adapter
re-implements the same core strategy as a self-contained module.

Strategy (faithful to brain3):
  - EMERGENCY: flee from incoming bullets/projectiles
  - HUNT: orbit the field center while moving toward the priority target
           (spawners > shooters > brains > grunts)
  - COLLECT: collect civilians when the field is safe
  - Orbital motion blend: 30% circular + 70% tactical
  - Hulk/electrode repulsion steering
  - Priority-locked target selection with hysteresis

Input:  info['data'] from gym — list of (pixel_x, pixel_y, sprite_type)
Output: (move_dir, shoot_dir) — integers 0-7 matching MultiDiscrete([8,8])

Coordinate systems:
  Gym pixels: x ∈ [0, 665], y ∈ [0, 492]  (y increases downward)
  Game units: GX ∈ [5, 145], GY ∈ [15, 230] (y increases downward)
  Conversion: gx = 5 + x/665*140,  gy = 15 + y/492*215
"""
import math
import time
from collections import defaultdict, deque
from typing import List, Tuple, Optional

# ── Coordinate system ─────────────────────────────────────────────────────────
GYM_W, GYM_H = 665.0, 492.0
GX_MIN, GX_MAX = 5.0, 145.0
GY_MIN, GY_MAX = 15.0, 230.0
GX_MID = (GX_MIN + GX_MAX) / 2   # 75.0
GY_MID = (GY_MIN + GY_MAX) / 2   # 122.5
FIELD_W = GX_MAX - GX_MIN         # 140.0
FIELD_H = GY_MAX - GY_MIN         # 215.0

# ── Gym sprite type → Brain3 label ────────────────────────────────────────────
GYM_TYPE_TO_LABEL = {
    'Grunt':         'G',
    'Electrode':     'E',
    'Hulk':          'H',
    'Sphereoid':     'S',
    'Quark':         'Q',
    'Brain':         'B',
    'Enforcer':      'F',
    'Tank':          'T',
    'Mommy':         'CW',
    'Daddy':         'CM',
    'Mikey':         'CC',
    'Prog':          'P',
    'Cruise':        'MS',
    'EnforcerBullet':'FB',
    'TankShell':     'TS',
    'PlayerBullet':  None,  # skip our own bullets
}

CIVILIAN_LABELS = frozenset({'CC', 'CW', 'CM'})
SPAWNERS  = frozenset({'S', 'Q'})
SHOOTERS  = frozenset({'F', 'T'})
BULLETS   = frozenset({'FB', 'MS', 'TS'})
THREATS   = frozenset({'G', 'F', 'FB', 'B', 'S', 'Q', 'P', 'E', 'H', 'T', 'MS', 'TS'})
LETHAL    = frozenset({'G', 'F', 'FB', 'B', 'P', 'E', 'H', 'S', 'T', 'MS', 'TS'})
HUNTABLE  = frozenset({'S', 'Q', 'F', 'T', 'B', 'P', 'G', 'MS'})
SHOOTABLE = frozenset({'G', 'F', 'FB', 'B', 'S', 'Q', 'P', 'E', 'T', 'MS', 'TS'})

# Priority tiers — lower = higher priority
TIER = {'S': 1, 'Q': 1, 'F': 2, 'T': 2, 'MS': 2, 'TS': 2, 'B': 2, 'P': 4, 'G': 5}

# ── Tunable parameters (matches brain3 defaults) ──────────────────────────────
HULK_STEER_R       = 42.0
HULK_STEER_STR     = 1.2
ELEC_STEER_R       = 25.0
ELEC_STEER_STR     = 0.8
EMERGENCY_RADIUS   = 15.0
SHIELD_RADIUS      = 22.0
ORBIT_TARGET_R     = 60.0
ORBIT_BLEND        = 0.30
GRUNT_STANDOFF     = 22.0
ELEC_FIRE_R        = 22.0
COLLECT_SAFE_R     = 30.0
MIN_LOCK_FRAMES    = 20

_D = 1.0 / math.sqrt(2.0)

# ── Direction lookup: gym action index → (dx, dy) in game coords (right/down+) ──
# Gym action 0-7 map to engine dirs 1-8 (with always_move=True, action_mod=1)
# 0=UP, 1=UP-RIGHT, 2=RIGHT, 3=DOWN-RIGHT, 4=DOWN, 5=DOWN-LEFT, 6=LEFT, 7=UP-LEFT
GYM_DIR_VECTORS = [
    (0.0, -1.0),   # 0: UP
    (_D,  -_D),    # 1: UP-RIGHT
    (1.0,  0.0),   # 2: RIGHT
    (_D,   _D),    # 3: DOWN-RIGHT
    (0.0,  1.0),   # 4: DOWN
    (-_D,  _D),    # 5: DOWN-LEFT
    (-1.0, 0.0),   # 6: LEFT
    (-_D, -_D),    # 7: UP-LEFT
]


def _dist(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def _normalize(dx, dy):
    mag = math.sqrt(dx*dx + dy*dy)
    if mag < 0.001:
        return 0.0, 0.0
    return dx / mag, dy / mag

def _vector_to_gym_dir(dx, dy):
    """Convert a (dx, dy) game-space vector to a gym action index 0-7.
    Returns 0 (UP) if vector is near zero."""
    if abs(dx) < 0.001 and abs(dy) < 0.001:
        return 0
    # Stick space: right=+x, up=+y (negate dy for game→stick)
    sx, sy = _normalize(dx, -dy)
    angle = math.atan2(sy, sx)
    # Map angle to gym direction: UP=0 at angle=π/2, clockwise
    angle_from_up = (math.pi / 2 - angle) % (2 * math.pi)
    return int(round(angle_from_up / (math.pi / 4))) % 8

def _best_fire_dir(dx, dy):
    """Return gym fire direction 0-7 toward (dx, dy)."""
    return _vector_to_gym_dir(dx, dy)

def _clamp_field(x, y, margin=8.0):
    return (max(GX_MIN + margin, min(GX_MAX - margin, x)),
            max(GY_MIN + margin, min(GY_MAX - margin, y)))


# ── Minimal Entity container ──────────────────────────────────────────────────
class Entity:
    __slots__ = ('slot', 'label', 'gx', 'gy')

    def __init__(self, slot, label, gx, gy):
        self.slot  = slot
        self.label = label
        self.gx    = gx
        self.gy    = gy


# ── Velocity tracker (per slot, game units/sec) ───────────────────────────────
class VelocityTracker:
    HISTORY = 5

    def __init__(self):
        # slot → deque of (t, gx, gy)
        self._history: dict = defaultdict(lambda: deque(maxlen=self.HISTORY))
        self._vel: dict = {}

    def update(self, entities: List[Entity], t: float):
        seen = set()
        for e in entities:
            s = e.slot
            seen.add(s)
            hist = self._history[s]
            hist.append((t, e.gx, e.gy))
            if len(hist) >= 2:
                t0, x0, y0 = hist[0]
                t1, x1, y1 = hist[-1]
                dt = t1 - t0
                if dt > 0.001:
                    self._vel[s] = ((x1 - x0) / dt, (y1 - y0) / dt)
        # Clean up dead slots
        for s in list(self._history.keys()):
            if s not in seen:
                del self._history[s]
                self._vel.pop(s, None)

    def get_vel(self, entity: Entity):
        return self._vel.get(entity.slot, (0.0, 0.0))

    def get_speed(self, entity: Entity):
        vx, vy = self.get_vel(entity)
        return math.hypot(vx, vy)

    def check_bullet_intercept(self, bullets, px, py, dt=0.25):
        """True if any bullet is heading toward player and will be close within dt secs."""
        for b in bullets:
            vx, vy = self.get_vel(b)
            if abs(vx) < 0.1 and abs(vy) < 0.1:
                continue
            # Predicted position
            fx = b.gx + vx * dt
            fy = b.gy + vy * dt
            cur_d  = _dist(b.gx, b.gy, px, py)
            pred_d = _dist(fx, fy, px, py)
            if pred_d < cur_d and pred_d < EMERGENCY_RADIUS * 2:
                return True
        return False


# ── Slot ID assignment: stable IDs via nearest-neighbor cross-frame match ─────
class SlotAssigner:
    """Assigns synthetic stable slot IDs to gym entities via nearest-neighbor matching."""

    MATCH_THRESHOLD = 20.0  # max GU distance to consider same entity
    NEXT_ID_BASE = {label: i * 200 for i, label in enumerate(
        ['G', 'E', 'H', 'S', 'Q', 'B', 'F', 'T', 'CW', 'CM', 'CC', 'P', 'MS', 'FB', 'TS']
    )}

    def __init__(self):
        # label → {slot_id: (last_gx, last_gy)}
        self._prev: dict = defaultdict(dict)
        self._next_id: dict = defaultdict(int)

    def reset(self):
        self._prev.clear()
        self._next_id.clear()

    def assign(self, entities_by_label: dict) -> dict:
        """Assign slot IDs. Returns {label: [Entity, ...]}.

        Args:
            entities_by_label: {label: [(gx, gy), ...]}
        """
        result = defaultdict(list)

        for label, positions in entities_by_label.items():
            prev = self._prev[label]  # {slot_id: (gx, gy)}
            new_prev = {}

            # Greedy nearest-neighbor assignment
            unmatched_prev = set(prev.keys())
            used_positions = set()
            matched = {}  # pos_idx → slot_id

            for pos_idx, (gx, gy) in enumerate(positions):
                best_slot, best_d = None, self.MATCH_THRESHOLD + 1
                for slot_id in unmatched_prev:
                    pgx, pgy = prev[slot_id]
                    d = _dist(gx, gy, pgx, pgy)
                    if d < best_d:
                        best_d, best_slot = d, slot_id
                if best_slot is not None:
                    matched[pos_idx] = best_slot
                    unmatched_prev.discard(best_slot)

            # Create entities
            for pos_idx, (gx, gy) in enumerate(positions):
                if pos_idx in matched:
                    slot_id = matched[pos_idx]
                else:
                    # New entity — assign fresh ID
                    base = self.NEXT_ID_BASE.get(label, 1000)
                    slot_id = base + self._next_id[label]
                    self._next_id[label] += 1
                new_prev[slot_id] = (gx, gy)
                result[label].append(Entity(slot_id, label, gx, gy))

            self._prev[label] = new_prev

        return result


# ── Target manager with hysteresis ────────────────────────────────────────────
class TargetManager:
    def __init__(self):
        self.target: Optional[Entity] = None
        self._lock_frames = 0

    def update(self, entities: List[Entity], px, py) -> Optional[Entity]:
        candidates = [e for e in entities if e.label in HUNTABLE]
        if not candidates:
            self.target = None
            self._lock_frames = 0
            return None

        best_tier = min(TIER.get(e.label, 10) for e in candidates)
        same_tier = [e for e in candidates if TIER.get(e.label, 10) == best_tier]
        best = min(same_tier, key=lambda e: _dist(px, py, e.gx, e.gy))

        if self.target is None:
            self.target = best
            self._lock_frames = MIN_LOCK_FRAMES
            return self.target

        # Check if current target still alive
        cur_alive = any(e.slot == self.target.slot and e.label == self.target.label
                        for e in candidates)
        if not cur_alive:
            self.target = best
            self._lock_frames = MIN_LOCK_FRAMES
            return self.target

        # Hysteresis: only switch to lower tier or significantly closer target
        cur_tier = TIER.get(self.target.label, 10)
        if self._lock_frames > 0:
            self._lock_frames -= 1
        else:
            if best_tier < cur_tier:
                self.target = best
                self._lock_frames = MIN_LOCK_FRAMES
            elif best_tier == cur_tier:
                # Refresh target pointer (position updated)
                updated = next((e for e in candidates if e.slot == self.target.slot), None)
                if updated:
                    self.target = updated

        return self.target


# ── Main Brain3GymAdapter ─────────────────────────────────────────────────────
class Brain3GymAdapter:
    """
    Drives the gym env using Brain3's strategy.

    Usage:
        adapter = Brain3GymAdapter()
        adapter.reset()
        for each step:
            move_dir, shoot_dir = adapter.act(info)
            obs, reward, done, trunc, info = env.step([move_dir, shoot_dir])
    """

    def __init__(self):
        self.vel_tracker   = VelocityTracker()
        self.slot_assigner = SlotAssigner()
        self.target_mgr    = TargetManager()
        self._prev_fire_dir  = 0  # fire direction hysteresis
        self._fire_lock      = 0
        self._emergency_frames = 0
        self._spawn_frames   = 0
        self._t = 0.0
        self._dt = 1.0 / 30.0  # simulated tick rate

    def reset(self):
        self.vel_tracker   = VelocityTracker()
        self.slot_assigner.reset()
        self.target_mgr    = TargetManager()
        self._prev_fire_dir  = 0
        self._fire_lock      = 0
        self._emergency_frames = 0
        self._spawn_frames   = 0
        self._t = 0.0

    def act(self, info: dict) -> Tuple[int, int]:
        """
        Args:
            info: gymnasium step info dict with 'data' key containing
                  [(pixel_x, pixel_y, sprite_type), ...] and optionally
                  'level' (0-indexed wave number).

        Returns:
            (move_dir, shoot_dir) — integers 0-7 for MultiDiscrete([8,8])
        """
        self._t += self._dt
        self._spawn_frames += 1
        wave = info.get('level', 0) + 1

        # ── Parse sprite data → game-unit entities ────────────────────────────
        px, py = GX_MID, GY_MID  # fallback
        entities_by_label: dict = defaultdict(list)

        for pixel_x, pixel_y, sprite_type in info.get('data', []):
            label = GYM_TYPE_TO_LABEL.get(sprite_type)
            if label is None:
                continue
            gx = GX_MIN + (pixel_x / GYM_W) * FIELD_W
            gy = GY_MIN + (pixel_y / GYM_H) * FIELD_H
            if sprite_type == 'Player':
                px, py = gx, gy
            else:
                entities_by_label[label].append((gx, gy))

        # ── Assign stable slot IDs ─────────────────────────────────────────────
        assigned = self.slot_assigner.assign(entities_by_label)
        entities: List[Entity] = [e for group in assigned.values() for e in group]

        # ── Update velocity tracker ────────────────────────────────────────────
        self.vel_tracker.update(entities, self._t)

        # ── Classify ──────────────────────────────────────────────────────────
        hulks     = assigned.get('H', [])
        electrodes= assigned.get('E', [])
        grunts    = assigned.get('G', [])
        brains    = assigned.get('B', [])
        civilians = [e for lbl in CIVILIAN_LABELS for e in assigned.get(lbl, [])]
        bullets   = [e for lbl in BULLETS for e in assigned.get(lbl, [])]
        threats   = [e for e in entities if e.label in THREATS]

        # ── Emergency detection ────────────────────────────────────────────────
        bullet_incoming = self.vel_tracker.check_bullet_intercept(bullets, px, py)
        bullets_close   = [e for e in bullets if _dist(px, py, e.gx, e.gy) < EMERGENCY_RADIUS]

        if bullet_incoming or bullets_close:
            self._emergency_frames = max(self._emergency_frames, 6)
        if self._emergency_frames > 0:
            self._emergency_frames -= 1

        # ── Mode selection ─────────────────────────────────────────────────────
        if self._emergency_frames > 0:
            mode = 'EMERGENCY'
        else:
            mode = self._determine_mode(entities, px, py, wave)

        # ── Target selection ───────────────────────────────────────────────────
        target = self.target_mgr.update(entities, px, py)

        # ── Movement ──────────────────────────────────────────────────────────
        lethals_close = [e for e in threats
                         if e.label in LETHAL and _dist(px, py, e.gx, e.gy) < SHIELD_RADIUS]

        if mode == 'EMERGENCY':
            mx, my = self._move_emergency(px, py, lethals_close, hulks)
        elif mode == 'COLLECT':
            mx, my = self._move_collect(px, py, civilians, brains, hulks, electrodes)
        else:
            mx, my = self._move_hunt(px, py, target, hulks, electrodes, civilians, entities)

        # ── Obstacle steering ─────────────────────────────────────────────────
        for h in hulks:
            d = max(1.0, _dist(px, py, h.gx, h.gy))
            if d < HULK_STEER_R:
                s = HULK_STEER_STR * (1.0 - d / HULK_STEER_R)
                mx -= (h.gx - px) / d * s
                my -= (h.gy - py) / d * s
        for e in electrodes:
            d = max(1.0, _dist(px, py, e.gx, e.gy))
            if d < ELEC_STEER_R:
                s = ELEC_STEER_STR * (1.0 - d / ELEC_STEER_R)
                mx -= (e.gx - px) / d * s
                my -= (e.gy - py) / d * s
        for g in grunts:
            d = max(1.0, _dist(px, py, g.gx, g.gy))
            if d < GRUNT_STANDOFF:
                s = 2.5 * (1.0 - d / GRUNT_STANDOFF)
                mx -= (g.gx - px) / d * s
                my -= (g.gy - py) / d * s

        # Wall repulsion
        for wd, wx, wy in [
            (px - GX_MIN,  1.0,  0.0),
            (GX_MAX - px, -1.0,  0.0),
            (py - GY_MIN,  0.0,  1.0),
            (GY_MAX - py,  0.0, -1.0),
        ]:
            if wd < 10.0:
                ws = 20.0 * (1.0 - wd / 10.0)
            elif wd < 20.0:
                ws = 1.5 * (1.0 - wd / 20.0)
            else:
                ws = 0.0
            mx += wx * ws
            my += wy * ws

        mx, my = _normalize(mx, my)
        move_dir = _vector_to_gym_dir(mx, my)

        # ── Fire direction ─────────────────────────────────────────────────────
        shoot_dir = self._compute_fire(
            mode, px, py, entities, target, lethals_close, grunts, hulks, civilians, wave
        )

        return move_dir, shoot_dir

    # ── Mode determination ────────────────────────────────────────────────────

    def _determine_mode(self, entities, px, py, wave):
        huntable  = [e for e in entities if e.label in HUNTABLE]
        civilians = [e for e in entities if e.label in CIVILIAN_LABELS]

        if huntable and all(e.label in ('H', 'E') for e in huntable):
            # Only hulks/electrodes left — KITE / orbital
            return 'HUNT'

        if not civilians:
            return 'HUNT'

        # COLLECT if no dangerous threats are close and civilians exist
        dangerous_close = any(
            e.label in LETHAL and e.label not in ('E', 'H')
            and _dist(px, py, e.gx, e.gy) < COLLECT_SAFE_R
            for e in entities
        )
        if not dangerous_close and civilians:
            return 'COLLECT'

        return 'HUNT'

    # ── Movement modes ────────────────────────────────────────────────────────

    def _move_emergency(self, px, py, lethals_close, hulks):
        fx, fy = 0.0, 0.0
        for e in lethals_close:
            d = max(1.0, _dist(px, py, e.gx, e.gy))
            strength = 3.0 * (1.0 - d / (EMERGENCY_RADIUS * 3))
            fx -= (e.gx - px) / d * strength
            fy -= (e.gy - py) / d * strength
        # Orbital component: perpendicular to center
        dx_c = px - GX_MID
        dy_c = py - GY_MID
        # Perp CCW: (-dy, dx)
        ox, oy = _normalize(-dy_c, dx_c)
        fx += ox * 0.5
        fy += oy * 0.5
        return _normalize(fx, fy)

    def _move_hunt(self, px, py, target, hulks, electrodes, civilians, entities):
        # Orbital motion toward field center at ORBIT_TARGET_R
        dx_c, dy_c = GX_MID - px, GY_MID - py
        r = math.hypot(dx_c, dy_c)
        if r > 0.1:
            # Perp CCW for orbit
            perp_x, perp_y = _normalize(-dy_c, dx_c)
            # Radial: push toward/away from center to maintain orbit radius
            radial_x, radial_y = dx_c / r, dy_c / r
            r_err = r - ORBIT_TARGET_R
            orbit_x = perp_x + radial_x * (r_err / ORBIT_TARGET_R) * 0.5
            orbit_y = perp_y + radial_y * (r_err / ORBIT_TARGET_R) * 0.5
            ox, oy = _normalize(orbit_x, orbit_y)
        else:
            ox, oy = 1.0, 0.0

        # Tactical: move toward target
        if target is not None:
            tx, ty = _normalize(target.gx - px, target.gy - py)
        else:
            tx, ty = ox, oy

        mx = ORBIT_BLEND * ox + (1.0 - ORBIT_BLEND) * tx
        my = ORBIT_BLEND * oy + (1.0 - ORBIT_BLEND) * ty
        return _normalize(mx, my)

    def _move_collect(self, px, py, civilians, brains, hulks, electrodes):
        if not civilians:
            # Fall back to orbiting
            return self._move_hunt(px, py, None, hulks, electrodes, [], [])
        nearest = min(civilians, key=lambda c: _dist(px, py, c.gx, c.gy))
        dx, dy = _normalize(nearest.gx - px, nearest.gy - py)
        return dx, dy

    # ── Fire computation ──────────────────────────────────────────────────────

    def _compute_fire(self, mode, px, py, entities, target,
                      lethals_close, grunts, hulks, civilians, wave):
        # 1. Close shield: shoot nearest lethal if very close
        mobile_close = [e for e in lethals_close if e.label not in ('E',)]
        if mobile_close:
            nearest = min(mobile_close, key=lambda e: _dist(px, py, e.gx, e.gy))
            return _best_fire_dir(nearest.gx - px, nearest.gy - py)

        # 2. Electrodes in fire radius
        elec_near = [e for e in entities
                     if e.label == 'E' and _dist(px, py, e.gx, e.gy) < ELEC_FIRE_R]
        if elec_near:
            nearest = min(elec_near, key=lambda e: _dist(px, py, e.gx, e.gy))
            return _best_fire_dir(nearest.gx - px, nearest.gy - py)

        # 3. COLLECT mode: shoot toward nearest civilian (help clear path)
        if mode == 'COLLECT' and civilians:
            nearest_civ = min(civilians, key=lambda c: _dist(px, py, c.gx, c.gy))
            return _best_fire_dir(nearest_civ.gx - px, nearest_civ.gy - py)

        # 4. Fire at priority target
        if target is not None and target.label not in CIVILIAN_LABELS:
            return _best_fire_dir(target.gx - px, target.gy - py)

        # 5. Nearest shootable
        shootable = [e for e in entities if e.label in SHOOTABLE]
        if shootable:
            nearest = min(shootable, key=lambda e: _dist(px, py, e.gx, e.gy))
            return _best_fire_dir(nearest.gx - px, nearest.gy - py)

        # 6. No target: keep previous fire direction
        return self._prev_fire_dir


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    from robotron import RobotronEnv
    from wrappers import MultiDiscreteToDiscrete, FrameSkipWrapper
    from position_wrapper import GroundTruthPositionWrapper

    print("=" * 70)
    print("Testing Brain3GymAdapter")
    print("=" * 70)

    env = RobotronEnv(
        level=3, lives=5, fps=0,
        config_path='curriculum_config.yaml',
        always_move=True, headless=True,
    )
    env = MultiDiscreteToDiscrete(env)
    env = FrameSkipWrapper(env, skip=4)
    env = GroundTruthPositionWrapper(env)

    adapter = Brain3GymAdapter()

    total_episodes = 3
    for ep in range(total_episodes):
        obs, info = env.reset()
        adapter.reset()
        ep_reward = 0.0
        steps = 0
        done = False

        while not done:
            move_dir, shoot_dir = adapter.act(info)

            assert 0 <= move_dir <= 7, f"Bad move_dir: {move_dir}"
            assert 0 <= shoot_dir <= 7, f"Bad shoot_dir: {shoot_dir}"

            action = [move_dir, shoot_dir]
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1
            done = terminated or truncated

        score = info.get('score', 0)
        level = info.get('level', 0)
        print(f"  Episode {ep+1}: score={score}  level={level}  "
              f"steps={steps}  reward={ep_reward:.1f}")

    env.close()
    print("\nBrain3GymAdapter test complete!")
    print("=" * 70)
