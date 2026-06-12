"""
Brain4GymAdapter — wraps Brain4's committed-goal + danger-map strategy for the gym.

Brain4 (~/win/code/robotron/brain4.py) can't be imported directly because it depends
on Windows-only modules (game_state, xenia_memory, reviewer, jit_entity_reader).
This adapter provides compatible Entity/GameState dataclasses and wraps Brain4's
core decision classes (DangerMap, PathFinder, GoalSelector, ShootingSystem).

Architecture (from brain4.py):
  Strategic: GoalSelector picks KILL/COLLECT/ESCAPE/ORBIT goal, commits to it
  Tactical:  DangerMap + A* PathFinder route around projected threats
  Shooting:  ShootingSystem fires independently of movement (predictive aim)
  Local:     Small avoidance push for immediate threats blended onto path direction

Input:  info['data'] from gym — list of (pixel_x, pixel_y, sprite_type)
Output: (move_dir, shoot_dir) — integers 0-7 matching MultiDiscrete([8,8])
"""
import math
import sys
import os
import types
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ── Coordinate system ─────────────────────────────────────────────────────────
GYM_W, GYM_H = 665.0, 492.0
GX_MIN, GX_MAX = 5.0, 145.0
GY_MIN, GY_MAX = 15.0, 230.0
GX_MID = (GX_MIN + GX_MAX) / 2
GY_MID = (GY_MIN + GY_MAX) / 2
FIELD_W = GX_MAX - GX_MIN
FIELD_H = GY_MAX - GY_MIN

# ── Gym sprite type → brain4 label ────────────────────────────────────────────
GYM_TYPE_TO_LABEL = {
    'Grunt':          'G',
    'Electrode':      'E',
    'Hulk':           'H',
    'Sphereoid':      'S',
    'Quark':          'Q',
    'Brain':          'B',
    'Enforcer':       'F',
    'Tank':           'T',
    'Mommy':          'CW',
    'Daddy':          'CM',
    'Mikey':          'CC',
    'Prog':           'P',
    'CruiseMissile':  'MS',
    'EnforcerBullet': 'FB',
    'TankShell':      'TS',
    'Bullet':         None,  # player bullets excluded
}


# ── Compatible dataclasses (matching what brain4.py expects) ──────────────────

@dataclass
class Entity:
    """Minimal entity matching brain4.py's Entity interface."""
    slot: int
    label: str
    gx: float
    gy: float


@dataclass
class GameState:
    """Minimal game state matching brain4.py's GameState interface."""
    player_gx: float
    player_gy: float
    entities: list
    wave: int
    score: int
    lives: int
    timestamp: float
    outer_loop_count: int = 0
    in_game: bool = True


# ── Slot assignment (stable IDs across frames for velocity tracking) ──────────

class SlotAssigner:
    """Greedy nearest-neighbor slot assignment for stable velocity tracking."""
    MATCH_THRESH = 20.0

    def __init__(self):
        self._slots: dict = {}  # slot_id → (label, gx, gy)
        self._next_id = 0

    def reset(self):
        self._slots.clear()
        self._next_id = 0

    def assign(self, entities_by_label: dict) -> List[Entity]:
        """Assign stable slot IDs to entities. Returns list of Entity objects."""
        new_entities = []
        for label, positions in entities_by_label.items():
            for gx, gy in positions:
                new_entities.append((label, gx, gy))

        # Match to existing slots by nearest-neighbor
        used_slots = set()
        result = []
        unmatched = []

        for label, gx, gy in new_entities:
            best_slot = None
            best_dist = self.MATCH_THRESH
            for sid, (slbl, sgx, sgy) in self._slots.items():
                if sid in used_slots or slbl != label:
                    continue
                d = math.hypot(gx - sgx, gy - sgy)
                if d < best_dist:
                    best_dist = d
                    best_slot = sid
            if best_slot is not None:
                used_slots.add(best_slot)
                self._slots[best_slot] = (label, gx, gy)
                result.append(Entity(slot=best_slot, label=label, gx=gx, gy=gy))
            else:
                unmatched.append((label, gx, gy))

        # New slots for unmatched
        for label, gx, gy in unmatched:
            sid = self._next_id
            self._next_id += 1
            self._slots[sid] = (label, gx, gy)
            result.append(Entity(slot=sid, label=label, gx=gx, gy=gy))

        # Prune dead slots
        self._slots = {sid: v for sid, v in self._slots.items() if sid in used_slots or sid >= self._next_id - len(unmatched)}

        return result


# ── Import brain4.py core classes by injecting stub modules ───────────────────

def _import_brain4():
    """Import brain4.py's core classes by providing stub modules for its dependencies."""
    brain4_path = os.path.expanduser('~/win/code/robotron/brain4.py')
    if not os.path.exists(brain4_path):
        raise FileNotFoundError(f"brain4.py not found at {brain4_path}")

    # Create stub modules so brain4.py's imports don't fail
    game_state_stub = types.ModuleType('game_state')
    game_state_stub.GameStateReader = type('GameStateReader', (), {})
    game_state_stub.GameState = GameState
    game_state_stub.Entity = Entity

    xenia_memory_stub = types.ModuleType('xenia_memory')
    xenia_memory_stub.XeniaMemory = type('XeniaMemory', (), {})

    reviewer_stub = types.ModuleType('reviewer')
    reviewer_stub.ReviewCapture = type('ReviewCapture', (), {})
    reviewer_stub.SpriteCropCollector = type('SpriteCropCollector', (), {})

    jit_entity_reader_stub = types.ModuleType('jit_entity_reader')
    jit_entity_reader_stub.CIVILIAN_LABELS = frozenset({'CC', 'CW', 'CM'})

    # Inject stubs
    sys.modules['game_state'] = game_state_stub
    sys.modules['xenia_memory'] = xenia_memory_stub
    sys.modules['reviewer'] = reviewer_stub
    sys.modules['jit_entity_reader'] = jit_entity_reader_stub

    # Add brain4's directory to path
    brain4_dir = os.path.dirname(brain4_path)
    if brain4_dir not in sys.path:
        sys.path.insert(0, brain4_dir)

    # Import brain4 module
    import importlib.util
    spec = importlib.util.spec_from_file_location('brain4', brain4_path)
    brain4_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(brain4_mod)

    return brain4_mod


# Import on module load
_brain4 = _import_brain4()
Brain4 = _brain4.Brain4


# ── Direction conversion ──────────────────────────────────────────────────────

def _vector_to_gym_dir(dx: float, dy: float) -> int:
    """Convert game-unit direction vector to gym direction index 0-7.

    Gym uses: 0=UP, 1=UP_RIGHT, 2=RIGHT, ..., 7=UP_LEFT
    Game uses: +x=right, +y=down
    """
    if abs(dx) < 0.01 and abs(dy) < 0.01:
        return 2  # default: right
    angle = math.atan2(-dy, dx)  # negate dy because gym UP = -game_y
    # angle: 0=right, pi/2=up, -pi/2=down
    idx = round(angle / (math.pi / 4)) % 8
    # Map from math angle indices to gym dir indices
    # math: 0=R, 1=UR, 2=U, 3=UL, 4=L(±π), 5=DL, 6=D, 7=DR
    # gym:  0=U, 1=UR, 2=R, 3=DR, 4=D, 5=DL, 6=L, 7=UL
    return [2, 1, 0, 7, 6, 5, 4, 3][idx]


def _stick_to_gym_dir(sx: float, sy: float) -> int:
    """Convert stick values (right/up positive) to gym direction 0-7.

    Stick: sx>0=right, sy>0=up
    Gym:   0=UP, 1=UP_RIGHT, 2=RIGHT, ...
    """
    if abs(sx) < 0.01 and abs(sy) < 0.01:
        return 0  # default: up (will be overridden if no target)
    angle = math.atan2(sy, sx)
    idx = round(angle / (math.pi / 4)) % 8
    # math: 0=R, 1=UR, 2=U, 3=UL, 4=L, 5=DL, 6=D, 7=DR
    # gym:  0=U, 1=UR, 2=R, 3=DR, 4=D, 5=DL, 6=L, 7=UL
    return [2, 1, 0, 7, 6, 5, 4, 3][idx]


# ── Main adapter ──────────────────────────────────────────────────────────────

class Brain4GymAdapter:
    """
    Drives the gym env using Brain4's committed-goal + danger-map strategy.

    Same interface as Brain3GymAdapter:
        adapter = Brain4GymAdapter()
        adapter.reset()
        for each step:
            move_dir, shoot_dir = adapter.act(info)
            obs, reward, done, trunc, info = env.step([move_dir, shoot_dir])
    """

    def __init__(self):
        self.brain = Brain4()
        self.slot_assigner = SlotAssigner()
        self._t = 0.0
        self._dt = 1.0 / 15.0  # match frame_skip=4 at 60fps
        self._wave = 0
        self._score = 0
        self._lives = 3

    def reset(self):
        self.brain = Brain4()
        self.slot_assigner.reset()
        self._t = 0.0
        self._wave = 0
        self._score = 0
        self._lives = 3

    def act(self, info: dict) -> Tuple[int, int]:
        """
        Args:
            info: gymnasium step info dict with 'data' key containing
                  [(pixel_x, pixel_y, sprite_type), ...] and optionally
                  'level' (current wave number), 'score', 'lives'.

        Returns:
            (move_dir, shoot_dir) — integers 0-7 for MultiDiscrete([8,8])
        """
        self._t += self._dt
        wave = info.get('level', self._wave)
        if wave == 0:
            wave = 1
        score = info.get('score', self._score)
        lives = info.get('lives', self._lives)

        # ── Parse sprite data → game-unit entities ────────────────────────
        px, py = GX_MID, GY_MID
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

        # ── Assign stable slot IDs ────────────────────────────────────────
        entities = self.slot_assigner.assign(entities_by_label)

        # ── Detect wave change ────────────────────────────────────────────
        if wave != self._wave:
            self.brain.on_wave_change(wave)
            self._wave = wave

        self._score = score
        self._lives = lives

        # ── Build GameState ───────────────────────────────────────────────
        state = GameState(
            player_gx=px,
            player_gy=py,
            entities=entities,
            wave=wave,
            score=score,
            lives=lives,
            timestamp=self._t,
        )

        # ── Run Brain4.think() ────────────────────────────────────────────
        mx_stick, my_stick, sx_stick, sy_stick = self.brain.think(state)

        # ── Convert stick values to gym direction indices ─────────────────
        # Brain4.think() returns stick values: (move_x, move_y, fire_x, fire_y)
        # move stick: game_delta_to_stick(dx, dy) → (dx_norm, -dy_norm) i.e. right/up positive
        # fire stick: fire_dir_to_stick(fd) → same convention
        move_dir = _stick_to_gym_dir(mx_stick, my_stick)
        shoot_dir = _stick_to_gym_dir(sx_stick, sy_stick)

        return move_dir, shoot_dir


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Testing Brain4GymAdapter")
    adapter = Brain4GymAdapter()
    adapter.reset()

    # Fake gym info with some entities
    fake_info = {
        'data': [
            (332.5, 246.0, 'Player'),
            (100.0, 100.0, 'Grunt'),
            (200.0, 200.0, 'Grunt'),
            (500.0, 400.0, 'Electrode'),
            (600.0, 100.0, 'Sphereoid'),
        ],
        'level': 1,
        'score': 0,
        'lives': 3,
    }

    for i in range(20):
        move_dir, shoot_dir = adapter.act(fake_info)
        dir_names = ['U', 'UR', 'R', 'DR', 'D', 'DL', 'L', 'UL']
        print(f"  step {i:3d}: move={dir_names[move_dir]} shoot={dir_names[shoot_dir]}")

    print("\nBrain4GymAdapter test complete!")
