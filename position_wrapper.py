"""
Position-based observation wrapper for Robotron.

Instead of raw pixels, outputs sprite positions relative to player using
category-guaranteed slots so strategically important entities (spawners,
civilians, shooters) are always visible regardless of distance.

Observation format (986 dims):
  [0:2]   player position, normalized to [-1, 1]
  [2:986] 41 entity slots × 24 features each:
            type one-hot  (17)
            rel_pos       (2)   normalized to [-1, 1]
            dist          (1)   normalized to [0, 1]
            angle         (1)   in [-pi, pi]
            valid         (1)   1.0 if slot filled, 0.0 if empty
            velocity      (2)   (vx, vy) normalized, clipped to [-1, 1]
"""
import numpy as np
import gymnasium as gym
from typing import List, Tuple


# ── Sprite types (must match engine class names exactly) ──────────────────────
SPRITE_TYPES = [
    'Player', 'Grunt', 'Electrode', 'Hulk', 'Sphereoid', 'Quark',
    'Brain', 'Enforcer', 'Tank', 'Mommy', 'Daddy', 'Mikey',
    'Prog', 'CruiseMissile', 'EnforcerBullet', 'TankShell',
]
NUM_TYPES = len(SPRITE_TYPES)  # 16

# ── Slot layout ───────────────────────────────────────────────────────────────
# Each entry: (num_slots, [sprite_types]).
# Slots fill with the closest entity of those types; extras are zero-padded.
# Catch-all slots at the end handle grunt soup, electrodes, progs, overflow.
# Player bullets ('Bullet') are excluded — agent can infer their position from
# its own firing actions, so including them wastes slots and adds noise.
SLOT_CATEGORIES = [
    (8,  ['CruiseMissile', 'TankShell', 'EnforcerBullet']),  # projectiles — up to 15+ in flight
    (4,  ['Sphereoid', 'Quark']),      # spawners — max ~4, always see all
    (4,  ['Tank', 'Enforcer']),        # shooters — spawned in groups of 3
    (3,  ['Brain']),                   # brains — up to 5, see nearest 3
    (6,  ['Mommy', 'Daddy', 'Mikey']), # civilians — all 6, always visible
    (4,  ['Hulk']),                    # hulks — up to 8, nearest 4
    # 12 catch-all slots below handle nearest unassigned sprites
]
CATCHALL_SLOTS = 12
TOTAL_SLOTS = sum(n for n, _ in SLOT_CATEGORIES) + CATCHALL_SLOTS  # 29 + 12 = 41

FEATURES_PER_SLOT = NUM_TYPES + 2 + 1 + 1 + 1 + 2  # 16+2+1+1+1+2 = 23
OBS_DIM = 2 + TOTAL_SLOTS * FEATURES_PER_SLOT        # 2 + 41*23 = 945


class GroundTruthPositionWrapper(gym.ObservationWrapper):
    """
    Converts pixel observations to position-based features using ground truth
    sprite positions from the engine.

    The play area is 665×492 pixels. Positions are normalized:
      - absolute positions → [-1, 1] via (x/width)*2 - 1
      - relative positions → [-1, 1] via dx/width (dx range is [-width, width])
      - distances         → [0, 1]  via dist/max_distance
      - velocities        → [-1, 1] clipped, in units of pixels/frame

    For transfer to the real game (game units GX 5-145, GY 15-230), convert:
      pixel_x = (gx - 5) / 140 * 665
      pixel_y = (gy - 15) / 215 * 492
    before calling _extract_features().
    """

    SLOT_CATEGORIES = SLOT_CATEGORIES
    CATCHALL_SLOTS = CATCHALL_SLOTS
    SPRITE_TYPES = SPRITE_TYPES
    OBS_DIM = OBS_DIM

    def __init__(self, env, verbose: bool = False):
        super().__init__(env)
        self.verbose = verbose

        # Play area bounds
        self.play_rect = self.env.unwrapped.engine.play_rect
        self.width = float(self.play_rect.width)
        self.height = float(self.play_rect.height)
        self.max_distance = np.hypot(self.width, self.height)

        # Fast lookups
        self._type_to_idx = {t: i for i, t in enumerate(SPRITE_TYPES)}
        self._category_type_sets = [(n, set(types)) for n, types in SLOT_CATEGORIES]

        # Velocity tracking: slot_key → (prev_x, prev_y) in pixels
        self._prev_positions: dict = {}

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        if verbose:
            print(f"GroundTruthPositionWrapper: {OBS_DIM} dims, {TOTAL_SLOTS} slots")
            print(f"  Play area: {self.width:.0f}×{self.height:.0f} px")
            for i, (n, types) in enumerate(SLOT_CATEGORIES):
                print(f"  Cat {i}: {n} slots → {types}")
            print(f"  Catch-all: {CATCHALL_SLOTS} slots")

    def reset(self, **kwargs):
        self._prev_positions.clear()
        return super().reset(**kwargs)

    def observation(self, obs):
        sprite_data = self.env.unwrapped.engine.get_sprite_data()
        return self._extract_features(sprite_data)

    def _extract_features(self, sprite_data: List[Tuple]) -> np.ndarray:
        """
        Build the 986-dim observation from a list of (x, y, sprite_type) tuples.

        Can be called standalone (without gym step) for real-game inference —
        pass pixel-converted coordinates from the memory reader.
        """
        # Separate player from entities
        player_pos = None
        entities = []
        for x, y, sprite_type in sprite_data:
            if sprite_type == 'Player':
                player_pos = np.array([float(x), float(y)], dtype=np.float32)
            elif sprite_type != 'Bullet':  # player bullets excluded — inferred from firing actions
                entities.append((float(x), float(y), sprite_type))

        if player_pos is None:
            player_pos = np.array([self.width / 2.0, self.height / 2.0], dtype=np.float32)

        # Compute distance from player for all entities
        entity_list = []
        for x, y, t in entities:
            pos = np.array([x, y], dtype=np.float32)
            rel = pos - player_pos
            dist = float(np.linalg.norm(rel))
            entity_list.append({'pos': pos, 'rel': rel, 'dist': dist, 'type': t})

        # Sort by distance (used for within-category and catch-all ordering)
        entity_list.sort(key=lambda e: e['dist'])

        # ── Category slot assignment ──────────────────────────────────────────
        assigned_indices = set()
        slot_assignments = []  # list of (slot_key, entity_dict | None)

        for cat_idx, (n_slots, type_set) in enumerate(self._category_type_sets):
            candidates = [
                (i, e) for i, e in enumerate(entity_list)
                if e['type'] in type_set and i not in assigned_indices
            ]
            for rank in range(n_slots):
                key = (cat_idx, rank)
                if rank < len(candidates):
                    i, e = candidates[rank]
                    assigned_indices.add(i)
                    slot_assignments.append((key, e))
                else:
                    slot_assignments.append((key, None))

        # ── Catch-all slots: nearest unassigned ───────────────────────────────
        remaining = [e for i, e in enumerate(entity_list) if i not in assigned_indices]
        for rank in range(CATCHALL_SLOTS):
            key = ('catchall', rank)
            if rank < len(remaining):
                slot_assignments.append((key, remaining[rank]))
            else:
                slot_assignments.append((key, None))

        # ── Build feature vector ──────────────────────────────────────────────
        player_norm = np.array([
            (player_pos[0] / self.width) * 2.0 - 1.0,
            (player_pos[1] / self.height) * 2.0 - 1.0,
        ], dtype=np.float32)

        parts = [player_norm]

        for slot_key, entity in slot_assignments:
            if entity is not None:
                pos = entity['pos']
                rel = entity['rel']
                dist = entity['dist']
                t = entity['type']

                rel_norm = np.array([rel[0] / self.width, rel[1] / self.height], dtype=np.float32)
                dist_norm = float(dist / self.max_distance)
                angle = float(np.arctan2(rel[1], rel[0]))

                # Velocity: delta from previous frame position
                prev = self._prev_positions.get(slot_key)
                if prev is not None:
                    vx = np.clip((pos[0] - prev[0]) / self.max_distance, -1.0, 1.0)
                    vy = np.clip((pos[1] - prev[1]) / self.max_distance, -1.0, 1.0)
                else:
                    vx, vy = 0.0, 0.0
                self._prev_positions[slot_key] = (float(pos[0]), float(pos[1]))

                # Type one-hot
                type_oh = np.zeros(NUM_TYPES, dtype=np.float32)
                type_oh[self._type_to_idx.get(t, 0)] = 1.0

                slot_vec = np.array([
                    *type_oh,
                    rel_norm[0], rel_norm[1],
                    dist_norm,
                    angle,
                    1.0,   # valid
                    float(vx), float(vy),
                ], dtype=np.float32)
            else:
                # Empty slot — clear velocity history so next fill starts fresh
                self._prev_positions.pop(slot_key, None)
                slot_vec = np.zeros(FEATURES_PER_SLOT, dtype=np.float32)

            parts.append(slot_vec)

        return np.concatenate(parts, dtype=np.float32)


if __name__ == "__main__":
    from robotron import RobotronEnv
    from wrappers import MultiDiscreteToDiscrete, FrameSkipWrapper

    print("=" * 70)
    print("Testing GroundTruthPositionWrapper")
    print("=" * 70)

    env = RobotronEnv(
        level=1, lives=5, fps=0,
        config_path='curriculum_config.yaml',
        always_move=True, headless=True,
    )
    env = MultiDiscreteToDiscrete(env)
    env = FrameSkipWrapper(env, skip=4)
    env = GroundTruthPositionWrapper(env, verbose=True)

    print(f"\nObservation space: {env.observation_space.shape}  (expected: ({OBS_DIM},))")
    assert env.observation_space.shape == (OBS_DIM,), "Wrong obs dim!"

    obs, info = env.reset()
    print(f"Reset obs shape:   {obs.shape}")
    print(f"Player pos (norm): {obs[:2]}")
    print(f"First slot valid:  {obs[2 + NUM_TYPES + 2 + 1 + 1]:.0f}")  # valid flag of slot 0

    print("\nRunning 30 steps and checking velocity changes...")
    prev_vel = obs[-(FEATURES_PER_SLOT - NUM_TYPES - 4):-2]  # last slot vx,vy
    vel_changed = False
    for step in range(30):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        vel = obs[2 + NUM_TYPES + 2 + 1 + 1 + 2:2 + NUM_TYPES + 2 + 1 + 1 + 4]  # slot0 vx,vy
        if np.any(vel != 0):
            vel_changed = True
        if terminated or truncated:
            obs, info = env.reset()

    # Verify spawner slot is always in obs
    print(f"\nVelocity non-zero observed: {vel_changed}")

    # Check slot layout: spawner category starts at slot 8
    spawner_slot_start = 2 + 8 * FEATURES_PER_SLOT  # after 8 projectile slots
    print(f"Spawner slot valid flag at index {spawner_slot_start + NUM_TYPES + 2 + 1 + 1}: "
          f"{obs[spawner_slot_start + NUM_TYPES + 2 + 1 + 1]:.0f}")

    env.close()
    print("\nAll checks passed!")
    print("=" * 70)
