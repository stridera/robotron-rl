"""
Position-based observation wrapper for Robotron.

Instead of raw pixels, outputs sprite positions relative to player.
Uses ground truth from engine.get_sprite_data() to test if position-based RL works.
"""
import numpy as np
import gymnasium as gym
from typing import List, Tuple, Dict


class GroundTruthPositionWrapper(gym.ObservationWrapper):
    """
    Converts pixel observations to position-based features using ground truth from engine.

    Observation format:
    - Player position: (2,) - [x, y] normalized to [-1, 1]
    - For each of 10 nearest sprites:
        - Type (10,) - one-hot encoded sprite type
        - Relative position (2,) - [dx, dy] normalized to [-1, 1]
        - Distance (1,) - normalized to [0, 1]
        - Angle (1,) - angle from player to sprite in [-pi, pi]
        - Valid flag (1,) - 1 if sprite exists, 0 if padding

    Total: 2 + 10 * (10 + 2 + 1 + 1 + 1) = 2 + 10 * 15 = 152 dims
    """

    # Sprite type encoding
    SPRITE_TYPES = [
        'Player', 'Grunt', 'Electrode', 'Hulk', 'Sphereoid', 'Quark',
        'Brain', 'Enforcer', 'Tank', 'Mommy', 'Daddy', 'Mikey',
        'Prog', 'Cruise', 'PlayerBullet', 'EnforcerBullet', 'TankShell'
    ]

    def __init__(self, env, max_sprites: int = 10, verbose: bool = False):
        """
        Args:
            env: Robotron environment
            max_sprites: Maximum number of sprites to include in observation
            verbose: Print debug info
        """
        super().__init__(env)
        self.max_sprites = max_sprites
        self.verbose = verbose

        # Get play area bounds for normalization
        self.play_rect = self.env.unwrapped.engine.play_rect
        self.width = self.play_rect.width
        self.height = self.play_rect.height
        self.max_distance = np.hypot(self.width, self.height)

        # Build type to index mapping
        self.type_to_idx = {t: i for i, t in enumerate(self.SPRITE_TYPES)}
        self.num_types = len(self.SPRITE_TYPES)

        # New observation space: position-based features
        # Player pos (2) + max_sprites * (type one-hot (num_types) + rel_pos (2) + dist (1) + angle (1) + valid (1))
        features_per_sprite = self.num_types + 2 + 1 + 1 + 1  # type, pos, dist, angle, valid
        obs_dim = 2 + self.max_sprites * features_per_sprite

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        if self.verbose:
            print(f"GroundTruthPositionWrapper initialized:")
            print(f"  Play area: {self.width}x{self.height}")
            print(f"  Max sprites: {self.max_sprites}")
            print(f"  Sprite types: {len(self.SPRITE_TYPES)}")
            print(f"  Observation dim: {obs_dim}")

    def observation(self, obs):
        """Convert pixel observation to position-based features."""
        # Get sprite data from engine
        sprite_data = self.env.unwrapped.engine.get_sprite_data()

        # Extract features
        features = self._extract_features(sprite_data)

        if self.verbose:
            player_pos = features[:2]
            print(f"Player: ({player_pos[0]:.2f}, {player_pos[1]:.2f}), Sprites: {len(sprite_data)}")

        return features

    def _extract_features(self, sprite_data: List[Tuple[int, int, str]]) -> np.ndarray:
        """
        Extract position-based features from sprite data.

        Args:
            sprite_data: List of (x, y, sprite_type) tuples

        Returns:
            Feature vector of shape (obs_dim,)
        """
        # Find player
        player_pos = None
        other_sprites = []

        for x, y, sprite_type in sprite_data:
            if sprite_type == 'Player':
                player_pos = np.array([x, y], dtype=np.float32)
            else:
                other_sprites.append((x, y, sprite_type))

        # If no player found (shouldn't happen), use center
        if player_pos is None:
            player_pos = np.array([self.width / 2, self.height / 2], dtype=np.float32)

        # Normalize player position to [-1, 1]
        player_pos_norm = self._normalize_position(player_pos)

        # Calculate features for each sprite
        sprite_features = []
        for x, y, sprite_type in other_sprites:
            pos = np.array([x, y], dtype=np.float32)

            # Relative position
            rel_pos = pos - player_pos
            rel_pos_norm = self._normalize_position(rel_pos)

            # Distance
            dist = np.linalg.norm(rel_pos)
            dist_norm = dist / self.max_distance

            # Angle (from player to sprite)
            angle = np.arctan2(rel_pos[1], rel_pos[0])

            # Type one-hot
            type_idx = self.type_to_idx.get(sprite_type, 0)
            type_onehot = np.zeros(self.num_types, dtype=np.float32)
            type_onehot[type_idx] = 1.0

            sprite_features.append({
                'pos': pos,
                'rel_pos_norm': rel_pos_norm,
                'dist': dist,
                'dist_norm': dist_norm,
                'angle': angle,
                'type_onehot': type_onehot,
            })

        # Sort by distance (nearest first)
        sprite_features.sort(key=lambda s: s['dist'])

        # Take only max_sprites nearest
        sprite_features = sprite_features[:self.max_sprites]

        # Build feature vector
        features = [player_pos_norm]

        for i in range(self.max_sprites):
            if i < len(sprite_features):
                s = sprite_features[i]
                # Concatenate: type_onehot, rel_pos, dist, angle, valid=1
                sprite_vec = np.concatenate([
                    s['type_onehot'],
                    s['rel_pos_norm'],
                    [s['dist_norm']],
                    [s['angle']],
                    [1.0]  # valid flag
                ])
            else:
                # Padding: all zeros except valid flag = 0
                sprite_vec = np.zeros(self.num_types + 2 + 1 + 1 + 1, dtype=np.float32)

            features.append(sprite_vec)

        # Flatten to single vector
        features = np.concatenate(features, dtype=np.float32)

        return features

    def _normalize_position(self, pos: np.ndarray) -> np.ndarray:
        """Normalize position to [-1, 1] range."""
        # Normalize x by width, y by height
        norm_x = (pos[0] / self.width) * 2 - 1
        norm_y = (pos[1] / self.height) * 2 - 1
        return np.array([norm_x, norm_y], dtype=np.float32)


if __name__ == "__main__":
    """Test the wrapper."""
    from robotron import RobotronEnv
    from wrappers import MultiDiscreteToDiscrete, FrameSkipWrapper

    print("="*80)
    print("Testing GroundTruthPositionWrapper")
    print("="*80)

    # Create environment
    env = RobotronEnv(
        level=1,
        lives=5,
        fps=0,
        config_path='ultra_simple_curriculum.yaml',
        always_move=True,
        headless=True
    )

    env = MultiDiscreteToDiscrete(env)
    env = FrameSkipWrapper(env, skip=4)
    env = GroundTruthPositionWrapper(env, max_sprites=10, verbose=True)

    print(f"\nObservation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Test a few steps
    print("\nRunning 10 steps...")
    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"First 10 values: {obs[:10]}")

    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Step {step}: reward={reward:.2f}, obs_shape={obs.shape}")

        if terminated or truncated:
            obs, info = env.reset()
            print("Episode ended, reset environment")

    env.close()

    print("\n" + "="*80)
    print("Test complete!")
    print("="*80)
