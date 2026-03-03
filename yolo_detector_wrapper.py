"""
YOLO11 Detector Wrapper for Robotron RL.

Integrates a pre-trained YOLO11 detector with the position-based RL policy.
Converts YOLO detections → position features → policy actions.
"""
import numpy as np
import gymnasium as gym
from typing import List, Tuple


class YOLODetectorWrapper(gym.ObservationWrapper):
    """
    Uses YOLO11 detector to convert pixel observations to position features.

    Compatible with position-based RL policy trained in Phase 1.5.
    """

    # Sprite type encoding (must match GroundTruthPositionWrapper)
    SPRITE_TYPES = [
        'Player', 'Grunt', 'Electrode', 'Hulk', 'Sphereoid', 'Quark',
        'Brain', 'Enforcer', 'Tank', 'Mommy', 'Daddy', 'Mikey',
        'Prog', 'Cruise', 'Bullet', 'EnforcerBullet', 'TankShell'
    ]

    def __init__(self, env, yolo_model_path: str, max_sprites: int = 20,
                 device: str = 'cuda', confidence_threshold: float = 0.5):
        """
        Args:
            env: Robotron environment
            yolo_model_path: Path to trained YOLO11 model (.pt file)
            max_sprites: Maximum sprites to include (matches training)
            device: 'cuda' or 'cpu'
            confidence_threshold: Minimum confidence for detections
        """
        super().__init__(env)
        self.max_sprites = max_sprites
        self.device = device
        self.confidence_threshold = confidence_threshold

        # Load YOLO11 model
        print(f"Loading YOLO11 model from {yolo_model_path}...")
        from ultralytics import YOLO
        self.yolo_model = YOLO(yolo_model_path)
        print(f"✅ YOLO11 model loaded on {device}")

        # Get play area bounds for normalization
        self.play_rect = self.env.unwrapped.engine.play_rect
        self.width = self.play_rect.width
        self.height = self.play_rect.height
        self.max_distance = np.hypot(self.width, self.height)

        # Build type to index mapping
        self.type_to_idx = {t: i for i, t in enumerate(self.SPRITE_TYPES)}
        self.num_types = len(self.SPRITE_TYPES)

        # New observation space: same as GroundTruthPositionWrapper
        features_per_sprite = self.num_types + 2 + 1 + 1 + 1  # type, pos, dist, angle, valid
        obs_dim = 2 + self.max_sprites * features_per_sprite

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        print(f"YOLODetectorWrapper initialized:")
        print(f"  Play area: {self.width}x{self.height}")
        print(f"  Max sprites: {self.max_sprites}")
        print(f"  Observation dim: {obs_dim}")

    def observation(self, obs):
        """Convert pixel observation to position-based features using YOLO."""
        # obs is the raw frame from environment
        # For YOLO, we need RGB image format

        # If obs is stacked frames (4, H, W), take the last frame
        if len(obs.shape) == 3 and obs.shape[0] == 4:
            frame = obs[-1]  # Take most recent frame
        else:
            frame = obs

        # Convert grayscale to RGB for YOLO (expects 3 channels)
        if len(frame.shape) == 2:
            frame_rgb = np.stack([frame, frame, frame], axis=-1)
        else:
            frame_rgb = frame

        # Run YOLO detection
        results = self.yolo_model(frame_rgb, conf=self.confidence_threshold,
                                 device=self.device, verbose=False)

        # Extract detections
        detections = self._parse_yolo_results(results[0])

        # Convert to position features (same format as GroundTruthPositionWrapper)
        features = self._extract_features(detections)

        return features

    def _parse_yolo_results(self, result) -> List[Tuple[float, float, str, float]]:
        """
        Parse YOLO11 results into sprite list.

        Returns:
            List of (x, y, sprite_type, confidence) tuples
        """
        detections = []

        # YOLO11 results format
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        for box in result.boxes:
            # Get box coordinates (xyxy format: [x_min, y_min, x_max, y_max])
            xyxy = box.xyxy[0].cpu().numpy()
            x_min, y_min, x_max, y_max = xyxy

            # Center position
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2

            # Get sprite type from class id
            class_id = int(box.cls[0].cpu().numpy())
            sprite_type = self._class_id_to_sprite_type(class_id)

            confidence = float(box.conf[0].cpu().numpy())

            detections.append((x_center, y_center, sprite_type, confidence))

        return detections

    def _class_id_to_sprite_type(self, class_id: int) -> str:
        """
        Map YOLO class ID to sprite type name.

        Based on /home/strider/Code/labelbox_to_yolo/datasets/Robotron/dataset.yaml
        """
        # Mapping from YOLO classes to RL policy sprite types
        class_map = {
            0: 'Player',              # YOLO: Player → RL: Player
            1: 'Mikey',               # YOLO: Civilian → RL: Mikey (family member)
            2: 'Grunt',               # YOLO: Grunt → RL: Grunt
            3: 'Hulk',                # YOLO: Hulk → RL: Hulk
            4: 'Sphereoid',           # YOLO: Sphereoid → RL: Sphereoid
            5: 'Enforcer',            # YOLO: Enforcer → RL: Enforcer
            6: 'Brain',               # YOLO: Brain → RL: Brain
            7: 'Tank',                # YOLO: Tank → RL: Tank
            8: 'Quark',               # YOLO: Quark → RL: Quark
            9: 'Electrode',           # YOLO: Electrode → RL: Electrode
            10: 'EnforcerBullet',     # YOLO: Enforcer Bullet → RL: EnforcerBullet
            11: 'Prog',               # YOLO: Converted Civilian → RL: Prog (brainwashed)
            12: 'Bullet',             # YOLO: Brain Bullet → RL: Bullet (generic)
        }
        return class_map.get(class_id, 'Grunt')  # Default to Grunt if unknown

    def _extract_features(self, detections: List[Tuple[float, float, str, float]]) -> np.ndarray:
        """
        Extract position-based features from detections.

        Same format as GroundTruthPositionWrapper for compatibility.
        """
        # Find player
        player_pos = None
        other_sprites = []

        for x, y, sprite_type, confidence in detections:
            if sprite_type == 'Player':
                player_pos = np.array([x, y], dtype=np.float32)
            else:
                other_sprites.append((x, y, sprite_type, confidence))

        # If no player found, use center of screen
        if player_pos is None:
            player_pos = np.array([self.width / 2, self.height / 2], dtype=np.float32)

        # Normalize player position to [-1, 1]
        player_pos_norm = self._normalize_position(player_pos)

        # Calculate features for each sprite
        sprite_features = []
        for x, y, sprite_type, confidence in other_sprites:
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
                'confidence': confidence,
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
        norm_x = (pos[0] / self.width) * 2 - 1
        norm_y = (pos[1] / self.height) * 2 - 1
        return np.array([norm_x, norm_y], dtype=np.float32)


if __name__ == "__main__":
    """Test the YOLO wrapper."""
    from robotron import RobotronEnv
    from wrappers import MultiDiscreteToDiscrete, FrameSkipWrapper
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    import sys
    import os

    if len(sys.argv) < 2:
        print("Usage: python yolo_detector_wrapper.py <rl_policy_path> [yolo_model_path]")
        print("Example: python yolo_detector_wrapper.py models/6l9t4lpc/checkpoints/ppo_progressive_checkpoint_3000000_steps.zip")
        print("         python yolo_detector_wrapper.py models/6l9t4lpc/checkpoints/ppo_progressive_checkpoint_3000000_steps.zip ../labelbox_to_yolo/runs/detect/train3/weights/best.pt")
        sys.exit(1)

    policy_path = sys.argv[1]

    # Default YOLO model path
    if len(sys.argv) >= 3:
        yolo_model_path = sys.argv[2]
    else:
        yolo_model_path = '../labelbox_to_yolo/runs/detect/train3/weights/best.pt'

    if not os.path.exists(yolo_model_path):
        print(f"❌ YOLO model not found at: {yolo_model_path}")
        print(f"Please provide the correct path to your YOLO11 model (.pt file)")
        sys.exit(1)

    print("="*80)
    print("TESTING YOLO DETECTOR + RL POLICY")
    print("="*80)

    # Create environment with YOLO detector
    def make_env():
        env = RobotronEnv(
            level=1,
            lives=5,
            fps=30,  # Visible speed
            config_path='progressive_curriculum.yaml',
            always_move=True,
            headless=False  # Show game
        )
        env = MultiDiscreteToDiscrete(env)
        env = FrameSkipWrapper(env, skip=4)

        # Use YOLO detector wrapper (replaces GroundTruthPositionWrapper)
        env = YOLODetectorWrapper(env, yolo_model_path, max_sprites=20)

        return env

    # Load environment and policy
    vec_env = DummyVecEnv([make_env])

    # Load VecNormalize stats from training
    run_id = policy_path.split('/')[1]
    vec_normalize_path = f"models/{run_id}/vec_normalize.pkl"
    vec_env = VecNormalize.load(vec_normalize_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    # Load trained policy
    model = PPO.load(policy_path, env=vec_env)

    print("\nRunning 5 episodes with YOLO detector + position-based policy...")
    print("Watch the game window to see the agent play!\n")

    for episode in range(5):
        obs = vec_env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < 5000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            total_reward += reward[0]
            steps += 1

        if 'score' in info[0]:
            score = info[0]['score']
            kills = score // 100
            print(f"Episode {episode+1}: Score={score}, Kills={kills}, Steps={steps}")

    vec_env.close()

    print("\n" + "="*80)
    print("Test complete!")
    print("="*80)
