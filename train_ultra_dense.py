"""
ULTRA-AGGRESSIVE dense reward shaping.

The problem with train_dense_rewards.py:
- Distance rewards (0.001/pixel) were 1000x weaker than kills (2.0)
- Wall penalties (0.01) were 200x weaker than kills
- Dense signals got drowned out by sparse kill rewards

This version:
- Distance rewards: 0.1/pixel (100x stronger) - comparable to kills
- Wall penalties: 1.0 at wall (100x stronger) - strong deterrent
- Score scaling: /25 (2x stronger) - even more emphasis on kills
- Action diversity bonus: reward using different actions
"""
import argparse
from typing import Callable
import numpy as np

# Suppress warnings
import warnings
warnings.filterwarnings('ignore', message='.*UnsupportedFieldAttribute.*')
warnings.filterwarnings('ignore', message='.*Field.*has no.*attribute.*')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='wandb')

from robotron import RobotronEnv
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback
from video_callback import VideoRecordingCallback
import wandb
import gymnasium as gym

from wrappers import MultiDiscreteToDiscrete, FrameSkipWrapper


class UltraDenseRewardWrapper(gym.Wrapper):
    """
    ULTRA-AGGRESSIVE dense reward shaping.

    Dense rewards are now COMPARABLE in magnitude to score rewards:
    - Distance: ±0.1 per pixel moved (was 0.001) = up to ±10 per step
    - Wall penalty: -1.0 at wall (was 0.01) = strong deterrent
    - Score: /25 (was /50) = 4.0 per grunt kill
    - Action diversity: +0.1 for using different action than last step
    """

    def __init__(self, env,
                 distance_reward_scale=0.1,    # 100x stronger!
                 wall_penalty_scale=1.0,       # 100x stronger!
                 score_scale=25.0,             # 2x stronger!
                 diversity_bonus=0.1,          # NEW: reward changing actions
                 verbose=False):
        super().__init__(env)
        self.distance_reward_scale = distance_reward_scale
        self.wall_penalty_scale = wall_penalty_scale
        self.score_scale = score_scale
        self.diversity_bonus = diversity_bonus
        self.verbose = verbose

        self.last_score = 0
        self.last_min_distance = None
        self.play_rect = None
        self.last_action = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_score = self.env.unwrapped.engine.score
        self.play_rect = self.env.unwrapped.engine.play_rect

        player_pos, enemy_positions = self._extract_positions(info)
        if player_pos and enemy_positions:
            self.last_min_distance = self._get_min_distance(player_pos, enemy_positions)
        else:
            self.last_min_distance = None

        self.last_action = None

        return obs, info

    def _extract_positions(self, info):
        """Extract player and enemy positions from sprite data."""
        data = info.get('data', [])
        player_pos = None
        enemy_positions = []

        for x, y, sprite_name in data:
            if sprite_name == 'Player':
                player_pos = (x, y)
            elif sprite_name in ['Grunt', 'Electrode', 'Hulk', 'Sphereoid', 'Quark', 'Brain',
                                'Enforcer', 'Tank', 'Prog', 'Cruise']:
                enemy_positions.append((x, y))

        return player_pos, enemy_positions

    def _get_min_distance(self, player_pos, enemy_positions):
        """Get minimum distance from player to any enemy."""
        if not enemy_positions:
            return None
        px, py = player_pos
        distances = [np.hypot(px - ex, py - ey) for ex, ey in enemy_positions]
        return min(distances)

    def _get_wall_distance(self, player_pos):
        """Get minimum distance from player to any wall."""
        if self.play_rect is None:
            return None
        px, py = player_pos
        left_dist = px
        right_dist = self.play_rect.width - px
        top_dist = py
        bottom_dist = self.play_rect.height - py
        return min(left_dist, right_dist, top_dist, bottom_dist)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 1. SCORE REWARD (main signal) - 2x stronger
        current_score = self.env.unwrapped.engine.score
        score_delta = current_score - self.last_score
        self.last_score = current_score
        score_reward = score_delta / self.score_scale  # Grunt kill = 4.0 (was 2.0)

        # 2. DISTANCE REWARD - 100x stronger!
        distance_reward = 0.0
        player_pos, enemy_positions = self._extract_positions(info)

        if player_pos and enemy_positions:
            current_min_distance = self._get_min_distance(player_pos, enemy_positions)

            if self.last_min_distance is not None and current_min_distance is not None:
                distance_delta = self.last_min_distance - current_min_distance
                distance_reward = distance_delta * self.distance_reward_scale
                # Moving 10 pixels closer = +1.0 reward (comparable to a kill!)

            self.last_min_distance = current_min_distance

        # 3. WALL PENALTY - 100x stronger!
        wall_penalty = 0.0
        if player_pos:
            wall_dist = self._get_wall_distance(player_pos)
            if wall_dist is not None:
                # Strong penalty near walls
                wall_threshold = 100.0
                if wall_dist < wall_threshold:
                    normalized_dist = wall_dist / wall_threshold
                    wall_penalty = -(1.0 - normalized_dist) * self.wall_penalty_scale
                    # At wall: -1.0 penalty (25% of a kill!)

        # 4. ACTION DIVERSITY BONUS (NEW!)
        diversity_reward = 0.0
        if self.last_action is not None:
            # Reward for changing action (encourages exploration)
            if not np.array_equal(action, self.last_action):
                diversity_reward = self.diversity_bonus
        self.last_action = action.copy() if hasattr(action, 'copy') else action

        # TOTAL REWARD
        total_reward = score_reward + distance_reward + wall_penalty + diversity_reward

        if self.verbose and (score_delta > 0 or abs(distance_reward) > 0.01 or wall_penalty < -0.1 or diversity_reward > 0):
            print(f"  Score: {score_reward:+.2f}  Dist: {distance_reward:+.2f}  Wall: {wall_penalty:+.2f}  Div: {diversity_reward:+.2f}  Tot: {total_reward:+.2f}")

        return obs, total_reward, terminated, truncated, info


class MetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_scores = []
        self.highest_score = 0

    def _on_step(self):
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info and 'score' in info:
                    score = info['score']
                    self.episode_scores.append(score)
                    self.highest_score = max(self.highest_score, score)
                    self.logger.record('robotron/episode_score', score)
                    self.logger.record('robotron/highest_score', self.highest_score)
        return True


class VecNormalizeCallback(BaseCallback):
    def __init__(self, vec_normalize_env, save_path, save_freq=100_000, verbose=0):
        super().__init__(verbose)
        self.vec_normalize_env = vec_normalize_env
        self.save_path = save_path
        self.save_freq = save_freq
        self.last_save = -1

    def _on_step(self):
        if self.num_timesteps >= self.save_freq and self.num_timesteps % self.save_freq == 0:
            if self.num_timesteps != self.last_save:
                self.vec_normalize_env.save(self.save_path)
                self.last_save = self.num_timesteps
                if self.verbose > 0:
                    print(f"✅ Saved at {self.num_timesteps:,} steps")
        return True


def make_env(config_path: str = None, level: int = 1, lives: int = 3,
             rank: int = 0, seed: int = 0, use_reward_shaping: bool = True,
             headless: bool = True, frame_skip: int = 4) -> Callable:
    def _init():
        env = RobotronEnv(level=level, lives=lives, fps=0, config_path=config_path,
                         always_move=True, headless=headless)
        env = MultiDiscreteToDiscrete(env)
        if frame_skip > 1:
            env = FrameSkipWrapper(env, skip=frame_skip)
        if use_reward_shaping:
            env = UltraDenseRewardWrapper(env, verbose=False)
        env = GrayscaleObservation(env, keep_dim=False)
        env = ResizeObservation(env, (84, 84))
        env = FrameStackObservation(env, stack_size=4)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def main(config_path: str = None, device: str = 'cuda:0', num_envs: int = 8):
    config = {
        'model': 'ppo',
        'total_timesteps': 2_000_000,  # Just 2M for testing
        'num_envs': num_envs,
        'reward_shaping': 'ultra_dense',
        'reward_params': {
            'distance_scale': 0.1,
            'wall_penalty': 1.0,
            'score_scale': 25.0,
            'diversity_bonus': 0.1,
        }
    }

    model_kwargs = {
        'policy': 'CnnPolicy',
        'n_steps': 128,
        'batch_size': 256,
        'n_epochs': 4,
        'gamma': 0.95,
        'gae_lambda': 0.95,
        'clip_range': 0.1,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'learning_rate': 1.0e-4,
        'policy_kwargs': {'normalize_images': False},
    }

    run = wandb.init(
        project="robotron",
        group="ppo_ultra_dense",
        config=config,
        sync_tensorboard=True,
        save_code=True,
        mode="offline",
    )

    print(f"Creating {num_envs} envs with ULTRA-DENSE rewards...")
    print(f"  Distance: ±0.1/pixel (100x stronger)")
    print(f"  Wall: -1.0 at wall (100x stronger)")
    print(f"  Score: /25 (2x stronger)")
    print(f"  Diversity: +0.1 for action change")

    envs = SubprocVecEnv([
        make_env(config_path, level=1, lives=5, rank=i, seed=42, headless=True)
        for i in range(num_envs)
    ])
    envs = VecNormalize(envs, norm_obs=True, norm_reward=False, clip_obs=10.)

    model = PPO(env=envs, verbose=1, tensorboard_log=f"runs/{run.id}",
                device=device, **model_kwargs)

    callbacks = [
        MetricsCallback(verbose=1),
        VecNormalizeCallback(envs, f"models/{run.id}/vec_normalize.pkl", 100_000, 1),
        VideoRecordingCallback(
            record_freq=100_000,
            video_length=500,
            config_path=config_path,
            level=1,
            lives=5,
            verbose=1
        ),
        WandbCallback(
            gradient_save_freq=1000,
            model_save_freq=100_000,
            model_save_path=f"models/{run.id}",
            verbose=2
        ),
        CheckpointCallback(
            save_freq=100_000 // num_envs,
            save_path=f"models/{run.id}/checkpoints",
            name_prefix="ppo_checkpoint"
        ),
    ]

    print("Starting ULTRA-DENSE training...")
    model.learn(total_timesteps=config['total_timesteps'], callback=callbacks)

    model.save(f"models/{run.id}/final_model")
    envs.save(f"models/{run.id}/vec_normalize.pkl")
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='curriculum_config.yaml')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--num-envs", type=int, default=8)
    args = parser.parse_args()

    main(config_path=args.config, device=args.device, num_envs=args.num_envs)
