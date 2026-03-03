"""
Progressive curriculum training for position-based RL.

Tests if position-based RL scales from 1 grunt to full game complexity.

Key changes from train_positions.py:
- Uses progressive_curriculum.yaml (1 grunt → 85 sprites over 21 levels)
- max_sprites=20 (was 10) - handles more realistic scenarios
- Larger MLP: 512x512 (was 256x256) - more capacity for complex scenarios
- Longer training: 3M steps (was 1M) - time to master full curriculum
"""
import argparse
from typing import Callable
import numpy as np

import warnings
warnings.filterwarnings('ignore', message='.*UnsupportedFieldAttribute.*')
warnings.filterwarnings('ignore', message='.*Field.*has no.*attribute.*')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='wandb')

from robotron import RobotronEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback
import wandb
import gymnasium as gym

from wrappers import MultiDiscreteToDiscrete, FrameSkipWrapper
from position_wrapper import GroundTruthPositionWrapper


class SimpleStrongRewardWrapper(gym.Wrapper):
    """Simple strong reward: score/10 (10x stronger than original)."""

    def __init__(self, env, score_scale=10.0, verbose=False):
        super().__init__(env)
        self.score_scale = score_scale
        self.verbose = verbose
        self.last_score = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_score = self.env.unwrapped.engine.score
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        current_score = self.env.unwrapped.engine.score
        score_delta = current_score - self.last_score
        self.last_score = current_score

        reward = score_delta / self.score_scale

        if self.verbose and score_delta > 0:
            print(f"  Kill! Score delta: {score_delta} → Reward: {reward:.1f}")

        return obs, reward, terminated, truncated, info


class MetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_scores = []
        self.episode_kills = []
        self.highest_score = 0
        self.episode_levels = []

    def _on_step(self):
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info and 'score' in info:
                    score = info['score']
                    self.episode_scores.append(score)
                    self.highest_score = max(self.highest_score, score)

                    kills = score // 100  # Rough estimate: 100 points per grunt
                    self.episode_kills.append(kills)

                    # Try to get level from info
                    level = info.get('level', 1)
                    self.episode_levels.append(level)

                    self.logger.record('robotron/episode_score', score)
                    self.logger.record('robotron/highest_score', self.highest_score)
                    self.logger.record('robotron/episode_kills', kills)
                    self.logger.record('robotron/episode_level', level)

                    if self.verbose > 0 and score > 0:
                        print(f"Episode: score={score}, kills={kills}, level={level}, best={self.highest_score}")
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
                    print(f"✅ VecNormalize saved at {self.num_timesteps:,} steps")
        return True


def make_env(config_path: str = None, level: int = 1, lives: int = 5,
             rank: int = 0, seed: int = 0, headless: bool = True,
             frame_skip: int = 4, max_sprites: int = 20) -> Callable:
    def _init():
        env = RobotronEnv(
            level=level,
            lives=lives,
            fps=0,
            config_path=config_path,
            always_move=True,
            headless=headless
        )
        env = MultiDiscreteToDiscrete(env)

        if frame_skip > 1:
            env = FrameSkipWrapper(env, skip=frame_skip)

        # Simple strong reward wrapper
        env = SimpleStrongRewardWrapper(env, score_scale=10.0, verbose=False)

        # KEY: Position wrapper with max_sprites=20 for realistic scenarios
        env = GroundTruthPositionWrapper(env, max_sprites=max_sprites, verbose=False)

        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def main(config_path: str = 'progressive_curriculum.yaml', device: str = 'cuda:0',
         num_envs: int = 8, total_timesteps: int = 3_000_000, max_sprites: int = 20):

    # Calculate observation dimension for logging
    # player_pos (2) + max_sprites * (type_onehot (17) + rel_pos (2) + dist (1) + angle (1) + valid (1))
    obs_dim = 2 + max_sprites * (17 + 2 + 1 + 1 + 1)

    config = {
        'model': 'ppo',
        'total_timesteps': total_timesteps,
        'num_envs': num_envs,
        'observation_type': 'positions_progressive',
        'score_scale': 10.0,
        'max_sprites': max_sprites,
        'obs_dim': obs_dim,
        'config_path': config_path,
    }

    # PPO with larger MLP for complex scenarios
    model_kwargs = {
        'policy': 'MlpPolicy',
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'learning_rate': 3e-4,
        'policy_kwargs': {
            'net_arch': [512, 512],  # Larger network for complex scenarios (was 256x256)
        },
    }

    run = wandb.init(
        project="robotron",
        group="ppo_positions_progressive",
        config=config,
        sync_tensorboard=True,
        save_code=True,
        mode="offline",
    )

    print("="*80)
    print("PROGRESSIVE CURRICULUM TRAINING (POSITION-BASED RL)")
    print("="*80)
    print(f"  Observation: Sprite positions ({obs_dim} dims)")
    print(f"  Policy: MlpPolicy (2x512 layers - larger for complex scenarios)")
    print(f"  Reward: score_delta / 10.0")
    print(f"  Config: {config_path}")
    print(f"  Max sprites: {max_sprites}")
    print(f"  Total timesteps: {total_timesteps:,}")
    print("="*80)
    print("\n🎯 TESTING: Can position-based RL scale to full game complexity?")
    print("\n📚 Curriculum Stages:")
    print("   Stage 1 (Levels 1-3):   1-3 grunts")
    print("   Stage 2 (Levels 4-6):   3-5 grunts + obstacles + family")
    print("   Stage 3 (Levels 7-9):   5-8 grunts + hulks (unkillable)")
    print("   Stage 4 (Levels 10-12): 10-15 grunts (realistic counts)")
    print("   Stage 5 (Levels 13-15): Add spawners (brains, sphereoids)")
    print("   Stage 6 (Levels 16-18): Add quarks (tank spawners)")
    print("   Stage 7 (Levels 19-21): Full difficulty (25-35 grunts + all enemies)")
    print("\n🎖️  Success Criteria:")
    print("   - Master Stage 1-3: 100% of episodes (already proven)")
    print("   - Master Stage 4: 10+ kills consistently")
    print("   - Master Stage 5-7: 30+ kills on realistic levels")
    print("   - Maintain action diversity throughout\n")

    envs = SubprocVecEnv([
        make_env(config_path, level=1, lives=5, rank=i, seed=42, headless=True, max_sprites=max_sprites)
        for i in range(num_envs)
    ])

    # VecNormalize for position features (helps with different scales)
    envs = VecNormalize(envs, norm_obs=True, norm_reward=False, clip_obs=10.)

    model = PPO(
        env=envs,
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
        device=device,
        **model_kwargs
    )

    callbacks = [
        MetricsCallback(verbose=1),
        VecNormalizeCallback(
            vec_normalize_env=envs,
            save_path=f"models/{run.id}/vec_normalize.pkl",
            save_freq=100_000,
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
            name_prefix="ppo_progressive_checkpoint"
        ),
    ]

    print("\nStarting training...")
    print(f"Expected timeline:")
    print(f"  100k-500k steps: Master Stage 1-3 (1-9 sprites)")
    print(f"  500k-1M steps: Master Stage 4 (10-15 grunts)")
    print(f"  1M-2M steps: Master Stage 5-6 (spawners)")
    print(f"  2M-3M steps: Master Stage 7 (full difficulty)\n")

    model.learn(total_timesteps=config['total_timesteps'], callback=callbacks)

    model.save(f"models/{run.id}/final_model")
    envs.save(f"models/{run.id}/vec_normalize.pkl")

    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)
    print("\n📊 Next steps:")
    print(f"  1. Evaluate: poetry run python check_position_model.py models/{run.id}/checkpoints/ppo_progressive_checkpoint_3000000_steps.zip")
    print(f"  2. If successful on complex levels → Position-based RL scales!")
    print(f"  3. If failed → May need:")
    print(f"     - Larger network (1024x1024)")
    print(f"     - More training (5M+ steps)")
    print(f"     - Different feature engineering")
    print(f"  4. If successful → Proceed to Phase 2 (sprite detector)")

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Progressive curriculum position-based RL")
    parser.add_argument("--config", type=str, default='progressive_curriculum.yaml',
                       help="Path to game config")
    parser.add_argument("--device", type=str, default='cuda:0',
                       help="Device to train on")
    parser.add_argument("--num-envs", type=int, default=8,
                       help="Number of parallel environments")
    parser.add_argument("--timesteps", type=int, default=3_000_000,
                       help="Total timesteps to train")
    parser.add_argument("--max-sprites", type=int, default=20,
                       help="Max sprites to track (default: 20)")

    args = parser.parse_args()
    main(
        config_path=args.config,
        device=args.device,
        num_envs=args.num_envs,
        total_timesteps=args.timesteps,
        max_sprites=args.max_sprites
    )
