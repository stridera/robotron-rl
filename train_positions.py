"""
Position-based RL training for Robotron.

Uses GroundTruthPositionWrapper to train on sprite positions instead of pixels.
This tests whether PPO can solve the task when vision is not a bottleneck.

Key differences from pixel-based training:
- Observation: 222-dim position vector (not 28k-dim pixels)
- Policy: MlpPolicy (not CnnPolicy - no CNN needed!)
- Expected to learn MUCH faster (10-100x)
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
from video_callback import VideoRecordingCallback
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

    def _on_step(self):
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info and 'score' in info:
                    score = info['score']
                    self.episode_scores.append(score)
                    self.highest_score = max(self.highest_score, score)

                    kills = score // 100  # Rough estimate: 100 points per grunt
                    self.episode_kills.append(kills)

                    self.logger.record('robotron/episode_score', score)
                    self.logger.record('robotron/highest_score', self.highest_score)
                    self.logger.record('robotron/episode_kills', kills)

                    if self.verbose > 0 and score > 0:
                        print(f"Episode: score={score}, kills={kills}, best={self.highest_score}")
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


def make_env(config_path: str = None, level: int = 1, lives: int = 5,
             rank: int = 0, seed: int = 0, headless: bool = True,
             frame_skip: int = 4, max_sprites: int = 10) -> Callable:
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

        # KEY: Use position wrapper instead of pixel observations
        env = GroundTruthPositionWrapper(env, max_sprites=max_sprites, verbose=False)

        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def main(config_path: str = None, device: str = 'cuda:0', num_envs: int = 8,
         total_timesteps: int = 1_000_000):

    config = {
        'model': 'ppo',
        'total_timesteps': total_timesteps,
        'num_envs': num_envs,
        'observation_type': 'positions',  # KEY MARKER
        'score_scale': 10.0,
        'max_sprites': 10,
    }

    # PPO with MLP policy (not CNN!)
    model_kwargs = {
        'policy': 'MlpPolicy',  # KEY: MlpPolicy for position vectors, not CnnPolicy!
        'n_steps': 2048,  # Longer rollouts for MLP (not constrained by memory like CNN)
        'batch_size': 64,  # Smaller batch for MLP
        'n_epochs': 10,  # More epochs
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,  # Lower entropy for position-based (easier task)
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'learning_rate': 3e-4,  # Standard MLP learning rate
        'policy_kwargs': {
            'net_arch': [256, 256],  # 2-layer MLP with 256 units each
        },
    }

    run = wandb.init(
        project="robotron",
        group="ppo_positions",
        config=config,
        sync_tensorboard=True,
        save_code=True,
        mode="offline",
    )

    print("="*80)
    print("POSITION-BASED RL TRAINING")
    print("="*80)
    print(f"  Observation: Sprite positions (222 dims, not 28k pixels!)")
    print(f"  Policy: MlpPolicy (2x256 layers, not CNN)")
    print(f"  Reward: score_delta / 10.0")
    print(f"  Config: {config_path or 'ultra_simple_curriculum.yaml'}")
    print(f"  Max sprites: {config['max_sprites']}")
    print("="*80)
    print("\n🎯 TESTING: Can PPO solve this with perfect position info?")
    print("   If YES → vision was the bottleneck, use detector approach")
    print("   If NO → PPO can't solve this task, need different algorithm\n")

    envs = SubprocVecEnv([
        make_env(config_path, level=1, lives=5, rank=i, seed=42, headless=True)
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
        # VideoRecordingCallback disabled for position-based training
        # (video callback expects pixel observations, but we're using positions)
        # VideoRecordingCallback(
        #     record_freq=100_000,
        #     video_length=500,
        #     config_path=config_path,
        #     level=1,
        #     lives=5,
        #     verbose=1
        # ),
        WandbCallback(
            gradient_save_freq=1000,
            model_save_freq=100_000,
            model_save_path=f"models/{run.id}",
            verbose=2
        ),
        CheckpointCallback(
            save_freq=100_000 // num_envs,
            save_path=f"models/{run.id}/checkpoints",
            name_prefix="ppo_positions_checkpoint"
        ),
    ]

    print("\nStarting training...")
    print(f"Expected: Should learn MUCH faster than pixel-based (~10-100x)")
    print(f"Target: Kill 1 grunt consistently within 100k-200k steps\n")

    model.learn(total_timesteps=config['total_timesteps'], callback=callbacks)

    model.save(f"models/{run.id}/final_model")
    envs.save(f"models/{run.id}/vec_normalize.pkl")

    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)
    print("\n📊 Next steps:")
    print(f"  1. Check: poetry run python check_dense_rewards.py models/{run.id}/checkpoints/ppo_positions_checkpoint_100000_steps.zip")
    print(f"  2. If successful: Move to Phase 2 (train sprite detector)")
    print(f"  3. If failed: Try different algorithm (Rainbow DQN, DreamerV3)")

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Position-based RL training")
    parser.add_argument("--config", type=str, default='ultra_simple_curriculum.yaml',
                       help="Path to game config")
    parser.add_argument("--device", type=str, default='cuda:0',
                       help="Device to train on")
    parser.add_argument("--num-envs", type=int, default=8,
                       help="Number of parallel environments")
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                       help="Total timesteps to train")

    args = parser.parse_args()
    main(
        config_path=args.config,
        device=args.device,
        num_envs=args.num_envs,
        total_timesteps=args.timesteps
    )
