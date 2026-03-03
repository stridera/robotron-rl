"""
SIMPLE, STRONG reward approach.

After multiple failed attempts with reward shaping:
- Death penalty removal: policy collapsed
- Weak dense rewards: policy collapsed (drowned out)
- Strong dense rewards: reward hacking (optimized shaped rewards, not kills)

This approach:
- NO reward shaping, NO auxiliary rewards
- ONLY raw score from kills
- But scaled VERY aggressively: score/10 instead of score/100
- Grunt kill = 10.0 reward (was 1.0)
- Extremely simple curriculum: 1 grunt only for first 500k steps
- Higher entropy for more exploration: 0.05 instead of 0.01

Theory: The agent needs an absolutely massive reward signal to learn from visual inputs.
Kill rewards need to be so strong that the value function can't ignore them.
"""
import argparse
from typing import Callable
import numpy as np

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


class SimpleStrongRewardWrapper(gym.Wrapper):
    """
    ONLY score rewards, but VERY strong.

    No auxiliary rewards, no shaping tricks.
    Just: reward = score_delta / 10.0

    Killing 1 grunt (100 pts) = 10.0 reward (was 1.0)
    Killing electrode (25 pts) = 2.5 reward (was 0.25)
    """

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

        # ONLY score-based reward, but 10x stronger
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
                    if self.verbose > 0 and score > 100:
                        print(f"Episode score: {score} (best: {self.highest_score})")
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
             rank: int = 0, seed: int = 0, headless: bool = True,
             frame_skip: int = 4) -> Callable:
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
        'total_timesteps': 2_000_000,
        'num_envs': num_envs,
        'reward_shaping': 'simple_strong',
        'score_scale': 10.0,  # 10x stronger than original
    }

    # PPO with HIGH entropy for exploration
    model_kwargs = {
        'policy': 'CnnPolicy',
        'n_steps': 128,
        'batch_size': 256,
        'n_epochs': 4,
        'gamma': 0.99,  # Back to 0.99 (not sparse anymore with /10 scaling)
        'gae_lambda': 0.95,
        'clip_range': 0.1,
        'ent_coef': 0.05,  # HIGH entropy for exploration (was 0.01)
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'learning_rate': 1.0e-4,
        'policy_kwargs': {'normalize_images': False},
    }

    run = wandb.init(
        project="robotron",
        group="ppo_simple_strong",
        config=config,
        sync_tensorboard=True,
        save_code=True,
        mode="offline",
    )

    print("="*80)
    print("SIMPLE, STRONG REWARD TRAINING")
    print("="*80)
    print(f"  Reward: score_delta / 10.0 (10x stronger)")
    print(f"  Grunt kill: 100/10 = 10.0 reward")
    print(f"  NO auxiliary rewards, NO shaping")
    print(f"  High entropy (0.05) for exploration")
    print(f"  Config: {config_path or 'curriculum_config.yaml'}")
    print("="*80)

    envs = SubprocVecEnv([
        make_env(config_path, level=1, lives=5, rank=i, seed=42, headless=True)
        for i in range(num_envs)
    ])

    # VecNormalize - normalize obs only, not rewards
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

    print("\nStarting training...")
    model.learn(total_timesteps=config['total_timesteps'], callback=callbacks)

    model.save(f"models/{run.id}/final_model")
    envs.save(f"models/{run.id}/vec_normalize.pkl")

    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple, strong reward training")
    parser.add_argument("--config", type=str, default='curriculum_config.yaml',
                       help="Path to game config")
    parser.add_argument("--device", type=str, default='cuda:0',
                       help="Device to train on")
    parser.add_argument("--num-envs", type=int, default=8,
                       help="Number of parallel environments")

    args = parser.parse_args()
    main(config_path=args.config, device=args.device, num_envs=args.num_envs)
