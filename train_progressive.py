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
warnings.filterwarnings('ignore', message='.*frozen.*', category=UserWarning, module='pydantic')
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')

from robotron import RobotronEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback
import wandb
import gymnasium as gym

from wrappers import MultiDiscreteToDiscrete, FrameSkipWrapper
from position_wrapper import GroundTruthPositionWrapper, OBS_DIM


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

        # Category-guaranteed position wrapper (41 slots, 986 dims)
        env = GroundTruthPositionWrapper(env)

        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def main(config_path: str = 'config.yaml', device: str = 'cuda:0',
         num_envs: int = 8, total_timesteps: int = 3_000_000,
         bc_checkpoint: str = None, start_level: int = 1, lives: int = 3,
         lr: float = None, clip_range: float = None, ent_coef: float = None,
         vf_coef: float = None):

    fine_tuning = bc_checkpoint is not None

    config = {
        'model': 'ppo',
        'total_timesteps': total_timesteps,
        'num_envs': num_envs,
        'observation_type': 'positions_category_slots',
        'score_scale': 10.0,
        'obs_dim': OBS_DIM,
        'config_path': config_path,
        'bc_checkpoint': bc_checkpoint,
        'start_level': start_level,
        'fine_tuning': fine_tuning,
    }

    if fine_tuning:
        # Tighter HPs to preserve BC initialization (overridable via CLI)
        model_kwargs = {
            'policy': 'MlpPolicy',
            'n_steps': 2048,
            'batch_size': 128,
            'n_epochs': 10,
            'gamma': 0.995,
            'gae_lambda': 0.95,
            'clip_range': clip_range if clip_range is not None else 0.1,
            'ent_coef': ent_coef if ent_coef is not None else 0.005,
            'vf_coef': vf_coef if vf_coef is not None else 0.5,
            'max_grad_norm': 0.5,
            'learning_rate': lr if lr is not None else 5e-5,
            'policy_kwargs': {
                'net_arch': [512, 512],
            },
        }
    else:
        # Scratch training HPs (overridable via CLI)
        model_kwargs = {
            'policy': 'MlpPolicy',
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': clip_range if clip_range is not None else 0.2,
            'ent_coef': ent_coef if ent_coef is not None else 0.01,
            'vf_coef': vf_coef if vf_coef is not None else 0.5,
            'max_grad_norm': 0.5,
            'learning_rate': lr if lr is not None else 3e-4,
            'policy_kwargs': {
                'net_arch': [512, 512],
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

    mode_str = f"BC fine-tuning from {bc_checkpoint}" if fine_tuning else "scratch"
    print("="*80)
    print("PROGRESSIVE CURRICULUM TRAINING (POSITION-BASED RL)")
    print("="*80)
    print(f"  Mode:        {mode_str}")
    print(f"  Observation: Category-slot positions ({OBS_DIM} dims, 41 slots)")
    print(f"  Policy:      MlpPolicy (2x512)")
    print(f"  Reward:      score_delta / 10.0")
    print(f"  Config:      {config_path}")
    print(f"  Start level: {start_level}")
    print(f"  Timesteps:   {total_timesteps:,}")
    if fine_tuning:
        print(f"  clip_range={model_kwargs['clip_range']}  ent_coef={model_kwargs['ent_coef']}  lr={model_kwargs['learning_rate']}  (fine-tuning HPs)")
    print("="*80)

    envs = SubprocVecEnv([
        make_env(config_path, level=start_level, lives=lives, rank=i, seed=42, headless=True)
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

    if fine_tuning and bc_checkpoint:
        print(f"\nLoading BC weights from {bc_checkpoint}...")
        bc_model = PPO.load(bc_checkpoint, env=envs, device=device)
        model.set_parameters(bc_model.get_parameters())
        print("BC weights loaded successfully.")

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

    if fine_tuning:
        print("\nStarting fine-tuning from BC checkpoint...")
        print("  Expected: W7+ at 500k steps, W10+ at 2M steps\n")
    else:
        print("\nStarting training from scratch...")
        print("  Expected: Playable at 1M+ steps\n")

    model.learn(total_timesteps=config['total_timesteps'], callback=callbacks)

    model.save(f"models/{run.id}/final_model")
    envs.save(f"models/{run.id}/vec_normalize.pkl")

    print("\n" + "="*80)
    print("Training complete!")
    print(f"  Model saved to: models/{run.id}/")
    print("="*80)

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Progressive curriculum position-based RL")
    parser.add_argument("--config",         type=str, default='config.yaml')
    parser.add_argument("--device",         type=str, default='cuda:0')
    parser.add_argument("--num-envs",       type=int, default=8)
    parser.add_argument("--timesteps",      type=int, default=3_000_000)
    parser.add_argument("--bc-checkpoint",  type=str, default=None,
                        help="Path to BC model zip (from train_bc.py). Triggers fine-tuning HPs.")
    parser.add_argument("--start-level",    type=int, default=1,
                        help="Curriculum starting level (use 5 with --bc-checkpoint)")
    parser.add_argument("--lives",          type=int, default=3,
                        help="Starting lives per episode (default: 3, original game value)")
    parser.add_argument("--lr",             type=float, default=None,
                        help="Learning rate override (default: 5e-5 fine-tune, 3e-4 scratch)")
    parser.add_argument("--clip-range",     type=float, default=None,
                        help="PPO clip range override (default: 0.1 fine-tune, 0.2 scratch)")
    parser.add_argument("--ent-coef",       type=float, default=None,
                        help="Entropy coefficient override (default: 0.005 fine-tune, 0.01 scratch)")
    parser.add_argument("--vf-coef",        type=float, default=None,
                        help="Value function coefficient override (default: 0.5)")

    args = parser.parse_args()
    main(
        config_path=args.config,
        device=args.device,
        num_envs=args.num_envs,
        total_timesteps=args.timesteps,
        bc_checkpoint=args.bc_checkpoint,
        start_level=args.start_level,
        lives=args.lives,
        lr=args.lr,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
    )
