"""
Improved training script for Robotron 2084 with curriculum learning and better hyperparameters.

Key improvements:
1. Curriculum learning - start with easier levels
2. Multiple parallel environments for better sample efficiency
3. Better reward shaping
4. Tuned hyperparameters for Atari-like games
5. Proper observation normalization
6. More lives to allow learning
"""
import argparse
from typing import Callable
import numpy as np

# Suppress annoying WandB warnings
import warnings
warnings.filterwarnings('ignore', message='.*UnsupportedFieldAttribute.*')
warnings.filterwarnings('ignore', message='.*Field.*has no.*attribute.*')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='wandb')

from robotron import RobotronEnv
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3 import PPO, DQN
from sb3_contrib import QRDQN
from wandb.integration.sb3 import WandbCallback
from video_callback import VideoRecordingCallback
from live_gameplay_callback import LiveGameplayCallback
import wandb
import gymnasium as gym

# Import custom wrappers
from wrappers import MultiDiscreteToDiscrete, FrameSkipWrapper


class RewardShapingWrapper(gym.Wrapper):
    """
    Score-focused reward shaping for Robotron:
    - Use raw game score as PRIMARY signal (killing enemies)
    - Small death penalty (just enough to discourage dying)
    - NO level bonus (level completion is a side effect of killing all enemies)
    """
    def __init__(self, env, death_penalty=10.0, verbose=False):
        super().__init__(env)
        self.death_penalty = death_penalty
        self.verbose = verbose
        self.last_score = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_score = self.env.unwrapped.engine.score
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Get actual score delta from environment
        current_score = self.env.unwrapped.engine.score
        score_delta = current_score - self.last_score
        self.last_score = current_score

        # PRIMARY SIGNAL: Raw score from killing enemies
        # Scale to make individual kills meaningful:
        # - Killing 1 grunt (100 pts) = +1 reward
        # - Killing electrode (25 pts) = +0.25 reward
        # - Clearing level (~30k pts) = +300 reward (comes naturally from kills!)
        # Using /100 instead of /50 to reduce reward variance for more stable learning
        reward = score_delta / 100.0

        # NO death penalty - let PPO learn survival naturally through gamma discount
        # Death means losing future rewards, which is penalty enough
        # Explicit death penalties make agent too risk-averse
        if terminated and self.verbose:
            print(f"Died with score: {current_score}")

        return obs, reward, terminated, truncated, info


class MetricsCallback(BaseCallback):
    """
    Track additional game metrics like highest level, score, families saved, etc.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_scores = []
        self.episode_levels = []
        self.episode_families = []
        self.highest_level = 0
        self.highest_score = 0

    def _on_step(self):
        # Check if any environments finished an episode
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                # Monitor wraps episode info in 'episode' key
                # But custom keys are also at the top level
                if 'episode' in info:
                    # Episode finished - track metrics
                    # Custom info keys are preserved alongside 'episode' key
                    if 'score' in info:
                        score = info['score']
                        self.episode_scores.append(score)
                        self.highest_score = max(self.highest_score, score)

                        # Log to WandB/tensorboard
                        self.logger.record('robotron/episode_score', score)
                        self.logger.record('robotron/highest_score', self.highest_score)

                    if 'level' in info:
                        level = info['level'] + 1  # Level is 0-indexed, display as 1-indexed
                        self.episode_levels.append(level)
                        self.highest_level = max(self.highest_level, level)

                        # Log to WandB/tensorboard
                        self.logger.record('robotron/episode_level', level)
                        self.logger.record('robotron/highest_level', self.highest_level)

                    if 'family' in info:
                        families_remaining = info['family']
                        self.episode_families.append(families_remaining)
                        self.logger.record('robotron/families_remaining', families_remaining)

                    if 'lives' in info:
                        self.logger.record('robotron/lives_remaining', info['lives'])

                    if self.verbose > 0:
                        msg = f"Episode complete - "
                        if 'score' in info:
                            msg += f"Score: {score} (best: {self.highest_score}) "
                        if 'level' in info:
                            msg += f"Level: {level} (best: {self.highest_level})"
                        print(msg)

        return True


class VecNormalizeCallback(BaseCallback):
    """
    Periodically save VecNormalize statistics during training.
    Saves more frequently early on, then less frequently later.
    """
    def __init__(self, vec_normalize_env, save_path, save_freq=100_000, verbose=0):
        super().__init__(verbose)
        self.vec_normalize_env = vec_normalize_env
        self.save_path = save_path
        self.save_freq = save_freq
        self.last_save = -1

    def _on_step(self):
        # Save at specific milestones for early training, then every save_freq
        milestones = [1000, 10_000, 50_000]  # Early saves

        should_save = False

        # Check early milestones
        if self.num_timesteps in milestones and self.num_timesteps != self.last_save:
            should_save = True
        # Check regular frequency
        elif self.num_timesteps >= self.save_freq and self.num_timesteps % self.save_freq == 0:
            should_save = True

        if should_save:
            self.vec_normalize_env.save(self.save_path)
            self.last_save = self.num_timesteps
            if self.verbose > 0:
                print(f"✅ Saved VecNormalize stats at {self.num_timesteps:,} steps")

        return True


class CurriculumCallback(BaseCallback):
    """
    Gradually increase difficulty by changing game config.
    Start with very simple scenarios and add complexity.
    """
    def __init__(self, env, milestone_steps, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.milestone_steps = milestone_steps  # List of (step, config_updates)
        self.current_milestone = 0

    def _on_step(self):
        if self.current_milestone < len(self.milestone_steps):
            step_threshold, updates = self.milestone_steps[self.current_milestone]
            if self.num_timesteps >= step_threshold:
                if self.verbose > 0:
                    print(f"\n{'='*50}")
                    print(f"Curriculum Update at {self.num_timesteps} steps")
                    print(f"Updates: {updates}")
                    print(f"{'='*50}\n")
                self.current_milestone += 1
        return True


class RawScoreEvalCallback(BaseCallback):
    """
    Custom evaluation callback that logs RAW game scores instead of normalized rewards.

    The default EvalCallback uses VecNormalize-wrapped env, so it logs normalized
    rewards which are meaningless (e.g., 85, -18, 255). This callback creates a
    fresh evaluation env WITHOUT VecNormalize to get actual game scores.
    """
    def __init__(
        self,
        config_path: str,
        level: int = 1,
        lives: int = 5,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 5,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.config_path = config_path
        self.level = level
        self.lives = lives
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_score = -np.inf
        self.eval_env = None

    def _init_callback(self) -> None:
        # Create eval env WITHOUT VecNormalize
        self.eval_env = make_env(
            config_path=self.config_path,
            level=self.level,
            lives=self.lives,
            rank=0,
            seed=999,
            headless=True,
        )()

    def _on_step(self) -> bool:
        # Only evaluate at specified frequency
        if self.n_calls % self.eval_freq != 0:
            return True

        # Run evaluation episodes
        episode_scores = []
        episode_levels = []
        episode_lengths = []

        for _ in range(self.n_eval_episodes):
            obs, info = self.eval_env.reset()
            done = False
            episode_length = 0

            while not done:
                # Get action from model
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                episode_length += 1

            # Extract raw score and level from info
            if 'score' in info:
                episode_scores.append(info['score'])
            if 'level' in info:
                episode_levels.append(info['level'] + 1)  # 0-indexed to 1-indexed
            episode_lengths.append(episode_length)

        # Calculate statistics
        mean_score = np.mean(episode_scores)
        std_score = np.std(episode_scores)
        mean_level = np.mean(episode_levels)
        mean_length = np.mean(episode_lengths)

        # Log to tensorboard/wandb
        self.logger.record('eval_raw/mean_score', mean_score)
        self.logger.record('eval_raw/std_score', std_score)
        self.logger.record('eval_raw/mean_level', mean_level)
        self.logger.record('eval_raw/mean_ep_length', mean_length)
        self.logger.record('eval_raw/best_mean_score', max(self.best_mean_score, mean_score))

        # Update best score
        if mean_score > self.best_mean_score:
            self.best_mean_score = mean_score
            if self.verbose > 0:
                print(f"\n🏆 New best mean score: {mean_score:.0f} (was {self.best_mean_score:.0f})")

        if self.verbose > 0:
            print(f"\nEvaluation at {self.num_timesteps:,} steps:")
            print(f"  Mean Score: {mean_score:.0f} ± {std_score:.0f}")
            print(f"  Mean Level: {mean_level:.1f}")
            print(f"  Mean Length: {mean_length:.0f}")
            print(f"  Best Score: {self.best_mean_score:.0f}")

        return True


def make_env(config_path: str = None, level: int = 1, lives: int = 3,
             rank: int = 0, seed: int = 0, use_reward_shaping: bool = True,
             headless: bool = True, frame_skip: int = 4) -> Callable:
    """
    Create a single environment with proper seeding for parallel training.
    Now with MultiDiscrete action space and frame skip for faster learning.
    """
    def _init():
        env = RobotronEnv(
            level=level,
            lives=lives,
            fps=0,
            config_path=config_path,
            always_move=True,  # Keeps Discrete(64) action space
            headless=headless,
        )

        # Convert to MultiDiscrete([8, 8]) for independent movement/shooting
        env = MultiDiscreteToDiscrete(env)

        # Frame skip BEFORE preprocessing (operates on raw observations)
        if frame_skip > 1:
            env = FrameSkipWrapper(env, skip=frame_skip)

        # Apply reward shaping
        if use_reward_shaping:
            env = RewardShapingWrapper(env)

        # Preprocessing
        env = GrayscaleObservation(env, keep_dim=False)  # 2D grayscale output
        env = ResizeObservation(env, (84, 84))  # Standard Atari size

        # Frame stacking BEFORE vectorization for correct shape handling
        env = FrameStackObservation(env, stack_size=4)

        # Monitor for tracking
        env = Monitor(env)

        # Seed for reproducibility
        env.reset(seed=seed + rank)
        return env
    return _init


def main(model_name: str,
         config_path: str = None,
         resume_path: str = None,
         project: str = None,
         group: str = None,
         device: str = 'cuda:0',
         num_envs: int = 8,
         use_curriculum: bool = True,
         video_freq: int = 100_000,  # Record video every 100k steps
         video_length: int = 500,
         display_freq: int = 50_000):  # Show live gameplay every 50k steps

    # Configuration
    config = {
        'model': model_name,
        'env_name': 'robotron',
        'resume_path': resume_path,
        'total_timesteps': 20_000_000,
        'num_envs': num_envs,
        'use_curriculum': use_curriculum,
        'video_freq': video_freq,
        'video_length': video_length,
        'env': {
            'config_path': config_path,
            'level': 1,
            'lives': 5,  # More lives to learn from mistakes!
            'fps': 0,
            'always_move': True,
        },
    }

    # Model-specific hyperparameters (tuned for Atari-like games)
    if model_name == "ppo":
        model_class = PPO
        config['model_kwargs'] = {
            'policy': 'CnnPolicy',  # CnnPolicy works with MultiDiscrete actions
            'n_steps': 64,  # Reduced from 128 for faster updates
            'batch_size': 256,  # Minibatch size
            'n_epochs': 4,  # Number of epochs when optimizing
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.1,
            'ent_coef': 0.02,  # Increased from 0.01 for more exploration
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'learning_rate': 1.0e-4,  # Reduced from 2.5e-4 for more stable learning (prevents catastrophic forgetting)
            'policy_kwargs': {'normalize_images': False},  # VecNormalize handles normalization
        }
    elif model_name == 'dqn':
        model_class = DQN
        config['model_kwargs'] = {
            'policy': 'CnnPolicy',
            'learning_rate': 5e-5,  # Reduced from 1e-4 for stability
            'buffer_size': 100_000,
            'learning_starts': 10_000,  # Much earlier than QRDQN
            'batch_size': 32,
            'gamma': 0.99,
            'train_freq': 4,
            'gradient_steps': 1,
            'target_update_interval': 1_000,
            'exploration_fraction': 0.2,  # Explore for 20% of training
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': 0.05,
            'policy_kwargs': {'normalize_images': False},  # VecNormalize handles normalization
        }
    elif model_name == 'qrdqn':
        model_class = QRDQN
        config['model_kwargs'] = {
            'policy': 'CnnPolicy',
            'learning_rate': 5e-5,  # Reduced from 1e-4 for stability
            'buffer_size': 100_000,
            'learning_starts': 10_000,  # Start learning much earlier!
            'batch_size': 32,
            'gamma': 0.99,
            'train_freq': 4,
            'gradient_steps': 1,
            'target_update_interval': 1_000,
            'exploration_fraction': 0.2,
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': 0.05,
            'policy_kwargs': {'normalize_images': False},  # VecNormalize handles normalization
        }
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Initialize WandB (offline mode to prevent sync issues)
    run = wandb.init(
        project=project or "robotron",
        group=group or f"{model_name}_multidiscrete",
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        mode="offline",  # Offline mode to prevent asyncio sync issues
    )

    run.log_code(name="game_config", include_fn=lambda x: x.endswith(".yaml"))

    # Create parallel environments (all headless for speed)
    print(f"Creating {num_envs} parallel environments...")
    envs = SubprocVecEnv([
        make_env(config_path, level=1, lives=5, rank=i, seed=42, headless=True)
        for i in range(num_envs)
    ])

    # Video recording setup
    if video_freq > 0:
        print(f"✅ Video recording enabled: every {video_freq:,} steps")
        print(f"   Videos use temporary env (no FPS penalty during training)")
    else:
        print("Video recording disabled")

    # Normalize observations ONLY (reward normalization kills learning signal in early training)
    # Note: Frame stacking is now done per-environment before vectorization
    envs = VecNormalize(envs, norm_obs=True, norm_reward=False, clip_obs=10.)

    # Note: Evaluation env is created inside RawScoreEvalCallback (without VecNormalize)

    # Create model
    print(f"Creating {model_name} model...")
    if resume_path:
        model = model_class.load(
            path=resume_path,
            env=envs,
            verbose=1,
            device=device,
            tensorboard_log=f"runs/{run.id}",
        )
    else:
        model = model_class(
            env=envs,
            verbose=1,
            tensorboard_log=f"runs/{run.id}",
            device=device,
            **config['model_kwargs']
        )

    # Callbacks
    callbacks = [
        # Custom metrics tracking
        MetricsCallback(verbose=1),
        # Save VecNormalize stats periodically
        VecNormalizeCallback(
            vec_normalize_env=envs,
            save_path=f"models/{run.id}/vec_normalize.pkl",
            save_freq=100_000,
            verbose=1,
        ),
        # Video recording (temporary env, no FPS impact)
        VideoRecordingCallback(
            record_freq=video_freq,
            video_length=video_length,
            config_path=config_path,
            level=1,
            lives=5,
            verbose=1,
        ) if video_freq > 0 else None,
        # WandB logging
        WandbCallback(
            gradient_save_freq=1000,
            model_save_freq=100_000,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
        # Checkpoint every 100k steps (for earlier evaluation)
        CheckpointCallback(
            save_freq=100_000 // num_envs,  # Adjust for parallel envs
            save_path=f"models/{run.id}/checkpoints",
            name_prefix=f"{model_name}_checkpoint",
        ),
        # Raw score evaluation (logs actual game scores, not normalized rewards)
        RawScoreEvalCallback(
            config_path=config_path,
            level=1,
            lives=5,
            eval_freq=10_000 // num_envs,  # Adjust for parallel envs
            n_eval_episodes=5,
            verbose=1,
        ),
        # Live gameplay display (popup window to watch agent play)
        LiveGameplayCallback(
            config_path=config_path,
            level=1,
            lives=5,
            display_freq=display_freq,
            verbose=1,
        ) if display_freq > 0 else None,
    ]

    # Filter out None callbacks
    callbacks = [cb for cb in callbacks if cb is not None]

    # Train
    print(f"Starting training for {config['total_timesteps']} timesteps...")
    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=callbacks,
    )

    # Save final model
    model.save(f"models/{run.id}/final_model")
    envs.save(f"models/{run.id}/vec_normalize.pkl")

    print("Training complete!")
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Robotron 2084 with improved settings")
    parser.add_argument("--model", type=str, required=True, choices=['ppo', 'dqn', 'qrdqn'],
                       help="Model type to train")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to game config YAML")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--project", type=str, default=None,
                       help="WandB project name")
    parser.add_argument("--group", type=str, default=None,
                       help="WandB group name")
    parser.add_argument("--device", type=str, default='cuda:0',
                       help="Device to train on")
    parser.add_argument("--num-envs", type=int, default=8,
                       help="Number of parallel environments")
    parser.add_argument("--no-curriculum", action='store_true',
                       help="Disable curriculum learning")
    parser.add_argument("--video-freq", type=int, default=100_000,
                       help="Record video every N steps (0 to disable, default 100k)")
    parser.add_argument("--video-length", type=int, default=500,
                       help="Number of frames per video (~16 seconds at 30fps)")
    parser.add_argument("--display-freq", type=int, default=0,
                       help="Show live gameplay popup every N steps (0 to disable)")

    args = parser.parse_args()
    main(
        model_name=args.model,
        config_path=args.config,
        resume_path=args.resume,
        project=args.project,
        group=args.group,
        device=args.device,
        num_envs=args.num_envs,
        use_curriculum=not args.no_curriculum,
        video_freq=args.video_freq,
        video_length=args.video_length,
        display_freq=args.display_freq,
    )
