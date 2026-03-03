"""
Training script with DENSE REWARD SHAPING to combat policy collapse.

Key improvements over train_improved.py:
1. Distance-based rewards (approach enemies, not avoid them)
2. Wall proximity penalties (don't get stuck in corners)
3. Stronger reward scaling (score/50 instead of score/100)
4. Lower gamma for sparse rewards (0.95 instead of 0.99)
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
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback
from video_callback import VideoRecordingCallback
import wandb
import gymnasium as gym

# Import custom wrappers
from wrappers import MultiDiscreteToDiscrete, FrameSkipWrapper


class DenseRewardShapingWrapper(gym.Wrapper):
    """
    Dense reward shaping to give the agent constant feedback:

    1. Score-based rewards (main signal):
       - Killing enemies = +reward (score_delta / 50)
       - Stronger than before (/50 instead of /100)

    2. Distance-based rewards (dense signal):
       - Moving toward nearest enemy = small positive reward
       - Moving away from nearest enemy = small negative reward
       - Helps agent learn to engage, not hide

    3. Wall proximity penalty (anti-camping):
       - Being near walls = small negative reward
       - Prevents corner-camping behavior
       - Encourages staying in center of play area
    """

    def __init__(self, env,
                 distance_reward_scale=0.001,  # Small reward for approaching enemies
                 wall_penalty_scale=0.01,      # Penalty for being near walls
                 score_scale=50.0,             # Scale score (1/50 for stronger signal)
                 verbose=False):
        super().__init__(env)
        self.distance_reward_scale = distance_reward_scale
        self.wall_penalty_scale = wall_penalty_scale
        self.score_scale = score_scale
        self.verbose = verbose

        self.last_score = 0
        self.last_min_distance = None
        self.play_rect = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_score = self.env.unwrapped.engine.score
        self.play_rect = self.env.unwrapped.engine.play_rect

        # Initialize last_min_distance
        player_pos, enemy_positions = self._extract_positions(info)
        if player_pos and enemy_positions:
            self.last_min_distance = self._get_min_distance(player_pos, enemy_positions)
        else:
            self.last_min_distance = None

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

        # Calculate distance to each wall
        left_dist = px
        right_dist = self.play_rect.width - px
        top_dist = py
        bottom_dist = self.play_rect.height - py

        return min(left_dist, right_dist, top_dist, bottom_dist)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 1. SCORE-BASED REWARD (main signal)
        current_score = self.env.unwrapped.engine.score
        score_delta = current_score - self.last_score
        self.last_score = current_score

        # Stronger signal: /50 instead of /100
        # Killing 1 grunt (100 pts) = +2.0 reward (was +1.0)
        score_reward = score_delta / self.score_scale

        # 2. DISTANCE-BASED REWARD (dense signal for movement)
        distance_reward = 0.0
        player_pos, enemy_positions = self._extract_positions(info)

        if player_pos and enemy_positions:
            current_min_distance = self._get_min_distance(player_pos, enemy_positions)

            if self.last_min_distance is not None and current_min_distance is not None:
                # Reward getting closer to enemies, penalize moving away
                distance_delta = self.last_min_distance - current_min_distance
                distance_reward = distance_delta * self.distance_reward_scale

            self.last_min_distance = current_min_distance

        # 3. WALL PROXIMITY PENALTY (anti-camping)
        wall_penalty = 0.0
        if player_pos:
            wall_dist = self._get_wall_distance(player_pos)
            if wall_dist is not None:
                # Penalty increases as we get closer to walls
                # At wall (dist=0): -0.01 penalty
                # At distance 50: -0.001 penalty
                # At distance 100+: ~0 penalty
                wall_threshold = 100.0
                if wall_dist < wall_threshold:
                    normalized_dist = wall_dist / wall_threshold  # 0 to 1
                    wall_penalty = -(1.0 - normalized_dist) * self.wall_penalty_scale

        # TOTAL REWARD
        total_reward = score_reward + distance_reward + wall_penalty

        if self.verbose and (score_delta > 0 or abs(distance_reward) > 0.001 or wall_penalty < -0.005):
            print(f"  Score: {score_reward:+.3f}  Distance: {distance_reward:+.5f}  Wall: {wall_penalty:+.5f}  Total: {total_reward:+.3f}")

        return obs, total_reward, terminated, truncated, info


# Copy MetricsCallback and other callbacks from train_improved.py
class MetricsCallback(BaseCallback):
    """Track additional game metrics like highest level, score, families saved, etc."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_scores = []
        self.episode_levels = []
        self.episode_families = []
        self.highest_level = 0
        self.highest_score = 0

    def _on_step(self):
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    if 'score' in info:
                        score = info['score']
                        self.episode_scores.append(score)
                        self.highest_score = max(self.highest_score, score)
                        self.logger.record('robotron/episode_score', score)
                        self.logger.record('robotron/highest_score', self.highest_score)

                    if 'level' in info:
                        level = info['level'] + 1
                        self.episode_levels.append(level)
                        self.highest_level = max(self.highest_level, level)
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
    """Periodically save VecNormalize statistics during training."""
    def __init__(self, vec_normalize_env, save_path, save_freq=100_000, verbose=0):
        super().__init__(verbose)
        self.vec_normalize_env = vec_normalize_env
        self.save_path = save_path
        self.save_freq = save_freq
        self.last_save = -1

    def _on_step(self):
        milestones = [1000, 10_000, 50_000]
        should_save = False

        if self.num_timesteps in milestones and self.num_timesteps != self.last_save:
            should_save = True
        elif self.num_timesteps >= self.save_freq and self.num_timesteps % self.save_freq == 0:
            should_save = True

        if should_save:
            self.vec_normalize_env.save(self.save_path)
            self.last_save = self.num_timesteps
            if self.verbose > 0:
                print(f"✅ Saved VecNormalize stats at {self.num_timesteps:,} steps")

        return True


class RawScoreEvalCallback(BaseCallback):
    """Custom evaluation callback that logs RAW game scores instead of normalized rewards."""
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
        self.eval_env = make_env(
            config_path=self.config_path,
            level=self.level,
            lives=self.lives,
            rank=0,
            seed=999,
            headless=True,
            use_reward_shaping=False,  # No reward shaping for eval
        )()

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        episode_scores = []
        episode_levels = []
        episode_lengths = []

        for _ in range(self.n_eval_episodes):
            obs, info = self.eval_env.reset()
            done = False
            episode_length = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                episode_length += 1

            if 'score' in info:
                episode_scores.append(info['score'])
            if 'level' in info:
                episode_levels.append(info['level'] + 1)
            episode_lengths.append(episode_length)

        mean_score = np.mean(episode_scores)
        std_score = np.std(episode_scores)
        mean_level = np.mean(episode_levels)
        mean_length = np.mean(episode_lengths)

        self.logger.record('eval_raw/mean_score', mean_score)
        self.logger.record('eval_raw/std_score', std_score)
        self.logger.record('eval_raw/mean_level', mean_level)
        self.logger.record('eval_raw/mean_ep_length', mean_length)
        self.logger.record('eval_raw/best_mean_score', max(self.best_mean_score, mean_score))

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
    """Create a single environment with dense reward shaping."""
    def _init():
        env = RobotronEnv(
            level=level,
            lives=lives,
            fps=0,
            config_path=config_path,
            always_move=True,
            headless=headless,
        )

        env = MultiDiscreteToDiscrete(env)

        if frame_skip > 1:
            env = FrameSkipWrapper(env, skip=frame_skip)

        # Apply DENSE reward shaping
        if use_reward_shaping:
            env = DenseRewardShapingWrapper(env, verbose=False)

        env = GrayscaleObservation(env, keep_dim=False)
        env = ResizeObservation(env, (84, 84))
        env = FrameStackObservation(env, stack_size=4)
        env = Monitor(env)

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
         video_freq: int = 100_000,
         video_length: int = 500):

    config = {
        'model': model_name,
        'env_name': 'robotron',
        'resume_path': resume_path,
        'total_timesteps': 20_000_000,
        'num_envs': num_envs,
        'video_freq': video_freq,
        'video_length': video_length,
        'reward_shaping': 'dense',  # Important marker
        'env': {
            'config_path': config_path,
            'level': 1,
            'lives': 5,
            'fps': 0,
            'always_move': True,
        },
    }

    # PPO with adjusted hyperparameters for dense rewards
    model_class = PPO
    config['model_kwargs'] = {
        'policy': 'CnnPolicy',
        'n_steps': 128,  # Back to 128 (more steps per update)
        'batch_size': 256,
        'n_epochs': 4,
        'gamma': 0.95,  # Lower discount for sparse rewards (was 0.99)
        'gae_lambda': 0.95,
        'clip_range': 0.1,
        'ent_coef': 0.01,  # Lower entropy (was 0.02) - let policy commit to good actions
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'learning_rate': 1.0e-4,
        'policy_kwargs': {'normalize_images': False},
    }

    run = wandb.init(
        project=project or "robotron",
        group=group or f"{model_name}_dense_rewards",
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        mode="offline",
    )

    run.log_code(name="game_config", include_fn=lambda x: x.endswith(".yaml"))

    print(f"Creating {num_envs} parallel environments with DENSE reward shaping...")
    envs = SubprocVecEnv([
        make_env(config_path, level=1, lives=5, rank=i, seed=42, headless=True, use_reward_shaping=True)
        for i in range(num_envs)
    ])

    # Normalize observations only
    envs = VecNormalize(envs, norm_obs=True, norm_reward=False, clip_obs=10.)

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

    callbacks = [
        MetricsCallback(verbose=1),
        VecNormalizeCallback(
            vec_normalize_env=envs,
            save_path=f"models/{run.id}/vec_normalize.pkl",
            save_freq=100_000,
            verbose=1,
        ),
        VideoRecordingCallback(
            record_freq=video_freq,
            video_length=video_length,
            config_path=config_path,
            level=1,
            lives=5,
            verbose=1,
        ) if video_freq > 0 else None,
        WandbCallback(
            gradient_save_freq=1000,
            model_save_freq=100_000,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
        CheckpointCallback(
            save_freq=100_000 // num_envs,
            save_path=f"models/{run.id}/checkpoints",
            name_prefix=f"{model_name}_checkpoint",
        ),
        RawScoreEvalCallback(
            config_path=config_path,
            level=1,
            lives=5,
            eval_freq=10_000 // num_envs,
            n_eval_episodes=5,
            verbose=1,
        ),
    ]

    callbacks = [cb for cb in callbacks if cb is not None]

    print(f"Starting training with DENSE rewards for {config['total_timesteps']} timesteps...")
    print(f"  - Score rewards: /50 (was /100)")
    print(f"  - Distance rewards: approach enemies = +reward")
    print(f"  - Wall penalties: near walls = -reward")
    print(f"  - Gamma: 0.95 (was 0.99)")

    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=callbacks,
    )

    model.save(f"models/{run.id}/final_model")
    envs.save(f"models/{run.id}/vec_normalize.pkl")

    print("Training complete!")
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Robotron with dense reward shaping")
    parser.add_argument("--model", type=str, default="ppo", choices=['ppo'],
                       help="Model type to train (only PPO for now)")
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
    parser.add_argument("--video-freq", type=int, default=100_000,
                       help="Record video every N steps (0 to disable)")
    parser.add_argument("--video-length", type=int, default=500,
                       help="Number of frames per video")

    args = parser.parse_args()
    main(
        model_name=args.model,
        config_path=args.config,
        resume_path=args.resume,
        project=args.project,
        group=args.group,
        device=args.device,
        num_envs=args.num_envs,
        video_freq=args.video_freq,
        video_length=args.video_length,
    )
