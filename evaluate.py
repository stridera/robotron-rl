"""
Evaluate a trained model on Robotron 2084.

Usage:
    python evaluate.py --model models/run_id/best/best_model.zip
    python evaluate.py --model models/run_id/best/best_model.zip --render --episodes 10
"""
import argparse
import numpy as np
from stable_baselines3 import PPO, DQN
from sb3_contrib import QRDQN
from robotron import RobotronEnv
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from wrappers import MultiDiscreteToDiscrete, FrameSkipWrapper
import time


def evaluate_model(model_path: str,
                   num_episodes: int = 5,
                   render: bool = False,
                   config_path: str = None,
                   level: int = 1,
                   lives: int = 5,
                   vec_normalize_path: str = None):
    """
    Evaluate a trained model.

    Args:
        model_path: Path to saved model
        num_episodes: Number of episodes to evaluate
        render: Whether to render the game
        config_path: Path to game config
        level: Starting level
        lives: Number of lives
        vec_normalize_path: Path to VecNormalize stats (if used during training)
    """
    print(f"Loading model from {model_path}")

    # Detect model type from path or try to load with each
    model_class = None
    for cls in [PPO, DQN, QRDQN]:
        try:
            model = cls.load(model_path)
            model_class = cls
            print(f"Loaded as {cls.__name__}")
            break
        except:
            continue

    if model_class is None:
        raise ValueError("Could not load model - unsupported type")

    # Create environment matching train_improved.py setup EXACTLY
    print(f"Creating environment (level={level}, lives={lives}, render={render})")

    def make_eval_env():
        env = RobotronEnv(
            level=level,
            lives=lives,
            fps=30 if render else 0,  # Visible speed if rendering
            config_path=config_path,
            always_move=True,
            headless=not render,
            render_mode='human' if render else None,
        )

        # IMPORTANT: Match train_improved.py preprocessing order exactly!
        # 1. Convert to MultiDiscrete (if trained with this wrapper)
        env = MultiDiscreteToDiscrete(env)

        # 2. Frame skip (match training - default is 4)
        env = FrameSkipWrapper(env, skip=4)

        # 3. Grayscale (keep_dim=False for 2D output)
        env = GrayscaleObservation(env, keep_dim=False)

        # 4. Resize to 84x84
        env = ResizeObservation(env, (84, 84))

        # 5. Frame stacking BEFORE vectorization (4 frames, channels-first)
        env = FrameStackObservation(env, stack_size=4)

        # 6. Monitor for tracking
        env = Monitor(env)

        return env

    env = DummyVecEnv([make_eval_env])

    # Load normalization stats if available
    if vec_normalize_path:
        print(f"Loading VecNormalize stats from {vec_normalize_path}")
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False  # Don't update stats during evaluation
        env.norm_reward = False

    # Run evaluation episodes
    episode_rewards = []
    episode_lengths = []
    episode_scores = []
    max_levels = []

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        max_level = level
        max_score = 0

        print(f"\n{'='*50}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*50}")

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            episode_reward += reward[0]
            episode_length += 1

            # Extract info from vectorized env
            if 'episode' in info[0]:
                # Episode finished
                final_info = info[0]['episode']
                max_score = final_info.get('r', max_score)
            elif len(info) > 0 and 'score' in info[0]:
                max_score = max(max_score, info[0]['score'])
                max_level = max(max_level, info[0].get('level', level))

            if render:
                time.sleep(0.01)  # Slight delay for human viewing

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_scores.append(max_score)
        max_levels.append(max_level)

        print(f"Episode {episode + 1} finished:")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Length: {episode_length} steps")
        print(f"  Score: {max_score}")
        print(f"  Max Level: {max_level}")

    # Summary statistics
    print(f"\n{'='*50}")
    print("EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"Episodes: {num_episodes}")
    print(f"\nRewards:")
    print(f"  Mean: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Min:  {np.min(episode_rewards):.2f}")
    print(f"  Max:  {np.max(episode_rewards):.2f}")
    print(f"\nEpisode Length:")
    print(f"  Mean: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"  Min:  {np.min(episode_lengths)}")
    print(f"  Max:  {np.max(episode_lengths)}")
    print(f"\nGame Score:")
    print(f"  Mean: {np.mean(episode_scores):.0f} ± {np.std(episode_scores):.0f}")
    print(f"  Min:  {np.min(episode_scores):.0f}")
    print(f"  Max:  {np.max(episode_scores):.0f}")
    print(f"\nMax Level Reached:")
    print(f"  Mean: {np.mean(max_levels):.1f}")
    print(f"  Max:  {np.max(max_levels)}")
    print(f"{'='*50}\n")

    env.close()

    return {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'scores': episode_scores,
        'levels': max_levels,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained Robotron model")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model (.zip file)")
    parser.add_argument("--episodes", type=int, default=5,
                       help="Number of episodes to evaluate")
    parser.add_argument("--render", action="store_true",
                       help="Render the game (slower)")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to game config YAML")
    parser.add_argument("--level", type=int, default=1,
                       help="Starting level")
    parser.add_argument("--lives", type=int, default=5,
                       help="Number of lives")
    parser.add_argument("--vec-normalize", type=str, default=None,
                       help="Path to vec_normalize.pkl (if used during training)")

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        num_episodes=args.episodes,
        render=args.render,
        config_path=args.config,
        level=args.level,
        lives=args.lives,
        vec_normalize_path=args.vec_normalize,
    )
