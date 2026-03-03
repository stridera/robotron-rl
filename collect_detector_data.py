"""
Data collection for sprite detector training.

Runs random policy to collect diverse (frame, sprite_positions) pairs.
Target: 500k training examples with ground truth labels from engine.

Output format:
- frames: (N, 84, 84) - single grayscale frames
- sprite_data: List of N lists of (x, y, sprite_type) tuples
- Saved as npz file for efficient loading
"""
import numpy as np
from robotron import RobotronEnv
from wrappers import MultiDiscreteToDiscrete, FrameSkipWrapper
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
import argparse
from tqdm import tqdm
import os


def make_collection_env(config_path='progressive_curriculum.yaml', headless=True):
    """Create environment for data collection."""
    env = RobotronEnv(
        level=1,
        lives=5,
        fps=0,
        config_path=config_path,
        always_move=True,
        headless=headless
    )

    env = MultiDiscreteToDiscrete(env)
    env = FrameSkipWrapper(env, skip=4)

    # Preprocessing: grayscale + resize only (no frame stacking for data collection)
    env = GrayscaleObservation(env, keep_dim=False)  # (84, 84) grayscale
    env = ResizeObservation(env, shape=(84, 84))

    return env


def collect_data(num_episodes=1000, config_path='progressive_curriculum.yaml',
                 output_path='detector_dataset.npz', max_steps_per_episode=1000):
    """
    Collect training data by running random policy.

    Args:
        num_episodes: Number of episodes to collect
        config_path: Game configuration
        output_path: Where to save dataset
        max_steps_per_episode: Max steps per episode (prevent infinite episodes)

    Expected output:
        ~500k frames if num_episodes=1000, max_steps=500
    """
    print("="*80)
    print("SPRITE DETECTOR DATA COLLECTION")
    print("="*80)
    print(f"Target episodes: {num_episodes}")
    print(f"Max steps per episode: {max_steps_per_episode}")
    print(f"Expected total frames: ~{num_episodes * max_steps_per_episode // 2:,}")
    print(f"Output: {output_path}")
    print("="*80)

    env = make_collection_env(config_path=config_path, headless=True)

    all_frames = []
    all_sprite_data = []

    total_frames = 0

    print("\nCollecting data...")

    for episode in tqdm(range(num_episodes), desc="Episodes"):
        obs, info = env.reset()

        for step in range(max_steps_per_episode):
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            # Collect frame (current observation) and sprite data from engine
            frame = obs  # (84, 84) grayscale frame
            sprite_data = env.unwrapped.engine.get_sprite_data()

            all_frames.append(frame)
            all_sprite_data.append(sprite_data)
            total_frames += 1

            if terminated or truncated:
                break

        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_sprites = np.mean([len(sd) for sd in all_sprite_data[-100:]])
            print(f"\n  Episode {episode + 1}: {total_frames:,} frames collected, "
                  f"avg sprites/frame: {avg_sprites:.1f}")

    env.close()

    print(f"\n{'='*80}")
    print(f"Collection complete!")
    print(f"{'='*80}")
    print(f"Total frames: {total_frames:,}")
    print(f"Total episodes: {num_episodes}")
    print(f"Avg frames/episode: {total_frames / num_episodes:.1f}")

    # Convert frames to numpy array
    print("\nConverting frames to numpy array...")
    all_frames = np.array(all_frames)

    print(f"Frame array shape: {all_frames.shape}")
    print(f"Frame array dtype: {all_frames.dtype}")
    print(f"Frame array size: {all_frames.nbytes / 1e9:.2f} GB")

    # Analyze sprite statistics
    print(f"\n{'='*80}")
    print("Dataset Statistics")
    print(f"{'='*80}")

    sprite_counts = [len(sd) for sd in all_sprite_data]
    print(f"Sprites per frame:")
    print(f"  Min: {np.min(sprite_counts)}")
    print(f"  Max: {np.max(sprite_counts)}")
    print(f"  Mean: {np.mean(sprite_counts):.1f}")
    print(f"  Median: {np.median(sprite_counts):.0f}")

    # Sprite type distribution
    from collections import Counter
    all_sprite_types = []
    for sprite_list in all_sprite_data:
        for x, y, sprite_type in sprite_list:
            all_sprite_types.append(sprite_type)

    type_counts = Counter(all_sprite_types)
    print(f"\nSprite type distribution:")
    for sprite_type, count in type_counts.most_common():
        pct = count / len(all_sprite_types) * 100
        print(f"  {sprite_type:15s}: {count:7,} ({pct:5.1f}%)")

    # Save dataset
    print(f"\n{'='*80}")
    print(f"Saving dataset to {output_path}...")
    print(f"{'='*80}")

    # Save as compressed npz
    np.savez_compressed(
        output_path,
        frames=all_frames,
        sprite_data=np.array(all_sprite_data, dtype=object),
        num_frames=total_frames,
        config_path=config_path
    )

    file_size = os.path.getsize(output_path) / 1e9
    print(f"✅ Dataset saved! File size: {file_size:.2f} GB")

    print(f"\n{'='*80}")
    print("Next steps:")
    print(f"{'='*80}")
    print("1. Train detector: python train_detector.py --dataset detector_dataset.npz")
    print("2. Evaluate detector: python eval_detector.py --model detector.pth")
    print("3. Test with RL: python test_detector_rl.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect training data for sprite detector")
    parser.add_argument("--episodes", type=int, default=1000,
                       help="Number of episodes to collect (default: 1000)")
    parser.add_argument("--config", type=str, default='progressive_curriculum.yaml',
                       help="Game configuration (default: progressive_curriculum.yaml)")
    parser.add_argument("--output", type=str, default='detector_dataset.npz',
                       help="Output file path (default: detector_dataset.npz)")
    parser.add_argument("--max-steps", type=int, default=1000,
                       help="Max steps per episode (default: 1000)")

    args = parser.parse_args()

    collect_data(
        num_episodes=args.episodes,
        config_path=args.config,
        output_path=args.output,
        max_steps_per_episode=args.max_steps
    )
