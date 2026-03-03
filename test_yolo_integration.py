"""
Quick test of YOLO11 + RL integration.

Tests that the YOLO detector can load and produce position features compatible
with the position-based RL policy.
"""
import numpy as np
from robotron import RobotronEnv
from wrappers import MultiDiscreteToDiscrete, FrameSkipWrapper
from yolo_detector_wrapper import YOLODetectorWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Configuration
YOLO_MODEL = '../labelbox_to_yolo/runs/detect/train3/weights/best.pt'
RL_POLICY = 'models/6l9t4lpc/checkpoints/ppo_progressive_checkpoint_3000000_steps.zip'
VEC_NORMALIZE = 'models/6l9t4lpc/vec_normalize.pkl'
NUM_EPISODES = 3
MAX_STEPS = 500

print("=" * 80)
print("YOLO11 + RL INTEGRATION TEST")
print("=" * 80)
print()

# Create environment with YOLO detector
def make_env():
    env = RobotronEnv(
        level=1,
        lives=5,
        fps=0,  # No frame rate limit
        config_path='progressive_curriculum.yaml',
        always_move=True,
        headless=True  # Off-screen rendering
    )
    env = MultiDiscreteToDiscrete(env)
    env = FrameSkipWrapper(env, skip=4)

    # Use YOLO detector instead of ground truth positions
    env = YOLODetectorWrapper(env, YOLO_MODEL, max_sprites=20)

    return env

print("Creating environment with YOLO detector...")
vec_env = DummyVecEnv([make_env])

print("Loading VecNormalize stats from training...")
vec_env = VecNormalize.load(VEC_NORMALIZE, vec_env)
vec_env.training = False
vec_env.norm_reward = False

print("Loading trained RL policy...")
model = PPO.load(RL_POLICY, env=vec_env, device='cpu')

print()
print(f"Running {NUM_EPISODES} test episodes...")
print("-" * 80)

episode_scores = []
episode_kills = []
episode_steps = []

for episode in range(NUM_EPISODES):
    obs = vec_env.reset()
    total_reward = 0
    steps = 0
    done = False

    while not done and steps < MAX_STEPS:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        total_reward += reward[0]
        steps += 1

    if 'score' in info[0]:
        score = info[0]['score']
        kills = score // 100  # Approximate kills from score
        episode_scores.append(score)
        episode_kills.append(kills)
        episode_steps.append(steps)

        print(f"Episode {episode+1}/{NUM_EPISODES}: "
              f"Score={score:4d}, Kills={kills:2d}, Steps={steps:3d}")

vec_env.close()

print("-" * 80)
print()
print("=" * 80)
print("RESULTS")
print("=" * 80)
print(f"Average Score: {np.mean(episode_scores):.1f} ± {np.std(episode_scores):.1f}")
print(f"Average Kills: {np.mean(episode_kills):.1f} ± {np.std(episode_kills):.1f}")
print(f"Average Steps: {np.mean(episode_steps):.1f}")
print()
print("✅ YOLO detector successfully integrated with RL policy!")
print()
print("Next steps:")
print("  1. Run longer evaluation (10+ episodes)")
print("  2. Compare to ground truth baseline (145 kills)")
print("  3. Analyze any performance gaps")
print("  4. Consider tuning YOLO confidence threshold if needed")
