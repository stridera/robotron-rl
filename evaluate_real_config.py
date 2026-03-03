"""
Evaluate current best model on REAL config.yaml (not progressive_curriculum.yaml).

Goal: Understand baseline performance and what's needed to reach level 40.
"""
import numpy as np
from robotron import RobotronEnv
from wrappers import MultiDiscreteToDiscrete, FrameSkipWrapper
from position_wrapper import GroundTruthPositionWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Configuration
RL_POLICY = 'models/6l9t4lpc/checkpoints/ppo_progressive_checkpoint_3000000_steps.zip'
VEC_NORMALIZE = 'models/6l9t4lpc/vec_normalize.pkl'
CONFIG = 'config.yaml'  # Real config!
NUM_EPISODES = 10
MAX_STEPS = 10000  # Allow long games

print("=" * 80)
print("EVALUATING ON REAL CONFIG")
print("=" * 80)
print(f"Goal: Reach level 40")
print(f"Current model: {RL_POLICY}")
print(f"Config: {CONFIG} (real game difficulty)")
print()

# Create environment with ground truth positions
def make_env():
    env = RobotronEnv(
        level=1,
        lives=5,
        fps=0,
        config_path=CONFIG,  # Real config
        always_move=True,
        headless=True
    )
    env = MultiDiscreteToDiscrete(env)
    env = FrameSkipWrapper(env, skip=4)
    env = GroundTruthPositionWrapper(env, max_sprites=20)
    return env

print("Creating environment...")
vec_env = DummyVecEnv([make_env])

print("Loading VecNormalize stats...")
vec_env = VecNormalize.load(VEC_NORMALIZE, vec_env)
vec_env.training = False
vec_env.norm_reward = False

print("Loading trained RL policy...")
model = PPO.load(RL_POLICY, env=vec_env, device='cpu')

print()
print(f"Running {NUM_EPISODES} episodes...")
print("-" * 80)

episode_results = []

for episode in range(NUM_EPISODES):
    obs = vec_env.reset()
    total_reward = 0
    steps = 0
    done = False
    max_level = 1

    while not done and steps < MAX_STEPS:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        total_reward += reward[0]
        steps += 1

        # Track max level reached
        if 'level' in info[0]:
            max_level = max(max_level, info[0]['level'])

    score = info[0].get('score', 0)
    kills = score // 100
    level = info[0].get('level', 1)

    episode_results.append({
        'episode': episode + 1,
        'score': score,
        'kills': kills,
        'level': level,
        'max_level': max_level,
        'steps': steps,
        'died': done
    })

    print(f"Episode {episode+1:2d}: "
          f"Level {max_level:2d}, "
          f"Score {score:6d}, "
          f"Kills {kills:3d}, "
          f"Steps {steps:5d}, "
          f"{'DIED' if done else 'MAX STEPS'}")

vec_env.close()

print("-" * 80)
print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

scores = [r['score'] for r in episode_results]
kills = [r['kills'] for r in episode_results]
levels = [r['max_level'] for r in episode_results]

print(f"Average Score:  {np.mean(scores):7.1f} ± {np.std(scores):6.1f}")
print(f"Average Kills:  {np.mean(kills):7.1f} ± {np.std(kills):6.1f}")
print(f"Average Level:  {np.mean(levels):7.1f} ± {np.std(levels):6.1f}")
print(f"Max Level:      {max(levels):7d}")
print(f"Deaths:         {sum(1 for r in episode_results if r['died'])}/{NUM_EPISODES}")
print()

# Level distribution
print("Level Distribution:")
for level in sorted(set(levels)):
    count = sum(1 for r in episode_results if r['max_level'] == level)
    bar = "█" * count
    print(f"  Level {level:2d}: {bar} ({count})")
print()

print("=" * 80)
print("ANALYSIS")
print("=" * 80)

avg_level = np.mean(levels)
max_level_reached = max(levels)

print(f"Current Performance:")
print(f"  - Average level: {avg_level:.1f}")
print(f"  - Best level: {max_level_reached}")
print(f"  - Goal: Level 40")
print(f"  - Gap: {40 - max_level_reached} levels")
print()

if max_level_reached < 5:
    print("❌ CRITICAL: Model struggles on real config (levels 1-4)")
    print()
    print("Likely issues:")
    print("  1. Trained on progressive_curriculum.yaml (easier than real config)")
    print("  2. Real config has more enemies from level 1")
    print("  3. Enemies faster/more aggressive in real config")
    print()
    print("Recommended fixes:")
    print("  - Train longer on progressive curriculum (more experience)")
    print("  - Fine-tune on real config levels 1-10")
    print("  - Increase network capacity (512→1024 hidden units)")
    print("  - Add curriculum that bridges progressive→real")
elif max_level_reached < 10:
    print("⚠️  Model reaches early-mid game but struggles (levels 5-9)")
    print()
    print("Likely issues:")
    print("  1. Network capacity (512x512 may be too small)")
    print("  2. Not enough training on dense enemy scenarios")
    print("  3. Spawner enemies (brains, sphereoids) not prioritized")
    print()
    print("Recommended fixes:")
    print("  - Increase network size (512x512 → 1024x1024)")
    print("  - Train longer (3M steps → 10M+ steps)")
    print("  - Add reward shaping for spawner priorities")
    print("  - Use curriculum with more spawner-heavy levels")
elif max_level_reached < 20:
    print("✓ Model reaches mid-game (levels 10-19)")
    print()
    print("To reach level 40:")
    print("  - Train much longer (10M-20M steps)")
    print("  - Larger network (1024x1024 or 1024x1024x512)")
    print("  - Better reward shaping (prioritize dangerous enemies)")
    print("  - Curriculum focusing on levels 15-30")
elif max_level_reached < 40:
    print("✓✓ Model reaches late-game but not level 40")
    print()
    print("To reach level 40:")
    print("  - Extended training (20M-50M steps)")
    print("  - Very large network (1024x1024x1024)")
    print("  - Expert demonstrations for late-game strategy")
    print("  - Curriculum with repeated level 30-40 practice")
else:
    print("🎉 MODEL REACHES LEVEL 40+!")
    print("Goal achieved!")

print()
print("=" * 80)
