"""
Evaluation script for progressive curriculum models (max_sprites=20).
"""
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from robotron import RobotronEnv
from wrappers import MultiDiscreteToDiscrete, FrameSkipWrapper
from position_wrapper import GroundTruthPositionWrapper
from stable_baselines3.common.monitor import Monitor
import sys
import gymnasium as gym

if len(sys.argv) < 2:
    print("Usage: python check_progressive_model.py <checkpoint_path>")
    sys.exit(1)

model_path = sys.argv[1]
run_id = model_path.split('/')[1]
vec_normalize_path = f"models/{run_id}/vec_normalize.pkl"

print("="*80)
print(f"EVALUATING PROGRESSIVE MODEL: {model_path}")
print("="*80)

# Simple reward wrapper for evaluation
class SimpleStrongRewardWrapper(gym.Wrapper):
    def __init__(self, env, score_scale=10.0):
        super().__init__(env)
        self.score_scale = score_scale
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
        return obs, reward, terminated, truncated, info

def make_env():
    env = RobotronEnv(
        level=1,
        lives=5,
        fps=0,
        config_path='progressive_curriculum.yaml',
        always_move=True,
        headless=False,  # Show the game
    )
    env = MultiDiscreteToDiscrete(env)
    env = FrameSkipWrapper(env, skip=4)
    env = SimpleStrongRewardWrapper(env)
    env = GroundTruthPositionWrapper(env, max_sprites=20, verbose=False)  # max_sprites=20!
    env = Monitor(env)
    return env

vec_env = DummyVecEnv([make_env])
vec_env = VecNormalize.load(vec_normalize_path, vec_env)
vec_env.training = False
vec_env.norm_reward = False

model = PPO.load(model_path, env=vec_env)

# Test deterministic policy
print("\nTesting deterministic policy (100 steps)...")
print("="*80)

obs = vec_env.reset()
det_actions = []
scores = []

for step in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    det_actions.append(action[0])

    if done[0]:
        if 'score' in info[0]:
            scores.append(info[0]['score'])
        obs = vec_env.reset()

det_actions = np.array(det_actions)

if det_actions.ndim == 3:
    det_actions = det_actions.squeeze(1)

movement_actions = det_actions[:, 0]
shooting_actions = det_actions[:, 1]

print(f"\nFirst 30 actions:")
print(f"  Movement: {movement_actions[:30]}")
print(f"  Shooting: {shooting_actions[:30]}")

movement_unique = len(np.unique(movement_actions))
shooting_unique = len(np.unique(shooting_actions))

print(f"\n✅ Action diversity:")
print(f"  Unique movement: {movement_unique} / 8")
print(f"  Unique shooting: {shooting_unique} / 8")

if movement_unique <= 2 or shooting_unique <= 2:
    print(f"\n❌ PROBLEM: Policy collapsed!")
    print(f"  Movement diversity: {movement_unique}/8")
    print(f"  Shooting diversity: {shooting_unique}/8")
else:
    print(f"\n✅ SUCCESS: Policy has good diversity!")
    print(f"  Movement using {movement_unique}/8 actions")
    print(f"  Shooting using {shooting_unique}/8 actions")

# Check action distribution
movement_counts = np.bincount(movement_actions, minlength=8)
shooting_counts = np.bincount(shooting_actions, minlength=8)

directions = ['center', 'up', 'up-right', 'right', 'down-right', 'down', 'down-left', 'left', 'up-left']

print(f"\nMovement distribution:")
for i, count in enumerate(movement_counts):
    pct = count / len(movement_actions) * 100
    if pct > 5:
        print(f"  {i} ({directions[i]:>10}): {count:>3} ({pct:>5.1f}%)")

print(f"\nShooting distribution:")
for i, count in enumerate(shooting_counts):
    pct = count / len(shooting_actions) * 100
    if pct > 5:
        print(f"  {i} ({directions[i]:>10}): {count:>3} ({pct:>5.1f}%)")

# Run 5 full episodes to check performance at different levels
print("\n" + "="*80)
print("Running 5 full episodes to check performance...")
print("="*80)

episode_results = []

for ep in range(5):
    obs = vec_env.reset()
    total_reward = 0
    steps = 0
    done = False
    max_level_reached = 1

    while not done and steps < 10000:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        total_reward += reward[0]
        steps += 1

        # Track max level reached
        if 'level' in info[0]:
            max_level_reached = max(max_level_reached, info[0]['level'])

    if 'score' in info[0]:
        final_score = info[0]['score']
        kills = final_score // 100
        episode_results.append({
            'score': final_score,
            'kills': kills,
            'steps': steps,
            'max_level': max_level_reached,
            'reward': total_reward
        })

        print(f"\nEpisode {ep+1}:")
        print(f"  Final score: {final_score}")
        print(f"  Kills: {kills}")
        print(f"  Max level reached: {max_level_reached}")
        print(f"  Steps survived: {steps}")
        print(f"  Total reward: {total_reward:.1f}")

vec_env.close()

# Summary
print("\n" + "="*80)
print("EVALUATION SUMMARY")
print("="*80)

if episode_results:
    avg_kills = np.mean([r['kills'] for r in episode_results])
    avg_level = np.mean([r['max_level'] for r in episode_results])
    max_level = max([r['max_level'] for r in episode_results])

    print(f"\nAverage performance (5 episodes):")
    print(f"  Avg kills: {avg_kills:.1f}")
    print(f"  Avg max level: {avg_level:.1f}")
    print(f"  Best level reached: {max_level}")

    print(f"\n📊 Progressive Curriculum Assessment:")

    if max_level >= 19:
        print(f"  🎉 EXCELLENT: Reached Stage 7 (level {max_level})!")
        print(f"     Position-based RL scales to full game difficulty!")
    elif max_level >= 13:
        print(f"  ✅ GOOD: Reached Stage 5 (level {max_level})")
        print(f"     Handles spawners but struggles with full difficulty")
    elif max_level >= 10:
        print(f"  ⚠️  PARTIAL: Reached Stage 4 (level {max_level})")
        print(f"     Handles 10-15 grunts but struggles with spawners")
    elif max_level >= 7:
        print(f"  ⚠️  LIMITED: Reached Stage 3 (level {max_level})")
        print(f"     Handles obstacles/hulks but can't scale to many grunts")
    else:
        print(f"  ❌ POOR: Only reached level {max_level}")
        print(f"     Didn't improve much beyond Phase 1 (1 grunt)")

    if avg_kills >= 30:
        print(f"\n✅ SUCCESS CRITERIA MET: {avg_kills:.1f} kills average (target: 30+)")
        print(f"   Ready to proceed to Phase 2 (sprite detector)")
    elif avg_kills >= 20:
        print(f"\n⚠️  PARTIAL SUCCESS: {avg_kills:.1f} kills average (target: 30+)")
        print(f"   May need more training or larger network")
    else:
        print(f"\n❌ BELOW TARGET: {avg_kills:.1f} kills average (target: 30+)")
        print(f"   Position-based RL needs improvement before Phase 2")

print("\n" + "="*80)
print("EVALUATION COMPLETE")
print("="*80)
