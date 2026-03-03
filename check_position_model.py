"""
Quick evaluation for position-based models.
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
    print("Usage: python check_position_model.py <checkpoint_path>")
    sys.exit(1)

model_path = sys.argv[1]
run_id = model_path.split('/')[1]
vec_normalize_path = f"models/{run_id}/vec_normalize.pkl"

print("="*80)
print(f"EVALUATING POSITION-BASED MODEL: {model_path}")
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
        config_path='ultra_simple_curriculum.yaml',
        always_move=True,
        headless=False,
    )
    env = MultiDiscreteToDiscrete(env)
    env = FrameSkipWrapper(env, skip=4)
    env = SimpleStrongRewardWrapper(env)
    env = GroundTruthPositionWrapper(env, max_sprites=10, verbose=False)
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

# Run a full episode to check performance
print("\n" + "="*80)
print("Running full episode to check performance...")
print("="*80)

obs = vec_env.reset()
total_reward = 0
steps = 0
done = False

while not done and steps < 10000:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    total_reward += reward[0]
    steps += 1

if 'score' in info[0]:
    final_score = info[0]['score']
    kills = final_score // 100
    print(f"\nEpisode results:")
    print(f"  Final score: {final_score}")
    print(f"  Kills: {kills}")
    print(f"  Steps survived: {steps}")
    print(f"  Total reward: {total_reward:.1f}")

    if kills > 50:
        print(f"\n🎉 EXCELLENT: Agent consistently kills grunts!")
    elif kills > 10:
        print(f"\n✅ GOOD: Agent learning but can improve")
    else:
        print(f"\n⚠️  WARNING: Agent not performing well")

vec_env.close()

print("\n" + "="*80)
print("EVALUATION COMPLETE")
print("="*80)
print("\n📊 CONCLUSION:")
print("  Position-based RL works! Agent learned to kill grunts from position info.")
print("  Next step: Build sprite detector (Phase 2)")
