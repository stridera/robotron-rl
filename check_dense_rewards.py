"""
Quick diagnostic to check if dense rewards fixed the policy collapse.
Run this after 100k-200k steps to see if the agent is behaving better.
"""
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from robotron import RobotronEnv
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
from wrappers import MultiDiscreteToDiscrete, FrameSkipWrapper
from stable_baselines3.common.monitor import Monitor
import sys

if len(sys.argv) < 2:
    print("Usage: python check_dense_rewards.py <checkpoint_path>")
    print("Example: python check_dense_rewards.py models/cddtkl23/checkpoints/ppo_checkpoint_100000_steps.zip")
    sys.exit(1)

model_path = sys.argv[1]
run_id = model_path.split('/')[1]
vec_normalize_path = f"models/{run_id}/vec_normalize.pkl"

print("="*80)
print(f"CHECKING POLICY BEHAVIOR: {model_path}")
print("="*80)

def make_env():
    env = RobotronEnv(
        level=1,
        lives=5,
        fps=0,
        config_path='curriculum_config.yaml',
        always_move=True,
        headless=True,
    )
    env = MultiDiscreteToDiscrete(env)
    env = FrameSkipWrapper(env, skip=4)
    env = GrayscaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, (84, 84))
    env = FrameStackObservation(env, stack_size=4)
    env = Monitor(env)
    return env

vec_env = DummyVecEnv([make_env])
vec_env = VecNormalize.load(vec_normalize_path, vec_env)
vec_env.training = False
vec_env.norm_reward = False

model = PPO.load(model_path, env=vec_env)

# Test deterministic policy
print("\n" + "="*80)
print("DETERMINISTIC POLICY (what you see during evaluation)")
print("="*80)

obs = vec_env.reset()
det_actions = []

for step in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    det_actions.append(action[0])

    if done[0]:
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
    print(f"\n🚨 PROBLEM: Policy still collapsed!")
    print(f"  Movement diversity: {movement_unique}/8")
    print(f"  Shooting diversity: {shooting_unique}/8")
    print(f"\n  Dense rewards didn't help. Need to try:")
    print(f"  1. Even stronger distance rewards (0.01 instead of 0.001)")
    print(f"  2. Stronger wall penalties (0.1 instead of 0.01)")
    print(f"  3. Even stronger score scaling (/25 instead of /50)")
    print(f"  4. Check if curriculum is too hard")
else:
    print(f"\n✅ SUCCESS: Policy has good diversity!")
    print(f"  Movement using {movement_unique}/8 actions")
    print(f"  Shooting using {shooting_unique}/8 actions")
    print(f"\n  Dense rewards are working! Agent is exploring different actions.")

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

# Check policy entropy
print("\n" + "="*80)
print("POLICY ENTROPY")
print("="*80)

obs = vec_env.reset()
model.policy.set_training_mode(False)
obs_tensor = model.policy.obs_to_tensor(obs)[0]
distribution = model.policy.get_distribution(obs_tensor)
entropy = distribution.entropy()

print(f"\nEntropy: {entropy.mean().item():.6f}")
if entropy.mean().item() < 0.1:
    print(f"  ⚠️  Very low entropy - policy is deterministic")
elif entropy.mean().item() < 1.0:
    print(f"  ✓ Moderate entropy - policy is learning to commit")
else:
    print(f"  ✓ High entropy - policy is exploring")

vec_env.close()

print("\n" + "="*80)
print("DONE")
print("="*80)
