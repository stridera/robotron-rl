"""
Check what the policy does in DETERMINISTIC mode (argmax).
This is what you see during evaluation.
"""
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from robotron import RobotronEnv
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
from wrappers import MultiDiscreteToDiscrete, FrameSkipWrapper
from stable_baselines3.common.monitor import Monitor

print("="*80)
print("DETERMINISTIC POLICY BEHAVIOR CHECK")
print("="*80)

# Load model
model_path = "models/6nqhjgz2/checkpoints/ppo_checkpoint_500000_steps.zip"
vec_normalize_path = "models/6nqhjgz2/vec_normalize.pkl"

def make_env():
    env = RobotronEnv(level=1, lives=5, fps=0, config_path='curriculum_config.yaml',
                      always_move=True, headless=True)
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

print("\nRunning 100 steps in DETERMINISTIC mode (argmax actions):")
print("="*80)

obs = vec_env.reset()
actions_det = []

for step in range(100):
    action, _ = model.predict(obs, deterministic=True)  # DETERMINISTIC
    obs, reward, done, info = vec_env.step(action)
    actions_det.append(action[0].copy())  # [movement, shooting]

    if done[0]:
        obs = vec_env.reset()

actions_det = np.array(actions_det)

print(f"\nDeterministic actions:")
print(f"  Movement: {actions_det[:30, 0]}")
print(f"  Shooting: {actions_det[:30, 1]}")

movement_unique = len(np.unique(actions_det[:, 0]))
shooting_unique = len(np.unique(actions_det[:, 1]))

print(f"\nAction diversity (deterministic):")
print(f"  Unique movement: {movement_unique} / 8")
print(f"  Unique shooting: {shooting_unique} / 8")

# Check if there's a dominant action
movement_counts = np.bincount(actions_det[:, 0], minlength=8)
shooting_counts = np.bincount(actions_det[:, 1], minlength=8)

print(f"\nMovement action frequencies:")
for i, count in enumerate(movement_counts):
    pct = count / len(actions_det) * 100
    if pct > 5:  # Only show actions used >5%
        direction = ['center', 'up', 'up-right', 'right', 'down-right', 'down', 'down-left', 'left', 'up-left'][i]
        print(f"  {i} ({direction:>10}): {count:>3} ({pct:>5.1f}%)")

print(f"\nShooting action frequencies:")
for i, count in enumerate(shooting_counts):
    pct = count / len(actions_det) * 100
    if pct > 5:
        direction = ['center', 'up', 'up-right', 'right', 'down-right', 'down', 'down-left', 'left', 'up-left'][i]
        print(f"  {i} ({direction:>10}): {count:>3} ({pct:>5.1f}%)")

# Check for stuck pattern (same action repeated)
movement_repeats = (actions_det[1:, 0] == actions_det[:-1, 0]).sum()
shooting_repeats = (actions_det[1:, 1] == actions_det[:-1, 1]).sum()

print(f"\nAction persistence (how often same action repeats):")
print(f"  Movement: {movement_repeats}/{len(actions_det)-1} ({movement_repeats/(len(actions_det)-1)*100:.1f}%)")
print(f"  Shooting: {shooting_repeats}/{len(actions_det)-1} ({shooting_repeats/(len(actions_det)-1)*100:.1f}%)")

if movement_repeats > 80 or shooting_repeats > 80:
    print(f"\n⚠️  PROBLEM: Policy is very repetitive in deterministic mode!")
    print(f"  The agent has learned a near-constant action policy.")
    print(f"\n  This happened because:")
    print(f"  1. Death penalty (-10) was too strong")
    print(f"  2. Agent learned 'safe' policy: run to corner, shoot in one direction")
    print(f"  3. This avoids death but doesn't score")
    print(f"\n  The fix (removing death penalty) should help in the NEXT training run.")

# Now check stochastic
print("\n" + "="*80)
print("Comparing with STOCHASTIC mode:")
print("="*80)

obs = vec_env.reset()
actions_stoch = []

for step in range(100):
    action, _ = model.predict(obs, deterministic=False)  # STOCHASTIC
    obs, reward, done, info = vec_env.step(action)
    actions_stoch.append(action[0].copy())

    if done[0]:
        obs = vec_env.reset()

actions_stoch = np.array(actions_stoch)

stoch_movement_unique = len(np.unique(actions_stoch[:, 0]))
stoch_shooting_unique = len(np.unique(actions_stoch[:, 1]))

print(f"\nStochastic action diversity:")
print(f"  Unique movement: {stoch_movement_unique} / 8")
print(f"  Unique shooting: {stoch_shooting_unique} / 8")

print(f"\n" + "="*80)
print("CONCLUSION:")
print("="*80)

if movement_unique <= 2 or shooting_unique <= 2:
    print("\n🚨 Policy has COLLAPSED to near-constant actions in deterministic mode")
    print("  This model learned a 'safe but useless' strategy")
    print("  It survives by hiding but doesn't kill enemies")
    print("\n✅ The fix (no death penalty) should prevent this in the next run")
else:
    print("\n✅ Policy seems reasonably diverse")
    print("  May need to investigate other issues")

vec_env.close()
