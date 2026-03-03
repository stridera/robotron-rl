"""
Diagnose why the policy is outputting fixed actions regardless of observations.
Check if VecNormalize or something else is destroying the observation signal.
"""
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from robotron import RobotronEnv
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
from wrappers import MultiDiscreteToDiscrete, FrameSkipWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

print("="*80)
print("OBSERVATION PIPELINE DIAGNOSTICS")
print("="*80)

# Load model and VecNormalize
model_path = "models/6nqhjgz2/checkpoints/ppo_checkpoint_500000_steps.zip"
vec_normalize_path = "models/6nqhjgz2/vec_normalize.pkl"

print(f"\nLoading model: {model_path}")
print(f"Loading VecNormalize: {vec_normalize_path}")

# Create single env with SAME pipeline as training
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

    # NO REWARD SHAPING for diagnostics

    env = GrayscaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, (84, 84))
    env = FrameStackObservation(env, stack_size=4)
    env = Monitor(env)
    return env

vec_env = DummyVecEnv([make_env])

# Load VecNormalize stats
vec_env = VecNormalize.load(vec_normalize_path, vec_env)
vec_env.training = False
vec_env.norm_reward = False

# Load model
model = PPO.load(model_path, env=vec_env)

print("\n" + "="*80)
print("VECNORMALIZE STATISTICS")
print("="*80)

print(f"\nObservation running mean stats:")
print(f"  Shape: {vec_env.obs_rms.mean.shape}")
print(f"  Mean range: [{vec_env.obs_rms.mean.min():.6f}, {vec_env.obs_rms.mean.max():.6f}]")
print(f"  Mean std: {vec_env.obs_rms.mean.std():.6f}")

print(f"\nObservation running variance stats:")
print(f"  Shape: {vec_env.obs_rms.var.shape}")
print(f"  Var range: [{vec_env.obs_rms.var.min():.6f}, {vec_env.obs_rms.var.max():.6f}]")
print(f"  Sqrt(var) range: [{np.sqrt(vec_env.obs_rms.var.min()):.6f}, {np.sqrt(vec_env.obs_rms.var.max()):.6f}]")

print(f"\nObservation count: {vec_env.obs_rms.count}")

# Check if variance is near zero (would cause division issues)
if np.any(vec_env.obs_rms.var < 1e-8):
    print(f"\n⚠️  WARNING: Some variances are near zero! This causes normalization issues.")
    print(f"  Pixels with var < 1e-8: {np.sum(vec_env.obs_rms.var < 1e-8)} / {vec_env.obs_rms.var.size}")

print("\n" + "="*80)
print("OBSERVATION VARIATION TEST")
print("="*80)

# Reset and collect observations over multiple steps
raw_obs_list = []
norm_obs_list = []
action_list = []

obs = vec_env.reset()
print(f"\nInitial observation (after VecNormalize):")
print(f"  Shape: {obs.shape}")
print(f"  Range: [{obs.min():.6f}, {obs.max():.6f}]")
print(f"  Mean: {obs.mean():.6f}, Std: {obs.std():.6f}")

# Collect 50 steps of data
for step in range(50):
    # Get raw obs BEFORE normalization (need to peek into wrapper)
    # We'll just check normalized obs variation

    # Get action from policy
    action, _ = model.predict(obs, deterministic=False)

    # Step
    obs, reward, done, info = vec_env.step(action)

    norm_obs_list.append(obs.copy())
    action_list.append(action.copy())

    if done[0]:
        obs = vec_env.reset()

# Convert to arrays
norm_obs_array = np.array(norm_obs_list)  # (50, 1, 4, 84, 84)
action_array = np.array(action_list)  # (50, 1, 2) for MultiDiscrete([8,8])

print(f"\nCollected {len(norm_obs_list)} steps")

print(f"\nNormalized observation statistics:")
print(f"  Shape: {norm_obs_array.shape}")
print(f"  Overall mean: {norm_obs_array.mean():.6f}")
print(f"  Overall std: {norm_obs_array.std():.6f}")
print(f"  Min: {norm_obs_array.min():.6f}, Max: {norm_obs_array.max():.6f}")

# Check temporal variation
temporal_std = norm_obs_array.std(axis=0).mean()
print(f"\nTemporal variation (std across time steps):")
print(f"  Mean temporal std: {temporal_std:.6f}")

if temporal_std < 0.01:
    print(f"  ⚠️  CRITICAL: Observations barely vary over time!")
    print(f"  This means the policy sees nearly identical inputs every step.")
    print(f"  VecNormalize has likely destroyed the observation signal.")

# Check action variation
print(f"\nAction statistics:")
print(f"  Shape: {action_array.shape}")
print(f"  Movement actions (0-7): {action_array[:, 0, 0][:20]}")
print(f"  Shooting actions (0-7): {action_array[:, 0, 1][:20]}")

movement_unique = len(np.unique(action_array[:, 0, 0]))
shooting_unique = len(np.unique(action_array[:, 0, 1]))

print(f"\nAction diversity:")
print(f"  Unique movement actions: {movement_unique} / 8")
print(f"  Unique shooting actions: {shooting_unique} / 8")

if movement_unique == 1 or shooting_unique == 1:
    print(f"  ⚠️  CRITICAL: Policy is outputting constant actions!")
    print(f"  Movement: always {action_array[0, 0, 0]}")
    print(f"  Shooting: always {action_array[0, 0, 1]}")

# Check policy entropy
print("\n" + "="*80)
print("POLICY ANALYSIS")
print("="*80)

# Get policy outputs
obs = vec_env.reset()
with model.policy.set_training_mode(False):
    obs_tensor = model.policy.obs_to_tensor(obs)[0]
    distribution = model.policy.get_distribution(obs_tensor)
    entropy = distribution.entropy()

print(f"\nPolicy entropy: {entropy.mean().item():.6f}")
print(f"  (Higher = more random, Lower = more deterministic)")

if entropy.mean().item() < 0.5:
    print(f"  ⚠️  WARNING: Very low entropy - policy is highly deterministic")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

issues = []

if np.any(vec_env.obs_rms.var < 1e-8):
    issues.append("VecNormalize variance collapsed (near zero)")

if temporal_std < 0.01:
    issues.append("Observations don't vary over time (VecNormalize destroyed signal)")

if movement_unique == 1 or shooting_unique == 1:
    issues.append("Policy outputs constant actions")

if entropy.mean().item() < 0.5:
    issues.append("Policy entropy too low (no exploration)")

if issues:
    print("\n🚨 CRITICAL ISSUES FOUND:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")

    print("\n💡 LIKELY ROOT CAUSE:")
    if "VecNormalize" in " ".join(issues):
        print("  VecNormalize has destroyed the observation signal.")
        print("  All observations look the same after normalization.")
        print("  The policy can't distinguish different game states.")
        print("\n💡 SOLUTION:")
        print("  Option 1: Don't use VecNormalize for observations (norm_obs=False)")
        print("  Option 2: Use VecNormalize with epsilon=1e-4 instead of default 1e-8")
        print("  Option 3: Clip observations more gently (clip_obs=100 instead of 10)")
        print("  Option 4: Don't clip at all (clip_obs=np.inf)")
else:
    print("\n✅ No obvious issues found - need deeper investigation")

vec_env.close()
