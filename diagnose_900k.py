"""
Comprehensive diagnostics for run fjmk0ac9 at 900k steps.
Agent still showing collapsed policy behavior despite death penalty removal.
"""
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from robotron import RobotronEnv
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
from wrappers import MultiDiscreteToDiscrete, FrameSkipWrapper
from stable_baselines3.common.monitor import Monitor

print("="*80)
print("COMPREHENSIVE DIAGNOSTICS - Run fjmk0ac9 @ 900k steps")
print("="*80)

# Load model
model_path = "models/fjmk0ac9/checkpoints/ppo_checkpoint_900000_steps.zip"
vec_normalize_path = "models/fjmk0ac9/vec_normalize.pkl"

print(f"\nLoading model: {model_path}")
print(f"Loading VecNormalize: {vec_normalize_path}")

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

# ============================================================================
# 1. VECNORMALIZE STATISTICS
# ============================================================================
print("\n" + "="*80)
print("1. VECNORMALIZE STATISTICS")
print("="*80)

print(f"\nObservation running mean:")
print(f"  Shape: {vec_env.obs_rms.mean.shape}")
print(f"  Range: [{vec_env.obs_rms.mean.min():.6f}, {vec_env.obs_rms.mean.max():.6f}]")
print(f"  Mean: {vec_env.obs_rms.mean.mean():.6f}")
print(f"  Std: {vec_env.obs_rms.mean.std():.6f}")

print(f"\nObservation running variance:")
print(f"  Shape: {vec_env.obs_rms.var.shape}")
print(f"  Range: [{vec_env.obs_rms.var.min():.6f}, {vec_env.obs_rms.var.max():.6f}]")
print(f"  Sqrt(var) range: [{np.sqrt(vec_env.obs_rms.var.min()):.6f}, {np.sqrt(vec_env.obs_rms.var.max()):.6f}]")

print(f"\nObservation count: {vec_env.obs_rms.count}")

# Check for collapsed variance
zero_var_pixels = np.sum(vec_env.obs_rms.var < 1e-8)
if zero_var_pixels > 0:
    print(f"\n⚠️  WARNING: {zero_var_pixels} pixels have near-zero variance!")
    print(f"  This means {zero_var_pixels}/{vec_env.obs_rms.var.size} pixels never change")

# ============================================================================
# 2. OBSERVATION VARIATION TEST
# ============================================================================
print("\n" + "="*80)
print("2. OBSERVATION VARIATION TEST")
print("="*80)

obs = vec_env.reset()
observations = []
actions = []
rewards = []

print(f"\nCollecting 100 steps of data...")

for step in range(100):
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, done, info = vec_env.step(action)

    observations.append(obs.copy())
    actions.append(action.copy())
    rewards.append(reward[0])

    if done[0]:
        obs = vec_env.reset()

observations = np.array(observations)  # (100, 1, 4, 84, 84)
actions = np.array(actions)  # (100, 1)
rewards = np.array(rewards)  # (100,)

print(f"\nNormalized observations:")
print(f"  Shape: {observations.shape}")
print(f"  Mean: {observations.mean():.6f}")
print(f"  Std: {observations.std():.6f}")
print(f"  Min: {observations.min():.6f}")
print(f"  Max: {observations.max():.6f}")

# Temporal variation (how much do observations change over time?)
temporal_std = observations.std(axis=0).mean()
print(f"\nTemporal variation (std across time):")
print(f"  Mean temporal std: {temporal_std:.6f}")

if temporal_std < 0.01:
    print(f"  🚨 CRITICAL: Observations barely change over time!")
    print(f"     VecNormalize has likely destroyed the observation signal.")
elif temporal_std < 0.05:
    print(f"  ⚠️  WARNING: Low temporal variation. Observations may not be informative.")
else:
    print(f"  ✓ Temporal variation seems reasonable")

# ============================================================================
# 3. DETERMINISTIC POLICY BEHAVIOR
# ============================================================================
print("\n" + "="*80)
print("3. DETERMINISTIC POLICY BEHAVIOR")
print("="*80)

obs = vec_env.reset()
det_actions = []

print(f"\nRunning 100 steps in DETERMINISTIC mode...")

for step in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    det_actions.append(action[0])

    if done[0]:
        obs = vec_env.reset()

det_actions = np.array(det_actions)

# Actions come as (N, 2) from the vectorized env
# Extract movement and shooting
if det_actions.ndim == 3:  # (N, 1, 2)
    det_actions = det_actions.squeeze(1)  # (N, 2)

movement_actions = det_actions[:, 0]
shooting_actions = det_actions[:, 1]

print(f"\nDeterministic actions (first 30):")
print(f"  Raw: {det_actions[:30]}")
print(f"  Movement: {movement_actions[:30]}")
print(f"  Shooting: {shooting_actions[:30]}")

movement_unique = len(np.unique(movement_actions))
shooting_unique = len(np.unique(shooting_actions))

print(f"\nAction diversity:")
print(f"  Unique movement: {movement_unique} / 8")
print(f"  Unique shooting: {shooting_unique} / 8")

if movement_unique <= 2 or shooting_unique <= 2:
    print(f"  🚨 CRITICAL: Policy has COLLAPSED to near-constant actions!")

    # Show the dominant actions
    movement_counts = np.bincount(movement_actions, minlength=8)
    shooting_counts = np.bincount(shooting_actions, minlength=8)

    directions = ['center', 'up', 'up-right', 'right', 'down-right', 'down', 'down-left', 'left', 'up-left']

    print(f"\n  Movement action distribution:")
    for i, count in enumerate(movement_counts):
        pct = count / len(movement_actions) * 100
        if pct > 5:
            dir_name = directions[i] if i < 8 else f"action_{i}"
            print(f"    {i} ({dir_name:>10}): {count:>3} ({pct:>5.1f}%)")

    print(f"\n  Shooting action distribution:")
    for i, count in enumerate(shooting_counts):
        pct = count / len(shooting_actions) * 100
        if pct > 5:
            dir_name = directions[i] if i < 8 else f"action_{i}"
            print(f"    {i} ({dir_name:>10}): {count:>3} ({pct:>5.1f}%)")

# ============================================================================
# 4. STOCHASTIC POLICY BEHAVIOR
# ============================================================================
print("\n" + "="*80)
print("4. STOCHASTIC POLICY BEHAVIOR")
print("="*80)

obs = vec_env.reset()
stoch_actions = []

print(f"\nRunning 100 steps in STOCHASTIC mode...")

for step in range(100):
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, done, info = vec_env.step(action)
    stoch_actions.append(action[0])

    if done[0]:
        obs = vec_env.reset()

stoch_actions = np.array(stoch_actions)

# Actions come as (N, 2) from the vectorized env
if stoch_actions.ndim == 3:  # (N, 1, 2)
    stoch_actions = stoch_actions.squeeze(1)  # (N, 2)

stoch_movement = stoch_actions[:, 0]
stoch_shooting = stoch_actions[:, 1]

stoch_movement_unique = len(np.unique(stoch_movement))
stoch_shooting_unique = len(np.unique(stoch_shooting))

print(f"\nStochastic action diversity:")
print(f"  Unique movement: {stoch_movement_unique} / 8")
print(f"  Unique shooting: {stoch_shooting_unique} / 8")

if stoch_movement_unique <= 2 or stoch_shooting_unique <= 2:
    print(f"  🚨 CRITICAL: Even stochastic policy is nearly collapsed!")
else:
    print(f"  ✓ Stochastic policy has reasonable diversity")

# ============================================================================
# 5. POLICY ENTROPY
# ============================================================================
print("\n" + "="*80)
print("5. POLICY ENTROPY")
print("="*80)

obs = vec_env.reset()
model.policy.set_training_mode(False)
obs_tensor = model.policy.obs_to_tensor(obs)[0]
distribution = model.policy.get_distribution(obs_tensor)
entropy = distribution.entropy()

print(f"\nPolicy entropy: {entropy.mean().item():.6f}")
print(f"  (Higher = more random, Lower = more deterministic)")

if entropy.mean().item() < 0.1:
    print(f"  🚨 CRITICAL: Entropy near zero - policy is completely deterministic!")
elif entropy.mean().item() < 0.5:
    print(f"  ⚠️  WARNING: Low entropy - policy has little exploration")
else:
    print(f"  ✓ Entropy seems reasonable")

# ============================================================================
# 6. REWARD SIGNAL ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("6. REWARD SIGNAL ANALYSIS")
print("="*80)

print(f"\nRewards from 100 steps (stochastic policy):")
print(f"  Mean: {rewards.mean():.6f}")
print(f"  Std: {rewards.std():.6f}")
print(f"  Min: {rewards.min():.6f}")
print(f"  Max: {rewards.max():.6f}")
print(f"  Non-zero: {np.count_nonzero(rewards)} / {len(rewards)}")

if rewards.std() < 0.01:
    print(f"  🚨 CRITICAL: Rewards have no variation!")
elif np.count_nonzero(rewards) < 10:
    print(f"  ⚠️  WARNING: Very sparse rewards ({np.count_nonzero(rewards)}/{len(rewards)} non-zero)")
else:
    print(f"  ✓ Reward signal seems present")

# ============================================================================
# 7. OVERALL DIAGNOSIS
# ============================================================================
print("\n" + "="*80)
print("7. OVERALL DIAGNOSIS")
print("="*80)

issues = []

if zero_var_pixels > vec_env.obs_rms.var.size * 0.1:
    issues.append(f"VecNormalize: {zero_var_pixels} pixels have zero variance")

if temporal_std < 0.01:
    issues.append("Observations don't vary over time")

if movement_unique <= 2 or shooting_unique <= 2:
    issues.append("Deterministic policy collapsed to constant actions")

if stoch_movement_unique <= 2 or stoch_shooting_unique <= 2:
    issues.append("Even stochastic policy nearly collapsed")

if entropy.mean().item() < 0.1:
    issues.append("Policy entropy near zero")

if rewards.std() < 0.01:
    issues.append("Rewards have no variation")

if issues:
    print("\n🚨 CRITICAL ISSUES FOUND:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")

    print("\n💡 LIKELY ROOT CAUSES:")

    if "Observations don't vary" in " ".join(issues):
        print("\n  → VecNormalize destroyed observation signal")
        print("    Solution: Train without VecNormalize (norm_obs=False)")

    if "Policy collapsed" in " ".join(issues) and "Observations don't vary" not in " ".join(issues):
        print("\n  → Policy learned a dominant strategy despite varying observations")
        print("    Solutions:")
        print("    1. Increase entropy coefficient (ent_coef=0.02 → 0.05)")
        print("    2. Increase reward scaling (score/100 → score/50)")
        print("    3. Add action diversity bonus")
        print("    4. Check if curriculum is too easy")

    if "entropy near zero" in " ".join(issues):
        print("\n  → Not enough exploration")
        print("    Solution: Increase ent_coef to 0.05 or higher")

    if "Rewards" in " ".join(issues):
        print("\n  → Agent not getting reward signal")
        print("    Solution: Check reward scaling and game difficulty")

else:
    print("\n✅ No obvious issues found")
    print("   Need to investigate training dynamics and hyperparameters")

vec_env.close()

print("\n" + "="*80)
print("DIAGNOSTICS COMPLETE")
print("="*80)
