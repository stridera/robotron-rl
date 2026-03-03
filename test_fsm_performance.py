"""
Test the existing FSM player on real config to establish expert baseline.
"""
import numpy as np
import time
from robotron import RobotronEnv

# Configuration
CONFIG = 'config.yaml'
NUM_EPISODES = 5
MAX_STEPS = 10000

# FSM imports and setup (copy key parts from robotron_fsm.py)
STAY = 0
UP = 1
UP_RIGHT = 2
RIGHT = 3
DOWN_RIGHT = 4
DOWN = 5
DOWN_LEFT = 6
LEFT = 7
UP_LEFT = 8

# Import the FSM decision function
import sys
sys.path.insert(0, '/home/strider/Code/robotron-rl')
from robotron_fsm import chooseOutputs

print("=" * 80)
print("FSM PLAYER PERFORMANCE TEST")
print("=" * 80)
print(f"Config: {CONFIG} (real game difficulty)")
print(f"Episodes: {NUM_EPISODES}")
print()

episode_results = []

for episode in range(NUM_EPISODES):
    print(f"\nEpisode {episode + 1}/{NUM_EPISODES}")
    print("-" * 40)

    env = RobotronEnv(
        level=1,
        lives=5,
        fps=0,  # No speed limit
        config_path=CONFIG,
        always_move=False,  # FSM uses separate move/fire
        headless=True
    )

    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)

    total_reward = 0
    steps = 0
    max_level = 1
    done = False

    while not done and steps < MAX_STEPS:
        # Get sprite data
        sprites_data = info['data']

        # FSM decision
        move_action, fire_action = chooseOutputs(sprites_data)

        # Encode action (move * 9 + fire for discrete action space)
        action = move_action * 9 + fire_action

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        steps += 1

        # Track max level
        if 'level' in info:
            max_level = max(max_level, info['level'])

        # Print progress every 1000 steps
        if steps % 1000 == 0:
            print(f"  Step {steps}: Level {info.get('level', 1)}, Score {info.get('score', 0)}")

    score = info.get('score', 0)
    level = info.get('level', 1)
    kills = score // 100

    episode_results.append({
        'episode': episode + 1,
        'score': score,
        'kills': kills,
        'level': level,
        'max_level': max_level,
        'steps': steps,
        'died': done
    })

    print(f"  Final: Level {max_level}, Score {score}, Kills {kills}, Steps {steps}")

    env.close()

print()
print("=" * 80)
print("FSM PLAYER SUMMARY")
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

print("Level Distribution:")
for level in sorted(set(levels)):
    count = sum(1 for r in episode_results if r['max_level'] == level)
    bar = "█" * count
    print(f"  Level {level:2d}: {bar} ({count})")
print()

print("=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"FSM Player:        Level {np.mean(levels):.1f} average, {max(levels)} max")
print(f"Current RL Model:  Level 1.4 average, 2 max")
print(f"Goal:              Level 40")
print()

if max(levels) >= 10:
    print("✅ FSM is strong expert (level 10+) - Perfect for imitation learning!")
elif max(levels) >= 5:
    print("✓ FSM is decent (level 5-9) - Good starting point for imitation learning")
else:
    print("⚠️  FSM struggles (level <5) - May need improvement before imitation learning")

print()
print("Next steps:")
print("  1. If FSM reaches level 10+: Collect demonstrations immediately")
print("  2. If FSM reaches level 5-9: Consider improving FSM first")
print("  3. If FSM reaches level <5: Definitely improve FSM before collecting demos")
