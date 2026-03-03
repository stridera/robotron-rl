"""
Watch FSM play with rendering enabled.
"""

import argparse
from robotron import RobotronEnv

parser = argparse.ArgumentParser(description='Watch FSM Play')
parser.add_argument('--version', type=int, default=6, choices=[1, 2, 3, 4, 5, 6], help='FSM version (1-6)')
parser.add_argument('--fps', type=int, default=30, help='FPS (30=visible, 0=unlimited)')
parser.add_argument('--level', type=int, default=1, help='Starting level')
parser.add_argument('--lives', type=int, default=5, help='Starting lives')
parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
parser.add_argument('--godmode', action='store_true', help='God mode (invincible)')

args = parser.parse_args()

# Import appropriate FSM
if args.version == 1:
    from expert_fsm import expert_decide
    print("Using FSM v1 (baseline)")
elif args.version == 2:
    from expert_fsm_v2 import expert_decide_v2 as expert_decide
    print("Using FSM v2 (with cluster avoidance + spawner hunting)")
elif args.version == 3:
    from expert_fsm_v3 import expert_decide_v3 as expert_decide
    print("Using FSM v3 (with shooting alignment fix)")
elif args.version == 4:
    from expert_fsm_v4 import expert_decide_v4 as expert_decide
    print("Using FSM v4 (family priority + level completion + safe alignment)")
elif args.version == 5:
    from expert_fsm_v5 import expert_decide_v5 as expert_decide
    print("Using FSM v5 (state-based human logic with edge circling)")
else:
    from expert_fsm_v6 import expert_decide_v6 as expert_decide
    print("Using FSM v6 (goal-oriented with entity tracking and collision prediction)")

print(f"FPS: {args.fps}")
print(f"Config: {args.config}")
print(f"Starting level: {args.level}")
print(f"God mode: {args.godmode}")
print()
print("Watch the FSM play! Press Ctrl+C to stop.")
print()

env = RobotronEnv(
    level=args.level,
    lives=args.lives,
    fps=args.fps,
    config_path=args.config,
    always_move=False,
    headless=False,  # RENDER!
    godmode=args.godmode
)

env.reset()
obs, reward, terminated, truncated, info = env.step(0)

steps = 0
total_reward = 0
max_level = args.level

try:
    while True:
        sprites = info['data']

        # FSM decision
        move_action, fire_action = expert_decide(sprites)

        # Execute
        action = move_action * 9 + fire_action
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

        if 'level' in info:
            max_level = max(max_level, info['level'])

        # Print status every 100 steps
        if steps % 100 == 0:
            # Get FSM state if available (v5 has it)
            state_info = ""
            if hasattr(_fsm if 'expert_fsm_v5' in str(expert_decide) else None, 'state'):
                from expert_fsm_v5 import _fsm
                state_info = f" | State: {_fsm.state.name}"

            print(f"Steps: {steps:4d} | Level: {info.get('level', 1)} | Score: {info.get('score', 0):5d} | Lives: {info.get('lives', 0)}{state_info}")

        if terminated or truncated:
            if args.godmode:
                # Reset and continue in god mode
                print(f"\nWave complete! Resetting...")
                env.reset()
                obs, reward, terminated, truncated, info = env.step(0)
            else:
                print(f"\nGame Over!")
                print(f"Final Stats:")
                print(f"  Steps: {steps}")
                print(f"  Max Level: {max_level}")
                print(f"  Score: {info.get('score', 0)}")
                print(f"  Total Reward: {total_reward:.1f}")
                break

except KeyboardInterrupt:
    print(f"\n\nStopped by user")
    print(f"Final Stats:")
    print(f"  Steps: {steps}")
    print(f"  Max Level: {max_level}")
    print(f"  Score: {info.get('score', 0)}")
    print(f"  Total Reward: {total_reward:.1f}")

env.close()
