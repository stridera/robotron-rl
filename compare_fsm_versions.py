"""
Compare FSM versions to quantify improvements.
"""

import numpy as np
from robotron import RobotronEnv


def test_fsm_version(version: int, episodes: int = 10, config: str = 'config.yaml'):
    """Test a specific FSM version."""

    if version == 1:
        from expert_fsm import expert_decide
        name = "v1 (baseline)"
    elif version == 2:
        from expert_fsm_v2 import expert_decide_v2 as expert_decide
        name = "v2 (cluster avoidance)"
    else:
        from expert_fsm_v3 import expert_decide_v3 as expert_decide
        name = "v3 (shooting alignment)"

    print(f"\nTesting FSM {name}...")
    print(f"  Running {episodes} episodes...")

    results = []

    for ep in range(episodes):
        env = RobotronEnv(
            level=1,
            lives=5,
            fps=0,
            config_path=config,
            always_move=False,
            headless=True
        )

        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)

        steps = 0
        max_level = 1
        done = False

        while not done and steps < 10000:
            sprites = info['data']
            move_action, fire_action = expert_decide(sprites)
            action = move_action * 9 + fire_action

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

            if 'level' in info:
                max_level = max(max_level, info['level'])

        score = info.get('score', 0)
        kills = score // 100

        results.append({
            'level': max_level,
            'score': score,
            'kills': kills,
            'steps': steps
        })

        env.close()

    # Compute statistics
    levels = [r['level'] for r in results]
    scores = [r['score'] for r in results]
    kills = [r['kills'] for r in results]
    steps = [r['steps'] for r in results]

    return {
        'name': name,
        'avg_level': np.mean(levels),
        'max_level': max(levels),
        'avg_score': np.mean(scores),
        'std_score': np.std(scores),
        'avg_kills': np.mean(kills),
        'std_kills': np.std(kills),
        'avg_steps': np.mean(steps),
        'std_steps': np.std(steps),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Compare FSM Versions')
    parser.add_argument('--episodes', type=int, default=10, help='Episodes per version')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--versions', type=int, nargs='+', default=[1, 3], help='Versions to compare (e.g., 1 2 3)')

    args = parser.parse_args()

    print("=" * 80)
    print("FSM VERSION COMPARISON")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Episodes per version: {args.episodes}")
    print()

    results = []
    for version in args.versions:
        result = test_fsm_version(version, args.episodes, args.config)
        results.append(result)

    # Print comparison table
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()

    print(f"{'Version':<30} {'Avg Level':<12} {'Max Level':<12} {'Avg Kills':<15} {'Avg Steps':<15}")
    print("-" * 80)

    for r in results:
        print(f"{r['name']:<30} {r['avg_level']:<12.1f} {r['max_level']:<12} "
              f"{r['avg_kills']:<6.1f} ± {r['std_kills']:<6.1f} {r['avg_steps']:<7.1f} ± {r['std_steps']:<6.1f}")

    # Calculate improvements
    if len(results) >= 2:
        baseline = results[0]

        print("\n" + "=" * 80)
        print("IMPROVEMENTS vs BASELINE")
        print("=" * 80)
        print()

        for i, r in enumerate(results[1:], 1):
            level_improve = ((r['avg_level'] / baseline['avg_level']) - 1) * 100
            kills_improve = ((r['avg_kills'] / baseline['avg_kills']) - 1) * 100
            steps_improve = ((r['avg_steps'] / baseline['avg_steps']) - 1) * 100

            print(f"{r['name']}:")
            print(f"  Level performance: {level_improve:+.1f}%")
            print(f"  Kills: {kills_improve:+.1f}%")
            print(f"  Survival time: {steps_improve:+.1f}%")
            print(f"  Max level reached: {r['max_level']} (baseline: {baseline['max_level']})")
            print()

    # Key findings
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print()

    best = max(results, key=lambda r: r['avg_level'])
    print(f"Best performing version: {best['name']}")
    print(f"  Avg Level: {best['avg_level']:.1f}")
    print(f"  Max Level: {best['max_level']}")
    print(f"  Avg Kills: {best['avg_kills']:.1f}")
    print()

    if best['max_level'] >= 2:
        print("✅ FSM is making progress beyond level 1!")
        print("   Next steps:")
        print("   - Optimize alignment threshold")
        print("   - Test on more episodes")
        print("   - Try god mode to see how far it can get")
    else:
        print("⚠️  Still stuck at level 1")
        print("   Need more improvements")
