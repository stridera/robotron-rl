"""
Diagnose FSM v2 specifically.
Modified version of diagnose_fsm.py for v2.
"""

import sys
# Use v2 FSM
from expert_fsm_v2 import expert_decide_v2 as expert_decide

# Import diagnostic infrastructure
from diagnose_fsm import run_diagnostic_episode, FSMDiagnostic
from robotron import RobotronEnv
import numpy as np
from collections import Counter


def run_diagnostic_episode_v2(config='config.yaml', max_steps=10000):
    """Run one episode with FSM v2."""

    diagnostic = FSMDiagnostic()

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

    step = 0
    done = False

    while not done and step < max_steps:
        sprites = info['data']

        # Get FSM v2 decision
        move_action, fire_action = expert_decide(sprites)

        # Analyze frame
        frame_stats = diagnostic.analyze_frame(sprites, move_action, fire_action, info, step)
        diagnostic.frame_history.append(frame_stats)

        # Execute action
        action = move_action * 9 + fire_action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        step += 1

    # Analyze death
    diagnostic.death_analysis = diagnostic.analyze_death()

    env.close()

    return diagnostic


def run_diagnostic_batch_v2(num_episodes=10, config='config.yaml'):
    """Run multiple episodes with FSM v2."""

    print("="*80)
    print("FSM v2 DIAGNOSTIC BATCH RUN (With Cluster Avoidance)")
    print("="*80)
    print(f"Config: {config}")
    print(f"Episodes: {num_episodes}")
    print()

    all_diagnostics = []

    for i in range(num_episodes):
        print(f"Running episode {i+1}/{num_episodes}...", end=' ')
        diagnostic = run_diagnostic_episode_v2(config)
        all_diagnostics.append(diagnostic)

        final = diagnostic.frame_history[-1]
        print(f"Level {final.level}, Score {final.score}, Steps {final.step}")

    # Aggregate analysis
    print("\n" + "="*80)
    print("AGGREGATE ANALYSIS - FSM v2")
    print("="*80)

    scores = [d.frame_history[-1].score for d in all_diagnostics]
    levels = [d.frame_history[-1].level for d in all_diagnostics]
    steps = [d.frame_history[-1].step for d in all_diagnostics]

    print(f"\nPerformance:")
    print(f"  Avg Score: {np.mean(scores):.1f} ± {np.std(scores):.1f}")
    print(f"  Avg Level: {np.mean(levels):.1f} ± {np.std(levels):.1f}")
    print(f"  Avg Survival: {np.mean(steps):.1f} steps")
    print(f"  Max Level: {max(levels)}")

    # Death causes
    print(f"\nDeath Causes:")
    causes = [d.death_analysis.get('likely_cause', 'Unknown') for d in all_diagnostics]
    cause_counts = Counter(causes)
    for cause, count in cause_counts.most_common():
        print(f"  {cause}: {count}/{num_episodes} ({count/num_episodes*100:.0f}%)")

    # Positioning
    pct_in_danger = []
    avg_edge_distances = []
    for d in all_diagnostics:
        pct_in_danger.append(np.mean([f.in_danger_zone for f in d.frame_history]) * 100)
        avg_edge_distances.append(np.mean([f.distance_to_nearest_edge for f in d.frame_history]))

    print(f"\nPositioning:")
    print(f"  Avg % time in danger zone: {np.mean(pct_in_danger):.1f}%")
    print(f"  Avg distance to edge: {np.mean(avg_edge_distances):.1f} px")

    # Threat exposure
    avg_close_enemies = []
    max_surrounded = []
    for d in all_diagnostics:
        avg_close_enemies.append(np.mean([f.enemies_within_120 for f in d.frame_history]))
        max_surrounded.append(max([f.enemies_within_120 for f in d.frame_history]))

    print(f"\nThreat Exposure:")
    print(f"  Avg enemies within 120px: {np.mean(avg_close_enemies):.1f}")
    print(f"  Max enemies nearby: {np.mean(max_surrounded):.1f}")

    print("\n" + "="*80)
    print("COMPARISON: FSM v1 vs v2")
    print("="*80)

    print("\n  FSM v1 (baseline):")
    print("    - Avg Level: 0.9")
    print("    - Max Level: 1")
    print("    - Death cause: Trapped in center (80%)")
    print("    - % in danger zone: 40.7%")

    print(f"\n  FSM v2 (with cluster avoidance):")
    print(f"    - Avg Level: {np.mean(levels):.1f}")
    print(f"    - Max Level: {max(levels)}")
    print(f"    - Death cause: {cause_counts.most_common(1)[0][0]} ({cause_counts.most_common(1)[0][1]*10}%)")
    print(f"    - % in danger zone: {np.mean(pct_in_danger):.1f}%")

    improvement = ((np.mean(levels) / 0.9) - 1) * 100
    print(f"\n  Improvement: {improvement:+.1f}% level performance")

    print("\n" + "="*80)
    print("NEXT IMPROVEMENTS NEEDED")
    print("="*80)

    if max(levels) < 5:
        print("⚠️  Still not reaching level 5 consistently")
        print()
        print("Key issues to address:")

        if 'Trapped in center' in [c for c, _ in cause_counts.most_common(2)]:
            print("  1. Still getting trapped - need better safe zone navigation")

        if np.mean(pct_in_danger) > 30:
            print("  2. Still spending too much time in center")

        print("  3. Need spawner hunting to clear levels faster")
        print("  4. Need better family collection timing")

    return all_diagnostics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='FSM v2 Diagnostic')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')

    args = parser.parse_args()

    diagnostics = run_diagnostic_batch_v2(args.episodes, args.config)
