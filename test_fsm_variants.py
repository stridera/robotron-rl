"""
Test multiple FSM parameter variants to find optimal settings.

Based on diagnostic findings:
- 80% deaths from "Trapped in center"
- Spending 36% time in danger zone (too much!)
- Need stronger edge bias

Will test 5 variants:
1. Baseline (current)
2. Strong edge bias (reduce danger zone size, stronger edge pull)
3. Very defensive (larger safety distances)
4. Aggressive (shorter distances, more killing)
5. Adaptive (distance scales with enemy count)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple
from diagnose_fsm import run_diagnostic_episode

@dataclass
class FSMConfig:
    """FSM parameter configuration."""
    name: str

    # Distance thresholds
    immediate_danger: float = 60
    danger_zone: float = 120
    optimal_distance: float = 180
    safe_distance: float = 250

    # Edge circling
    edge_threshold: float = 100
    danger_zone_center: float = 200  # Radius of center danger zone

    # Behavior
    edge_bias_strength: float = 1.0  # Multiplier for edge attraction
    retreat_threshold_enemies: int = 5  # How many close enemies trigger retreat


# Define variants
VARIANTS = [
    FSMConfig(
        name="v1_baseline",
        # Current parameters (from expert_fsm.py)
        immediate_danger=60,
        danger_zone=120,
        optimal_distance=180,
        safe_distance=250,
        edge_threshold=100,
        danger_zone_center=200,
        edge_bias_strength=1.0,
        retreat_threshold_enemies=5,
    ),

    FSMConfig(
        name="v2_strong_edge",
        # Stronger edge bias - stay near walls always
        immediate_danger=60,
        danger_zone=120,
        optimal_distance=180,
        safe_distance=250,
        edge_threshold=150,  # Wider edge zone
        danger_zone_center=150,  # Smaller center danger zone (more of map is "center")
        edge_bias_strength=2.0,  # 2x edge attraction
        retreat_threshold_enemies=4,  # Retreat sooner
    ),

    FSMConfig(
        name="v3_very_defensive",
        # Larger safety bubbles - more conservative
        immediate_danger=80,  # Start dodging earlier
        danger_zone=150,  # Larger danger bubble
        optimal_distance=220,  # Stay farther from enemies
        safe_distance=300,  # Only advance when very safe
        edge_threshold=100,
        danger_zone_center=180,
        edge_bias_strength=1.5,
        retreat_threshold_enemies=4,
    ),

    FSMConfig(
        name="v4_aggressive",
        # Shorter distances - more aggressive killing
        immediate_danger=50,  # Accept more risk
        danger_zone=100,  # Smaller danger bubble
        optimal_distance=150,  # Fight closer
        safe_distance=200,  # Advance more often
        edge_threshold=80,
        danger_zone_center=200,
        edge_bias_strength=0.8,  # Less edge bias
        retreat_threshold_enemies=6,  # Tolerate more enemies
    ),

    FSMConfig(
        name="v5_adaptive",
        # Base parameters that will scale with enemy count
        # (Implementation would need to dynamically adjust in FSM)
        immediate_danger=70,
        danger_zone=130,
        optimal_distance=190,
        safe_distance=260,
        edge_threshold=120,
        danger_zone_center=180,
        edge_bias_strength=1.2,
        retreat_threshold_enemies=4,
    ),
]


def patch_fsm_with_config(config: FSMConfig):
    """
    Monkey-patch the expert_fsm module with new parameters.
    Not elegant but allows testing without rewriting FSM code.
    """
    import expert_fsm

    # Update the class constants
    expert_fsm.ExpertFSM.IMMEDIATE_DANGER = config.immediate_danger
    expert_fsm.ExpertFSM.DANGER_ZONE = config.danger_zone
    expert_fsm.ExpertFSM.OPTIMAL_DISTANCE = config.optimal_distance
    expert_fsm.ExpertFSM.SAFE_DISTANCE = config.safe_distance

    # Update module-level constants
    expert_fsm.EDGE_THRESHOLD = config.edge_threshold
    expert_fsm.DANGER_ZONE_CENTER = config.danger_zone_center

    # Reset the global FSM instance
    expert_fsm._fsm = expert_fsm.ExpertFSM()

    print(f"Patched FSM with config: {config.name}")
    print(f"  Danger distances: {config.immediate_danger}/{config.danger_zone}/{config.optimal_distance}/{config.safe_distance}")
    print(f"  Edge params: edge_threshold={config.edge_threshold}, danger_zone_center={config.danger_zone_center}")


def test_variant(config: FSMConfig, num_episodes: int = 10) -> dict:
    """Test a single variant."""

    print(f"\n{'='*80}")
    print(f"Testing Variant: {config.name}")
    print(f"{'='*80}")

    # Patch FSM with new config
    patch_fsm_with_config(config)

    # Run episodes
    diagnostics = []
    for i in range(num_episodes):
        print(f"  Episode {i+1}/{num_episodes}...", end=' ')
        diag = run_diagnostic_episode('config.yaml', max_steps=10000)
        diagnostics.append(diag)

        final = diag.frame_history[-1]
        print(f"Level {final.level}, Score {final.score}")

    # Aggregate results
    scores = [d.frame_history[-1].score for d in diagnostics]
    levels = [d.frame_history[-1].level for d in diagnostics]
    steps = [d.frame_history[-1].step for d in diagnostics]

    # Death causes
    from collections import Counter
    causes = [d.death_analysis.get('likely_cause', 'Unknown') for d in diagnostics]
    cause_counts = Counter(causes)

    # Positioning
    pct_in_danger = []
    avg_edge_dist = []
    for d in diagnostics:
        pct_in_danger.append(np.mean([f.in_danger_zone for f in d.frame_history]) * 100)
        avg_edge_dist.append(np.mean([f.distance_to_nearest_edge for f in d.frame_history]))

    # Threat exposure
    avg_close_enemies = []
    for d in diagnostics:
        avg_close_enemies.append(np.mean([f.enemies_within_120 for f in d.frame_history]))

    results = {
        'config': config,
        'avg_score': np.mean(scores),
        'std_score': np.std(scores),
        'avg_level': np.mean(levels),
        'max_level': max(levels),
        'avg_survival': np.mean(steps),
        'death_causes': dict(cause_counts),
        'pct_in_danger': np.mean(pct_in_danger),
        'avg_edge_dist': np.mean(avg_edge_dist),
        'avg_close_enemies': np.mean(avg_close_enemies),
    }

    # Print summary
    print(f"\n{config.name} Results:")
    print(f"  Score: {results['avg_score']:.1f} ± {results['std_score']:.1f}")
    print(f"  Level: {results['avg_level']:.1f} (max {results['max_level']})")
    print(f"  Survival: {results['avg_survival']:.0f} steps")
    print(f"  % in danger zone: {results['pct_in_danger']:.1f}%")
    print(f"  Avg edge distance: {results['avg_edge_dist']:.1f} px")
    print(f"  Top death cause: {cause_counts.most_common(1)[0]}")

    return results


def main():
    """Test all variants and compare."""

    print("="*80)
    print("FSM PARAMETER SWEEP")
    print("="*80)
    print(f"Testing {len(VARIANTS)} variants, 10 episodes each")
    print()

    all_results = []

    for variant in VARIANTS:
        results = test_variant(variant, num_episodes=10)
        all_results.append(results)

    # Final comparison
    print("\n" + "="*80)
    print("VARIANT COMPARISON")
    print("="*80)
    print()

    print(f"{'Variant':<20} {'Avg Level':<12} {'Max Level':<12} {'Avg Score':<15} {'% Danger':<12} {'Edge Dist'}")
    print("-"*80)

    for r in all_results:
        print(f"{r['config'].name:<20} "
              f"{r['avg_level']:<12.1f} "
              f"{r['max_level']:<12} "
              f"{r['avg_score']:<15.0f} "
              f"{r['pct_in_danger']:<12.1f} "
              f"{r['avg_edge_dist']:.1f}")

    print()

    # Find best by level
    best_by_level = max(all_results, key=lambda r: r['max_level'])
    print(f"🏆 Best max level: {best_by_level['config'].name} (Level {best_by_level['max_level']})")

    # Find best by average level
    best_by_avg = max(all_results, key=lambda r: r['avg_level'])
    print(f"🏆 Best avg level: {best_by_avg['config'].name} ({best_by_avg['avg_level']:.1f})")

    # Find most consistent (lowest danger zone %)
    best_positioning = min(all_results, key=lambda r: r['pct_in_danger'])
    print(f"🏆 Best positioning: {best_positioning['config'].name} ({best_positioning['pct_in_danger']:.1f}% in danger)")

    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    # Recommend the variant with best max level achievement
    if best_by_level['max_level'] >= 5:
        print(f"✅ Use {best_by_level['config'].name} - reaches level {best_by_level['max_level']}")
        print(f"   This is good enough for imitation learning!")
    elif best_by_level['max_level'] >= 3:
        print(f"✓ Use {best_by_level['config'].name} - reaches level {best_by_level['max_level']}")
        print(f"   Decent baseline, but could improve more")
    else:
        print(f"⚠️  Best variant ({best_by_level['config'].name}) only reaches level {best_by_level['max_level']}")
        print(f"   Need more improvements before imitation learning")
        print(f"   Next steps:")
        print(f"   - Implement cluster avoidance")
        print(f"   - Add spawner hunting mode")
        print(f"   - Improve projectile prediction")


if __name__ == "__main__":
    main()
