"""
FSM Diagnostic Tool

Visualizes FSM decision-making to understand failure patterns.
Records detailed stats on why/how the FSM dies.
"""

import numpy as np
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from robotron import RobotronEnv
from expert_fsm import expert_decide

@dataclass
class FrameStats:
    """Statistics for a single frame."""
    step: int
    player_x: float
    player_y: float
    move_action: int
    fire_action: int

    # Enemy counts by type
    total_enemies: int
    spawners: int  # Brain, Sphereoid, Quark
    shooters: int  # Enforcer, Tank
    regular: int   # Grunt

    # Distances
    closest_enemy_dist: float
    closest_projectile_dist: float
    closest_family_dist: float

    # Positioning
    distance_to_center: float
    distance_to_nearest_edge: float
    in_danger_zone: bool  # In center 200x200 area

    # Threats
    enemies_within_60: int   # IMMEDIATE_DANGER
    enemies_within_120: int  # DANGER_ZONE
    enemies_within_180: int  # OPTIMAL_DISTANCE
    projectiles_close: int   # Within 120

    # Game state
    score: int
    level: int
    lives: int


class FSMDiagnostic:
    """Records and analyzes FSM gameplay."""

    def __init__(self):
        self.frame_history: List[FrameStats] = []
        self.death_analysis = None

    def analyze_frame(self, sprites, move_action, fire_action, info, step) -> FrameStats:
        """Analyze a single frame."""

        # Find player
        player = None
        for sprite in sprites:
            if sprite[2] == 'Player':
                player = sprite
                break

        if not player:
            # Player not found - return dummy stats
            return FrameStats(
                step=step, player_x=0, player_y=0,
                move_action=move_action, fire_action=fire_action,
                total_enemies=0, spawners=0, shooters=0, regular=0,
                closest_enemy_dist=999, closest_projectile_dist=999, closest_family_dist=999,
                distance_to_center=0, distance_to_nearest_edge=0, in_danger_zone=False,
                enemies_within_60=0, enemies_within_120=0, enemies_within_180=0,
                projectiles_close=0,
                score=info.get('score', 0), level=info.get('level', 1), lives=info.get('lives', 0)
            )

        px, py = player[0], player[1]

        # Categorize sprites
        spawners = []
        shooters = []
        regular_enemies = []
        projectiles = []
        family = []

        for sprite in sprites:
            x, y, stype = sprite

            if stype in ['Brain', 'Sphereoid', 'Quark']:
                spawners.append((x, y))
            elif stype in ['Enforcer', 'Tank']:
                shooters.append((x, y))
            elif stype in ['Grunt']:
                regular_enemies.append((x, y))
            elif stype in ['EnforcerBullet', 'TankShell', 'CruiseMissile']:
                projectiles.append((x, y))
            elif stype in ['Mommy', 'Daddy', 'Mikey']:
                family.append((x, y))

        all_enemies = spawners + shooters + regular_enemies

        # Calculate distances
        def get_dist(x, y):
            return np.hypot(px - x, py - y)

        closest_enemy = min([get_dist(x, y) for x, y in all_enemies], default=999)
        closest_projectile = min([get_dist(x, y) for x, y in projectiles], default=999)
        closest_family = min([get_dist(x, y) for x, y in family], default=999)

        # Position analysis
        center_x, center_y = 665/2, 492/2
        dist_to_center = get_dist(center_x, center_y)

        dist_to_edges = [px, 665 - px, py, 492 - py]
        dist_to_nearest_edge = min(dist_to_edges)

        in_danger_zone = abs(px - center_x) < 200 and abs(py - center_y) < 200

        # Threat counts
        enemies_60 = sum(1 for x, y in all_enemies if get_dist(x, y) < 60)
        enemies_120 = sum(1 for x, y in all_enemies if get_dist(x, y) < 120)
        enemies_180 = sum(1 for x, y in all_enemies if get_dist(x, y) < 180)
        projectiles_120 = sum(1 for x, y in projectiles if get_dist(x, y) < 120)

        stats = FrameStats(
            step=step,
            player_x=px,
            player_y=py,
            move_action=move_action,
            fire_action=fire_action,
            total_enemies=len(all_enemies),
            spawners=len(spawners),
            shooters=len(shooters),
            regular=len(regular_enemies),
            closest_enemy_dist=closest_enemy,
            closest_projectile_dist=closest_projectile,
            closest_family_dist=closest_family,
            distance_to_center=dist_to_center,
            distance_to_nearest_edge=dist_to_nearest_edge,
            in_danger_zone=in_danger_zone,
            enemies_within_60=enemies_60,
            enemies_within_120=enemies_120,
            enemies_within_180=enemies_180,
            projectiles_close=projectiles_120,
            score=info.get('score', 0),
            level=info.get('level', 1),
            lives=info.get('lives', 0)
        )

        return stats

    def analyze_death(self) -> Dict[str, Any]:
        """Analyze the last 50 frames before death."""
        if len(self.frame_history) < 2:
            return {}

        death_frame = self.frame_history[-1]
        last_50 = self.frame_history[-min(50, len(self.frame_history)):]

        analysis = {
            'death_step': death_frame.step,
            'death_score': death_frame.score,
            'death_level': death_frame.level,
            'death_position': (death_frame.player_x, death_frame.player_y),

            # What killed us?
            'closest_enemy_at_death': death_frame.closest_enemy_dist,
            'closest_projectile_at_death': death_frame.closest_projectile_dist,
            'enemies_very_close': death_frame.enemies_within_60,
            'projectiles_close': death_frame.projectiles_close,

            # Positioning at death
            'was_in_danger_zone': death_frame.in_danger_zone,
            'distance_to_edge': death_frame.distance_to_nearest_edge,

            # Last 50 frames analysis
            'avg_enemies': np.mean([f.total_enemies for f in last_50]),
            'avg_spawners': np.mean([f.spawners for f in last_50]),
            'avg_closest_enemy': np.mean([f.closest_enemy_dist for f in last_50]),
            'avg_in_danger_zone': np.mean([f.in_danger_zone for f in last_50]),
            'avg_edge_distance': np.mean([f.distance_to_nearest_edge for f in last_50]),

            # Did we get swarmed?
            'max_enemies_close': max([f.enemies_within_120 for f in last_50]),
            'times_surrounded': sum(1 for f in last_50 if f.enemies_within_120 >= 5),

            # Family collection
            'family_nearby': sum(1 for f in last_50 if f.closest_family_dist < 100),
        }

        # Likely cause of death
        if death_frame.closest_projectile_dist < 30:
            analysis['likely_cause'] = 'Hit by projectile'
        elif death_frame.enemies_within_60 >= 3:
            analysis['likely_cause'] = 'Surrounded by enemies'
        elif death_frame.in_danger_zone:
            analysis['likely_cause'] = 'Trapped in center'
        elif death_frame.distance_to_nearest_edge < 20:
            analysis['likely_cause'] = 'Cornered at edge'
        else:
            analysis['likely_cause'] = 'Unknown - gradual overwhelm'

        return analysis

    def print_summary(self, episode_num: int):
        """Print episode summary."""
        if not self.frame_history:
            print(f"Episode {episode_num}: No data")
            return

        final = self.frame_history[-1]

        print(f"\n{'='*80}")
        print(f"Episode {episode_num} Summary")
        print(f"{'='*80}")
        print(f"Survived: {final.step} steps")
        print(f"Score: {final.score}")
        print(f"Level: {final.level}")
        print(f"Final Lives: {final.lives}")

        if self.death_analysis:
            print(f"\nDeath Analysis:")
            print(f"  Likely cause: {self.death_analysis['likely_cause']}")
            print(f"  Position: ({self.death_analysis['death_position'][0]:.0f}, {self.death_analysis['death_position'][1]:.0f})")
            print(f"  In danger zone: {self.death_analysis['was_in_danger_zone']}")
            print(f"  Closest enemy: {self.death_analysis['closest_enemy_at_death']:.1f} pixels")
            print(f"  Closest projectile: {self.death_analysis['closest_projectile_at_death']:.1f} pixels")
            print(f"  Enemies within 60px: {self.death_analysis['enemies_very_close']}")
            print(f"  Times surrounded (last 50 frames): {self.death_analysis['times_surrounded']}")
            print(f"  Avg enemies in last 50 frames: {self.death_analysis['avg_enemies']:.1f}")
            print(f"  Max enemies within 120px: {self.death_analysis['max_enemies_close']}")

        # Performance metrics
        avg_enemies = np.mean([f.total_enemies for f in self.frame_history])
        avg_spawners = np.mean([f.spawners for f in self.frame_history])
        avg_closest = np.mean([f.closest_enemy_dist for f in self.frame_history])
        pct_in_danger = np.mean([f.in_danger_zone for f in self.frame_history]) * 100
        avg_edge_dist = np.mean([f.distance_to_nearest_edge for f in self.frame_history])

        print(f"\nPerformance Metrics:")
        print(f"  Avg enemies on screen: {avg_enemies:.1f}")
        print(f"  Avg spawners: {avg_spawners:.1f}")
        print(f"  Avg distance to closest enemy: {avg_closest:.1f} px")
        print(f"  % time in danger zone (center): {pct_in_danger:.1f}%")
        print(f"  Avg distance to edge: {avg_edge_dist:.1f} px")


def run_diagnostic_episode(config='config.yaml', max_steps=10000):
    """Run one episode with full diagnostics."""

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

        # Get FSM decision
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


def run_diagnostic_batch(num_episodes=10, config='config.yaml'):
    """Run multiple episodes and aggregate statistics."""

    print("="*80)
    print("FSM DIAGNOSTIC BATCH RUN")
    print("="*80)
    print(f"Config: {config}")
    print(f"Episodes: {num_episodes}")
    print()

    all_diagnostics = []

    for i in range(num_episodes):
        print(f"Running episode {i+1}/{num_episodes}...", end=' ')
        diagnostic = run_diagnostic_episode(config)
        all_diagnostics.append(diagnostic)

        # Quick summary
        final = diagnostic.frame_history[-1]
        print(f"Level {final.level}, Score {final.score}, Steps {final.step}")

    # Aggregate analysis
    print("\n" + "="*80)
    print("AGGREGATE ANALYSIS")
    print("="*80)

    # Performance
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
    from collections import Counter
    cause_counts = Counter(causes)
    for cause, count in cause_counts.most_common():
        print(f"  {cause}: {count}/{num_episodes} ({count/num_episodes*100:.0f}%)")

    # Positioning analysis
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
    print(f"  Max times surrounded: {np.mean(max_surrounded):.1f} enemies")

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    # Generate recommendations
    if np.mean(pct_in_danger) > 30:
        print("⚠️  Spending too much time in center (danger zone)")
        print("   → Increase edge bias")
        print("   → Make 'retreat to edge' trigger earlier")

    if np.mean(avg_close_enemies) > 5:
        print("⚠️  Too many enemies getting close")
        print("   → Increase DANGER_ZONE threshold")
        print("   → Retreat earlier when enemies approach")

    if 'Surrounded by enemies' in [c for c, _ in cause_counts.most_common(2)]:
        print("⚠️  Frequently getting surrounded")
        print("   → Improve cluster detection")
        print("   → Never path through enemy groups")

    if 'Hit by projectile' in [c for c, _ in cause_counts.most_common(2)]:
        print("⚠️  Not dodging projectiles well")
        print("   → Increase projectile avoidance priority")
        print("   → Predict projectile trajectories")

    if np.mean(levels) < 3:
        print("⚠️  Not clearing level 1 consistently")
        print("   → Focus on spawner elimination")
        print("   → Tune distance thresholds")
        print("   → Test more aggressive parameters")

    return all_diagnostics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='FSM Diagnostic Tool')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--single', action='store_true', help='Run single episode with detailed output')

    args = parser.parse_args()

    if args.single:
        # Single episode with detailed output
        diagnostic = run_diagnostic_episode(args.config)
        diagnostic.print_summary(1)

        # Save detailed frame history
        with open('fsm_diagnostic_frames.json', 'w') as f:
            json.dump([asdict(frame) for frame in diagnostic.frame_history], f, indent=2)
        print(f"\nSaved frame history to fsm_diagnostic_frames.json")
    else:
        # Batch run with aggregate stats
        diagnostics = run_diagnostic_batch(args.episodes, args.config)
