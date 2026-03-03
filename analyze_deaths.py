"""
Death Analysis Tool for FSM v6

Records the last N frames before death and analyzes common failure patterns.
Helps identify specific situations that need improvement.
"""

import argparse
import json
import math
from collections import defaultdict
from typing import List, Tuple, Dict
from dataclasses import dataclass, asdict
from robotron import RobotronEnv
from expert_fsm_v6 import expert_decide_v6


@dataclass
class FrameData:
    """Data for a single frame before death."""
    frame_offset: int  # Frames before death (0 = death frame)
    player_x: float
    player_y: float
    move_action: int
    fire_action: int
    sprites: List[Tuple[float, float, str]]  # (x, y, type)
    enemies_nearby: List[Dict]  # Detailed enemy info
    safe_moves: List[int]


@dataclass
class DeathScenario:
    """Complete death scenario with last N frames."""
    episode: int
    level: int
    score: int
    steps: int
    frames: List[FrameData]
    cause: str  # What killed us
    closest_threats: List[Dict]  # What was near us when we died


class DeathRecorder:
    """Records frames before death for analysis."""

    def __init__(self, buffer_size: int = 15):
        self.buffer_size = buffer_size
        self.frame_buffer = []
        self.deaths = []

    def record_frame(self, sprite_data, move_action, fire_action, player_pos, safe_moves):
        """Add frame to circular buffer."""
        # Extract player position
        player_x, player_y = player_pos.x, player_pos.y

        # Find nearby enemies (within 200px)
        enemies_nearby = []
        for x, y, sprite_type in sprite_data:
            if sprite_type in ['Player', 'Bullet']:
                continue

            dist = math.hypot(x - player_x, y - player_y)
            if dist < 200:
                enemies_nearby.append({
                    'type': sprite_type,
                    'x': x,
                    'y': y,
                    'distance': dist
                })

        frame = FrameData(
            frame_offset=-1,  # Will be set when death occurs
            player_x=player_x,
            player_y=player_y,
            move_action=move_action,
            fire_action=fire_action,
            sprites=sprite_data,
            enemies_nearby=enemies_nearby,
            safe_moves=safe_moves
        )

        self.frame_buffer.append(frame)
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)

    def record_death(self, episode, level, score, steps):
        """Death occurred - save buffered frames."""
        if not self.frame_buffer:
            return

        # Set frame offsets (0 = death frame, negative = frames before)
        frames = []
        for i, frame in enumerate(self.frame_buffer):
            frame.frame_offset = i - len(self.frame_buffer) + 1
            frames.append(frame)

        # Analyze what killed us
        death_frame = frames[-1]
        cause, closest_threats = self.analyze_death(death_frame)

        scenario = DeathScenario(
            episode=episode,
            level=level,
            score=score,
            steps=steps,
            frames=frames,
            cause=cause,
            closest_threats=closest_threats
        )

        self.deaths.append(scenario)
        self.frame_buffer = []  # Clear buffer for next life

    def analyze_death(self, death_frame: FrameData) -> Tuple[str, List[Dict]]:
        """Determine what killed us."""
        # Sort enemies by distance
        nearby = sorted(death_frame.enemies_nearby, key=lambda e: e['distance'])

        if not nearby:
            return "UNKNOWN", []

        closest = nearby[0]

        # Categorize death cause
        if closest['type'] in ['EnforcerBullet', 'TankShell', 'CruiseMissile']:
            cause = f"HIT_BY_BULLET ({closest['type']})"
        elif closest['type'] == 'Hulk':
            cause = "HIT_BY_HULK"
        elif closest['type'] == 'Electrode':
            cause = "HIT_ELECTRODE"
        elif closest['type'] in ['Grunt', 'Prog']:
            cause = f"HIT_BY_ENEMY ({closest['type']})"
        elif closest['type'] in ['Enforcer', 'Tank']:
            cause = f"HIT_BY_SHOOTER ({closest['type']})"
        else:
            cause = f"HIT_BY_OTHER ({closest['type']})"

        # Add distance and context
        cause += f" dist={closest['distance']:.1f}px"

        # Check if we were trapped (no safe moves)
        if len(death_frame.safe_moves) == 0:
            cause += " [TRAPPED - no safe moves]"
        elif len(death_frame.safe_moves) <= 2:
            cause += f" [LIMITED - only {len(death_frame.safe_moves)} safe moves]"

        return cause, nearby[:5]  # Return top 5 threats


def run_death_analysis(episodes: int, config: str, level: int = 1, lives: int = 5):
    """Run episodes and collect death data."""
    print("=" * 80)
    print("DEATH ANALYSIS - FSM v6")
    print("=" * 80)
    print(f"Running {episodes} episodes to collect death data...")
    print()

    recorder = DeathRecorder(buffer_size=15)
    episode_count = 0
    death_count = 0

    for episode in range(episodes):
        env = RobotronEnv(
            level=level,
            lives=lives,
            fps=0,
            config_path=config,
            always_move=False,
            headless=True
        )

        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)

        steps = 0
        max_level = level
        current_lives = lives

        while not (terminated or truncated) and steps < 10000:
            sprite_data = info['data']

            # FSM decision
            move_action, fire_action = expert_decide_v6(sprite_data)

            # Record frame BEFORE executing
            from expert_fsm_v6 import _fsm
            recorder.record_frame(
                sprite_data,
                move_action,
                fire_action,
                _fsm.player_pos,
                _fsm.find_safe_moves() if _fsm.player_pos else []
            )

            # Execute
            action = move_action * 9 + fire_action
            obs, reward, terminated, truncated, info = env.step(action)

            steps += 1

            if 'level' in info:
                max_level = max(max_level, info['level'])

            # Check if we lost a life
            new_lives = info.get('lives', 0)
            if new_lives < current_lives:
                # Death occurred!
                death_count += 1
                recorder.record_death(episode + 1, max_level, info.get('score', 0), steps)
                print(f"  Death #{death_count}: Episode {episode + 1}, Level {max_level}, Score {info.get('score', 0)}, Step {steps}")
                current_lives = new_lives

        episode_count += 1
        if episode_count % 10 == 0:
            print(f"  Completed {episode_count}/{episodes} episodes, {death_count} deaths recorded")

        env.close()

    print()
    print(f"Analysis complete: {death_count} deaths recorded from {episodes} episodes")
    print()

    return recorder.deaths


def analyze_death_patterns(deaths: List[DeathScenario]):
    """Analyze common patterns in deaths."""
    print("=" * 80)
    print("DEATH PATTERN ANALYSIS")
    print("=" * 80)
    print()

    # Categorize deaths
    death_causes = defaultdict(int)
    bullet_deaths = []
    hulk_deaths = []
    trapped_deaths = []
    enemy_deaths = []

    for death in deaths:
        death_causes[death.cause.split(" dist=")[0]] += 1

        if "BULLET" in death.cause:
            bullet_deaths.append(death)
        elif "HULK" in death.cause:
            hulk_deaths.append(death)
        elif "TRAPPED" in death.cause or "LIMITED" in death.cause:
            trapped_deaths.append(death)
        elif "ENEMY" in death.cause or "SHOOTER" in death.cause:
            enemy_deaths.append(death)

    # Print summary
    print(f"Total Deaths: {len(deaths)}")
    print()
    print("Death Causes:")
    for cause, count in sorted(death_causes.items(), key=lambda x: -x[1]):
        percentage = (count / len(deaths)) * 100
        print(f"  {cause:40s} {count:3d} ({percentage:5.1f}%)")

    print()
    print("=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)

    # Analyze bullet deaths
    if bullet_deaths:
        print()
        print(f"BULLET DEATHS ({len(bullet_deaths)} total)")
        print("-" * 80)

        # Sample a few for detailed analysis
        for i, death in enumerate(bullet_deaths[:3]):
            print(f"\nExample {i+1}: Episode {death.episode}, Level {death.level}")
            print(f"Cause: {death.cause}")

            # Last 5 frames
            for frame in death.frames[-5:]:
                print(f"  Frame {frame.frame_offset:+3d}: Player at ({frame.player_x:.0f}, {frame.player_y:.0f})")
                print(f"              Safe moves: {len(frame.safe_moves)}, Nearby threats: {len(frame.enemies_nearby)}")
                if frame.enemies_nearby:
                    closest = frame.enemies_nearby[0]
                    print(f"              Closest: {closest['type']} at {closest['distance']:.0f}px")

    # Analyze Hulk deaths
    if hulk_deaths:
        print()
        print(f"HULK DEATHS ({len(hulk_deaths)} total)")
        print("-" * 80)
        print("Hulks are immortal obstacles - we should NEVER hit them!")

        for i, death in enumerate(hulk_deaths[:3]):
            print(f"\nExample {i+1}: Episode {death.episode}, Level {death.level}")
            print(f"Cause: {death.cause}")

            death_frame = death.frames[-1]
            print(f"  Death frame: Player at ({death_frame.player_x:.0f}, {death_frame.player_y:.0f})")
            print(f"  Safe moves available: {len(death_frame.safe_moves)}")

            # Find the Hulk
            for threat in death.closest_threats:
                if threat['type'] == 'Hulk':
                    print(f"  Hulk at ({threat['x']:.0f}, {threat['y']:.0f}), dist={threat['distance']:.0f}px")

    # Analyze trapped deaths
    if trapped_deaths:
        print()
        print(f"TRAPPED DEATHS ({len(trapped_deaths)} total)")
        print("-" * 80)
        print("These deaths occurred when we had no (or very few) safe moves.")

        for i, death in enumerate(trapped_deaths[:3]):
            print(f"\nExample {i+1}: Episode {death.episode}, Level {death.level}")
            print(f"Cause: {death.cause}")

            # Show last 3 frames
            for frame in death.frames[-3:]:
                print(f"  Frame {frame.frame_offset:+3d}: Safe moves: {len(frame.safe_moves)}, Threats: {len(frame.enemies_nearby)}")
                if frame.enemies_nearby:
                    print(f"              Threats within 200px: {[t['type'] for t in frame.enemies_nearby[:5]]}")

    # Analyze regular enemy deaths
    if enemy_deaths:
        print()
        print(f"ENEMY DEATHS ({len(enemy_deaths)} total)")
        print("-" * 80)

        for i, death in enumerate(enemy_deaths[:3]):
            print(f"\nExample {i+1}: Episode {death.episode}, Level {death.level}")
            print(f"Cause: {death.cause}")

            death_frame = death.frames[-1]
            print(f"  Death frame: Player at ({death_frame.player_x:.0f}, {death_frame.player_y:.0f})")
            print(f"  Safe moves available: {len(death_frame.safe_moves)}")
            threat_list = [(t['type'], f"{t['distance']:.0f}px") for t in death.closest_threats[:3]]
            print(f"  Nearby threats: {threat_list}")

    print()
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()

    # Generate recommendations based on patterns
    recommendations = []

    if bullet_deaths:
        pct = (len(bullet_deaths) / len(deaths)) * 100
        recommendations.append(
            f"1. BULLET AVOIDANCE ({pct:.1f}% of deaths)\n"
            f"   - Increase bullet dodge priority (currently triggers at <100px)\n"
            f"   - Improve bullet trajectory prediction\n"
            f"   - Add 'bullet danger zones' to avoid"
        )

    if hulk_deaths:
        pct = (len(hulk_deaths) / len(deaths)) * 100
        recommendations.append(
            f"2. HULK COLLISION ({pct:.1f}% of deaths)\n"
            f"   - CRITICAL: Collision prediction should prevent this!\n"
            f"   - Check is_move_safe() for Hulk detection bugs\n"
            f"   - Increase Hulk collision radius (currently {20}px)"
        )

    if trapped_deaths:
        pct = (len(trapped_deaths) / len(deaths)) * 100
        recommendations.append(
            f"3. GETTING TRAPPED ({pct:.1f}% of deaths)\n"
            f"   - Add 'escape route' planning\n"
            f"   - Avoid situations with <3 safe moves\n"
            f"   - Maintain distance from walls when enemies nearby"
        )

    if enemy_deaths:
        pct = (len(enemy_deaths) / len(deaths)) * 100
        recommendations.append(
            f"4. ENEMY COLLISIONS ({pct:.1f}% of deaths)\n"
            f"   - Check collision radius (currently {20}px)\n"
            f"   - Improve distance maintenance (currently 100-200px)\n"
            f"   - Add 'personal space' buffer"
        )

    for rec in recommendations:
        print(rec)
        print()


def save_death_data(deaths: List[DeathScenario], filename: str = "death_analysis.json"):
    """Save death data to JSON file."""
    # Convert to serializable format
    data = []
    for death in deaths:
        death_dict = {
            'episode': death.episode,
            'level': death.level,
            'score': death.score,
            'steps': death.steps,
            'cause': death.cause,
            'closest_threats': death.closest_threats,
            'frames': []
        }

        for frame in death.frames:
            frame_dict = {
                'frame_offset': frame.frame_offset,
                'player_x': frame.player_x,
                'player_y': frame.player_y,
                'move_action': frame.move_action,
                'fire_action': frame.fire_action,
                'safe_moves': frame.safe_moves,
                'enemies_nearby': frame.enemies_nearby
            }
            death_dict['frames'].append(frame_dict)

        data.append(death_dict)

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Death data saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze FSM v6 Deaths')
    parser.add_argument('--episodes', type=int, default=50, help='Number of episodes to run')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--level', type=int, default=1, help='Starting level')
    parser.add_argument('--lives', type=int, default=5, help='Starting lives')
    parser.add_argument('--output', type=str, default='death_analysis.json', help='Output JSON file')

    args = parser.parse_args()

    # Run episodes and collect deaths
    deaths = run_death_analysis(args.episodes, args.config, args.level, args.lives)

    if deaths:
        # Analyze patterns
        analyze_death_patterns(deaths)

        # Save data
        save_death_data(deaths, args.output)
    else:
        print("No deaths recorded! FSM is invincible! 🎉")
