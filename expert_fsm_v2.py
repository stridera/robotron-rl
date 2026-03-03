"""
Expert FSM v2 - With Cluster Avoidance

Key improvements over v1:
1. **Cluster Detection**: Identifies when 3+ enemies group together
2. **Cluster Avoidance**: Never paths through enemy clusters
3. **Safe Zone Management**: Knows which corners are safe
4. **Spawner Hunting**: Aggressively eliminates spawners (Brain, Sphereoid, Quark)
5. **Predictive Positioning**: Anticipates where enemies will converge

Based on diagnostic findings:
- 80% deaths from "trapped in center"
- Need to avoid enemy clusters
- Must hunt spawners before they create swarms
"""

import numpy as np
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

# Constants
BOARD_WIDTH = 665
BOARD_HEIGHT = 492
CENTER_X = BOARD_WIDTH / 2
CENTER_Y = BOARD_HEIGHT / 2

# Actions
STAY = 0
UP = 1
UP_RIGHT = 2
RIGHT = 3
DOWN_RIGHT = 4
DOWN = 5
DOWN_LEFT = 6
LEFT = 7
UP_LEFT = 8

@dataclass
class Sprite:
    x: float
    y: float
    type: str
    distance: float = 0.0
    angle: float = 0.0

@dataclass
class Cluster:
    """A group of nearby enemies."""
    center_x: float
    center_y: float
    enemy_count: int
    danger_score: float
    enemies: List[Sprite]

# Safe corners (edges with escape routes)
SAFE_CORNERS = [
    (80, 80),         # Top-left
    (585, 80),        # Top-right
    (80, 412),        # Bottom-left
    (585, 412),       # Bottom-right
]


class ExpertFSMv2:
    """
    Improved FSM with cluster avoidance and spawner hunting.
    """

    # Parameters (from v4_aggressive - best performer)
    IMMEDIATE_DANGER = 50
    DANGER_ZONE = 100
    OPTIMAL_DISTANCE = 150
    SAFE_DISTANCE = 200

    # Cluster parameters
    CLUSTER_RADIUS = 80      # Enemies within 80px = cluster
    CLUSTER_MIN_SIZE = 3     # 3+ enemies = cluster
    CLUSTER_AVOID_DIST = 150 # Stay this far from clusters

    # Spawner hunting
    SPAWNER_PRIORITY_DIST = 300  # Hunt spawners within this distance

    def __init__(self):
        self.player_pos = None
        self.enemies = []
        self.spawners = []
        self.projectiles = []
        self.family = []
        self.obstacles = []
        self.clusters = []

    def parse_sprites(self, sprite_data: List[Tuple]) -> None:
        """Parse and categorize sprites."""
        self.enemies = []
        self.spawners = []
        self.projectiles = []
        self.family = []
        self.obstacles = []
        self.player_pos = None

        for sprite in sprite_data:
            x, y, sprite_type = sprite

            if sprite_type == 'Player':
                self.player_pos = Sprite(x, y, sprite_type)
                continue

            if sprite_type == 'Bullet':
                continue

            s = Sprite(x, y, sprite_type)

            # Categorize
            if sprite_type in ['Mommy', 'Daddy', 'Mikey']:
                self.family.append(s)
            elif sprite_type in ['EnforcerBullet', 'TankShell', 'CruiseMissile']:
                self.projectiles.append(s)
            elif sprite_type in ['Electrode', 'Hulk']:
                self.obstacles.append(s)
            elif sprite_type in ['Brain', 'Sphereoid', 'Quark']:
                self.spawners.append(s)
                self.enemies.append(s)  # Also count as enemies
            else:
                self.enemies.append(s)

        # Calculate distances
        if self.player_pos:
            for sprites in [self.enemies, self.spawners, self.projectiles, self.family, self.obstacles]:
                for s in sprites:
                    dx = s.x - self.player_pos.x
                    dy = s.y - self.player_pos.y
                    s.distance = math.hypot(dx, dy)
                    s.angle = math.atan2(dy, dx)

            # Sort by distance
            self.enemies.sort(key=lambda s: s.distance)
            self.spawners.sort(key=lambda s: s.distance)
            self.projectiles.sort(key=lambda s: s.distance)
            self.family.sort(key=lambda s: s.distance)

            # Detect clusters
            self.clusters = self.detect_clusters()

    def detect_clusters(self) -> List[Cluster]:
        """Detect enemy clusters (groups of 3+ enemies within CLUSTER_RADIUS)."""
        if len(self.enemies) < self.CLUSTER_MIN_SIZE:
            return []

        clusters = []
        processed = set()

        for i, enemy in enumerate(self.enemies):
            if i in processed:
                continue

            # Find all enemies near this one
            nearby = []
            for j, other in enumerate(self.enemies):
                if i == j or j in processed:
                    continue

                dist = math.hypot(enemy.x - other.x, enemy.y - other.y)
                if dist < self.CLUSTER_RADIUS:
                    nearby.append(j)

            # If 3+ enemies cluster, mark it
            if len(nearby) + 1 >= self.CLUSTER_MIN_SIZE:
                cluster_enemies = [enemy] + [self.enemies[j] for j in nearby]

                # Calculate cluster center
                center_x = np.mean([e.x for e in cluster_enemies])
                center_y = np.mean([e.y for e in cluster_enemies])

                # Danger score based on size and proximity
                dist_to_player = math.hypot(center_x - self.player_pos.x,
                                           center_y - self.player_pos.y)
                danger = len(cluster_enemies) * (1.0 / (dist_to_player + 1))

                cluster = Cluster(
                    center_x=center_x,
                    center_y=center_y,
                    enemy_count=len(cluster_enemies),
                    danger_score=danger,
                    enemies=cluster_enemies
                )
                clusters.append(cluster)

                # Mark as processed
                processed.add(i)
                for j in nearby:
                    processed.add(j)

        # Sort by danger score
        clusters.sort(key=lambda c: c.danger_score, reverse=True)
        return clusters

    def get_direction_to(self, target_x: float, target_y: float) -> int:
        """Get direction from player to target."""
        if not self.player_pos:
            return STAY

        dx = target_x - self.player_pos.x
        dy = target_y - self.player_pos.y

        if abs(dx) < 1 and abs(dy) < 1:
            return STAY

        angle = math.atan2(dy, dx)
        angle_deg = math.degrees(angle)
        if angle_deg < 0:
            angle_deg += 360

        # Convert to 8-direction
        directions = [
            (0, 45, RIGHT),
            (45, 90, UP_RIGHT),
            (90, 135, UP),
            (135, 180, UP_LEFT),
            (180, 225, LEFT),
            (225, 270, DOWN_LEFT),
            (270, 315, DOWN),
            (315, 360, DOWN_RIGHT),
        ]

        for low, high, direction in directions:
            if low <= angle_deg < high:
                return direction

        return RIGHT

    def get_direction_away(self, threat_x: float, threat_y: float) -> int:
        """Get direction away from threat."""
        toward = self.get_direction_to(threat_x, threat_y)
        # Rotate 180 degrees (4 directions in 8-direction system)
        away = ((toward - 1 + 4) % 8) + 1
        return away

    def check_wall_collision(self, direction: int) -> int:
        """Adjust direction if heading into wall."""
        if not self.player_pos:
            return direction

        x, y = self.player_pos.x, self.player_pos.y
        margin = 25

        # Near edges - adjust direction
        if x < margin and direction in [LEFT, UP_LEFT, DOWN_LEFT]:
            if y < BOARD_HEIGHT / 2:
                return UP if direction == UP_LEFT else RIGHT
            else:
                return DOWN if direction == DOWN_LEFT else RIGHT

        if x > BOARD_WIDTH - margin and direction in [RIGHT, UP_RIGHT, DOWN_RIGHT]:
            if y < BOARD_HEIGHT / 2:
                return UP if direction == UP_RIGHT else LEFT
            else:
                return DOWN if direction == DOWN_RIGHT else LEFT

        if y < margin and direction in [UP, UP_LEFT, UP_RIGHT]:
            if x < BOARD_WIDTH / 2:
                return RIGHT if direction == UP else DOWN_RIGHT
            else:
                return LEFT if direction == UP else DOWN_LEFT

        if y > BOARD_HEIGHT - margin and direction in [DOWN, DOWN_LEFT, DOWN_RIGHT]:
            if x < BOARD_WIDTH / 2:
                return RIGHT if direction == DOWN else UP_RIGHT
            else:
                return LEFT if direction == DOWN else UP_LEFT

        return direction

    def get_nearest_safe_corner(self) -> Tuple[float, float]:
        """Find safest corner (fewest enemies nearby)."""
        if not self.player_pos:
            return SAFE_CORNERS[0]

        corner_scores = []
        for corner_x, corner_y in SAFE_CORNERS:
            # Score = distance to player - (enemies nearby * 50)
            dist_to_player = math.hypot(corner_x - self.player_pos.x,
                                       corner_y - self.player_pos.y)

            enemies_nearby = sum(1 for e in self.enemies
                               if math.hypot(e.x - corner_x, e.y - corner_y) < 150)

            score = -dist_to_player - (enemies_nearby * 50)
            corner_scores.append((corner_x, corner_y, score))

        # Return corner with highest score
        return max(corner_scores, key=lambda c: c[2])[:2]

    def is_path_clear_to_target(self, target: Sprite, max_enemies_between: int = 2) -> bool:
        """Check if path to target is relatively clear."""
        if not self.player_pos:
            return False

        # Count enemies between player and target
        enemies_between = 0
        for enemy in self.enemies:
            if enemy.type == target.type:
                continue

            # Check if enemy is on line between player and target
            # Simple check: is enemy closer to target than player is?
            dist_enemy_to_target = math.hypot(enemy.x - target.x, enemy.y - target.y)
            if dist_enemy_to_target < target.distance * 0.7:  # In the way
                enemies_between += 1

        return enemies_between <= max_enemies_between

    def should_hunt_spawner(self) -> Optional[Sprite]:
        """Determine if we should hunt a spawner."""
        if not self.spawners:
            return None

        nearest_spawner = self.spawners[0]

        # Only hunt if close enough and path is clearish
        if nearest_spawner.distance < self.SPAWNER_PRIORITY_DIST:
            if self.is_path_clear_to_target(nearest_spawner, max_enemies_between=3):
                return nearest_spawner

        return None

    def avoid_clusters(self) -> Optional[int]:
        """
        Check if we should avoid clusters.
        Returns direction to move if avoidance needed, None otherwise.
        """
        if not self.clusters:
            return None

        most_dangerous = self.clusters[0]
        dist_to_cluster = math.hypot(most_dangerous.center_x - self.player_pos.x,
                                     most_dangerous.center_y - self.player_pos.y)

        # If approaching a cluster, move away
        if dist_to_cluster < self.CLUSTER_AVOID_DIST:
            # Move away from cluster center
            away_direction = self.get_direction_away(most_dangerous.center_x,
                                                     most_dangerous.center_y)
            return away_direction

        return None

    def decide_action(self, sprite_data: List[Tuple]) -> Tuple[int, int]:
        """
        Main decision function with improved tactics.
        """
        self.parse_sprites(sprite_data)

        if not self.player_pos:
            return (STAY, UP)

        move_action = STAY
        fire_action = STAY

        # ===== PRIORITY 1: IMMEDIATE DANGER - PROJECTILES =====
        if self.projectiles and self.projectiles[0].distance < self.IMMEDIATE_DANGER:
            proj = self.projectiles[0]
            move_action = self.get_direction_away(proj.x, proj.y)
            fire_action = self.get_direction_to(proj.x, proj.y)
            move_action = self.check_wall_collision(move_action)
            return (move_action, fire_action)

        # ===== PRIORITY 2: CLUSTER AVOIDANCE (NEW!) =====
        cluster_avoidance = self.avoid_clusters()
        if cluster_avoidance is not None:
            move_action = cluster_avoidance

            # Still shoot at nearest threat while retreating
            if self.enemies:
                fire_action = self.get_direction_to(self.enemies[0].x, self.enemies[0].y)

            move_action = self.check_wall_collision(move_action)
            return (move_action, fire_action)

        # ===== PRIORITY 3: SPAWNER HUNTING (NEW!) =====
        spawner_target = self.should_hunt_spawner()
        if spawner_target is not None:
            # Hunt the spawner aggressively
            move_action = self.get_direction_to(spawner_target.x, spawner_target.y)
            fire_action = self.get_direction_to(spawner_target.x, spawner_target.y)

            # But still dodge if enemies get too close
            if self.enemies and self.enemies[0].distance < self.DANGER_ZONE:
                move_action = self.get_direction_away(self.enemies[0].x, self.enemies[0].y)

            move_action = self.check_wall_collision(move_action)
            return (move_action, fire_action)

        # ===== PRIORITY 4: OVERWHELMED - RETREAT TO SAFE CORNER =====
        close_enemies = [e for e in self.enemies if e.distance < self.DANGER_ZONE]
        if len(close_enemies) >= 5:
            safe_x, safe_y = self.get_nearest_safe_corner()
            move_action = self.get_direction_to(safe_x, safe_y)

            if self.enemies:
                fire_action = self.get_direction_to(self.enemies[0].x, self.enemies[0].y)

            move_action = self.check_wall_collision(move_action)
            return (move_action, fire_action)

        # ===== PRIORITY 5: FAMILY COLLECTION (when safe) =====
        if self.family:
            nearest_family = self.family[0]

            # Check if path is clear and no nearby enemies
            enemies_close = len([e for e in self.enemies if e.distance < self.DANGER_ZONE])

            if enemies_close < 2 and nearest_family.distance < 150:
                move_action = self.get_direction_to(nearest_family.x, nearest_family.y)

                if self.enemies:
                    fire_action = self.get_direction_to(self.enemies[0].x, self.enemies[0].y)

                move_action = self.check_wall_collision(move_action)
                return (move_action, fire_action)

        # ===== PRIORITY 6: KITING - MAINTAIN OPTIMAL DISTANCE =====
        if self.enemies:
            closest = self.enemies[0]

            fire_action = self.get_direction_to(closest.x, closest.y)

            if closest.distance < self.DANGER_ZONE:
                # Too close - retreat
                move_action = self.get_direction_away(closest.x, closest.y)
            elif closest.distance > self.SAFE_DISTANCE:
                # Too far - advance (but stay near edges)
                player_dist_to_center = math.hypot(self.player_pos.x - CENTER_X,
                                                   self.player_pos.y - CENTER_Y)

                if player_dist_to_center < 150:  # In danger zone
                    # Move to nearest edge instead
                    if self.player_pos.x < CENTER_X:
                        move_action = LEFT if self.player_pos.x > 100 else UP if self.player_pos.y > CENTER_Y else DOWN
                    else:
                        move_action = RIGHT if self.player_pos.x < BOARD_WIDTH - 100 else UP if self.player_pos.y > CENTER_Y else DOWN
                else:
                    # Advance carefully
                    move_action = self.get_direction_to(closest.x, closest.y)
            else:
                # Optimal distance - circle/kite
                angle_to_enemy = closest.angle
                circle_angle = angle_to_enemy + math.pi / 2

                circle_x = self.player_pos.x + 80 * math.cos(circle_angle)
                circle_y = self.player_pos.y + 80 * math.sin(circle_angle)

                move_action = self.get_direction_to(circle_x, circle_y)

            move_action = self.check_wall_collision(move_action)
            return (move_action, fire_action)

        # ===== DEFAULT: PATROL EDGES =====
        # Move to nearest edge
        if self.player_pos.x < CENTER_X:
            move_action = LEFT if self.player_pos.x > 80 else (UP if self.player_pos.y > CENTER_Y else DOWN)
        else:
            move_action = RIGHT if self.player_pos.x < BOARD_WIDTH - 80 else (UP if self.player_pos.y > CENTER_Y else DOWN)

        fire_action = UP
        move_action = self.check_wall_collision(move_action)
        return (move_action, fire_action)


# Global FSM instance
_fsm_v2 = ExpertFSMv2()


def expert_decide_v2(sprite_data: List[Tuple]) -> Tuple[int, int]:
    """
    Wrapper function for FSMv2.
    """
    return _fsm_v2.decide_action(sprite_data)


if __name__ == "__main__":
    """Quick test of FSM v2."""
    import argparse
    from robotron import RobotronEnv

    parser = argparse.ArgumentParser(description='Expert FSM v2')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')

    args = parser.parse_args()

    print("="*80)
    print("EXPERT FSM v2 - WITH CLUSTER AVOIDANCE")
    print("="*80)
    print()

    episode_results = []

    for episode in range(args.episodes):
        env = RobotronEnv(
            level=1,
            lives=5,
            fps=0,
            config_path=args.config,
            always_move=False,
            headless=True
        )

        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)

        steps = 0
        max_level = 1
        done = False

        while not done and steps < 10000:
            move, fire = expert_decide_v2(info['data'])
            action = move * 9 + fire

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

            if 'level' in info:
                max_level = max(max_level, info['level'])

        score = info.get('score', 0)
        episode_results.append({'score': score, 'level': max_level, 'steps': steps})

        print(f"Episode {episode+1}: Level {max_level}, Score {score}, Steps {steps}")
        env.close()

    print()
    print("="*80)
    scores = [r['score'] for r in episode_results]
    levels = [r['level'] for r in episode_results]

    import numpy as np
    print(f"Average Score: {np.mean(scores):.1f} ± {np.std(scores):.1f}")
    print(f"Average Level: {np.mean(levels):.1f}")
    print(f"Max Level: {max(levels)}")
