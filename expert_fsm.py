"""
Expert FSM Player for Robotron 2084

Based on human expert strategies:
1. **Edge Circling**: Stay near edges, avoid center (enemies converge in center)
2. **Family Priority**: Always collect family members when safe
3. **Spawner Priority**: Kill brains, sphereoids, quarks first (they create more enemies)
4. **Projectile Avoidance**: Dodge bullets/missiles as top priority
5. **Smart Positioning**: Maintain distance from enemies while shooting

Key improvements over basic FSM:
- Strategic positioning (edge bias)
- Better enemy prioritization
- Predictive movement (anticipate enemy positions)
- Aggressive family collection when clear
- Kiting behavior (maintain optimal shooting distance)
"""

import numpy as np
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Constants
BOARD_WIDTH = 665
BOARD_HEIGHT = 492

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

# Edge circling zones
EDGE_THRESHOLD = 100  # Stay within 100 pixels of edge
DANGER_ZONE_CENTER = 200  # Avoid 200x200 center area
CENTER_X = BOARD_WIDTH / 2
CENTER_Y = BOARD_HEIGHT / 2

@dataclass
class Sprite:
    x: float
    y: float
    type: str
    distance: float = 0.0
    angle: float = 0.0
    velocity_x: float = 0.0  # For prediction (if available)
    velocity_y: float = 0.0


class ExpertFSM:
    """
    Expert-level FSM player using human strategies.
    """

    # Enemy priorities (higher = more urgent to kill)
    PRIORITY = {
        # Spawners (highest priority - create more enemies!)
        'Brain': 10,           # Spawns cruise missiles
        'Sphereoid': 9,        # Spawns enforcers
        'Quark': 8,            # Spawns tanks

        # Dangerous enemies (shoot on sight)
        'Enforcer': 7,         # Shoots homing bullets
        'Tank': 6,             # Shoots shells
        'Prog': 5,             # Brainwashed family (chases)

        # Projectiles (dodge > shoot)
        'EnforcerBullet': 4,
        'TankShell': 4,
        'CruiseMissile': 4,

        # Regular enemies
        'Grunt': 2,            # Basic enemy

        # Obstacles/unkillable
        'Electrode': 1,        # Obstacle
        'Hulk': 0,             # Unkillable

        # Family (collect, don't shoot!)
        'Mommy': -1,
        'Daddy': -1,
        'Mikey': -1,
    }

    # Threat distances
    IMMEDIATE_DANGER = 60    # Run away!
    DANGER_ZONE = 120        # Keep shooting while retreating
    OPTIMAL_DISTANCE = 180   # Ideal kiting distance
    SAFE_DISTANCE = 250      # Can advance toward enemies

    def __init__(self):
        self.player_pos = None
        self.previous_player_pos = None
        self.enemies = []
        self.projectiles = []
        self.family = []
        self.obstacles = []

    def parse_sprites(self, sprite_data: List[Tuple]) -> None:
        """Parse sprite data into categorized lists."""
        self.enemies = []
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
                continue  # Player's own bullets

            # Calculate distance from player (will be updated after player found)
            s = Sprite(x, y, sprite_type)

            # Categorize
            if sprite_type in ['Mommy', 'Daddy', 'Mikey']:
                self.family.append(s)
            elif sprite_type in ['EnforcerBullet', 'TankShell', 'CruiseMissile']:
                self.projectiles.append(s)
            elif sprite_type in ['Electrode', 'Hulk']:
                self.obstacles.append(s)
            else:
                self.enemies.append(s)

        # Calculate distances and angles
        if self.player_pos:
            for sprites in [self.enemies, self.projectiles, self.family, self.obstacles]:
                for s in sprites:
                    dx = s.x - self.player_pos.x
                    dy = s.y - self.player_pos.y
                    s.distance = math.hypot(dx, dy)
                    s.angle = math.atan2(dy, dx)

            # Sort by distance
            self.enemies.sort(key=lambda s: s.distance)
            self.projectiles.sort(key=lambda s: s.distance)
            self.family.sort(key=lambda s: s.distance)
            self.obstacles.sort(key=lambda s: s.distance)

    def get_direction_to(self, target_x: float, target_y: float) -> int:
        """Get direction from player to target."""
        if not self.player_pos:
            return STAY

        dx = target_x - self.player_pos.x
        dy = target_y - self.player_pos.y

        if abs(dx) < 1 and abs(dy) < 1:
            return STAY

        angle = math.atan2(dy, dx)

        # Convert angle to 8-direction
        # 0 rad = RIGHT, π/2 = UP, π = LEFT, -π/2 = DOWN
        angle_deg = math.degrees(angle)
        if angle_deg < 0:
            angle_deg += 360

        if angle_deg < 22.5 or angle_deg >= 337.5:
            return RIGHT
        elif angle_deg < 67.5:
            return UP_RIGHT
        elif angle_deg < 112.5:
            return UP
        elif angle_deg < 157.5:
            return UP_LEFT
        elif angle_deg < 202.5:
            return LEFT
        elif angle_deg < 247.5:
            return DOWN_LEFT
        elif angle_deg < 292.5:
            return DOWN
        else:
            return DOWN_RIGHT

    def get_direction_away(self, threat_x: float, threat_y: float) -> int:
        """Get direction away from threat."""
        # Just reverse the direction
        toward = self.get_direction_to(threat_x, threat_y)
        away = (toward + 3) % 8 + 1  # Rotate 180 degrees
        return away

    def check_wall_collision(self, direction: int) -> int:
        """Adjust direction if heading into wall."""
        if not self.player_pos:
            return direction

        x, y = self.player_pos.x, self.player_pos.y
        margin = 30

        # Near left edge
        if x < margin:
            if direction in [LEFT, UP_LEFT, DOWN_LEFT]:
                if y < BOARD_HEIGHT / 2:
                    return UP_RIGHT if direction == UP_LEFT else RIGHT if direction == LEFT else DOWN_RIGHT
                else:
                    return DOWN_RIGHT if direction == DOWN_LEFT else RIGHT if direction == LEFT else UP_RIGHT

        # Near right edge
        if x > BOARD_WIDTH - margin:
            if direction in [RIGHT, UP_RIGHT, DOWN_RIGHT]:
                if y < BOARD_HEIGHT / 2:
                    return UP_LEFT if direction == UP_RIGHT else LEFT if direction == RIGHT else DOWN_LEFT
                else:
                    return DOWN_LEFT if direction == DOWN_RIGHT else LEFT if direction == RIGHT else UP_LEFT

        # Near top edge
        if y < margin:
            if direction in [UP, UP_LEFT, UP_RIGHT]:
                if x < BOARD_WIDTH / 2:
                    return RIGHT if direction == UP else DOWN_RIGHT if direction == UP_RIGHT else DOWN_LEFT
                else:
                    return LEFT if direction == UP else DOWN_LEFT if direction == UP_LEFT else DOWN_RIGHT

        # Near bottom edge
        if y > BOARD_HEIGHT - margin:
            if direction in [DOWN, DOWN_LEFT, DOWN_RIGHT]:
                if x < BOARD_WIDTH / 2:
                    return RIGHT if direction == DOWN else UP_RIGHT if direction == DOWN_RIGHT else UP_LEFT
                else:
                    return LEFT if direction == DOWN else UP_LEFT if direction == DOWN_LEFT else UP_RIGHT

        return direction

    def get_edge_direction(self) -> int:
        """Get direction toward nearest edge (strategic positioning)."""
        if not self.player_pos:
            return STAY

        x, y = self.player_pos.x, self.player_pos.y

        # Calculate distances to each edge
        to_left = x
        to_right = BOARD_WIDTH - x
        to_top = y
        to_bottom = BOARD_HEIGHT - y

        # Find nearest edge
        min_dist = min(to_left, to_right, to_top, to_bottom)

        # Bias toward corners (more defensive)
        if min_dist == to_left:
            return UP_LEFT if to_top < to_bottom else DOWN_LEFT
        elif min_dist == to_right:
            return UP_RIGHT if to_top < to_bottom else DOWN_RIGHT
        elif min_dist == to_top:
            return UP_LEFT if to_left < to_right else UP_RIGHT
        else:
            return DOWN_LEFT if to_left < to_right else DOWN_RIGHT

    def is_in_danger_zone(self) -> bool:
        """Check if player is in dangerous center area."""
        if not self.player_pos:
            return False

        x, y = self.player_pos.x, self.player_pos.y
        dx = abs(x - CENTER_X)
        dy = abs(y - CENTER_Y)

        return dx < DANGER_ZONE_CENTER and dy < DANGER_ZONE_CENTER

    def get_shooting_target(self) -> Optional[Sprite]:
        """Get best target to shoot at (prioritize spawners/dangerous enemies)."""
        # Priority order: projectiles in range > spawners > shooting enemies > regular enemies

        # Shoot nearby projectiles
        for proj in self.projectiles:
            if proj.distance < self.DANGER_ZONE:
                return proj

        # Shoot spawners (high priority)
        spawners = [e for e in self.enemies if e.type in ['Brain', 'Sphereoid', 'Quark']]
        if spawners:
            return spawners[0]

        # Shoot dangerous enemies
        dangerous = [e for e in self.enemies if e.type in ['Enforcer', 'Tank']]
        if dangerous and dangerous[0].distance < self.SAFE_DISTANCE:
            return dangerous[0]

        # Shoot closest enemy if in range
        if self.enemies and self.enemies[0].distance < self.OPTIMAL_DISTANCE:
            return self.enemies[0]

        return None

    def decide_action(self, sprite_data: List[Tuple]) -> Tuple[int, int]:
        """
        Main decision function.

        Returns:
            (move_action, fire_action) tuple
        """
        self.parse_sprites(sprite_data)

        if not self.player_pos:
            return (STAY, UP)

        move_action = STAY
        fire_action = STAY

        # ===== PRIORITY 1: IMMEDIATE DANGER - DODGE PROJECTILES =====
        if self.projectiles:
            closest_projectile = self.projectiles[0]
            if closest_projectile.distance < self.IMMEDIATE_DANGER:
                # Emergency dodge!
                move_action = self.get_direction_away(closest_projectile.x, closest_projectile.y)
                fire_action = self.get_direction_to(closest_projectile.x, closest_projectile.y)
                move_action = self.check_wall_collision(move_action)
                return (move_action, fire_action)

        # ===== PRIORITY 2: SURROUNDED - RETREAT TO EDGE =====
        close_enemies = [e for e in self.enemies if e.distance < self.DANGER_ZONE]
        if len(close_enemies) >= 5 or (self.is_in_danger_zone() and close_enemies):
            # Too many enemies close or in center - retreat to edge
            move_action = self.get_edge_direction()
            target = self.get_shooting_target()
            if target:
                fire_action = self.get_direction_to(target.x, target.y)
            move_action = self.check_wall_collision(move_action)
            return (move_action, fire_action)

        # ===== PRIORITY 3: FAMILY COLLECTION (when safe) =====
        if self.family:
            nearest_family = self.family[0]

            # Check if path to family is clear (no close enemies)
            enemies_between = [e for e in self.enemies if e.distance < nearest_family.distance and e.distance < self.DANGER_ZONE]

            if not enemies_between and nearest_family.distance < 200:
                # Safe to collect family
                move_action = self.get_direction_to(nearest_family.x, nearest_family.y)

                # Still shoot at threats while collecting
                target = self.get_shooting_target()
                if target:
                    fire_action = self.get_direction_to(target.x, target.y)

                move_action = self.check_wall_collision(move_action)
                return (move_action, fire_action)

        # ===== PRIORITY 4: KITING - MAINTAIN OPTIMAL DISTANCE =====
        if self.enemies:
            closest_enemy = self.enemies[0]

            target = self.get_shooting_target()
            if target:
                fire_action = self.get_direction_to(target.x, target.y)

            if closest_enemy.distance < self.DANGER_ZONE:
                # Too close - retreat while shooting
                move_action = self.get_direction_away(closest_enemy.x, closest_enemy.y)
            elif closest_enemy.distance > self.SAFE_DISTANCE:
                # Too far - advance toward enemies (but stay near edges)
                if self.is_in_danger_zone():
                    # Move to edge instead of advancing
                    move_action = self.get_edge_direction()
                else:
                    # Advance toward closest enemy
                    move_action = self.get_direction_to(closest_enemy.x, closest_enemy.y)
            else:
                # Optimal distance - circle around enemy
                # Move perpendicular to enemy (kiting)
                angle_to_enemy = math.atan2(closest_enemy.y - self.player_pos.y,
                                           closest_enemy.x - self.player_pos.x)
                # Add 90 degrees for perpendicular movement
                circle_angle = angle_to_enemy + math.pi / 2

                circle_x = self.player_pos.x + 100 * math.cos(circle_angle)
                circle_y = self.player_pos.y + 100 * math.sin(circle_angle)

                move_action = self.get_direction_to(circle_x, circle_y)

            move_action = self.check_wall_collision(move_action)
            return (move_action, fire_action)

        # ===== DEFAULT: PATROL EDGES =====
        move_action = self.get_edge_direction()
        fire_action = UP  # Default shooting direction

        move_action = self.check_wall_collision(move_action)
        return (move_action, fire_action)


# Global FSM instance
_fsm = ExpertFSM()


def expert_decide(sprite_data: List[Tuple]) -> Tuple[int, int]:
    """
    Wrapper function matching robotron_fsm.py interface.

    Args:
        sprite_data: List of (x, y, type) tuples

    Returns:
        (move_action, fire_action) tuple
    """
    return _fsm.decide_action(sprite_data)


if __name__ == "__main__":
    """Test the expert FSM."""
    import argparse
    from robotron import RobotronEnv

    parser = argparse.ArgumentParser(description='Expert FSM Player')
    parser.add_argument('--level', type=int, default=1, help='Start Level')
    parser.add_argument('--lives', type=int, default=5, help='Start Lives')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes')
    parser.add_argument('--fps', type=int, default=0, help='FPS (0=unlimited)')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--headless', action='store_true', help='Headless mode')

    args = parser.parse_args()

    print("=" * 80)
    print("EXPERT FSM PLAYER")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Episodes: {args.episodes}")
    print()

    episode_results = []

    for episode in range(args.episodes):
        env = RobotronEnv(
            level=args.level,
            lives=args.lives,
            fps=args.fps,
            config_path=args.config,
            always_move=False,
            headless=args.headless
        )

        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)

        total_reward = 0
        steps = 0
        max_level = args.level
        done = False

        while not done and steps < 10000:
            sprite_data = info['data']

            # Get FSM decision
            move_action, fire_action = expert_decide(sprite_data)

            # Encode action
            action = move_action * 9 + fire_action

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            steps += 1

            if 'level' in info:
                max_level = max(max_level, info['level'])

        score = info.get('score', 0)
        kills = score // 100

        episode_results.append({
            'score': score,
            'kills': kills,
            'level': max_level,
            'steps': steps
        })

        print(f"Episode {episode + 1}: Level {max_level}, Score {score}, Kills {kills}")

        env.close()

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    scores = [r['score'] for r in episode_results]
    kills = [r['kills'] for r in episode_results]
    levels = [r['level'] for r in episode_results]

    import numpy as np
    print(f"Average Score:  {np.mean(scores):.1f} ± {np.std(scores):.1f}")
    print(f"Average Kills:  {np.mean(kills):.1f} ± {np.std(kills):.1f}")
    print(f"Average Level:  {np.mean(levels):.1f} ± {np.std(levels):.1f}")
    print(f"Max Level:      {max(levels)}")
