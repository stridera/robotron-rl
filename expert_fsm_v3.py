"""
Expert FSM Player v3 for Robotron 2084

CRITICAL FIX: Shooting Alignment
- Robotron bullets only hit enemies on exact 8-way grid (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)
- v1/v2 would shoot at enemies without positioning to align first
- v3 positions the player to get enemies on the firing line before shooting

Based on human expert strategies:
1. **Shooting Alignment**: Position to get enemies on 8-way firing lines
2. **Edge Circling**: Stay near edges, avoid center (enemies converge in center)
3. **Family Priority**: Always collect family members when safe
4. **Spawner Priority**: Kill brains, sphereoids, quarks first (they create more enemies)
5. **Projectile Avoidance**: Dodge bullets/missiles as top priority
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
EDGE_THRESHOLD = 80
DANGER_ZONE_CENTER = 200
CENTER_X = BOARD_WIDTH / 2
CENTER_Y = BOARD_HEIGHT / 2

# Alignment threshold - how close to a firing line before we shoot
ALIGNMENT_THRESHOLD = 15  # pixels


@dataclass
class Sprite:
    x: float
    y: float
    type: str
    distance: float = 0.0
    angle: float = 0.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0


class ExpertFSMv3:
    """
    Expert-level FSM player v3 with shooting alignment.
    """

    # Enemy priorities (higher = more urgent to kill)
    PRIORITY = {
        # Spawners (highest priority - create more enemies!)
        'Brain': 10,
        'Sphereoid': 9,
        'Quark': 8,
        # Dangerous enemies
        'Enforcer': 7,
        'Tank': 6,
        'Prog': 5,
        # Projectiles
        'EnforcerBullet': 4,
        'TankShell': 4,
        'CruiseMissile': 4,
        # Regular enemies
        'Grunt': 2,
        # Obstacles
        'Electrode': 1,
        'Hulk': 0,
        # Family
        'Mommy': -1,
        'Daddy': -1,
        'Mikey': -1,
    }

    # Threat distances (v3: slightly more aggressive)
    IMMEDIATE_DANGER = 50
    DANGER_ZONE = 100
    OPTIMAL_DISTANCE = 150
    SAFE_DISTANCE = 200

    def __init__(self):
        self.player_pos = None
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
                continue

            s = Sprite(x, y, sprite_type)

            if sprite_type in ['Mommy', 'Daddy', 'Mikey']:
                self.family.append(s)
            elif sprite_type in ['EnforcerBullet', 'TankShell', 'CruiseMissile']:
                self.projectiles.append(s)
            elif sprite_type in ['Electrode', 'Hulk']:
                self.obstacles.append(s)
            else:
                self.enemies.append(s)

        if self.player_pos:
            for sprites in [self.enemies, self.projectiles, self.family, self.obstacles]:
                for s in sprites:
                    dx = s.x - self.player_pos.x
                    dy = s.y - self.player_pos.y
                    s.distance = math.hypot(dx, dy)
                    s.angle = math.atan2(dy, dx)

            self.enemies.sort(key=lambda s: s.distance)
            self.projectiles.sort(key=lambda s: s.distance)
            self.family.sort(key=lambda s: s.distance)
            self.obstacles.sort(key=lambda s: s.distance)

    def angle_to_direction(self, angle: float) -> int:
        """Convert angle (radians) to 8-way direction."""
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

    def get_direction_to(self, target_x: float, target_y: float) -> int:
        """Get direction from player to target."""
        if not self.player_pos:
            return STAY

        dx = target_x - self.player_pos.x
        dy = target_y - self.player_pos.y

        if abs(dx) < 1 and abs(dy) < 1:
            return STAY

        angle = math.atan2(dy, dx)
        return self.angle_to_direction(angle)

    def get_direction_away(self, threat_x: float, threat_y: float) -> int:
        """Get direction away from threat."""
        toward = self.get_direction_to(threat_x, threat_y)
        if toward == STAY:
            return STAY
        away = (toward + 3) % 8 + 1
        return away

    def check_wall_collision(self, direction: int) -> int:
        """Adjust direction if heading into wall."""
        if not self.player_pos:
            return direction

        x, y = self.player_pos.x, self.player_pos.y
        margin = 30

        # Near left edge
        if x < margin and direction in [LEFT, UP_LEFT, DOWN_LEFT]:
            return DOWN if y < BOARD_HEIGHT / 2 else UP

        # Near right edge
        if x > BOARD_WIDTH - margin and direction in [RIGHT, UP_RIGHT, DOWN_RIGHT]:
            return DOWN if y < BOARD_HEIGHT / 2 else UP

        # Near top edge
        if y < margin and direction in [UP, UP_LEFT, UP_RIGHT]:
            return RIGHT if x < BOARD_WIDTH / 2 else LEFT

        # Near bottom edge
        if y > BOARD_HEIGHT - margin and direction in [DOWN, DOWN_LEFT, DOWN_RIGHT]:
            return RIGHT if x < BOARD_WIDTH / 2 else LEFT

        return direction

    def is_aligned_with_target(self, target: Sprite) -> Tuple[bool, Optional[int]]:
        """
        Check if target is on one of the 8 firing lines.

        Returns:
            (is_aligned, fire_direction)
            is_aligned: True if we can hit the target by shooting
            fire_direction: Which direction to fire (if aligned)
        """
        if not self.player_pos:
            return (False, None)

        dx = target.x - self.player_pos.x
        dy = target.y - self.player_pos.y

        # Check alignment for each of the 8 directions
        # Alignment means the target is within ALIGNMENT_THRESHOLD pixels of the firing line

        # RIGHT (0°)
        if abs(dy) < ALIGNMENT_THRESHOLD and dx > 0:
            return (True, RIGHT)

        # UP_RIGHT (45°)
        if abs(dx - dy) < ALIGNMENT_THRESHOLD and dx > 0 and dy < 0:
            return (True, UP_RIGHT)

        # UP (90°)
        if abs(dx) < ALIGNMENT_THRESHOLD and dy < 0:
            return (True, UP)

        # UP_LEFT (135°)
        if abs(dx + dy) < ALIGNMENT_THRESHOLD and dx < 0 and dy < 0:
            return (True, UP_LEFT)

        # LEFT (180°)
        if abs(dy) < ALIGNMENT_THRESHOLD and dx < 0:
            return (True, LEFT)

        # DOWN_LEFT (225°)
        if abs(dx - dy) < ALIGNMENT_THRESHOLD and dx < 0 and dy > 0:
            return (True, DOWN_LEFT)

        # DOWN (270°)
        if abs(dx) < ALIGNMENT_THRESHOLD and dy > 0:
            return (True, DOWN)

        # DOWN_RIGHT (315°)
        if abs(dx + dy) < ALIGNMENT_THRESHOLD and dx > 0 and dy > 0:
            return (True, DOWN_RIGHT)

        return (False, None)

    def get_alignment_move(self, target: Sprite) -> int:
        """
        Calculate which direction to move to align with target.

        Strategy: Move perpendicular to the angle to get onto nearest firing line.
        """
        if not self.player_pos:
            return STAY

        dx = target.x - self.player_pos.x
        dy = target.y - self.player_pos.y

        # Calculate angle to target
        angle = math.atan2(dy, dx)
        angle_deg = math.degrees(angle)
        if angle_deg < 0:
            angle_deg += 360

        # Find which 45° sector the target is in
        sector = int((angle_deg + 22.5) / 45) % 8

        # For each sector, determine which direction aligns us better
        # We want to move perpendicular to get onto the nearest 45° line

        # Determine if we should move clockwise or counter-clockwise
        # to get onto the nearest firing line

        # Simple approach: move toward the nearest cardinal/diagonal axis
        remainder = (angle_deg + 22.5) % 45

        # If in first half of sector, move counter-clockwise (perpendicular)
        # If in second half, move clockwise (perpendicular)

        if remainder < 22.5:
            # Move perpendicular counter-clockwise
            perpendicular = angle + math.pi / 2
        else:
            # Move perpendicular clockwise
            perpendicular = angle - math.pi / 2

        # But we also need to consider distance - if target is far, approach it
        # If target is close, strafe to align

        if target.distance > self.OPTIMAL_DISTANCE:
            # Target far - move toward it while trying to align
            # Blend: 50% toward target, 50% alignment
            blend_angle = angle * 0.7 + perpendicular * 0.3
            return self.angle_to_direction(blend_angle)
        else:
            # Target close - prioritize alignment (strafing)
            return self.angle_to_direction(perpendicular)

    def get_shooting_target(self) -> Optional[Sprite]:
        """Get best target to shoot at."""
        # Priority: projectiles > spawners > dangerous enemies > closest enemy

        # Shoot nearby projectiles
        for proj in self.projectiles:
            if proj.distance < self.DANGER_ZONE:
                return proj

        # Shoot spawners
        spawners = [e for e in self.enemies if e.type in ['Brain', 'Sphereoid', 'Quark']]
        if spawners and spawners[0].distance < self.SAFE_DISTANCE:
            return spawners[0]

        # Shoot dangerous enemies
        dangerous = [e for e in self.enemies if e.type in ['Enforcer', 'Tank', 'Prog']]
        if dangerous and dangerous[0].distance < self.OPTIMAL_DISTANCE:
            return dangerous[0]

        # Shoot closest enemy if in range
        if self.enemies and self.enemies[0].distance < self.OPTIMAL_DISTANCE:
            return self.enemies[0]

        return None

    def is_in_danger_zone(self) -> bool:
        """Check if player is in dangerous center area."""
        if not self.player_pos:
            return False

        dx = abs(self.player_pos.x - CENTER_X)
        dy = abs(self.player_pos.y - CENTER_Y)

        return dx < DANGER_ZONE_CENTER / 2 and dy < DANGER_ZONE_CENTER / 2

    def get_edge_direction(self) -> int:
        """Get direction toward nearest edge."""
        if not self.player_pos:
            return STAY

        x, y = self.player_pos.x, self.player_pos.y

        to_left = x
        to_right = BOARD_WIDTH - x
        to_top = y
        to_bottom = BOARD_HEIGHT - y

        min_dist = min(to_left, to_right, to_top, to_bottom)

        if min_dist == to_left:
            return UP_LEFT if to_top < to_bottom else DOWN_LEFT
        elif min_dist == to_right:
            return UP_RIGHT if to_top < to_bottom else DOWN_RIGHT
        elif min_dist == to_top:
            return UP_LEFT if to_left < to_right else UP_RIGHT
        else:
            return DOWN_LEFT if to_left < to_right else DOWN_RIGHT

    def decide_action(self, sprite_data: List[Tuple]) -> Tuple[int, int]:
        """
        Main decision function with shooting alignment.

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
                # Emergency dodge - just run!
                move_action = self.get_direction_away(closest_projectile.x, closest_projectile.y)

                # Try to shoot while dodging if aligned
                is_aligned, fire_dir = self.is_aligned_with_target(closest_projectile)
                if is_aligned:
                    fire_action = fire_dir

                move_action = self.check_wall_collision(move_action)
                return (move_action, fire_action)

        # ===== PRIORITY 2: OVERWHELMED - RETREAT TO EDGE =====
        close_enemies = [e for e in self.enemies if e.distance < self.DANGER_ZONE]
        if len(close_enemies) >= 5 or (self.is_in_danger_zone() and len(close_enemies) >= 2):
            # Retreat to edge
            move_action = self.get_edge_direction()

            # Shoot at nearest if aligned
            target = self.get_shooting_target()
            if target:
                is_aligned, fire_dir = self.is_aligned_with_target(target)
                if is_aligned:
                    fire_action = fire_dir

            move_action = self.check_wall_collision(move_action)
            return (move_action, fire_action)

        # ===== PRIORITY 3: FAMILY COLLECTION (when safe) =====
        if self.family:
            nearest_family = self.family[0]
            enemies_between = [e for e in self.enemies
                             if e.distance < nearest_family.distance and e.distance < self.DANGER_ZONE]

            if not enemies_between and nearest_family.distance < 200:
                # Safe to collect
                move_action = self.get_direction_to(nearest_family.x, nearest_family.y)

                # Shoot threats if aligned
                target = self.get_shooting_target()
                if target:
                    is_aligned, fire_dir = self.is_aligned_with_target(target)
                    if is_aligned:
                        fire_action = fire_dir

                move_action = self.check_wall_collision(move_action)
                return (move_action, fire_action)

        # ===== PRIORITY 4: COMBAT - ALIGN AND SHOOT =====
        target = self.get_shooting_target()

        if target:
            # Check if we're aligned with the target
            is_aligned, fire_dir = self.is_aligned_with_target(target)

            if is_aligned:
                # We can hit it! Shoot!
                fire_action = fire_dir

                # Movement: maintain optimal distance
                if target.distance < self.DANGER_ZONE:
                    # Too close - back up while staying aligned
                    move_action = self.get_direction_away(target.x, target.y)
                elif target.distance > self.SAFE_DISTANCE:
                    # Too far - move closer
                    move_action = self.get_direction_to(target.x, target.y)
                else:
                    # Good distance - strafe to maintain alignment
                    # Move perpendicular to keep enemy on firing line
                    angle_to_target = math.atan2(target.y - self.player_pos.y,
                                                target.x - self.player_pos.x)
                    strafe_angle = angle_to_target + math.pi / 2
                    strafe_x = self.player_pos.x + 50 * math.cos(strafe_angle)
                    strafe_y = self.player_pos.y + 50 * math.sin(strafe_angle)
                    move_action = self.get_direction_to(strafe_x, strafe_y)

            else:
                # Not aligned - move to get target on a firing line
                move_action = self.get_alignment_move(target)

                # Don't shoot yet (will miss)
                fire_action = STAY

        else:
            # No targets - patrol edges
            move_action = self.get_edge_direction()
            fire_action = STAY

        move_action = self.check_wall_collision(move_action)
        return (move_action, fire_action)


# Global FSM instance
_fsm = ExpertFSMv3()


def expert_decide_v3(sprite_data: List[Tuple]) -> Tuple[int, int]:
    """
    Wrapper function for v3 FSM.

    Args:
        sprite_data: List of (x, y, type) tuples

    Returns:
        (move_action, fire_action) tuple
    """
    return _fsm.decide_action(sprite_data)


if __name__ == "__main__":
    """Test the expert FSM v3."""
    import argparse
    from robotron import RobotronEnv

    parser = argparse.ArgumentParser(description='Expert FSM v3 Player')
    parser.add_argument('--level', type=int, default=1, help='Start Level')
    parser.add_argument('--lives', type=int, default=5, help='Start Lives')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes')
    parser.add_argument('--fps', type=int, default=0, help='FPS (0=unlimited)')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--headless', action='store_true', help='Headless mode')

    args = parser.parse_args()

    print("=" * 80)
    print("EXPERT FSM v3 PLAYER (With Shooting Alignment)")
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

            # Get FSM v3 decision
            move_action, fire_action = expert_decide_v3(sprite_data)

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

        print(f"Episode {episode + 1}: Level {max_level}, Score {score}, Kills {kills}, Steps {steps}")

        env.close()

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    scores = [r['score'] for r in episode_results]
    kills = [r['kills'] for r in episode_results]
    levels = [r['level'] for r in episode_results]
    steps_list = [r['steps'] for r in episode_results]

    import numpy as np
    print(f"Average Score:  {np.mean(scores):.1f} ± {np.std(scores):.1f}")
    print(f"Average Kills:  {np.mean(kills):.1f} ± {np.std(kills):.1f}")
    print(f"Average Level:  {np.mean(levels):.1f} ± {np.std(levels):.1f}")
    print(f"Max Level:      {max(levels)}")
    print(f"Average Steps:  {np.mean(steps_list):.1f}")
