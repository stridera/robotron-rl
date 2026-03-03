"""
Expert FSM Player v4 for Robotron 2084

NEW in v4:
1. **Family Collection Priority**: Aggressively collect family early in wave
2. **Level Completion Mode**: Hunt down remaining enemies to clear levels
3. **Safe Alignment**: Only align if movement is safe, otherwise retreat
4. **Spawner Hunting**: Carefully hunt spawners when path is clear

v3 achievements: +54% kills, first time reaching level 2
v4 goal: Consistently clear level 1, reach level 3-5

Based on human expert strategies:
1. **Shooting Alignment**: Position to get enemies on 8-way firing lines
2. **Smart Family Collection**: Collect early when safe
3. **Level Completion**: Actively hunt remaining enemies
4. **Edge Circling**: Stay near edges, avoid center
5. **Spawner Priority**: Kill brains, sphereoids, quarks ASAP
6. **Projectile Avoidance**: Dodge bullets/missiles as top priority
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

# Alignment threshold (increased to reduce jittering - easier to get aligned)
ALIGNMENT_THRESHOLD = 25  # pixels


@dataclass
class Sprite:
    x: float
    y: float
    type: str
    distance: float = 0.0
    angle: float = 0.0


class ExpertFSMv4:
    """
    Expert-level FSM player v4 with family priority and level completion.
    """

    # Enemy priorities
    PRIORITY = {
        'Brain': 10,
        'Sphereoid': 9,
        'Quark': 8,
        'Enforcer': 7,
        'Tank': 6,
        'Prog': 5,
        'EnforcerBullet': 4,
        'TankShell': 4,
        'CruiseMissile': 4,
        'Grunt': 2,
        'Electrode': 1,
        'Hulk': 0,
        'Mommy': -1,
        'Daddy': -1,
        'Mikey': -1,
    }

    # Threat distances (increased ranges to shoot earlier)
    IMMEDIATE_DANGER = 50
    DANGER_ZONE = 100
    OPTIMAL_DISTANCE = 200  # Increased from 150 - shoot from further away
    SAFE_DISTANCE = 250  # Increased from 200

    def __init__(self):
        self.player_pos = None
        self.enemies = []
        self.projectiles = []
        self.family = []
        self.obstacles = []
        self.total_enemies_seen = 0  # Track total enemies for level completion

        # Decision persistence to avoid jittering
        self.current_target = None
        self.target_lock_frames = 0
        self.target_lock_duration = 15  # Stick with target for at least 15 frames

        self.previous_move = STAY
        self.move_persistence_frames = 0
        self.move_persistence_duration = 10  # Commit to movement for 10 frames (increased from 5)

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

        # Track max enemies
        total = len(self.enemies) + len(self.obstacles)
        if total > self.total_enemies_seen:
            self.total_enemies_seen = total

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

        if x < margin and direction in [LEFT, UP_LEFT, DOWN_LEFT]:
            return DOWN if y < BOARD_HEIGHT / 2 else UP

        if x > BOARD_WIDTH - margin and direction in [RIGHT, UP_RIGHT, DOWN_RIGHT]:
            return DOWN if y < BOARD_HEIGHT / 2 else UP

        if y < margin and direction in [UP, UP_LEFT, UP_RIGHT]:
            return RIGHT if x < BOARD_WIDTH / 2 else LEFT

        if y > BOARD_HEIGHT - margin and direction in [DOWN, DOWN_LEFT, DOWN_RIGHT]:
            return RIGHT if x < BOARD_WIDTH / 2 else LEFT

        return direction

    def simulate_move(self, direction: int, distance: float = 20) -> Tuple[float, float]:
        """Simulate where player would be after moving in direction."""
        if not self.player_pos:
            return (0, 0)

        x, y = self.player_pos.x, self.player_pos.y

        # Direction to angle
        dir_angles = {
            STAY: 0,
            RIGHT: 0,
            UP_RIGHT: -math.pi/4,
            UP: -math.pi/2,
            UP_LEFT: -3*math.pi/4,
            LEFT: math.pi,
            DOWN_LEFT: 3*math.pi/4,
            DOWN: math.pi/2,
            DOWN_RIGHT: math.pi/4,
        }

        angle = dir_angles.get(direction, 0)
        new_x = x + distance * math.cos(angle)
        new_y = y + distance * math.sin(angle)

        return (new_x, new_y)

    def is_move_safe(self, direction: int) -> bool:
        """Check if moving in this direction is safe (won't walk into enemies)."""
        new_x, new_y = self.simulate_move(direction, distance=30)

        # Count enemies near future position
        danger_count = 0
        for enemy in self.enemies:
            dist = math.hypot(enemy.x - new_x, enemy.y - new_y)
            if dist < 60:  # Would be very close
                danger_count += 1

        # Also check projectiles
        for proj in self.projectiles:
            dist = math.hypot(proj.x - new_x, proj.y - new_y)
            if dist < 40:
                danger_count += 2  # Projectiles are more dangerous

        return danger_count < 2

    def is_aligned_with_target(self, target: Sprite) -> Tuple[bool, Optional[int]]:
        """Check if target is on one of the 8 firing lines."""
        if not self.player_pos:
            return (False, None)

        dx = target.x - self.player_pos.x
        dy = target.y - self.player_pos.y

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
        """Calculate which direction to move to align with target."""
        if not self.player_pos:
            return STAY

        dx = target.x - self.player_pos.x
        dy = target.y - self.player_pos.y

        angle = math.atan2(dy, dx)
        angle_deg = math.degrees(angle)
        if angle_deg < 0:
            angle_deg += 360

        remainder = (angle_deg + 22.5) % 45

        if remainder < 22.5:
            perpendicular = angle + math.pi / 2
        else:
            perpendicular = angle - math.pi / 2

        if target.distance > self.OPTIMAL_DISTANCE:
            blend_angle = angle * 0.7 + perpendicular * 0.3
            return self.angle_to_direction(blend_angle)
        else:
            return self.angle_to_direction(perpendicular)

    def get_shooting_target(self) -> Optional[Sprite]:
        """Get best target to shoot at with target locking to avoid jittering."""

        # Check if we should keep current target (target locking)
        if self.current_target and self.target_lock_frames < self.target_lock_duration:
            # Find current target in sprite lists
            for proj in self.projectiles:
                if proj.type == self.current_target.type and proj.distance < self.SAFE_DISTANCE:
                    self.target_lock_frames += 1
                    return proj

            for enemy in self.enemies:
                if enemy.type == self.current_target.type and enemy.distance < self.SAFE_DISTANCE:
                    self.target_lock_frames += 1
                    return enemy

        # Target lock expired or no current target - select new target
        self.target_lock_frames = 0

        # Shoot nearby projectiles
        for proj in self.projectiles:
            if proj.distance < self.DANGER_ZONE:
                self.current_target = proj
                return proj

        # Shoot spawners
        spawners = [e for e in self.enemies if e.type in ['Brain', 'Sphereoid', 'Quark']]
        if spawners and spawners[0].distance < self.SAFE_DISTANCE:
            self.current_target = spawners[0]
            return spawners[0]

        # Shoot dangerous enemies
        dangerous = [e for e in self.enemies if e.type in ['Enforcer', 'Tank', 'Prog']]
        if dangerous and dangerous[0].distance < self.OPTIMAL_DISTANCE:
            self.current_target = dangerous[0]
            return dangerous[0]

        # Shoot closest enemy if in range
        if self.enemies and self.enemies[0].distance < self.OPTIMAL_DISTANCE:
            self.current_target = self.enemies[0]
            return self.enemies[0]

        self.current_target = None
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

    def should_enter_completion_mode(self) -> bool:
        """Determine if we should aggressively hunt remaining enemies to complete level."""
        # Only a few enemies left
        total_remaining = len(self.enemies) + len(self.obstacles)

        if total_remaining <= 3:
            return True

        # Killed most enemies (assuming ~20 total on level 1)
        if self.total_enemies_seen > 0:
            percent_remaining = total_remaining / self.total_enemies_seen
            if percent_remaining < 0.25:  # Less than 25% left
                return True

        return False

    def is_safe_window_for_family(self) -> bool:
        """Check if it's a safe window to collect family."""
        # Early in wave (few enemies)
        if len(self.enemies) < 5:
            return True

        # Not many close enemies
        close_enemies = [e for e in self.enemies if e.distance < self.DANGER_ZONE]
        if len(close_enemies) < 2:
            return True

        return False

    def should_hunt_spawner(self) -> Optional[Sprite]:
        """Determine if we should hunt a spawner."""
        spawners = [e for e in self.enemies if e.type in ['Brain', 'Sphereoid', 'Quark']]

        if not spawners:
            return None

        nearest = spawners[0]

        # Don't hunt if too far
        if nearest.distance > 300:
            return None

        # Don't hunt if overwhelmed
        close_enemies = [e for e in self.enemies if e.distance < self.DANGER_ZONE]
        if len(close_enemies) > 5:
            return None

        # Check path
        enemies_between = sum(1 for e in self.enemies
                            if e.distance < nearest.distance and e.distance < 150)

        if enemies_between > 3:
            return None  # Path blocked

        return nearest

    def apply_movement_persistence(self, new_move: int) -> int:
        """Apply movement persistence to avoid jittering back and forth."""
        # If we're persisting a movement, keep it unless there's a big change needed
        if self.move_persistence_frames > 0:
            # Check if new move is very different (>90 degrees different)
            if new_move != STAY and self.previous_move != STAY:
                # Calculate angle difference
                diff = abs(new_move - self.previous_move)
                if diff <= 2 or diff >= 6:  # Adjacent or opposite directions
                    # Keep previous move
                    self.move_persistence_frames -= 1
                    return self.previous_move

        # New movement direction - start persistence
        self.previous_move = new_move
        self.move_persistence_frames = self.move_persistence_duration
        return new_move

    def decide_action(self, sprite_data: List[Tuple]) -> Tuple[int, int]:
        """
        Main decision function with v4 improvements.

        Priority order:
        1. Dodge projectiles (immediate danger)
        2. Retreat if overwhelmed
        3. FAMILY COLLECTION (improved priority)
        4. Level completion mode (hunt remaining enemies)
        5. Spawner hunting
        6. Combat (align and shoot)
        7. Default patrol

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
                move_action = self.get_direction_away(closest_projectile.x, closest_projectile.y)

                is_aligned, fire_dir = self.is_aligned_with_target(closest_projectile)
                if is_aligned:
                    fire_action = fire_dir

                move_action = self.apply_movement_persistence(move_action)
                move_action = self.check_wall_collision(move_action)
                return (move_action, fire_action)

        # ===== PRIORITY 2: OVERWHELMED - RETREAT TO EDGE =====
        close_enemies = [e for e in self.enemies if e.distance < self.DANGER_ZONE]
        if len(close_enemies) >= 6 or (self.is_in_danger_zone() and len(close_enemies) >= 3):
            move_action = self.get_edge_direction()

            target = self.get_shooting_target()
            if target:
                is_aligned, fire_dir = self.is_aligned_with_target(target)
                if is_aligned:
                    fire_action = fire_dir

            move_action = self.apply_movement_persistence(move_action)
            move_action = self.check_wall_collision(move_action)
            return (move_action, fire_action)

        # ===== PRIORITY 3: FAMILY COLLECTION (IMPROVED) =====
        if self.family and self.is_safe_window_for_family():
            nearest_family = self.family[0]

            # More aggressive family collection
            if nearest_family.distance < 250:
                # Check enemies in path
                enemies_in_path = [e for e in self.enemies
                                 if e.distance < nearest_family.distance and e.distance < 100]

                # Collect if path is reasonably clear OR family is very close
                if len(enemies_in_path) < 2 or nearest_family.distance < 80:
                    move_action = self.get_direction_to(nearest_family.x, nearest_family.y)

                    # Shoot threats if aligned
                    target = self.get_shooting_target()
                    if target:
                        is_aligned, fire_dir = self.is_aligned_with_target(target)
                        if is_aligned:
                            fire_action = fire_dir

                    move_action = self.apply_movement_persistence(move_action)
                    move_action = self.check_wall_collision(move_action)
                    return (move_action, fire_action)

        # ===== PRIORITY 4: LEVEL COMPLETION MODE =====
        if self.should_enter_completion_mode():
            # Hunt down remaining enemies aggressively
            if self.enemies:
                # Target furthest enemy (often stragglers)
                target = max(self.enemies, key=lambda e: e.distance)

                is_aligned, fire_dir = self.is_aligned_with_target(target)

                if is_aligned:
                    fire_action = fire_dir
                    # Move toward target
                    move_action = self.get_direction_to(target.x, target.y)
                else:
                    # Try to align
                    alignment_move = self.get_alignment_move(target)
                    if self.is_move_safe(alignment_move):
                        move_action = alignment_move
                    else:
                        # Not safe - approach target cautiously
                        move_action = self.get_direction_to(target.x, target.y)

                move_action = self.apply_movement_persistence(move_action)
                move_action = self.check_wall_collision(move_action)
                return (move_action, fire_action)

        # ===== PRIORITY 5: SPAWNER HUNTING =====
        spawner = self.should_hunt_spawner()
        if spawner:
            is_aligned, fire_dir = self.is_aligned_with_target(spawner)

            if is_aligned:
                fire_action = fire_dir
                # Maintain good distance while shooting
                if spawner.distance < self.DANGER_ZONE:
                    move_action = self.get_direction_away(spawner.x, spawner.y)
                else:
                    move_action = self.get_direction_to(spawner.x, spawner.y)
            else:
                # Move to align
                alignment_move = self.get_alignment_move(spawner)
                if self.is_move_safe(alignment_move):
                    move_action = alignment_move
                else:
                    # Approach cautiously
                    move_action = self.get_direction_to(spawner.x, spawner.y)

            move_action = self.apply_movement_persistence(move_action)
            move_action = self.check_wall_collision(move_action)
            return (move_action, fire_action)

        # ===== PRIORITY 6: COMBAT - SAFE ALIGNMENT AND SHOOT =====
        target = self.get_shooting_target()

        if target:
            is_aligned, fire_dir = self.is_aligned_with_target(target)

            if is_aligned:
                fire_action = fire_dir

                # Movement: maintain optimal distance
                if target.distance < self.DANGER_ZONE:
                    move_action = self.get_direction_away(target.x, target.y)
                elif target.distance > self.SAFE_DISTANCE:
                    move_action = self.get_direction_to(target.x, target.y)
                else:
                    # Strafe
                    angle_to_target = math.atan2(target.y - self.player_pos.y,
                                                target.x - self.player_pos.x)
                    strafe_angle = angle_to_target + math.pi / 2
                    strafe_x = self.player_pos.x + 50 * math.cos(strafe_angle)
                    strafe_y = self.player_pos.y + 50 * math.sin(strafe_angle)
                    move_action = self.get_direction_to(strafe_x, strafe_y)

            else:
                # Try to align - but only if safe!
                alignment_move = self.get_alignment_move(target)

                if self.is_move_safe(alignment_move):
                    move_action = alignment_move
                else:
                    # Not safe - retreat to edge instead
                    move_action = self.get_edge_direction()

                fire_action = STAY

        else:
            # No targets - patrol edges
            move_action = self.get_edge_direction()
            fire_action = STAY

        # Apply movement persistence to reduce jittering
        move_action = self.apply_movement_persistence(move_action)

        move_action = self.check_wall_collision(move_action)
        return (move_action, fire_action)


# Global FSM instance
_fsm = ExpertFSMv4()


def expert_decide_v4(sprite_data: List[Tuple]) -> Tuple[int, int]:
    """
    Wrapper function for v4 FSM.

    Args:
        sprite_data: List of (x, y, type) tuples

    Returns:
        (move_action, fire_action) tuple
    """
    return _fsm.decide_action(sprite_data)


if __name__ == "__main__":
    """Test the expert FSM v4."""
    import argparse
    from robotron import RobotronEnv

    parser = argparse.ArgumentParser(description='Expert FSM v4 Player')
    parser.add_argument('--level', type=int, default=1, help='Start Level')
    parser.add_argument('--lives', type=int, default=5, help='Start Lives')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes')
    parser.add_argument('--fps', type=int, default=0, help='FPS (0=unlimited)')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--headless', action='store_true', help='Headless mode')

    args = parser.parse_args()

    print("=" * 80)
    print("EXPERT FSM v4 PLAYER (Family Priority + Level Completion + Safe Alignment)")
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

            move_action, fire_action = expert_decide_v4(sprite_data)
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
