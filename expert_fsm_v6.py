"""
Expert FSM Player v6 for Robotron 2084

MAJOR REDESIGN: Goal-Oriented Planning with Entity Tracking

Key Improvements from v5:
1. Entity Tracking - Track position history to calculate velocity
2. Collision Prediction - Check if moves will get us hit (NEVER GET HIT!)
3. Goal-Oriented Planning - Commit to TARGETS (goals), not ACTIONS (directions)
4. Lead Shots - Shoot where enemies WILL BE, not where they are
5. Intercept Paths - Move to where enemies WILL BE (faster than chasing)
6. Bullet Dodge Prediction - Predict bullet trajectories and avoid

Philosophy:
- Turn-based game with perfect information = We should NEVER get hit
- Commit to goals, not actions
- Plan ahead instead of reacting
- Predict movement, don't chase current positions
"""

import math
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from collections import defaultdict, deque

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

# Direction names for debugging
DIRECTION_NAMES = {
    0: "STAY",
    1: "UP",
    2: "UP_RIGHT",
    3: "RIGHT",
    4: "DOWN_RIGHT",
    5: "DOWN",
    6: "DOWN_LEFT",
    7: "LEFT",
    8: "UP_LEFT"
}

# Movement deltas (pixels per frame)
MOVE_DELTAS = {
    STAY: (0, 0),
    UP: (0, -5),
    UP_RIGHT: (3.5, -3.5),
    RIGHT: (5, 0),
    DOWN_RIGHT: (3.5, 3.5),
    DOWN: (0, 5),
    DOWN_LEFT: (-3.5, 3.5),
    LEFT: (-5, 0),
    UP_LEFT: (-3.5, -3.5),
}


@dataclass
class Sprite:
    x: float
    y: float
    type: str
    distance: float = 0.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    id: str = ""  # Unique identifier


class EntityTracker:
    """
    Tracks entity positions over time to calculate velocity and predict future positions.
    """

    def __init__(self, history_size: int = 5):
        self.history_size = history_size
        # entity_id -> deque of (x, y, frame) tuples
        self.position_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self.frame_count = 0

    def update(self, sprites: List[Sprite]):
        """Update position history for all sprites."""
        self.frame_count += 1

        # Track all sprites
        for sprite in sprites:
            # Create ID from type and approximate position (for matching across frames)
            sprite_id = self.get_sprite_id(sprite)
            sprite.id = sprite_id

            # Add to history
            self.position_history[sprite_id].append((sprite.x, sprite.y, self.frame_count))

    def get_sprite_id(self, sprite: Sprite) -> str:
        """
        Generate a consistent ID for a sprite across frames.
        Uses type and grid position to match sprites.
        """
        # Round position to nearest 20px grid to handle small movements
        grid_x = int(sprite.x / 20) * 20
        grid_y = int(sprite.y / 20) * 20
        return f"{sprite.type}_{grid_x}_{grid_y}"

    def get_velocity(self, sprite: Sprite) -> Tuple[float, float]:
        """
        Calculate sprite's velocity from recent frames.
        Returns (vx, vy) in pixels per frame.
        """
        sprite_id = sprite.id or self.get_sprite_id(sprite)
        history = self.position_history.get(sprite_id, [])

        if len(history) < 2:
            return (0.0, 0.0)

        # Use last few frames for smoothing
        recent = list(history)[-min(3, len(history)):]

        # Calculate average velocity
        x1, y1, f1 = recent[0]
        x2, y2, f2 = recent[-1]

        frame_diff = f2 - f1
        if frame_diff == 0:
            return (0.0, 0.0)

        vx = (x2 - x1) / frame_diff
        vy = (y2 - y1) / frame_diff

        return (vx, vy)

    def predict_position(self, sprite: Sprite, frames_ahead: int) -> Tuple[float, float]:
        """
        Predict where sprite will be in N frames.
        Returns (future_x, future_y).
        """
        vx, vy = self.get_velocity(sprite)
        future_x = sprite.x + vx * frames_ahead
        future_y = sprite.y + vy * frames_ahead

        # Clamp to board boundaries
        future_x = max(0, min(BOARD_WIDTH, future_x))
        future_y = max(0, min(BOARD_HEIGHT, future_y))

        return (future_x, future_y)


class ExpertFSMv6:
    """
    Goal-oriented FSM with entity tracking and collision prediction.
    """

    def __init__(self):
        self.player_pos = None
        self.enemies = []
        self.family = []

        # Entity tracking
        self.tracker = EntityTracker(history_size=5)

        # Goal-oriented planning
        self.current_goal = None  # The sprite we're pursuing
        self.current_goal_type = None  # 'enemy' or 'civilian'
        self.plan = []  # List of moves to execute

        # Strafing (for "spraying" enemies)
        self.strafe_direction = None  # Direction to strafe
        self.strafe_frames = 0  # How many frames to hold strafe

        # Constants
        # Type-specific collision radii (from death analysis)
        self.COLLISION_RADII = {
            'Electrode': 35,  # Was 20 - accounts for visual size
            'Hulk': 40,       # Was 20 - immortal, MUST avoid (reduced when surrounded)
            'Grunt': 35,      # Iteration 6: Was 30px, increased to 35px (deaths at 30-39px range)
            'Prog': 35,       # Same as Grunt (converted civilians)
            'default': 20     # Other enemies
        }
        self.PLAYER_SPEED = 5  # Player movement speed (pixels/frame)
        self.BULLET_SPEED = 8  # Bullet speed (pixels/frame)
        self.GRUNT_SPEED = 3  # Grunt chase speed
        self.HULK_SPEED = 2  # Hulk wander speed

        # Debug
        self._debug_frame_counter = 0

    def parse_sprites(self, sprite_data: List[Tuple]) -> None:
        """Parse sprite data and update entity tracker."""
        self.enemies = []
        self.family = []
        self.player_pos = None

        all_sprites = []

        for sprite in sprite_data:
            x, y, sprite_type = sprite

            if sprite_type == 'Player':
                self.player_pos = Sprite(x, y, sprite_type)
                all_sprites.append(self.player_pos)
                continue

            if sprite_type == 'Bullet':
                continue

            s = Sprite(x, y, sprite_type)
            all_sprites.append(s)

            if sprite_type in ['Mommy', 'Daddy', 'Mikey']:
                self.family.append(s)
            elif sprite_type != 'Hulk':  # Hulk is immortal, but track as obstacle
                self.enemies.append(s)

            # Also track Hulks as obstacles
            if sprite_type == 'Hulk':
                self.enemies.append(s)

        # Update entity tracker
        self.tracker.update(all_sprites)

        # Calculate distances and velocities
        if self.player_pos:
            for sprites in [self.enemies, self.family]:
                for s in sprites:
                    dx = s.x - self.player_pos.x
                    dy = s.y - self.player_pos.y
                    s.distance = math.hypot(dx, dy)

                    # Get velocity from tracker
                    vx, vy = self.tracker.get_velocity(s)
                    s.velocity_x = vx
                    s.velocity_y = vy

            self.enemies.sort(key=lambda s: s.distance)
            self.family.sort(key=lambda s: s.distance)

    def is_in_center(self, x: float, y: float) -> bool:
        """Check if position is in dangerous center area."""
        center_x = BOARD_WIDTH / 2
        center_y = BOARD_HEIGHT / 2

        dx = abs(x - center_x)
        dy = abs(y - center_y)

        # Center danger zone is 200x200 area
        return dx < 100 and dy < 100

    def is_near_wall(self, x: float, y: float, threshold: float = 100) -> bool:
        """
        Check if position is near a wall/corner.

        Iteration 5: Detect corner danger to avoid getting trapped by bullets.
        User feedback: "spawners spawn enforcers which force us into a corner"

        Args:
            x, y: Position to check
            threshold: Distance from wall to consider "near"

        Returns:
            True if within threshold pixels of any wall
        """
        dist_to_left = x
        dist_to_right = BOARD_WIDTH - x
        dist_to_top = y
        dist_to_bottom = BOARD_HEIGHT - y

        min_dist = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)
        return min_dist < threshold

    def count_escape_corridors_from(self, x: float, y: float) -> int:
        """
        Count number of directions with open escape routes from a given position.

        Iteration 6: Proactive escape planning to avoid getting trapped.
        Death analysis showed 24% of deaths are "TRAPPED - no safe moves".

        An escape corridor is a direction where:
        - No walls within 150px
        - No bullets within 100px in that direction

        Args:
            x, y: Position to check from

        Returns:
            Number of safe escape directions (0-8)
        """
        escape_count = 0

        for move in [UP, UP_RIGHT, RIGHT, DOWN_RIGHT, DOWN, DOWN_LEFT, LEFT, UP_LEFT]:
            dx, dy = MOVE_DELTAS[move]

            # Check 150px in this direction
            check_x = x + dx * 30  # 30 frames * 5px = 150px
            check_y = y + dy * 30

            # Wall check - is this direction blocked by wall?
            if not (0 < check_x < BOARD_WIDTH and 0 < check_y < BOARD_HEIGHT):
                continue  # This direction leads to wall

            # Threat check - are there bullets in this direction?
            has_threat = False
            for enemy in self.enemies:
                if enemy.type in ['EnforcerBullet', 'TankShell', 'CruiseMissile']:
                    # Check if bullet is in this direction (within 45° cone)
                    threat_dx = enemy.x - x
                    threat_dy = enemy.y - y

                    # Only check if bullet is somewhat close
                    if math.hypot(threat_dx, threat_dy) > 150:
                        continue

                    # Calculate angle between direction and threat
                    if abs(dx) > 0.1 or abs(dy) > 0.1:
                        # Normalize direction vector
                        dir_mag = math.hypot(dx, dy)
                        dir_x = dx / dir_mag
                        dir_y = dy / dir_mag

                        # Normalize threat vector
                        threat_mag = math.hypot(threat_dx, threat_dy)
                        if threat_mag > 0:
                            threat_x = threat_dx / threat_mag
                            threat_y = threat_dy / threat_mag

                            # Dot product to check alignment (> 0.7 means within ~45°)
                            alignment = dir_x * threat_x + dir_y * threat_y
                            if alignment > 0.7 and math.hypot(threat_dx, threat_dy) < 100:
                                has_threat = True
                                break

            if not has_threat:
                escape_count += 1

        return escape_count

    def count_escape_corridors(self) -> int:
        """
        Count number of directions with open escape routes from current position.

        Iteration 6: Proactive escape planning to avoid getting trapped.

        Returns:
            Number of safe escape directions (0-8)
        """
        if not self.player_pos:
            return 0
        return self.count_escape_corridors_from(self.player_pos.x, self.player_pos.y)

    def predict_bullet_collision(self, bullet: Sprite) -> Tuple[bool, int, Tuple[float, float]]:
        """
        Iteration 7: Predict if bullet will collide with us based on its trajectory.

        Level 9 analysis: ALL bullet deaths at 16-28px despite 120px dodge threshold!
        Problem: Bullets move at 8-10 px/frame, we move at 5 px/frame.
        Solution: Predict trajectory, not just distance.

        Args:
            bullet: The bullet sprite to check

        Returns:
            (will_collide, frames_until_collision, collision_point)
        """
        if not self.player_pos:
            return (False, -1, (0, 0))

        bx, by = bullet.velocity_x, bullet.velocity_y

        # If bullet not moving or very slow, use distance-based fallback
        if abs(bx) < 0.5 and abs(by) < 0.5:
            # Stationary bullet - just check distance
            will_hit = bullet.distance < 120
            frames = int(bullet.distance / 8) if will_hit else -1
            return (will_hit, frames, (bullet.x, bullet.y))

        # Predict bullet path for next 30 frames
        for frame in range(1, 31):
            # Bullet future position
            future_bx = bullet.x + bx * frame
            future_by = bullet.y + by * frame

            # Check if bullet goes off board
            if not (0 < future_bx < BOARD_WIDTH and 0 < future_by < BOARD_HEIGHT):
                break

            # Player current position (assuming we don't move for worst-case)
            # This is intentionally conservative - if bullet will hit our current pos
            future_px = self.player_pos.x
            future_py = self.player_pos.y

            # Check collision
            dist = math.hypot(future_bx - future_px, future_by - future_py)
            if dist < 35:  # Collision radius for bullets
                return (True, frame, (future_bx, future_by))

        return (False, -1, (0, 0))

    def get_bullet_dodge_direction(self, bullet: Sprite) -> int:
        """
        Iteration 7: Get best direction to dodge bullet based on its trajectory.

        Key insight: Move PERPENDICULAR to bullet path, not just away!
        This makes the bullet MISS us instead of just delaying the hit.

        Args:
            bullet: The bullet to dodge

        Returns:
            Best dodge direction
        """
        if not self.player_pos:
            return STAY

        bx, by = bullet.velocity_x, bullet.velocity_y

        # If bullet not moving, just move away
        if abs(bx) < 0.5 and abs(by) < 0.5:
            return self.get_direction_away(bullet.x, bullet.y)

        # Calculate PERPENDICULAR directions to bullet trajectory
        # Bullet moving in direction (bx, by)
        # Perpendicular directions: (by, -bx) and (-by, bx)

        # Try both perpendicular directions and pick safer one
        perp1_x = by
        perp1_y = -bx
        perp2_x = -by
        perp2_y = bx

        # Normalize perpendicular vectors
        perp1_mag = math.hypot(perp1_x, perp1_y)
        perp2_mag = math.hypot(perp2_x, perp2_y)

        if perp1_mag > 0:
            perp1_x = perp1_x / perp1_mag * 30
            perp1_y = perp1_y / perp1_mag * 30
        if perp2_mag > 0:
            perp2_x = perp2_x / perp2_mag * 30
            perp2_y = perp2_y / perp2_mag * 30

        # Score both directions
        future1_x = self.player_pos.x + perp1_x
        future1_y = self.player_pos.y + perp1_y
        future2_x = self.player_pos.x + perp2_x
        future2_y = self.player_pos.y + perp2_y

        # Score based on wall distance (want to stay away from walls)
        score1 = min(future1_x, BOARD_WIDTH - future1_x,
                    future1_y, BOARD_HEIGHT - future1_y)
        score2 = min(future2_x, BOARD_WIDTH - future2_x,
                    future2_y, BOARD_HEIGHT - future2_y)

        # Choose better perpendicular direction
        if score1 > score2:
            return self.get_direction_to(future1_x, future1_y)
        else:
            return self.get_direction_to(future2_x, future2_y)

    def will_collide(self, my_future_pos: Tuple[float, float], threat: Sprite, frames_ahead: int, collision_radius: float = None) -> bool:
        """
        Check if we'll collide with a threat at a future position.

        Args:
            my_future_pos: Our predicted position (x, y)
            threat: The threatening sprite
            frames_ahead: How many frames to look ahead
            collision_radius: Optional override for collision radius

        Returns:
            True if collision predicted, False otherwise
        """
        if collision_radius is None:
            collision_radius = self.COLLISION_RADII.get(threat.type, self.COLLISION_RADII['default'])

        threat_future_x, threat_future_y = self.tracker.predict_position(threat, frames_ahead)

        # Check collision at each frame
        for frame in range(1, frames_ahead + 1):
            # Interpolate our position
            t = frame / frames_ahead
            my_x = self.player_pos.x + (my_future_pos[0] - self.player_pos.x) * t
            my_y = self.player_pos.y + (my_future_pos[1] - self.player_pos.y) * t

            # Interpolate threat position
            t_x = threat.x + (threat_future_x - threat.x) * t
            t_y = threat.y + (threat_future_y - threat.y) * t

            # Check collision
            dist = math.hypot(my_x - t_x, my_y - t_y)
            if dist < collision_radius:
                return True

        return False

    def is_move_safe(self, move_action: int, look_ahead_frames: int = 3, allow_aggressive: bool = False) -> bool:
        """
        Check if a move is safe (won't get us hit).

        This is the KEY to never getting hit - check every move before making it!
        Uses type-specific collision radii from death analysis.

        Args:
            move_action: The move to check
            look_ahead_frames: How many frames to predict ahead
            allow_aggressive: If True, use relaxed Hulk collision checks when pursuing high-value targets
        """
        if not self.player_pos:
            return True

        # Calculate future position
        dx, dy = MOVE_DELTAS[move_action]
        future_x = self.player_pos.x + dx * look_ahead_frames
        future_y = self.player_pos.y + dy * look_ahead_frames

        # Clamp to board
        future_x = max(0, min(BOARD_WIDTH, future_x))
        future_y = max(0, min(BOARD_HEIGHT, future_y))

        my_future_pos = (future_x, future_y)

        # Detect Hulk-box situation (surrounded by multiple Hulks)
        # User constraint: "Just be sure we don't lock ourself in a box of multiple hulks"
        hulk_count_nearby = sum(1 for e in self.enemies if e.type == 'Hulk' and e.distance < 150)

        # Check collision with all threats
        for enemy in self.enemies:
            # Get type-specific collision radius
            collision_radius = self.COLLISION_RADII.get(enemy.type, self.COLLISION_RADII['default'])

            # Special handling for Hulks:
            if enemy.type == 'Hulk':
                # If surrounded by 3+ Hulks, reduce radius to allow threading through
                if hulk_count_nearby >= 3:
                    collision_radius = 25  # Reduced from 40px
                # User feedback: "if a grunt is surrounded by hulks, we're afraid to go kill them"
                # Solution: When aggressively pursuing enemies, use relaxed Hulk collision (30px instead of 40px)
                elif allow_aggressive:
                    collision_radius = 30  # Relaxed from 40px to allow pursuing enemies near Hulks

            # Electrodes don't move, just check distance
            if enemy.type == 'Electrode':
                dist = math.hypot(future_x - enemy.x, future_y - enemy.y)
                if dist < collision_radius:
                    return False
                continue

            # Hulks are dangerous (immortal obstacles)
            if enemy.type == 'Hulk':
                if self.will_collide(my_future_pos, enemy, look_ahead_frames, collision_radius):
                    return False

            # Projectiles are VERY dangerous
            if enemy.type in ['EnforcerBullet', 'TankShell', 'CruiseMissile']:
                if self.will_collide(my_future_pos, enemy, look_ahead_frames, collision_radius):
                    return False

            # Regular enemies - if collision predicted, mark as unsafe
            # (Fixed: was only marking unsafe if distance < 30, causing Grunt deaths at 31-37px)
            if self.will_collide(my_future_pos, enemy, look_ahead_frames, collision_radius):
                return False

        return True

    def find_safe_moves(self, allow_aggressive: bool = False) -> List[int]:
        """
        Find all safe moves (moves that won't get us hit).
        Returns list of safe move directions.

        Args:
            allow_aggressive: If True, use relaxed Hulk collision checks when pursuing high-value targets
        """
        safe_moves = []

        # Check all 8 directions + stay
        for move in [STAY, UP, UP_RIGHT, RIGHT, DOWN_RIGHT, DOWN, DOWN_LEFT, LEFT, UP_LEFT]:
            if self.is_move_safe(move, allow_aggressive=allow_aggressive):
                safe_moves.append(move)

        return safe_moves

    def count_future_safe_moves(self, move_action: int) -> int:
        """
        Simulate a move and count how many safe moves we'd have after making it.
        Used for escape route planning.
        """
        if not self.player_pos:
            return 0

        # Temporarily move player
        original_x = self.player_pos.x
        original_y = self.player_pos.y

        dx, dy = MOVE_DELTAS[move_action]
        self.player_pos.x += dx
        self.player_pos.y += dy

        # Count safe moves from this position
        future_safe_count = len(self.find_safe_moves())

        # Restore original position
        self.player_pos.x = original_x
        self.player_pos.y = original_y

        return future_safe_count

    def calculate_lead_shot(self, target: Sprite) -> Optional[int]:
        """
        Calculate where to shoot to hit a moving target.
        Shoot where enemy WILL BE, not where they are.
        """
        if not self.player_pos:
            return None

        # If target isn't moving, just shoot directly
        if abs(target.velocity_x) < 0.5 and abs(target.velocity_y) < 0.5:
            return self.can_shoot_at_position(target.x, target.y)

        # Calculate time for bullet to reach target
        distance = target.distance
        travel_time = distance / self.BULLET_SPEED

        # Predict where target will be
        future_x, future_y = self.tracker.predict_position(target, int(travel_time))

        # Check if we can shoot at that future position
        return self.can_shoot_at_position(future_x, future_y)

    def can_shoot_at_position(self, target_x: float, target_y: float) -> Optional[int]:
        """
        Check if we can shoot at a position (is it on a firing line?).
        Returns fire direction if yes, None if no.
        """
        if not self.player_pos:
            return None

        dx = target_x - self.player_pos.x
        dy = target_y - self.player_pos.y

        THRESHOLD = 30

        # RIGHT
        if abs(dy) < THRESHOLD and dx > 0:
            return RIGHT
        # UP_RIGHT
        if abs(dx - dy) < THRESHOLD and dx > 0 and dy < 0:
            return UP_RIGHT
        # UP
        if abs(dx) < THRESHOLD and dy < 0:
            return UP
        # UP_LEFT
        if abs(dx + dy) < THRESHOLD and dx < 0 and dy < 0:
            return UP_LEFT
        # LEFT
        if abs(dy) < THRESHOLD and dx < 0:
            return LEFT
        # DOWN_LEFT
        if abs(dx - dy) < THRESHOLD and dx < 0 and dy > 0:
            return DOWN_LEFT
        # DOWN
        if abs(dx) < THRESHOLD and dy > 0:
            return DOWN
        # DOWN_RIGHT
        if abs(dx + dy) < THRESHOLD and dx > 0 and dy > 0:
            return DOWN_RIGHT

        return None

    def can_shoot_at(self, target: Sprite) -> Optional[int]:
        """Legacy method - just use current position."""
        return self.can_shoot_at_position(target.x, target.y)

    def find_opportunistic_shot(self) -> Optional[int]:
        """
        ALWAYS shoot at SOMETHING.

        Iteration 7: Prioritize INCOMING bullets that will hit us (active defense!)

        Priority:
        1a. INCOMING bullets that will hit us - DESTROY THEM! (NEW)
        1b. Other bullets within 300px
        2. Spawners (create more enemies)
        3. Shooters (dangerous)
        4. Regular enemies (Grunts)
        5. Obstacles (Electrodes) - give points!
        6. Last resort: shoot toward nearest threat
        """
        # PRIORITY 1a: INCOMING bullets that will hit us - DESTROY THEM!
        # User insight: "remember we can shoot bullets" - use offense as defense!
        for enemy in self.enemies:
            if enemy.type in ['EnforcerBullet', 'TankShell', 'CruiseMissile']:
                # Check if bullet is coming AT us
                will_hit, frames, collision_pt = self.predict_bullet_collision(enemy)

                if will_hit and frames < 20:  # Will hit us in next 20 frames!
                    # Try to shoot it!
                    fire_dir = self.calculate_lead_shot(enemy)
                    if fire_dir is not None:
                        # CRITICAL: Shoot incoming bullets first!
                        return fire_dir

        # PRIORITY 1b: Other bullets within shooting range
        for enemy in self.enemies:
            if enemy.type in ['EnforcerBullet', 'TankShell', 'CruiseMissile']:
                if enemy.distance < 300:
                    fire_dir = self.calculate_lead_shot(enemy)
                    if fire_dir is not None:
                        return fire_dir

        # PRIORITY 2: Spawners (Brain, Sphereoid, Quark)
        for enemy in self.enemies:
            if enemy.type in ['Brain', 'Sphereoid', 'Quark']:
                if enemy.distance < 300:
                    fire_dir = self.calculate_lead_shot(enemy)
                    if fire_dir is not None:
                        return fire_dir

        # PRIORITY 3: Shooters (Enforcer, Tank)
        for enemy in self.enemies:
            if enemy.type in ['Enforcer', 'Tank']:
                if enemy.distance < 300:
                    fire_dir = self.calculate_lead_shot(enemy)
                    if fire_dir is not None:
                        return fire_dir

        # PRIORITY 4: Regular enemies (Grunts, Progs, etc.)
        for enemy in self.enemies:
            if enemy.type not in ['Electrode', 'Hulk', 'EnforcerBullet', 'TankShell', 'CruiseMissile',
                                 'Brain', 'Sphereoid', 'Quark', 'Enforcer', 'Tank']:
                if enemy.distance < 300:
                    fire_dir = self.calculate_lead_shot(enemy)
                    if fire_dir is not None:
                        return fire_dir

        # PRIORITY 5: Obstacles (Electrodes) - give points and block paths!
        for enemy in self.enemies:
            if enemy.type == 'Electrode':
                if enemy.distance < 300:
                    # Electrodes don't move - just shoot directly
                    fire_dir = self.can_shoot_at(enemy)
                    if fire_dir is not None:
                        return fire_dir

        # PRIORITY 6: Last resort - shoot toward nearest threat (even if not perfectly aligned)
        # This ensures we're ALWAYS shooting
        if self.enemies:
            nearest = self.enemies[0]
            if nearest.type != 'Hulk':  # Don't waste shots on immortal Hulks
                # Shoot in general direction
                dx = nearest.x - self.player_pos.x
                dy = nearest.y - self.player_pos.y

                # Find closest firing direction
                angle = math.atan2(-dy, dx)
                angle_deg = math.degrees(angle)
                if angle_deg < 0:
                    angle_deg += 360

                # Convert to nearest 8-way direction
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

        return STAY  # Only if no enemies at all

    def calculate_strafe_intercept(self, target: Sprite) -> Tuple[float, float]:
        """
        STRAFE enemies instead of trying to hold perfect position.

        Key insight: "Spray" enemies by strafing across them while shooting.
        This avoids jittering and is more decisive.

        Strategy:
        1. Get close to enemy (within shooting range)
        2. Strafe perpendicular to them (circle around)
        3. Shoot while moving (natural "spraying")
        4. Commit to strafe direction to avoid jittering
        """
        if not self.player_pos:
            return (target.x, target.y)

        # Predict where target will be
        if abs(target.velocity_x) < 0.5 and abs(target.velocity_y) < 0.5:
            future_x, future_y = target.x, target.y
        else:
            distance_to_target = target.distance
            frames_ahead = int(distance_to_target / self.PLAYER_SPEED)
            future_x, future_y = self.tracker.predict_position(target, frames_ahead)

        current_dist = target.distance

        # If too far (>200px), just move toward enemy
        if current_dist > 200:
            return (future_x, future_y)

        # If too close (<70px), back up
        if current_dist < 70:
            # Move away from enemy
            away_x = self.player_pos.x + (self.player_pos.x - future_x) * 2
            away_y = self.player_pos.y + (self.player_pos.y - future_y) * 2
            return (away_x, away_y)

        # In good range (70-200px) - STRAFE!
        # Move perpendicular to enemy to "spray" them

        # Calculate perpendicular direction (tangent to circle around enemy)
        dx = future_x - self.player_pos.x
        dy = future_y - self.player_pos.y

        # Perpendicular vector (rotate 90 degrees)
        # Use strafe direction if we have one, otherwise pick one
        if self.strafe_frames > 0:
            # Continue current strafe
            self.strafe_frames -= 1
        else:
            # Choose new strafe direction (perpendicular to enemy)
            # Randomly pick clockwise or counter-clockwise
            import random
            clockwise = random.choice([True, False])

            if clockwise:
                # Rotate 90 degrees clockwise: (x, y) -> (y, -x)
                perp_dx = dy
                perp_dy = -dx
            else:
                # Rotate 90 degrees counter-clockwise: (x, y) -> (-y, x)
                perp_dx = -dy
                perp_dy = dx

            # Normalize and scale
            perp_dist = math.hypot(perp_dx, perp_dy)
            if perp_dist > 0:
                perp_dx = perp_dx / perp_dist * 100
                perp_dy = perp_dy / perp_dist * 100

            self.strafe_direction = (perp_dx, perp_dy)
            self.strafe_frames = 15  # Commit for 15 frames (no jittering!)

        # Move in strafe direction
        if self.strafe_direction:
            strafe_x = self.player_pos.x + self.strafe_direction[0]
            strafe_y = self.player_pos.y + self.strafe_direction[1]

            # Clamp to board
            strafe_x = max(20, min(BOARD_WIDTH - 20, strafe_x))
            strafe_y = max(20, min(BOARD_HEIGHT - 20, strafe_y))

            return (strafe_x, strafe_y)

        # Fallback
        return (future_x, future_y)

    def calculate_intercept_path(self, target: Sprite) -> Tuple[float, float]:
        """
        Calculate where to move to engage target.
        Uses strafing for enemies (spray them), direct path for civilians.
        """
        if self.current_goal_type == 'civilian':
            # Civilians: just go to them directly
            return (target.x, target.y)
        else:
            # Enemies: strafe to spray them
            return self.calculate_strafe_intercept(target)

    def get_direction_away(self, target_x: float, target_y: float) -> int:
        """
        Get direction AWAY from target position.
        Uses 8-way direction based on angle.
        """
        if not self.player_pos:
            return STAY

        # Calculate direction TO target, then reverse it
        dx = target_x - self.player_pos.x
        dy = target_y - self.player_pos.y

        # Reverse direction (move away)
        dx = -dx
        dy = -dy

        if abs(dx) < 5 and abs(dy) < 5:
            return STAY

        # Calculate angle (negate dy for screen coordinates)
        angle = math.atan2(-dy, dx)
        angle_deg = math.degrees(angle)
        if angle_deg < 0:
            angle_deg += 360

        # Convert to 8-way direction
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
        """
        Get direction to target position.
        Uses 8-way direction based on angle.
        """
        if not self.player_pos:
            return STAY

        dx = target_x - self.player_pos.x
        dy = target_y - self.player_pos.y

        if abs(dx) < 5 and abs(dy) < 5:
            return STAY

        # Calculate angle (negate dy for screen coordinates)
        angle = math.atan2(-dy, dx)
        angle_deg = math.degrees(angle)
        if angle_deg < 0:
            angle_deg += 360

        # Convert to 8-way direction
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

    def find_goal(self) -> Optional[Sprite]:
        """
        Choose a goal (target) to pursue.

        Iteration 7: Use trajectory prediction for bullet threats (not just distance!)

        Priority:
        1. Dodge immediate threats (bullets, Hulks very close)
        2. Keep current goal if still valid
        3. Civilians if safe (for extra lives)
        4. Dangerous enemies (Spawners > Shooters > Others)
        """

        # PRIORITY 1: Immediate threats that we MUST dodge
        # Iteration 7: Use TRAJECTORY PREDICTION for bullets (not just distance!)
        immediate_threats = []
        for enemy in self.enemies:
            # Bullets are highest threat - use trajectory prediction!
            if enemy.type in ['EnforcerBullet', 'TankShell', 'CruiseMissile']:
                # Check if bullet will hit us based on trajectory
                will_hit, frames, collision_pt = self.predict_bullet_collision(enemy)

                if will_hit and frames < 15:  # Will hit in next 15 frames!
                    # This bullet is ACTUALLY coming at us - must dodge!
                    immediate_threats.append(enemy)
                elif enemy.distance < 80:  # Also dodge very close bullets (backup)
                    immediate_threats.append(enemy)

            # Hulks are immortal - must dodge
            elif enemy.type == 'Hulk':
                if enemy.distance < 50:
                    immediate_threats.append(enemy)

        if immediate_threats:
            # Just return closest threat (we'll dodge in decide_action)
            closest_threat = min(immediate_threats, key=lambda e: e.distance)
            self.current_goal = closest_threat
            self.current_goal_type = 'dodge'
            return self.current_goal

        # PRIORITY 2: Keep current goal if still valid
        if self.current_goal and self.current_goal_type in ['enemy', 'civilian']:
            goal_list = self.enemies if self.current_goal_type == 'enemy' else self.family

            # Find goal in current list (match by approximate position)
            for sprite in goal_list:
                if (abs(sprite.x - self.current_goal.x) < 50 and
                    abs(sprite.y - self.current_goal.y) < 50 and
                    sprite.type == self.current_goal.type):
                    # Goal still exists - keep it
                    self.current_goal = sprite
                    return self.current_goal

            # Goal no longer exists - clear it and reset strafe
            self.current_goal = None
            self.strafe_frames = 0  # Reset strafe when target dies

        # PRIORITY 3: Civilians if safe (they give extra lives!)
        if self.family:
            nearest_civilian = self.family[0]

            # Collect if:
            # - No dangerous enemies within 150px
            # - OR civilian is very close (< 80px)
            dangerous_close = [e for e in self.enemies
                             if e.distance < 150 and
                             e.type not in ['Electrode', 'Hulk']]

            if len(dangerous_close) == 0 or nearest_civilian.distance < 80:
                self.current_goal = nearest_civilian
                self.current_goal_type = 'civilian'
                return self.current_goal

        # PRIORITY 4: Dangerous enemies
        if self.enemies:
            # Filter out obstacles
            targetable_enemies = [e for e in self.enemies
                                if e.type not in ['Electrode', 'Hulk']]

            if targetable_enemies:
                # Score enemies by danger
                def enemy_danger_score(enemy):
                    base_score = 200 - enemy.distance

                    # Spawners (CRITICAL priority - they spawn enemies that overwhelm us!)
                    # User feedback: "we wait too long for spawners... which overwhelm us"
                    if enemy.type in ['Brain', 'Sphereoid', 'Quark']:
                        base_score += 1000  # Was 300, now 1000 - kill spawners ASAP!

                    # Shooters (DANGEROUS from range - their bullets corner us!)
                    # Iteration 5: Boost priority when close to prevent bullet swarm
                    # User feedback: "spawners spawn enforcers which force us into a corner"
                    if enemy.type in ['Enforcer', 'Tank']:
                        if enemy.distance < 200:
                            base_score += 500  # High priority when close!
                        else:
                            base_score += 200  # Normal priority when far

                    # Grunts (chase aggressively)
                    if enemy.type == 'Grunt':
                        base_score += 50

                    return base_score

                most_dangerous = max(targetable_enemies, key=enemy_danger_score)

                # Reset strafe if switching to new target
                if self.current_goal != most_dangerous:
                    self.strafe_frames = 0

                self.current_goal = most_dangerous
                self.current_goal_type = 'enemy'
                return self.current_goal

        # PRIORITY 5: Any remaining civilians
        if self.family:
            self.current_goal = self.family[0]
            self.current_goal_type = 'civilian'
            return self.current_goal

        # No goals
        self.current_goal = None
        return None

    def decide_action(self, sprite_data: List[Tuple]) -> Tuple[int, int]:
        """
        Main decision function with goal-oriented planning and collision prediction.

        Decision Hierarchy:
        1. SAFETY CHECK: Will this move get us hit? → Find safe move
        2. SHOOT: Can we shoot at any enemy? → Fire
        3. EXECUTE PLAN: Move toward goal using intercept path
        """
        self.parse_sprites(sprite_data)

        if not self.player_pos:
            return (STAY, STAY)

        move_action = STAY
        fire_action = STAY
        move_reason = "No valid move"
        shoot_reason = "Not shooting"

        # STEP 1: Find/maintain goal
        goal = self.find_goal()

        # STEP 2: ALWAYS shoot opportunistically (with lead shots!)
        opportunistic_shot = self.find_opportunistic_shot()
        if opportunistic_shot:
            fire_action = opportunistic_shot
            # Find which enemy we're shooting at (check all priorities)
            found_target = False

            # Check all enemy types in priority order
            for enemy in self.enemies:
                if enemy.distance < 300:
                    if enemy.type == 'Electrode':
                        # Electrodes don't move
                        fire_dir = self.can_shoot_at(enemy)
                        if fire_dir == opportunistic_shot:
                            shoot_reason = f"Shooting Electrode (dist={enemy.distance:.0f})"
                            found_target = True
                            break
                    elif enemy.type != 'Hulk':
                        # Moving enemies - use lead shot
                        lead_shot = self.calculate_lead_shot(enemy)
                        if lead_shot == opportunistic_shot:
                            shoot_reason = f"Lead shot at {enemy.type} (dist={enemy.distance:.0f}, v=[{enemy.velocity_x:.1f},{enemy.velocity_y:.1f}])"
                            found_target = True
                            break

            if not found_target:
                # Must be shooting toward nearest (fallback)
                if self.enemies:
                    nearest = self.enemies[0]
                    shoot_reason = f"Toward {nearest.type} (dist={nearest.distance:.0f})"

        # STEP 3: Find safe moves
        # Use aggressive pursuit mode when targeting enemies (not dodging)
        # This allows us to pursue enemies near Hulks with relaxed collision checks
        allow_aggressive = self.current_goal_type == 'enemy'
        safe_moves = self.find_safe_moves(allow_aggressive=allow_aggressive)

        if not safe_moves:
            # NO SAFE MOVES! Emergency - just stay and hope for the best
            move_action = STAY
            move_reason = "TRAPPED - no safe moves!"

        # PRIORITY 2 FIX: Emergency backup when enemy CRITICALLY close (<40px)
        # Tuned: 50px was too cautious, 40px is truly critical distance
        elif goal and self.current_goal_type == 'enemy' and goal.distance < 40:
            # EMERGENCY! Too close - back up immediately
            backup_move = self.get_direction_away(goal.x, goal.y)
            if backup_move in safe_moves:
                move_action = backup_move
                move_reason = f"EMERGENCY BACKUP - {goal.type} CRITICAL distance ({goal.distance:.0f}px < 40px)!"
            else:
                # Backup direction not safe, find safest move away
                best_move = None
                best_distance = 0
                for move in safe_moves:
                    dx, dy = MOVE_DELTAS[move]
                    future_x = self.player_pos.x + dx * 3
                    future_y = self.player_pos.y + dy * 3
                    dist = math.hypot(future_x - goal.x, future_y - goal.y)
                    if dist > best_distance:
                        best_distance = dist
                        best_move = move
                if best_move:
                    move_action = best_move
                    move_reason = f"EMERGENCY - {goal.type} too close, escaping!"

        # Iteration 6+7: ESCAPE MODE - Proactive escape when getting boxed in
        # Death analysis showed 24% of deaths are "TRAPPED - no safe moves"
        # Iteration 7: Added bullet collision trigger (< 10 frames)
        # Enter escape mode when safe move count is low (< 5) OR bullet will hit soon
        else:
            # Check for escape mode triggers
            escape_triggers = []

            # Trigger 1: Low safe moves (Iteration 7: changed from < 4 to < 5)
            if len(safe_moves) < 5 and len(safe_moves) > 0:
                escape_triggers.append(f"low_safe_moves={len(safe_moves)}")

            # Trigger 2: Bullet will hit soon (Iteration 7: NEW!)
            if len(safe_moves) > 0:  # Only check if we have moves available
                for enemy in self.enemies:
                    if enemy.type in ['EnforcerBullet', 'TankShell', 'CruiseMissile']:
                        will_hit, frames, _ = self.predict_bullet_collision(enemy)
                        if will_hit and frames < 10:  # Will hit in next 10 frames!
                            escape_triggers.append(f"bullet_collision_in_{frames}f")
                            break  # One bullet trigger is enough

            if escape_triggers:
                # ESCAPE MODE: Find move that maximizes future escape corridors
                best_move = None
                best_future_corridors = -1

                for move in safe_moves:
                    dx, dy = MOVE_DELTAS[move]
                    future_x = self.player_pos.x + dx * 3
                    future_y = self.player_pos.y + dy * 3

                    # Count escape corridors from future position
                    future_corridors = self.count_escape_corridors_from(future_x, future_y)

                    if future_corridors > best_future_corridors:
                        best_future_corridors = future_corridors
                        best_move = move

                if best_move:
                    move_action = best_move
                    triggers_str = ", ".join(escape_triggers)
                    move_reason = f"ESCAPE MODE ({triggers_str}) → seeking open space ({best_future_corridors} corridors)"
                elif len(safe_moves) > 0:
                    # Fall through to goal logic below
                    pass
                else:
                    # This case handled by earlier check
                    pass

            # If not in escape mode, continue with goal logic
            if not escape_triggers and goal:
                # We have a goal - move toward it

                if self.current_goal_type == 'dodge':
                    # Emergency dodge - move AWAY from threat
                    threat = goal

                    # Iteration 7: Use PERPENDICULAR dodging for bullets (trajectory-based!)
                    if threat.type in ['EnforcerBullet', 'TankShell', 'CruiseMissile']:
                        # For bullets: Try perpendicular dodge first (makes bullet MISS)
                        perp_dodge = self.get_bullet_dodge_direction(threat)

                        if perp_dodge in safe_moves:
                            # Perpendicular dodge is safe - use it!
                            move_action = perp_dodge
                            move_reason = f"PERPENDICULAR DODGE {threat.type} (trajectory-based, dist={threat.distance:.0f})"
                        else:
                            # Perpendicular not safe - fall back to distance-based
                            # Find safest move AWAY from threat
                            best_move = None
                            best_score = -999999

                            nearby_bullets = [e for e in self.enemies
                                             if e.type in ['EnforcerBullet', 'TankShell', 'CruiseMissile']
                                             and e.distance < 200]

                            for move in safe_moves:
                                dx, dy = MOVE_DELTAS[move]
                                future_x = self.player_pos.x + dx * 3
                                future_y = self.player_pos.y + dy * 3

                                # Distance from threat (higher is better)
                                dist = math.hypot(future_x - threat.x, future_y - threat.y)
                                score = dist

                                # Wall avoidance
                                if len(nearby_bullets) >= 1:
                                    if self.is_near_wall(future_x, future_y, threshold=150):
                                        score -= 250

                                if score > best_score:
                                    best_score = score
                                    best_move = move

                            if best_move:
                                move_action = best_move
                                move_reason = f"DODGING {threat.type} (fallback, dist={threat.distance:.0f})"
                    else:
                        # For non-bullets (Hulks): Use distance-based dodge
                        best_move = None
                        best_score = -999999

                        for move in safe_moves:
                            dx, dy = MOVE_DELTAS[move]
                            future_x = self.player_pos.x + dx * 3
                            future_y = self.player_pos.y + dy * 3

                            # Distance from threat (higher is better)
                            dist = math.hypot(future_x - threat.x, future_y - threat.y)
                            score = dist

                            if score > best_score:
                                best_score = score
                                best_move = move

                        if best_move:
                            move_action = best_move
                            move_reason = f"DODGING {threat.type} (dist={threat.distance:.0f})"

                else:
                    # Normal goal pursuit - use intercept path
                    intercept_x, intercept_y = self.calculate_intercept_path(goal)

                    # Get direction to intercept point
                    desired_move = self.get_direction_to(intercept_x, intercept_y)

                    # Check if desired move is safe
                    if desired_move in safe_moves:
                        move_action = desired_move
                        if self.current_goal_type == 'civilian':
                            move_reason = f"Intercept {goal.type} at ({intercept_x:.0f},{intercept_y:.0f}) dist={goal.distance:.0f}"
                        else:
                            strafe_info = f" STRAFE({self.strafe_frames})" if self.strafe_frames > 0 else ""
                            move_reason = f"Strafe {goal.type} at ({intercept_x:.0f},{intercept_y:.0f}) dist={goal.distance:.0f}{strafe_info}"
                    else:
                        # Desired move isn't safe - find closest safe alternative
                        best_move = None
                        best_score = -999999

                        # Iteration 6: Check for bullet swarm (more aggressive)
                        nearby_bullets = [e for e in self.enemies
                                         if e.type in ['EnforcerBullet', 'TankShell', 'CruiseMissile']
                                         and e.distance < 200]

                        for move in safe_moves:
                            dx, dy = MOVE_DELTAS[move]
                            future_x = self.player_pos.x + dx * 3
                            future_y = self.player_pos.y + dy * 3

                            # Score: closer to intercept point + not in center
                            dist_to_intercept = math.hypot(future_x - intercept_x,
                                                          future_y - intercept_y)
                            score = -dist_to_intercept

                            # Penalty for center
                            if self.is_in_center(future_x, future_y):
                                score -= 100

                            # Iteration 6: More aggressive wall avoidance
                            # - Trigger with 1+ bullets (was 2+)
                            # - 150px threshold (was 100px)
                            # - Higher penalty (200 vs 150)
                            if len(nearby_bullets) >= 1:  # Changed from >= 2
                                if self.is_near_wall(future_x, future_y, threshold=150):  # Changed from 100
                                    score -= 200  # Increased from 150

                            if score > best_score:
                                best_score = score
                                best_move = move

                        if best_move:
                            move_action = best_move
                            move_reason = f"Safe alternative to {goal.type} (desired unsafe)"
            elif not escape_triggers and len(safe_moves) > 0:
                # No goal - patrol edges
                # Choose safe move that keeps us on edges
                best_move = None
                best_score = -999999

                for move in safe_moves:
                    dx, dy = MOVE_DELTAS[move]
                    future_x = self.player_pos.x + dx * 3
                    future_y = self.player_pos.y + dy * 3

                    # Score: distance from center + distance from nearest edge
                    center_dist = math.hypot(future_x - BOARD_WIDTH/2,
                                            future_y - BOARD_HEIGHT/2)

                    # Edge distance (want to be close to an edge)
                    edge_dist = min(
                        future_x,
                        BOARD_WIDTH - future_x,
                        future_y,
                        BOARD_HEIGHT - future_y
                    )

                    # Want high center distance, low edge distance
                    score = center_dist - edge_dist

                    if score > best_score:
                        best_score = score
                        best_move = move

                if best_move:
                    move_action = best_move
                    move_reason = "PATROLLING edges (no goal)"

        # Debug output (every 20 frames)
        self._debug_frame_counter += 1
        debug_this_frame = self._debug_frame_counter % 20 == 0

        if debug_this_frame:
            player_info = f"Player at ({self.player_pos.x:.0f}, {self.player_pos.y:.0f})"

            goal_info = "No goal"
            if goal:
                goal_info = f"{self.current_goal_type.upper()}: {goal.type} at ({goal.x:.0f}, {goal.y:.0f}) dist={goal.distance:.0f} v=[{goal.velocity_x:.1f},{goal.velocity_y:.1f}]"

            # Enemies list
            enemies_list = "No enemies"
            if self.enemies:
                enemy_strs = [f"{e.type}({e.distance:.0f})" for e in self.enemies[:10]]
                enemies_list = f"Enemies: {', '.join(enemy_strs)}"
                if len(self.enemies) > 10:
                    enemies_list += f" ... +{len(self.enemies) - 10} more"

            # Safe moves
            safe_move_names = [DIRECTION_NAMES[m] for m in safe_moves]
            safe_info = f"Safe moves: {', '.join(safe_move_names)}"

            move_dir_name = DIRECTION_NAMES.get(move_action, f"UNKNOWN({move_action})")
            fire_dir_name = DIRECTION_NAMES.get(fire_action, f"UNKNOWN({fire_action})")

            print(f"[FSM v6] {player_info}")
            print(f"[FSM v6] {goal_info}")
            print(f"[FSM v6] {enemies_list}")
            print(f"[FSM v6] {safe_info}")
            print(f"[FSM v6] Move: {move_dir_name} - {move_reason}")
            print(f"[FSM v6] Fire: {fire_dir_name} - {shoot_reason}")
            print()

        return (move_action, fire_action)


# Global FSM instance
_fsm = ExpertFSMv6()


def expert_decide_v6(sprite_data: List[Tuple]) -> Tuple[int, int]:
    """
    Wrapper function for v6 FSM.
    """
    return _fsm.decide_action(sprite_data)


if __name__ == "__main__":
    """Test the expert FSM v6."""
    import argparse
    from robotron import RobotronEnv

    parser = argparse.ArgumentParser(description='Expert FSM v6 Player')
    parser.add_argument('--level', type=int, default=1, help='Start Level')
    parser.add_argument('--lives', type=int, default=5, help='Start Lives')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes')
    parser.add_argument('--fps', type=int, default=0, help='FPS (0=unlimited)')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--headless', action='store_true', help='Headless mode')

    args = parser.parse_args()

    print("=" * 80)
    print("EXPERT FSM v6 PLAYER (Goal-Oriented with Entity Tracking)")
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

            move_action, fire_action = expert_decide_v6(sprite_data)
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
