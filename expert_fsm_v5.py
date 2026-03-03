"""
Expert FSM Player v5 for Robotron 2084

COMPLETE REDESIGN: State-based FSM matching human thought process

Human player states:
1. IMMEDIATE THREAT: Enemy very close - kill it or run
2. COLLECT CIVILIAN: Civilian nearby and safe - go get it
3. HUNT ENEMY: Find nearest killable enemy and engage
4. SHOOT OPPORTUNISTICALLY: Always shoot if enemy is on a firing line

No jittering, no overthinking, just clear decisions.
"""

import math
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

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

# Game state
class State(Enum):
    IMMEDIATE_THREAT = 1
    COLLECT_CIVILIAN = 2
    HUNT_ENEMY = 3
    PATROL = 4


@dataclass
class Sprite:
    x: float
    y: float
    type: str
    distance: float = 0.0


class ExpertFSMv5:
    """
    State-based FSM matching human decision making.
    """

    def __init__(self):
        self.player_pos = None
        self.enemies = []
        self.family = []
        self.state = State.PATROL

        # Target persistence - lock onto a target until killed/collected
        self.current_target = None  # The sprite we're pursuing
        self.current_target_type = None  # 'enemy' or 'civilian'

        # Movement persistence - commit to a direction
        self.committed_move = None
        self.commit_frames_remaining = 0
        self.commit_duration = 8  # Hold direction for 8 frames

        # Trapped detection - detect when stuck at walls
        self.last_position = None
        self.stuck_frames = 0

    def parse_sprites(self, sprite_data: List[Tuple]) -> None:
        """Parse sprite data."""
        self.enemies = []
        self.family = []
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
            elif sprite_type != 'Hulk':  # Hulk is unkillable, don't include it
                # Include everything else (even Electrodes for avoidance)
                self.enemies.append(s)

        if self.player_pos:
            for sprites in [self.enemies, self.family]:
                for s in sprites:
                    dx = s.x - self.player_pos.x
                    dy = s.y - self.player_pos.y
                    s.distance = math.hypot(dx, dy)

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

    def get_safe_direction_to(self, target_x: float, target_y: float) -> int:
        """
        Get direction to target while avoiding center.
        If direct path goes through center, take an edge route.
        """
        if not self.player_pos:
            return STAY

        dx = target_x - self.player_pos.x
        dy = target_y - self.player_pos.y

        if abs(dx) < 5 and abs(dy) < 5:
            return STAY

        # Calculate direct angle
        # NOTE: Screen coordinates have y=0 at top, y increasing downward
        # atan2(dy, dx) where dy > 0 means target is BELOW player
        # We NEGATE dy so atan2 gives us correct angles: 0°=RIGHT, 90°=UP, 180°=LEFT, 270°=DOWN
        angle = math.atan2(-dy, dx)
        angle_deg = math.degrees(angle)
        if angle_deg < 0:
            angle_deg += 360

        # Check if we're currently in center - if so, get out first
        if self.is_in_center(self.player_pos.x, self.player_pos.y):
            # Move toward nearest edge
            center_x = BOARD_WIDTH / 2
            center_y = BOARD_HEIGHT / 2

            # Determine which edge is closest
            to_left = self.player_pos.x
            to_right = BOARD_WIDTH - self.player_pos.x
            to_top = self.player_pos.y
            to_bottom = BOARD_HEIGHT - self.player_pos.y

            min_dist = min(to_left, to_right, to_top, to_bottom)

            if min_dist == to_left:
                return LEFT
            elif min_dist == to_right:
                return RIGHT
            elif min_dist == to_top:
                return UP
            else:
                return DOWN

        # Check if direct path to target goes through center
        # Simple check: is midpoint in center?
        mid_x = (self.player_pos.x + target_x) / 2
        mid_y = (self.player_pos.y + target_y) / 2

        if self.is_in_center(mid_x, mid_y):
            # Direct path goes through center - take edge route
            # Decide whether to go around top/bottom or left/right
            center_x = BOARD_WIDTH / 2
            center_y = BOARD_HEIGHT / 2

            # If we're on left side, stay on left while moving toward target
            # If we're on right side, stay on right
            if self.player_pos.x < center_x:
                # Left side - prefer left edge
                if target_y > self.player_pos.y:
                    return DOWN_LEFT
                elif target_y < self.player_pos.y:
                    return UP_LEFT
                else:
                    return LEFT
            else:
                # Right side - prefer right edge
                if target_y > self.player_pos.y:
                    return DOWN_RIGHT
                elif target_y < self.player_pos.y:
                    return UP_RIGHT
                else:
                    return RIGHT

        # Direct path is safe - convert angle to 8-way direction
        # After negating dy: 0°=RIGHT, 45°=UP_RIGHT, 90°=UP, 135°=UP_LEFT,
        # 180°=LEFT, 225°=DOWN_LEFT, 270°=DOWN, 315°=DOWN_RIGHT
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
        toward = self.get_safe_direction_to(threat_x, threat_y)
        if toward == STAY:
            return STAY
        # Opposite direction
        away = ((toward - 1 + 4) % 8) + 1
        return away

    def get_alignment_move(self, target: Sprite) -> Optional[int]:
        """
        Calculate movement to align with target on a firing line.
        Returns direction to move to get target on firing line, or None if already well-aligned.

        Key: We need to get the offset to NEAR ZERO, not just within threshold.
        """
        if not self.player_pos:
            return None

        dx = target.x - self.player_pos.x
        dy = target.y - self.player_pos.y

        # Calculate which firing line we're closest to
        # Try each firing line and see which requires smallest movement
        best_move = None
        best_offset = float('inf')

        # RIGHT line (move vertically to get dy near 0)
        if dx > 0:
            vert_offset = abs(dy)
            if vert_offset < best_offset and vert_offset > 5:  # Stop when very close
                best_offset = vert_offset
                best_move = UP if dy < 0 else DOWN

        # LEFT line (move vertically to get dy near 0)
        if dx < 0:
            vert_offset = abs(dy)
            if vert_offset < best_offset and vert_offset > 5:
                best_offset = vert_offset
                best_move = UP if dy < 0 else DOWN

        # UP line (move horizontally to get dx near 0)
        if dy < 0:
            horiz_offset = abs(dx)
            if horiz_offset < best_offset and horiz_offset > 5:
                best_offset = horiz_offset
                best_move = LEFT if dx < 0 else RIGHT

        # DOWN line (move horizontally to get dx near 0)
        if dy > 0:
            horiz_offset = abs(dx)
            if horiz_offset < best_offset and horiz_offset > 5:
                best_offset = horiz_offset
                best_move = LEFT if dx < 0 else RIGHT

        # UP_RIGHT diagonal (move to get |dx| == |dy|)
        if dx > 0 and dy < 0:
            diag_offset = abs(abs(dx) - abs(dy))
            if diag_offset < best_offset and diag_offset > 5:
                best_offset = diag_offset
                # Move perpendicular to equalize dx and dy
                if abs(dx) > abs(dy):
                    best_move = UP  # Need more vertical (increase |dy|)
                else:
                    best_move = RIGHT  # Need more horizontal (increase |dx|)

        # DOWN_RIGHT diagonal
        if dx > 0 and dy > 0:
            diag_offset = abs(abs(dx) - abs(dy))
            if diag_offset < best_offset and diag_offset > 5:
                best_offset = diag_offset
                if abs(dx) > abs(dy):
                    best_move = DOWN
                else:
                    best_move = RIGHT

        # UP_LEFT diagonal
        if dx < 0 and dy < 0:
            diag_offset = abs(abs(dx) - abs(dy))
            if diag_offset < best_offset and diag_offset > 5:
                best_offset = diag_offset
                if abs(dx) > abs(dy):
                    best_move = UP
                else:
                    best_move = LEFT

        # DOWN_LEFT diagonal
        if dx < 0 and dy > 0:
            diag_offset = abs(abs(dx) - abs(dy))
            if diag_offset < best_offset and diag_offset > 5:
                best_offset = diag_offset
                if abs(dx) > abs(dy):
                    best_move = DOWN
                else:
                    best_move = LEFT

        return best_move

    def can_shoot_at(self, target: Sprite) -> Optional[int]:
        """
        Check if we can shoot at target (is it on a firing line?).
        Returns fire direction if yes, None if no.

        More lenient than v4 - 30px threshold.
        """
        if not self.player_pos:
            return None

        dx = target.x - self.player_pos.x
        dy = target.y - self.player_pos.y

        THRESHOLD = 30  # More lenient

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

    def find_opportunistic_shot(self) -> Optional[int]:
        """
        Always look for opportunistic shots at any enemy on a firing line.
        This is what human players do - shoot whenever possible.
        """
        # Check all enemies, prioritize closest
        for enemy in self.enemies:
            if enemy.distance < 300:  # Within reasonable range
                fire_dir = self.can_shoot_at(enemy)
                if fire_dir is not None:
                    return fire_dir
        return None

    def find_multi_enemy_alignment(self) -> Optional[Tuple[int, int]]:
        """
        When surrounded by multiple enemies, find a position where we can shoot
        multiple enemies at once. Returns (move_direction, num_enemies_aligned) or None.

        This is critical for survival when surrounded - moving to align multiple
        enemies lets us clear them faster.
        """
        if not self.player_pos or not self.enemies:
            return None

        # Only use this when surrounded (3+ close enemies)
        close_enemies = [e for e in self.enemies if e.distance < 150 and e.type != 'Electrode']
        if len(close_enemies) < 3:
            return None

        # Test each possible move direction
        best_move = None
        best_count = 0

        # Simulate moving in each direction by a small amount (15 pixels)
        test_moves = [
            (UP, 0, -15),
            (DOWN, 0, 15),
            (LEFT, -15, 0),
            (RIGHT, 15, 0),
            (UP_RIGHT, 15, -15),
            (UP_LEFT, -15, -15),
            (DOWN_RIGHT, 15, 15),
            (DOWN_LEFT, -15, 15),
        ]

        for move_dir, dx, dy in test_moves:
            # Simulate new position
            test_x = self.player_pos.x + dx
            test_y = self.player_pos.y + dy

            # Skip if this would put us too close to center
            if self.is_in_center(test_x, test_y):
                continue

            # Count how many enemies would be on firing lines from this position
            aligned_count = 0
            for enemy in close_enemies:
                # Calculate relative position from test position
                enemy_dx = enemy.x - test_x
                enemy_dy = enemy.y - test_y

                # Check if enemy would be on a firing line (30px threshold)
                THRESHOLD = 30

                # Horizontal lines (RIGHT or LEFT)
                if abs(enemy_dy) < THRESHOLD:
                    aligned_count += 1
                    continue

                # Vertical lines (UP or DOWN)
                if abs(enemy_dx) < THRESHOLD:
                    aligned_count += 1
                    continue

                # Diagonal lines
                if abs(abs(enemy_dx) - abs(enemy_dy)) < THRESHOLD:
                    aligned_count += 1
                    continue

            if aligned_count > best_count:
                best_count = aligned_count
                best_move = move_dir

        # Only return if we can align at least 2 enemies (worth the effort)
        if best_count >= 2:
            return (best_move, best_count)

        return None

    def find_target(self) -> Optional[Sprite]:
        """
        Choose a target and lock onto it.

        Priority:
        1. Immediate threats (enemies in danger zone <80px)
        2. Current target if still valid
        3. Civilians if only 1 enemy left and safe
        4. Dangerous enemies (by distance + type priority)
        5. Civilians if safe
        6. Any remaining enemies
        """

        # PRIORITY 1: Immediate threat - enemy very close
        danger_zone_enemies = [e for e in self.enemies if e.distance < 80]
        if danger_zone_enemies:
            # Immediate threat - must deal with it
            self.current_target = danger_zone_enemies[0]
            self.current_target_type = 'enemy'
            return self.current_target

        # PRIORITY 2: Keep current target if still valid
        if self.current_target:
            # Check if target still exists
            target_list = self.enemies if self.current_target_type == 'enemy' else self.family

            # Find target in current list
            for sprite in target_list:
                # Match by position (close enough) and type
                if (abs(sprite.x - self.current_target.x) < 50 and
                    abs(sprite.y - self.current_target.y) < 50 and
                    sprite.type == self.current_target.type):
                    # Target still exists - keep pursuing it
                    self.current_target = sprite  # Update position
                    return self.current_target

            # Target no longer exists - clear it
            self.current_target = None

        # PRIORITY 3: Civilians if few enemies left OR civilian very close
        if self.family:
            nearest_civilian = self.family[0]

            # Collect civilians if:
            # 1. Only 1-2 enemies left and civilian is closer than closest enemy
            # 2. OR civilian is VERY close (< 80px) and no immediate threats
            # 3. OR no enemies at all

            if len(self.enemies) == 0:
                # No enemies - always collect civilians
                # Break commitment when switching to civilian
                if self.current_target_type != 'civilian':
                    self.commit_frames_remaining = 0
                self.current_target = nearest_civilian
                self.current_target_type = 'civilian'
                return self.current_target

            if len(self.enemies) <= 2:
                # Few enemies - collect if civilian is closer
                closest_enemy = self.enemies[0]
                if nearest_civilian.distance < closest_enemy.distance:
                    # Break commitment when switching to civilian
                    if self.current_target_type != 'civilian':
                        self.commit_frames_remaining = 0
                    self.current_target = nearest_civilian
                    self.current_target_type = 'civilian'
                    return self.current_target

            # Civilian very close - grab it if no immediate threats
            if nearest_civilian.distance < 80:
                immediate_threats = [e for e in self.enemies if e.distance < 100]
                if len(immediate_threats) == 0:
                    # Break commitment when switching to civilian
                    if self.current_target_type != 'civilian':
                        self.commit_frames_remaining = 0
                    self.current_target = nearest_civilian
                    self.current_target_type = 'civilian'
                    return self.current_target

        # PRIORITY 4: Dangerous enemies (spawners first, then by distance)
        # NOTE: Hulks are IMMORTAL obstacles - avoid but don't target them
        if self.enemies:
            # Filter out obstacles and Hulks (Hulks are already filtered in parse_sprites)
            # Electrodes are obstacles, not enemies
            targetable_enemies = [e for e in self.enemies if e.type != 'Electrode']

            if targetable_enemies:
                # Score enemies by danger: closer = more dangerous, spawners = high priority
                def enemy_danger_score(enemy):
                    base_score = 100 - enemy.distance  # Closer = higher score

                    # Spawner bonus (highest priority - they create more enemies)
                    if enemy.type in ['Brain', 'Sphereoid', 'Quark']:
                        base_score += 150

                    # Shooter bonus (they're dangerous from range)
                    if enemy.type in ['Enforcer', 'Tank']:
                        base_score += 100

                    # Projectile bonus (instant death - very high priority)
                    if enemy.type in ['EnforcerBullet', 'TankShell', 'CruiseMissile']:
                        base_score += 200  # HIGHEST priority - they kill you!

                    # Grunt bonus (they chase you aggressively)
                    if enemy.type == 'Grunt':
                        base_score += 30

                    return base_score

                most_dangerous = max(targetable_enemies, key=enemy_danger_score)
                self.current_target = most_dangerous
                self.current_target_type = 'enemy'
                return self.current_target

        # PRIORITY 5: Civilians if no enemies
        if self.family:
            nearest_civilian = self.family[0]
            self.current_target = nearest_civilian
            self.current_target_type = 'civilian'
            return self.current_target

        # No targets
        self.current_target = None
        return None

    def decide_action(self, sprite_data: List[Tuple]) -> Tuple[int, int]:
        """
        Main decision function with target persistence.

        Logic:
        1. Choose/maintain a target
        2. Always shoot opportunistically
        3. Move toward target (avoiding center, handling threats)
        """
        self.parse_sprites(sprite_data)

        if not self.player_pos:
            return (STAY, STAY)

        move_action = STAY
        fire_action = STAY

        # STEP 1: Find/maintain target
        target = self.find_target()

        # Debug output (every 20 frames)
        if hasattr(self, '_debug_frame_counter'):
            self._debug_frame_counter += 1
        else:
            self._debug_frame_counter = 0

        debug_this_frame = self._debug_frame_counter % 20 == 0

        # STEP 2: ALWAYS shoot opportunistically at ANY enemy on firing line
        opportunistic_shot = self.find_opportunistic_shot()
        shoot_reason = "Not shooting"
        if opportunistic_shot:
            fire_action = opportunistic_shot
            # Find which enemy we're shooting at for debug
            for enemy in self.enemies:
                if enemy.distance < 300 and self.can_shoot_at(enemy) == opportunistic_shot:
                    shoot_reason = f"Shooting at {enemy.type} (dist={enemy.distance:.0f})"
                    break

        # STEP 3: Check for emergency override (enemy TOO close OR stuck at wall)
        override_commitment = False
        closest_enemy = None
        if self.enemies:
            closest = self.enemies[0]
            closest_enemy = closest
            if closest.distance < 60:  # Emergency!
                override_commitment = True

        # Check if stuck (position not changing)
        if self.last_position:
            pos_diff = math.hypot(self.player_pos.x - self.last_position.x,
                                 self.player_pos.y - self.last_position.y)
            if pos_diff < 1.0:  # Barely moved
                self.stuck_frames += 1
                if self.stuck_frames >= 3:  # Stuck for 3+ frames
                    override_commitment = True
            else:
                self.stuck_frames = 0
        else:
            self.stuck_frames = 0

        # Save position for next frame
        self.last_position = Sprite(self.player_pos.x, self.player_pos.y, 'Player')

        # STEP 4: Move toward target
        # Only use commitment if target hasn't changed significantly
        target_changed = False
        if target and self.current_target:
            # Check if we're still targeting same sprite (by position)
            pos_diff = math.hypot(target.x - self.current_target.x,
                                 target.y - self.current_target.y)
            if pos_diff > 100:  # Target moved a lot or it's a new target
                target_changed = True

        move_reason = "No valid move"

        if self.commit_frames_remaining > 0 and not override_commitment and not target_changed:
            # Still committed to previous move
            self.commit_frames_remaining -= 1
            move_action = self.committed_move
            move_reason = f"Committed move ({self.commit_frames_remaining} frames left)"
        else:
            # Calculate new move based on target
            if target:
                target_x, target_y = target.x, target.y

                # Check if target is an immediate threat
                # NOTE: Electrodes are obstacles, not threats - don't back up from them!
                is_obstacle = target.type in ['Electrode', 'Hulk']

                if self.current_target_type == 'enemy' and target.distance < 80 and not is_obstacle:
                    # Immediate threat - handle specially
                    # First check if we can align multiple enemies while backing up
                    multi_align = self.find_multi_enemy_alignment()

                    if multi_align:
                        # We can align multiple enemies - move to that position!
                        move_action, aligned_count = multi_align
                        move_reason = f"MULTI-ALIGN: {aligned_count} enemies (surrounded)"
                    else:
                        # Projectiles are instant death - back up aggressively!
                        is_projectile = target.type in ['EnforcerBullet', 'TankShell', 'CruiseMissile']

                        if is_projectile:
                            backup_distance = 80   # Bullets kill instantly - stay away!
                        else:
                            backup_distance = 70   # Normal enemies (including Grunts)

                        if target.distance < backup_distance and not self.can_shoot_at(target):
                            # Too close and can't shoot - RUN
                            move_action = self.get_direction_away(target_x, target_y)
                            move_reason = f"FLEEING from {target.type} (dist={target.distance:.0f}, can't shoot)"
                        elif target.distance < backup_distance:
                            # Too close - back up while shooting
                            move_action = self.get_direction_away(target_x, target_y)
                            move_reason = f"BACKING UP from {target.type} (dist={target.distance:.0f}, too close)"
                        else:
                            # Close but manageable - circle to keep on firing line
                            angle_to_enemy = math.atan2(target_y - self.player_pos.y,
                                                       target_x - self.player_pos.x)
                            circle_angle = angle_to_enemy + math.pi / 2
                            circle_x = self.player_pos.x + 70 * math.cos(circle_angle)
                            circle_y = self.player_pos.y + 70 * math.sin(circle_angle)
                            move_action = self.get_safe_direction_to(circle_x, circle_y)
                            move_reason = f"CIRCLING {target.type} (dist={target.distance:.0f})"
                else:
                    # Normal pursuit
                    # Strategy: If target is enemy and within range, try to align for shooting
                    # Otherwise, just move toward target
                    if self.current_target_type == 'enemy' and target.distance < 250:
                        # Close enough to care about alignment
                        alignment_move = self.get_alignment_move(target)

                        if alignment_move:
                            # Not aligned - move to align
                            move_action = alignment_move
                            move_reason = f"ALIGNING to shoot {target.type} (dist={target.distance:.0f})"
                        else:
                            # Already aligned - maintain distance or approach
                            if target.distance > 150:
                                # Too far - move closer
                                move_action = self.get_safe_direction_to(target_x, target_y)
                                move_reason = f"Moving closer to {target.type} (dist={target.distance:.0f})"
                            else:
                                # Good distance - circle to maintain firing line
                                angle_to_enemy = math.atan2(target_y - self.player_pos.y,
                                                           target_x - self.player_pos.x)
                                circle_angle = angle_to_enemy + math.pi / 2
                                circle_x = self.player_pos.x + 100 * math.cos(circle_angle)
                                circle_y = self.player_pos.y + 100 * math.sin(circle_angle)
                                move_action = self.get_safe_direction_to(circle_x, circle_y)
                                move_reason = f"STRAFING {target.type} (dist={target.distance:.0f})"
                    else:
                        # Far away or civilian - just move toward it
                        # For civilians, go direct (no center avoidance)
                        # For enemies, avoid center
                        if self.current_target_type == 'civilian':
                            # Direct path to civilian - calculate angle
                            dx = target_x - self.player_pos.x
                            dy = target_y - self.player_pos.y
                            # NEGATE dy for screen coordinates (y=0 at top)
                            angle = math.atan2(-dy, dx)
                            angle_deg = math.degrees(angle)
                            if angle_deg < 0:
                                angle_deg += 360

                            # Convert to 8-way direction
                            if angle_deg < 22.5 or angle_deg >= 337.5:
                                move_action = RIGHT
                            elif angle_deg < 67.5:
                                move_action = UP_RIGHT
                            elif angle_deg < 112.5:
                                move_action = UP
                            elif angle_deg < 157.5:
                                move_action = UP_LEFT
                            elif angle_deg < 202.5:
                                move_action = LEFT
                            elif angle_deg < 247.5:
                                move_action = DOWN_LEFT
                            elif angle_deg < 292.5:
                                move_action = DOWN
                            else:
                                move_action = DOWN_RIGHT
                            move_reason = f"Moving DIRECT toward civilian {target.type} (dist={target.distance:.0f})"
                        else:
                            # Enemy - use safe path (avoid center)
                            move_action = self.get_safe_direction_to(target_x, target_y)
                            move_reason = f"Moving toward enemy {target.type} (dist={target.distance:.0f})"

            # Never stop moving - if we calculated STAY, patrol instead
            if move_action == STAY:
                # Default patrol: circle the edges
                # Move along nearest edge
                if self.is_in_center(self.player_pos.x, self.player_pos.y):
                    # Get out of center first
                    center_x = BOARD_WIDTH / 2
                    center_y = BOARD_HEIGHT / 2
                    if self.player_pos.x < center_x:
                        move_action = LEFT
                        move_reason = "ESCAPING center (moving left)"
                    else:
                        move_action = RIGHT
                        move_reason = "ESCAPING center (moving right)"
                else:
                    # Circle along edges clockwise
                    x, y = self.player_pos.x, self.player_pos.y
                    # Determine which edge we're on
                    to_edges = [
                        ('left', x),
                        ('right', BOARD_WIDTH - x),
                        ('top', y),
                        ('bottom', BOARD_HEIGHT - y)
                    ]
                    nearest_edge = min(to_edges, key=lambda e: e[1])[0]

                    # Move along that edge
                    if nearest_edge == 'left':
                        move_action = DOWN_LEFT if y < BOARD_HEIGHT / 2 else UP_LEFT
                    elif nearest_edge == 'right':
                        move_action = UP_RIGHT if y < BOARD_HEIGHT / 2 else DOWN_RIGHT
                    elif nearest_edge == 'top':
                        move_action = UP_RIGHT if x < BOARD_WIDTH / 2 else UP_LEFT
                    else:  # bottom
                        move_action = DOWN_LEFT if x < BOARD_WIDTH / 2 else DOWN_RIGHT

                    move_reason = f"PATROLLING ({nearest_edge} edge)"

            # Commit to this new move
            self.committed_move = move_action
            self.commit_frames_remaining = self.commit_duration

        # Wall avoidance - ONLY redirect if moving INTO wall
        # Skip wall redirect for civilians (let them be collected near walls)
        # Skip wall redirect if NOT moving into wall (prevents getting stuck at walls)
        if self.player_pos and self.current_target_type != 'civilian':
            x, y = self.player_pos.x, self.player_pos.y
            margin = 40

            # ONLY redirect if we're AT wall AND moving INTO it
            # Don't redirect if we're moving AWAY from wall or along it
            if x < margin and move_action in [LEFT, UP_LEFT, DOWN_LEFT]:
                # At left wall AND moving into it - redirect along wall
                if target and target.x > self.player_pos.x:
                    # Target is to the right - move toward it (away from wall)
                    # Don't redirect! Let the FSM move right toward target
                    pass
                else:
                    # Target is on same side as wall - move along wall
                    if target and target.y < self.player_pos.y:
                        move_action = UP
                    else:
                        move_action = DOWN
                    move_reason = f"WALL REDIRECT (left edge -> {['UP', 'DOWN'][move_action == DOWN]})"
                    self.committed_move = move_action
                    self.commit_frames_remaining = self.commit_duration

            elif x > BOARD_WIDTH - margin and move_action in [RIGHT, UP_RIGHT, DOWN_RIGHT]:
                # At right wall AND moving into it
                if target and target.x < self.player_pos.x:
                    # Target is to the left - move toward it (away from wall)
                    # Don't redirect! Let the FSM move left toward target
                    pass
                else:
                    # Target is on same side as wall - move along wall
                    if target and target.y < self.player_pos.y:
                        move_action = UP
                    else:
                        move_action = DOWN
                    move_reason = f"WALL REDIRECT (right edge -> {['UP', 'DOWN'][move_action == DOWN]})"
                    self.committed_move = move_action
                    self.commit_frames_remaining = self.commit_duration

            elif y < margin and move_action in [UP, UP_LEFT, UP_RIGHT]:
                # At top wall AND moving into it
                if target and target.y > self.player_pos.y:
                    # Target is below - move toward it (away from wall)
                    pass
                else:
                    # Target is on same side as wall - move along wall
                    if target and target.x < self.player_pos.x:
                        move_action = LEFT
                    else:
                        move_action = RIGHT
                    move_reason = f"WALL REDIRECT (top edge -> {['LEFT', 'RIGHT'][move_action == RIGHT]})"
                    self.committed_move = move_action
                    self.commit_frames_remaining = self.commit_duration

            elif y > BOARD_HEIGHT - margin and move_action in [DOWN, DOWN_LEFT, DOWN_RIGHT]:
                # At bottom wall AND moving into it
                if target and target.y < self.player_pos.y:
                    # Target is above - move toward it (away from wall)
                    pass
                else:
                    # Target is on same side as wall - move along wall
                    if target and target.x < self.player_pos.x:
                        move_action = LEFT
                    else:
                        move_action = RIGHT
                    move_reason = f"WALL REDIRECT (bottom edge -> {['LEFT', 'RIGHT'][move_action == RIGHT]})"
                    self.committed_move = move_action
                    self.commit_frames_remaining = self.commit_duration

        # Print debug info
        if debug_this_frame:
            # Player position
            player_info = f"Player at ({self.player_pos.x:.0f}, {self.player_pos.y:.0f})"

            # Target info with coordinates
            target_info = "No target"
            if target:
                target_info = f"{self.current_target_type.upper()}: {target.type} at ({target.x:.0f}, {target.y:.0f}) dist={target.distance:.0f}"

            # Closest enemy info
            closest_info = "No enemies"
            if closest_enemy:
                closest_info = f"Closest: {closest_enemy.type} at ({closest_enemy.x:.0f}, {closest_enemy.y:.0f}) dist={closest_enemy.distance:.0f}"

            # All enemies list (first 10 for readability)
            enemies_list = "No enemies"
            if self.enemies:
                enemy_strs = [f"{e.type}({e.distance:.0f})" for e in self.enemies[:10]]
                enemies_list = f"Enemies: {', '.join(enemy_strs)}"
                if len(self.enemies) > 10:
                    enemies_list += f" ... +{len(self.enemies) - 10} more"

            # Civilians list
            civilians_list = ""
            if self.family:
                civilian_strs = [f"{c.type}({c.distance:.0f})" for c in self.family]
                civilians_list = f"Civilians: {', '.join(civilian_strs)}"

            # Direction names
            move_dir_name = DIRECTION_NAMES.get(move_action, f"UNKNOWN({move_action})")
            fire_dir_name = DIRECTION_NAMES.get(fire_action, f"UNKNOWN({fire_action})")

            print(f"[FSM] {player_info}")
            print(f"[FSM] {target_info}")
            print(f"[FSM] {closest_info}")
            print(f"[FSM] {enemies_list}")
            if civilians_list:
                print(f"[FSM] {civilians_list}")
            print(f"[FSM] Move: {move_dir_name} - {move_reason}")
            print(f"[FSM] Fire: {fire_dir_name} - {shoot_reason}")
            print()

        return (move_action, fire_action)


# Global FSM instance
_fsm = ExpertFSMv5()


def expert_decide_v5(sprite_data: List[Tuple]) -> Tuple[int, int]:
    """
    Wrapper function for v5 FSM.
    """
    return _fsm.decide_action(sprite_data)


if __name__ == "__main__":
    """Test the expert FSM v5."""
    import argparse
    from robotron import RobotronEnv

    parser = argparse.ArgumentParser(description='Expert FSM v5 Player')
    parser.add_argument('--level', type=int, default=1, help='Start Level')
    parser.add_argument('--lives', type=int, default=5, help='Start Lives')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes')
    parser.add_argument('--fps', type=int, default=0, help='FPS (0=unlimited)')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--headless', action='store_true', help='Headless mode')

    args = parser.parse_args()

    print("=" * 80)
    print("EXPERT FSM v5 PLAYER (State-Based Human Logic)")
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

            move_action, fire_action = expert_decide_v5(sprite_data)
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
