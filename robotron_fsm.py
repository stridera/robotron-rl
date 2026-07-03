"""
HOW TO RUN THIS
cd robotron2084gym/
python3.11 -mvenv .venv
source .venv/bin/activate
pip install -r requirements.txt
python robotron_fsm.py
OR
python robotron_fsm.py > out.txt

"""

"""
Robotron Finite State Machine

Relies on simple player to provide all on-screen elements as input, and sends
joystick directions as output. Game framework will act on those outputs, update
the screen, and send it back to the FSM.

This is the simplest way to run the game.  It will run the game at 30 fps and will provide random
actions to the game.  It will print out the score, level, lives, reward, and if the player is dead.
It will also print out the family remaining and the objects on the board.
Objects can be:
    Player
    Mommy
    Daddy
    Mikey
    Tank
    TankShell
    Grunt
    Electrode
    Hulk
    Bullet
    CruiseMissile
    Brain
    Enforcer
    EnforcerBullet
    Sphereoid
    Quark - these become tanks
    Prog - originally missing from this list
"""


"""
from robotron import RobotronEnv

NEXT

Check scores, especially for each family
Levels 59 and 89
Add AWAY_ANGLE for EnforcerBullet
Improve logic when surrounded, and to prevent too many projectiles and seeker enemies
Fix borders and movement adjust
Player lives rolls over to 0 after 255, so always walk into an enemy if lives are above a limit

"""

# CONSTANTS

import argparse
import math
import time
from os import path
from robotron import RobotronEnv
INVALID = -1

# Grid and Distance
MAX_TOP = 492
MAX_BOTTOM = 0
MAX_LEFT = 0
MAX_RIGHT = 665

BORDER_ADJUST = 20

ADJ_TOP = MAX_TOP - 2
ADJ_BOTTOM = MAX_BOTTOM + BORDER_ADJUST + 9
ADJ_LEFT = MAX_LEFT + 2
ADJ_RIGHT = MAX_RIGHT - BORDER_ADJUST

Y_AXIS_INVERSION = 492

""" Are these direction constants available somewhere else?"""
# Joystick directions
STAY = 0
UP = 1
UP_RIGHT = 2
RIGHT = 3
DOWN_RIGHT = 4
DOWN = 5
DOWN_LEFT = 6
LEFT = 7
UP_LEFT = 8

# Move directives
TOWARD = 1
AWAY = 0

# Object categories and names
PLAYER = 'Player'
OBSTACLE = 'Electrode'
HULK = 'Hulk'

FAMILY = {'Mommy', 'Daddy', 'Mikey'}
ENEMIES = {'Grunt', 'Brain'}
CHASE_ENEMIES = {'Sphereoid', 'Quark'}
PRIORITY_ENEMIES = {'Enforcer', 'Tank'}
PROJECTILES = {'TankShell', 'CruiseMissile', 'EnforcerBullet', 'Prog'}
BULLET = 'Bullet'  # Bullet is from the player, and not an enemy projectile

FAMILY_TYPE = 'Family'
ENEMY_TYPE = 'Enemy'
PRIORITY_ENEMY_TYPE = 'PriorityEnemy'
CHASE_ENEMY_TYPE = 'ChaseEnemy'
PROJECTILE_TYPE = 'Projectile'
QUARK = 'Quark'
QUARK_TYPE = 'QuarkThreat'

# Distance values for tuning behavior
ADJACENT = 40
ADJACENT_HULK = 50
# STRUCTURAL Quark fix (2026-06-28). Death forensics (536 deaths): Quark = 32% =
# #1 killer, concentrated at wave 7 (108/172); player dies at median 7.5px (direct
# contact, quark does NOT die with it -> player moved INTO it), not cornered, ~always
# >=2 quarks present. Root cause: Quarks are CHASE_ENEMIES so the FSM moves TOWARD them
# to ~50px, but quarks TELEPORT (>40px -> contact within one frameskip-4 step) so the
# 40px ADJACENT flee radius reacts too late, AND on quark waves an Enforcer/bullet wins
# the adjacent-priority contest so the player flees the bullet INTO the quark.
# Fix: treat a close Quark as its own top-priority flee threat (just below PROJECTILE),
# with a LARGE reaction radius so the player kites it (fire while backing away) instead
# of chasing. A prior CLOSE_MOVE_QUARK standoff-threshold tweak was a WASH because the
# chase path never actively flees. ADJACENT_QUARK<=0 reverts to the old chase behavior
# (for A/B). Tunable via EVOLVE_PARAMS / env.
import os as _os
ADJACENT_QUARK = float(_os.environ.get('FSM_ADJACENT_QUARK', '75'))

# STRUCTURAL fix #2: THREAT-FIELD-AWARE flee (opt-in FSM_THREAT_FIELD=1). Death forensics
# after the quark fix: Hulk became #1 killer (18%->26%) and 46% of hulk deaths had a Quark
# also nearby -> the away-from-ONE-threat flee trades quark deaths for hulk deaths (flee the
# quark, back INTO an invincible hulk). Fix: when fleeing, move away from the WEIGHTED CENTROID
# of ALL nearby threats (hulks weighted heaviest since unshootable) -> toward genuinely open
# space. Implemented by summing each threat's toward-vector*weight/dist and passing it through
# the existing getMoveStick(...,AWAY,...) (the AWAY flip + wall handling are reused as-is).
THREAT_FIELD = _os.environ.get('FSM_THREAT_FIELD', '0') == '1'
FIELD_RADIUS = float(_os.environ.get('FSM_FIELD_RADIUS', '140'))
FIELD_MAX_THREATS = int(_os.environ.get('FSM_FIELD_MAX_THREATS', '3'))  # pincer gate (see threatFieldMove)

# STRUCTURAL fix #3 (opt-in FSM_HULK_DEFLECT=1): surgical alternative to the (dead-end) threat
# field. Keep the decisive nearest-threat flee; ONLY when that direction points straight at a
# nearby invincible HULK, nudge it +/-45deg to the nearest clear heading. Targets the flee-into
# -hulk trade (46% of post-quark-fix hulk deaths had a quark nearby) without disturbing swarms.
HULK_DEFLECT = _os.environ.get('FSM_HULK_DEFLECT', '1') == '1'  # default ON: validated +1.26 wave (N=100 paired)

# STRUCTURAL fix #4 (opt-in FSM_HULK_NOFIRE=1): never waste shots on invincible Hulks. Wave-12
# forensics: 69% of w12 deaths are Quark kills with a Hulk present in 98% — when a Hulk is the
# adjacent threat the FSM fires AT it (getFireStick(adjacent)) = a wasted shot, so the killable
# Quark survives and contact-kills. Fix: when the nearest threat is a Hulk, fire at the nearest
# KILLABLE threat instead (twin-stick: flee the hulk, shoot the quark).
HULK_NOFIRE = _os.environ.get('FSM_HULK_NOFIRE', '0') == '1'
HULK_DEFLECT_R = float(_os.environ.get('FSM_HULK_DEFLECT_R', '60'))
# STRUCTURAL fix #5 (opt-in FSM_EDGE_DEFLECT=1): combteacher death-forensics (2026-06-30) found
# EnfBullet/Spark = #2 killer (20%, 73% of those deaths at board edges/corners) — the FSM flees
# INTO an edge and a fast spark traps it in the 20-70px band BEFORE the existing at-the-wall
# handling (margin 2-20px) fires. Surgical earlier deflection: when a flee heading points into a
# near edge, slide +/-45deg ALONG it. Same shape as the winning hulkDeflect.
EDGE_DEFLECT = _os.environ.get('FSM_EDGE_DEFLECT', '0') == '1'
EDGE_DEFLECT_M = float(_os.environ.get('FSM_EDGE_DEFLECT_M', '55'))

# STRUCTURAL fix #6 (opt-in FSM_HULK_PUSH=1): hulks are indestructible but a laser hit
# SHOVES them ~6-12px in the shot direction (asm:471 HULK_BULLET_COLLISION_HANDLER =
# away from the player). Death taxonomy (2026-07-01, correct labels): hulk-pin = 30% of
# deaths — cornered (STAY) with a hulk closing and no dodge. Fix: when cornered with a
# hulk in push range, redirect fire AT it to shove it open. (Also reframes the old
# HULK_NOFIRE regression: not firing at hulks removed this push.)
HULK_PUSH = _os.environ.get('FSM_HULK_PUSH', '0') == '1'
HULK_PUSH_R = float(_os.environ.get('FSM_HULK_PUSH_R', '40'))

# $F1 shell-budget exploit (2026-07-01, ENEMY_MODEL.md §9, robomame.asm:7830-7836): the
# per-wave shell counter only decrements when a shell is SHOT or hits the player — shells
# that expire naturally are never decremented, so after ~20 fired shells the game thinks 20
# are always live and TANKS FALL SILENT for the rest of the wave. Dodging shells (never
# shooting them) permanently exhausts the wave's shell budget; shooting one frees capacity
# for another aimed shot. Opt-in: redirect fire off TankShells to the nearest killable
# threat instead (all FLEE behavior toward shells is unchanged).
NO_SHOOT_SHELLS = _os.environ.get('FSM_NO_SHOOT_SHELLS', '0') == '1'

# Rescue-seeking (2026-07-01): family collection was purely opportunistic — pursuit only
# inside CLOSE_MOVE_CIVILIAN (~92px of a 665px board) and the idle fallback preferred
# chasing enemies over rescuing. But family is the LIFE ENGINE: rescues escalate
# 1000->2000->...->5000 within a wave, brain waves carry 15-25 family (65k+ = 2.6 lives on
# wave 5 alone), extra man every 25k, and the measured sustainability deficit is only
# ~0.1 lives/wave. RESCUE_SEEK: when idle (no threat set the move), head for the nearest
# family member at ANY distance, before the chase-enemy fallback.
# BRAIN_RESCUE_MULT: while a Brain is on screen, multiply the civilian pursuit radius —
# brains NEVER chase the player while any human lives (asm:1BFF-1C08) and all initially
# cluster on Mikey via the $B354 empty-list bug (asm:2394-2405), so brain waves are
# structurally safe to sweep aggressively.
RESCUE_SEEK = _os.environ.get('FSM_RESCUE_SEEK', '0') == '1'
BRAIN_RESCUE_MULT = float(_os.environ.get('FSM_BRAIN_RESCUE_MULT', '1.0'))

# Endgame hunt (2026-07-03): the idle fallback only ever pursued CHASE_ENEMIES and
# family — a last remaining Grunt/Brain/Tank beyond the fire radius was NEVER
# approached, so the FSM stood still (or got herded to a wall by hulk-flee) waiting
# for it to wander into range. That wall-hiding posture is exactly the pinned
# positioning that dominates the death taxonomy, and the dead time lets grunts
# speed up (they accelerate within a wave). HUNT_KILLABLE: when idle, advance on
# the nearest killable enemy (priority or regular) to HUNT_STANDOFF and keep firing
# at it even beyond the CLOSE_FIRE radii (shots are free). Settable from the
# evolved-params json (floats; truthy enables) for clean A/B via eval_protocol.
HUNT_KILLABLE = float(_os.environ.get('FSM_HUNT', '0'))
HUNT_STANDOFF = float(_os.environ.get('FSM_HUNT_STANDOFF', '90'))

# Positioning buffer: scale the flee-INITIATION radii so the FSM keeps more distance
# from threats (targets swarm/pin positioning failures = 81% of deaths). 1.0 = evolved
# default. Applied lazily on first chooseOutputs() so it composes with EVOLVE_PARAMS
# (which the consumer setattr()s after import).
BUFFER_SCALE = float(_os.environ.get('FSM_BUFFER_SCALE', '1.0'))
_buffer_applied = False
_BUFFER_KEYS = ('ADJACENT', 'ADJACENT_HULK', 'ADJACENT_QUARK', 'CLOSE_MOVE_PROJECTILE',
                'CLOSE_MOVE_PRIORITY_ENEMY', 'CLOSE_MOVE_CHASE_ENEMY',
                'CLOSE_MOVE_ENEMY', 'CLOSE_MOVE_HULK')
def _apply_buffer_scale():
    global _buffer_applied
    if _buffer_applied:
        return
    _buffer_applied = True
    if BUFFER_SCALE == 1.0:
        return
    g = globals()
    for k in _BUFFER_KEYS:
        v = g.get(k)
        if isinstance(v, (int, float)):
            g[k] = v * BUFFER_SCALE
# Direction (1..8) -> unit vector in the flipped frame (Y up): N,NE,E,SE,S,SW,W,NW.
_DIRVEC = {1: (0.0, 1.0), 2: (0.707, 0.707), 3: (1.0, 0.0), 4: (0.707, -0.707),
           5: (0.0, -1.0), 6: (-0.707, -0.707), 7: (-1.0, 0.0), 8: (-0.707, 0.707)}

"""
CLOSE = 75 # regular enemy
"""
CLOSE_FIRE_PRIORITY_ENEMY = 150
CLOSE_MOVE_PRIORITY_ENEMY = 50
CLOSE_FIRE_PROJECTILE = 150
CLOSE_MOVE_PROJECTILE = 100
CLOSE_FIRE_CHASE_ENEMY = 175
CLOSE_MOVE_CHASE_ENEMY = 50
# Quark-specific flee distance. Death forensics (2026-06-27): Quark = 32% of FSM
# deaths (#1 killer) — Quarks are in CHASE_ENEMIES, so the FSM chases them to ~50px
# to shoot and their erratic bounce kills on contact. None => behave as
# CLOSE_MOVE_CHASE_ENEMY (no change, safe default); set larger (e.g. 100) to keep
# distance from Quarks. A/B via the evolved-params json.
CLOSE_MOVE_QUARK = None
CLOSE_FIRE_ENEMY = 75
CLOSE_MOVE_ENEMY = 50
CLOSE_FIRE_HULK = 50
CLOSE_MOVE_HULK = 50
CLOSE_FIRE_OBSTACLE = 40
CLOSE_MOVE_OBSTACLE = 40
CLOSE_MOVE_COUNT_LIMIT = 5
CLOSE_FIRE_COUNT_LIMIT = 3
CLOSE_MOVE_CIVILIAN = 75

# Object data array element positions
X_POS = 0
Y_POS = 1
TYPE = 2

DISTANCE = 0
X_DISTANCE = 1
Y_DISTANCE = 2

# Print statement levels
DEBUG_LEVEL = 0
DEBUG_OFF = 0
DEBUG_LOW = 1
DEBUG_MED = 2
DEBUG_HIGH = 3

"""
Return an array of calculated total distance and each direction: [DISTANCE, X_DISTANCE, Y_DISTANCE]
X and Y are necessary to determine direction relative to the player
"""


def getDistance(playerLocation, objX, objY):
    pX = playerLocation[X_POS]
    pY = playerLocation[Y_POS]

    if DEBUG_LEVEL >= DEBUG_HIGH:
        print(f"getDistance inputs player x {pX} y {pY} target x {objX} y {objY}")

    xDistance = objX - playerLocation[X_POS]
    yDistance = objY - playerLocation[Y_POS]

    if DEBUG_LEVEL >= DEBUG_HIGH:
        print(f"getDistance differences px {pX} ox {objX} dx {xDistance} py {pY} oy {objY} dy {yDistance}")

    """ 0 for either direction doesn't need math """
    if xDistance == 0:
        return [abs(yDistance), xDistance, yDistance]
    if yDistance == 0:
        return [abs(xDistance), xDistance, yDistance]

    """ skip the math for the diagonal adjacent case """
    if abs(xDistance) == 1 and abs(yDistance) == 1:
        return [1, xDistance, yDistance]

    """ Pythagorean """
    absDistance = int(math.sqrt(pow(abs(xDistance), 2) + pow(abs(yDistance), 2)))
    return [absDistance, xDistance, yDistance]


"""
targetDistanceData - X_DISTANCE, Y_DISTANCE, TYPE
TYPE doesn't matter at this point, though
Use the X and Y distance values of the target to calculate the direction to fire.
The Y Axis is inverted. It is 0 at the top border, and INCREASES toward the bottom.
Return a value from the Joystick Directions constants.
"""


def getFireStick(targetDistanceData):
    xDistance = targetDistanceData[X_DISTANCE]
    yDistance = targetDistanceData[Y_DISTANCE]

    if DEBUG_LEVEL >= DEBUG_MED:
        print(f"getFireStick {targetDistanceData} x {xDistance} y {yDistance}")

    # skip the math if one direction is 0
    if xDistance == 0:
        if yDistance >= 0:
            return UP
        else:
            return DOWN
    if yDistance == 0:
        if xDistance >= 0:
            return RIGHT
        else:
            return LEFT

    """ y first, x second """
    radians = math.atan2(yDistance, xDistance)
    if DEBUG_LEVEL >= DEBUG_HIGH:
        print(f"getFireStick radians {radians}")

    # 45 degrees per segment, and negatives are possible
    if radians == 0 or (radians > 0 and radians < 0.392699081):
        # 0 to 22.5 = RIGHT
        return RIGHT
    elif (radians > 0 and radians < 1.178097245):
        # 22.5 to 67.5 = UP_RIGHT
        return UP_RIGHT
    elif (radians > 0 and radians < 1.963495408):
        # 67.5 to 112.5 = UP
        return UP
    elif (radians > 0 and radians < 2.748893572):
        # 112.5 to 157.5 = UP_LEFT
        return UP_LEFT
    elif (radians > 0 and radians < 3.534291735):
        # 157.5 to 202.5 = LEFT
        return LEFT
    elif (radians > 0 and radians < 4.3196898899):
        # 202.5 to 247.5 = DOWN_LEFT
        return DOWN_LEFT
    elif (radians > 0 and radians < 5.105088062):
        # 247.5 to 292.5 = DOWN
        return DOWN
    elif (radians > 0 and radians < 5.890486225):
        # 292.5 to 337.5 = DOWN_RIGHT
        return DOWN_RIGHT
    elif (radians > 0 and radians <= 6.283185308):
        # 337.5 to 360 = RIGHT
        return RIGHT
    elif (radians < 0 and radians > -0.392699081):
        # 0 to -22.5 = RIGHT
        return RIGHT
    elif (radians < 0 and radians > -1.178097245):
        # -22.5 to -67.5 = DOWN_RIGHT
        return DOWN_RIGHT
    elif (radians < 0 and radians > -1.963495408):
        # -66.75 to -112.5 = DOWN
        return DOWN
    elif (radians < 0 and radians > -2.748893572):
        # -112.5 to -157.5 = DOWN_LEFT
        return DOWN_LEFT
    elif (radians < 0 and radians > -3.53429173):
        # -157.5 to -202.5 = LEFT
        return LEFT
    elif (radians < 0 and radians > -4.3196898899):
        # -202.4 to -247.5 = UP_LEFT
        return UP_LEFT
    elif (radians < 0 and radians > -5.105088062):
        # -247.5 to -292.5 = UP
        return UP
    elif (radians < 0 and radians > -5.890486225):
        # -292.5 to -337.5 = UP_RIGHT
        return UP_RIGHT
    elif (radians < 0 and radians >= -6.283185308):
        # -337.5 to -360 = RIGHT
        return RIGHT
    else:
        if DEBUG_LEVEL >= DEBUG_LOW:
            print(f"getFireStick did not match {radians}")
        return RIGHT


"""
targetDistanceData - X_DISTANCE, Y_DISTANCE, TYPE
TYPE doesn't matter at this point, though
moveDirective - either TOWARD or AWAY
playerLocation - X and Y coordinates
Use the X and Y distance values of the target plus the move directive to figure out which direction to move
Return a value from the Joystick Directions constants.
"""


def getMoveStick(targetDistanceData, moveDirective, playerLocation):
    """ flipping the signs should reverse the direction relative to the player """
    """
    if moveDirective == AWAY:
    	# can this be done without an temporary variable? 
    	xDistance = targetDistanceData[X_DISTANCE]
    	yDistance = targetDistanceData[Y_DISTANCE]
    	targetDistanceData[X_DISTANCE] = -xDistance
    	targetDistanceData[Y_DISTANCE] = -yDistance
    """

    """ it's the same calculation """
    moveDirection = getFireStick(targetDistanceData)

    # If the solution above to flip the signs of the targetDistanceData is better, remove this section. Can't have both.
    if moveDirective == AWAY:
        if DEBUG_LEVEL >= DEBUG_HIGH:
            print(f"before Away {moveDirection}")
        moveDirection = (moveDirection + 4) % 8
        if moveDirection == 0:
            moveDirection = 8
        if DEBUG_LEVEL >= DEBUG_HIGH:
            print(f"after Away  {moveDirection}")

    """ Wall handling """
    playerXPos = playerLocation[X_POS]
    playerYPos = playerLocation[Y_POS]

    if playerXPos <= ADJ_LEFT:
        if DEBUG_LEVEL >= DEBUG_MED:
            print(f"LEFT WALL {ADJ_LEFT} playerX {playerXPos} playerY {playerYPos}")
        if playerYPos >= ADJ_TOP:
            if DEBUG_LEVEL >= DEBUG_MED:
                print(f"TOP LEFT CORNER LEFT {ADJ_LEFT} playerX {playerXPos} TOP {ADJ_TOP} playerY {playerYPos}")
            if moveDirection == LEFT or moveDirection == UP_LEFT:
                moveDirection = DOWN
            elif moveDirection == UP or moveDirection == UP_RIGHT:
                moveDirection = RIGHT
        elif playerYPos <= ADJ_BOTTOM:
            if DEBUG_LEVEL >= DEBUG_MED:
                print(
                    f"BOTTOM LEFT CORNER LEFT {ADJ_LEFT} playerX {playerXPos} BOTTOM {ADJ_BOTTOM} playerY {playerYPos}")
            if moveDirection == LEFT or moveDirection == DOWN_LEFT:
                moveDirection = UP
            elif moveDirection == DOWN or moveDirection == DOWN_RIGHT:
                moveDirection = RIGHT
        elif moveDirection == UP_LEFT:
            moveDirection = UP
        elif moveDirection == DOWN_LEFT:
            moveDirection = DOWN
        elif moveDirection == LEFT:
            if playerYPos < MAX_TOP / 2:
                moveDirection = UP
            else:
                moveDirection = DOWN

    if playerXPos >= ADJ_RIGHT:
        if DEBUG_LEVEL >= DEBUG_MED:
            print(f"RIGHT WALL {ADJ_RIGHT} playerX {playerXPos} playerY {playerYPos}")
        if playerYPos >= ADJ_TOP:
            if DEBUG_LEVEL >= DEBUG_MED:
                print(f"TOP RIGHT CORNER RIGHT {ADJ_RIGHT} playerX {playerXPos} TOP {ADJ_TOP} playerY {playerYPos}")
            if moveDirection == RIGHT or moveDirection == UP_RIGHT:
                moveDirection = DOWN
            elif moveDirection == UP or moveDirection == UP_LEFT:
                moveDirection = LEFT
        elif playerYPos <= ADJ_BOTTOM:
            if DEBUG_LEVEL >= DEBUG_MED:
                print(
                    f"BOTTOM RIGHT CORNER RIGHT {ADJ_RIGHT} playerX {playerXPos} BOTTOM {ADJ_BOTTOM} playerY {playerYPos}")
            if moveDirection == RIGHT or moveDirection == DOWN_RIGHT:
                moveDirection = UP
            elif moveDirection == DOWN or moveDirection == DOWN_LEFT:
                moveDirection = LEFT
        elif moveDirection == UP_RIGHT:
            moveDirection = UP
        elif moveDirection == DOWN_RIGHT:
            moveDirection = DOWN
        elif moveDirection == RIGHT:
            if playerYPos < MAX_TOP / 2:
                moveDirection = UP
            else:
                moveDirection = DOWN

    if playerYPos >= ADJ_TOP:
        if DEBUG_LEVEL >= DEBUG_MED:
            print(f"TOP WALL {ADJ_TOP} playerX {playerXPos} playerY {playerYPos}")
        if moveDirection == UP_RIGHT:
            moveDirection = RIGHT
        elif moveDirection == UP_LEFT:
            moveDirection = LEFT
        elif moveDirection == UP:
            if playerYPos < MAX_RIGHT / 2:
                moveDirection = RIGHT
            else:
                moveDirection = LEFT

    if playerYPos <= ADJ_BOTTOM:
        if DEBUG_LEVEL >= DEBUG_MED:
            print(f"BOTTOM WALL {ADJ_BOTTOM} playerX {playerXPos} playerY {playerYPos}")
        if moveDirection == DOWN_RIGHT:
            moveDirection = RIGHT
        elif moveDirection == DOWN_LEFT:
            moveDirection = LEFT
        elif moveDirection == DOWN:
            if playerYPos < MAX_RIGHT / 2:
                moveDirection = RIGHT
            else:
                moveDirection = LEFT

    return moveDirection


# Per-type repulsion weights for the threat-field flee. Hulks heaviest (invincible, the
# post-quark-fix #1 killer); quarks/projectiles high (teleport/fast); electrodes light.
THREAT_WEIGHTS = {}
THREAT_WEIGHTS[HULK] = 2.2
THREAT_WEIGHTS['Quark'] = 1.6
for _t in PROJECTILES:
    THREAT_WEIGHTS[_t] = 1.5
for _t in PRIORITY_ENEMIES:
    THREAT_WEIGHTS[_t] = 1.0
for _t in CHASE_ENEMIES:
    THREAT_WEIGHTS.setdefault(_t, 1.0)
for _t in ENEMIES:
    THREAT_WEIGHTS[_t] = 1.0
THREAT_WEIGHTS[OBSTACLE] = 0.5

# Adjacent-types that mean "we are fleeing" (so the threat-field override applies). FAMILY
# is excluded (we move TOWARD family, not away).
_FLEE_TYPES = {QUARK_TYPE, PROJECTILE_TYPE, PRIORITY_ENEMY_TYPE, CHASE_ENEMY_TYPE,
               ENEMY_TYPE, OBSTACLE, HULK}


def threatFieldMove(playerLocation, objectList):
    """Flee the WEIGHTED CENTROID of all nearby threats (toward open space), not just the one
    adjacent threat. Returns a stick direction (with wall handling) or None if no threats in
    range. Sums each threat's toward-vector * weight / dist, then reuses getMoveStick(AWAY)."""
    pX, pY = playerLocation[X_POS], playerLocation[Y_POS]
    r2 = FIELD_RADIUS * FIELD_RADIUS
    Tx = Ty = 0.0
    n = 0
    for obj in objectList:
        objX, objY, objType = obj
        w = THREAT_WEIGHTS.get(objType)
        if w is None:
            continue
        objY = Y_AXIS_INVERSION - objY          # match the flipped-Y frame used everywhere
        dx = objX - pX
        dy = objY - pY
        d2 = dx * dx + dy * dy
        if d2 > r2 or d2 < 1.0:
            continue
        n += 1
        inv = w / math.sqrt(d2)                 # 1/dist falloff: nearer threats dominate
        Tx += dx * inv
        Ty += dy * inv
    # PINCER GATE: the field only helps when a FEW threats box the player (flee-into-2nd-threat).
    # In a swarm the weighted centroid sits ~on the player -> weak/erratic vector that is WORSE
    # than fleeing the single nearest threat (broad version validated NET-NEGATIVE -1.2 wave).
    # So defer to the simple nearest-flee unless 2..FIELD_MAX_THREATS threats are in range.
    if n < 2 or n > FIELD_MAX_THREATS or (Tx == 0.0 and Ty == 0.0):
        return None
    # [dist, x, y] toward the weighted threat centroid; AWAY flips it to flee + wall-handles.
    return getMoveStick([1, Tx, Ty], AWAY, playerLocation)


def _hulk_ahead(direction, playerLocation, objectList):
    """True if an invincible Hulk lies within HULK_DEFLECT_R and ~<=45deg of `direction`."""
    vx, vy = _DIRVEC[direction]
    pX, pY = playerLocation[X_POS], playerLocation[Y_POS]
    for obj in objectList:
        objX, objY, objType = obj
        if objType != HULK:
            continue
        dx = objX - pX
        dy = (Y_AXIS_INVERSION - objY) - pY
        d = math.sqrt(dx * dx + dy * dy)
        if d < 1.0 or d > HULK_DEFLECT_R:
            continue
        # cos(angle between flee heading and hulk bearing); 0.7 ~ 45deg cone
        if (dx * vx + dy * vy) / d >= 0.7:
            return True
    return False


def hulkDeflect(moveStick, playerLocation, objectList):
    """If the flee heading points at a near hulk, nudge to the nearest +/-45deg heading that's
    clear of hulks. Keeps the decisive nearest-flee everywhere else."""
    if moveStick == STAY or moveStick == INVALID:
        return moveStick
    if not _hulk_ahead(moveStick, playerLocation, objectList):
        return moveStick
    for step in (1, -1, 2, -2):                 # try ever-wider deflections, nearest first
        cand = ((moveStick - 1 + step) % 8) + 1
        if not _hulk_ahead(cand, playerLocation, objectList):
            return cand
    return moveStick                             # boxed in by hulks — keep original


def _edge_ahead(direction, playerLocation):
    """True if moving `direction` drives the player further INTO a near board edge."""
    vx, vy = _DIRVEC[direction]
    pX, pY = playerLocation[X_POS], playerLocation[Y_POS]
    M = EDGE_DEFLECT_M
    if pX > MAX_RIGHT - M and vx > 0.3:
        return True
    if pX < MAX_LEFT + M and vx < -0.3:
        return True
    if pY > MAX_TOP - M and vy > 0.3:
        return True
    if pY < MAX_BOTTOM + M and vy < -0.3:
        return True
    return False


def edgeDeflect(moveStick, playerLocation):
    """If the flee heading drives into a near edge, nudge to the nearest +/-45deg heading that
    slides along it instead. Runs earlier (55px) than the at-the-wall handling (2-20px)."""
    if moveStick == STAY or moveStick == INVALID:
        return moveStick
    if not _edge_ahead(moveStick, playerLocation):
        return moveStick
    for step in (1, -1, 2, -2):
        cand = ((moveStick - 1 + step) % 8) + 1
        if not _edge_ahead(cand, playerLocation):
            return cand
    return moveStick                             # boxed into a corner — keep original


def _nearest_killable(*cands):
    """Closest non-INVALID threat-distance among killable categories (excludes hulks)."""
    best = INVALID
    for c in cands:
        if c != INVALID and (best == INVALID or c[DISTANCE] < best[DISTANCE]):
            best = c
    return best


"""
objectList - array of objects detected on screen with 3 values: objX, objY, objType
Evaluate all objects on screen and determine the best output to the joysticks.
Return a pair of values from the Joystick Directions constants: moveStick, fireStick
"""


def chooseOutputs(objectList):
    # This global probably doesn't need to be declared
    global Y_AXIS_INVERSION

    _apply_buffer_scale()   # one-time flee-radius scaling (after EVOLVE_PARAMS load)

    moveStick = INVALID
    fireStick = INVALID

    playerLocation = INVALID
    playerFound = INVALID

    adjacentType = INVALID
    adjacent = INVALID
    closeMoveCount = 0
    closeFireCount = 0

    nearestProjectile = INVALID
    nearestPriorityEnemy = INVALID
    nearestChaseEnemy = INVALID
    nearestEnemy = INVALID
    nearestHulk = INVALID
    nearestWall = INVALID
    nearestCivilian = INVALID
    nearestObstacle = INVALID

    # NO_SHOOT_SHELLS bookkeeping: shells stay in the projectile bucket for all
    # flee decisions, but fire gets redirected to a killable (non-shell) target.
    nearestKillableProjectile = INVALID
    adjacentIsShell = False
    nearestProjectileIsShell = False
    brainSeen = False   # brain-wave rescue aggression (see BRAIN_RESCUE_MULT)

    projectileDistance = INVALID
    priorityEnemyDistance = INVALID
    chaseEnemyDistance = INVALID
    enemyDistance = INVALID
    hulkDistance = INVALID
    civilianDistance = INVALID
    obstacleDistance = INVALID

    """ Find the Player first, as distance only matters relative to the player """
    for obj in objectList:
        objX, objY, objType = obj
        """ Y Axis is inverted. It is 0 at the top border, and increases downward """
        """ This should adjust the Y value so that 0 is at the bottom border, and it increases upwward """
        objY = Y_AXIS_INVERSION - objY

        if objType == PLAYER:
            playerFound = True
            playerLocation = [objX, objY]
            if (objX == 332 and objY == 246):
                if DEBUG_LEVEL >= DEBUG_LOW:
                    print("PLAYER IN CENTER - MAYBE NEW LEVEL")
            if DEBUG_LEVEL >= DEBUG_LOW:
                print(f"Player x {objX} y {objY}")
        break

    if playerFound == INVALID:
        if DEBUG_LEVEL >= DEBUG_LOW:
            print("Player location not found")
        return [STAY, UP]

    """ CHECK ALL OBJECTS """
    for obj in objectList:
        objX, objY, objType = obj

        """ Y Axis is inverted. It is 0 at the top border, and increases downward """
        """ This should adjust the Y value so that 0 is at the bottom border, and it increases upwward """
        objY = Y_AXIS_INVERSION - objY

        if DEBUG_LEVEL >= DEBUG_HIGH:
            print(f"Object List Loop {obj} x {objX} y {objY} type {objType}")

        """ skip the PLAYER and his BULLETs """
        if objType == PLAYER or objType == BULLET:
            continue

        if objType in PROJECTILES:
            _isShell = objType == 'TankShell'
            projectileDistance = getDistance(playerLocation, objX, objY)
            if DEBUG_LEVEL >= DEBUG_HIGH:
                print(f"Enemy x {objX} y {objY} dist {projectileDistance} prior nearest Projectile {nearestProjectile}")
            if projectileDistance[DISTANCE] <= CLOSE_MOVE_PROJECTILE:
                if DEBUG_LEVEL >= DEBUG_MED:
                    print(f"Projectile is CLOSE_MOVE {projectileDistance}")
                closeMoveCount += 1
            if projectileDistance[DISTANCE] <= CLOSE_FIRE_PROJECTILE:
                if DEBUG_LEVEL >= DEBUG_MED:
                    print(f"Projectile is CLOSE_FIRE {projectileDistance}")
                closeFireCount += 1
            if projectileDistance[DISTANCE] <= ADJACENT:
                if adjacent == INVALID or adjacentType != PROJECTILE_TYPE or projectileDistance[DISTANCE] < adjacent[DISTANCE]:
                    adjacentType = PROJECTILE_TYPE
                    adjacent = projectileDistance
                    adjacentIsShell = _isShell
                    if DEBUG_LEVEL >= DEBUG_HIGH:
                        print("Adjacent set to new Projectile")
                if DEBUG_LEVEL >= DEBUG_MED:
                    print("Projectile is Adjacent")
                    print(f"Projectile {obj} dist {projectileDistance}")
            if nearestProjectile == INVALID or nearestProjectile[DISTANCE] > projectileDistance[DISTANCE]:
                nearestProjectile = projectileDistance
                nearestProjectileIsShell = _isShell
                if DEBUG_LEVEL >= DEBUG_HIGH:
                    print(f"NearestProjectile updated to {projectileDistance}")
            if not _isShell and (nearestKillableProjectile == INVALID
                                 or nearestKillableProjectile[DISTANCE] > projectileDistance[DISTANCE]):
                nearestKillableProjectile = projectileDistance

        elif objType in PRIORITY_ENEMIES:
            priorityEnemyDistance = getDistance(playerLocation, objX, objY)
            if DEBUG_LEVEL >= DEBUG_HIGH:
                print(
                    f"Priority Enemy x {objX} y {objY} dist {priorityEnemyDistance} prior nearest Priority Enemy {nearestPriorityEnemy}")
            if priorityEnemyDistance[DISTANCE] <= CLOSE_MOVE_PRIORITY_ENEMY:
                if DEBUG_LEVEL >= DEBUG_MED:
                    print(f"Priority Enemy is CLOSE_MOVE {priorityEnemyDistance}")
                closeMoveCount += 1
            if priorityEnemyDistance[DISTANCE] <= CLOSE_FIRE_PRIORITY_ENEMY:
                if DEBUG_LEVEL >= DEBUG_MED:
                    print(f"Priority Enemy is CLOSE_FIRE {priorityEnemyDistance}")
                closeFireCount += 1
            if priorityEnemyDistance[DISTANCE] <= ADJACENT:
                if adjacent == INVALID or adjacentType not in (PROJECTILE_TYPE, QUARK_TYPE) or (adjacentType == PRIORITY_ENEMY_TYPE and priorityEnemyDistance[DISTANCE] < adjacent[DISTANCE]):
                    adjacentType = PRIORITY_ENEMY_TYPE
                    adjacent = priorityEnemyDistance
                    if DEBUG_LEVEL >= DEBUG_MED:
                        print("Priority Enemy is Adjacent")
                        print(f"Priority Enemy {obj} dist {priorityEnemyDistance}")
            if nearestPriorityEnemy == INVALID or nearestPriorityEnemy[DISTANCE] > priorityEnemyDistance[DISTANCE]:
                nearestPriorityEnemy = priorityEnemyDistance
                if DEBUG_LEVEL >= DEBUG_HIGH:
                    print(f"NearestPrioirtyEnemy updated to {priorityEnemyDistance}")

        elif objType == QUARK and ADJACENT_QUARK > 0:
            # Structural Quark handling: kite (fire + flee) at a large radius, top flee
            # priority just below projectiles. Still counts toward close-fire so it gets
            # shot at range while backing away.
            quarkDistance = getDistance(playerLocation, objX, objY)
            if DEBUG_LEVEL >= DEBUG_MED:
                print(f"Quark x {objX} y {objY} dist {quarkDistance} (structural)")
            if quarkDistance[DISTANCE] <= CLOSE_FIRE_CHASE_ENEMY:
                closeFireCount += 1
            if quarkDistance[DISTANCE] <= ADJACENT_QUARK:
                # Override any adjacent threat except a projectile or a closer quark.
                if adjacent == INVALID or (adjacentType != PROJECTILE_TYPE and adjacentType != QUARK_TYPE) \
                        or (adjacentType == QUARK_TYPE and quarkDistance[DISTANCE] < adjacent[DISTANCE]):
                    adjacentType = QUARK_TYPE
                    adjacent = quarkDistance
                    if DEBUG_LEVEL >= DEBUG_MED:
                        print(f"Quark is Adjacent (structural flee) {quarkDistance}")
            if nearestChaseEnemy == INVALID or nearestChaseEnemy[DISTANCE] > quarkDistance[DISTANCE]:
                nearestChaseEnemy = quarkDistance

        elif objType in CHASE_ENEMIES:
            chaseEnemyDistance = getDistance(playerLocation, objX, objY)
            if DEBUG_LEVEL >= DEBUG_MED:
                print(
                    f"Chase Enemy x {objX} y {objY} dist {chaseEnemyDistance} prior nearest Chase Enemy {nearestChaseEnemy}")
            _chaseMoveThresh = (CLOSE_MOVE_QUARK if (objType == 'Quark' and CLOSE_MOVE_QUARK is not None)
                                else CLOSE_MOVE_CHASE_ENEMY)
            if chaseEnemyDistance[DISTANCE] <= _chaseMoveThresh:
                if DEBUG_LEVEL >= DEBUG_MED:
                    print(f"Chase Enemy is CLOSE_MOVE {chaseEnemyDistance}")
                closeMoveCount += 1
            if chaseEnemyDistance[DISTANCE] <= CLOSE_FIRE_CHASE_ENEMY:
                if DEBUG_LEVEL >= DEBUG_MED:
                    print(f"Chase Enemy is CLOSE_FIRE {chaseEnemyDistance}")
                closeFireCount += 1
            if chaseEnemyDistance[DISTANCE] <= ADJACENT:
                if adjacent == INVALID or (adjacentType not in (PROJECTILE_TYPE, QUARK_TYPE, PRIORITY_ENEMY_TYPE)) or (adjacentType == CHASE_ENEMY_TYPE and chaseEnemyDistance[DISTANCE] < adjacent[DISTANCE]):
                    adjacentType = CHASE_ENEMY_TYPE
                    adjacent = chaseEnemyDistance
                    if DEBUG_LEVEL >= DEBUG_MED:
                        print("Chase Enemy is Adjacent")
                        print(f"Chase Enemy {obj} dist {chaseEnemyDistance}")
            if nearestChaseEnemy == INVALID or nearestChaseEnemy[DISTANCE] > chaseEnemyDistance[DISTANCE]:
                nearestChaseEnemy = chaseEnemyDistance
                if DEBUG_LEVEL >= DEBUG_MED:
                    print(f"NearestChaseEnemy updated to {chaseEnemyDistance}")

        elif objType in ENEMIES:
            if objType == 'Brain':
                brainSeen = True
            enemyDistance = getDistance(playerLocation, objX, objY)
            if DEBUG_LEVEL >= DEBUG_HIGH:
                print(f"Enemy x {objX} y {objY} dist {enemyDistance} prior nearest {nearestEnemy}")
            if enemyDistance[DISTANCE] <= CLOSE_MOVE_ENEMY:
                if DEBUG_LEVEL >= DEBUG_MED:
                    print(f"Enemy is CLOSE_MOVE {enemyDistance}")
                closeMoveCount += 1
            if enemyDistance[DISTANCE] <= CLOSE_FIRE_ENEMY:
                if DEBUG_LEVEL >= DEBUG_MED:
                    print(f"Enemy is CLOSE_FIRE {enemyDistance}")
                closeFireCount += 1
            if enemyDistance[DISTANCE] <= ADJACENT:
                if adjacent == INVALID or (adjacentType not in (PROJECTILE_TYPE, QUARK_TYPE, PRIORITY_ENEMY_TYPE, CHASE_ENEMY_TYPE)) or (adjacentType == ENEMY_TYPE and enemyDistance[DISTANCE] < adjacent[DISTANCE]):
                    adjacentType = ENEMY_TYPE
                    adjacent = enemyDistance
                    if DEBUG_LEVEL >= DEBUG_MED:
                        print("Enemy is Adjacent")
                        print(f"Enemy {obj} dist {enemyDistance}")
            if nearestEnemy == INVALID or nearestEnemy[DISTANCE] > enemyDistance[DISTANCE]:
                nearestEnemy = enemyDistance
                if DEBUG_LEVEL >= DEBUG_HIGH:
                    print(f"NearestEnemy updated to {enemyDistance}")

        elif objType == HULK:
            hulkDistance = getDistance(playerLocation, objX, objY)
            if hulkDistance[DISTANCE] <= CLOSE_MOVE_HULK:
                if DEBUG_LEVEL >= DEBUG_MED:
                    print(f"Hulk is CLOSE_MOVE {hulkDistance}")
                closeMoveCount += 1
            if hulkDistance[DISTANCE] <= CLOSE_FIRE_HULK:
                if DEBUG_LEVEL >= DEBUG_MED:
                    print(f"Hulk is CLOSE_FIRE {hulkDistance}")
                closeFireCount += 1
            if hulkDistance[DISTANCE] <= ADJACENT_HULK:
                if adjacent == INVALID or (adjacentType not in (PROJECTILE_TYPE, QUARK_TYPE, PRIORITY_ENEMY_TYPE, CHASE_ENEMY_TYPE, ENEMY_TYPE)) or (adjacentType == HULK and hulkDistance[DISTANCE] < adjacent[DISTANCE]):
                    adjacentType = HULK
                    adjacent = hulkDistance
                    if DEBUG_LEVEL >= DEBUG_MED:
                        print("Hulk is Adjacent")
                        print(f"Hulk {obj} dist {hulkDistance}")
                """ do not break, because other adjacent things would be more important """
            if nearestHulk == INVALID or nearestHulk[DISTANCE] > hulkDistance[DISTANCE]:
                nearestHulk = hulkDistance
                if DEBUG_LEVEL >= DEBUG_HIGH:
                    print(f"NearestHulk updated to {enemyDistance}")

        elif objType == OBSTACLE:
            obstacleDistance = getDistance(playerLocation, objX, objY)
            if obstacleDistance[DISTANCE] <= CLOSE_MOVE_OBSTACLE:
                if DEBUG_LEVEL >= DEBUG_MED:
                    print(f"Obstacle is CLOSE_MOVE {obstacleDistance}")
                closeMoveCount += 1
            if obstacleDistance[DISTANCE] <= CLOSE_FIRE_OBSTACLE:
                if DEBUG_LEVEL >= DEBUG_MED:
                    print(f"Obstacle is CLOSE_FIRE {obstacleDistance}")
                closeFireCount += 1
            if obstacleDistance[DISTANCE] <= ADJACENT:
                if adjacent == INVALID or (adjacentType not in (PROJECTILE_TYPE, QUARK_TYPE, PRIORITY_ENEMY_TYPE, CHASE_ENEMY_TYPE, ENEMY_TYPE, HULK)) or (adjacentType == OBSTACLE and obstacleDistance[DISTANCE] < adjacent[DISTANCE]):
                    adjacentType = OBSTACLE
                    adjacent = obstacleDistance
                    if DEBUG_LEVEL >= DEBUG_MED:
                        print("Obstacle is Adjacent")
                        print(f"Obstacle {obj} dist {obstacleDistance}")
                """ do not break, because other adjacent things would be more important """
            if nearestObstacle == INVALID or nearestObstacle[DISTANCE] > obstacleDistance[DISTANCE]:
                nearestObstacle = obstacleDistance
                if DEBUG_LEVEL >= DEBUG_HIGH:
                    print(f"NearestObstacle updated to {obstacleDistance}")

        elif objType in FAMILY:
            civilianDistance = getDistance(playerLocation, objX, objY)
            if civilianDistance[DISTANCE] <= ADJACENT:
                if adjacent == INVALID or (adjacentType == FAMILY_TYPE and civilianDistance[DISTANCE] < adjacent[DISTANCE]):
                    adjacentType = FAMILY_TYPE
                    adjacent = civilianDistance
                    if DEBUG_LEVEL >= DEBUG_MED:
                        print("Civilian is Adjacent")
                        print(f"Civilian {obj} dist {civilianDistance}")
                """ do not break, because other adjacent things would be more important """
            if nearestCivilian == INVALID or nearestCivilian[DISTANCE] > civilianDistance[DISTANCE]:
                nearestCivilian = civilianDistance
                if DEBUG_LEVEL >= DEBUG_HIGH:
                    print(f"NearestCivilian updated to {civilianDistance}")

        else:
            if DEBUG_LEVEL >= DEBUG_LOW:
                print(f"Unknown object {obj}")

    if DEBUG_LEVEL >= DEBUG_HIGH:
        print("Done looping over objects")

    """ DONE CHECK ALL OBJECTS """

    """ CHECK ADJACENT """

    """ actions for when something is right next to the player """
    if adjacent != INVALID:
        if DEBUG_LEVEL >= DEBUG_MED:
            print(f"ADJACENT {adjacentType} is not invalid {adjacent}")
        if adjacentType == QUARK_TYPE:
            # Kite the quark: fire at it AND always actively flee (no STAY) — standing
            # still in a teleporting-quark swarm is the dominant death (median 7.5px).
            fireStick = getFireStick(adjacent)
            moveStick = getMoveStick(adjacent, AWAY, playerLocation)
            if DEBUG_LEVEL >= DEBUG_MED:
                print(f"Adjacent Quark (structural flee) {adjacent} and player {playerLocation}")
        elif adjacentType == PROJECTILE_TYPE:
            if NO_SHOOT_SHELLS and adjacentIsShell:
                # $F1 exploit: don't shoot the shell (let it expire and eat the wave's
                # shell budget) — dodge it and spend the shot on a killable threat.
                _kt = _nearest_killable(nearestKillableProjectile, nearestChaseEnemy,
                                        nearestPriorityEnemy, nearestEnemy)
                fireStick = getFireStick(_kt) if _kt != INVALID else getFireStick(adjacent)
            else:
                fireStick = getFireStick(adjacent)
            if closeMoveCount > CLOSE_MOVE_COUNT_LIMIT:
                moveStick = STAY
            else:
                moveStick = getMoveStick(adjacent, AWAY, playerLocation)
            if DEBUG_LEVEL >= DEBUG_MED:
                print(f"Adjacent Projectile {adjacent} and player {playerLocation}")
        elif adjacentType == PRIORITY_ENEMY_TYPE:
            fireStick = getFireStick(adjacent)
            if closeMoveCount > CLOSE_MOVE_COUNT_LIMIT:
                moveStick = STAY
            else:
                moveStick = getMoveStick(adjacent, AWAY, playerLocation)
            if DEBUG_LEVEL >= DEBUG_MED:
                print(f"Adjacent Priority Enemy {adjacent} and player {playerLocation}")
        elif adjacentType == CHASE_ENEMY_TYPE:
            fireStick = getFireStick(adjacent)
            if closeMoveCount > CLOSE_MOVE_COUNT_LIMIT:
                moveStick = STAY
            else:
                moveStick = getMoveStick(adjacent, AWAY, playerLocation)
            if DEBUG_LEVEL >= DEBUG_MED:
                print(f"Adjacent Chase Enemy {adjacent} and player {playerLocation}")
        elif adjacentType == ENEMY_TYPE:
            fireStick = getFireStick(adjacent)
            if closeMoveCount > CLOSE_MOVE_COUNT_LIMIT:
                moveStick = STAY
            else:
                moveStick = getMoveStick(adjacent, AWAY, playerLocation)
            if DEBUG_LEVEL >= DEBUG_MED:
                print(f"Adjacent Enemy {adjacent} and player {playerLocation}")
        elif adjacentType == OBSTACLE:
            fireStick = getFireStick(adjacent)
            if closeMoveCount > CLOSE_MOVE_COUNT_LIMIT:
                moveStick = STAY
            else:
                moveStick = getMoveStick(adjacent, AWAY, playerLocation)
            if DEBUG_LEVEL >= DEBUG_MED:
                print(f"Adjacent Obstacle {adjacent} and player {playerLocation}")
        elif adjacentType == HULK:
            # A hulk is invincible — firing at it wastes the shot. Redirect fire to the nearest
            # KILLABLE threat (opt-in) so quarks/enemies die instead of closing in (wave-12 fix).
            if HULK_NOFIRE:
                _kt = _nearest_killable(
                    nearestKillableProjectile if NO_SHOOT_SHELLS else nearestProjectile,
                    nearestChaseEnemy, nearestPriorityEnemy, nearestEnemy)
                fireStick = getFireStick(_kt) if _kt != INVALID else getFireStick(adjacent)
            else:
                fireStick = getFireStick(adjacent)
            if closeMoveCount > CLOSE_MOVE_COUNT_LIMIT:
                moveStick = STAY
            else:
                moveStick = getMoveStick(adjacent, AWAY, playerLocation)
            if DEBUG_LEVEL >= DEBUG_MED:
                print(f"Adjacent Hulk {adjacent} and player {playerLocation}")
        elif adjacentType == FAMILY_TYPE:
            if closeMoveCount > CLOSE_MOVE_COUNT_LIMIT:
                moveStick = STAY
            else:
                moveStick = getMoveStick(adjacent, TOWARD, playerLocation)
            if DEBUG_LEVEL >= DEBUG_HIGH:
                print(f"Adjacent Family {adjacent} and player {playerLocation}")
        else:
            if DEBUG_LEVEL >= DEBUG_LOW:
                print(f"SHOULD NEVER HAPPEN Nothing critical is Adjacent {adjacent}")

        # Threat-field override: when actually fleeing (not STAY, not family), steer toward
        # open space away from ALL nearby threats instead of away from the single adjacent one.
        if THREAT_FIELD and adjacentType in _FLEE_TYPES and moveStick not in (STAY, INVALID):
            _fm = threatFieldMove(playerLocation, objectList)
            if _fm is not None:
                moveStick = _fm
        # Surgical hulk-deflection (independent opt-in): nudge the flee off a hulk in its path.
        if HULK_DEFLECT and adjacentType in _FLEE_TYPES and moveStick not in (STAY, INVALID):
            moveStick = hulkDeflect(moveStick, playerLocation, objectList)
        # Surgical edge-deflection (independent opt-in): nudge the flee off a near board edge.
        if EDGE_DEFLECT and adjacentType in _FLEE_TYPES and moveStick not in (STAY, INVALID):
            moveStick = edgeDeflect(moveStick, playerLocation)

        # Hulk-push (opt-in): when cornered (STAY) with a hulk in push range, redirect
        # fire AT the hulk to shove it away and open an escape (asm:471). Only in the
        # STAY case — the pin — so it doesn't steal fire from killable threats while fleeing.
        if HULK_PUSH and moveStick == STAY and nearestHulk != INVALID \
                and nearestHulk[DISTANCE] <= HULK_PUSH_R:
            fireStick = getFireStick(nearestHulk)
            if DEBUG_LEVEL >= DEBUG_MED:
                print(f"Hulk-push: cornered, shoving hulk {nearestHulk}")

    if moveStick != INVALID and fireStick != INVALID:
        if DEBUG_LEVEL >= DEBUG_LOW:
            print(
                f"ADJACENT decision move {moveStick} fire {fireStick} closeMoveCount {closeMoveCount} Adjacent {adjacentType} {adjacent}")
        return [moveStick, fireStick]

    if DEBUG_LEVEL >= DEBUG_MED:
        print(f"Non Adjacent projectile {nearestProjectile} enemy {nearestEnemy}")
    if (moveStick != INVALID):
        if DEBUG_LEVEL >= DEBUG_HIGH:
            print(f"ONLY IF FAMILY WAS ADJACENT moveStick already set {moveStick}")

    """ DONE CHECK ADJACENT """

    """ CHECK CLOSE """

    """ actions for non-adjacent cases"""
    if nearestProjectile != INVALID:
        if nearestProjectile[DISTANCE] <= CLOSE_MOVE_PROJECTILE:
            if DEBUG_LEVEL >= DEBUG_MED:
                print(f"Projectile is CLOSE_MOVE {nearestProjectile}")
            if moveStick == INVALID:
                moveStick = getMoveStick(nearestProjectile, AWAY, playerLocation)
                if DEBUG_LEVEL >= DEBUG_LOW:
                    print(f"nearest Projectile is CLOSE_MOVE fire {fireStick} move {moveStick} player {playerLocation}")
        if nearestProjectile[DISTANCE] <= CLOSE_FIRE_PROJECTILE:
            if DEBUG_LEVEL >= DEBUG_MED:
                print(f"Projectile is CLOSE_FIRE {nearestProjectile}")
            if fireStick == INVALID:
                if NO_SHOOT_SHELLS and nearestProjectileIsShell:
                    # $F1 exploit: skip the shell; shoot the nearest killable projectile
                    # if one is in range, else fall through to the chase/priority chain.
                    if nearestKillableProjectile != INVALID \
                            and nearestKillableProjectile[DISTANCE] <= CLOSE_FIRE_PROJECTILE:
                        fireStick = getFireStick(nearestKillableProjectile)
                else:
                    fireStick = getFireStick(nearestProjectile)
                if DEBUG_LEVEL >= DEBUG_LOW:
                    print(f"nearest Projectile is CLOSE_FIRE fire {fireStick} move {moveStick} player {playerLocation}")

    if nearestChaseEnemy != INVALID:
        if nearestChaseEnemy[DISTANCE] <= CLOSE_MOVE_CHASE_ENEMY:
            if DEBUG_LEVEL >= DEBUG_MED:
                print(f"Chase Enemy is CLOSE_MOVE {nearestChaseEnemy}")
            if moveStick == INVALID:
                moveStick = getMoveStick(nearestChaseEnemy, AWAY, playerLocation)
                if DEBUG_LEVEL >= DEBUG_LOW:
                    print(
                        f"nearest Chase Enemy is CLOSE_MOVE fire {fireStick} move {moveStick} player {playerLocation}")
        if nearestChaseEnemy[DISTANCE] <= CLOSE_FIRE_CHASE_ENEMY:
            if DEBUG_LEVEL >= DEBUG_MED:
                print(f"Chase Enemy is CLOSE_FIRE {nearestChaseEnemy}")
            if fireStick == INVALID:
                fireStick = getFireStick(nearestChaseEnemy)
                if DEBUG_LEVEL >= DEBUG_LOW:
                    print(
                        f"nearest Chase Enemy is CLOSE_FIRE fire {fireStick} move {moveStick} player {playerLocation}")

    if nearestPriorityEnemy != INVALID:
        if nearestPriorityEnemy[DISTANCE] <= CLOSE_MOVE_PRIORITY_ENEMY:
            if DEBUG_LEVEL >= DEBUG_MED:
                print(f"Priority Enemy is CLOSE_MOVE {nearestPriorityEnemy}")
            if moveStick == INVALID:
                moveStick = getMoveStick(nearestPriorityEnemy, AWAY, playerLocation)
                if DEBUG_LEVEL >= DEBUG_LOW:
                    print(
                        f"nearest Priority Enemy is CLOSE_MOVE fire {fireStick} move {moveStick} player {playerLocation}")
        if nearestPriorityEnemy[DISTANCE] <= CLOSE_FIRE_PRIORITY_ENEMY:
            if DEBUG_LEVEL >= DEBUG_MED:
                print(f"Priority Enemy is CLOSE_FIRE {nearestPriorityEnemy}")
            if fireStick == INVALID:
                fireStick = getFireStick(nearestPriorityEnemy)
                if DEBUG_LEVEL >= DEBUG_LOW:
                    print(
                        f"nearest Priority Enemy is CLOSE_FIRE fire {fireStick} move {moveStick} player {playerLocation}")

    if nearestEnemy != INVALID:
        if nearestEnemy[DISTANCE] <= CLOSE_MOVE_ENEMY:
            if DEBUG_LEVEL >= DEBUG_MED:
                print(f"Enemy is CLOSE_MOVE {nearestEnemy}")
            if moveStick == INVALID:
                moveStick = getMoveStick(nearestEnemy, AWAY, playerLocation)
                if DEBUG_LEVEL >= DEBUG_LOW:
                    print(f"nearest Enemy is CLOSE_MOVE fire {fireStick} move {moveStick} player {playerLocation}")
        if nearestEnemy[DISTANCE] <= CLOSE_FIRE_ENEMY:
            if DEBUG_LEVEL >= DEBUG_MED:
                print(f"Enemy is CLOSE_FIRE {nearestEnemy}")
            if fireStick == INVALID:
                fireStick = getFireStick(nearestEnemy)
                if DEBUG_LEVEL >= DEBUG_LOW:
                    print(f"nearest Enemy is CLOSE_FIRE fire {fireStick} move {moveStick} player {playerLocation}")

    if nearestHulk != INVALID:
        if nearestHulk[DISTANCE] <= CLOSE_MOVE_HULK:
            if DEBUG_LEVEL >= DEBUG_MED:
                print(f"Hulk is CLOSE_MOVE {nearestHulk}")
            if moveStick == INVALID:
                moveStick = getMoveStick(nearestHulk, AWAY, playerLocation)
                if DEBUG_LEVEL >= DEBUG_LOW:
                    print(f"nearest Hulk is CLOSE_MOVE fire {fireStick} move {moveStick} player {playerLocation}")
        if nearestHulk[DISTANCE] <= CLOSE_FIRE_HULK:
            if DEBUG_LEVEL >= DEBUG_MED:
                print(f"Hulk is CLOSE_FIRE {nearestHulk}")
            if fireStick == INVALID:
                fireStick = getFireStick(nearestHulk)
                if DEBUG_LEVEL >= DEBUG_LOW:
                    print(f"nearest Hulk is CLOSE_FIRE fire {fireStick} move {moveStick} player {playerLocation}")

    if nearestObstacle != INVALID:
        if nearestObstacle[DISTANCE] <= CLOSE_MOVE_OBSTACLE:
            if DEBUG_LEVEL >= DEBUG_MED:
                print(f"Obstacle is CLOSE_MOVE {nearestObstacle}")
            if moveStick == INVALID:
                moveStick = getMoveStick(nearestObstacle, AWAY, playerLocation)
                if DEBUG_LEVEL >= DEBUG_LOW:
                    print(f"nearest Obstacle is CLOSE_MOVE fire {fireStick} move {moveStick} player {playerLocation}")
        if nearestObstacle[DISTANCE] <= CLOSE_FIRE_OBSTACLE:
            if DEBUG_LEVEL >= DEBUG_MED:
                print(f"Obstacle is CLOSE_FIRE {nearestObstacle}")
            if fireStick == INVALID:
                fireStick = getFireStick(nearestObstacle)
                if DEBUG_LEVEL >= DEBUG_LOW:
                    print(f"nearest Obstacle is CLOSE_FIRE fire {fireStick} move {moveStick} player {playerLocation}")

    if nearestCivilian != INVALID:
        if DEBUG_LEVEL >= DEBUG_HIGH:
            print(f"Found a nearest civilian {nearestCivilian}")
        # Pursuit radius scales up on brain waves (brains ignore the player while any
        # human lives, asm:1BFF-1C08 — sweeping is structurally safe). Legacy path keeps
        # the pre-existing last-seen-distance gate for A/B purity; RESCUE_SEEK uses the
        # nearest civilian's own distance (the legacy gate compared against whichever
        # family member happened to be processed last).
        _civ_r = CLOSE_MOVE_CIVILIAN * (BRAIN_RESCUE_MULT if brainSeen else 1.0)
        _civ_d = nearestCivilian[DISTANCE] if RESCUE_SEEK else civilianDistance[DISTANCE]
        if _civ_d <= _civ_r:
            if DEBUG_LEVEL >= DEBUG_MED:
                print(f"Civilian is CLOSE_MOVE {nearestCivilian}")
            if moveStick == INVALID:
                moveStick = getMoveStick(nearestCivilian, TOWARD, playerLocation)
                if DEBUG_LEVEL >= DEBUG_LOW:
                    print(f"nearest Civilian is CLOSE_MOVE fire {fireStick} move {moveStick} player {playerLocation}")

    """ post 199 logic change to move if there's a projectile """
    if (closeMoveCount > CLOSE_MOVE_COUNT_LIMIT and nearestProjectile == INVALID):
        moveStick = STAY
        if DEBUG_LEVEL >= DEBUG_LOW:
            print(f"No projectiles and closeMoveCount {closeMoveCount} past limit {CLOSE_MOVE_COUNT_LIMIT} so STAY")

    if DEBUG_LEVEL >= DEBUG_MED:
        print(f"Nearest and Close checks Complete closeMoveCount {closeMoveCount} fire {fireStick} move {moveStick}")

    """ DONE CHECK CLOSE """

    """ if nothing is close, shoot the nearest high priority target, and STAY """
    """ if there are no close high priority targets, do not fire and save it for when something is close """
    """ if there are 0 enemies and projectiles, the level should have ended """
    if fireStick == INVALID:
        """
        if nearestProjectile != INVALID:
            if DEBUG_LEVEL >= DEBUG_MED:
                print(f"get fire for nearest Projectile")
            fireStick = getFireStick(nearestProjectile)
            if DEBUG_LEVEL >= DEBUG_MED:
                print(f"fire for nearest Projectile {fireStick}")
        elif nearestPriorityEnemy != INVALID:
            if DEBUG_LEVEL >= DEBUG_MED:
                print(f"get fire for nearest Priority Enemy")
            fireStick = getFireStick(nearestPriorityEnemy)
            if DEBUG_LEVEL >= DEBUG_MED:
                print(f"fire for nearest Priority Enemy {fireStick}")
        else:
        """
        fireStick = STAY
        if DEBUG_LEVEL >= DEBUG_LOW:
            print(f"fireStick was INVALID until the end, so STAY")
    """
        elif nearestEnemy != INVALID:
            print(f"get fire for nearest Enemy")
            fireStick = getFireStick(nearestEnemy)
            print(f"fire for nearest Enemy {fireStick}")
        elif nearestObstacle != INVALID:
            print(f"NEVER HAPPEN no enemies or projectiles, so fire on Obstacle {nearestObstacle}")
            fireStick = getFireStick(nearestObstacle)
        elif nearestHulk != INVALID:
            print(f"NEVER HAPPEN no enemies, projectiles, or obstacles, so fire on Hulk {nearestHulk}")
            fireStick = getFireStick(nearestHulk)
        else:
            fireStick = UP
            print("fireStick and nearest target were INVALID, which should never happen")
    """

    if moveStick == INVALID:
        if DEBUG_LEVEL >= DEBUG_LOW:
            print("moveStick was INVALID until the end")
        # RESCUE_SEEK: rescue outranks hunting when idle — family is the life engine
        # (escalating 1000->5000 per rescue; extra man every 25k).
        if RESCUE_SEEK and nearestCivilian != INVALID:
            if DEBUG_LEVEL >= DEBUG_LOW:
                print(f"rescue-seek toward civilian {nearestCivilian}")
            moveStick = getMoveStick(nearestCivilian, TOWARD, playerLocation)
        elif nearestChaseEnemy != INVALID:
            if DEBUG_LEVEL >= DEBUG_LOW:
                print(f"move toward Chase Enemy {nearestChaseEnemy}")
            moveStick = getMoveStick(nearestChaseEnemy, TOWARD, playerLocation)
        elif nearestCivilian != INVALID:
            if DEBUG_LEVEL >= DEBUG_LOW:
                print(f"move toward civilian {nearestCivilian}")
            moveStick = getMoveStick(nearestCivilian, TOWARD, playerLocation)
        elif HUNT_KILLABLE and _nearest_killable(nearestPriorityEnemy, nearestEnemy) != INVALID:
            # Endgame hunt: close on the last killable enemy instead of hiding at a
            # wall until it wanders into range. Stop at HUNT_STANDOFF (the CLOSE_MOVE
            # flee radii take over below that) and fire at it the whole way in.
            _ht = _nearest_killable(nearestPriorityEnemy, nearestEnemy)
            if _ht[DISTANCE] > HUNT_STANDOFF:
                moveStick = getMoveStick(_ht, TOWARD, playerLocation)
            else:
                moveStick = STAY
            if fireStick == STAY:
                fireStick = getFireStick(_ht)
            if DEBUG_LEVEL >= DEBUG_LOW:
                print(f"hunt killable enemy {_ht} move {moveStick} fire {fireStick}")
        else:
            if DEBUG_LEVEL >= DEBUG_LOW:
                print("moveStick default to Stay")
            moveStick = STAY

    return [moveStick, fireStick]


def main(starting_level: int = 1, lives: int = 3, fps: int = 30, godmode: bool = False):
    global DEBUG_LEVEL, MAX_RIGHT, MAX_TOP, Y_AXIS_INVERSION, ADJ_TOP, ADJ_BOTTOM, ADJ_LEFT, ADJ_RIGHT

    config_path = path.join(path.dirname(__file__), "config.yaml")
    env = RobotronEnv(level=starting_level, lives=lives, fps=fps, config_path=config_path, godmode=godmode)
    board_size = env.get_board_size()
    # print(f"Board Size: {board_size}")  # Default Board Size: (665, 492)

    """ Set debug level for print statements here """
    DEBUG_LEVEL = DEBUG_OFF  # DEBUG_LOW DEBUG_MED DEBUG_HIGH

    MAX_RIGHT, MAX_TOP = board_size
    MAX_BOTTOM = 0
    MAX_LEFT = 0

    Y_AXIS_INVERSION = MAX_TOP

    BORDER_ADJUST = 20

    ADJ_TOP = MAX_TOP - 2  # BORDER_ADJUST
    ADJ_BOTTOM = MAX_BOTTOM + BORDER_ADJUST + 9
    ADJ_LEFT = MAX_LEFT + 2  # + BORDER_ADJUST
    ADJ_RIGHT = MAX_RIGHT - BORDER_ADJUST

    if DEBUG_LEVEL >= DEBUG_LOW:
        print(f"Board Size: {board_size} {MAX_RIGHT} {MAX_TOP}")  # Default Board Size: (665, 492)
        # Adjusted Board Size: (2-645, 29-490)
        print(f"adj top {ADJ_TOP} bot {ADJ_BOTTOM} left {ADJ_LEFT} right {ADJ_RIGHT}")

    env.reset()
    _, _, isDead, _, data = env.step(0)

    while True:
        if DEBUG_LEVEL >= DEBUG_LOW:
            print(f"FRAME START")
        if DEBUG_LEVEL >= DEBUG_HIGH:
            print(f"Objects: {data}")

        actionArray = chooseOutputs(data["data"])

        if DEBUG_LEVEL >= DEBUG_LOW:
            print(f"Move and Fire: {actionArray}")

        encodedAction = actionArray[0] * 9 + actionArray[1]

        if DEBUG_LEVEL >= DEBUG_HIGH:
            print(f"encoded action: {encodedAction}")

        _, _, isDead, _, data = env.step(encodedAction)

        # _image, reward, isDead, data = env.step(env.action_space.sample())
        # score, level, lives, family, data = data.values()
        # print(f"Score: {score} | Level: {level} | Lives: {lives} | Reward: {reward} | Dead: {isDead}")
        # print(f"Family Remaining: {family} | Objects: {data}")
        """
        Should look like:
        Score: 0 | Level: 1 | Lives: 3 | Reward: 0.0 | Dead: False
            Family Remaining: 2 | Objects: [(337, 246, 'Player'), (38, 292, 'Mommy'), (578, 223, 'Daddy'), 
            (489, 15, 'Grunt'), (7, 439, 'Grunt'), (613, 241, 'Grunt'), (195, 137, 'Electrode'), 
            (136, 123, 'Electrode'), (214, 161, 'Electrode'), (187, 334, 'Electrode'), (625, 186, 'Electrode'), 
            (352, 261, 'Bullet')]
        """

        if isDead:
            if DEBUG_LEVEL >= DEBUG_LOW:
                print("GAME OVER")
            if DEBUG_LEVEL >= DEBUG_LOW:
                time.sleep(100000)

        if DEBUG_LEVEL >= DEBUG_LOW:
            print("FRAME END")
            print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rainbow')
    parser.add_argument('--level', type=int, default=1, help='Start Level')
    parser.add_argument('--lives', type=int, default=3, help='Start Lives')
    parser.add_argument('--fps', type=int, default=200, help='FPS')
    parser.add_argument('--godmode', action='store_true', help='Enable GOD Mode (Can\'t die.)')

    args = parser.parse_args()
    main(args.level, args.lives, args.fps, args.godmode)
