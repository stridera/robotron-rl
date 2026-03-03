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

# Distance values for tuning behavior
ADJACENT = 40
ADJACENT_HULK = 50

"""
CLOSE = 75 # regular enemy
"""
CLOSE_FIRE_PRIORITY_ENEMY = 150
CLOSE_MOVE_PRIORITY_ENEMY = 50
CLOSE_FIRE_PROJECTILE = 150
CLOSE_MOVE_PROJECTILE = 100
CLOSE_FIRE_CHASE_ENEMY = 175
CLOSE_MOVE_CHASE_ENEMY = 50
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


"""
objectList - array of objects detected on screen with 3 values: objX, objY, objType
Evaluate all objects on screen and determine the best output to the joysticks.
Return a pair of values from the Joystick Directions constants: moveStick, fireStick
"""


def chooseOutputs(objectList):
    # This global probably doesn't need to be declared
    global Y_AXIS_INVERSION

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
                    if DEBUG_LEVEL >= DEBUG_HIGH:
                        print("Adjacent set to new Projectile")
                if DEBUG_LEVEL >= DEBUG_MED:
                    print("Projectile is Adjacent")
                    print(f"Projectile {obj} dist {projectileDistance}")
            if nearestProjectile == INVALID or nearestProjectile[DISTANCE] > projectileDistance[DISTANCE]:
                nearestProjectile = projectileDistance
                if DEBUG_LEVEL >= DEBUG_HIGH:
                    print(f"NearestProjectile updated to {projectileDistance}")

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
                if adjacent == INVALID or adjacentType != PROJECTILE_TYPE or (adjacentType == PRIORITY_ENEMY_TYPE and priorityEnemyDistance[DISTANCE] < adjacent[DISTANCE]):
                    adjacentType = PRIORITY_ENEMY_TYPE
                    adjacent = priorityEnemyDistance
                    if DEBUG_LEVEL >= DEBUG_MED:
                        print("Priority Enemy is Adjacent")
                        print(f"Priority Enemy {obj} dist {priorityEnemyDistance}")
            if nearestPriorityEnemy == INVALID or nearestPriorityEnemy[DISTANCE] > priorityEnemyDistance[DISTANCE]:
                nearestPriorityEnemy = priorityEnemyDistance
                if DEBUG_LEVEL >= DEBUG_HIGH:
                    print(f"NearestPrioirtyEnemy updated to {priorityEnemyDistance}")

        elif objType in CHASE_ENEMIES:
            chaseEnemyDistance = getDistance(playerLocation, objX, objY)
            if DEBUG_LEVEL >= DEBUG_MED:
                print(
                    f"Chase Enemy x {objX} y {objY} dist {chaseEnemyDistance} prior nearest Chase Enemy {nearestChaseEnemy}")
            if chaseEnemyDistance[DISTANCE] <= CLOSE_MOVE_CHASE_ENEMY:
                if DEBUG_LEVEL >= DEBUG_MED:
                    print(f"Chase Enemy is CLOSE_MOVE {chaseEnemyDistance}")
                closeMoveCount += 1
            if chaseEnemyDistance[DISTANCE] <= CLOSE_FIRE_CHASE_ENEMY:
                if DEBUG_LEVEL >= DEBUG_MED:
                    print(f"Chase Enemy is CLOSE_FIRE {chaseEnemyDistance}")
                closeFireCount += 1
            if chaseEnemyDistance[DISTANCE] <= ADJACENT:
                if adjacent == INVALID or (adjacentType not in (PROJECTILE_TYPE, PRIORITY_ENEMY_TYPE)) or (adjacentType == CHASE_ENEMY_TYPE and chaseEnemyDistance[DISTANCE] < adjacent[DISTANCE]):
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
                if adjacent == INVALID or (adjacentType not in (PROJECTILE_TYPE, PRIORITY_ENEMY_TYPE, CHASE_ENEMY_TYPE)) or (adjacentType == ENEMY_TYPE and enemyDistance[DISTANCE] < adjacent[DISTANCE]):
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
                if adjacent == INVALID or (adjacentType not in (PROJECTILE_TYPE, PRIORITY_ENEMY_TYPE, CHASE_ENEMY_TYPE, ENEMY_TYPE)) or (adjacentType == HULK and hulkDistance[DISTANCE] < adjacent[DISTANCE]):
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
                if adjacent == INVALID or (adjacentType not in (PROJECTILE_TYPE, PRIORITY_ENEMY_TYPE, CHASE_ENEMY_TYPE, ENEMY_TYPE, HULK)) or (adjacentType == OBSTACLE and obstacleDistance[DISTANCE] < adjacent[DISTANCE]):
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
        if adjacentType == PROJECTILE_TYPE:
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
        if civilianDistance[DISTANCE] <= CLOSE_MOVE_CIVILIAN:
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
        if nearestChaseEnemy != INVALID:
            if DEBUG_LEVEL >= DEBUG_LOW:
                print(f"move toward Chase Enemy {nearestChaseEnemy}")
            moveStick = getMoveStick(nearestChaseEnemy, TOWARD, playerLocation)
        elif nearestCivilian != INVALID:
            if DEBUG_LEVEL >= DEBUG_LOW:
                print(f"move toward civilian {nearestCivilian}")
            moveStick = getMoveStick(nearestCivilian, TOWARD, playerLocation)
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
