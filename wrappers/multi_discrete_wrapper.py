"""Wrapper to convert MultiDiscrete action space to Discrete for RobotronEnv.

This allows the agent to learn movement and shooting independently as two separate
8-way directional choices, rather than learning a single 64-way combined action.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MultiDiscreteToDiscrete(gym.ActionWrapper):
    """Convert MultiDiscrete([8, 8]) actions to Discrete(64) for RobotronEnv.

    The RobotronEnv with always_move=True expects actions encoded as:
        action = movement_direction * 8 + shooting_direction

    This wrapper allows the agent to output two separate 8-way choices:
        action[0] = movement_direction (0-7)
        action[1] = shooting_direction (0-7)

    Direction encoding for both movement and shooting:
        8   1   2
          \ | /
        7 - 0 - 3
          / | \
        6   5   4

    Where:
        0 = no movement/shooting
        1 = up
        2 = up-right
        3 = right
        4 = down-right
        5 = down
        6 = down-left
        7 = left
        8 = up-left

    Args:
        env: RobotronEnv with Discrete(64) action space
    """

    def __init__(self, env: gym.Env):
        """Initialize wrapper.

        Args:
            env: RobotronEnv with Discrete(64) action space
        """
        super().__init__(env)

        # Verify the base env has the expected action space
        if not isinstance(env.action_space, spaces.Discrete):
            raise ValueError(f"Expected Discrete action space, got {type(env.action_space)}")

        if env.action_space.n != 64:
            raise ValueError(f"Expected Discrete(64) action space, got Discrete({env.action_space.n})")

        # Change action space to MultiDiscrete([8, 8])
        self.action_space = spaces.MultiDiscrete([8, 8])

    def action(self, action: np.ndarray) -> int:
        """Convert MultiDiscrete action to Discrete action.

        Args:
            action: Array of shape (2,) with [movement, shooting] directions

        Returns:
            Discrete action (0-63) for RobotronEnv
        """
        # Ensure action is a numpy array
        if not isinstance(action, np.ndarray):
            action = np.array(action)

        # Validate action shape
        if action.shape != (2,):
            raise ValueError(f"Expected action shape (2,), got {action.shape}")

        # Validate action ranges
        if not (0 <= action[0] < 8 and 0 <= action[1] < 8):
            raise ValueError(f"Action values must be in range [0, 7], got {action}")

        # Convert: discrete_action = movement * 8 + shooting
        movement_dir = int(action[0])
        shooting_dir = int(action[1])
        discrete_action = movement_dir * 8 + shooting_dir

        return discrete_action
