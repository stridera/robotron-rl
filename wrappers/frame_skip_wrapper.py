"""Frame skip wrapper for faster learning."""

import numpy as np
import gymnasium as gym
from typing import Tuple, Any, Dict


class FrameSkipWrapper(gym.Wrapper):
    """Repeat actions for multiple frames to speed up learning.

    The agent chooses an action, and that action is repeated for `skip` frames.
    The rewards are accumulated over these frames.

    This wrapper should be applied BEFORE GrayScaleObservation and ResizeObservation,
    so it operates on the raw environment observations.

    Args:
        env: The environment to wrap
        skip: Number of frames to repeat each action (default: 4)
    """

    def __init__(self, env: gym.Env, skip: int = 4):
        """Initialize frame skip wrapper.

        Args:
            env: Environment to wrap
            skip: Number of frames to repeat actions (default: 4, standard for Atari)
        """
        super().__init__(env)
        self.skip = skip

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict]:
        """Repeat action for skip frames and accumulate reward.

        Args:
            action: Action to repeat

        Returns:
            Tuple of (observation, total_reward, terminated, truncated, info)
        """
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        # Repeat action for skip frames
        for i in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            # Stop early if episode ends
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info
