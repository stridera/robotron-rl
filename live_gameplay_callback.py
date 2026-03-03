"""
Live Gameplay Callback - Display agent gameplay in a popup window periodically during training.

This callback creates a temporary environment with rendering enabled and displays
the agent playing for a short period. Useful for visually monitoring training progress.
"""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from robotron import RobotronEnv
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
from wrappers import MultiDiscreteToDiscrete, FrameSkipWrapper
import time


class LiveGameplayCallback(BaseCallback):
    """
    Periodically display live gameplay in a popup window.

    Creates a temporary rendering environment to show the agent playing.
    Does not impact training performance as it uses a separate environment.
    """

    def __init__(
        self,
        config_path: str = None,
        level: int = 1,
        lives: int = 5,
        display_freq: int = 50_000,
        display_steps: int = 500,
        fps: int = 30,
        verbose: int = 1,
    ):
        """
        Args:
            config_path: Path to game config YAML
            level: Starting level for display
            lives: Number of lives
            display_freq: Display gameplay every N steps
            display_steps: Number of environment steps to display
            fps: Frames per second for display
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.config_path = config_path
        self.level = level
        self.lives = lives
        self.display_freq = display_freq
        self.display_steps = display_steps
        self.fps = fps
        self.display_env = None

    def _init_callback(self) -> None:
        """Initialize the display environment (with rendering)"""
        # We'll create the env on-demand to avoid issues with forked processes
        pass

    def _create_display_env(self):
        """Create a temporary environment for display (matches training setup)"""
        import pygame

        # Ensure pygame is fully initialized (handles subprocess issues)
        if pygame.get_init():
            pygame.quit()  # Clean slate
        pygame.init()
        pygame.font.init()  # Explicitly init font module

        env = RobotronEnv(
            level=self.level,
            lives=self.lives,
            fps=self.fps,
            config_path=self.config_path,
            always_move=True,
            headless=False,  # Enable rendering!
        )

        # Match training preprocessing
        env = MultiDiscreteToDiscrete(env)
        env = FrameSkipWrapper(env, skip=4)
        env = GrayscaleObservation(env, keep_dim=False)
        env = ResizeObservation(env, (84, 84))
        env = FrameStackObservation(env, stack_size=4)

        return env

    def _on_step(self) -> bool:
        # Only display at specified frequency
        if self.n_calls % self.display_freq != 0:
            return True

        if self.verbose > 0:
            print(f"\n{'='*60}")
            print(f"LIVE GAMEPLAY DISPLAY - Step {self.num_timesteps:,}")
            print(f"Showing {self.display_steps} steps of gameplay...")
            print(f"{'='*60}\n")

        # Create temporary display environment
        display_env = self._create_display_env()

        try:
            obs, info = display_env.reset()
            display_env.render()  # Initial render
            steps_displayed = 0
            episode_reward = 0

            while steps_displayed < self.display_steps:
                # Get action from current policy
                action, _ = self.model.predict(obs, deterministic=False)
                obs, reward, terminated, truncated, info = display_env.step(action)
                display_env.render()  # Render after each step

                episode_reward += reward
                steps_displayed += 1

                # Small delay for human viewing
                time.sleep(0.01)

                if terminated or truncated:
                    # Start new episode
                    obs, info = display_env.reset()
                    display_env.render()  # Render reset
                    if self.verbose > 0:
                        score = info.get('score', 0)
                        level = info.get('level', 0) + 1
                        print(f"  Episode ended - Score: {score}, Level: {level}")
                    episode_reward = 0

            if self.verbose > 0:
                print(f"\nLive gameplay display complete!")
                print(f"{'='*60}\n")

        finally:
            # Clean up display environment
            display_env.close()

        return True
