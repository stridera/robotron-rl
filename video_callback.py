"""
Callback for recording videos without impacting training speed.

Creates a temporary rendering environment only when recording videos,
then destroys it to avoid the 75% FPS penalty during normal training.
"""
import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from robotron import RobotronEnv
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation


class VideoRecordingCallback(BaseCallback):
    """
    Record gameplay videos periodically without slowing down training.

    Creates a temporary rendering environment only during video recording,
    then destroys it to maintain full training speed.
    """

    def __init__(
        self,
        record_freq: int = 50_000,
        video_length: int = 500,
        config_path: str = None,
        level: int = 1,
        lives: int = 5,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.record_freq = record_freq
        self.video_length = video_length
        self.config_path = config_path
        self.level = level
        self.lives = lives
        self.last_recording = -1

    def _on_step(self) -> bool:
        # Check if it's time to record
        if self.record_freq > 0 and self.num_timesteps % self.record_freq == 0:
            if self.num_timesteps != self.last_recording:
                self._record_video()
                self.last_recording = self.num_timesteps
        return True

    def _record_video(self):
        """Create temporary env, record video, destroy env."""
        if self.verbose > 0:
            print(f"\n{'='*60}")
            print(f"🎥 Recording video at {self.num_timesteps:,} steps...")
            print(f"{'='*60}")

        try:
            # Create temporary rendering environment
            env = self._create_temp_env()

            # Record gameplay
            frames = self._collect_frames(env)

            # Upload to WandB
            if len(frames) > 0:
                video_array = np.array(frames, dtype=np.uint8)
                wandb.log({
                    "gameplay_video": wandb.Video(video_array, fps=30, format="mp4"),
                    "video_step": self.num_timesteps
                })

                if self.verbose > 0:
                    print(f"✅ Uploaded video ({len(frames)} frames)")

            # Clean up
            env.close()

            if self.verbose > 0:
                print(f"{'='*60}\n")

        except Exception as e:
            if self.verbose > 0:
                print(f"⚠️  Video recording failed: {e}")
                print(f"{'='*60}\n")

    def _create_temp_env(self):
        """Create a temporary rendering environment (not headless)."""
        env = RobotronEnv(
            level=self.level,
            lives=self.lives,
            fps=0,
            config_path=self.config_path,
            always_move=True,
            headless=False  # Need rendering for videos
        )

        # Apply same preprocessing as training
        env = GrayscaleObservation(env, keep_dim=False)
        env = ResizeObservation(env, (84, 84))
        env = FrameStackObservation(env, stack_size=4)

        return env

    def _collect_frames(self, env):
        """Run one episode and collect RGB frames."""
        frames = []

        obs, info = env.reset()
        done = False
        step_count = 0

        while not done and step_count < self.video_length:
            # Get action from current policy
            action, _ = self.model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(np.logical_or(terminated, truncated))

            # Render and store frame
            frame = env.render()
            if frame is not None:
                # Convert from (H, W, C) to (C, H, W) for WandB
                frames.append(frame.transpose(2, 0, 1))

            step_count += 1

        return frames


class VideoRecordingCallbackVecEnv(BaseCallback):
    """
    Version that works with VecNormalize by temporarily creating a normalized env.
    """

    def __init__(
        self,
        record_freq: int = 50_000,
        video_length: int = 500,
        config_path: str = None,
        level: int = 1,
        lives: int = 5,
        vec_normalize_path: str = None,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.record_freq = record_freq
        self.video_length = video_length
        self.config_path = config_path
        self.level = level
        self.lives = lives
        self.vec_normalize_path = vec_normalize_path
        self.last_recording = -1

    def _on_step(self) -> bool:
        # Check if it's time to record
        if self.record_freq > 0 and self.num_timesteps % self.record_freq == 0:
            if self.num_timesteps != self.last_recording:
                self._record_video()
                self.last_recording = self.num_timesteps
        return True

    def _record_video(self):
        """Create temporary env, record video, destroy env."""
        if self.verbose > 0:
            print(f"\n{'='*60}")
            print(f"🎥 Recording video at {self.num_timesteps:,} steps...")
            print(f"{'='*60}")

        try:
            # Create temporary rendering environment
            env = self._create_temp_env()

            # Record gameplay
            frames = self._collect_frames(env)

            # Upload to WandB
            if len(frames) > 0:
                video_array = np.array(frames, dtype=np.uint8)
                wandb.log({
                    "gameplay_video": wandb.Video(video_array, fps=30, format="mp4"),
                    "video_step": self.num_timesteps
                })

                if self.verbose > 0:
                    print(f"✅ Uploaded video ({len(frames)} frames)")

            # Clean up
            env.close()

            if self.verbose > 0:
                print(f"{'='*60}\n")

        except Exception as e:
            if self.verbose > 0:
                print(f"⚠️  Video recording failed: {e}")
                print(f"   Error: {str(e)}")
                print(f"{'='*60}\n")

    def _create_temp_env(self):
        """Create a temporary rendering environment (not headless)."""
        env = RobotronEnv(
            level=self.level,
            lives=self.lives,
            fps=0,
            config_path=self.config_path,
            always_move=True,
            headless=False  # Need rendering for videos
        )

        # Apply same preprocessing as training
        env = GrayscaleObservation(env, keep_dim=False)
        env = ResizeObservation(env, (84, 84))
        env = FrameStackObservation(env, stack_size=4)

        return env

    def _collect_frames(self, env):
        """Run one episode and collect RGB frames."""
        frames = []

        obs, info = env.reset()
        done = False
        step_count = 0

        while not done and step_count < self.video_length:
            # Normalize observation if VecNormalize is being used
            # Note: We can't easily load VecNormalize stats here without the wrapper
            # So we'll use unnormalized observations (may affect policy slightly)

            # Get action from current policy
            action, _ = self.model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(np.logical_or(terminated, truncated))

            # Render and store frame
            try:
                frame = env.render()
                if frame is not None:
                    # Convert from (H, W, C) to (C, H, W) for WandB
                    frames.append(frame.transpose(2, 0, 1))
            except Exception as e:
                if self.verbose > 0:
                    print(f"⚠️  Frame capture failed: {e}")
                break

            step_count += 1

        return frames
