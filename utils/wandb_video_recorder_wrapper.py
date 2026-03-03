import os
from typing import Callable
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvWrapper
import wandb


class WandBVideoRecorderWrapper(VecEnvWrapper):
    def __init__(self, venv: VecEnv, record_video_trigger: Callable[[int], bool], video_length: int = 200, use_obs: bool = False):
        VecEnvWrapper.__init__(self, venv)

        self.env = venv
        self.use_obs = use_obs
        self.record_video_trigger = record_video_trigger

        self.step_id = 0
        self.recording = False
        self.recorded_frames = 0
        self.video_length = video_length
        self.frames = None

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        self.start_video_recorder(obs)
        return obs

    def start_video_recorder(self, obs) -> None:
        self.close_video_recorder()
        self.recording = True
        self.save_frame(obs)

    def _video_enabled(self) -> bool:
        return self.record_video_trigger(self.step_id)

    def save_frame(self, obs) -> None:
        if self.recording:
            if self.frames is None:
                self.frames = []

            if self.use_obs:
                self.frames.append(obs[0].transpose(2, 0, 1))
            else:
                # RobotronEnv.render() returns RGB array directly (no mode argument)
                frame = self.env.render()
                if frame is not None:
                    # Handle both single env and vectorized env
                    if isinstance(frame, list):
                        frame = frame[0]  # Take first env
                    self.frames.append(frame.transpose(2, 0, 1))

            self.recorded_frames += 1

    def step_wait(self) -> VecEnvStepReturn:
        obs, rews, dones, infos = self.venv.step_wait()
        self.step_id += 1
        if self.recording:
            self.save_frame(obs)

            if self.recorded_frames > self.video_length:
                frames = np.array(self.frames, dtype=np.uint8)
                wandb.log({"video": wandb.Video(frames, fps=30)})
                self.close_video_recorder()
        elif self._video_enabled():
            self.start_video_recorder(obs)

        return obs, rews, dones, infos

    def close_video_recorder(self) -> None:
        self.frames = None
        self.recording = False
        self.recorded_frames = 0

    def close(self) -> None:
        VecEnvWrapper.close(self)
        self.close_video_recorder()

    def __del__(self):
        self.close()
