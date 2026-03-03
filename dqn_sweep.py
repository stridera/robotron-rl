import argparse

from robotron import RobotronEnv
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import QRDQN
from utils import WandBVideoRecorderWrapper
from wandb.integration.sb3 import WandbCallback
import wandb


def main(args):
    device = "cuda:0"
    total_timesteps = 5_000_000
    env_config = {
        "config_path": "game_config.yaml",
        "level": 2,
        "lives": 0,
        "always_move": True,
    }

    run = wandb.init(
        project="robotron",
        config=args,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )
    config = wandb.config

    run.log({'env_config': env_config})
    run.log_code()
    run.log_code(name="game_config", include_fn=lambda x: x.endswith(".yaml"))

    env = RobotronEnv(**env_config)
    env = GrayscaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, (123, 166))
    env = Monitor(env, info_keywords=('score', 'level'))
    env = DummyVecEnv([lambda: env])
    env = WandBVideoRecorderWrapper(env, record_video_trigger=lambda x: x % 2000 == 0, video_length=200)
    env = VecFrameStack(env, 4, channels_order='first')

    env.reset()
    model = QRDQN(env=env, verbose=1, tensorboard_log=f"runs/{run.id}", device=device, **config)
    model.learn(
        total_timesteps=total_timesteps,
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_freq=500_000,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
