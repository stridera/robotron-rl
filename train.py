"""
Train using Stable Baselines3
"""
import argparse

from robotron import RobotronEnv
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from sb3_contrib import QRDQN
from utils import WandBVideoRecorderWrapper
from wandb.integration.sb3 import WandbCallback
import wandb


def main(model_name: str, config_path: str = None, resume_path: str = None, project: str = None, group: str = None, device: str = 'cuda:0'):
    config = {
        'model': model_name,
        "env_name": "robotron",
        'resume_path': resume_path,
        "total_timesteps": 55_500_000,

        'env': {
            'config_path': config_path,
            "level": 1,
            "lives": 0,
            "fps": 0,
            "always_move": True,
        },
    }
    if model_name == "ppo":
        model_class = PPO
        config['model_kwargs'] = {}
    elif model_name == 'qrdqn':
        model_class = QRDQN
        config['model_kwargs'] = {
            "policy": "CnnPolicy",
            "learning_rate": 0.00025,
            "gamma": 0.99,
            "batch_size": 32,
            "train_freq": 4,
            "target_update_interval": 10_000,
            "learning_starts": 200_000,
            "buffer_size": 500_000,
            "max_grad_norm": 10,
            "exploration_fraction": 0.1,
            "exploration_final_eps": 0.01,
        }
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    run = wandb.init(
        project=project or "robotron",
        group=group or f"{model_name}_test",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    run.log_code(name="game_config", include_fn=lambda x: x.endswith(".yaml"))

    env = RobotronEnv(**config['env'])
    env = GrayscaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, (123, 166))
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = WandBVideoRecorderWrapper(env, record_video_trigger=lambda x: x % 2000 == 0, video_length=200)
    env = VecFrameStack(env, 4, channels_order='first')

    env.reset()

    if resume_path:
        model = model_class.load(path=resume_path, env=env, verbose=1, device=device,
                                 tensorboard_log=f"runs/{run.id}", **config['model_kwargs'])
    else:
        model = model_class(env=env, verbose=1,
                            tensorboard_log=f"runs/{run.id}", device=device, **config['model_kwargs'])

    model.learn(
        total_timesteps=config["total_timesteps"],
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
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--device", type=str, default='cuda:0')
    args = parser.parse_args()
    main(args.model, args.config, args.resume, args.project, args.group, args.device)
