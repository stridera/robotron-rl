"""PPO training on the MAME Robotron gym (faithful Williams emulation).

Same recipe family as train_native.py, but on mame_gym's MameRobotronEnv:
  - 945-dim category-slot obs (same extractor as native/python gyms)
  - MultiDiscrete([8,8]) actions
  - Reward: score_delta/10 + survival + Option A wave bonuses + death penalty
    + spawner/shooter/brain kill bonuses
  - No corruption guard needed (MAME is faithful; no wave-transition bug)

Fresh training is the default: native-gym checkpoints learned non-physical
player dynamics (player 3.5x too slow vs enemies) and score below random on
faithful hardware. --bc-checkpoint exists for warmstart experiments anyway.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "mame_gym"))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch as th
import torch.nn as nn
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from mame_robotron_env import MameRobotronEnv


class GridCNN(BaseFeaturesExtractor):
    """Small CNN for the (11,36,24) spatial-grid obs. NatureCNN's strides
    collapse a 36x24 input to nothing, so use 3x3 convs with modest downsampling
    to keep the global field layout legible to the policy."""
    def __init__(self, observation_space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_in = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_in, 32, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),   # 36x24 -> 18x12
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),   # -> 9x6
            nn.Flatten(),
        )
        with th.no_grad():
            n_flat = self.cnn(th.zeros(1, *observation_space.shape)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())

    def forward(self, obs):
        return self.linear(self.cnn(obs))


class MameMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.highest_score = 0
        self.highest_wave = 0
        self.death_count = 0

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                score = info.get("score", 0)
                wave = info.get("wave", 0)
                self.highest_score = max(self.highest_score, score)
                self.highest_wave = max(self.highest_wave, wave)
                if info.get("lives", 1) == 0:
                    self.death_count += 1
                self.logger.record("robotron/episode_score", score)
                self.logger.record("robotron/episode_level", wave)
                self.logger.record("robotron/highest_score", self.highest_score)
                self.logger.record("robotron/highest_wave", self.highest_wave)
                self.logger.record("robotron/death_eps", self.death_count)
        return True


class VecNormSaver(BaseCallback):
    def __init__(self, vec, path, freq=100_000):
        super().__init__()
        self.vec, self.path, self.freq = vec, path, freq
    def _on_step(self):
        if self.num_timesteps % self.freq < self.locals.get("n_steps", 2048):
            self.vec.save(self.path)
        return True


def make_env(rank: int, base_port: int, frameskip: int, reset_pool=None,
             auto_capture_min_wave=None, obs_mode="slot"):
    def _init():
        env = MameRobotronEnv(rank=rank, base_port=base_port, frameskip=frameskip,
                              reset_pool=reset_pool,
                              auto_capture_min_wave=auto_capture_min_wave,
                              obs_mode=obs_mode)
        return Monitor(env, info_keywords=("score", "wave", "lives"))
    return _init


def main(num_envs=8, total_timesteps=3_000_000, bc_checkpoint=None,
         vec_normalize=None, lr=3e-4, clip_range=0.2, ent_coef=0.01,
         gamma=0.999, device="cpu", base_port=9800, frameskip=4,
         target_kl=None, reset_pool=None, auto_capture_min_wave=None,
         obs_mode="slot"):

    fine_tuning = bc_checkpoint is not None
    run = wandb.init(project="robotron", group="ppo_mame_chain",
                     config={"model": "ppo", "total_timesteps": total_timesteps,
                             "num_envs": num_envs, "obs_dim": 945, "gym": "mame",
                             "bc_checkpoint": bc_checkpoint, "frameskip": frameskip},
                     sync_tensorboard=True, save_code=True, mode="offline")
    print("=" * 78)
    print("MAME-GYM PPO TRAINING (faithful Williams emulation)")
    print("=" * 78)
    print(f"  Mode:      {'fine-tune from ' + bc_checkpoint if fine_tuning else 'FRESH (native checkpoints have non-physical dynamics)'}")
    print(f"  Timesteps: {total_timesteps:,}  Envs: {num_envs}  Frameskip: {frameskip}")
    print(f"  HPs:       lr={lr} clip={clip_range} ent_coef={ent_coef} gamma={gamma}")
    print("=" * 78, flush=True)

    envs = SubprocVecEnv([make_env(i, base_port, frameskip, reset_pool,
                                   auto_capture_min_wave, obs_mode) for i in range(num_envs)])
    # Grid obs is already well-scaled (counts + clipped velocity); skip obs
    # normalization (per-element stats on a sparse grid amplify noise). Slot obs
    # keeps the running normalizer.
    envs = VecNormalize(envs, norm_obs=(obs_mode != "grid"), norm_reward=False, clip_obs=10.)
    if vec_normalize:
        envs = VecNormalize.load(vec_normalize, envs.venv)
        envs.training = True
        envs.norm_reward = False

    if fine_tuning:
        model = PPO.load(bc_checkpoint, env=envs, device=device)
        model.learning_rate = lr
        model.clip_range = lambda _: clip_range
        model.ent_coef = ent_coef
        model.gamma = gamma
        model.target_kl = target_kl
    elif obs_mode == "grid":
        model = PPO(policy="MlpPolicy", env=envs, device=device, verbose=1,
                    n_steps=2048, batch_size=128, n_epochs=10,
                    gamma=gamma, gae_lambda=0.95, clip_range=clip_range,
                    ent_coef=ent_coef, vf_coef=0.5, max_grad_norm=0.5,
                    learning_rate=lr,
                    policy_kwargs={"features_extractor_class": GridCNN,
                                   "features_extractor_kwargs": {"features_dim": 512},
                                   "net_arch": [256]},
                    target_kl=target_kl,
                    tensorboard_log=f"runs/{run.id}")
    else:
        model = PPO(policy="MlpPolicy", env=envs, device=device, verbose=1,
                    n_steps=2048, batch_size=128, n_epochs=10,
                    gamma=gamma, gae_lambda=0.95, clip_range=clip_range,
                    ent_coef=ent_coef, vf_coef=0.5, max_grad_norm=0.5,
                    learning_rate=lr, policy_kwargs={"net_arch": [512, 512]},
                    target_kl=target_kl,
                    tensorboard_log=f"runs/{run.id}")

    out_dir = Path(f"models/{run.id}")
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    callbacks = [
        MameMetricsCallback(),
        VecNormSaver(envs, str(out_dir / "vec_normalize.pkl"), freq=100_000),
        CheckpointCallback(save_freq=max(1, 100_000 // num_envs),
                           save_path=str(out_dir / "checkpoints"),
                           name_prefix="ppo_mame_checkpoint"),
        WandbCallback(verbose=0),
    ]

    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks,
                    log_interval=4, progress_bar=False)
    finally:
        model.save(str(out_dir / "final_model"))
        envs.save(str(out_dir / "vec_normalize.pkl"))
        model.save(str(out_dir / "model"))
        envs.close()
        run.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bc-checkpoint", type=str, default=None)
    p.add_argument("--vec-normalize", type=str, default=None)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--timesteps", type=int, default=3_000_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--base-port", type=int, default=9800)
    p.add_argument("--frameskip", type=int, default=4)
    p.add_argument("--target-kl", type=float, default=None)
    p.add_argument("--reset-pool", type=str, default=None,
                   help="comma-separated state indices for episode starts, e.g. '0,0,1,2' (0=wave-1 boot, N=w5_N)")
    p.add_argument("--auto-capture-min-wave", type=int, default=None,
                   help="save a reset state whenever a training env enters a wave >= this (harvest into the next link's pool)")
    p.add_argument("--obs-mode", type=str, default="slot", choices=["slot", "grid"],
                   help="'slot'=945-dim MLP obs; 'grid'=(11,36,24) spatial CNN obs")
    args = p.parse_args()
    pool = [int(x) for x in args.reset_pool.split(",")] if args.reset_pool else None
    main(num_envs=args.num_envs, total_timesteps=args.timesteps,
         bc_checkpoint=args.bc_checkpoint, vec_normalize=args.vec_normalize,
         lr=args.lr, clip_range=args.clip_range, ent_coef=args.ent_coef,
         gamma=args.gamma, device=args.device, base_port=args.base_port,
         obs_mode=args.obs_mode,
         frameskip=args.frameskip, target_kl=args.target_kl, reset_pool=pool,
         auto_capture_min_wave=args.auto_capture_min_wave)
