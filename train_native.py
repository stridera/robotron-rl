"""PPO training on the robotron_native gym (real Williams 6809 ROM).

Bridges the native gym to our chain-trained 945-dim category-slot policy via
the existing rl_bridge.PositionObsBuilder. Wraps each native env with:

  - SnapshotRotator: cycles through /tmp/cap_001..020_6809.bin on reset (5
    snapshots span waves 1-3).
  - PositionObsAdapter: projects native 300-byte slot pool → 945-dim obs.
  - ActionAdapter: MultiDiscrete([8,8]) → MultiDiscrete([9,9]) (+1 offset).
  - CorruptionGuard: detects wave-transition corruption (wave jump >1, lives
    >5, score delta >50k, player out of bounds) and terminates the episode
    cleanly, banking a wave-clear bonus if it looks like a legitimate clear.
  - NativeReward: score_delta/10 + survival + Option A step-function wave
    bonuses + death penalty (same shape as SimpleStrongRewardWrapper).

Why corruption guard: known limitation in robotron_native — unimplemented
$A3 SUBD-indexed opcode corrupts the wave-transition path when a competent
agent clears a wave through bullet kills. Until the emulator is fixed,
treat clear-or-corruption as episode-end and reset to a fresh snapshot.

Resume the chain: --bc-checkpoint models/bpu3nzbg/checkpoints/...3000000_steps.zip
"""
from __future__ import annotations
import argparse
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete

import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

# robotron_native + bridge
_NATIVE = Path(os.environ.get("ROBOTRON_NATIVE_ROOT", "/home/strider/Code/robotron_native"))
sys.path.insert(0, str(_NATIVE / "core"))

# ROBOTRON_NATIVE_SNAPSHOTS env var overrides snapshot rotation (comma-separated paths)
_env_snapshots = os.environ.get("ROBOTRON_NATIVE_SNAPSHOTS", "").strip()
if _env_snapshots:
    SNAPSHOTS = [s.strip() for s in _env_snapshots.split(",") if s.strip()]
else:
    SNAPSHOTS = sorted(str(p) for p in Path("/tmp").glob("cap_*_6809.bin"))
if not SNAPSHOTS:
    raise SystemExit("No /tmp/cap_*_6809.bin snapshots found")


# ── Action: MultiDiscrete([8,8]) (move,fire) with dir 0=N..7=NW (always-move)
#    → native MultiDiscrete([9,9]) with 0=none, 1=N..8=NW. Bridge: +1 each.

class NativeRobotronEnv(gym.Env):
    """Single-process native gym with snapshot rotation + 945-dim obs +
    MultiDiscrete([8,8]) action surface."""

    metadata = {"render_modes": []}

    def __init__(self, rank: int = 0, seed: int = 0, lives: int = 3):
        super().__init__()
        from gym_env import RobotronEnv
        from rl_bridge import PositionObsBuilder

        self._RobotronEnv = RobotronEnv
        self._inner: Optional[gym.Env] = None
        self._obs_builder = PositionObsBuilder()
        self._lives_override = lives
        self._rng = np.random.default_rng(seed + rank * 7919)

        # Match what the chain-trained policy expects.
        self.action_space = MultiDiscrete([8, 8])
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(945,), dtype=np.float32
        )

        # Corruption guard state
        self._last_wave = 0
        self._last_lives = 0
        self._last_score = 0
        self._last_grunt_count = 99
        self._last_spawner_count = 0
        self._last_shooter_count = 0
        self._last_brain_count = 0
        self._step_count = 0

    def _read_score(self, core) -> int:
        raw = core.read_range(0xBDE5, 3)
        return ((raw[0] >> 4) * 10 + (raw[0] & 0xF)) * 10000 + \
               ((raw[1] >> 4) * 10 + (raw[1] & 0xF)) * 100 + \
               ((raw[2] >> 4) * 10 + (raw[2] & 0xF))

    def _read_grunts(self, core) -> int:
        return core.read8(0xBE68)

    # Spawner SWs (Sphereoid, Quark) and shooter SWs (Enforcer, Tank).
    # Brains are wave-5+ exclusive enemies; tougher than shooters (cruise missile
    # + civilian capture). Bigger bonus to incentivize learning brain combat.
    _SPAWNER_SWS = (0x12C8, 0x4BC9)
    _SHOOTER_SWS = (0x1483, 0x4800)
    _BRAIN_SWS   = (0x1DD6, 0x2119)

    def _count_spawners_shooters(self, core) -> tuple[int, int, int]:
        buf = core.read_range(0x98D4, 101 * 24)
        spawner = shooter = brain = 0
        for i in range(101):
            base = i * 24
            sw = (buf[base + 4] << 8) | buf[base + 5]
            if sw in self._SPAWNER_SWS: spawner += 1
            elif sw in self._SHOOTER_SWS: shooter += 1
            elif sw in self._BRAIN_SWS: brain += 1
        return spawner, shooter, brain

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        snapshot = SNAPSHOTS[int(self._rng.integers(0, len(SNAPSHOTS)))]
        # Recreate the inner env each reset (fresh CPU + snapshot).
        if self._inner is not None:
            try:
                self._inner.close()
            except Exception:
                pass
        self._inner = self._RobotronEnv(snapshot_path=snapshot)
        _, info = self._inner.reset()
        core = self._inner._core
        # Set our preferred starting lives.
        core.write8(0xBDEC, self._lives_override)
        self._obs_builder.reset()
        self._last_wave = core.read8(0xBDED)
        self._last_lives = self._lives_override
        self._last_score = self._read_score(core)
        self._last_grunt_count = self._read_grunts(core)
        self._last_spawner_count, self._last_shooter_count, self._last_brain_count = self._count_spawners_shooters(core)
        self._step_count = 0
        obs = self._obs_builder(core).astype(np.float32)
        return obs, {"snapshot": snapshot, "wave": self._last_wave,
                     "score": self._last_score, "lives": self._last_lives}

    def step(self, action):
        assert self._inner is not None
        a = [int(action[0]) + 1, int(action[1]) + 1]   # (move,fire) → native (+1 each)
        _, _, term_inner, trunc_inner, _ = self._inner.step(a)
        self._step_count += 1
        core = self._inner._core
        wave = core.read8(0xBDED)
        lives = core.read8(0xBDEC)
        score = self._read_score(core)
        grunts = self._read_grunts(core)
        score_delta = score - self._last_score

        # ── Corruption guard ───────────────────────────────────────────────
        wave_jumped = (wave - self._last_wave) > 1 if wave >= self._last_wave else True
        lives_inflated = lives > 10   # 1-up bonuses can push lives to ~7-9 legitimately
        score_exploded = score_delta > 50_000
        oob = (core.read8(0x9864) > 200) or (core.read8(0x9866) > 250)
        wave_cleared_clean = (wave == self._last_wave + 1) and not wave_jumped \
                             and not lives_inflated and not score_exploded

        corrupted = wave_jumped or lives_inflated or score_exploded or oob

        # ── Reward: same shape family as SimpleStrongRewardWrapper ────────
        if corrupted:
            causes = []
            if wave_jumped: causes.append("wave_jumped")
            if lives_inflated: causes.append("lives_inflated")
            if score_exploded: causes.append("score_exploded")
            if oob: causes.append("oob")
            # Treat as best-effort wave clear (the corruption fires AT the
            # transition, so the player almost certainly just cleared a wave).
            reward = self._last_wave * 250.0
            for crossed in range(self._last_wave + 1, self._last_wave + 2):
                if crossed >= 10: reward += 8000.0
                elif crossed >= 7: reward += 3000.0
                elif crossed >= 5: reward += 1000.0
            terminated = True
            truncated = False
            info = {
                "score": self._last_score, "lives": self._last_lives,
                "wave": self._last_wave, "wave_cleared": True,
                "corrupted": True, "corruption_cause": ",".join(causes),
                "corruption_wave_jumped": int(wave_jumped),
                "corruption_lives_inflated": int(lives_inflated),
                "corruption_score_exploded": int(score_exploded),
                "corruption_oob": int(oob),
                "corruption_pre_wave": self._last_wave,
                "corruption_post_wave": int(wave),
                "corruption_pre_lives": self._last_lives,
                "corruption_post_lives": int(lives),
                "corruption_score_delta": int(score_delta),
                "snapshot_steps": self._step_count,
            }
            return np.zeros(945, dtype=np.float32), reward, terminated, truncated, info

        # Normal step
        reward = score_delta / 10.0
        if lives < self._last_lives:
            reward += -20.0
        else:
            reward += 0.3 * max(1, int(wave))
        if wave > self._last_wave:
            reward += 250.0 * wave
            for crossed in range(self._last_wave + 1, wave + 1):
                if crossed >= 10: reward += 8000.0
                elif crossed >= 7: reward += 3000.0
                elif crossed >= 5: reward += 1000.0

        # Spawner / shooter kill bonuses (port from python SimpleStrongRewardWrapper).
        # Only credit a drop when score also moved up — guards against snapshot-rollover
        # noise. Magnitudes 50 / 20 match the python wrapper, scaled to native's
        # score-on-10 base (~5x kill point value).
        spawner_count, shooter_count, brain_count = self._count_spawners_shooters(core)
        if score_delta > 0:
            killed_spawners = max(0, self._last_spawner_count - spawner_count)
            killed_shooters = max(0, self._last_shooter_count - shooter_count)
            killed_brains   = max(0, self._last_brain_count   - brain_count)
            reward += 50.0 * killed_spawners + 20.0 * killed_shooters + 100.0 * killed_brains

        terminated = term_inner or (lives == 0)
        truncated = trunc_inner

        self._last_wave = wave
        self._last_lives = lives
        self._last_score = score
        self._last_grunt_count = grunts
        self._last_spawner_count = spawner_count
        self._last_shooter_count = shooter_count
        self._last_brain_count = brain_count

        obs = self._obs_builder(core).astype(np.float32)
        info = {"score": score, "lives": lives, "wave": wave,
                "level": wave, "wave_cleared": False, "corrupted": False}
        return obs, reward, terminated, truncated, info

    def close(self):
        if self._inner is not None:
            try:
                self._inner.close()
            except Exception:
                pass
        self._inner = None


# ── SB3 callbacks ────────────────────────────────────────────────────────

class NativeMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.highest_score = 0
        self.highest_wave = 0
        self.corruption_count = 0
        self.wave_clear_count = 0
        self.death_count = 0
        self.corrupt_wave_jumped = 0
        self.corrupt_lives_inflated = 0
        self.corrupt_score_exploded = 0
        self.corrupt_oob = 0
        self.last_corruption_log: dict = {}

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                score = info.get("score", 0)
                wave = info.get("wave", 0)
                self.highest_score = max(self.highest_score, score)
                self.highest_wave = max(self.highest_wave, wave)
                if info.get("corrupted"):
                    self.corruption_count += 1
                    self.corrupt_wave_jumped += info.get("corruption_wave_jumped", 0)
                    self.corrupt_lives_inflated += info.get("corruption_lives_inflated", 0)
                    self.corrupt_score_exploded += info.get("corruption_score_exploded", 0)
                    self.corrupt_oob += info.get("corruption_oob", 0)
                    self.last_corruption_log = {
                        "step": int(self.num_timesteps),
                        "cause": info.get("corruption_cause", ""),
                        "pre_wave": info.get("corruption_pre_wave"),
                        "post_wave": info.get("corruption_post_wave"),
                        "pre_lives": info.get("corruption_pre_lives"),
                        "post_lives": info.get("corruption_post_lives"),
                        "score_delta": info.get("corruption_score_delta"),
                        "snapshot_steps": info.get("snapshot_steps"),
                    }
                    print(f"[CORRUPTION] step={self.num_timesteps} "
                          f"cause={info.get('corruption_cause')} "
                          f"pre_wave={info.get('corruption_pre_wave')} "
                          f"post_wave={info.get('corruption_post_wave')} "
                          f"pre_lives={info.get('corruption_pre_lives')} "
                          f"post_lives={info.get('corruption_post_lives')} "
                          f"score_delta={info.get('corruption_score_delta')} "
                          f"snapshot_steps={info.get('snapshot_steps')}", flush=True)
                if info.get("wave_cleared"):
                    self.wave_clear_count += 1
                if info.get("lives", 1) == 0:
                    self.death_count += 1
                self.logger.record("robotron/episode_score", score)
                self.logger.record("robotron/episode_level", wave)
                self.logger.record("robotron/highest_score", self.highest_score)
                self.logger.record("robotron/highest_wave", self.highest_wave)
                self.logger.record("robotron/corrupted_eps", self.corruption_count)
                self.logger.record("robotron/wave_clear_eps", self.wave_clear_count)
                self.logger.record("robotron/death_eps", self.death_count)
                self.logger.record("robotron/corrupt_wave_jumped", self.corrupt_wave_jumped)
                self.logger.record("robotron/corrupt_lives_inflated", self.corrupt_lives_inflated)
                self.logger.record("robotron/corrupt_score_exploded", self.corrupt_score_exploded)
                self.logger.record("robotron/corrupt_oob", self.corrupt_oob)
        return True


class VecNormSaver(BaseCallback):
    def __init__(self, vec, path, freq=100_000):
        super().__init__()
        self.vec, self.path, self.freq = vec, path, freq
    def _on_step(self):
        if self.num_timesteps % self.freq < self.locals.get("n_steps", 2048):
            self.vec.save(self.path)
        return True


# ── Env factory ──────────────────────────────────────────────────────────

def make_env(rank: int, seed: int, lives: int) -> Callable[[], gym.Env]:
    def _init():
        env = NativeRobotronEnv(rank=rank, seed=seed, lives=lives)
        env = Monitor(env, info_keywords=("score", "wave", "lives",
                                          "wave_cleared", "corrupted"))
        return env
    return _init


# ── Main ─────────────────────────────────────────────────────────────────

def main(num_envs=8, total_timesteps=1_000_000, bc_checkpoint=None,
         vec_normalize=None, lives=3, lr=2e-4, clip_range=0.2, ent_coef=0.02,
         device="cpu", target_kl=None, gamma=0.995):

    fine_tuning = bc_checkpoint is not None
    config = {
        "model": "ppo", "total_timesteps": total_timesteps,
        "num_envs": num_envs, "obs_dim": 945, "gym": "robotron_native",
        "bc_checkpoint": bc_checkpoint, "fine_tuning": fine_tuning,
        "snapshots": [Path(p).name for p in SNAPSHOTS],
    }
    run = wandb.init(project="robotron", group="ppo_native_chain",
                     config=config, sync_tensorboard=True, save_code=True,
                     mode="offline")
    print("=" * 78)
    print("NATIVE-GYM PPO TRAINING (real 6809 ROM)")
    print("=" * 78)
    print(f"  Mode:        {'BC fine-tuning from ' + bc_checkpoint if fine_tuning else 'scratch'}")
    print(f"  Observation: 945-dim category-slot (bridge from native slot pool)")
    print(f"  Snapshots:   {len(SNAPSHOTS)} ({[Path(p).name for p in SNAPSHOTS]})")
    print(f"  Timesteps:   {total_timesteps:,}  Envs: {num_envs}  Lives: {lives}")
    print(f"  HPs:         lr={lr} clip={clip_range} ent_coef={ent_coef}")
    print(f"  Corruption guard: wave-jump>1, lives>5, score-delta>50k → terminate")
    print("=" * 78)

    envs = SubprocVecEnv([make_env(i, 42, lives) for i in range(num_envs)])
    envs = VecNormalize(envs, norm_obs=True, norm_reward=False, clip_obs=10.)
    if vec_normalize:
        envs = VecNormalize.load(vec_normalize, envs.venv)
        envs.training = True
        envs.norm_reward = False

    model_kwargs = dict(
        policy="MlpPolicy", n_steps=2048, batch_size=128, n_epochs=10,
        gamma=gamma, gae_lambda=0.95, clip_range=clip_range,
        ent_coef=ent_coef, vf_coef=0.5, max_grad_norm=0.5,
        learning_rate=lr, policy_kwargs={"net_arch": [512, 512]},
        target_kl=target_kl,
    )
    if fine_tuning:
        model = PPO.load(bc_checkpoint, env=envs, device=device)
        # Apply HP overrides
        model.learning_rate = lr
        model.clip_range = lambda _: clip_range
        model.ent_coef = ent_coef
        model.target_kl = target_kl
        model.gamma = gamma
    else:
        model = PPO(env=envs, device=device, verbose=1,
                    tensorboard_log=f"runs/{run.id}", **model_kwargs)

    out_dir = Path(f"models/{run.id}")
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    callbacks = [
        NativeMetricsCallback(),
        VecNormSaver(envs, str(out_dir / "vec_normalize.pkl"), freq=100_000),
        CheckpointCallback(save_freq=max(1, 100_000 // num_envs),
                           save_path=str(out_dir / "checkpoints"),
                           name_prefix="ppo_native_checkpoint"),
        WandbCallback(verbose=0),
    ]

    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks,
                    log_interval=4, progress_bar=False)
    finally:
        model.save(str(out_dir / "final_model"))
        envs.save(str(out_dir / "vec_normalize.pkl"))
        model.save(str(out_dir / "model"))
        run.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bc-checkpoint", type=str, default=None)
    p.add_argument("--vec-normalize", type=str, default=None)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--lives", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.02)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--target-kl", type=float, default=None,
                   help="PPO early-stop epochs when KL exceeds this (default: off)")
    p.add_argument("--gamma", type=float, default=0.995,
                   help="PPO discount factor (default 0.995)")
    args = p.parse_args()
    main(
        num_envs=args.num_envs, total_timesteps=args.timesteps,
        bc_checkpoint=args.bc_checkpoint, vec_normalize=args.vec_normalize,
        lives=args.lives, lr=args.lr, clip_range=args.clip_range,
        ent_coef=args.ent_coef, device=args.device,
        target_kl=args.target_kl, gamma=args.gamma,
    )
