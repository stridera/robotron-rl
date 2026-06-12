"""Capture wave-5 full-machine save states by driving the chain-head policy.

Runs one MAME bridge; on entering wave 5, settles 120 frames, then CMD_SAVE.
States land in MAME's state dir as w5_1..w5_N and are loadable by any env
via reset_pool (MameRobotronEnv).

Usage: .venv/bin/python3 mame_gym/capture_wave5_states.py <model.zip> <vec_normalize.pkl> [n_states]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from mame_robotron_env import MameRobotronEnv
from mame_obs import parse_header

MODEL = sys.argv[1]
VECNORM = sys.argv[2]
N_STATES = int(sys.argv[3]) if len(sys.argv) > 3 else 8
TARGET_WAVE = int(sys.argv[4]) if len(sys.argv) > 4 else 5
START_IDX = int(sys.argv[5]) if len(sys.argv) > 5 else 0   # state index offset
RESET_POOL = [int(x) for x in sys.argv[6].split(",")] if len(sys.argv) > 6 else [0]
SETTLE_STEPS = 30   # 30 steps * 4 frameskip = 120 frames after wave entry


def main():
    env = MameRobotronEnv(rank=0, base_port=9990, frameskip=4, reset_pool=RESET_POOL)
    venv = DummyVecEnv([lambda: env])
    venv = VecNormalize.load(VECNORM, venv)
    venv.training = False
    venv.norm_reward = False
    model = PPO.load(MODEL, env=venv, device="cpu")

    saved = 0
    episodes = 0
    obs = venv.reset()
    settle_left = -1
    while saved < N_STATES and episodes < 300:
        action, _ = model.predict(obs, deterministic=False)
        obs, _, dones, infos = venv.step(action)
        info = infos[0]
        if dones[0]:
            episodes += 1
            settle_left = -1
            continue
        wave = info.get("wave", 0)
        if wave == TARGET_WAVE and settle_left < 0:
            settle_left = SETTLE_STEPS
        elif wave != TARGET_WAVE:
            settle_left = -1
        if settle_left > 0:
            settle_left -= 1
        elif settle_left == 0:
            saved += 1
            pkt = env._bridge.save_state(START_IDX + saved)
            h = parse_header(pkt)
            print(f"saved w5_{START_IDX + saved}: wave={h['wave']} score={h['score']} lives={h['lives']} (ep {episodes})", flush=True)
            settle_left = -1
    print(f"DONE: {saved}/{N_STATES} states in {episodes} episodes", flush=True)
    venv.close()


if __name__ == "__main__":
    main()
