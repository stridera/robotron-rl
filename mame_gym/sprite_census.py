"""Sprite census: enumerate every state-word that appears in the slot pool
across waves, flagging unmapped ones. Unmapped SWs = entities INVISIBLE to the
policy (dropped from the 945-dim obs).

Runs the chain-head policy from both wave-1 and wave-5 starts to cover deep
waves. For each SW: occurrence count, example positions, waves seen in.

Usage: .venv/bin/python3 mame_gym/sprite_census.py <model.zip> <vecnorm.pkl>
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from mame_robotron_env import MameRobotronEnv
from mame_obs import parse_header, ENTITY_TYPES, _TYPE_MAP

MODEL = sys.argv[1]
VECNORM = sys.argv[2]
SLOT_BASE, STRIDE, NSLOT = 10, 24, 101


def census_packet(pkt, wave, stats):
    for i in range(NSLOT):
        off = SLOT_BASE + i * STRIDE
        sw = (pkt[off+4] << 8) | pkt[off+5]
        if sw == 0:
            continue
        s = stats[sw]
        s["count"] += 1
        s["waves"].add(wave)
        if len(s["examples"]) < 3:
            s["examples"].append((pkt[off], pkt[off+1], i))


def run_phase(env, venv, model, steps, stats):
    obs = venv.reset()
    for _ in range(steps):
        a, _ = model.predict(obs, deterministic=False)
        obs, _, dones, infos = venv.step(a)
        pkt = env._last_packet
        if pkt is not None:
            census_packet(pkt, infos[0].get("wave", 0), stats)
        if dones[0]:
            obs = venv.reset()


def main():
    stats = defaultdict(lambda: {"count": 0, "waves": set(), "examples": []})

    for pool, label, steps in [([0], "wave-1 starts", 4000),
                               ([1, 2, 3, 4, 5, 6, 7, 8], "wave-5 starts", 6000)]:
        env = MameRobotronEnv(rank=0, base_port=9995, frameskip=4, reset_pool=pool)
        venv = DummyVecEnv([lambda e=env: e])
        venv = VecNormalize.load(VECNORM, venv)
        venv.training = False
        venv.norm_reward = False
        model = PPO.load(MODEL, env=venv, device="cpu")
        print(f"--- phase: {label} ({steps} steps) ---", flush=True)
        run_phase(env, venv, model, steps, stats)
        venv.close()

    print(f"\n{'SW':>7s} {'name':<18s} {'mapped':<7s} {'count':>8s} {'waves':<16s} examples")
    unmapped = []
    for sw in sorted(stats):
        s = stats[sw]
        name = ENTITY_TYPES.get(sw, "???")
        mapped = "YES" if (name != "???" and _TYPE_MAP.get(name)) else ("excl" if name != "???" else "NO")
        waves = ",".join(str(w) for w in sorted(s["waves"]))
        print(f"${sw:04X}  {name:<18s} {mapped:<7s} {s['count']:>8d} {waves:<16s} {s['examples']}")
        if mapped == "NO":
            unmapped.append((sw, s["count"]))
    print(f"\nUNMAPPED state-words (invisible to policy): {[(hex(sw), c) for sw, c in unmapped]}")


if __name__ == "__main__":
    main()
