"""SLOT-SWEEP: measure how the obs-limited FSM ceiling rises as we widen the slot
budget. Both full-info (12.9) and obs-limited (10.4) use the SAME position-only
chooseOutputs — the only difference is entity TRUNCATION (41 slots). This sweeps
intermediate slot budgets to find how many slots recover the +2.5-wave gap, i.e. the
target obs dimension for a richer-obs rebuild.

Usage: MAME_RL_RESEED=1 .venv/bin/python mame_gym/fsm_slotsweep_on_mame.py [N] [PORT]
"""
import sys
import statistics
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from mame_robotron_env import MameRobotronEnv
from mame_obs import MameObsBuilder
import robotron_fsm as fsm

N = int(sys.argv[1]) if len(sys.argv) > 1 else 15
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 9984

W, H = 665, 492
fsm.DEBUG_LEVEL = 0
fsm.MAX_RIGHT, fsm.MAX_TOP = W, H
fsm.MAX_BOTTOM, fsm.MAX_LEFT = 0, 0
fsm.Y_AXIS_INVERSION = H
B = 20
fsm.ADJ_TOP, fsm.ADJ_BOTTOM, fsm.ADJ_LEFT, fsm.ADJ_RIGHT = H - 2, 0 + B + 9, 0 + 2, W - B

builder = MameObsBuilder()

# (label, category caps, catchall, total slots)
CONFIGS = [
    ("cur-41",  [(8, ['CruiseMissile', 'TankShell', 'EnforcerBullet']),
                 (4, ['Sphereoid', 'Quark']), (4, ['Tank', 'Enforcer']),
                 (3, ['Brain']), (6, ['Mommy', 'Daddy', 'Mikey']), (4, ['Hulk'])], 12),
    ("exp-79",  [(16, ['CruiseMissile', 'TankShell', 'EnforcerBullet']),
                 (4, ['Sphereoid', 'Quark']), (8, ['Tank', 'Enforcer']),
                 (5, ['Brain']), (6, ['Mommy', 'Daddy', 'Mikey']), (8, ['Hulk'])], 32),
    ("full",    None, None),   # all sprites
]


def select_data(packet, cats, catchall):
    sprites = builder._sprites_from_packet(packet)
    player = next((s for s in sprites if s[2] == "Player"), None)
    if player is None:
        return []
    px, py = player[0], player[1]
    others = [s for s in sprites if s[2] != "Player"]
    if cats is None:  # full
        return [(px, py, "Player")] + [(s[0], s[1], s[2]) for s in others]

    def d2(s):
        return (s[0] - px) ** 2 + (s[1] - py) ** 2
    used, sel = set(), []
    for n, types in cats:
        for s in sorted([s for s in others if s[2] in types and id(s) not in used], key=d2)[:n]:
            used.add(id(s)); sel.append(s)
    sel += sorted([s for s in others if id(s) not in used], key=d2)[:catchall]
    return [(px, py, "Player")] + [(s[0], s[1], s[2]) for s in sel]


def run_config(env, label, cats, catchall):
    results = []
    for ep in range(N):
        env.reset()
        packet = env._last_packet
        maxw = env._last_wave
        while True:
            try:
                mv, fr = fsm.chooseOutputs(select_data(packet, cats, catchall))
            except Exception:
                mv, fr = 1, 1
            mi = (mv - 1) if mv >= 1 else 0
            fi = (fr - 1) if fr >= 1 else mi
            _, _, t, tr, info = env.step(np.array([mi, fi]))
            packet = env._last_packet
            maxw = max(maxw, info.get("wave", 0))
            if t or tr:
                break
        results.append(maxw)
    tot = "all" if cats is None else (sum(n for n, _ in cats) + catchall)
    print(f"[{label}] slots={tot}  N={N}  mean={statistics.mean(results):.1f} "
          f"median={statistics.median(results)} max={max(results)} "
          f"dist={dict(sorted(Counter(results).items()))}", flush=True)
    return results


def main():
    env = MameRobotronEnv(rank=0, base_port=PORT, frameskip=4, reset_pool=[0], obs_mode="slot")
    for cfg in CONFIGS:
        run_config(env, *cfg)
    env.close()


if __name__ == "__main__":
    main()
