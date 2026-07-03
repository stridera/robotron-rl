"""OBS-GAP test: run the good FSM (chooseOutputs) on MAME, but feed it ONLY the
entities the 945-dim obs actually conveys — i.e. the per-category-nearest-N + 12
catch-all selection (max 41 slots), exactly as MameObsBuilder/_extract_features slots
them. If the FSM still hits ~16, the obs is sufficient and RL's failure is training.
If it collapses toward ~4, the obs TRUNCATION is the bottleneck (RL is blind to most
of the swarm). Compare against fsm_choose_on_mame.py (full info = wave 16)."""
import sys
import statistics
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from mame_robotron_env import MameRobotronEnv
from mame_obs import MameObsBuilder
from position_wrapper import SLOT_CATEGORIES, CATCHALL_SLOTS
import robotron_fsm as fsm

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 9982
FULL = "full" in sys.argv[3:]   # control: feed ALL sprites (should reproduce wave 16)

W, H = 665, 492
fsm.DEBUG_LEVEL = 0
fsm.MAX_RIGHT, fsm.MAX_TOP = W, H
fsm.MAX_BOTTOM, fsm.MAX_LEFT = 0, 0
fsm.Y_AXIS_INVERSION = H
B = 20
fsm.ADJ_TOP, fsm.ADJ_BOTTOM, fsm.ADJ_LEFT, fsm.ADJ_RIGHT = H - 2, 0 + B + 9, 0 + 2, W - B

builder = MameObsBuilder()


def obs_limited_data(packet):
    """Replicate the 945-dim obs's entity selection, return chooseOutputs data."""
    sprites = builder._sprites_from_packet(packet)        # [(x,y,'Player'), (x,y,name,vx,vy)...]
    player = next((s for s in sprites if s[2] == "Player"), None)
    if player is None:
        return []
    px, py = player[0], player[1]
    others = [s for s in sprites if s[2] != "Player"]
    if FULL:
        return [(px, py, "Player")] + [(s[0], s[1], s[2]) for s in others]

    def d2(s):
        return (s[0] - px) ** 2 + (s[1] - py) ** 2
    used, selected = set(), []
    for n, types in SLOT_CATEGORIES:
        cands = sorted([s for s in others if s[2] in types and id(s) not in used], key=d2)[:n]
        for s in cands:
            used.add(id(s)); selected.append(s)
    remaining = sorted([s for s in others if id(s) not in used], key=d2)[:CATCHALL_SLOTS]
    selected += remaining
    return [(px, py, "Player")] + [(s[0], s[1], s[2]) for s in selected]


def main():
    env = MameRobotronEnv(rank=0, base_port=PORT, frameskip=4, reset_pool=[0], obs_mode="slot")
    results = []
    for ep in range(N):
        env.reset()
        packet = env._last_packet
        maxw = env._last_wave
        steps = 0
        while True:
            try:
                mv, fr = fsm.chooseOutputs(obs_limited_data(packet))
            except Exception:
                mv, fr = 1, 1
            mi = (mv - 1) if mv >= 1 else 0
            fi = (fr - 1) if fr >= 1 else mi
            _, _, t, tr, info = env.step(np.array([mi, fi]))
            packet = env._last_packet
            maxw = max(maxw, info.get("wave", 0))
            steps += 1
            if t or tr:
                break
        results.append(maxw)
        print(f"ep {ep+1}: wave {maxw}  ({steps} steps)", flush=True)
    label = "FULL sprites (control)" if FULL else "OBS-LIMITED (41-slot truncation)"
    print(f"\n=== {N} chooseOutputs runs on MAME, {label} ===")
    print(f"wave: min={min(results)} max={max(results)} mean={statistics.mean(results):.1f} "
          f"median={statistics.median(results)}  dist={dict(sorted(Counter(results).items()))}")
    env.close()


if __name__ == "__main__":
    main()
