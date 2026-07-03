"""validate_fsm.py — play N continuous wave-1 MAME games with a given FSM config, report wave dist.

Used to validate an evolved FSM (models/fsm_evolved_<tag>.json best_params) over MANY games vs the
hand-tuned baseline (mean ~11.4). Pass 'default' to measure the baseline with the same protocol.

Usage: .venv/bin/python3 validate_fsm.py <fsm_evolved_*.json | default> [n_games] [port]
"""
import sys, json, statistics
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "mame_gym"))
import numpy as np
from mame_robotron_env import MameRobotronEnv
from mame_obs import MameObsBuilder
from position_wrapper import SLOT_CATEGORIES, CATCHALL_SLOTS
import robotron_fsm as fsm

SRC = sys.argv[1] if len(sys.argv) > 1 else "default"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
PORT = int(sys.argv[3]) if len(sys.argv) > 3 else 9968
STEPCAP = 8000

W, H = 665, 492
fsm.DEBUG_LEVEL = 0
fsm.MAX_RIGHT, fsm.MAX_TOP = W, H
fsm.MAX_BOTTOM, fsm.MAX_LEFT = 0, 0
fsm.Y_AXIS_INVERSION = H

if SRC != "default":
    params = json.loads(Path(SRC).read_text())["best_params"]
    for name, val in params.items():
        setattr(fsm, name, val)
    print(f"loaded {len(params)} evolved params from {SRC}", flush=True)
else:
    print("using DEFAULT (hand-tuned) FSM params", flush=True)
B = float(getattr(fsm, "BORDER_ADJUST", 20))
fsm.ADJ_TOP, fsm.ADJ_BOTTOM = H - 2, 0 + B + 9
fsm.ADJ_LEFT, fsm.ADJ_RIGHT = 0 + 2, W - B

builder = MameObsBuilder()


def fsm_action(packet):
    sprites = builder._sprites_from_packet(packet)
    player = next((s for s in sprites if s[2] == "Player"), None)
    if player is None:
        return 0, 0
    px, py = player[0], player[1]
    others = [s for s in sprites if s[2] != "Player"]
    def d2(s):
        return (s[0] - px) ** 2 + (s[1] - py) ** 2
    used, sel = set(), []
    for cnt, types in SLOT_CATEGORIES:
        for s in sorted([s for s in others if s[2] in types and id(s) not in used], key=d2)[:cnt]:
            used.add(id(s)); sel.append(s)
    sel += sorted([s for s in others if id(s) not in used], key=d2)[:CATCHALL_SLOTS]
    data = [(px, py, "Player")] + [(s[0], s[1], s[2]) for s in sel]
    try:
        mv, fr = fsm.chooseOutputs(data)
    except Exception:
        mv, fr = 1, 1
    mi = (mv - 1) if mv >= 1 else 0
    fi = (fr - 1) if fr >= 1 else mi
    return mi, fi


def main():
    env = MameRobotronEnv(rank=0, base_port=PORT, frameskip=4, reset_pool=[0], obs_mode="slot")
    waves, scores = [], []
    for g in range(N):
        env.reset(); packet = env._last_packet
        mw = env._last_wave; sc = env._last_score; steps = 0
        while steps < STEPCAP:
            mi, fi = fsm_action(packet)
            _, _, term, trunc, info = env.step(np.array([mi, fi]))
            packet = env._last_packet
            mw = max(mw, info.get("wave", mw))
            if not (term or trunc):
                sc = info.get("score", sc)
            steps += 1
            if term or trunc:
                break
        waves.append(mw); scores.append(sc)
        print(f"game {g+1}: wave={mw} score={sc} steps={steps}", flush=True)
    env.close()
    print(f"\n=== {N} games ({SRC}) ===")
    print(f"wave: min={min(waves)} max={max(waves)} mean={statistics.mean(waves):.2f} median={statistics.median(waves)}")
    print(f"score: max={max(scores)} mean={statistics.mean(scores):.0f}")
    print(f"dist: {dict(sorted(Counter(waves).items()))}")


if __name__ == "__main__":
    main()
