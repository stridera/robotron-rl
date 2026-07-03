"""collect_fsm_demos.py — generate behavior-cloning demos from the FSM teacher.

Runs the obs-limited chooseOutputs FSM on the MAME env (the SAME env + 945-dim obs the
PPO policy uses) and records (obs, action) pairs. The FSM decides from only what the obs
conveys, so the policy can reproduce it. Diversity via a broad reset pool (wave-1 + deep
states) + epsilon-random exploration (DAgger-style: expert labels over a wide state set).

Output: demos/fsm_mame_demos.npz  with  obs (N,945) float32, actions (N,2) int32 [0-7].

Usage: .venv/bin/python3 mame_gym/collect_fsm_demos.py <target_pairs> <port> [epsilon]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from mame_robotron_env import MameRobotronEnv
from mame_obs import MameObsBuilder
from position_wrapper import SLOT_CATEGORIES, CATCHALL_SLOTS
import robotron_fsm as fsm

TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 9982
EPSILON = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10
import os as _os0
OUT = Path(_os0.environ.get("DEMO_OUT",
      str(Path(__file__).parent.parent / "demos" / "fsm_mame_demos.npz")))
# CHAMPION_DEMOS=1: label with the FULL champion v2 action (FSM + minimal-deviation
# clearance override, clearance_planner.py) instead of the bare FSM — plus whatever
# FSM_* flags are set (e.g. FSM_RESCUE_SEEK). This is the corrected-perception
# teacher re-pass (2026-07-02): all prior BC/anchored-RL demos were collected with
# broken entity labels (tanks invisible, shells-as-quarks).
CHAMPION = _os0.environ.get("CHAMPION_DEMOS", "0") == "1"
if CHAMPION:
    from clearance_planner import clearance_search as _clear

# FSM global setup (665x492 px board)
W, H = 665, 492
fsm.DEBUG_LEVEL = 0
fsm.MAX_RIGHT, fsm.MAX_TOP = W, H
fsm.MAX_BOTTOM, fsm.MAX_LEFT = 0, 0
fsm.Y_AXIS_INVERSION = H
# Optional: load an evolved FSM's best_params (e.g. fsm_evolved_reseed_v2.json) so the
# demos are labeled by the stronger teacher. Set EVOLVE_PARAMS=<path-to-json>.
import os, json as _json
_ep = os.environ.get("EVOLVE_PARAMS", "")
if _ep and Path(_ep).exists():
    for _n, _v in _json.loads(Path(_ep).read_text())["best_params"].items():
        setattr(fsm, _n, _v)
    print(f"loaded evolved FSM params from {_ep}", flush=True)
B = float(getattr(fsm, "BORDER_ADJUST", 20))
fsm.ADJ_TOP, fsm.ADJ_BOTTOM, fsm.ADJ_LEFT, fsm.ADJ_RIGHT = H - 2, 0 + B + 9, 0 + 2, W - B

builder = MameObsBuilder()
rng = np.random.default_rng(12345)


def fsm_action(packet):
    """obs-limited chooseOutputs -> (move_idx, fire_idx) in 0-7."""
    sprites = builder._sprites_from_packet(packet)
    player = next((s for s in sprites if s[2] == "Player"), None)
    if player is None:
        return 0, 0
    px, py = player[0], player[1]
    others = [s for s in sprites if s[2] != "Player"]

    def d2(s):
        return (s[0] - px) ** 2 + (s[1] - py) ** 2
    used, sel = set(), []
    for n, types in SLOT_CATEGORIES:
        for s in sorted([s for s in others if s[2] in types and id(s) not in used], key=d2)[:n]:
            used.add(id(s)); sel.append(s)
    sel += sorted([s for s in others if id(s) not in used], key=d2)[:CATCHALL_SLOTS]
    data = [(px, py, "Player")] + [(s[0], s[1], s[2]) for s in sel]
    try:
        mv, fr = fsm.chooseOutputs(data)
    except Exception:
        mv, fr = 1, 1
    mv = mv if mv >= 1 else 1
    fr = fr if fr >= 1 else mv
    if CHAMPION:
        mv, fr = _clear(sprites, mv, fr)
    return mv - 1, fr - 1


def main():
    pool = [int(x) for x in (Path(__file__).parent / "bc_collect_pool.txt").read_text().split(",")]
    env = MameRobotronEnv(rank=0, base_port=PORT, frameskip=4, reset_pool=pool, obs_mode="slot")
    obs_buf = np.zeros((TARGET, 945), dtype=np.float32)
    act_buf = np.zeros((TARGET, 2), dtype=np.int32)
    n = 0
    t0 = time.time()
    obs, _ = env.reset()
    packet = env._last_packet
    waves_seen = []
    while n < TARGET:
        mi, fi = fsm_action(packet)        # expert label (always recorded)
        obs_buf[n] = obs
        act_buf[n] = (mi, fi)
        n += 1
        if rng.random() < EPSILON:          # DAgger-style exploration of the step taken
            step_a = np.array([rng.integers(0, 8), rng.integers(0, 8)])
        else:
            step_a = np.array([mi, fi])
        obs, _, term, trunc, info = env.step(step_a)
        packet = env._last_packet
        if term or trunc:
            waves_seen.append(info.get("wave", 0))
            obs, _ = env.reset()
            packet = env._last_packet
        if n % 20000 == 0:
            print(f"  {n}/{TARGET}  ({n/(time.time()-t0):.0f}/s, {len(waves_seen)} eps)", flush=True)
    env.close()
    OUT.parent.mkdir(exist_ok=True)
    np.savez_compressed(OUT, obs=obs_buf, actions=act_buf)
    from collections import Counter
    am = Counter(int(a) for a in act_buf[:, 0])
    af = Counter(int(a) for a in act_buf[:, 1])
    print(f"\nsaved {OUT}: {n} pairs, {len(waves_seen)} episodes")
    print(f"move-dir hist: {dict(sorted(am.items()))}")
    print(f"fire-dir hist: {dict(sorted(af.items()))}")


if __name__ == "__main__":
    main()
