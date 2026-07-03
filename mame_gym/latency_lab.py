"""latency_lab.py — measure + fix the Xenia perception-latency gap on MAME.

Context (2026-07-03): the hunt champion does W100+ on MAME (deaths/wave ~1.09) but
the SAME brain on Xenia stays W9-27 with deaths/wave ~1.7-1.8 in every era. The
Xenia agent's diagnosis: entity positions arrive through a ~250ms write-accumulator
(≈4 decision steps at 15Hz) while the player position is read fresh — the planner
dodges where threats WERE. YOLO input on real hardware will have its own latency,
so compensation is on the critical path of the end goal regardless.

This lab reproduces that exact failure mode on MAME and measures the fix:
  mode=base   : act on the current packet (control)
  mode=delay  : entities from packet t-K, player from packet t (Xenia's mix)
  mode=comp   : same as delay, but each entity is extrapolated forward K steps
                along its tracked per-step velocity (linear; the planner's chaser
                model handles re-aiming beyond that), positions clamped to board.

Usage:
  MAME_RL_RESEED=1 .venv/bin/python mame_gym/latency_lab.py <mode> <K> [N] [PORT] [STEPCAP]
Reports per game: max wave, deaths, score; aggregate mean wave + deaths/wave.
"""
import sys, os
from collections import deque
from pathlib import Path

os.environ.setdefault("FSM_RESCUE_SEEK", "1")     # champion recipe

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "mame_gym"))
import json
import numpy as np
from mame_robotron_env import MameRobotronEnv
from mame_obs import MameObsBuilder
from position_wrapper import SLOT_CATEGORIES, CATCHALL_SLOTS
import robotron_fsm as fsm
from clearance_planner import clearance_search

MODE   = sys.argv[1]                      # base | delay | comp
K      = int(sys.argv[2])                 # delay in decision steps (4 ≈ 267ms ≈ Xenia)
N      = int(sys.argv[3]) if len(sys.argv) > 3 else 20
PORT   = int(sys.argv[4]) if len(sys.argv) > 4 else 9960
STEPCAP = int(sys.argv[5]) if len(sys.argv) > 5 else 15000
PARAMS = os.environ.get("LAT_PARAMS", str(ROOT / "models/fsm_evolved_planner_v2_hunt.json"))
assert MODE in ("base", "delay", "comp")

W, H = 665, 492


def setup_fsm():
    fsm.DEBUG_LEVEL = 0
    fsm.MAX_RIGHT, fsm.MAX_TOP = W, H
    fsm.MAX_BOTTOM, fsm.MAX_LEFT = 0, 0
    fsm.Y_AXIS_INVERSION = H
    for nm, vl in json.loads(Path(PARAMS).read_text())["best_params"].items():
        setattr(fsm, nm, vl)
    B = float(getattr(fsm, "BORDER_ADJUST", 20))
    fsm.ADJ_TOP, fsm.ADJ_BOTTOM = H - 2, 0 + B + 9
    fsm.ADJ_LEFT, fsm.ADJ_RIGHT = 0 + 2, W - B


def act(sprites):
    """FSM + clearance planner on a prepared sprite list (eval_protocol actor body)."""
    player = next((s for s in sprites if s[2] == "Player"), None)
    if player is None:
        return np.array([0, 0])
    px, py = player[0], player[1]
    others = [s for s in sprites if s[2] != "Player"]
    def d2(s): return (s[0] - px) ** 2 + (s[1] - py) ** 2
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
    mv = mv if mv >= 1 else 1
    fr = fr if fr >= 1 else mv
    mv, fr = clearance_search(sprites, mv, fr)
    return np.array([mv - 1, fr - 1])


def perceived_sprites(builder, hist, cur_packet):
    """Build the sprite list the brain 'sees' under the configured latency model.
    hist[0] is the OLDEST buffered packet (t-K), hist[-1] == cur_packet."""
    if MODE == "base" or len(hist) <= 1:
        return builder._sprites_from_packet(cur_packet)
    stale = builder._sprites_from_packet(hist[0])   # feeds velocity tracker in order
    # fresh player position from the CURRENT header (Xenia reads it directly)
    from mame_obs import _gxgy_to_pixel
    fpx, fpy = _gxgy_to_pixel(cur_packet[5], cur_packet[7])
    out = [(fpx, fpy, "Player")]
    lag = len(hist) - 1                              # actual staleness in steps
    for s in stale:
        if s[2] == "Player":
            continue
        if MODE == "comp" and len(s) == 5:
            x = min(max(s[0] + s[3] * lag, 0.0), float(W))
            y = min(max(s[1] + s[4] * lag, 0.0), float(H))
            out.append((x, y, s[2], s[3], s[4]))
        else:
            out.append(s)
    return out


def main():
    setup_fsm()
    env = MameRobotronEnv(rank=0, base_port=PORT, frameskip=4, reset_pool=[0], obs_mode="slot")
    games = []
    for g in range(N):
        env.reset()
        builder = MameObsBuilder()                   # fresh velocity state per game
        hist = deque(maxlen=K + 1)
        hist.append(env._last_packet)
        info = {"wave": env._last_wave, "lives": env._last_lives, "score": env._last_score}
        start_lives = info["lives"]
        mw, last_lives, score, steps = info["wave"], info["lives"], info["score"], 0
        term = trunc = False
        recovered = False
        while steps < STEPCAP:
            a = act(perceived_sprites(builder, hist, hist[-1]))
            _, _, term, trunc, info = env.step(a)
            hist.append(env._last_packet)
            steps += 1
            mw = max(mw, info.get("wave", 0))
            last_lives = info.get("lives", last_lives)
            if info.get("recovered"):
                recovered = True
            if not (term or trunc):
                score = info.get("score", score)
            if term or trunc:
                break
        # Flicker-proof death count (the $BDEC lives byte flickers 1->2->1 during
        # the 0x7F wave transition, so per-step drop counting phantom-counts):
        # deaths = start + earned(score-based) - remaining (0 if the game ended).
        deaths = start_lives + score // 25000 - (0 if term else max(last_lives, 0))
        tag = " RECOVERED-INVALID" if recovered else ""
        if not recovered:
            games.append({"wave": mw, "deaths": deaths, "score": score, "steps": steps})
        print(f"game {g+1}/{N}: wave={mw} deaths={deaths} score={score} steps={steps}{tag}",
              flush=True)
    env.close()
    if games:
        tw = sum(gm["wave"] for gm in games); td = sum(gm["deaths"] for gm in games)
        waves = [gm["wave"] for gm in games]
        print(f"\n=== {MODE} K={K} (params {Path(PARAMS).name}, cap {STEPCAP}, n={len(games)}) ===")
        print(f"wave mean={np.mean(waves):.2f} median={np.median(waves):.1f} "
              f"min={min(waves)} max={max(waves)}")
        print(f"deaths/wave={td/max(tw,1):.3f}  (total {td} deaths / {tw} waves)")


if __name__ == "__main__":
    main()
