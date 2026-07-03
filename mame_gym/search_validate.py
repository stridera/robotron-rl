"""search_validate.py — measure the policy-improvement SEARCH teacher's own wave
performance on MAME, built on top of the EVOLVED chooseOutputs FSM (not the simple
fsm_oracle base that search_oracle.py wires in).

Teacher = keep the evolved-FSM fire (already near-optimal at the nearest threat) and
SEARCH the 8 move directions by an H-frame save/restore rollout, continuing with the
evolved FSM as the rollout policy, scoring survival (primary) + score gained - death.
By construction the chosen move is >= the evolved FSM's own move in H-step value, so
this should validate AT OR ABOVE the evolved FSM's 12.72 (v2) — quantifies whether
lookahead beats the reactive FSM ceiling before committing to a full ExIt distill.

Usage: MAME_RL_RESEED=1 .venv/bin/python mame_gym/search_validate.py <evolved.json|default> [N] [PORT] [H]
"""
import sys, json, statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "mame_gym"))
import numpy as np
from mame_robotron_env import MameRobotronEnv
from mame_obs import MameObsBuilder, parse_header
from position_wrapper import SLOT_CATEGORIES, CATCHALL_SLOTS
import robotron_fsm as fsm

SRC = sys.argv[1] if len(sys.argv) > 1 else "default"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 12
PORT = int(sys.argv[3]) if len(sys.argv) > 3 else 9966
H = int(sys.argv[4]) if len(sys.argv) > 4 else 16

W, Hpx = 665, 492
fsm.DEBUG_LEVEL = 0
fsm.MAX_RIGHT, fsm.MAX_TOP = W, Hpx
fsm.MAX_BOTTOM, fsm.MAX_LEFT = 0, 0
fsm.Y_AXIS_INVERSION = Hpx
if SRC != "default":
    params = json.loads(Path(SRC).read_text())["best_params"]
    for name, val in params.items():
        setattr(fsm, name, val)
    print(f"loaded {len(params)} evolved params from {SRC}", flush=True)
else:
    print("using DEFAULT FSM params", flush=True)
B = float(getattr(fsm, "BORDER_ADJUST", 20))
fsm.ADJ_TOP, fsm.ADJ_BOTTOM = Hpx - 2, 0 + B + 9
fsm.ADJ_LEFT, fsm.ADJ_RIGHT = 0 + 2, W - B

builder = MameObsBuilder()
SCRATCH_IDX = 60002
DEATH_PENALTY = 50_000.0
SURVIVE_W = 1000.0


def obs_fsm_action(packet):
    """Obs-limited evolved chooseOutputs -> (move0_7, fire0_7), same selection as the 945-dim obs."""
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


def _dead(h, base_wave):
    return (h["lives"] == 0) or (h["game_state"] == 0x1B) \
        or not (base_wave <= h["wave"] <= base_wave + 1)


def search_move(bridge, cur_packet):
    """Return (move, fire) in SERVER dirs 1..8; leaves live game restored to cur_packet."""
    h0 = parse_header(cur_packet)
    base_score, base_wave = h0["score"], h0["wave"]
    fmv, ffr = obs_fsm_action(cur_packet)
    first_fire = ffr + 1
    bridge.save_state(SCRATCH_IDX)
    best_mv, best_val = fmv + 1, -1e18
    for mv in range(1, 9):
        bridge.reset(SCRATCH_IDX)
        pkt = cur_packet
        survived, died = 0, False
        for k in range(H):
            if k == 0:
                a_mv, a_fr = mv, first_fire
            else:
                cm, cf = obs_fsm_action(pkt)
                a_mv, a_fr = cm + 1, cf + 1
            pkt, recovered = bridge.step(a_mv, a_fr)
            if recovered:
                died = True; break
            h = parse_header(pkt)
            if _dead(h, base_wave):
                died = True; break
            survived += 1
        val = SURVIVE_W * survived + (parse_header(pkt)["score"] - base_score)
        if died:
            val -= DEATH_PENALTY
        if val > best_val:
            best_val, best_mv = val, mv
    bridge.reset(SCRATCH_IDX)
    return best_mv, first_fire


def main():
    env = MameRobotronEnv(rank=0, base_port=PORT, frameskip=4, reset_pool=[0], obs_mode="slot")
    waves, scores = [], []
    for g in range(N):
        env.reset(); packet = env._last_packet
        mw = env._last_wave; sc = env._last_score; steps = 0
        while steps < 8000:
            mv, fr = search_move(env._bridge, packet)
            _, _, term, trunc, info = env.step(np.array([mv - 1, fr - 1]))
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
    print(f"\n=== {N} games SEARCH-teacher (H={H}, base={SRC}) ===")
    print(f"wave: min={min(waves)} max={max(waves)} mean={statistics.mean(waves):.2f} median={statistics.median(waves)}")
    print(f"score: max={max(scores)} mean={statistics.mean(scores):.0f}")
    print(f"dist: {dict(sorted(Counter(waves).items()))}")


if __name__ == "__main__":
    main()
