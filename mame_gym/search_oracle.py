"""search_oracle.py — Expert-Iteration search teacher on the MAME gym.

DAgger hit the FSM-fidelity ceiling (robust median wave 8). Expert Iteration
replaces the FSM *labeler* with shallow lookahead SEARCH that beats the FSM,
then distills the improved labels with the existing dagger.py machinery
(swap only the labeler in collect()).

Where the FSM is weak is MOVEMENT/positioning — it already fires near-optimally
(at the nearest threat). So this teacher fixes fire = FSM's fire and SEARCHES
the 8 move directions by a K-frame rollout (hold the candidate action), scoring
by score gained minus a heavy death penalty. By construction the chosen move is
>= the FSM's own move in K-step rollout value.

Fast because MAME savestates are ~2-3ms over the bridge socket: save current ->
for each move {restore, roll out K frames, score} -> restore. ~100ms/state for
8 moves x 4 frames (~10 states/sec) — fine for both validation and a full
distillation collect (~10 min for 6k labels).

Server direction convention: bridge.step wants 1..8 (0=none); fsm_action and the
945-dim policy use 0..7. Helpers below keep the boundary explicit.
"""
import numpy as np

from mame_obs import parse_header
from fsm_oracle import fsm_action

SCRATCH_IDX   = 60001        # disk savestate slot reserved for search (above pool/auto-capture)
DEATH_PENALTY = 50_000       # >> any few-frame score; dominates the rollout value


def _dead(h, base_wave):
    # mirrors MameRobotronEnv.step's terminal logic: out of lives, the
    # KILL_PLAYER game_state, or a non-sequential wave move (left play mode).
    return (h["lives"] == 0) or (h["game_state"] == 0x1B) \
        or not (base_wave <= h["wave"] <= base_wave + 1)


SURVIVE_W = 1000.0           # value per frame survived — survival dominates score


def search_move_action(bridge, cur_packet, H=20, scratch=SCRATCH_IDX,
                       death_penalty=DEATH_PENALTY, survive_w=SURVIVE_W):
    """Policy-improvement search: for each first MOVE, follow the FSM as the
    rollout policy for H frames and score by survival (primary) + score gained.
    Keeps the FSM's fire on the first frame. By construction the chosen move is
    >= the FSM's own (one candidate branch is "FSM-move then FSM" = pure FSM).

    bridge      : MameBridge (env._bridge)
    cur_packet  : current raw obs packet (bridge.step/reset return value)
    returns     : (move, fire, best_val) in SERVER dirs 1..8.

    Side effect: leaves the live game restored to `cur_packet`'s state.
    """
    h0 = parse_header(cur_packet)
    base_score, base_wave = h0["score"], h0["wave"]
    fsm = fsm_action(cur_packet)            # [move0_7, fire0_7]
    first_fire = int(fsm[1]) + 1            # FSM fire (toward nearest threat)

    bridge.save_state(scratch)
    best_mv, best_val = int(fsm[0]) + 1, -1e18
    for mv in range(1, 9):                  # 8 real move directions (skip 0=none)
        bridge.reset(scratch)
        pkt = cur_packet
        survived = 0
        died = False
        for k in range(H):
            if k == 0:
                a_mv, a_fr = mv, first_fire          # candidate move, FSM fire
            else:
                fa = fsm_action(pkt)                 # FSM continuation (rollout policy)
                a_mv, a_fr = int(fa[0]) + 1, int(fa[1]) + 1
            pkt, recovered = bridge.step(a_mv, a_fr)
            if recovered:
                died = True
                break
            h = parse_header(pkt)
            if _dead(h, base_wave):
                died = True
                break
            survived += 1
        val = survive_w * survived + (parse_header(pkt)["score"] - base_score)
        if died:
            val -= death_penalty
        if val > best_val:
            best_val, best_mv = val, mv
    bridge.reset(scratch)                   # restore the real state for the caller
    return best_mv, first_fire, best_val


def fsm_move_action(cur_packet):
    """FSM baseline action in SERVER dirs 1..8 (for A/B comparison)."""
    a = fsm_action(cur_packet)
    return int(a[0]) + 1, int(a[1]) + 1
