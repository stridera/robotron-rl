"""residual_env.py — FSM-residual control wrapper (SUGGESTIONS.md strongest rec).

The evolved FSM (12.72) proposes (move, fire) every step. The RL policy outputs a
GATE + an override move:
  action = MultiDiscrete([2, 8]) = (gate, override_move)
  gate==0 -> use the FSM's move   (default: keep the teacher)
  gate==1 -> use override_move    (the policy takes over movement this step)
  fire    -> ALWAYS the FSM's fire (it already fires near-optimally)

Reward = base env reward - OVERRIDE_PENALTY when gate==1. This biases the policy to
keep the FSM and override ONLY when it's worth it (e.g. dodging a Quark/Hulk that the
FSM would walk into). Conservative recipe: it starts near the FSM and learns targeted
movement overrides in death-risk states — a much smaller problem than cloning the FSM.

Goal (per death forensics): cut deaths/wave (Quark=32% of FSM deaths) while keeping the
FSM's 12.72 capability, so the residual beats the pure FSM on eval_protocol paired games.
"""
import os
import json
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete

import robotron_fsm as fsm
from mame_obs import MameObsBuilder
from position_wrapper import SLOT_CATEGORIES, CATCHALL_SLOTS

OVERRIDE_PENALTY = float(os.environ.get("RESIDUAL_OVERRIDE_PENALTY", "0.2"))


def _setup_fsm():
    W, H = 665, 492
    fsm.DEBUG_LEVEL = 0
    fsm.MAX_RIGHT, fsm.MAX_TOP = W, H
    fsm.MAX_BOTTOM, fsm.MAX_LEFT = 0, 0
    fsm.Y_AXIS_INVERSION = H
    ep = os.environ.get("EVOLVE_PARAMS", "")
    if ep and Path(ep).exists():
        for n, v in json.loads(Path(ep).read_text())["best_params"].items():
            setattr(fsm, n, v)
    B = float(getattr(fsm, "BORDER_ADJUST", 20))
    fsm.ADJ_TOP, fsm.ADJ_BOTTOM = H - 2, 0 + B + 9
    fsm.ADJ_LEFT, fsm.ADJ_RIGHT = 0 + 2, W - B


class ResidualWrapper(gym.Wrapper):
    """Wrap a MameRobotronEnv (slot obs). Policy gates FSM movement; fire stays FSM."""

    def __init__(self, env):
        super().__init__(env)
        _setup_fsm()
        self._builder = MameObsBuilder()
        self.action_space = MultiDiscrete([2, 8])
        # observation_space unchanged (945-dim slot obs from the wrapped env)

    def _fsm_action(self, packet):
        sprites = self._builder._sprites_from_packet(packet)
        player = next((s for s in sprites if s[2] == "Player"), None)
        if player is None:
            return 0, 0
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
        mi = (mv - 1) if mv >= 1 else 0
        fi = (fr - 1) if fr >= 1 else mi
        return mi, fi

    def step(self, action):
        gate, override_move = int(action[0]), int(action[1])
        fsm_move, fsm_fire = self._fsm_action(self.env.unwrapped._last_packet)
        move = fsm_move if gate == 0 else override_move
        obs, reward, term, trunc, info = self.env.step(np.array([move, fsm_fire]))
        if gate == 1:
            reward -= OVERRIDE_PENALTY
        info["override"] = gate
        return obs, reward, term, trunc, info
