"""obs_decode.py — decode the 945-dim slot obs back into the FSM's entity list, and
query the FSM (chooseOutputs) from EXACTLY what the policy perceives. Used for clean BC
labels and for DAgger (label the policy's own visited obs with the expert action)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from position_wrapper import SPRITE_TYPES
import robotron_fsm as fsm

W, H = 665.0, 492.0

# Configure the FSM module globals once (as robotron_fsm.main() does, 665x492 board).
fsm.DEBUG_LEVEL = 0
fsm.MAX_RIGHT, fsm.MAX_TOP = W, H
fsm.MAX_BOTTOM, fsm.MAX_LEFT = 0, 0
fsm.Y_AXIS_INVERSION = H
_B = 20
fsm.ADJ_TOP, fsm.ADJ_BOTTOM, fsm.ADJ_LEFT, fsm.ADJ_RIGHT = H - 2, _B + 9, 2, W - _B


def obs_to_fsm_data(obs):
    """945-dim obs -> [(x,y,name), ...] in pixel coords (the chooseOutputs format)."""
    px = (float(obs[0]) + 1.0) / 2.0 * W
    py = (float(obs[1]) + 1.0) / 2.0 * H
    data = [(px, py, "Player")]
    for i in range(41):
        off = 2 + i * 23
        if obs[off + 20] < 0.5:          # valid flag
            continue
        name = SPRITE_TYPES[int(np.argmax(obs[off:off + 16]))]
        if name == "Player":             # spurious (type-idx 0 fallback) — skip
            continue
        rx = float(obs[off + 16]) * W
        ry = float(obs[off + 17]) * H
        data.append((px + rx, py + ry, name))
    return data


def fsm_action_from_obs(obs):
    """(move_idx, fire_idx) in 0-7 from the FSM, given a 945-dim obs."""
    try:
        mv, fr = fsm.chooseOutputs(obs_to_fsm_data(obs))
    except Exception:
        mv, fr = 1, 1
    mi = (mv - 1) if mv >= 1 else 0
    fi = (fr - 1) if fr >= 1 else mi
    return mi, fi


if __name__ == "__main__":
    # Slot-match check: does FSM-on-decoded-obs agree with the stored demo labels
    # (FSM-on-replicated-selection)? High agreement => slot replication was fine and
    # the 60% BC ceiling is covariate shift, not a label mismatch.
    d = np.load("demos/fsm_mame_demos.npz")
    obs, act = d["obs"], d["actions"]
    n = min(20000, len(obs))
    idx = np.random.default_rng(0).choice(len(obs), n, replace=False)
    mm = ff = 0
    for j in idx:
        mi, fi = fsm_action_from_obs(obs[j])
        mm += (mi == act[j, 0]); ff += (fi == act[j, 1])
    print(f"slot-match check on {n} demos:")
    print(f"  move agreement (decode-FSM vs stored label): {mm/n:.3f}")
    print(f"  fire agreement: {ff/n:.3f}")
