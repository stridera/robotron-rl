"""Savestate-based differential search for the playable player + score/wave/lives.
Warm up past Coin+Start, capture checkpoint, then for each direction: reset to
checkpoint, hold direction for K frames, record delta. Anything that responds
consistently across all four cardinal directions is signal, not noise."""
from __future__ import annotations
import numpy as np
from robotron_mame_env import RobotronMAMEEnv, OBS_BYTES

N_HOLD = 120     # frames to hold each direction after reset
WARMUP = 240     # frames to step in idle before capturing checkpoint

def hold(env, action, steps):
    obs0, last = None, None
    for i in range(steps):
        obs, *_ = env.step(action)
        if i == 0: obs0 = obs.copy()
        last = obs
    return obs0, last

def main():
    env = RobotronMAMEEnv(verbose=False)
    try:
        obs, _ = env.reset()
        print(f"[init] obs head: {obs[:10].tolist()}")
        # Warm up so the game has time to settle past Coin+Start into a wave.
        for _ in range(WARMUP): obs, *_ = env.step([0, 0])
        print(f"[warmup] after {WARMUP} idle steps obs head: {obs[:10].tolist()}")
        env.save_checkpoint()
        print("[checkpoint] saved")

        diffs = {}
        for dirname, code in [("RIGHT", 3), ("LEFT", 7), ("DOWN", 5), ("UP", 1)]:
            env.reset()
            b, a = hold(env, [code, 0], N_HOLD)
            d = a.astype(int) - b.astype(int)
            diffs[dirname] = d
            moved = np.where(d != 0)[0]
            top = sorted(moved, key=lambda i: -abs(d[i]))[:6]
            print(f"[{dirname:>5}] {len(moved):3d} bytes moved; top: " +
                  " ".join(f"obs[{i}]:{int(b[i])}->{int(a[i])}({d[i]:+d})" for i in top))

        # Also one IDLE baseline run for noise reference
        env.reset()
        b, a = hold(env, [0, 0], N_HOLD)
        diffs["IDLE"] = a.astype(int) - b.astype(int)

        # Robust signal: byte that goes UP under RIGHT and DOWN under LEFT (and
        # vice-versa for vertical), with the sign reliably opposite. Idle noise
        # is allowed to be nonzero but small.
        print("\n[X candidates] obs bytes where RIGHT and LEFT have opposite sign:")
        r, l, d_, u, idle = (diffs[k] for k in ["RIGHT","LEFT","DOWN","UP","IDLE"])
        for i in range(OBS_BYTES):
            if r[i] != 0 and l[i] != 0 and r[i] * l[i] < 0:
                if abs(r[i]) + abs(l[i]) > abs(idle[i]) * 2:    # signal > noise
                    print(f"  obs[{i}:3d] R={r[i]:+4d} L={l[i]:+4d} "
                          f"D={d_[i]:+4d} U={u[i]:+4d} idle={idle[i]:+4d}")

        print("\n[Y candidates] obs bytes where DOWN and UP have opposite sign:")
        for i in range(OBS_BYTES):
            if d_[i] != 0 and u[i] != 0 and d_[i] * u[i] < 0:
                if abs(d_[i]) + abs(u[i]) > abs(idle[i]) * 2:
                    print(f"  obs[{i}:3d] D={d_[i]:+4d} U={u[i]:+4d} "
                          f"R={r[i]:+4d} L={l[i]:+4d} idle={idle[i]:+4d}")
    finally:
        env.close()

if __name__ == "__main__":
    main()
