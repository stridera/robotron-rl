"""Paired FSM A/B eval: play the champion FSM N games and record per-game max wave.

Pairing: with a fixed MAME_RL_SEED_BASE, game i faces the same RNG seed in every run,
so two runs (baseline vs a FSM_* knob) are paired by game index. Run this twice (knob
off/on, same seed base) then pair with fsm_ab_compare.py.

Usage:
  MAME_RL_RESEED=1 MAME_RL_SEED_BASE=777 EVOLVE_PARAMS=models/fsm_evolved_combteacher.json \
  [FSM_HULK_PUSH=1] .venv/bin/python3 mame_gym/fsm_ab_eval.py <N> <port> <out.json>
"""
import os, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from mame_robotron_env import MameRobotronEnv
from mame_obs import parse_header, iter_entities, classify_sw
import robotron_fsm as fsm

N    = int(sys.argv[1]) if len(sys.argv) > 1 else 100
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 9970
OUT  = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("/tmp/fsm_ab.json")

W, H = 665, 492
fsm.DEBUG_LEVEL = 0
fsm.MAX_RIGHT, fsm.MAX_TOP, fsm.MAX_BOTTOM, fsm.MAX_LEFT = W, H, 0, 0
fsm.Y_AXIS_INVERSION = H
_ep = os.environ.get("EVOLVE_PARAMS", "")
if _ep and Path(_ep).exists():
    for k, v in json.loads(Path(_ep).read_text())["best_params"].items():
        setattr(fsm, k, v)
B = float(getattr(fsm, "BORDER_ADJUST", 20))
fsm.ADJ_TOP, fsm.ADJ_BOTTOM, fsm.ADJ_LEFT, fsm.ADJ_RIGHT = H - 2, B + 9, 2, W - B
def topix(gx, gy): return ((gx - 5) / 140 * W, (gy - 15) / 215 * H)
NM = {"Brain": "Brain", "Cruise Missile": "CruiseMissile", "Dad": "Daddy",
      "Electrode": "Electrode", "EnfBullet/Spark": "EnforcerBullet", "Enforcer": "Enforcer",
      "Grunt": "Grunt", "Hulk": "Hulk", "Mikey": "Mikey", "Mom": "Mommy", "Prog": "Prog",
      "Quark": "Quark", "Spheroid": "Sphereoid", "Tank": "Tank", "TankShell": "TankShell"}
def bd(p):
    h = parse_header(p); o = [(*topix(h["player_x"], h["player_y"]), "Player")]
    for a, l, sw, x, y in iter_entities(p):
        nm = NM.get(classify_sw(sw))
        if nm: o.append((*topix(x, y), nm))
    return o

def main():
    env = MameRobotronEnv(rank=0, base_port=PORT, frameskip=4, reset_pool=[0], obs_mode="slot")
    waves = []
    for g in range(N):
        env.reset(); p = env._last_packet; mw = env._last_wave
        while True:
            try: mv, fr = fsm.chooseOutputs(bd(p))
            except Exception: mv, fr = 1, 1
            mi = (mv - 1) if mv >= 1 else 0
            fi = (fr - 1) if fr >= 1 else mi
            _, _, t, tr, info = env.step(np.array([mi, fi]))
            p = env._last_packet; mw = max(mw, info.get("wave", 0))
            if t or tr: break
        waves.append(mw)
        if (g + 1) % 10 == 0: print(f"  {g+1}/{N} games, mean {np.mean(waves):.2f}", flush=True)
    env.close()
    OUT.write_text(json.dumps({"waves": waves, "push": fsm.HULK_PUSH,
                               "buffer": fsm.BUFFER_SCALE, "n": N}))
    print(f"DONE {N} games: mean {np.mean(waves):.2f} median {int(np.median(waves))} "
          f"max {max(waves)} -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
