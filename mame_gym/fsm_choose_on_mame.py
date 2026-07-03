"""Run the USER's actual FSM (robotron_fsm.chooseOutputs, proven 23 on python / 8-9 on
Xenia) on the MAME env. Decisive: 8-9 here => MAME faithful, agents just weak; ~4 here
=> MAME has a real defect making it harder than the same ROM on Xenia."""
import sys
import statistics
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from mame_robotron_env import MameRobotronEnv
from mame_obs import parse_header, iter_entities, classify_sw
import robotron_fsm as fsm

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 9982
FRAMESKIP = int(sys.argv[3]) if len(sys.argv) > 3 else 4

# python-gym pixel board + replicate robotron_fsm.main() global setup
W, H = 665, 492
fsm.DEBUG_LEVEL = 0
fsm.MAX_RIGHT, fsm.MAX_TOP = W, H
fsm.MAX_BOTTOM, fsm.MAX_LEFT = 0, 0
fsm.Y_AXIS_INVERSION = H
B = 20
fsm.ADJ_TOP, fsm.ADJ_BOTTOM, fsm.ADJ_LEFT, fsm.ADJ_RIGHT = H - 2, 0 + B + 9, 0 + 2, W - B

# MAME classify_sw name -> FSM name
NAME = {"Brain": "Brain", "Brain (alt)": "Brain", "Cruise Missile": "CruiseMissile",
        "Dad": "Daddy", "Electrode": "Electrode", "EnfBullet/Spark": "EnforcerBullet",
        "Enforcer": "Enforcer", "Grunt": "Grunt", "Hulk": "Hulk", "Mikey": "Mikey",
        "Mom": "Mommy", "PlayerBullet": "Bullet", "Quark": "Quark",
        "Spheroid": "Sphereoid", "Tank": "Tank"}
GX_MIN, GX_R, GY_MIN, GY_R = 5, 140, 15, 215


def topix(gx, gy):
    return ((gx - GX_MIN) / GX_R * W, (gy - GY_MIN) / GY_R * H)


def build_data(packet):
    h = parse_header(packet)
    objs = [(*topix(h["player_x"], h["player_y"]), "Player")]
    for addr, lid, sw, x, y in iter_entities(packet):
        pn = NAME.get(classify_sw(sw))
        if pn is None:   # PlayerIcon / unknown -> skip
            continue
        objs.append((*topix(x, y), pn))
    return objs


def main():
    env = MameRobotronEnv(rank=0, base_port=PORT, frameskip=FRAMESKIP, reset_pool=[0], obs_mode="slot")
    results = []
    for ep in range(N):
        env.reset()
        packet = env._last_packet
        maxw = env._last_wave
        steps = 0
        while True:
            try:
                mv, fr = fsm.chooseOutputs(build_data(packet))   # 0-8 each (0=none)
            except Exception:
                mv, fr = 1, 1
            # MAME is always-move/always-fire (idx -> dir+1). Map FSM no-op -> a move.
            move_idx = (mv - 1) if mv >= 1 else 0
            fire_idx = (fr - 1) if fr >= 1 else move_idx
            _, _, t, tr, info = env.step(np.array([move_idx, fire_idx]))
            packet = env._last_packet
            maxw = max(maxw, info.get("wave", 0))
            steps += 1
            if t or tr:
                break
        results.append(maxw)
        print(f"ep {ep+1}: wave {maxw}  ({steps} steps)", flush=True)
    print(f"\n=== {N} chooseOutputs-FSM runs (MAME env) ===")
    print(f"wave: min={min(results)} max={max(results)} mean={statistics.mean(results):.1f} "
          f"median={statistics.median(results)}  dist={dict(sorted(Counter(results).items()))}")
    env.close()


if __name__ == "__main__":
    main()
