"""Collect per-death PRE-DEATH TRAJECTORIES for the champion FSM.

MAME save-state is lossy (no rewind), so we use the env's packet-history circular
queue (MAME_PKT_HISTORY) as the rewind: on every life loss, dump the full history
(all entities + player each frame) + the FSM's chosen move/fire, to JSONL. The
escape-route analyzer (analyze_escape.py) then reconstructs, per death, whether the
killer behaved normally and whether a safe move existed that the FSM didn't take.

Usage:
  MAME_RL_RESEED=1 MAME_PKT_HISTORY=40 EVOLVE_PARAMS=models/fsm_evolved_combteacher.json \
  .venv/bin/python3 mame_gym/collect_deaths_fsm.py [n_games] [port] [out.jsonl]
"""
import os, sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from mame_robotron_env import MameRobotronEnv
from mame_obs import parse_header, iter_entities, classify_sw
import robotron_fsm as fsm

N_GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 40
PORT    = int(sys.argv[2]) if len(sys.argv) > 2 else 9962
OUT     = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("deaths_fixed_v1/trajectories.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MAME_PKT_HISTORY", "40")

W, H = 665, 492
fsm.DEBUG_LEVEL = 0
fsm.MAX_RIGHT, fsm.MAX_TOP = W, H
fsm.MAX_BOTTOM, fsm.MAX_LEFT = 0, 0
fsm.Y_AXIS_INVERSION = H
_ep = os.environ.get("EVOLVE_PARAMS", "")
if _ep and Path(_ep).exists():
    for n, v in json.loads(Path(_ep).read_text())["best_params"].items():
        setattr(fsm, n, v)
B = float(getattr(fsm, "BORDER_ADJUST", 20))
fsm.ADJ_TOP, fsm.ADJ_BOTTOM, fsm.ADJ_LEFT, fsm.ADJ_RIGHT = H - 2, 0 + B + 9, 0 + 2, W - B
GX_MIN, GX_R, GY_MIN, GY_R = 5, 140, 15, 215
def topix(gx, gy): return ((gx - GX_MIN) / GX_R * W, (gy - GY_MIN) / GY_R * H)
FSMNAME = {"Brain": "Brain", "Cruise Missile": "CruiseMissile", "Dad": "Daddy",
           "Electrode": "Electrode", "EnfBullet/Spark": "EnforcerBullet",
           "Enforcer": "Enforcer", "Grunt": "Grunt", "Hulk": "Hulk", "Mikey": "Mikey",
           "Mom": "Mommy", "Prog": "Prog", "Quark": "Quark", "Spheroid": "Sphereoid",
           "Tank": "Tank", "TankShell": "TankShell"}

def build_data(packet):
    h = parse_header(packet)
    objs = [(*topix(h["player_x"], h["player_y"]), "Player")]
    for addr, lid, sw, x, y in iter_entities(packet):
        pn = FSMNAME.get(classify_sw(sw))
        if pn is not None:
            objs.append((*topix(x, y), pn))
    return objs

def decode_frame(packet):
    """All entities this frame in GAME UNITS (raw display coords, addr identity)."""
    h = parse_header(packet)
    ents = []
    for addr, lid, sw, x, y in iter_entities(packet):
        name = classify_sw(sw)
        if name is None:
            continue
        ents.append({"a": addr, "sw": sw, "n": name, "x": x, "y": y})
    return {"wave": h["wave"], "lives": h["lives"], "px": h["player_x"],
            "py": h["player_y"], "gs": h["game_state"], "ents": ents}

def main():
    env = MameRobotronEnv(rank=0, base_port=PORT, frameskip=4, reset_pool=[0], obs_mode="slot")
    out = open(OUT, "w")
    n_deaths = 0
    for game in range(N_GAMES):
        env.reset()
        pkt = env._last_packet
        prev_lives = parse_header(pkt)["lives"]
        maxw = env._last_wave
        acts = []   # (move,fire) aligned to _pkt_history frames
        while True:
            try:
                mv, fr = fsm.chooseOutputs(build_data(pkt))
            except Exception:
                mv, fr = 1, 1
            move_idx = (mv - 1) if mv >= 1 else 0
            fire_idx = (fr - 1) if fr >= 1 else move_idx
            acts.append([move_idx, fire_idx])
            if len(acts) > env._pkt_history_len:
                acts.pop(0)
            _, _, t, tr, info = env.step(np.array([move_idx, fire_idx]))
            pkt = env._last_packet
            maxw = max(maxw, info.get("wave", 0))
            lives = parse_header(pkt)["lives"]
            if lives < prev_lives:   # a life was lost -> dump the pre-death history
                traj = [decode_frame(p) for p in env._pkt_history]
                out.write(json.dumps({"game": game, "wave": maxw, "death_idx": n_deaths,
                                      "frames": traj, "actions": acts[-len(traj):]}) + "\n")
                out.flush()
                n_deaths += 1
            prev_lives = lives
            if t or tr:
                break
        print(f"game {game+1}: wave {maxw}, deaths so far {n_deaths}", flush=True)
    out.close()
    env.close()
    print(f"DONE: {n_deaths} death trajectories -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
