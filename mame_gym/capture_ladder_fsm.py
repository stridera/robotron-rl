"""capture_ladder_fsm.py — build a deep-wave save-state pool driven by the EVOLVED FSM.

Like capture_ladder_states.py but driven by the obs-limited evolved chooseOutputs FSM
(reaches w16-23, deeper than any policy ~13-16) instead of a PPO model — so we can
capture states in the wave 10-22 band to seed a deep-wave curriculum. Per the ASM
wave-cycle (configs cycle period-20 after wave 40), practicing waves toward the 21-40
band is what carries a policy to 41-100.

Saves a state at each NEW wave entry within [MIN_WAVE, MAX_WAVE] up to a per-wave quota,
persists to mame_bridge.STATE_DIR, writes index->wave mapping to ladder_fsm_state_map.json.

Usage: MAME_RL_RESEED=1 EVOLVE_PARAMS=models/fsm_evolved_reseed_v2.json \
       .venv/bin/python mame_gym/capture_ladder_fsm.py [min_wave] [max_wave] [quota] [start_idx] [max_eps] [port]
"""
import os, json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from mame_robotron_env import MameRobotronEnv
from mame_obs import MameObsBuilder, parse_header
from position_wrapper import SLOT_CATEGORIES, CATCHALL_SLOTS
import robotron_fsm as fsm

MIN_WAVE = int(sys.argv[1]) if len(sys.argv) > 1 else 10
MAX_WAVE = int(sys.argv[2]) if len(sys.argv) > 2 else 20
QUOTA    = int(sys.argv[3]) if len(sys.argv) > 3 else 6
START_IDX = int(sys.argv[4]) if len(sys.argv) > 4 else 30000   # high base, away from collect pool idxs
MAX_EPISODES = int(sys.argv[5]) if len(sys.argv) > 5 else 400
PORT = int(sys.argv[6]) if len(sys.argv) > 6 else 9990
SETTLE_STEPS = 30
MAPPING_PATH = Path(__file__).parent / "ladder_fsm_state_map.json"

W, Hpx = 665, 492
fsm.DEBUG_LEVEL = 0
fsm.MAX_RIGHT, fsm.MAX_TOP = W, Hpx
fsm.MAX_BOTTOM, fsm.MAX_LEFT = 0, 0
fsm.Y_AXIS_INVERSION = Hpx
_ep = os.environ.get("EVOLVE_PARAMS", "")
if _ep and Path(_ep).exists():
    for _n, _v in json.loads(Path(_ep).read_text())["best_params"].items():
        setattr(fsm, _n, _v)
    print(f"loaded evolved params from {_ep}", flush=True)
B = float(getattr(fsm, "BORDER_ADJUST", 20))
fsm.ADJ_TOP, fsm.ADJ_BOTTOM = Hpx - 2, 0 + B + 9
fsm.ADJ_LEFT, fsm.ADJ_RIGHT = 0 + 2, W - B
builder = MameObsBuilder()


def fsm_action(packet):
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


def main():
    env = MameRobotronEnv(rank=0, base_port=PORT, frameskip=4, reset_pool=[0], obs_mode="slot")
    got = Counter()
    mapping = {}
    next_idx = START_IDX + 1
    episodes = 0
    env.reset(); packet = env._last_packet
    settle_left = -1
    pending_wave = -1
    fresh_episode = True
    while episodes < MAX_EPISODES:
        if all(got[w] >= QUOTA for w in range(MIN_WAVE, MAX_WAVE + 1)):
            break
        mi, fi = fsm_action(packet)
        _, _, term, trunc, info = env.step(np.array([mi, fi]))
        packet = env._last_packet
        if term or trunc:
            episodes += 1
            env.reset(); packet = env._last_packet
            settle_left = -1; pending_wave = -1; fresh_episode = True
            continue
        wave = info.get("wave", 0)
        if fresh_episode:
            pending_wave = wave
            fresh_episode = False
            continue
        if wave != pending_wave:
            pending_wave = wave
            settle_left = SETTLE_STEPS if (MIN_WAVE <= wave <= MAX_WAVE and got[wave] < QUOTA) else -1
        if settle_left > 0:
            settle_left -= 1
        elif settle_left == 0:
            settle_left = -1
            pkt = env._bridge.save_state(next_idx)
            h = parse_header(pkt)
            if h["wave"] != pending_wave:
                continue
            got[h["wave"]] += 1
            mapping[next_idx] = {"wave": h["wave"], "score": h["score"], "lives": h["lives"]}
            print(f"saved idx {next_idx}: wave={h['wave']} score={h['score']} lives={h['lives']} "
                  f"(ep {episodes}) coverage={dict(sorted(got.items()))}", flush=True)
            next_idx += 1
    old = json.loads(MAPPING_PATH.read_text()) if MAPPING_PATH.exists() else {}
    old.update({str(k): v for k, v in mapping.items()})
    MAPPING_PATH.write_text(json.dumps(old, indent=1, sort_keys=True))
    print(f"DONE in {episodes} episodes. coverage={dict(sorted(got.items()))} next_idx={next_idx}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
