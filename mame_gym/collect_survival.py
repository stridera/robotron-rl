"""collect_survival.py — survival-labeled data for a death-risk VALUE model.

Runs the evolved FSM (the best controller) and records, per step, the 945-dim obs and
frames-until-death (clamped to HORIZON). A value model V(obs) ~ P(survive >= H) trained
on this powers a value-guided SEARCH teacher: score a candidate state by predicted future
survival instead of raw H-frame rollouts (which fail — rollout-policy != play-policy).

Output: demos/survival_<tag>.npz  with obs (N,945) f32, ttl (N,) i32 (frames-until-death,
clamped to HORIZON; episodes that hit the step cap are right-censored and dropped at the
tail so labels stay valid).

Usage: MAME_RL_RESEED=1 EVOLVE_PARAMS=models/fsm_evolved_reseed_v2.json \
       .venv/bin/python mame_gym/collect_survival.py <target_pairs> <port> [tag] [horizon]
"""
import os, sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from mame_robotron_env import MameRobotronEnv
from mame_obs import MameObsBuilder
from position_wrapper import SLOT_CATEGORIES, CATCHALL_SLOTS
import robotron_fsm as fsm

TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 300_000
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 9988
TAG = sys.argv[3] if len(sys.argv) > 3 else "v2"
HORIZON = int(sys.argv[4]) if len(sys.argv) > 4 else 60   # frames (env steps, frameskip 4)
OUT = Path(__file__).parent.parent / "demos" / f"survival_{TAG}.npz"
STEPCAP = 8000

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
fsm.ADJ_TOP, fsm.ADJ_BOTTOM = H - 2, 0 + B + 9
fsm.ADJ_LEFT, fsm.ADJ_RIGHT = 0 + 2, W - B
builder = MameObsBuilder()


def fsm_action(packet):
    sprites = builder._sprites_from_packet(packet)
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


def main():
    pool = [int(x) for x in (Path(__file__).parent / "bc_collect_pool.txt").read_text().split(",")]
    env = MameRobotronEnv(rank=0, base_port=PORT, frameskip=4, reset_pool=pool, obs_mode="slot")
    obs_buf = np.zeros((TARGET, 945), np.float32)
    ttl_buf = np.zeros((TARGET,), np.int32)
    n = 0
    ep_obs, ep_died = [], 0
    obs, _ = env.reset(); packet = env._last_packet
    steps = 0
    while n < TARGET:
        mi, fi = fsm_action(packet)
        ep_obs.append(obs.copy())
        obs, _, term, trunc, info = env.step(np.array([mi, fi]))
        packet = env._last_packet
        steps += 1
        died = bool(term)  # term => death; trunc/stepcap => censored
        if died or trunc or steps >= STEPCAP:
            L = len(ep_obs)
            if died:
                # label each step with frames-until-death, clamp to HORIZON
                for i, o in enumerate(ep_obs):
                    if n >= TARGET:
                        break
                    obs_buf[n] = o
                    ttl_buf[n] = min(L - i, HORIZON)
                    n += 1
                ep_died += 1
            else:
                # censored episode: only keep steps whose horizon window fully fits
                # (>= HORIZON from the censor point) -> label HORIZON (survived the window)
                keep = max(0, L - HORIZON)
                for i in range(keep):
                    if n >= TARGET:
                        break
                    obs_buf[n] = ep_obs[i]; ttl_buf[n] = HORIZON; n += 1
            ep_obs = []
            obs, _ = env.reset(); packet = env._last_packet; steps = 0
            if ep_died % 20 == 0 and ep_died > 0:
                print(f"  {n}/{TARGET} states, {ep_died} death-episodes", flush=True)
    env.close()
    OUT.parent.mkdir(exist_ok=True)
    np.savez_compressed(OUT, obs=obs_buf[:n], ttl=ttl_buf[:n])
    pct_short = float((ttl_buf[:n] < HORIZON).mean())
    print(f"saved {OUT}: {n} states, {ep_died} deaths, {pct_short*100:.0f}% within {HORIZON}f of death", flush=True)


if __name__ == "__main__":
    main()
