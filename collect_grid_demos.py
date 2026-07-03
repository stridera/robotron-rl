"""collect_grid_demos.py — parallel FSM behavior-cloning demos in GRID obs form.

The from-scratch grid-CNN run (9iwa3zzv) walled at mean wave 3 — same wall pure
RL hit on the slot obs. The slot policy only broke past it via a DAgger-on-FSM BC
WARMSTART (dagger_x3_best -> yiwzpqq7, mean 8.1). This collects the equivalent
warmstart set for the grid CNN: the obs-limited FSM teacher drives MAME, and we
record (grid_obs, action) pairs so a CNN policy can be BC-pretrained, then RL-tuned
(train_mame.py --obs-mode grid --bc-checkpoint ...). Tests the grid's TRUE ceiling
(richer obs preserves global threat layout) vs slot's 8.1. See project_goal_wave100.

The FSM labels from the raw PACKET (obs-mode-agnostic), so labels are identical to
the slot collector; only the recorded obs differs. Grid obs is (11,72,48) ~ 40x the
slot's 945 floats, so we store float16 + COMPRESSED shards (the grid is very sparse
-> tiny on disk). The BC trainer streams shards (never all in RAM -> respects the
WSL 48GB cap; see project_wsl_memory_cap).

Usage: .venv/bin/python3 collect_grid_demos.py [n_workers] [per_worker] [base_port] [epsilon] [tag]
Merged manifest -> demos/grid_<tag>_shards/  (shard_*.npz, each obs float16 + actions int8)
"""
import sys
import multiprocessing as mp
from pathlib import Path

ROOT = Path(__file__).parent

N        = int(sys.argv[1]) if len(sys.argv) > 1 else 12
PER      = int(sys.argv[2]) if len(sys.argv) > 2 else 40_000
BASEPORT = int(sys.argv[3]) if len(sys.argv) > 3 else 9920
EPSILON  = float(sys.argv[4]) if len(sys.argv) > 4 else 0.10
TAG      = sys.argv[5] if len(sys.argv) > 5 else "v1"


def worker(wid, n, port, epsilon, shard_path):
    import sys as _s
    _s.path.insert(0, str(ROOT)); _s.path.insert(0, str(ROOT / "mame_gym"))
    import time
    import numpy as np
    from mame_robotron_env import MameRobotronEnv
    from mame_obs import MameObsBuilder
    from position_wrapper import SLOT_CATEGORIES, CATCHALL_SLOTS
    from spatial_obs import NUM_CHANNELS, GRID_H, GRID_W
    import robotron_fsm as fsm

    # FSM global setup (665x492 px board) — identical to collect_fsm_demos.py
    W, H = 665, 492
    fsm.DEBUG_LEVEL = 0
    fsm.MAX_RIGHT, fsm.MAX_TOP = W, H
    fsm.MAX_BOTTOM, fsm.MAX_LEFT = 0, 0
    fsm.Y_AXIS_INVERSION = H
    Bm = 20
    fsm.ADJ_TOP, fsm.ADJ_BOTTOM, fsm.ADJ_LEFT, fsm.ADJ_RIGHT = H - 2, 0 + Bm + 9, 0 + 2, W - Bm

    builder = MameObsBuilder()
    rng = np.random.default_rng(12345 + wid)

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

    pool = [int(x) for x in (ROOT / "mame_gym/bc_collect_pool.txt").read_text().split(",")]
    env = MameRobotronEnv(rank=wid, base_port=port, frameskip=4, reset_pool=pool, obs_mode="grid")

    ob = np.zeros((n, NUM_CHANNELS, GRID_H, GRID_W), dtype=np.float16)
    ac = np.zeros((n, 2), dtype=np.int8)
    obs, _ = env.reset(); packet = env._last_packet
    k = 0; t0 = time.time()
    while k < n:
        mi, fi = fsm_action(packet)
        ob[k] = obs.astype(np.float16); ac[k] = (mi, fi); k += 1
        if rng.random() < epsilon:
            step_a = np.array([rng.integers(0, 8), rng.integers(0, 8)])
        else:
            step_a = np.array([mi, fi])
        obs, _, term, trunc, _ = env.step(step_a)
        packet = env._last_packet
        if term or trunc:
            obs, _ = env.reset(); packet = env._last_packet
        if k % 5000 == 0:
            print(f"  [w{wid}] {k}/{n} ({k/(time.time()-t0):.1f}/s)", flush=True)
    np.savez_compressed(shard_path, obs=ob, actions=ac)
    env.close()
    print(f"  [w{wid}] DONE {n} -> {shard_path}", flush=True)


def main():
    import time
    mp.set_start_method("spawn", force=True)
    shard_dir = ROOT / "demos" / f"grid_{TAG}_shards"; shard_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    print(f"launching {N} workers x {PER} grid states (eps={EPSILON}) -> {shard_dir}", flush=True)
    t0 = time.time()
    for wid in range(N):
        sp = str(shard_dir / f"shard_{wid}.npz")
        p = mp.Process(target=worker, args=(wid, PER, BASEPORT + wid * 2, EPSILON, sp))
        p.start(); procs.append(p)
    for p in procs:
        p.join()
    total = 0
    for wid in range(N):
        sp = shard_dir / f"shard_{wid}.npz"
        if sp.exists():
            total += PER
        else:
            print(f"  WARN: shard {wid} missing", flush=True)
    dt = time.time() - t0
    print(f"\nDONE {total:,} grid demos across {N} shards -> {shard_dir}  in {dt/60:.1f} min "
          f"({total/dt:.1f}/s aggregate)", flush=True)


if __name__ == "__main__":
    main()
