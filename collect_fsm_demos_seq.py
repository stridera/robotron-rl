"""collect_fsm_demos_seq.py — parallel FSM demos in SLOT obs WITH episode boundaries.

For the recurrent/LSTM lever: a MlpLstmPolicy needs episode-ordered sequences (not the
shuffled flat (obs,action) of dagger_agg). This is the slot collector (collect_fsm_demos.py
labels) in the parallel-shard form of collect_grid_demos.py, PLUS a per-step `dones` array
(True on the last step of each episode) so bc_recurrent.py can split shards into contiguous
episodes and reset the LSTM hidden state at each boundary.

The FSM labels from the raw PACKET (obs-agnostic), identical to the slot collector. Lower
epsilon (0.05) than the flat collector: recurrent BC wants trajectories close to FSM behavior
so the hidden state learns coherent dynamics, while a little noise still adds recovery coverage.

Output: demos/seq_<tag>_shards/shard_*.npz  (obs float16 (n,945), actions int8 (n,2),
dones bool (n,)). Episode order is preserved WITHIN each shard.

Usage: .venv/bin/python3 collect_fsm_demos_seq.py [n_workers] [per_worker] [base_port] [epsilon] [tag]
"""
import sys
import multiprocessing as mp
from pathlib import Path

ROOT = Path(__file__).parent

N        = int(sys.argv[1]) if len(sys.argv) > 1 else 12
PER      = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
BASEPORT = int(sys.argv[3]) if len(sys.argv) > 3 else 9920
EPSILON  = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
TAG      = sys.argv[5] if len(sys.argv) > 5 else "v1"


def worker(wid, n, port, epsilon, shard_path):
    import sys as _s
    _s.path.insert(0, str(ROOT)); _s.path.insert(0, str(ROOT / "mame_gym"))
    import time
    import numpy as np
    from mame_robotron_env import MameRobotronEnv
    from mame_obs import MameObsBuilder
    from position_wrapper import SLOT_CATEGORIES, CATCHALL_SLOTS
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
    env = MameRobotronEnv(rank=wid, base_port=port, frameskip=4, reset_pool=pool, obs_mode="slot")

    ob = np.zeros((n, 945), dtype=np.float16)
    ac = np.zeros((n, 2), dtype=np.int8)
    dn = np.zeros((n,), dtype=bool)
    obs, _ = env.reset(); packet = env._last_packet
    k = 0; t0 = time.time(); eps_count = 0
    while k < n:
        mi, fi = fsm_action(packet)
        ob[k] = obs.astype(np.float16); ac[k] = (mi, fi)
        if rng.random() < epsilon:
            step_a = np.array([rng.integers(0, 8), rng.integers(0, 8)])
        else:
            step_a = np.array([mi, fi])
        obs, _, term, trunc, _ = env.step(step_a)
        packet = env._last_packet
        done = bool(term or trunc)
        dn[k] = done
        k += 1
        if done:
            eps_count += 1
            obs, _ = env.reset(); packet = env._last_packet
        if k % 5000 == 0:
            print(f"  [w{wid}] {k}/{n} ({k/(time.time()-t0):.1f}/s, {eps_count} eps)", flush=True)
    # ensure the final partial episode is marked terminal so sequence-splitting is clean
    dn[k - 1] = True
    np.savez_compressed(shard_path, obs=ob, actions=ac, dones=dn)
    env.close()
    print(f"  [w{wid}] DONE {n} ({eps_count} eps) -> {shard_path}", flush=True)


def main():
    import time
    mp.set_start_method("spawn", force=True)
    shard_dir = ROOT / "demos" / f"seq_{TAG}_shards"; shard_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    print(f"launching {N} workers x {PER} slot+done states (eps={EPSILON}) -> {shard_dir}", flush=True)
    t0 = time.time()
    for wid in range(N):
        sp = str(shard_dir / f"shard_{wid}.npz")
        p = mp.Process(target=worker, args=(wid, PER, BASEPORT + wid * 2, EPSILON, sp))
        p.start(); procs.append(p)
    for p in procs:
        p.join()
    total = sum(PER for wid in range(N) if (shard_dir / f"shard_{wid}.npz").exists())
    dt = time.time() - t0
    print(f"\nDONE {total:,} seq demos across {N} shards -> {shard_dir}  in {dt/60:.1f} min "
          f"({total/dt:.1f}/s aggregate)", flush=True)


if __name__ == "__main__":
    main()
