"""parallel_search_collect.py — parallel ExIt search-labeling across N MAME instances.

Single-instance search labeling (~360ms/state) caps ExIt label volume far below
distillation scale. This fans the labeling out over N worker processes, each
running the best policy (dagger_x3_best) + its own MAME instance + the search
teacher, writing a label shard. ~N x throughput -> distillation-scale search
labels feasible (e.g. 12 workers x 20k = 240k labels in ~one single-instance
shard's wall-time).

Then distill: train on the FULL median-8 base (dagger_agg_x3) + the merged
search labels (blended DUP=1) — see exit_dagger / sweep_dup for the distill half.

Usage: .venv/bin/python3 parallel_search_collect.py [n_workers] [per_worker] [H] [base_port] [tag]
Merged labels -> demos/psearch_<tag>.npz
"""
import sys, time
import multiprocessing as mp
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent
MODEL = str(ROOT / "models/dagger_x3_best/final_model.zip")
VEC   = str(ROOT / "models/dagger_x3_best/vec_normalize.pkl")

N        = int(sys.argv[1]) if len(sys.argv) > 1 else 12
PER      = int(sys.argv[2]) if len(sys.argv) > 2 else 20_000
H        = int(sys.argv[3]) if len(sys.argv) > 3 else 12
BASEPORT = int(sys.argv[4]) if len(sys.argv) > 4 else 9990
TAG      = sys.argv[5] if len(sys.argv) > 5 else "p1"


def worker(wid, n, port, H, shard_path):
    import sys as _s
    _s.path.insert(0, str(ROOT)); _s.path.insert(0, str(ROOT / "mame_gym"))
    import numpy as np
    import gymnasium as gym
    from gymnasium.spaces import Box, MultiDiscrete
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from mame_robotron_env import MameRobotronEnv
    from search_oracle import search_move_action

    class _D(gym.Env):
        observation_space = Box(-np.inf, np.inf, (945,), np.float32)
        action_space = MultiDiscrete([8, 8])
        def reset(self, *, seed=None, options=None): return np.zeros(945, np.float32), {}
        def step(self, a): return np.zeros(945, np.float32), 0.0, False, False, {}

    venv = VecNormalize.load(VEC, DummyVecEnv([_D])); venv.training = False
    model = PPO.load(MODEL, device="cpu")
    pool = [int(x) for x in (ROOT / "mame_gym/bc_collect_pool.txt").read_text().split(",")]
    env = MameRobotronEnv(rank=wid, base_port=port, frameskip=4, reset_pool=pool, obs_mode="slot")

    ob = np.zeros((n, 945), np.float32); ac = np.zeros((n, 2), np.int32)
    obs, _ = env.reset(); k = 0; t0 = time.time()
    while k < n:
        nobs = venv.normalize_obs(obs.reshape(1, -1).astype(np.float32))[0]
        a, _ = model.predict(nobs, deterministic=False)
        mv, fr, _ = search_move_action(env._bridge, env._last_packet, H=H)
        ob[k] = obs; ac[k] = (mv - 1, fr - 1); k += 1
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            obs, _ = env.reset()
        if k % 2000 == 0:
            print(f"  [w{wid}] {k}/{n} ({k/(time.time()-t0):.1f}/s)", flush=True)
    np.savez_compressed(shard_path, obs=ob, actions=ac)
    env.close()
    print(f"  [w{wid}] DONE {n} -> {shard_path}", flush=True)


def main():
    mp.set_start_method("spawn", force=True)
    shard_dir = ROOT / "demos" / f"psearch_{TAG}_shards"; shard_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    print(f"launching {N} workers x {PER} states (H={H}) ...", flush=True)
    t0 = time.time()
    for wid in range(N):
        sp = str(shard_dir / f"shard_{wid}.npz")
        p = mp.Process(target=worker, args=(wid, PER, BASEPORT + wid * 2, H, sp))
        p.start(); procs.append(p)
    for p in procs:
        p.join()
    # merge shards
    obs_all, act_all = [], []
    for wid in range(N):
        sp = shard_dir / f"shard_{wid}.npz"
        if sp.exists():
            d = np.load(sp); obs_all.append(d["obs"]); act_all.append(d["actions"])
        else:
            print(f"  WARN: shard {wid} missing", flush=True)
    obs = np.concatenate(obs_all); act = np.concatenate(act_all)
    out = ROOT / "demos" / f"psearch_{TAG}.npz"
    np.savez_compressed(out, obs=obs, actions=act)
    dt = time.time() - t0
    print(f"\nMERGED {len(obs):,} search labels -> {out}  in {dt/60:.1f} min "
          f"({len(obs)/dt:.1f}/s aggregate)", flush=True)


if __name__ == "__main__":
    main()
