"""collect_policy_rollouts.py — collect a policy's own play for RE-DISTILLATION.

Anchored RL gives a ONE-SHOT climb from a clean BC base (bc_bignet 8.1 -> bcanchor_best
det 9.8) but RL-on-RL-tuned collapses (v5 -> 2.2). To iterate the gain (AlphaZero-style),
distill the RL-improved policy back into a FRESH clean BC network, then one-shot anchored
RL again from that cleaner base. Step 1 (here): record the policy's (obs, det-action) pairs
as it plays from wave 1, so bc can clone it. eps-random STEPS give state diversity, but the
recorded LABEL is always the policy's deterministic action (clean target).

Usage: .venv/bin/python3 collect_policy_rollouts.py [model_dir] [n_workers] [per_worker] [base_port] [epsilon] [tag]
  model_dir e.g. bcanchor_best  -> demos/redistill_<tag>_shards/
"""
import sys
import multiprocessing as mp
from pathlib import Path

ROOT = Path(__file__).parent
MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "bcanchor_best"
N        = int(sys.argv[2]) if len(sys.argv) > 2 else 12
PER      = int(sys.argv[3]) if len(sys.argv) > 3 else 100_000
BASEPORT = int(sys.argv[4]) if len(sys.argv) > 4 else 9960
EPSILON  = float(sys.argv[5]) if len(sys.argv) > 5 else 0.08
TAG      = sys.argv[6] if len(sys.argv) > 6 else "v1"


def worker(wid, n, port, epsilon, model_dir, shard_path):
    import sys as _s
    _s.path.insert(0, str(ROOT)); _s.path.insert(0, str(ROOT / "mame_gym"))
    import time
    import numpy as np
    import gymnasium as gym
    from gymnasium.spaces import Box, MultiDiscrete
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from mame_robotron_env import MameRobotronEnv

    class _D(gym.Env):
        observation_space = Box(-np.inf, np.inf, (945,), np.float32)
        action_space = MultiDiscrete([8, 8])
        def reset(self, *, seed=None, options=None): return np.zeros(945, np.float32), {}
        def step(self, a): return np.zeros(945, np.float32), 0.0, False, False, {}

    md = ROOT / "models" / model_dir
    venv = VecNormalize.load(str(md / "vec_normalize.pkl"), DummyVecEnv([_D])); venv.training = False
    model = PPO.load(str(md / "final_model"), device="cpu")
    rng = np.random.default_rng(7000 + wid)
    env = MameRobotronEnv(rank=wid, base_port=port, frameskip=4, reset_pool=[0], obs_mode="slot")

    ob = np.zeros((n, 945), np.float32); ac = np.zeros((n, 2), np.int8)
    obs, _ = env.reset(); k = 0; t0 = time.time()
    while k < n:
        nobs = venv.normalize_obs(obs.reshape(1, -1).astype(np.float32))[0]
        a, _ = model.predict(nobs, deterministic=True)          # clean det LABEL
        ob[k] = obs; ac[k] = (int(a[0]), int(a[1])); k += 1
        if rng.random() < epsilon:                               # diversity in STATES visited
            step_a = np.array([rng.integers(0, 8), rng.integers(0, 8)])
        else:
            step_a = a
        obs, _, term, trunc, _ = env.step(step_a)
        if term or trunc:
            obs, _ = env.reset()
        if k % 5000 == 0:
            print(f"  [w{wid}] {k}/{n} ({k/(time.time()-t0):.1f}/s)", flush=True)
    np.savez_compressed(shard_path, obs=ob, actions=ac)
    env.close()
    print(f"  [w{wid}] DONE {n} -> {shard_path}", flush=True)


def main():
    import time
    mp.set_start_method("spawn", force=True)
    shard_dir = ROOT / "demos" / f"redistill_{TAG}_shards"; shard_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    print(f"redistill collect: {MODEL_DIR} x {N} workers x {PER} (eps={EPSILON}) -> {shard_dir}", flush=True)
    t0 = time.time()
    for wid in range(N):
        sp = str(shard_dir / f"shard_{wid}.npz")
        p = mp.Process(target=worker, args=(wid, PER, BASEPORT + wid * 2, EPSILON, MODEL_DIR, sp))
        p.start(); procs.append(p)
    for p in procs:
        p.join()
    total = sum(PER for wid in range(N) if (shard_dir / f"shard_{wid}.npz").exists())
    print(f"\nDONE {total:,} redistill demos -> {shard_dir} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
