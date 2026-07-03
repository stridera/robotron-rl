"""Controlled test: does distilling the search labels help at ANY upweight?

v1 regressed (median 7->3) when 12k search labels were upweighted x20. Reuse
those SAVED labels (no re-collection) + the same 1.5M FSM base, train at several
DUP factors, and eval each. DUP=0 is the base (should ~median 7). If no DUP
beats the base, search-distillation-via-upweight is the wrong integration.
"""
import sys
sys.path.insert(0, "mame_gym")
import numpy as np, statistics
from collections import Counter

import exit_dagger as ed                      # safe: run with no args -> ed uses defaults
from mame_robotron_env import MameRobotronEnv

ed.START_N = 1_500_000                          # match v1's base subsample (seed 0)
base_o, base_a = ed._seed_base()
d = np.load("demos/exit_search_v1.npz")
sea_o = d["obs"].astype(np.float32); sea_a = d["actions"].astype(np.int32)
print(f"base={len(base_o):,}  search={len(sea_o):,}", flush=True)

pool = [int(x) for x in open("mame_gym/bc_collect_pool.txt").read().split(",")]
env = MameRobotronEnv(rank=0, base_port=9981, frameskip=4, reset_pool=pool, obs_mode="slot")
env.reset()

for DUP in [0, 1, 5, 20]:
    if DUP == 0:
        o, a = base_o, base_a
    else:
        o = np.concatenate([base_o, np.tile(sea_o, (DUP, 1))])
        a = np.concatenate([base_a, np.tile(sea_a, (DUP, 1))])
    model, venv = ed.train_bc(o, a)
    waves = ed.evaluate(env, model, venv, n=21)
    print(f"DUP={DUP:2d} train={len(o):>9,} median={statistics.median(waves)} "
          f"mean={statistics.mean(waves):.1f} dist={dict(sorted(Counter(waves).items()))}",
          flush=True)
env.close()
print("sweep done", flush=True)
