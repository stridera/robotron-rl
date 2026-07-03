"""distill_psearch.py — distill parallel-collected search labels onto a strong base.

One-shot ExIt distillation: train on a subsample of the median-8 DAgger aggregate
(strong base) blended 1:1 (DUP) with the parallel-collected search labels, save the
model, and print a quick (noisy) in-process eval. The REAL verdict is an independent
stochastic eval (mame_gym/eval_continuous.py ... 25) run separately afterward.

Usage: .venv/bin/python3 distill_psearch.py [base_n] [dup] [port] [psearch_npz] [tag]
"""
import sys
from pathlib import Path
sys.path.insert(0, "mame_gym")
import numpy as np, statistics
from collections import Counter

import exit_dagger as ed                       # safe: no args -> ed uses defaults
from mame_robotron_env import MameRobotronEnv

BASE_N  = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
DUP     = int(sys.argv[2]) if len(sys.argv) > 2 else 1
PORT    = int(sys.argv[3]) if len(sys.argv) > 3 else 9981
PSEARCH = sys.argv[4] if len(sys.argv) > 4 else "demos/psearch_p1.npz"
TAG     = sys.argv[5] if len(sys.argv) > 5 else "p1"

ed.START_N = BASE_N
base_o, base_a = ed._seed_base()
d = np.load(PSEARCH)
so = d["obs"].astype(np.float32); sa = d["actions"].astype(np.int32)
print(f"base={len(base_o):,}  psearch={len(so):,}  DUP={DUP}", flush=True)

o = np.concatenate([base_o, np.tile(so, (DUP, 1))])
a = np.concatenate([base_a, np.tile(sa, (DUP, 1))])
model, venv = ed.train_bc(o, a)

out = Path(f"models/distill_{TAG}"); out.mkdir(parents=True, exist_ok=True)
model.save(str(out / "final_model")); venv.save(str(out / "vec_normalize.pkl"))
print(f"saved -> {out}", flush=True)

pool = [int(x) for x in open("mame_gym/bc_collect_pool.txt").read().split(",")]
env = MameRobotronEnv(rank=0, base_port=PORT, frameskip=4, reset_pool=pool, obs_mode="slot")
env.reset()
waves = ed.evaluate(env, model, venv, n=21)
print(f"distill_{TAG} IN-PROCESS(noisy) median={statistics.median(waves)} "
      f"mean={statistics.mean(waves):.1f} dist={dict(sorted(Counter(waves).items()))}", flush=True)
env.close()
print("distill done", flush=True)
