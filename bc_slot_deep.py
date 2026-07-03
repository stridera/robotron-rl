"""bc_slot_deep.py — BC the bigger net on shallow aggregate + DEEP-wave demos.

The BC-fidelity gap (obs-limited FSM det 11.4 vs policy ~8) traces to deep-wave data
coverage: bc_collect_pool had ZERO wave-8+ states, so the aggregate undersampled wave
8-12. This adds 1.2M FSM demos collected from wave-8/9 save states (dense deep-wave
supervision) to a subsample of the median-8 aggregate, and BC-trains the [1024,1024,512]
net. RAM-capped: all deep + 3.5M aggregate subsample (~4.7M, ~18GB, under WSL 48GB cap).
Eval deterministically vs bc_bignet (det 8.1) and the FSM ceiling (11.4).

Usage: .venv/bin/python3 bc_slot_deep.py [epochs] [agg_cap] [net]
"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "mame_gym"))

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 14
AGG_CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 4_500_000
DEEP_CAP = int(sys.argv[3]) if len(sys.argv) > 3 else 400_000
NET = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [1024, 1024, 512]
sys.argv = sys.argv[:1]   # blank before importing exit_dagger (fsm_oracle argv parse)

import numpy as np
import exit_dagger as ed

import os
AGG = ROOT / "demos" / "dagger_agg_x3.npz"
DEEP_DIR = ROOT / "demos" / os.environ.get("DEEP_SHARDS", "slotdeep_v1_shards")
OUT = ROOT / "models" / os.environ.get("DEEP_OUT", "bc_deep")
ed.NET = NET; ed.EPOCHS = EPOCHS

# deep demos (all)
deep_o, deep_a = [], []
for sp in sorted(DEEP_DIR.glob("shard_*.npz")):
    with np.load(sp) as d:
        deep_o.append(np.asarray(d["obs"], dtype=np.float32))
        deep_a.append(np.asarray(d["actions"], dtype=np.int32))
deep_o = np.concatenate(deep_o); deep_a = np.concatenate(deep_a)
if len(deep_o) > DEEP_CAP:
    di = np.random.default_rng(1).choice(len(deep_o), DEEP_CAP, replace=False)
    deep_o, deep_a = deep_o[di], deep_a[di]
print(f"deep demos: {len(deep_o):,}", flush=True)

# aggregate subsample
d = np.load(AGG)
ao = np.asarray(d["obs"], dtype=np.float32); aa = np.asarray(d["actions"], dtype=np.int32)
if len(ao) > AGG_CAP:
    idx = np.random.default_rng(0).choice(len(ao), AGG_CAP, replace=False)
    ao, aa = ao[idx], aa[idx]
print(f"aggregate subsample: {len(ao):,}", flush=True)

obs = np.concatenate([ao, deep_o]); act = np.concatenate([aa, deep_a])
del ao, aa, deep_o, deep_a
print(f"combined: {len(obs):,} states ({obs.nbytes/1e9:.1f}GB)  net={NET} epochs={EPOCHS}", flush=True)

model, venv = ed.train_bc(obs, act)
OUT.mkdir(parents=True, exist_ok=True)
model.save(str(OUT / "final_model")); venv.save(str(OUT / "vec_normalize.pkl"))
print(f"saved -> {OUT}", flush=True)
