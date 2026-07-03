"""bc_slot_bignet.py — test whether policy CAPACITY is the BC-fidelity bottleneck.

The obs-limited FSM (what BC clones) plays to mean wave 11.4 on the slot obs, but our
best cloned policy (dagger_x3_best, [512,512] MLP) only reaches ~8 — a ~3-4 wave BC-
fidelity gap (EXPERIMENT_STATE 2026-06-19). Hypothesis: a [512,512] net can't capture
the FSM's deep-wave decision precision. Re-BC the SAME median-8 aggregate with a bigger,
deeper net and eval deterministically. If it closes the gap (->10-11), capacity was the
limit and we have a new best on the existing obs; if it plateaus ~8, the gap is elsewhere.

Usage: .venv/bin/python3 bc_slot_bignet.py [epochs] [net]   net e.g. 1024,1024,512
"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "mame_gym"))

# parse OUR args, then blank sys.argv before importing exit_dagger (it transitively
# imports fsm_oracle, which parses sys.argv at module load and chokes on our net arg).
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 14
NET = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1024, 1024, 512]
sys.argv = sys.argv[:1]

import numpy as np
import exit_dagger as ed
AGG = ROOT / "demos" / "dagger_agg_x3.npz"
OUT = ROOT / "models" / "bc_bignet"

ed.NET = NET
ed.EPOCHS = EPOCHS
print(f"BC bignet: net={NET} epochs={EPOCHS} agg={AGG.name}", flush=True)
d = np.load(AGG)
obs = np.asarray(d["obs"], dtype=np.float32); act = np.asarray(d["actions"], dtype=np.int32)
print(f"loaded {len(obs):,} states ({obs.nbytes/1e9:.1f}GB)", flush=True)
model, venv = ed.train_bc(obs, act)
OUT.mkdir(parents=True, exist_ok=True)
model.save(str(OUT / "final_model")); venv.save(str(OUT / "vec_normalize.pkl"))
print(f"saved -> {OUT}", flush=True)
