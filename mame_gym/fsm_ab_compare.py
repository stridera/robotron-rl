"""Pair two fsm_ab_eval.py outputs (same seed base) and report the delta + CI."""
import json, sys
import numpy as np
A = json.loads(open(sys.argv[1]).read())
B = json.loads(open(sys.argv[2]).read())
a, b = np.array(A["waves"]), np.array(B["waves"])
n = min(len(a), len(b)); a, b = a[:n], b[:n]
d = b - a   # B (variant) minus A (baseline)
wins = int((d > 0).sum()); losses = int((d < 0).sum()); ties = int((d == 0).sum())
# bootstrap CI on mean paired delta
rng = np.random.default_rng(0)
boot = [d[rng.integers(0, n, n)].mean() for _ in range(2000)]
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"A (baseline): mean {a.mean():.2f} median {int(np.median(a))} max {a.max()}  "
      f"push={A.get('push')} buf={A.get('buffer')}")
print(f"B (variant):  mean {b.mean():.2f} median {int(np.median(b))} max {b.max()}  "
      f"push={B.get('push')} buf={B.get('buffer')}")
print(f"\npaired delta (B-A): mean {d.mean():+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  (n={n})")
print(f"win/tie/loss: {wins}/{ties}/{losses}  ({100*wins/n:.0f}% win)")
print("VERDICT:", "B better (CI>0)" if lo > 0 else ("A better (CI<0)" if hi < 0 else "inconclusive (CI spans 0)"))
