"""train_value.py — death-risk VALUE model for the value-guided search teacher.

Trains V(obs) ~ frames-until-death (clamped to HORIZON) on demos/survival_<tag>.npz from
collect_survival.py. The search teacher scores a candidate move's H-frame-rollout endpoint
by V(endpoint) (higher = safer / survives longer under FSM play) instead of raw survival+
score — fixing the rollout-policy != play-policy mismatch that made plain search play wave 3-4.

Output: models/value_<tag>/  (state_dict.pt + norm.npz with obs mean/var + HORIZON).
Usage: .venv/bin/python train_value.py [tag]   (reads demos/survival_<tag>.npz)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
import numpy as np
import torch as th
import torch.nn as nn

TAG = sys.argv[1] if len(sys.argv) > 1 else "v2b"
SRC = ROOT / "demos" / f"survival_{TAG}.npz"
OUT = ROOT / "models" / f"value_{TAG}"
DEVICE = "cuda" if th.cuda.is_available() else "cpu"
EPOCHS = 25
BATCH = 2048


class ValueNet(nn.Module):
    def __init__(self, d=945):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 512), nn.ReLU(),
                                 nn.Linear(512, 512), nn.ReLU(),
                                 nn.Linear(512, 1), nn.Sigmoid())  # -> [0,1] = ttl/HORIZON

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    d = np.load(SRC)
    obs = np.asarray(d["obs"], dtype=np.float32)
    ttl = np.asarray(d["ttl"], dtype=np.float32)
    HORIZON = float(ttl.max())
    y = ttl / HORIZON                       # normalize target to [0,1]
    mean = obs.mean(0); var = obs.var(0) + 1e-8
    obs_n = (obs - mean) / np.sqrt(var)
    n = len(obs); ntr = int(n * 0.95)
    perm = np.random.default_rng(0).permutation(n)
    tr, va = perm[:ntr], perm[ntr:]
    ot = th.as_tensor(obs_n); yt = th.as_tensor(y)
    model = ValueNet().to(DEVICE)
    opt = th.optim.Adam(model.parameters(), lr=3e-4)
    lossf = nn.MSELoss()
    print(f"train_value {TAG}: {n} states, HORIZON={HORIZON:.0f}, "
          f"{100*(ttl<HORIZON).mean():.0f}% in danger zone", flush=True)
    for ep in range(EPOCHS):
        model.train()
        idx = tr[th.randperm(len(tr)).numpy()]
        for i in range(0, len(idx), BATCH):
            b = idx[i:i + BATCH]
            xb = ot[b].to(DEVICE); yb = yt[b].to(DEVICE)
            loss = lossf(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        # val
        model.eval()
        with th.no_grad():
            pv = model(ot[va].to(DEVICE)).cpu().numpy()
        yv = y[va]
        vmse = float(((pv - yv) ** 2).mean())
        corr = float(np.corrcoef(pv, yv)[0, 1])
        if ep % 5 == 0 or ep == EPOCHS - 1:
            print(f"  ep {ep}: val_mse={vmse:.4f} corr={corr:.3f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    th.save(model.state_dict(), OUT / "state_dict.pt")
    np.savez(OUT / "norm.npz", mean=mean, var=var, horizon=HORIZON)
    print(f"saved -> {OUT} (val corr {corr:.3f})", flush=True)


if __name__ == "__main__":
    main()
