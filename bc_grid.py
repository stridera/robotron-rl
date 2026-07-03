"""bc_grid.py — behavior-cloning warmstart for the GRID-CNN policy.

From-scratch grid RL walled at mean wave 3 (run 9iwa3zzv) — the same wall pure RL
hits on the slot obs. The slot policy only broke past it via DAgger-on-FSM BC
(dagger_x3_best -> RL -> yiwzpqq7, mean 8.1). This is the grid equivalent: BC-pretrain
the GridCNN policy on the FSM grid demos (demos/grid_v1_shards/), then RL fine-tune
via train_mame.py --obs-mode grid --bc-checkpoint models/grid_bc_v1/final_model.

Architecture MUST match train_mame.py's grid branch so PPO.load works:
  MlpPolicy + features_extractor=GridCNN(features_dim=512) + net_arch=[512,256],
  VecNormalize(norm_obs=False) (grid obs is already well-scaled).

Streams shards one at a time (float16 in RAM, ->float32 on GPU per batch) to respect
the WSL 48GB cap (see project_wsl_memory_cap): 480k grids would be ~37GB if all
decompressed at once.

Usage: .venv/bin/python3 bc_grid.py [epochs] [batch] [shard_glob_tag] [out_tag]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mame_gym"))
import numpy as np
import torch as th
import gymnasium as gym
from gymnasium.spaces import Box, MultiDiscrete
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from spatial_obs import NUM_CHANNELS, GRID_H, GRID_W
from train_mame import GridCNN

EPOCHS    = int(sys.argv[1]) if len(sys.argv) > 1 else 12
BATCH     = int(sys.argv[2]) if len(sys.argv) > 2 else 512
SHARD_TAG = sys.argv[3] if len(sys.argv) > 3 else "v1"
OUT_TAG   = sys.argv[4] if len(sys.argv) > 4 else "v1"
DEVICE = "cuda" if th.cuda.is_available() else "cpu"
SHARD_DIR = ROOT / "demos" / f"grid_{SHARD_TAG}_shards"
OUT = ROOT / "models" / f"grid_bc_{OUT_TAG}"


class _DummyGrid(gym.Env):
    observation_space = Box(-np.inf, np.inf, (NUM_CHANNELS, GRID_H, GRID_W), np.float32)
    action_space = MultiDiscrete([8, 8])
    def reset(self, *, seed=None, options=None):
        return np.zeros((NUM_CHANNELS, GRID_H, GRID_W), np.float32), {}
    def step(self, a):
        return np.zeros((NUM_CHANNELS, GRID_H, GRID_W), np.float32), 0.0, False, False, {}


def main():
    shards = sorted(SHARD_DIR.glob("shard_*.npz"))
    if not shards:
        print(f"no shards in {SHARD_DIR}"); sys.exit(1)
    n_total = 0
    for sp in shards:
        with np.load(sp) as d:
            n_total += len(d["actions"])
    print(f"{len(shards)} shards, {n_total:,} grid demos  device={DEVICE}", flush=True)

    venv = DummyVecEnv([_DummyGrid])
    # match train_mame grid: norm_obs=False, norm_reward=False, clip_obs=10
    venv = VecNormalize(venv, norm_obs=False, norm_reward=False, clip_obs=10.0)
    model = PPO("MlpPolicy", venv, device=DEVICE, verbose=0,
                policy_kwargs={"features_extractor_class": GridCNN,
                               "features_extractor_kwargs": {"features_dim": 512},
                               "net_arch": [512, 256]})
    pol = model.policy
    opt = th.optim.Adam(pol.parameters(), lr=3e-4)

    for ep in range(EPOCHS):
        order = np.random.permutation(len(shards))
        ep_loss, ep_n = 0.0, 0
        for si in order:
            with np.load(shards[si]) as d:
                obs = np.asarray(d["obs"])              # (n,11,72,48) float16
                act = np.asarray(d["actions"]).astype(np.int64)
            ot = th.from_numpy(obs); at = th.from_numpy(act)
            n = len(ot)
            perm = th.randperm(n)
            pol.train()
            for i in range(0, n, BATCH):
                idx = perm[i:i + BATCH]
                xb = ot[idx].to(DEVICE, dtype=th.float32)
                ab = at[idx].to(DEVICE)
                _, lp, _ = pol.evaluate_actions(xb, ab)
                loss = -lp.mean()
                opt.zero_grad(); loss.backward()
                th.nn.utils.clip_grad_norm_(pol.parameters(), 0.5); opt.step()
                ep_loss += float(loss) * len(idx); ep_n += len(idx)
            del ot, at, obs, act
        print(f"[bc_grid ep {ep+1}/{EPOCHS}] nll={ep_loss/ep_n:.4f} (n={ep_n:,})", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    model.save(str(OUT / "final_model")); venv.save(str(OUT / "vec_normalize.pkl"))
    print(f"saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
