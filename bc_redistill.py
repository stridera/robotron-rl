"""bc_redistill.py — BC a fresh CLEAN clone of an RL-improved policy (re-distillation).

Anchored RL gave a one-shot climb (bc_bignet 8.1 -> bcanchor_best det 9.8) but RL-on-RL
collapses. To iterate, distill bcanchor_best's play (collect_policy_rollouts.py ->
demos/redistill_<tag>_shards) into a FRESH clean BC net. The clean clone removes RL
fragility; one-shot anchored RL from it can then climb again. Mirrors bc_slot_bignet:
MlpPolicy [1024,1024,512] + VecNorm(norm_obs from data), per-batch GPU norm (RAM-safe).

Usage: .venv/bin/python3 bc_redistill.py [epochs] [shard_tag] [out_tag] [net]
"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "mame_gym"))
import numpy as np
import torch as th
import gymnasium as gym
from gymnasium.spaces import Box, MultiDiscrete
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

EPOCHS    = int(sys.argv[1]) if len(sys.argv) > 1 else 16
SHARD_TAG = sys.argv[2] if len(sys.argv) > 2 else "v1"
OUT_TAG   = sys.argv[3] if len(sys.argv) > 3 else "v1"
NET = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [1024, 1024, 512]
DEVICE = "cuda" if th.cuda.is_available() else "cpu"
DIM = 945; BATCH = 1024
SHARD_DIR = ROOT / "demos" / f"redistill_{SHARD_TAG}_shards"
OUT = ROOT / "models" / f"bc_redistill_{OUT_TAG}"


class _Dummy(gym.Env):
    observation_space = Box(-np.inf, np.inf, (DIM,), np.float32)
    action_space = MultiDiscrete([8, 8])
    def reset(self, *, seed=None, options=None): return np.zeros(DIM, np.float32), {}
    def step(self, a): return np.zeros(DIM, np.float32), 0.0, False, False, {}


def main():
    shards = sorted(SHARD_DIR.glob("shard_*.npz"))
    if not shards:
        print(f"no shards in {SHARD_DIR}"); sys.exit(1)
    ol, al = [], []
    for sp in shards:
        with np.load(sp) as d:
            ol.append(np.asarray(d["obs"], dtype=np.float32)); al.append(np.asarray(d["actions"]).astype(np.int64))
    obs = np.concatenate(ol); act = np.concatenate(al); del ol, al
    print(f"{len(shards)} shards, {len(obs):,} redistill demos ({obs.nbytes/1e9:.1f}GB) net={NET} device={DEVICE}", flush=True)

    venv = DummyVecEnv([_Dummy]); venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    venv.obs_rms.mean = obs.mean(0).astype(np.float64); venv.obs_rms.var = obs.var(0).astype(np.float64) + 1e-8
    venv.obs_rms.count = float(len(obs))
    model = PPO("MlpPolicy", venv, device=DEVICE, verbose=0, policy_kwargs={"net_arch": NET})
    pol = model.policy
    mean = th.as_tensor(venv.obs_rms.mean, dtype=th.float32, device=DEVICE)
    std = th.sqrt(th.as_tensor(venv.obs_rms.var, dtype=th.float32, device=DEVICE) + venv.epsilon)
    clip = float(venv.clip_obs)
    ot = th.as_tensor(obs); at = th.as_tensor(act)
    opt = th.optim.Adam(pol.parameters(), lr=3e-4)
    n = len(ot)
    for ep in range(EPOCHS):
        pol.train(); perm = th.randperm(n); el = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            xb = th.clamp((ot[idx].to(DEVICE) - mean) / std, -clip, clip)
            _, lp, _ = pol.evaluate_actions(xb, at[idx].to(DEVICE))
            loss = -lp.mean(); opt.zero_grad(); loss.backward()
            th.nn.utils.clip_grad_norm_(pol.parameters(), 0.5); opt.step(); el += float(loss)*len(idx)
        print(f"[bc_redistill ep {ep+1}/{EPOCHS}] nll={el/n:.4f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    model.save(str(OUT / "final_model")); venv.save(str(OUT / "vec_normalize.pkl"))
    print(f"saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
