"""bc_hybrid.py — behavior-cloning the FSM on the HYBRID obs (945 slot + 12 global).

Mirrors the proven slot BC (exit_dagger.train_bc): MlpPolicy [512,512] + VecNormalize
(norm_obs=True, obs_rms fitted from the demo data), per-batch GPU normalization (no
full normalized copy -> WSL-RAM-safe, see project_wsl_memory_cap). Hybrid obs is 957
floats so the full 1.2M-demo set fits in RAM (~4.6GB), unlike the grid (would be 37GB).

Architecture matches train_mame.py's hybrid branch (MlpPolicy net [512,512]) so
train_mame.py --obs-mode hybrid --bc-checkpoint loads it cleanly for RL fine-tune.

The slot part lets BC trivially reproduce the FSM (~the slot ceiling); the 12 global
features are along for the ride (FSM ignores them) — RL then exploits them to flee
open-field threats and push past 8. See EXPERIMENT_STATE / project_goal_wave100.

Usage: .venv/bin/python3 bc_hybrid.py [epochs] [batch] [shard_tag] [out_tag]
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
from hybrid_obs import HYBRID_DIM

EPOCHS    = int(sys.argv[1]) if len(sys.argv) > 1 else 14
BATCH     = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
SHARD_TAG = sys.argv[3] if len(sys.argv) > 3 else "v1"
OUT_TAG   = sys.argv[4] if len(sys.argv) > 4 else "v1"
DEVICE = "cuda" if th.cuda.is_available() else "cpu"
NET = [512, 512]
SHARD_DIR = ROOT / "demos" / f"hybrid_{SHARD_TAG}_shards"
OUT = ROOT / "models" / f"hybrid_bc_{OUT_TAG}"


class _Dummy(gym.Env):
    observation_space = Box(-np.inf, np.inf, (HYBRID_DIM,), np.float32)
    action_space = MultiDiscrete([8, 8])
    def reset(self, *, seed=None, options=None): return np.zeros(HYBRID_DIM, np.float32), {}
    def step(self, a): return np.zeros(HYBRID_DIM, np.float32), 0.0, False, False, {}


def main():
    shards = sorted(SHARD_DIR.glob("shard_*.npz"))
    if not shards:
        print(f"no shards in {SHARD_DIR}"); sys.exit(1)
    obs_list, act_list = [], []
    for sp in shards:
        with np.load(sp) as d:
            obs_list.append(np.asarray(d["obs"], dtype=np.float32))
            act_list.append(np.asarray(d["actions"], dtype=np.int64))
    obs = np.concatenate(obs_list); act = np.concatenate(act_list)
    del obs_list, act_list
    print(f"{len(shards)} shards, {len(obs):,} hybrid demos ({obs.nbytes/1e9:.1f}GB)  device={DEVICE}", flush=True)

    venv = DummyVecEnv([_Dummy])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    venv.obs_rms.mean = obs.mean(0).astype(np.float64)
    venv.obs_rms.var = obs.var(0).astype(np.float64) + 1e-8
    venv.obs_rms.count = float(len(obs))
    model = PPO("MlpPolicy", venv, device=DEVICE, verbose=0, policy_kwargs={"net_arch": NET})
    pol = model.policy
    # per-batch GPU normalization (no full normalized copy) — WSL-RAM-safe
    mean = th.as_tensor(venv.obs_rms.mean, dtype=th.float32, device=DEVICE)
    std = th.sqrt(th.as_tensor(venv.obs_rms.var, dtype=th.float32, device=DEVICE) + venv.epsilon)
    clip = float(venv.clip_obs)
    ot = th.as_tensor(obs); at = th.as_tensor(act)
    opt = th.optim.Adam(pol.parameters(), lr=3e-4)
    n = len(ot)
    for ep in range(EPOCHS):
        pol.train()
        perm = th.randperm(n)
        ep_loss = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb = th.clamp((ot[idx].to(DEVICE) - mean) / std, -clip, clip)
            _, lp, _ = pol.evaluate_actions(xb, at[idx].to(DEVICE))
            loss = -lp.mean()
            opt.zero_grad(); loss.backward()
            th.nn.utils.clip_grad_norm_(pol.parameters(), 0.5); opt.step()
            ep_loss += float(loss) * len(idx)
        print(f"[bc_hybrid ep {ep+1}/{EPOCHS}] nll={ep_loss/n:.4f}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    model.save(str(OUT / "final_model")); venv.save(str(OUT / "vec_normalize.pkl"))
    print(f"saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
