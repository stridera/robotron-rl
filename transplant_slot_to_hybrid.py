"""transplant_slot_to_hybrid.py — graft the strong slot policy into a hybrid policy.

RL doesn't climb over a weak BC on this gym (grid/hybrid BC->RL stayed flat); results
are BC-capped at FSM fidelity (~8, reached by the slot policy dagger_x3_best -> yiwzpqq7).
Rebuilding a strong hybrid BC via DAgger is slow. Instead: TRANSPLANT the strong slot
policy (dagger_x3_best, median 8) into a hybrid (957-dim) policy so it STARTS at 8, then
RL can learn to use the 12 extra global-threat features to exceed 8.

Surgery: hybrid = slot for all params; the first Linear of policy_net & value_net is
[512,957] vs slot's [512,945] -> copy slot into [:, :945], ZERO [:, 945:957]. With zero
weights on the global features they contribute nothing, so the hybrid is bit-identical to
slot at init (median 8). VecNormalize: first 945 stats from slot, last 12 from the hybrid
demo data. Output loads via train_mame.py --obs-mode hybrid --bc-checkpoint.

Usage: .venv/bin/python3 transplant_slot_to_hybrid.py
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

SLOT_DIR = ROOT / "models" / "dagger_x3_best"
SHARD_DIR = ROOT / "demos" / "hybrid_v1_shards"
OUT = ROOT / "models" / "hybrid_transplant"
NET = [512, 512]


class _Dummy(gym.Env):
    observation_space = Box(-np.inf, np.inf, (HYBRID_DIM,), np.float32)
    action_space = MultiDiscrete([8, 8])
    def reset(self, *, seed=None, options=None): return np.zeros(HYBRID_DIM, np.float32), {}
    def step(self, a): return np.zeros(HYBRID_DIM, np.float32), 0.0, False, False, {}


def main():
    slot = PPO.load(str(SLOT_DIR / "final_model"), device="cpu")
    slot_sd = slot.policy.state_dict()

    venv = DummyVecEnv([_Dummy])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    hybrid = PPO("MlpPolicy", venv, device="cpu", verbose=0, policy_kwargs={"net_arch": NET})
    hyb_sd = hybrid.policy.state_dict()

    transplanted, padded, skipped = 0, [], []
    new_sd = {}
    for k, hv in hyb_sd.items():
        if k not in slot_sd:
            new_sd[k] = hv; skipped.append(k); continue
        sv = slot_sd[k]
        if sv.shape == hv.shape:
            new_sd[k] = sv; transplanted += 1
        elif sv.dim() == 2 and hv.dim() == 2 and sv.shape[0] == hv.shape[0] and sv.shape[1] < hv.shape[1]:
            # first Linear: [512,945] -> [512,957]; copy slot, zero the new cols
            w = th.zeros_like(hv); w[:, :sv.shape[1]] = sv
            new_sd[k] = w; padded.append((k, tuple(sv.shape), tuple(hv.shape)))
        else:
            new_sd[k] = hv; skipped.append(k)
    hybrid.policy.load_state_dict(new_sd)
    print(f"transplanted {transplanted} params; padded {len(padded)}: {padded}; skipped {skipped}")

    # VecNormalize: first 945 from slot, last 12 from hybrid demo global features
    slot_venv = VecNormalize.load(str(SLOT_DIR / "vec_normalize.pkl"), DummyVecEnv([_Dummy_slot]))
    smean = slot_venv.obs_rms.mean; svar = slot_venv.obs_rms.var
    shard = sorted(SHARD_DIR.glob("shard_*.npz"))[0]
    with np.load(shard) as d:
        gobs = np.asarray(d["obs"], dtype=np.float64)[:, 945:]
    gmean = gobs.mean(0); gvar = gobs.var(0) + 1e-8
    venv.obs_rms.mean = np.concatenate([smean, gmean])
    venv.obs_rms.var = np.concatenate([svar, gvar])
    venv.obs_rms.count = float(slot_venv.obs_rms.count)
    print(f"vecnorm: slot945 + global12  mean[945:]={venv.obs_rms.mean[945:].round(3)}")

    OUT.mkdir(parents=True, exist_ok=True)
    hybrid.save(str(OUT / "final_model")); venv.save(str(OUT / "vec_normalize.pkl"))
    print(f"saved -> {OUT}")


class _Dummy_slot(gym.Env):
    observation_space = Box(-np.inf, np.inf, (945,), np.float32)
    action_space = MultiDiscrete([8, 8])
    def reset(self, *, seed=None, options=None): return np.zeros(945, np.float32), {}
    def step(self, a): return np.zeros(945, np.float32), 0.0, False, False, {}


if __name__ == "__main__":
    main()
