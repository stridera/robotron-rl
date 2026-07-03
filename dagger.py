"""dagger.py — DAgger loop to fix BC covariate shift with the FSM as expert.

Iterates: train BC on the aggregated dataset -> run the policy in MAME (it visits its
OWN states, including mistakes) -> label every visited obs with the FSM (decode obs ->
chooseOutputs) -> aggregate -> retrain. Each iteration the policy learns to recover
from the states it actually reaches. Evals continuous-from-wave-1 each iter.

Usage: .venv/bin/python3 dagger.py [iters] [collect_per_iter] [port]
Best policy saved to models/dagger_best/.
"""
import sys
import statistics
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "mame_gym"))
import numpy as np
import torch as th
import gymnasium as gym
from gymnasium.spaces import Box, MultiDiscrete
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from mame_robotron_env import MameRobotronEnv
from obs_decode import fsm_action_from_obs

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
COLLECT = int(sys.argv[2]) if len(sys.argv) > 2 else 60_000
PORT = int(sys.argv[3]) if len(sys.argv) > 3 else 9982
EXTRACTOR = sys.argv[4] if len(sys.argv) > 4 else "mlp"   # 'mlp' or 'attn'
TAG = sys.argv[5] if len(sys.argv) > 5 else EXTRACTOR
RESUME = sys.argv[6] if len(sys.argv) > 6 else ""         # path to a saved aggregate npz
NET = [512, 512]
DEVICE = "cuda" if th.cuda.is_available() else "cpu"
EPOCHS = 18
BATCH = 2048


class _Dummy(gym.Env):
    observation_space = Box(-np.inf, np.inf, (945,), np.float32)
    action_space = MultiDiscrete([8, 8])
    def reset(self, *, seed=None, options=None): return np.zeros(945, np.float32), {}
    def step(self, a): return np.zeros(945, np.float32), 0.0, False, False, {}


def train_bc(obs, act):
    venv = DummyVecEnv([_Dummy])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    venv.obs_rms.mean = obs.mean(0).astype(np.float64)
    venv.obs_rms.var = obs.var(0).astype(np.float64) + 1e-8
    venv.obs_rms.count = float(len(obs))
    if EXTRACTOR == "attn":
        from slot_policy import SlotAttnExtractor
        pk = {"features_extractor_class": SlotAttnExtractor,
              "features_extractor_kwargs": {"features_dim": 256}, "net_arch": [256]}
    else:
        pk = {"net_arch": NET}
    model = PPO("MlpPolicy", venv, device=DEVICE, verbose=0, policy_kwargs=pk)
    pol = model.policy
    # MEMORY: do NOT materialize a normalized full-size copy of obs. At 3.6M x 945 float32
    # the raw aggregate is ~13.6GB; a second normalized copy held alongside it through the
    # whole epoch loop (~27GB obs resident) is what OOM-crashed WSL on the x3 resume iter 1.
    # Keep ONLY the raw obs on CPU (as_tensor shares the numpy buffer, no copy) and apply
    # VecNormalize's exact transform per-batch on the GPU -- math-identical to normalize_obs.
    mean = th.as_tensor(venv.obs_rms.mean, dtype=th.float32, device=DEVICE)
    std = th.sqrt(th.as_tensor(venv.obs_rms.var, dtype=th.float32, device=DEVICE) + venv.epsilon)
    clip = float(venv.clip_obs)
    ot = th.as_tensor(obs); at = th.as_tensor(act.astype(np.int64))
    opt = th.optim.Adam(pol.parameters(), lr=3e-4)
    n = len(ot)
    for ep in range(EPOCHS):
        pol.train()
        perm = th.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb = th.clamp((ot[idx].to(DEVICE) - mean) / std, -clip, clip)
            _, lp, _ = pol.evaluate_actions(xb, at[idx].to(DEVICE))
            loss = -lp.mean()
            opt.zero_grad(); loss.backward()
            th.nn.utils.clip_grad_norm_(pol.parameters(), 0.5); opt.step()
    return model, venv


def _norm(venv, obs):
    return venv.normalize_obs(obs.reshape(1, -1).astype(np.float32))[0]


def collect(env, model, venv, n):
    ob = np.zeros((n, 945), np.float32); ac = np.zeros((n, 2), np.int32)
    obs, _ = env.reset(); k = 0
    while k < n:
        a, _ = model.predict(_norm(venv, obs), deterministic=False)   # policy drives (stochastic)
        ob[k] = obs; ac[k] = fsm_action_from_obs(obs)                 # expert labels its state
        k += 1
        obs, _, t, tr, _ = env.step(a)
        if t or tr:
            obs, _ = env.reset()
    return ob, ac


def evaluate(env, model, venv, n=15):
    # TRUE continuous metric: force wave-1 boots (reset_pool=[0]), not the diverse
    # collection pool — otherwise episodes start at deep states and inflate the wave.
    saved_pool = env._reset_pool
    env._reset_pool = [0]
    waves = []
    for _ in range(n):
        obs, _ = env.reset(); mw = env._last_wave
        while True:
            a, _ = model.predict(_norm(venv, obs), deterministic=True)   # precise metric (MLP doesn't lock)
            obs, _, t, tr, info = env.step(a); mw = max(mw, info.get("wave", 0))
            if t or tr:
                break
        waves.append(mw)
    env._reset_pool = saved_pool
    return waves


def main():
    pool = [int(x) for x in (Path("mame_gym/bc_collect_pool.txt").read_text()).split(",")]
    env = MameRobotronEnv(rank=0, base_port=PORT, frameskip=4, reset_pool=pool, obs_mode="slot")
    agg_path = Path(f"demos/dagger_agg_{TAG}.npz")
    if RESUME and Path(RESUME).exists():       # resume from a saved aggregate
        d = np.load(RESUME); print(f"RESUMING from {RESUME} ({len(d['obs']):,} states)")
    else:
        d = np.load("demos/fsm_mame_demos.npz")
    # np.asarray avoids a copy when dtype already matches (npz obs is float32) -- the old
    # .astype copied another ~13.6GB at load. The remaining transient peak is the per-iter
    # np.concatenate below (old + result, ~27GB for a few seconds), well under the WSL cap.
    agg_o = np.asarray(d["obs"], dtype=np.float32); agg_a = np.asarray(d["actions"], dtype=np.int32)
    best_med, best = -1, None
    for it in range(0, ITERS + 1):
        model, venv = train_bc(agg_o, agg_a)
        waves = evaluate(env, model, venv)
        med = statistics.median(waves)
        print(f"[dagger iter {it}] agg={len(agg_o):,}  eval median={med} mean={statistics.mean(waves):.1f} "
              f"dist={dict(sorted(Counter(waves).items()))}", flush=True)
        out = Path(f"models/dagger_{TAG}_{it}"); out.mkdir(parents=True, exist_ok=True)
        model.save(str(out / "final_model")); venv.save(str(out / "vec_normalize.pkl"))
        if med > best_med:
            best_med, best = med, it
            bo = Path(f"models/dagger_{TAG}_best"); bo.mkdir(parents=True, exist_ok=True)
            model.save(str(bo / "final_model")); venv.save(str(bo / "vec_normalize.pkl"))
        if it < ITERS:
            co, ca = collect(env, model, venv, COLLECT)
            agg_o = np.concatenate([agg_o, co]); agg_a = np.concatenate([agg_a, ca])
            np.savez_compressed(agg_path, obs=agg_o, actions=agg_a)  # checkpoint aggregate
    env.close()
    print(f"\nDAgger done. best median wave={best_med} at iter {best} -> models/dagger_best/")


if __name__ == "__main__":
    main()
