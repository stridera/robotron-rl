"""exit_dagger.py — Expert Iteration (search-as-teacher) on the MAME gym.

DAgger plateaued at the FSM-fidelity ceiling (robust median wave 8). This loop
replaces the FSM labeler with the validated policy-improvement SEARCH teacher
(mame_gym/search_oracle.search_move_action) — which beats the FSM (+2 waves in
A/B) — and distills its labels with the same train_bc/evaluate machinery as
dagger.py. Goal: push the policy past median 8 toward wave 100
(see memory project_goal_wave100).

Iterate: train BC on the aggregate -> policy runs in MAME (visits its OWN
states) -> SEARCH labels each visited state (8-move lookahead, FSM rollout, score
by survival) -> aggregate -> retrain. The aggregate is seeded from a SUBSAMPLE of
the DAgger x3 aggregate so (a) RAM/training stay light and (b) the slower search
labels are not drowned by millions of FSM labels; over iters the search labels
accumulate and dominate.

Usage: .venv/bin/python3 exit_dagger.py [iters] [collect_per_iter] [H] [port] [start_n] [tag]
Resumable aggregate -> demos/exit_agg_<tag>.npz. Best -> models/exit_<tag>_best/.
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
from search_oracle import search_move_action

# Defaults; the CLI overrides these in main(). Kept import-safe (no argv parsing
# at module level) so other scripts can `import exit_dagger` to reuse train_bc/
# evaluate/_seed_base without their own argv being misread.
ITERS, COLLECT, H, PORT, START_N, TAG = 5, 12_000, 12, 9978, 1_000_000, "v1"
SEED_AGG = "demos/dagger_agg_x3.npz"     # the median-8 DAgger aggregate to seed from
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
    model = PPO("MlpPolicy", venv, device=DEVICE, verbose=0, policy_kwargs={"net_arch": NET})
    pol = model.policy
    # MEMORY: normalize per-batch on GPU (no full-size normalized copy) — same fix
    # as dagger.py to avoid the WSL OOM at multi-million-state aggregates.
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
    """Policy drives (visits its own states); the SEARCH teacher labels each
    visited state. Labels stored in policy convention (0..7)."""
    ob = np.zeros((n, 945), np.float32); ac = np.zeros((n, 2), np.int32)
    obs, _ = env.reset(); k = 0
    while k < n:
        a, _ = model.predict(_norm(venv, obs), deterministic=False)   # policy drives
        mv, fr, _ = search_move_action(env._bridge, env._last_packet, H=H)  # search labels
        ob[k] = obs; ac[k] = (mv - 1, fr - 1)                         # server 1..8 -> policy 0..7
        k += 1
        obs, _, t, tr, _ = env.step(a)
        if t or tr:
            obs, _ = env.reset()
        if k % 1000 == 0:
            print(f"  collected {k}/{n}", flush=True)
    return ob, ac


def evaluate(env, model, venv, n=15):
    saved_pool = env._reset_pool
    env._reset_pool = [0]
    waves = []
    for _ in range(n):
        obs, _ = env.reset(); mw = env._last_wave
        while True:
            a, _ = model.predict(_norm(venv, obs), deterministic=True)
            obs, _, t, tr, info = env.step(a); mw = max(mw, info.get("wave", 0))
            if t or tr:
                break
        waves.append(mw)
    env._reset_pool = saved_pool
    return waves


DUP = 1    # blend search labels 1:1 with the FSM base. Sweep (2026-06-18) showed
           # DUP=1 lifts median 6->8 (19/21 at w8); DUP>=5 OVERFITS the narrow
           # search set and collapses (median 5 then 4). Do NOT upweight.


def _seed_base():
    d = np.load(SEED_AGG)
    o = np.asarray(d["obs"], dtype=np.float32); a = np.asarray(d["actions"], dtype=np.int32)
    if len(o) > START_N:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(o), START_N, replace=False)
        o, a = o[idx], a[idx]
    return o, a


def main():
    global ITERS, COLLECT, H, PORT, START_N, TAG
    if len(sys.argv) > 1: ITERS   = int(sys.argv[1])
    if len(sys.argv) > 2: COLLECT = int(sys.argv[2])
    if len(sys.argv) > 3: H       = int(sys.argv[3])
    if len(sys.argv) > 4: PORT    = int(sys.argv[4])
    if len(sys.argv) > 5: START_N = int(sys.argv[5])
    if len(sys.argv) > 6: TAG     = sys.argv[6]
    pool = [int(x) for x in (Path("mame_gym/bc_collect_pool.txt").read_text()).split(",")]
    env = MameRobotronEnv(rank=0, base_port=PORT, frameskip=4, reset_pool=pool, obs_mode="slot")
    base_o, base_a = _seed_base()                          # FIXED FSM base (competence/anti-forget)
    print(f"FSM base: {len(base_o):,} (from {SEED_AGG})")
    sa_path = Path(f"demos/exit_search_{TAG}.npz")
    if sa_path.exists():                                   # accumulated SEARCH labels (resumable)
        d = np.load(sa_path)
        sea_o = np.asarray(d["obs"], dtype=np.float32); sea_a = np.asarray(d["actions"], dtype=np.int32)
        print(f"RESUMING search labels {sa_path} ({len(sea_o):,})")
    else:
        sea_o = np.zeros((0, 945), np.float32); sea_a = np.zeros((0, 2), np.int32)

    def training_set():
        # FSM base + DUP copies of the (better) search labels so they carry weight.
        if len(sea_o) == 0:
            return base_o, base_a
        o = np.concatenate([base_o, np.tile(sea_o, (DUP, 1))])
        a = np.concatenate([base_a, np.tile(sea_a, (DUP, 1))])
        return o, a

    best_med, best = -1, None
    for it in range(0, ITERS + 1):
        to, ta = training_set()
        model, venv = train_bc(to, ta)
        waves = evaluate(env, model, venv)
        med = statistics.median(waves)
        print(f"[exit iter {it}] train={len(to):,} (search={len(sea_o):,}x{DUP})  "
              f"eval median={med} mean={statistics.mean(waves):.1f} "
              f"dist={dict(sorted(Counter(waves).items()))}", flush=True)
        out = Path(f"models/exit_{TAG}_{it}"); out.mkdir(parents=True, exist_ok=True)
        model.save(str(out / "final_model")); venv.save(str(out / "vec_normalize.pkl"))
        if med > best_med:
            best_med, best = med, it
            bo = Path(f"models/exit_{TAG}_best"); bo.mkdir(parents=True, exist_ok=True)
            model.save(str(bo / "final_model")); venv.save(str(bo / "vec_normalize.pkl"))
        if it < ITERS:
            co, ca = collect(env, model, venv, COLLECT)
            sea_o = np.concatenate([sea_o, co]); sea_a = np.concatenate([sea_a, ca])
            np.savez_compressed(sa_path, obs=sea_o, actions=sea_a)
    env.close()
    print(f"\nExIt done. best median wave={best_med} at iter {best} -> models/exit_{TAG}_best/")


if __name__ == "__main__":
    main()
