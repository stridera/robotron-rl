"""train_bc_anchored.py — RL that IMPROVES the BC policy instead of forgetting it.

Naive PPO fine-tunes REGRESS strong BC policies (catastrophic forgetting), NOT proof
the game is unlearnable. Fix: alternate self-contained PPO phases with BC "pull-back"
phases toward the FSM demos (the dagger_agg_x3 bc_bignet was trained on). Start from
bc_bignet (det 8.1) and test whether RL can climb PAST wave 8 without regressing.

CORRECT design (v3, after two buggy tries):
- CHUNKED ALTERNATION: model.learn(PPO_CHUNK) is fully self-consistent (rollout collected
  and trained by the same policy), THEN a separate BC phase. (v2 ran BC in on_rollout_end,
  between collection and PPO.train -> broke PPO's on-policy ratio -> destabilized to mean 4.4.)
- SEPARATE BC OPTIMIZER so BC gradients never corrupt PPO's Adam state.
- Checkpoints save vec_normalize too (so evals are exact).

Usage: .venv/bin/python3 train_bc_anchored.py [timesteps] [ppo_chunk] [bc_steps_per_cycle] [bc_lr] [lr] [ent] [port] [tag]
"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "mame_gym"))

TIMESTEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 8_000_000
PPO_CHUNK = int(sys.argv[2]) if len(sys.argv) > 2 else 32_768     # 2 rollouts/cycle
BC_STEPS  = int(sys.argv[3]) if len(sys.argv) > 3 else 150        # BC pull-back steps/cycle
BC_LR     = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-4
LR        = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-4
ENT       = float(sys.argv[6]) if len(sys.argv) > 6 else 0.005
PORT      = int(sys.argv[7]) if len(sys.argv) > 7 else 9940
TAG       = sys.argv[8] if len(sys.argv) > 8 else "v3"
sys.argv = sys.argv[:1]

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from train_mame import make_env

import os
BASE = ROOT / "models" / os.environ.get("ANCHOR_BASE", "bc_bignet")
AGG = ROOT / "demos" / os.environ.get("ANCHOR_AGG", "dagger_agg_x3.npz")
DEMO_N = 1_500_000
OUT = ROOT / "models" / f"bcanchor_{TAG}"
DEVICE = "cuda" if th.cuda.is_available() else "cpu"
NENV = int(os.environ.get("NENV", 8))
BC_BATCH = 512
CK_EVERY = int(os.environ.get("CK_EVERY", 200_000))


def main():
    # CURRICULUM_POOL: comma-separated reset-state indices (e.g. "0,0,30001,30002,...") so
    # training can reset INTO deep captured waves, not just wave-1 boots. Default [0].
    _cp = os.environ.get("CURRICULUM_POOL", "")
    reset_pool = [int(x) for x in _cp.split(",")] if _cp else [0]
    print(f"reset_pool: {len(reset_pool)} idxs, {len(set(reset_pool))} unique (head {reset_pool[:8]})", flush=True)
    envs = SubprocVecEnv([make_env(i, PORT, 4, reset_pool, None, "slot", 0) for i in range(NENV)])
    envs = VecNormalize.load(str(BASE / "vec_normalize.pkl"), envs)
    envs.training = True; envs.norm_reward = False

    model = PPO.load(str(BASE / "final_model"), env=envs, device=DEVICE)
    model.learning_rate = LR; model.ent_coef = ENT; model.lr_schedule = lambda _: LR

    d = np.load(AGG)
    o = np.asarray(d["obs"], dtype=np.float32); a = np.asarray(d["actions"], dtype=np.int64)
    if len(o) > DEMO_N:
        idx = np.random.default_rng(0).choice(len(o), DEMO_N, replace=False)
        o, a = o[idx], a[idx]
    ot = th.as_tensor(o); at = th.as_tensor(a)
    mean = th.as_tensor(envs.obs_rms.mean, dtype=th.float32, device=DEVICE)
    std = th.sqrt(th.as_tensor(envs.obs_rms.var, dtype=th.float32, device=DEVICE) + envs.epsilon)
    clip = float(envs.clip_obs)
    bc_opt = th.optim.Adam(model.policy.parameters(), lr=BC_LR)   # SEPARATE from PPO's optimizer
    N = len(ot)

    ckdir = OUT / "checkpoints"; ckdir.mkdir(parents=True, exist_ok=True)
    print(f"BC-anchored v3 (chunked+sep-opt): demos={N:,} ppo_chunk={PPO_CHUNK} bc_steps={BC_STEPS} "
          f"bc_lr={BC_LR} lr={LR} ent={ENT} steps={TIMESTEPS:,} -> {OUT.name}", flush=True)

    done = 0; last_ck = 0
    while done < TIMESTEPS:
        # --- PPO phase (self-contained) ---
        model.learn(total_timesteps=PPO_CHUNK, reset_num_timesteps=False, progress_bar=False)
        done += PPO_CHUNK
        # --- BC pull-back phase (separate optimizer) ---
        model.policy.train()
        bc_tot = 0.0
        for _ in range(BC_STEPS):
            idx = th.randint(0, N, (BC_BATCH,))
            xb = th.clamp((ot[idx].to(DEVICE) - mean) / std, -clip, clip)
            _, logp, _ = model.policy.evaluate_actions(xb, at[idx].to(DEVICE))
            loss = -logp.mean()
            bc_opt.zero_grad(); loss.backward()
            th.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5); bc_opt.step()
            bc_tot += float(loss)
        print(f"[cycle done={done:,}] bc_nll={bc_tot/BC_STEPS:.3f}", flush=True)
        # --- checkpoint (with vec) ---
        if done - last_ck >= CK_EVERY:
            tag = f"{done//1000}k"
            model.save(str(ckdir / f"ck_{tag}")); envs.save(str(ckdir / f"vec_{tag}.pkl"))
            last_ck = done

    OUT.mkdir(parents=True, exist_ok=True)
    model.save(str(OUT / "final_model")); envs.save(str(OUT / "vec_normalize.pkl"))
    print(f"saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
