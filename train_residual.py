"""train_residual.py — PPO on the FSM-residual env (mame_gym/residual_env.ResidualWrapper).

The FSM (evolved v2, EVOLVE_PARAMS) proposes move+fire; the policy learns a gated move
override (action MultiDiscrete([2,8]) = gate, override_move), fire stays FSM. Reward =
shaped env reward - RESIDUAL_OVERRIDE_PENALTY per override. Goal: cut deaths/wave while
keeping the FSM's 12.72 capability.

Usage: EVOLVE_PARAMS=models/fsm_evolved_reseed_v2.json MAME_RL_RESEED=1 NENV=8 \
       .venv/bin/python train_residual.py [timesteps] [port] [tag]
Judge with eval_protocol.py (residual mode) paired vs the pure FSM.
"""
import os, sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "mame_gym"))
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from mame_robotron_env import MameRobotronEnv
from residual_env import ResidualWrapper

TIMESTEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3_000_000
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 9940
TAG = sys.argv[3] if len(sys.argv) > 3 else "residual1"
NENV = int(os.environ.get("NENV", 8))
CK_EVERY = int(os.environ.get("CK_EVERY", 200_000))
OUT = ROOT / "models" / f"residual_{TAG}"
DEVICE = "cuda" if th.cuda.is_available() else "cpu"


def make_env(rank):
    def _init():
        env = MameRobotronEnv(rank=rank, base_port=PORT, frameskip=4, reset_pool=[0], obs_mode="slot")
        env = ResidualWrapper(env)
        return Monitor(env, info_keywords=("score", "wave", "lives", "override"))
    return _init


def main():
    envs = SubprocVecEnv([make_env(i) for i in range(NENV)])
    envs = VecNormalize(envs, norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = PPO("MlpPolicy", envs, device=DEVICE, verbose=1,
                n_steps=2048, batch_size=512, n_epochs=4, gamma=0.99, gae_lambda=0.95,
                ent_coef=0.005, learning_rate=2.5e-4, clip_range=0.2,
                policy_kwargs={"net_arch": [512, 512]})
    # PRETRAIN_GATE=1: BC the policy to output [gate=0, fsm_move] from FSM demos so it
    # STARTS at FSM-level (a no-op residual), then PPO learns only useful overrides. Fixes
    # the from-scratch failure (residual1/2 never defaulted to keep-FSM -> stayed below FSM).
    if os.environ.get("PRETRAIN_GATE") == "1":
        import numpy as np
        d = np.load(ROOT / "demos" / "fsm_mame_demos.npz")
        o = np.asarray(d["obs"], dtype=np.float32)
        mv = np.asarray(d["actions"][:, 0], dtype=np.int64)
        Npre = min(len(o), 300_000); o, mv = o[:Npre], mv[:Npre]
        envs.obs_rms.mean = o.mean(0).astype("float64")
        envs.obs_rms.var = o.var(0).astype("float64") + 1e-8
        envs.obs_rms.count = float(Npre)
        mean = th.as_tensor(envs.obs_rms.mean, dtype=th.float32, device=DEVICE)
        std = th.sqrt(th.as_tensor(envs.obs_rms.var, dtype=th.float32, device=DEVICE) + envs.epsilon)
        ot = th.as_tensor(o)
        tgt = th.as_tensor(np.stack([np.zeros(Npre, np.int64), mv], 1))
        opt = th.optim.Adam(model.policy.parameters(), lr=3e-4)
        model.policy.train()
        loss = th.tensor(0.0)
        for ep in range(3):
            perm = th.randperm(Npre)
            for i in range(0, Npre, 1024):
                idx = perm[i:i + 1024]
                xb = th.clamp((ot[idx].to(DEVICE) - mean) / std, -10, 10)
                _, lp, _ = model.policy.evaluate_actions(xb, tgt[idx].to(DEVICE))
                loss = -lp.mean()
                opt.zero_grad(); loss.backward(); opt.step()
        print(f"PRETRAIN_GATE: cloned [gate=0, fsm_move] on {Npre} demos (nll {float(loss):.3f})", flush=True)
    ckdir = OUT / "checkpoints"; ckdir.mkdir(parents=True, exist_ok=True)
    print(f"residual PPO: tag={TAG} nenv={NENV} steps={TIMESTEPS:,} "
          f"override_penalty={os.environ.get('RESIDUAL_OVERRIDE_PENALTY','0.2')} "
          f"fsm={os.environ.get('EVOLVE_PARAMS','default')} -> {OUT.name}", flush=True)
    done, last_ck = 0, 0
    CHUNK = 100_000
    while done < TIMESTEPS:
        model.learn(total_timesteps=CHUNK, reset_num_timesteps=False, progress_bar=False)
        done += CHUNK
        if done - last_ck >= CK_EVERY:
            tag = f"{done//1000}k"
            model.save(str(ckdir / f"ck_{tag}")); envs.save(str(ckdir / f"vec_{tag}.pkl"))
            last_ck = done
            print(f"[ck {tag}] saved", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    model.save(str(OUT / "final_model")); envs.save(str(OUT / "vec_normalize.pkl"))
    print(f"saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
