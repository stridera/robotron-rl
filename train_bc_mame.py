"""train_bc_mame.py — behavior-clone the FSM teacher into the MAME MLP policy.

Loads demos/fsm_mame_demos.npz (obs 945, actions [move,fire] 0-7) and trains a
MlpPolicy (net_arch [512,512], MultiDiscrete([8,8]) — IDENTICAL to train_mame.py) by
maximizing the log-prob of the FSM's actions (SB3 evaluate_actions). Saves a PPO
checkpoint + VecNormalize (stats fitted on the demos) so train_mame.py can RL-fine-tune:

  python train_mame.py --bc-checkpoint models/bc_fsm/final_model.zip \
      --vec-normalize models/bc_fsm/vec_normalize.pkl --num-envs 12 ...

Usage: .venv/bin/python3 train_bc_mame.py [demos.npz] [epochs] [device]
"""
import sys
from pathlib import Path

import numpy as np
import torch as th
import gymnasium as gym
from gymnasium.spaces import Box, MultiDiscrete
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

DEMOS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("demos/fsm_mame_demos.npz")
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 25
DEVICE = sys.argv[3] if len(sys.argv) > 3 else ("cuda" if th.cuda.is_available() else "cpu")
NET_ARCH = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [512, 512]
OUT = Path("models") / (sys.argv[5] if len(sys.argv) > 5 else "bc_fsm")
EXTRACTOR = sys.argv[6] if len(sys.argv) > 6 else "mlp"   # 'mlp' or 'attn'
BATCH = 2048


class _DummyEnv(gym.Env):
    observation_space = Box(-np.inf, np.inf, (945,), np.float32)
    action_space = MultiDiscrete([8, 8])

    def reset(self, *, seed=None, options=None):
        return np.zeros(945, np.float32), {}

    def step(self, a):
        return np.zeros(945, np.float32), 0.0, False, False, {}


def main():
    d = np.load(DEMOS)
    obs = d["obs"].astype(np.float32)
    actions = d["actions"].astype(np.int64)
    print(f"loaded {len(obs):,} demos  obs{obs.shape}  actions{actions.shape}  device={DEVICE}")

    venv = DummyVecEnv([_DummyEnv])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    venv.obs_rms.mean = obs.mean(0).astype(np.float64)
    venv.obs_rms.var = obs.var(0).astype(np.float64) + 1e-8
    venv.obs_rms.count = float(len(obs))

    print(f"net_arch={NET_ARCH}  out={OUT}  extractor={EXTRACTOR}")
    if EXTRACTOR == "attn":
        sys.path.insert(0, str(Path(__file__).parent / "mame_gym"))
        from slot_policy import SlotAttnExtractor
        pk = {"features_extractor_class": SlotAttnExtractor,
              "features_extractor_kwargs": {"features_dim": 256}, "net_arch": [256]}
    else:
        pk = {"net_arch": NET_ARCH}
    model = PPO("MlpPolicy", venv, device=DEVICE, verbose=0, policy_kwargs=pk)
    policy = model.policy

    obs_n = venv.normalize_obs(obs).astype(np.float32)
    obs_t = th.as_tensor(obs_n, device=DEVICE)
    act_t = th.as_tensor(actions, device=DEVICE)
    opt = th.optim.Adam(policy.parameters(), lr=3e-4)
    sched = th.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    n = len(obs_t)
    n_val = max(2000, n // 20)
    tr_idx = th.arange(n_val, n, device=DEVICE)
    va_idx = th.arange(0, n_val, device=DEVICE)

    for ep in range(EPOCHS):
        policy.train()
        perm = tr_idx[th.randperm(len(tr_idx), device=DEVICE)]
        tot = 0.0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            _, log_prob, _ = policy.evaluate_actions(obs_t[idx], act_t[idx])
            loss = -log_prob.mean()
            opt.zero_grad(); loss.backward()
            th.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        policy.eval()
        with th.no_grad():
            feats = policy.extract_features(obs_t[va_idx])
            latent_pi, _ = policy.mlp_extractor(feats)
            dist = policy._get_action_dist_from_latent(latent_pi)
            pm = dist.distribution[0].probs.argmax(-1)
            pf = dist.distribution[1].probs.argmax(-1)
            ma = (pm == act_t[va_idx, 0]).float().mean().item()
            fa = (pf == act_t[va_idx, 1]).float().mean().item()
        print(f"epoch {ep+1:2d}  nll={tot/len(tr_idx):.3f}  val move_acc={ma:.3f}  fire_acc={fa:.3f}",
              flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    model.save(str(OUT / "final_model"))
    venv.save(str(OUT / "vec_normalize.pkl"))
    print(f"\nsaved BC policy -> {OUT}/final_model.zip + vec_normalize.pkl")
    print(f"EVAL:      .venv/bin/python3 mame_gym/eval_continuous.py {OUT}/final_model.zip {OUT}/vec_normalize.pkl 25 9982 stoch slot")
    print(f"FINE-TUNE: python train_mame.py --bc-checkpoint {OUT}/final_model.zip --vec-normalize {OUT}/vec_normalize.pkl --num-envs 12 --timesteps 3000000 --ent-coef 0.005 --base-port 9920 --device cpu")


if __name__ == "__main__":
    main()
