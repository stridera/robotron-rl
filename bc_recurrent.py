"""bc_recurrent.py — BC-pretrain a RecurrentPPO MlpLstmPolicy on FSM sequence demos.

The recurrent lever for the wave-8 wall: the slot obs is a positional SNAPSHOT (no velocity),
so the policy can't anticipate converging grunts. An LSTM gives it temporal memory. This
clones the obs-limited FSM (the same teacher behind bc_bignet=8.1) into a recurrent policy
so it STARTS at FSM fidelity, then (later) anchored-RL can push past 8 with anticipatory motion.

Design (lowest-risk correct recurrent BC):
- demos/seq_v1_shards: episode-ordered (obs,action,done). Split into FIXED-LENGTH chunks of
  L=SEQ_LEN; each chunk is an independent sequence with the LSTM hidden state reset at its
  start (episode_starts=[1,0,...,0]). L=32 frames (x4 frameskip ~ 2s) is ample motion context
  and avoids the padding/masking needed for variable-length-episode batching.
- sb3_contrib RecurrentActorCriticPolicy.evaluate_actions(obs, actions, RNNStates, episode_starts)
  reshapes (n_seq*L, .) -> (n_seq, L, .); we feed B chunks as (B*L, .). loss = -log_prob.mean().
- obs normalized with bc_bignet's VecNormalize stats (env-determined, identical 945-dim slot obs);
  the SAME pkl is copied as the recurrent model's vec_normalize.pkl so eval_continuous ... rnn matches.

Usage: .venv/bin/python3 bc_recurrent.py [epochs] [seq_len] [batch_chunks] [lr] [tag] [shard_glob]
Output: models/<tag>/ (final_model.zip + vec_normalize.pkl)
"""
import sys
import glob
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "mame_gym"))

EPOCHS   = int(sys.argv[1]) if len(sys.argv) > 1 else 8
SEQ_LEN  = int(sys.argv[2]) if len(sys.argv) > 2 else 32
BATCH    = int(sys.argv[3]) if len(sys.argv) > 3 else 64        # chunks per optimizer step
LR       = float(sys.argv[4]) if len(sys.argv) > 4 else 3e-4
TAG      = sys.argv[5] if len(sys.argv) > 5 else "rbc_v1"
GLOB     = sys.argv[6] if len(sys.argv) > 6 else "demos/seq_v1_shards/shard_*.npz"

import numpy as np
import torch as th
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.type_aliases import RNNStates

DEVICE = "cuda" if th.cuda.is_available() else "cpu"
BASE = ROOT / "models" / "bc_bignet"
OUT = ROOT / "models" / TAG


def build_episodes(glob_pat, maxlen):
    """Load episode-ordered shards -> LIST of (obs,actions) per episode (variable length).

    CRITICAL: train on whole episodes so the LSTM hidden-state distribution MATCHES inference
    (eval_continuous rnn: zero init, carry forward over the whole game). The earlier
    per-chunk-reset (L=32) version mismatched this -> the policy drifted out of distribution
    after ~32 steps and died at wave 2. Episodes here are ~1500 steps (FSM survives deep), so we
    return variable-length episodes and length-bucket them in the train loop (pad per-batch, no
    global truncation/waste). Only the rare episode > maxlen is capped."""
    eps = []
    for f in sorted(glob.glob(str(ROOT / glob_pat))) or sorted(glob.glob(glob_pat)):
        d = np.load(f)
        o, a, dn = d["obs"], d["actions"], d["dones"]
        start = 0
        for i in range(len(o)):
            if dn[i]:
                ep_o, ep_a = o[start:i + 1], a[start:i + 1]
                start = i + 1
                L = min(len(ep_o), maxlen)
                if L >= 2:
                    eps.append((ep_o[:L].astype(np.float16), ep_a[:L].astype(np.int8)))
    if not eps:
        raise SystemExit(f"no episodes built from {glob_pat}")
    return eps


def main():
    MAXLEN = SEQ_LEN   # arg2 = per-episode length CAP (rare long episodes capped); else full episodes
    eps = build_episodes(GLOB, MAXLEN)
    E = len(eps)
    lens = np.array([len(o) for o, _ in eps])
    print(f"built {E:,} episodes (cap={MAXLEN}, mean_len={lens.mean():.0f}, max={lens.max()}, "
          f"{int(lens.sum()):,} labeled steps) from {GLOB}", flush=True)

    # obs normalization from bc_bignet stats (identical env/obs); reuse pkl for eval too
    vn = VecNormalize.load(str(BASE / "vec_normalize.pkl"), DummyVecEnv([lambda: _StubEnv()]))
    mean = th.as_tensor(vn.obs_rms.mean, dtype=th.float32, device=DEVICE)
    std = th.sqrt(th.as_tensor(vn.obs_rms.var, dtype=th.float32, device=DEVICE) + vn.epsilon)
    clip = float(vn.clip_obs)

    venv = DummyVecEnv([lambda: _StubEnv()])
    model = RecurrentPPO(
        "MlpLstmPolicy", venv, device=DEVICE, n_steps=MAXLEN, batch_size=MAXLEN,
        policy_kwargs=dict(lstm_hidden_size=256, n_lstm_layers=1,
                           net_arch=dict(pi=[256, 256], vf=[256, 256])),
    )
    policy = model.policy
    nlstm = policy.lstm_actor.num_layers
    hid = policy.lstm_actor.hidden_size
    opt = th.optim.Adam(policy.parameters(), lr=LR)

    # length-bucket: sort by length, form fixed batches of similar-length episodes (pad per-batch).
    # episode_starts ALL ZERO + zero init state => fast cuDNN path, each row processed full from
    # zero hidden state (== reset at step 0), exactly matching inference. No global truncation/waste.
    order = np.argsort(lens)
    batches = [order[i:i + BATCH] for i in range(0, E, BATCH)]
    rng = np.random.default_rng(0)

    OUT.mkdir(parents=True, exist_ok=True)
    for ep in range(EPOCHS):
        rng.shuffle(batches)            # shuffle batch order, keep similar lengths grouped
        tot, nb = 0.0, 0
        policy.train()
        for bidx in batches:
            B = len(bidx)
            Lmax = int(max(len(eps[j][0]) for j in bidx))
            ob = th.zeros(B, Lmax, 945, device=DEVICE)
            ac = th.zeros(B, Lmax, 2, dtype=th.long, device=DEVICE)
            msk = th.zeros(B, Lmax, device=DEVICE)
            for r, j in enumerate(bidx):
                o, a = eps[j]; L = len(o)
                ob[r, :L] = th.as_tensor(o.astype(np.float32), device=DEVICE)
                ac[r, :L] = th.as_tensor(a.astype(np.int64), device=DEVICE)
                msk[r, :L] = 1.0
            ob = th.clamp((ob - mean) / std, -clip, clip)
            ob_f = ob.reshape(B * Lmax, 945)
            ac_f = ac.reshape(B * Lmax, 2)
            es_f = th.zeros(B * Lmax, device=DEVICE)      # all zero -> fast path, zero init per row
            h0 = th.zeros(nlstm, B, hid, device=DEVICE)
            states = RNNStates((h0, h0.clone()), (h0.clone(), h0.clone()))
            _, log_prob, _ = policy.evaluate_actions(ob_f, ac_f, states, es_f)
            mflat = msk.reshape(B * Lmax)
            loss = -(log_prob * mflat).sum() / mflat.sum()
            opt.zero_grad(); loss.backward()
            th.nn.utils.clip_grad_norm_(policy.parameters(), 0.5); opt.step()
            tot += float(loss.detach()); nb += 1
        print(f"epoch {ep+1}/{EPOCHS}  bc_nll={tot/max(nb,1):.3f}", flush=True)
        model.save(str(OUT / "final_model"))

    model.save(str(OUT / "final_model"))
    vn.save(str(OUT / "vec_normalize.pkl"))
    print(f"saved -> {OUT} (final_model.zip + vec_normalize.pkl)", flush=True)


class _StubEnv(gym.Env):
    """Minimal gym.Env exposing the MAME spaces so RecurrentPPO/VecNormalize build without booting MAME."""
    metadata = {"render_modes": []}
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(945,), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete([8, 8])
        self.render_mode = None
    def reset(self, *a, **k):
        return np.zeros(945, dtype=np.float32), {}
    def step(self, a):
        return np.zeros(945, dtype=np.float32), 0.0, False, False, {}


if __name__ == "__main__":
    main()
