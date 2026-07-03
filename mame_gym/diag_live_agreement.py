"""Diagnostic: does the trained policy take the FSM's action at EVAL time, on the LIVE obs?
BC reports ~80% on the demo dataset but dies at wave 1-2 — if LIVE agreement is much lower,
the eval-time obs/normalization differs from training (a bug all approaches inherit)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from mame_robotron_env import MameRobotronEnv
from obs_decode import fsm_action_from_obs
import slot_policy  # noqa: register SlotAttnExtractor for load

MODEL = sys.argv[1]
VECNORM = sys.argv[2]
PORT = int(sys.argv[3]) if len(sys.argv) > 3 else 9970
N = int(sys.argv[4]) if len(sys.argv) > 4 else 5
DET = "stoch" not in sys.argv[5:]

env = MameRobotronEnv(rank=0, base_port=PORT, frameskip=4, reset_pool=[0], obs_mode="slot")
venv = VecNormalize.load(VECNORM, DummyVecEnv([lambda: env])); venv.training = False
model = PPO.load(MODEL, device="cpu")


def norm(o):
    return venv.normalize_obs(o.reshape(1, -1).astype(np.float32))[0]


for ep in range(N):
    obs = env.reset()[0]
    mw = env._last_wave
    mv_match = fr_match = steps = 0
    while True:
        a, _ = model.predict(norm(obs), deterministic=DET)
        fmi, ffi = fsm_action_from_obs(obs)            # FSM action on the SAME live obs
        mv_match += int(a[0] == fmi); fr_match += int(a[1] == ffi); steps += 1
        obs, _, t, tr, info = env.step(a)
        mw = max(mw, info.get("wave", 0))
        if t or tr:
            break
    print(f"ep {ep+1}: reached w{mw} in {steps} steps | LIVE agreement move={mv_match/steps:.2f} "
          f"fire={fr_match/steps:.2f}", flush=True)
env.close()
