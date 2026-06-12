"""Rebuild the wave-ladder save-state pool in one driving pass.

Unlike capture_wave5_states.py (one target wave per run), this saves a state
at EVERY wave entry within [MIN_WAVE, MAX_WAVE] up to a per-wave quota, so a
single deep episode contributes states for each wave it passes through.
Written after the 2026-06-11 reboot wiped /tmp/mame_states (the whole
wave-5..12 pool); states now persist in mame_bridge.STATE_DIR.

Prints an index→wave mapping at the end for building reset pools.

Usage: .venv/bin/python3 mame_gym/capture_ladder_states.py <model.zip> <vecnorm.pkl> \
           [min_wave] [max_wave] [per_wave_quota] [start_idx] [reset_pool]
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from mame_robotron_env import MameRobotronEnv
from mame_obs import parse_header

MODEL = sys.argv[1]
VECNORM = sys.argv[2]
MIN_WAVE = int(sys.argv[3]) if len(sys.argv) > 3 else 5
MAX_WAVE = int(sys.argv[4]) if len(sys.argv) > 4 else 13
QUOTA = int(sys.argv[5]) if len(sys.argv) > 5 else 5
START_IDX = int(sys.argv[6]) if len(sys.argv) > 6 else 0
RESET_POOL = [int(x) for x in sys.argv[7].split(",")] if len(sys.argv) > 7 else [0]
SETTLE_STEPS = 30   # 120 frames past wave entry: skip the transition flash
MAX_EPISODES = 400
MAPPING_PATH = Path(__file__).parent / "ladder_state_map.json"


def main():
    env = MameRobotronEnv(rank=0, base_port=9990, frameskip=4, reset_pool=RESET_POOL)
    venv = DummyVecEnv([lambda: env])
    venv = VecNormalize.load(VECNORM, venv)
    venv.training = False
    venv.norm_reward = False
    model = PPO.load(MODEL, env=venv, device="cpu")

    got = Counter()
    mapping = {}        # state index -> {wave, score, lives}
    next_idx = START_IDX + 1
    episodes = 0
    obs = venv.reset()
    settle_left = -1
    pending_wave = -1
    fresh_episode = True   # first step after reset: adopt wave WITHOUT saving
    while episodes < MAX_EPISODES:
        if all(got[w] >= QUOTA for w in range(MIN_WAVE, MAX_WAVE + 1)):
            break
        action, _ = model.predict(obs, deterministic=False)
        obs, _, dones, infos = venv.step(action)
        info = infos[0]
        if dones[0]:
            episodes += 1
            settle_left = -1
            pending_wave = -1
            fresh_episode = True
            continue
        wave = info.get("wave", 0)
        if fresh_episode:
            # Episode may START at an in-range wave (deep reset states):
            # saving here would just duplicate the reset state. Only save on
            # transitions observed within the episode (caught 2026-06-11:
            # 5 "wave-10" states were byte-near copies of the w5_28 start).
            pending_wave = wave
            fresh_episode = False
            continue
        if wave != pending_wave:
            # New wave entered: arm a save if it's in range and under quota.
            pending_wave = wave
            settle_left = SETTLE_STEPS if (MIN_WAVE <= wave <= MAX_WAVE
                                           and got[wave] < QUOTA) else -1
        if settle_left > 0:
            settle_left -= 1
        elif settle_left == 0:
            settle_left = -1
            pkt = env._bridge.save_state(next_idx)
            h = parse_header(pkt)
            if h["wave"] != pending_wave:
                continue   # died/transitioned during settle; state still saved but skip counting
            got[h["wave"]] += 1
            mapping[next_idx] = {"wave": h["wave"], "score": h["score"], "lives": h["lives"]}
            print(f"saved w5_{next_idx}: wave={h['wave']} score={h['score']} "
                  f"lives={h['lives']} (ep {episodes})  coverage={dict(sorted(got.items()))}",
                  flush=True)
            next_idx += 1
    # Merge with any prior mapping (multi-pass rebuilds).
    old = json.loads(MAPPING_PATH.read_text()) if MAPPING_PATH.exists() else {}
    old.update({str(k): v for k, v in mapping.items()})
    MAPPING_PATH.write_text(json.dumps(old, indent=1, sort_keys=True))
    print(f"DONE in {episodes} episodes. coverage={dict(sorted(got.items()))} "
          f"next_idx={next_idx}", flush=True)
    venv.close()


if __name__ == "__main__":
    main()
