"""Continuous wave-1 evaluation: the REAL hardware test.

The ladder trains per-wave competence by resetting into deep save states. This
script measures the thing that actually matters for Xbox deployment: how far a
policy gets in ONE continuous game starting from wave 1 with full lives, no
save-state resets. The distribution of waves reached answers "can it chain
1 -> N", which the save-state-started ATH episodes do NOT.

Usage: .venv/bin/python3 mame_gym/eval_continuous.py <model.zip> <vecnorm.pkl> [n_episodes] [port]
"""
import sys
import statistics
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from mame_robotron_env import MameRobotronEnv

MODEL = sys.argv[1]
VECNORM = sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 15
PORT = int(sys.argv[4]) if len(sys.argv) > 4 else 9970
DETERMINISTIC = len(sys.argv) > 5 and sys.argv[5] == "det"


def main():
    env = MameRobotronEnv(rank=0, base_port=PORT, frameskip=4, reset_pool=[0])
    venv = DummyVecEnv([lambda: env])
    venv = VecNormalize.load(VECNORM, venv)
    venv.training = False
    venv.norm_reward = False
    model = PPO.load(MODEL, env=venv, device="cpu")

    results = []
    for ep in range(N):
        obs = venv.reset()
        start_lives = env._last_lives
        start_wave = env._last_wave
        max_wave = start_wave
        final_score = env._last_score
        steps = 0
        while True:
            a, _ = model.predict(obs, deterministic=DETERMINISTIC)
            obs, _, dones, infos = venv.step(a)
            info = infos[0]
            max_wave = max(max_wave, info.get("wave", 0))
            if not dones[0]:
                final_score = info.get("score", final_score)
            steps += 1
            if dones[0]:
                break
        results.append((max_wave, final_score))
        print(f"ep {ep+1}: start=w{start_wave}/{start_lives}lives  reached=w{max_wave}  "
              f"score={final_score}  steps={steps}", flush=True)

    waves = [r[0] for r in results]
    scores = [r[1] for r in results]
    print(f"\n=== {N} CONTINUOUS wave-1 runs (full lives, no save-state resets) ===")
    print(f"wave reached:  min={min(waves)} max={max(waves)} "
          f"mean={statistics.mean(waves):.1f} median={statistics.median(waves)}")
    print(f"score:         max={max(scores)} mean={statistics.mean(scores):.0f}")
    print(f"wave distribution: {dict(sorted(Counter(waves).items()))}")
    venv.close()


if __name__ == "__main__":
    main()
