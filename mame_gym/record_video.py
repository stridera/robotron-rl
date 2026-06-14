"""record_video.py — record a real-MAME-frame video of a policy playing.

Drives one MAME instance with the policy from a wave-1 boot, grabs a screen
snapshot each env-step (so the video is real game frames, not an obs overlay),
and assembles an mp4. Optional --wandb logs it to a W&B run. Intended to be run
at each major chain checkpoint so progress is watchable and issues are visible.

Usage:
  .venv/bin/python3 mame_gym/record_video.py <model.zip> <vecnorm.pkl> \
      [out.mp4] [max_steps] [obs_mode=slot|grid] [--wandb] [--deterministic]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import imageio.v2 as imageio
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from mame_robotron_env import MameRobotronEnv
from mame_obs import parse_header

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
FLAGS = {a for a in sys.argv[1:] if a.startswith("--")}
MODEL, VECNORM = ARGS[0], ARGS[1]
OUT = ARGS[2] if len(ARGS) > 2 else "/tmp/robotron_run.mp4"
MAX_STEPS = int(ARGS[3]) if len(ARGS) > 3 else 1500
OBS_MODE = ARGS[4] if len(ARGS) > 4 else "slot"
DET = "--deterministic" in FLAGS
SNAP_DIR = Path.home() / ".mame" / "snap" / "robotron"
FPS = 15   # one frame per env-step = 60fps/frameskip-4


def _grab(before: set):
    """Return the newest PNG that appeared since `before` (or None)."""
    now = set(SNAP_DIR.glob("*.png"))
    new = sorted(now - before)
    return new[-1] if new else None


def main():
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    env = MameRobotronEnv(rank=0, base_port=9974, frameskip=4, reset_pool=[0], obs_mode=OBS_MODE)
    venv = DummyVecEnv([lambda: env])
    venv = VecNormalize.load(VECNORM, venv)
    venv.training = False
    venv.norm_reward = False
    model = PPO.load(MODEL, env=venv, device="cpu")

    obs = venv.reset()
    frames, max_wave, last_score = [], 1, 0
    for step in range(MAX_STEPS):
        a, _ = model.predict(obs, deterministic=DET)
        obs, _, dones, infos = venv.step(a)
        h = parse_header(env._last_packet)
        max_wave = max(max_wave, h["wave"]); last_score = h["score"]
        before = set(SNAP_DIR.glob("*.png"))
        env._bridge.snapshot()
        png = None
        for _ in range(30):                  # wait up to ~1.5s for MAME to write it
            png = _grab(before)
            if png:
                break
            time.sleep(0.05)
        if png is not None:
            try:
                frames.append(np.asarray(Image.open(png).convert("RGB")))
            finally:
                png.unlink(missing_ok=True)
        if dones[0]:
            break
    venv.close()

    if not frames:
        print("ERROR: captured 0 frames (snapshot dir issue?)", flush=True)
        return
    imageio.mimsave(OUT, frames, fps=FPS, quality=8, macro_block_size=None)
    print(f"wrote {OUT}: {len(frames)} frames, reached wave {max_wave}, score {last_score}", flush=True)

    if "--wandb" in FLAGS:
        import wandb
        run = wandb.init(project="robotron", group="videos", mode="offline",
                         config={"model": MODEL, "reached_wave": max_wave, "score": last_score})
        run.log({"gameplay": wandb.Video(OUT, fps=FPS, format="mp4"),
                 "reached_wave": max_wave, "score": last_score})
        run.finish()
        print("logged to W&B (offline)", flush=True)


if __name__ == "__main__":
    main()
