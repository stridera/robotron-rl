"""record_video_overlay.py — gameplay video annotated with what the MODEL sees
and what it outputs, for verification.

Each frame overlays on the real MAME screen:
  - every entity the model perceives (classified from the typed entity list),
    drawn as a labeled box at its game position (so you can confirm the model's
    perception matches the pixels);
  - the player box (lime) + the chosen MOVE arrow (cyan) and FIRE arrow (red);
  - a header line: wave / lives / score / game_state / move,fire dirs.

Game-unit -> snapshot-pixel map: px = gx*2, py = gy on the native 292x240 frame
(verified: player boot (74,124) -> screen centre). Scaled up 3x for legibility.

Usage:
  .venv/bin/python3 mame_gym/record_video_overlay.py <model.zip> <vecnorm.pkl> \
      [out.mp4] [max_steps] [obs_mode] [--deterministic]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from collections import deque

import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from mame_robotron_env import MameRobotronEnv
from mame_obs import parse_header, iter_entities, _LIST1_SW, _LIST2_SW, _LIST3_SW

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
FLAGS = {a for a in sys.argv[1:] if a.startswith("--")}
MODEL, VECNORM = ARGS[0], ARGS[1]
OUT = ARGS[2] if len(ARGS) > 2 else "/tmp/robotron_overlay.mp4"
MAX_STEPS = int(ARGS[3]) if len(ARGS) > 3 else 600
OBS_MODE = ARGS[4] if len(ARGS) > 4 else "slot"
DET = "--deterministic" in FLAGS
SNAP_DIR = Path.home() / ".mame" / "snap" / "robotron"
SCALE = 3
FPS = 15

# move/fire dir (1-8) -> unit (dx, dy) in screen space (y down). 0 = none.
DIR_VEC = {0: (0, 0), 1: (0, -1), 2: (0.7, -0.7), 3: (1, 0), 4: (0.7, 0.7),
           5: (0, 1), 6: (-0.7, 0.7), 7: (-1, 0), 8: (-0.7, -0.7)}


def classify(list_id, sw):
    if list_id == 4: return "Electrode"
    if list_id == 1: return _LIST1_SW.get(sw, "TankShell")
    if list_id == 2: return _LIST2_SW.get(sw)
    if list_id == 3: return _LIST3_SW.get(sw)
    return None


def gp(gx, gy):
    return gx * 2 * SCALE, gy * SCALE


def main():
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    env = MameRobotronEnv(rank=0, base_port=9974, frameskip=4, reset_pool=[0], obs_mode=OBS_MODE)
    venv = DummyVecEnv([lambda: env])
    venv = VecNormalize.load(VECNORM, venv)
    venv.training = False; venv.norm_reward = False
    model = PPO.load(MODEL, env=venv, device="cpu")

    # Recent NOTABLE reward events (rescue / wave-clear / death / kills), newest
    # first — so we can verify the right rewards fire against the gameplay.
    NOTABLE = {"death", "RESCUE", "WAVE_CLEAR", "spawner_kill", "shooter_kill", "brain_kill"}
    reward_log = deque(maxlen=16)

    obs = venv.reset()
    frames = []
    for step in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=DET)
        move, fire = int(action[0][0]) + 1, int(action[0][1]) + 1
        before = set(SNAP_DIR.glob("*.png"))
        pkt = env._last_packet
        obs, rews, dones, infos = venv.step(action)
        parts = infos[0].get("reward_parts", {})
        notable = {k: v for k, v in parts.items() if k in NOTABLE}
        if notable:
            reward_log.appendleft((step, float(rews[0]), notable))
        env._bridge.snapshot()
        png = None
        for _ in range(30):
            new = sorted(set(SNAP_DIR.glob("*.png")) - before)
            if new: png = new[-1]; break
            time.sleep(0.05)
        if png is None:
            if dones[0]: break
            continue
        img = Image.open(png).convert("RGB").resize((292 * SCALE, 240 * SCALE), Image.NEAREST)
        png.unlink(missing_ok=True)
        d = ImageDraw.Draw(img)
        h = parse_header(pkt)
        # entities the model perceives
        for addr, lid, sw, x, y in iter_entities(pkt):
            nm = classify(lid, sw)
            if nm is None: continue
            px, py = gp(x, y)
            fam = nm in ("Mommy", "Daddy", "Mikey")
            color = "yellow" if fam else "magenta"
            d.rectangle([px - 9, py - 9, px + 9, py + 9], outline=color)
            d.text((px - 9, py - 20), nm[:4], fill=color)
        # player + action arrows
        ppx, ppy = gp(h["player_x"], h["player_y"])
        d.rectangle([ppx - 11, ppy - 11, ppx + 11, ppy + 11], outline="lime")
        mv = DIR_VEC[move]; fr = DIR_VEC[fire]
        d.line([ppx, ppy, ppx + mv[0] * 38, ppy + mv[1] * 38], fill="cyan", width=3)
        d.line([ppx, ppy, ppx + fr[0] * 26, ppy + fr[1] * 26], fill="red", width=2)
        d.text((6, 4), f"W{h['wave']} L{h['lives']} S{h['score']} gs={h['game_state']:#04x} "
                       f"move={move} fire={fire}", fill="white")
        # recent reward events stacked at top-right (newest first)
        W = img.size[0]
        d.text((W - 250, 4), "recent rewards:", fill="white")
        for i, (s, tot, nb) in enumerate(reward_log):
            lbl = " ".join(f"{k}+{int(v)}" if v >= 0 else f"{k}{int(v)}" for k, v in nb.items())
            col = ("red" if "death" in nb else "yellow" if "RESCUE" in nb
                   else "cyan" if "WAVE_CLEAR" in nb else "magenta")
            d.text((W - 250, 20 + i * 14), f"s{s} {lbl}", fill=col)
        frames.append(np.asarray(img))
        if dones[0]: break
    venv.close()
    if not frames:
        print("ERROR: 0 frames"); return
    imageio.mimsave(OUT, frames, fps=FPS, quality=8, macro_block_size=None)
    print(f"wrote {OUT}: {len(frames)} frames", flush=True)


if __name__ == "__main__":
    main()
