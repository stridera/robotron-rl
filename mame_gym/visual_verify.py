"""Visual classification verification + YOLO dataset generator.

Drives the chain-head policy, takes paired (screenshot, typed-entity-array)
samples at random intervals across waves, and renders labeled overlay images
for human review. The raw pairs (PNG + JSON) double as YOLO training data.

Game-units -> screen-pixels: the Williams playfield is 292x240 visible pixels;
entity display coords (node +4/+5) are in game units. Calibration anchors on
the player sprite (header pX/pY), which we know renders at the player's
position. Empirically game units map ~2x horizontally: px = gx*2, py = gy.
The overlay tool draws BOTH the box and the label so misalignment is obvious.

Usage:
  .venv/bin/python3 mame_gym/visual_verify.py <model.zip> <vecnorm.pkl> [n_samples] [out_dir]
Then review out_dir/overlay_*.png by eye.
"""
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image, ImageDraw

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from mame_robotron_env import MameRobotronEnv
from mame_obs import iter_entities, parse_header, _LIST1_SW, _LIST2_SW, _LIST3_SW

MODEL = sys.argv[1]
VECNORM = sys.argv[2]
N_SAMPLES = int(sys.argv[3]) if len(sys.argv) > 3 else 15
OUT_DIR = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("/tmp/visual_verify")
SNAP_DIR = Path("/home/strider/.mame/snap/robotron")  # MAME default snap dir

# game-units -> screen px (calibrated against player anchor; refine by eye)
def gu_to_px(gx, gy, img_w, img_h):
    # Williams blitter: X in bytes (2 px/byte) -> px = gx*2; Y direct.
    return gx * 2 * img_w / 292.0, gy * img_h / 240.0


def label_for(list_id, sw):
    if list_id == 4: return "Electrode"
    if list_id == 2: return _LIST2_SW.get(sw, f"FAM?{sw:04X}")
    if list_id == 1: return _LIST1_SW.get(sw, "TankShell?")
    if list_id == 3: return _LIST3_SW.get(sw, f"L3?{sw:04X}")
    return f"?{sw:04X}"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = MameRobotronEnv(rank=98, base_port=9860, frameskip=4,
                          reset_pool=[0, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    venv = DummyVecEnv([lambda: env])
    venv = VecNormalize.load(VECNORM, venv)
    venv.training = False
    venv.norm_reward = False
    model = PPO.load(MODEL, env=venv, device="cpu")

    rng = np.random.default_rng(7)
    obs = venv.reset()
    taken = 0
    step_i = 0
    next_snap = int(rng.integers(20, 120))
    while taken < N_SAMPLES and step_i < 30000:
        a, _ = model.predict(obs, deterministic=False)
        obs, _, dones, infos = venv.step(a)
        step_i += 1
        if dones[0]:
            obs = venv.reset()
            continue
        if step_i >= next_snap:
            next_snap = step_i + int(rng.integers(60, 200))
            pre_pkt = env._last_packet           # obs aligned with screenshot
            before = set(SNAP_DIR.glob("*.png")) if SNAP_DIR.exists() else set()
            env._bridge.snapshot()
            time.sleep(0.3)
            after = set(SNAP_DIR.glob("*.png")) if SNAP_DIR.exists() else set()
            new = sorted(after - before)
            if not new:
                continue
            taken += 1
            img_path = new[-1]
            h = parse_header(pre_pkt)
            ents = [{"addr": a_, "list": l, "sw": f"0x{s:04X}", "x": x, "y": y,
                     "label": label_for(l, s)}
                    for a_, l, s, x, y in iter_entities(pre_pkt)]
            sample = {"header": h, "entities": ents}
            (OUT_DIR / f"sample_{taken:02d}.json").write_text(json.dumps(sample, indent=1))
            # Overlay
            img = Image.open(img_path).convert("RGB")
            img = img.resize((img.width * 3, img.height * 3), Image.NEAREST)
            d = ImageDraw.Draw(img)
            W, H = img.size
            for e in ents:
                px, py = gu_to_px(e["x"], e["y"], W, H)
                d.rectangle([px - 12, py - 12, px + 12, py + 12], outline="red")
                d.text((px - 12, py - 24), e["label"], fill="yellow")
            ppx, ppy = gu_to_px(h["player_x"], h["player_y"], W, H)
            d.rectangle([ppx - 14, ppy - 14, ppx + 14, ppy + 14], outline="lime")
            d.text((ppx - 14, ppy - 28), "PLAYER", fill="lime")
            img.save(OUT_DIR / f"overlay_{taken:02d}_w{h['wave']}.png")
            shutil.copy(img_path, OUT_DIR / f"raw_{taken:02d}_w{h['wave']}.png")
            print(f"sample {taken}: wave={h['wave']} entities={len(ents)} -> overlay_{taken:02d}_w{h['wave']}.png", flush=True)
    print(f"DONE {taken} samples in {OUT_DIR}", flush=True)
    venv.close()


if __name__ == "__main__":
    main()
