"""Targeted visual verification: snapshot when specific (rare/spawned) entity
types are present — Enforcers, EnforcerBullets, Progs, CruiseMissiles,
mid-lifecycle Spheroids, and (wave 7+) Quarks/Tanks/TankShells.

Uses the same obs builder the policy sees, so a labeled "Prog" overlay is a
direct visual test of the velocity-disambiguation heuristic.

Usage: .venv/bin/python3 mame_gym/visual_verify_targeted.py <model.zip> <vecnorm.pkl> [out_dir]
"""
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image, ImageDraw

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from mame_robotron_env import MameRobotronEnv
from mame_obs import MameObsBuilder, iter_entities, parse_header, _LIST1_SW, _LIST2_SW, _LIST3_SW

MODEL = sys.argv[1]
VECNORM = sys.argv[2]
OUT_DIR = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("/tmp/visual_verify_targeted")
SNAP_DIR = Path("/home/strider/.mame/snap/robotron")

QUOTAS = {"Enforcer": 3, "EnforcerBullet": 3, "Prog": 4, "CruiseMissile": 3,
          "Quark": 2, "Tank": 2, "TankShell": 2}


def gu_to_px(gx, gy, w, h):
    return gx * 2 * w / 292.0, gy * h / 240.0


def typed_sprites(builder, packet):
    """Run the SAME classification path the policy obs uses, returning
    (label, x, y) in game units (player excluded)."""
    sprites = builder._sprites_from_packet(packet)
    out = []
    for spx, spy, name in sprites[1:]:
        gx = spx / 665.0 * 140 + 5
        gy = spy / 492.0 * 215 + 15
        out.append((name, gx, gy))
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = MameRobotronEnv(rank=97, base_port=9850, frameskip=4,
                          reset_pool=[17, 18, 19, 20, 21, 22, 23, 24])  # wave-5 starts
    venv = DummyVecEnv([lambda: env])
    venv = VecNormalize.load(VECNORM, venv)
    venv.training = False
    venv.norm_reward = False
    model = PPO.load(MODEL, env=venv, device="cpu")
    builder = MameObsBuilder()   # independent tracker for label rendering

    got = Counter()
    obs = venv.reset()
    builder.reset()
    taken = 0
    cooldown = 0
    for step_i in range(60000):
        a, _ = model.predict(obs, deterministic=False)
        obs, _, dones, infos = venv.step(a)
        if dones[0]:
            obs = venv.reset()
            builder.reset()
            continue
        pkt = env._last_packet
        # Skip death/respawn frames: sprites aren't rendered (entities
        # teleporting), so overlays would show empty boxes (header byte 9 =
        # $9848 player-being-killed flag).
        if pkt[9] != 0:
            cooldown = max(cooldown, 8)
        labels = typed_sprites(builder, pkt)
        cooldown = max(0, cooldown - 1)
        want = [n for n, _, _ in labels if n in QUOTAS and got[n] < QUOTAS[n]]
        if not want or cooldown:
            continue
        cooldown = 30
        before = set(SNAP_DIR.glob("*.png")) if SNAP_DIR.exists() else set()
        env._bridge.snapshot()
        time.sleep(0.3)
        new = sorted((set(SNAP_DIR.glob("*.png")) if SNAP_DIR.exists() else set()) - before)
        if not new:
            continue
        taken += 1
        for n in set(want):
            got[n] += 1
        h = parse_header(pkt)
        img = Image.open(new[-1]).convert("RGB")
        img = img.resize((img.width * 3, img.height * 3), Image.NEAREST)
        d = ImageDraw.Draw(img)
        W, H = img.size
        for name, gx, gy in labels:
            px, py = gu_to_px(gx, gy, W, H)
            color = "cyan" if name in QUOTAS else "red"
            d.rectangle([px - 12, py - 12, px + 12, py + 12], outline=color)
            d.text((px - 12, py - 24), name, fill="yellow" if name not in QUOTAS else "cyan")
        ppx, ppy = gu_to_px(h["player_x"], h["player_y"], W, H)
        d.rectangle([ppx - 14, ppy - 14, ppx + 14, ppy + 14], outline="lime")
        tag = "_".join(sorted(set(want)))
        img.save(OUT_DIR / f"target_{taken:02d}_w{h['wave']}_{tag}.png")
        shutil.copy(new[-1], OUT_DIR / f"raw_{taken:02d}.png")
        print(f"capture {taken}: wave={h['wave']} targets={sorted(set(want))} quotas={dict(got)}", flush=True)
        if all(got[k] >= v for k, v in QUOTAS.items() if k not in ("Quark", "Tank", "TankShell")):
            break   # tanks/quarks need wave 7+; don't block on them
    print(f"DONE {taken} captures, quotas={dict(got)}", flush=True)
    venv.close()


if __name__ == "__main__":
    main()
