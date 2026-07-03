"""Auto-labeled YOLO dataset collector for the Xbox-hardware perception path.

MAME gives us both the rendered frame (snapshot) and perfect entity ground
truth (RAM walk) — free labels. The FSM plays; every SNAP_EVERY steps we take
a screenshot and write YOLO-format labels from the obs held at snapshot time
(the bridge's CMD_SNAP screenshots the CURRENT frame, then returns the obs one
frame later — so labels come from the obs the caller already holds; worst-case
skew is one frame, <=2px).

Coordinates: entity record x = blitter column in BYTES (2px each), y = line in
pixels, on the native 292x240 screen. Boxes use per-type nominal sprite sizes
(v1 approximation — refine by emitting anim-metadata dims from Lua if needed).

Usage:
  MAME_SNAP_DIR=... python mame_gym/collect_yolo_dataset.py OUT_DIR [N_FRAMES] [PORT]

Output layout (YOLO): OUT_DIR/images/*.png + OUT_DIR/labels/*.txt + classes.txt.
Wave diversity via random resets over the save-state pool (rl_reset + w5_*).
"""
from __future__ import annotations

import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mame_gym"))

from mame_bridge import MameBridge, STATE_DIR          # noqa: E402
from mame_obs import iter_entities, classify_sw, parse_header  # noqa: E402
from mame_obs import MameObsBuilder                    # noqa: E402
from position_wrapper import SLOT_CATEGORIES, CATCHALL_SLOTS  # noqa: E402
import robotron_fsm as fsm                             # noqa: E402

SCREEN_W, SCREEN_H = 292, 240

# YOLO class ids. Player is class 0 (the detector must find the player too —
# on hardware there is no RAM to ask).
CLASSES = ["Player", "Grunt", "Electrode", "Hulk", "Brain", "Enforcer",
           "EnfBullet/Spark", "Spheroid", "Quark", "Tank", "TankShell",
           "Cruise Missile", "Prog", "Mikey", "Mom", "Dad"]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}

# Nominal sprite sizes in pixels (w, h). Player/grunt sprites are 4 bytes wide
# (8px) x 12 high (e.g. player anim $3603: 04 0C, asm:4338). Small projectiles
# get compact boxes; big enemies wider ones. v1 approximations.
BOX = {
    "Player": (8, 12), "Grunt": (8, 12), "Electrode": (8, 10),
    "Hulk": (12, 14), "Brain": (10, 14), "Enforcer": (8, 10),
    "EnfBullet/Spark": (6, 6), "Spheroid": (10, 10), "Quark": (10, 10),
    "Tank": (12, 10), "TankShell": (6, 6), "Cruise Missile": (6, 8),
    "Prog": (8, 12), "Mikey": (8, 12), "Mom": (8, 12), "Dad": (8, 12),
}


def fsm_action(builder, packet):
    sprites = builder._sprites_from_packet(packet)
    player = next((s for s in sprites if s[2] == "Player"), None)
    if player is None:
        return 1, 1
    px, py = player[0], player[1]
    others = [s for s in sprites if s[2] != "Player"]
    d2 = lambda s: (s[0] - px) ** 2 + (s[1] - py) ** 2  # noqa: E731
    used, sel = set(), []
    for cnt, types in SLOT_CATEGORIES:
        for s in sorted([s for s in others if s[2] in types and id(s) not in used], key=d2)[:cnt]:
            used.add(id(s)); sel.append(s)
    sel += sorted([s for s in others if id(s) not in used], key=d2)[:CATCHALL_SLOTS]
    data = [(px, py, "Player")] + [(s[0], s[1], s[2]) for s in sel]
    try:
        mv, fr = fsm.chooseOutputs(data)
    except Exception:
        mv, fr = 1, 1
    return max(mv, 1), max(fr, 1)


def labels_from_packet(packet) -> list[str]:
    h = parse_header(packet)
    rows = []

    def add(name, px_x, px_y):
        cid = CLASS_ID.get(name)
        if cid is None:
            return
        w, hgt = BOX[name]
        # blitter dest = sprite TOP-LEFT corner; YOLO wants the box center
        cx, cy = (px_x + w / 2) / SCREEN_W, (px_y + hgt / 2) / SCREEN_H
        if not (0 <= cx <= 1 and 0 <= cy <= 1):
            return
        rows.append(f"{cid} {cx:.4f} {cy:.4f} {w / SCREEN_W:.4f} {hgt / SCREEN_H:.4f}")

    # player: header x is in BYTES (2px), y in lines
    add("Player", h["player_x"] * 2, h["player_y"])
    for _addr, _lst, sw, x, y in iter_entities(packet):
        name = classify_sw(sw)
        if name in (None, "PlayerIcon", "PlayerBullet"):
            continue
        add(name, x * 2, y)
    return rows


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "yolo_dataset")
    n_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 9956
    snap_every = int(os.environ.get("SNAP_EVERY", "8"))
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "classes.txt").write_text("\n".join(CLASSES) + "\n")

    snap_dir = Path(os.environ.get("MAME_SNAP_DIR", str(out / "_snaps")))
    snap_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MAME_SNAP_DIR"] = str(snap_dir)

    # evolved champion params for realistic play
    src = os.environ.get("EVOLVE_PARAMS", str(ROOT / "models" / "fsm_evolved_combteacher.json"))
    if Path(src).exists():
        import json
        for k, v in json.loads(Path(src).read_text())["best_params"].items():
            setattr(fsm, k, v)
    fsm.DEBUG_LEVEL = 0
    W, Hpx = 665, 492
    fsm.MAX_RIGHT, fsm.MAX_TOP = W, Hpx
    fsm.MAX_BOTTOM, fsm.MAX_LEFT = 0, 0
    fsm.Y_AXIS_INVERSION = Hpx
    B = float(getattr(fsm, "BORDER_ADJUST", 20))
    fsm.ADJ_TOP, fsm.ADJ_BOTTOM = Hpx - 2, 0 + B + 9
    fsm.ADJ_LEFT, fsm.ADJ_RIGHT = 0 + 2, W - B

    # state pool: boot + every saved ladder state (wave diversity)
    states = [0] + sorted(
        int(m.group(1)) for f in (Path(STATE_DIR) / "robotron").glob("w5_*.sta")
        if (m := re.match(r"w5_(\d+)\.sta", f.name)))
    random.seed(1234)

    bridge = MameBridge(port=port)
    builder = MameObsBuilder()
    frames = 0
    ep = 0
    t0 = time.time()
    while frames < n_frames:
        idx = 0 if ep % 4 == 0 else random.choice(states)  # 25% wave-1 boots
        ep += 1
        try:
            packet = bridge.reset(idx)
        except Exception:
            continue
        builder.reset()
        for step in range(600):
            if frames >= n_frames:
                break
            mv, fr = fsm_action(builder, packet)
            if step % snap_every == 0:
                held = packet                      # frame that the PNG will show
                before = set(snap_dir.rglob("*.png"))
                packet, rec = bridge.snapshot(), False
                new = set(snap_dir.rglob("*.png")) - before
                if new:
                    png = new.pop()
                    rows = labels_from_packet(held)
                    name = f"f{frames:06d}"
                    shutil.move(str(png), out / "images" / f"{name}.png")
                    (out / "labels" / f"{name}.txt").write_text("\n".join(rows) + "\n")
                    frames += 1
                    if frames % 50 == 0:
                        print(f"{frames}/{n_frames} frames "
                              f"({frames / (time.time() - t0):.1f}/s)", flush=True)
            pk, recovered = bridge.step(mv, fr)
            packet = pk
            h = parse_header(packet)
            if recovered or h["lives"] == 0 or h["game_state"] == 0x1B:
                break
    bridge.close()
    print(f"done: {frames} labeled frames in {out}")


if __name__ == "__main__":
    main()
