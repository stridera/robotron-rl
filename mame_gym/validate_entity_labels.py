"""Independent entity-label validation (built 2026-07-01).

The old visual tools labeled sprites with OUR OWN classifier, so they could only
confirm self-consistency, never catch an SW-table mislabel (tank shells labeled
Quark went undetected for weeks). This tool cross-checks every decoded label
against TWO sources that are INDEPENDENT of the +8/9 collision-handler SW we type
on:

  1. The animation-frame pointer (node+$02, gated by MAME_EMIT_ANIM=1) — the
     actual sprite bitmap the hardware blits. If two labels share an anim cluster
     they're the same sprite; if one label spans two clusters it's a merge/mislabel.
  2. Measured per-step velocity vs the ASM motion bounds (ENEMY_MODEL.md §4):
     a "Quark" that moves like a fast bouncing projectile is not a quark.

Drives the champion evolved FSM (reaches wave ~12, where tanks/shells appear).
Snapshots on first sighting of each label and on anomalies (SW-change at a reused
node = recycling; >25 u/step jump = identity aliasing).

Usage:
  MAME_RL_RESEED=1 EVOLVE_PARAMS=models/fsm_evolved_combteacher.json \
  .venv/bin/python3 mame_gym/validate_entity_labels.py [n_games] [port] [out_dir]
"""
import os, sys, json, shutil, time
from collections import Counter, defaultdict
from pathlib import Path

os.environ["MAME_EMIT_ANIM"] = "1"   # set BEFORE importing the bridge (it reads
                                     # this at module load to size the obs read)
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from mame_robotron_env import MameRobotronEnv
from mame_obs import parse_header, iter_entities, classify_sw, ENTITY_ARRAY_OFFSET
import robotron_fsm as fsm

N_GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 6
PORT    = int(sys.argv[2]) if len(sys.argv) > 2 else 9964
OUT_DIR = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("/tmp/validate_labels")
SNAP_DIR = Path("/home/strider/.mame/snap/robotron")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- FSM setup (mirrors fsm_choose_on_mame.py) --------------------------------
W, H = 665, 492
fsm.DEBUG_LEVEL = 0
fsm.MAX_RIGHT, fsm.MAX_TOP = W, H
fsm.MAX_BOTTOM, fsm.MAX_LEFT = 0, 0
fsm.Y_AXIS_INVERSION = H
_ep = os.environ.get("EVOLVE_PARAMS", "")
if _ep and Path(_ep).exists():
    for n, v in json.loads(Path(_ep).read_text())["best_params"].items():
        setattr(fsm, n, v)
B = float(getattr(fsm, "BORDER_ADJUST", 20))
fsm.ADJ_TOP, fsm.ADJ_BOTTOM, fsm.ADJ_LEFT, fsm.ADJ_RIGHT = H - 2, 0 + B + 9, 0 + 2, W - B
GX_MIN, GX_R, GY_MIN, GY_R = 5, 140, 15, 215
def topix(gx, gy): return ((gx - GX_MIN) / GX_R * W, (gy - GY_MIN) / GY_R * H)
FSMNAME = {"Brain": "Brain", "Brain (alt)": "Brain", "Cruise Missile": "CruiseMissile",
           "Dad": "Daddy", "Electrode": "Electrode", "EnfBullet/Spark": "EnforcerBullet",
           "Enforcer": "Enforcer", "Grunt": "Grunt", "Hulk": "Hulk", "Mikey": "Mikey",
           "Mom": "Mommy", "Quark": "Quark", "Spheroid": "Sphereoid", "Tank": "Tank",
           "TankShell": "TankShell"}

def build_data(packet):
    h = parse_header(packet)
    objs = [(*topix(h["player_x"], h["player_y"]), "Player")]
    for addr, lid, sw, x, y in iter_entities(packet):
        pn = FSMNAME.get(classify_sw(sw))
        if pn is None:
            continue
        objs.append((*topix(x, y), pn))
    return objs

# ---- anim-ptr trailing block --------------------------------------------------
def iter_with_anim(packet):
    """Yield (addr, list_id, sw, x, y, anim_ptr). anim_ptr from the n*2-byte block
    appended after the entity array when MAME_EMIT_ANIM=1 (0 if absent)."""
    n = packet[ENTITY_ARRAY_OFFSET]
    rec_base = ENTITY_ARRAY_OFFSET + 1
    anim_base = rec_base + n * 7
    have_anim = len(packet) >= anim_base + n * 2
    for i in range(n):
        off = rec_base + i * 7
        addr = (packet[off] << 8) | packet[off + 1]
        sw = (packet[off + 3] << 8) | packet[off + 4]
        anim = 0
        if have_anim:
            ao = anim_base + i * 2
            anim = (packet[ao] << 8) | packet[ao + 1]
        yield addr, packet[off + 2], sw, packet[off + 5], packet[off + 6], anim

# Known anim-frame ranges from robomame.asm (ENEMY_MODEL.md §5) for name-check.
KNOWN_ANIM = [
    (0x1A34, 0x1A40, "Spark"), (0x206B, 0x2080, "CruiseMissile"),
    (0x4FEE, 0x50E0, "TankShell"), (0x35AE, 0x35D0, "PlayerLaser"),
]
def anim_name(a):
    for lo, hi, nm in KNOWN_ANIM:
        if lo <= a <= hi:
            return nm
    return "?"

# ASM per-step (frameskip-4) chebyshev velocity expectation, display units.
# Loose: this is a sanity band, the anim cluster is the primary oracle.
SLOW = {"Quark", "Sphereoid", "Tank", "Hulk", "Brain", "Grunt", "Electrode",
        "Mikey", "Mommy", "Daddy"}          # ground movers, expect <= ~12 u/step
FAST = {"TankShell", "EnforcerBullet", "CruiseMissile", "Enforcer"}  # projectiles/divers

def snap(env, tag, note):
    tag = tag.replace("/", "-").replace(" ", "").replace("(", "").replace(")", "")
    before = set(SNAP_DIR.glob("*.png")) if SNAP_DIR.exists() else set()
    try:
        env._bridge.snapshot(); time.sleep(0.3)
    except Exception as e:
        print(f"  [snap failed: {e}]"); return
    new = sorted((set(SNAP_DIR.glob("*.png")) if SNAP_DIR.exists() else set()) - before)
    if new:
        dst = OUT_DIR / f"{tag}.png"
        shutil.copy(new[-1], dst)
        print(f"  [snap -> {dst.name}] {note}")

def main():
    env = MameRobotronEnv(rank=0, base_port=PORT, frameskip=4, reset_pool=[0], obs_mode="slot")
    label_anim = defaultdict(Counter)   # label -> {anim_ptr: count}
    label_sw   = defaultdict(Counter)   # label -> {sw: count}
    label_vel  = defaultdict(list)      # label -> [cheby per-step velocity]
    seen_labels = set()
    anomalies = []
    prev = {}   # addr -> (sw, x, y)
    anim_checked = False
    total_steps = 0

    for game in range(N_GAMES):
        env.reset()
        pkt = env._last_packet
        prev.clear()
        maxw = env._last_wave
        while True:
            if not anim_checked:
                n = pkt[ENTITY_ARRAY_OFFSET]
                need = ENTITY_ARRAY_OFFSET + 1 + n * 7 + n * 2
                print(f"anim block present: {len(pkt) >= need} (packet={len(pkt)}, need>={need}, n={n})")
                anim_checked = True
            h = parse_header(pkt)
            for addr, lid, sw, x, y, anim in iter_with_anim(pkt):
                name = classify_sw(sw)
                if name is None:
                    continue
                label_anim[name][f"0x{anim:04X}"] += 1
                label_sw[name][f"0x{sw:04X}"] += 1
                p = prev.get(addr)
                if p is not None:
                    if p[0] != sw:   # same node, different SW = slot recycling
                        anomalies.append((h["wave"], "RECYCLE", addr, f"0x{p[0]:04X}->0x{sw:04X}", name))
                        if len(anomalies) <= 8:
                            snap(env, f"anom_recycle_w{h['wave']}_{addr:04X}", f"node {addr:#06x} SW {p[0]:#06x}->{sw:#06x}")
                    else:
                        d = max(abs(x - p[1]), abs(y - p[2]))
                        if d > 25:   # bigger than any real 1-step move = aliasing
                            anomalies.append((h["wave"], "JUMP", addr, f"{d}u/step", name))
                            if len([a for a in anomalies if a[1] == "JUMP"]) <= 6:
                                snap(env, f"anom_jump_w{h['wave']}_{addr:04X}", f"{name} jumped {d}u/step")
                        else:
                            label_vel[name].append(d)
                if name not in seen_labels:
                    seen_labels.add(name)
                    snap(env, f"first_{name}_w{h['wave']}", f"first {name} (sw 0x{sw:04X} anim 0x{anim:04X})")
            prev = {addr: (sw, x, y) for addr, lid, sw, x, y, _ in iter_with_anim(pkt)}

            try:
                mv, fr = fsm.chooseOutputs(build_data(pkt))
            except Exception:
                mv, fr = 1, 1
            move_idx = (mv - 1) if mv >= 1 else 0
            fire_idx = (fr - 1) if fr >= 1 else move_idx
            _, _, t, tr, info = env.step(np.array([move_idx, fire_idx]))
            pkt = env._last_packet
            maxw = max(maxw, info.get("wave", 0))
            total_steps += 1
            if t or tr:
                break
        print(f"game {game+1}: reached wave {maxw}")
    env.close()

    # ---- report ----
    def pct(c):
        tot = sum(c.values()) or 1
        return ", ".join(f"{k}:{v}({100*v/tot:.0f}%)" for k, v in c.most_common(4))
    print("\n" + "=" * 78)
    print("PER-LABEL VALIDATION  (label <- our SW table;  anim/vel = INDEPENDENT check)")
    print("=" * 78)
    hdr = f"{'label':<15}{'n':>6}  {'vel med/p95/max':>16}  anim-ptr clusters (independent identity)"
    print(hdr)
    for name in sorted(label_anim, key=lambda k: -sum(label_anim[k].values())):
        v = sorted(label_vel[name])
        vs = f"{v[len(v)//2]}/{v[int(len(v)*0.95)]}/{v[-1]}" if v else "-"
        top = label_anim[name].most_common(3)
        cl = ", ".join(f"0x{int(a,16):04X}[{anim_name(int(a,16))}]:{c}" for a, c in top)
        band = "FAST" if name in FAST else ("SLOW" if name in SLOW else "?")
        print(f"{name:<15}{sum(label_anim[name].values()):>6}  {vs:>16}  {cl}")
    print("\nKEY CROSS-CHECK — do Quark / Tank / TankShell resolve to DISTINCT sprites?")
    for name in ("Quark", "Tank", "TankShell"):
        if name in label_anim:
            print(f"  {name:<10} sw={pct(label_sw[name])}  anim={pct(label_anim[name])}")
    print(f"\nanomalies: {len(anomalies)} (RECYCLE={sum(1 for a in anomalies if a[1]=='RECYCLE')}, "
          f"JUMP={sum(1 for a in anomalies if a[1]=='JUMP')})")
    for a in anomalies[:12]:
        print(f"  wave{a[0]:>2} {a[1]:<8} node0x{a[2]:04X} {a[3]} ({a[4]})")
    print(f"\nsnapshots + this report: {OUT_DIR}")

if __name__ == "__main__":
    main()
