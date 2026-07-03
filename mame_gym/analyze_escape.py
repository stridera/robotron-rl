"""Escape-route / death post-mortem analyzer (pillar 2 of the FSM-improvement loop).

For each recorded pre-death trajectory (collect_deaths_fsm.py), reconstruct:
  1. KILLER: the lethal entity that reached kill range (identified from the frames
     BEFORE death, since mutual-destruction unlinks it at the death frame).
  2. NORMAL?: did the killer move within its ASM speed bound (ENEMY_MODEL §4)?
     An out-of-bound "jump" means a decode artifact (node recycling), not a real
     death — flags remaining perception bugs.
  3. ESCAPE?: a few steps before death, was there a move direction whose projected
     path (player extrapolated at observed speed, threats extrapolated at observed
     velocity) kept clearance > kill radius? If yes AND the FSM went elsewhere, the
     death was AVOIDABLE — a target for an FSM fix.

Aggregates the dominant avoidable pattern to drive the next FSM change.

Usage: .venv/bin/python3 mame_gym/analyze_escape.py <trajectories.jsonl>
"""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path

TRAJ = sys.argv[1] if len(sys.argv) > 1 else "deaths_fixed_v1/trajectories.jsonl"
KILL_R = 15
LETHAL = {"Grunt", "Electrode", "Hulk", "Sphereoid", "Quark", "Brain", "Prog",
          "Enforcer", "Tank", "TankShell", "Cruise Missile", "EnfBullet/Spark"}
# ASM per-step (frameskip-4) chebyshev speed ceilings, display units (ENEMY_MODEL §4).
# Generous (2x nominal) so only gross artifacts flag.
SPEED_CAP = {"Quark": 12, "Sphereoid": 12, "Tank": 8, "Hulk": 12, "Brain": 8,
             "Grunt": 12, "Electrode": 2, "Prog": 12, "Enforcer": 20,
             "TankShell": 24, "EnfBullet/Spark": 24, "Cruise Missile": 12}
# 8 move directions -> (dx,dy) in display units (x=col is ~2px, but we work in the
# raw display space the entities live in). N=up=-y.
DIRV = {0:(0,-1),1:(1,-1),2:(1,0),3:(1,1),4:(0,1),5:(-1,1),6:(-1,0),7:(-1,-1)}

# Everything is destructible by the player laser EXCEPT the Hulk (its $00B6 handler
# only shoves it — asm:471). Sparks ($14DC) and tank shells ($4FD5) DO die to a
# laser hit (asm:14E0/4FD9, +25 pts). So for any non-Hulk incoming killer, shooting
# it is a valid escape if there was time + aim.
UNSHOOTABLE = {"Hulk"}

def cheby(ax, ay, bx, by): return max(abs(ax-bx), abs(ay-by))

import math
_UNIT = {d: (vx/math.hypot(vx, vy), vy/math.hypot(vx, vy)) for d, (vx, vy) in
         {0:(0,-1),1:(1,-1),2:(1,0),3:(1,1),4:(0,1),5:(-1,1),6:(-1,0),7:(-1,-1)}.items()}
def octant(dx, dy):
    if dx == 0 and dy == 0: return None
    n = math.hypot(dx, dy)
    return max(_UNIT, key=lambda d: (_UNIT[d][0]*dx + _UNIT[d][1]*dy)/n)
def octant_adj(a, b):   # within +/-1 octant (45 deg)
    if a is None or b is None: return False
    return min((a-b) % 8, (b-a) % 8) <= 1

def track(frames, addr):
    """Per-frame (x,y) for a node addr across history (None where absent)."""
    return [next((e for e in f["ents"] if e["a"] == addr), None) for f in frames]

def analyze(rec):
    frames, acts = rec["frames"], rec["actions"]
    if len(frames) < 4:
        return None
    death = frames[-1]
    dpx, dpy = death["px"], death["py"]
    # Killer: nearest lethal to the player over the last 3 frames (the killer may be
    # gone at the death frame; take the closest approach across the tail).
    best = None
    for f in frames[-3:]:
        for e in f["ents"]:
            if e["n"] not in LETHAL:
                continue
            d = cheby(e["x"], e["y"], f["px"], f["py"])
            if best is None or d < best[0]:
                best = (d, e["n"], e["a"], f)
    if best is None:
        return {"killer": None}
    kd, kname, kaddr, _ = best
    # Killer behaved normally? max per-step speed vs cap.
    tr = [e for e in track(frames, kaddr) if e]
    maxstep = 0
    for a, b in zip(tr, tr[1:]):
        maxstep = max(maxstep, cheby(a["x"], a["y"], b["x"], b["y"]))
    normal = maxstep <= SPEED_CAP.get(kname, 20)
    # Escape? evaluate at t* = 3 steps before death.
    t = max(0, len(frames) - 4)
    f0 = frames[t]
    pspeed = 6
    # observed player step size (median), fallback 6
    steps = [cheby(frames[i]["px"], frames[i]["py"], frames[i+1]["px"], frames[i+1]["py"])
             for i in range(len(frames)-1)]
    steps = [s for s in steps if s > 0]
    if steps: pspeed = sorted(steps)[len(steps)//2]
    threats = [e for e in f0["ents"] if e["n"] in LETHAL]
    # threat velocities from t-1 -> t
    tvel = {}
    if t > 0:
        for e in threats:
            p = next((x for x in frames[t-1]["ents"] if x["a"] == e["a"]), None)
            tvel[e["a"]] = (e["x"]-p["x"], e["y"]-p["y"]) if p else (0, 0)
    HOR = 3   # look-ahead steps
    def clearance(dx, dy):
        cx, cy, mind = f0["px"], f0["py"], 1e9
        for k in range(1, HOR+1):
            cx += dx*pspeed; cy += dy*pspeed
            for e in threats:
                vx, vy = tvel.get(e["a"], (0, 0))
                ex, ey = e["x"]+vx*k, e["y"]+vy*k
                mind = min(mind, cheby(cx, cy, ex, ey))
        return mind
    dir_clear = {d: clearance(*v) for d, v in DIRV.items()}
    best_dir = max(dir_clear, key=dir_clear.get)
    fsm_move = acts[t][0] if t < len(acts) else None
    fsm_fire = acts[t][1] if t < len(acts) else None
    fsm_clear = dir_clear.get(fsm_move, -1)
    MARGIN = 8
    # DODGE escape: a STRICTLY BETTER safe move existed than the FSM's own.
    dodge = (dir_clear[best_dir] > KILL_R) and (dir_clear[best_dir] > fsm_clear + MARGIN)
    attributed = kd <= 22
    # SHOOT escape: for a non-Hulk killer, was the FSM aiming at it? and how
    # crowded was it (can't shoot your way out of a multi-direction swarm).
    kf = best[3]
    ke = next((e for e in kf["ents"] if e["a"] == kaddr), None)
    shootable = kname not in UNSHOOTABLE
    kill_oct = octant(ke["x"]-kf["px"], ke["y"]-kf["py"]) if ke else None
    fire_at_killer = octant_adj(fsm_fire, kill_oct)
    n_near = sum(1 for e in threats if cheby(e["x"], e["y"], f0["px"], f0["py"]) <= 35)
    return {"killer": kname, "kd": kd, "normal": normal, "maxstep": maxstep,
            "attributed": attributed, "dodge": dodge, "best_dir": best_dir,
            "best_clear": round(dir_clear[best_dir],1),
            "shootable": shootable, "fire_at_killer": fire_at_killer, "n_near": n_near,
            "fsm_move": fsm_move, "fsm_fire": fsm_fire, "wave": rec["wave"]}

def main():
    recs = [json.loads(l) for l in open(TRAJ)]
    allr = [r for r in (analyze(x) for x in recs) if r and r.get("killer")]
    res = [r for r in allr if r["attributed"]]
    n = len(res)
    print(f"{TRAJ}: {len(recs)} deaths, {len(allr)} with killer, {n} attributed (kd<=22)\n")

    def category(r):
        if not r["normal"]:                 return "decode-artifact"
        if r["dodge"]:                      return "dodge-avoidable"
        if r["shootable"] and r["n_near"] >= 3: return "swarm(hard)"
        if r["shootable"] and not r["fire_at_killer"]: return "shoot-avoidable"
        if r["shootable"] and r["fire_at_killer"]:     return "shot-but-too-late"
        return "hulk-dodge(hard)"   # non-shootable, no better move

    cats = Counter(category(r) for r in res)
    print("death taxonomy (attributed, corrected labels):")
    for c, k in cats.most_common(): print(f"  {c:<20} {k:>4} ({100*k/n:.0f}%)")

    for label in ("dodge-avoidable", "shoot-avoidable"):
        sub = [r for r in res if category(r) == label]
        print(f"\n{label.upper()} ({len(sub)}, {100*len(sub)/n:.0f}%) — by killer:")
        for kk, c in Counter(r["killer"] for r in sub).most_common():
            print(f"  {kk:<16} {c:>4}")
        for r in sub[:6]:
            print(f"  wave{r['wave']:>2} {r['killer']:<14} kd={r['kd']} n_near={r['n_near']} "
                  f"move={r['fsm_move']} fire={r['fsm_fire']} fire_at_killer={r['fire_at_killer']}")

    print(f"\ndecode-artifact (node recycling, perception bug): {cats['decode-artifact']} "
          f"({100*cats['decode-artifact']/n:.0f}%) — fix item 5")
    print("\n=> shootable single/few-threat deaths where the FSM wasn't aiming at the")
    print("   killer are the new lever: prioritize firing at the nearest INCOMING threat.")

if __name__ == "__main__":
    main()
