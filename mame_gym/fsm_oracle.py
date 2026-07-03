"""fsm_oracle.py — a SIMPLE flee-and-shoot FSM driven directly in the MAME env.

Oracle test for "why is RL stuck at median wave 3 on MAME?": if a simple rule-based
strategy (the kind that reaches wave 8-9 on the real game) reaches deep waves HERE,
then RL's median-3 is a pipeline bug, not a fundamental limit. If this FSM ALSO caps
near wave 3, the MAME env itself is the suspect.

Logic (potential field): fire toward the nearest threat; move along the repulsion
vector away from nearby threats + away from the four walls; drift to center if clear.
Reads the raw packet (ground-truth positions/types), NOT the 945-dim obs — so it also
isolates whether the obs encoding is the bottleneck (compare vs an obs-fed FSM later).
"""
import sys
import math
import statistics
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from mame_robotron_env import MameRobotronEnv
from mame_obs import parse_header, iter_entities

GX_MIN, GX_MAX = 5, 145
GY_MIN, GY_MAX = 15, 230
CX, CY = (GX_MIN + GX_MAX) / 2, (GY_MIN + GY_MAX) / 2
FAMILY_LIST = 2

N = int(sys.argv[1]) if len(sys.argv) > 1 else 15
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 9982
FRAMESKIP = int(sys.argv[3]) if len(sys.argv) > 3 else 4


def compass_dir(dx, dy):
    # dx>0=right(E), dy>0=down(S). Returns game dir 1=N 2=NE 3=E 4=SE 5=S 6=SW 7=W 8=NW.
    ang = math.atan2(dy, dx)
    sect = int(round(ang / (math.pi / 4))) % 8
    return (3, 4, 5, 6, 7, 8, 1, 2)[sect]


def fsm_action(packet):
    h = parse_header(packet)
    px, py = h["player_x"], h["player_y"]
    threats, family = [], []
    for addr, lid, sw, ex, ey in iter_entities(packet):
        (family if lid == FAMILY_LIST else threats).append((ex, ey))
    # FIRE toward nearest threat
    fire = 0
    if threats:
        nx, ny = min(threats, key=lambda t: max(abs(t[0] - px), abs(t[1] - py)))
        fire = compass_dir(nx - px, ny - py)
    # MOVE along repulsion from nearby threats + walls
    fx = fy = 0.0
    nearest_threat_d = min((max(abs(t[0]-px), abs(t[1]-py)) for t in threats), default=999)
    for ex, ey in threats:
        dx, dy = px - ex, py - ey
        d2 = dx * dx + dy * dy + 1.0
        if d2 < 55 * 55:
            fx += dx / d2 * 500
            fy += dy / d2 * 500
    _ = (family, nearest_threat_d)   # family-seeking reverted (it walked into deaths)
    margin = 28
    if px - GX_MIN < margin: fx += (margin - (px - GX_MIN)) * 0.30
    if GX_MAX - px < margin: fx -= (margin - (GX_MAX - px)) * 0.30
    if py - GY_MIN < margin: fy += (margin - (py - GY_MIN)) * 0.30
    if GY_MAX - py < margin: fy -= (margin - (GY_MAX - py)) * 0.30
    if abs(fx) < 1e-6 and abs(fy) < 1e-6:   # clear: drift to center
        fx, fy = CX - px, CY - py
    move = compass_dir(fx, fy) if (abs(fx) > 1e-6 or abs(fy) > 1e-6) else (fire or 1)
    return np.array([move - 1, (fire - 1) if fire else 0])


def main():
    env = MameRobotronEnv(rank=0, base_port=PORT, frameskip=FRAMESKIP, reset_pool=[0], obs_mode="slot")
    results = []
    for ep in range(N):
        env.reset()
        packet = env._last_packet
        max_wave = env._last_wave
        final_score = env._last_score
        steps = 0
        start_lives = parse_header(packet)["lives"]
        max_lives_seen = start_lives
        extra_life_earned = False
        prev_lives = start_lives
        while True:
            a = fsm_action(packet)
            _, _, term, trunc, info = env.step(a)
            packet = env._last_packet
            h = parse_header(packet)
            lv = h["lives"]
            if lv > prev_lives:
                extra_life_earned = True
            max_lives_seen = max(max_lives_seen, lv)
            prev_lives = lv
            max_wave = max(max_wave, info.get("wave", 0))
            if not (term or trunc):
                final_score = info.get("score", final_score)
            steps += 1
            if term or trunc:
                break
        results.append((max_wave, final_score))
        print(f"ep {ep+1}: reached=w{max_wave}  score={final_score}  steps={steps}  "
              f"start_lives={start_lives} max_lives={max_lives_seen} extra_earned={extra_life_earned}",
              flush=True)
    waves = [r[0] for r in results]
    print(f"\n=== {N} CONTINUOUS FSM runs (MAME env, from wave 1) ===")
    print(f"wave reached:  min={min(waves)} max={max(waves)} "
          f"mean={statistics.mean(waves):.1f} median={statistics.median(waves)}")
    print(f"score: max={max(r[1] for r in results)} mean={statistics.mean(r[1] for r in results):.0f}")
    print(f"wave distribution: {dict(sorted(Counter(waves).items()))}")
    env.close()


if __name__ == "__main__":
    main()
