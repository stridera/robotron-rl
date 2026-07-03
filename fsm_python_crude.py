"""Run the SAME crude potential-field FSM (as fsm_oracle on MAME) on the PYTHON gym.
Disambiguates: if this crude FSM reaches deep here but caps at ~4 on MAME, the ENV is
the difference. If it also caps low here, the FSM is just crude (not an env defect)."""
import sys
import math
import statistics
from collections import Counter
from os import path

sys.path.insert(0, path.dirname(__file__))
from robotron import RobotronEnv

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
YSIGN = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0   # flip if fleeing wrong way
FAMILY = {"Mommy", "Daddy", "Mikey"}
NONTHREAT = FAMILY | {"Player"}


def compass_dir(dx, dy):
    ang = math.atan2(dy, dx)
    sect = int(round(ang / (math.pi / 4))) % 8
    return (3, 4, 5, 6, 7, 8, 1, 2)[sect]


def act(objs, W, H):
    player = next((o for o in objs if o[2] == "Player"), None)
    if player is None:
        return 0, 0
    px, py = player[0], player[1]
    threats = [(o[0], o[1]) for o in objs if o[2] not in NONTHREAT]
    fire = 0
    if threats:
        nx, ny = min(threats, key=lambda t: max(abs(t[0] - px), abs(t[1] - py)))
        fire = compass_dir(nx - px, (ny - py) * YSIGN)
    fx = fy = 0.0
    R = 130.0
    for ex, ey in threats:
        dx, dy = px - ex, (py - ey)
        d2 = dx * dx + dy * dy + 1.0
        if d2 < R * R:
            fx += dx / d2 * 3000
            fy += dy / d2 * 3000
    m = 60
    if px < m: fx += (m - px) * 0.3
    if W - px < m: fx -= (m - (W - px)) * 0.3
    if py < m: fy += (m - py) * 0.3
    if H - py < m: fy -= (m - (H - py)) * 0.3
    if abs(fx) < 1e-6 and abs(fy) < 1e-6:
        fx, fy = W / 2 - px, H / 2 - py
    move = compass_dir(fx, fy * YSIGN)
    return move, fire


config = path.join(path.dirname(__file__), "config.yaml")
env = RobotronEnv(level=1, lives=3, fps=0, config_path=config, headless=True)
W, H = env.get_board_size()
results = []
for ep in range(N):
    env.reset()
    _, _, dead, trunc, data = env.step(0)
    maxlevel = data.get("level", 1)
    steps = 0
    while not (dead or trunc):
        mv, fr = act(data["data"], W, H)
        _, _, dead, trunc, data = env.step(mv * 9 + fr)
        maxlevel = max(maxlevel, data.get("level", 1))
        steps += 1
        if steps > 40000:
            break
    results.append(maxlevel)
    print(f"ep {ep+1}: level {maxlevel} ({steps} steps)", flush=True)
print(f"\n=== {N} CRUDE-FSM runs (PYTHON gym) ysign={YSIGN} ===")
print(f"level: min={min(results)} max={max(results)} mean={statistics.mean(results):.1f} "
      f"median={statistics.median(results)}  dist={dict(sorted(Counter(results).items()))}")
env.close()
