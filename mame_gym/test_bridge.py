"""Decisive test: does input injection actually move the player when applied
every frame? Drive Move Right, then Move Left, then idle. Log player X/Y over
time. If X moves, slot+0 is confirmed X and inputs work. If it doesn't, we
know that's the real blocker."""
from __future__ import annotations
import sys, time
from robotron_mame_env import RobotronMAMEEnv

DIR = dict(none=0, N=1, NE=2, E=3, SE=4, S=5, SW=6, W=7, NW=8)

def log(env, obs, tag, t0):
    d = env.decode(obs)
    active = sum(1 for s in d["slots"] if s["sw"] != 0)
    print(f"[t+{time.perf_counter()-t0:5.2f}s {tag:>10s}] "
          f"player X={d['player_x']:3d} Y={d['player_y']:3d}  active_slots={active}")

def main():
    env = RobotronMAMEEnv(verbose=False)
    t0 = time.perf_counter()
    try:
        obs, _ = env.reset()
        log(env, obs, "reset", t0)

        # Hold Move Right (E) every frame for 600 steps; sample every 60.
        for phase, dirname in [("RIGHT", "E"), ("LEFT", "W"),
                               ("DOWN", "S"),  ("UP",  "N"), ("IDLE", "none")]:
            d = DIR[dirname]
            for i in range(600):
                obs, *_ = env.step([d, 0])
                if i % 60 == 0:
                    log(env, obs, phase, t0)
        steps = 600 * 5
        elapsed = time.perf_counter() - t0
        print(f"\n[done] {steps} steps in {elapsed:.2f}s = {steps/elapsed:.0f} steps/sec")
    finally:
        env.close()

if __name__ == "__main__":
    sys.exit(main())
