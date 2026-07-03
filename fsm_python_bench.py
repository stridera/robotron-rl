"""Run the FSM's own decision logic (chooseOutputs) on the PYTHON gym, continuous
episodes, report the level reached. Compared with fsm_oracle on MAME, this isolates
whether the wave ceiling is env-specific (python allows deep, MAME caps ~4)."""
import sys
import statistics
from collections import Counter
from os import path

sys.path.insert(0, path.dirname(__file__))
from robotron import RobotronEnv
import robotron_fsm as fsm

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
config = path.join(path.dirname(fsm.__file__), "config.yaml")
env = RobotronEnv(level=1, lives=3, fps=0, config_path=config, headless=True)
bw, bh = env.get_board_size()
# replicate robotron_fsm.main()'s global setup
fsm.DEBUG_LEVEL = 0
fsm.MAX_RIGHT, fsm.MAX_TOP = bw, bh
fsm.MAX_BOTTOM = 0
fsm.MAX_LEFT = 0
fsm.Y_AXIS_INVERSION = bh
B = 20
fsm.ADJ_TOP = bh - 2
fsm.ADJ_BOTTOM = 0 + B + 9
fsm.ADJ_LEFT = 0 + 2
fsm.ADJ_RIGHT = bw - B

results = []
for ep in range(N):
    env.reset()
    _, _, dead, trunc, data = env.step(0)
    maxlevel = data.get("level", 1)
    steps = 0
    while not (dead or trunc):
        action = fsm.chooseOutputs(data["data"])
        enc = action[0] * 9 + action[1]
        _, _, dead, trunc, data = env.step(enc)
        maxlevel = max(maxlevel, data.get("level", 1))
        steps += 1
        if steps > 40000:
            break
    results.append(maxlevel)
    print(f"ep {ep+1}: level {maxlevel}  ({steps} steps)", flush=True)

print(f"\n=== {N} FSM runs (PYTHON gym, from level 1, 3 lives) ===")
print(f"level: min={min(results)} max={max(results)} mean={statistics.mean(results):.1f} "
      f"median={statistics.median(results)}")
print(f"distribution: {dict(sorted(Counter(results).items()))}")
env.close()
