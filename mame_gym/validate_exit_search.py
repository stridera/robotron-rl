"""Validate the ExIt search teacher beats the FSM on the MAME gym.

Drives the bridge directly (fair A/B; same loop for both) from each start state:
  - FSM policy        : fsm_move_action(packet)
  - search teacher    : search_move_action(bridge, packet, K)
Reports wave reached + score for each. The teacher should match-or-exceed.

Run: .venv/bin/python3 mame_gym/validate_exit_search.py [K] [max_steps] [port]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from mame_robotron_env import MameRobotronEnv
from mame_obs import parse_header
from search_oracle import search_move_action, fsm_move_action

K         = int(sys.argv[1]) if len(sys.argv) > 1 else 4
MAX_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1200
PORT      = int(sys.argv[3]) if len(sys.argv) > 3 else 9977


def run(bridge, start_idx, policy, max_steps):
    pkt = bridge.reset(start_idx)
    h = parse_header(pkt)
    base_wave, max_wave, score = h["wave"], h["wave"], h["score"]
    for _ in range(max_steps):
        if policy == "fsm":
            mv, fr = fsm_move_action(pkt)
        else:
            mv, fr, _ = search_move_action(bridge, pkt, H=K)
        pkt, recovered = bridge.step(mv, fr)
        if recovered:
            break
        h = parse_header(pkt)
        if h["lives"] == 0 or not (base_wave <= h["wave"] <= max_wave + 1):
            break
        max_wave = max(max_wave, h["wave"])
        score = h["score"]
        base_wave = h["wave"]   # advance the legitimate-wave window
    return max_wave, score


def main():
    env = MameRobotronEnv(rank=0, base_port=PORT, frameskip=4, reset_pool=[0], obs_mode="slot")
    env.reset()                      # boots the MAME instance
    br = env._bridge
    starts = [0]                     # wave-1 boot (deterministic). Add pool idxs for more.
    print(f"K={K} max_steps={MAX_STEPS}")
    for s in starts:
        fw, fsc = run(br, s, "fsm", MAX_STEPS)
        sw, ssc = run(br, s, "search", MAX_STEPS)
        print(f"start={s}: FSM wave={fw} score={fsc:,} | SEARCH wave={sw} score={ssc:,} "
              f"| dwave={sw-fw:+d} dscore={ssc-fsc:+,}")
    env.close()


if __name__ == "__main__":
    main()
