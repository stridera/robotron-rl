"""Diagnose what kills the player in the early waves, using the CORRECT typed
entity list (iter_entities) at the exact death instant (game_state==0x1B).
The $98D4 'slot pool' that the old forensics used overlaps font memory."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from collections import Counter
from mame_bridge import MameBridge
from mame_obs import parse_header, iter_entities, _LIST1_SW, _LIST2_SW, _LIST3_SW


def name_of(list_id, sw):
    if list_id == 4: return "Electrode"
    if list_id == 1: return _LIST1_SW.get(sw, "TankShell")
    if list_id == 2: return _LIST2_SW.get(sw)
    if list_id == 3: return _LIST3_SW.get(sw)
    return None


def main():
    rng = np.random.default_rng(11)
    b = MameBridge(port=9974, frameskip=4, boot_timeout=90)
    killers = Counter(); deaths = 0
    for epi in range(8):
        pkt = b.reset(0); prev_gs = 0; prevpkt = pkt
        for step in range(1200):
            pkt, rec = b.step(int(rng.integers(1, 9)), int(rng.integers(0, 9)))
            h = parse_header(pkt)
            if h['game_state'] == 0x1B and prev_gs != 0x1B:
                deaths += 1
                ph = parse_header(prevpkt); px, py = ph['player_x'], ph['player_y']
                near = []
                for addr, lid, sw, x, y in iter_entities(prevpkt):
                    near.append((max(abs(x - px), abs(y - py)), name_of(lid, sw), x, y))
                near.sort()
                print(f"death {deaths} wave {h['wave']} player=({px},{py}): "
                      f"nearest={[(d, nm) for d, nm, _, _ in near[:4]]}", flush=True)
                if near:
                    killers[near[0][1]] += 1
            prev_gs = h['game_state']; prevpkt = pkt
            if rec or h['lives'] == 0:
                break
    b.close()
    print(f"\nKILLER tally (nearest at death, {deaths} deaths):", dict(killers.most_common()), flush=True)
    print("DIAG DONE", flush=True)


if __name__ == "__main__":
    main()
