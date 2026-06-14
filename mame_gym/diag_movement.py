"""Clean speed check: reset before each measurement to avoid wall/drift
artifacts. Player speed (hold a direction from center) vs grunt speed (track one
grunt by stable node address over idle steps)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from mame_bridge import MameBridge
from mame_obs import parse_header, iter_entities

DIRS = {1: "up", 3: "right", 5: "down", 7: "left"}


def main():
    b = MameBridge(port=9978, frameskip=4, boot_timeout=90)
    print("=== player speed (reset, hold dir 8 steps from center) ===")
    for d in (1, 3, 5, 7):
        h0 = parse_header(b.reset(0)); x0, y0 = h0["player_x"], h0["player_y"]
        for _ in range(8):
            pkt, _ = b.step(d, 0)
        h = parse_header(pkt)
        dx, dy = h["player_x"] - x0, h["player_y"] - y0
        dist = abs(dx) + abs(dy)
        print(f"  {DIRS[d]:>6}: dx={dx:+d} dy={dy:+d}  ({dist} u / 8 steps = {dist/8:.2f} u/step)")
    print("=== grunt speed (reset, track ONE grunt by address, 8 idle steps) ===")
    pkt = b.reset(0)
    # pick a grunt (list 3) by address
    tgt = None
    for addr, lid, sw, x, y in iter_entities(pkt):
        if lid == 3:
            tgt = (addr, x, y); break
    if tgt:
        for _ in range(8):
            pkt, _ = b.step(0, 0)
        # find same addr
        now = None
        for addr, lid, sw, x, y in iter_entities(pkt):
            if addr == tgt[0]:
                now = (x, y); break
        if now:
            gdx, gdy = now[0] - tgt[1], now[1] - tgt[2]
            gd = abs(gdx) + abs(gdy)
            print(f"  grunt {hex(tgt[0])}: moved dx={gdx:+d} dy={gdy:+d} ({gd} u / 8 steps = {gd/8:.2f} u/step)")
        else:
            print("  grunt vanished (killed/unlinked)")
    b.close()
    print("SPEED DIAG DONE")


if __name__ == "__main__":
    main()
