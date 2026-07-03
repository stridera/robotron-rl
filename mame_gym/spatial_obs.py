"""spatial_obs.py — absolute spatial-grid observation for a CNN policy.

Renders a MAME packet into a multi-channel 2D grid (channels = entity-type
threat groups + player + aggregate enemy velocity). Unlike the 945-dim
distance-ranked slot obs — which shows only the nearest-N per category and
obscures the global threat layout — the grid preserves field geometry so a CNN
can reason about WHERE all threats are and which open region is safe. That is
the capability the slot+MLP setup lacked: continuous play walled at wave ~3,
dying in the OPEN field (63% of deaths) to converging grunts, not at walls.

Prototype for the architecture pivot. Reuses MameObsBuilder for all the entity
classification (Prog disambiguation, Quark variants) and identity-stable
velocity, then bins sprites into the grid.

Grid: absolute playfield, rows=y (36), cols=x (24) — matches the ~1.5 field
aspect (pixel field 665w x 492h). Channels-first (C, H, W) for SB3 NatureCNN.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mame_obs import MameObsBuilder  # noqa: E402

PIX_W, PIX_H = 665.0, 492.0
GRID_W, GRID_H = 48, 72   # finer grid (was 24x36): ~14x7px cells (~1 sprite each)
                          # so the CNN can resolve individual enemy positions for
                          # dodging — tests whether the 24x36 grid's underperformance
                          # vs slot was a resolution limit. Fresh run required.

# sprite name -> presence channel. Grouped by threat class.
_CH = {
    "Player": 0,
    "Grunt": 1,
    "Hulk": 2,                                  # invincible — avoid, don't waste shots
    "Brain": 3, "Prog": 3,                      # brain-wave threats
    "Enforcer": 4, "Sphereoid": 4,              # shooters / spawners
    "EnforcerBullet": 5,                        # fast projectiles
    "Electrode": 6,                             # static hazards
    "Tank": 7, "CruiseMissile": 7, "Quark": 7, "TankShell": 7,
    "Mommy": 8, "Daddy": 8, "Mikey": 8,         # rescue targets (the point source)
}
NUM_PRESENCE = 9
# + 2 aggregate enemy-velocity channels (vx, vy summed per cell, normalized)
NUM_CHANNELS = NUM_PRESENCE + 2
VEL_NORM = 20.0   # ~max per-step pixel displacement


class SpatialGridObsBuilder:
    def __init__(self):
        self._src = MameObsBuilder()

    def reset(self):
        self._src.reset()

    def __call__(self, packet: bytes) -> np.ndarray:
        grid = np.zeros((NUM_CHANNELS, GRID_H, GRID_W), dtype=np.float32)
        for sp in self._src._sprites_from_packet(packet):
            px, py, name = sp[0], sp[1], sp[2]
            ch = _CH.get(name)
            if ch is None:
                continue
            col = min(GRID_W - 1, max(0, int(px / PIX_W * GRID_W)))
            row = min(GRID_H - 1, max(0, int(py / PIX_H * GRID_H)))
            grid[ch, row, col] += 1.0
            # Aggregate enemy velocity (not player, not family) into vel channels.
            if len(sp) >= 5 and ch not in (0, 8):
                grid[NUM_PRESENCE, row, col] += np.clip(sp[3] / VEL_NORM, -1, 1)
                grid[NUM_PRESENCE + 1, row, col] += np.clip(sp[4] / VEL_NORM, -1, 1)
        return grid


def _render(grid: np.ndarray) -> str:
    """ASCII overlay of player (@) + threats (#) + family (F) for validation."""
    rows = []
    for r in range(GRID_H):
        line = []
        for c in range(GRID_W):
            if grid[0, r, c] > 0:
                ch = "@"
            elif grid[8, r, c] > 0:
                ch = "F"
            elif grid[1:8, r, c].sum() > 0:
                ch = "#"
            else:
                ch = "."
            line.append(ch)
        rows.append("".join(line))
    return "\n".join(rows)


if __name__ == "__main__":
    from mame_bridge import MameBridge
    from mame_obs import parse_header
    b = MameBridge(port=9966, frameskip=4, boot_timeout=90)
    bld = SpatialGridObsBuilder(); bld.reset()
    pkt = b.reset(0)
    for _ in range(20):
        pkt = b.step(3, 3)[0]
    grid = bld(pkt)
    h = parse_header(pkt)
    print(f"grid shape: {grid.shape}  finite: {np.all(np.isfinite(grid))}")
    pop = {i: int(grid[i].sum()) for i in range(NUM_PRESENCE) if grid[i].sum() > 0}
    print(f"populated presence channels (idx->count): {pop}")
    print(f"player header pos gx={h['player_x']} gy={h['player_y']} wave={h['wave']}")
    print(f"velocity channel abs-sum: {float(np.abs(grid[NUM_PRESENCE:]).sum()):.1f}")
    print(_render(grid))
    b.close()
