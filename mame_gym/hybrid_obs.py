"""hybrid_obs.py — 945-dim slot obs + compact GLOBAL threat summary.

The slot+MLP policy walls at mean wave ~8 (yiwzpqq7) — 63% of deaths are in the
OPEN field to converging grunts (spatial_obs.py forensics). Root cause: the 945-dim
slot obs shows only the nearest-N entities per category, hiding the GLOBAL threat
layout, so the policy can't tell which open region is safe. A full spatial-grid CNN
obs preserves layout but is far too sample-inefficient to train (grid runs 9iwa3zzv:
mean ~3 even after BC; the slot obs is the FSM's native nearest-N representation so
slot BC trivially hits ~8, the grid forces a CNN to re-derive it). See
project_goal_wave100 + EXPERIMENT_STATE 2026-06-19.

HYBRID = keep the proven 945-dim slot obs (strong BC ~8) + APPEND 12 cheap global
features so an MLP can still learn easily:
  - 8 directional threat-pressure features (one per 45° sector around the player):
    sum of clamped 1/distance over all threats in that sector -> "how much danger
    lies in each direction" (aligned with the 8 movement dirs).
  - 4 wall-openness features: normalized distance to each playfield edge -> how much
    room to retreat in each cardinal direction.
BC reproduces the FSM (~8) using mostly the slot part (the FSM ignores these extras),
then RL can exploit the global features to flee toward open, low-pressure regions and
push PAST 8. (957-dim total.)

Reuses MameObsBuilder for all entity classification + identity-stable velocity.
"""
from __future__ import annotations
import sys
import math
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mame_obs import MameObsBuilder  # noqa: E402

PIX_W, PIX_H = 665.0, 492.0
N_SECTORS = 8
N_GLOBAL = N_SECTORS + 4          # 8 directional pressures + 4 wall distances
HYBRID_DIM = 945 + N_GLOBAL       # 957
_PRESSURE_NORM = 4.0              # ~saturation for summed clamped inverse-distance
_DIST_CLAMP = 60.0               # px; threats closer than this contribute ~1.0

# Non-threat sprite names (Player handled separately; family are rescue targets).
_NONTHREAT = {"Player", "Mommy", "Daddy", "Mikey"}


class HybridObsBuilder:
    def __init__(self):
        self._src = MameObsBuilder()

    def reset(self):
        self._src.reset()

    def __call__(self, packet: bytes) -> np.ndarray:
        sprites = self._src._sprites_from_packet(packet)
        base = self._src._extractor._extract_features(sprites).astype(np.float32)
        g = np.zeros(N_GLOBAL, dtype=np.float32)
        px, py = sprites[0][0], sprites[0][1]
        for s in sprites[1:]:
            if s[2] in _NONTHREAT:
                continue
            dx, dy = s[0] - px, s[1] - py
            dist = math.hypot(dx, dy)
            if dist < 1.0:
                dist = 1.0
            # sector by angle (atan2), 8 bins of 45 deg
            ang = math.atan2(dy, dx)            # [-pi, pi]
            sec = int((ang + math.pi) / (2 * math.pi) * N_SECTORS) % N_SECTORS
            g[sec] += min(1.0, _DIST_CLAMP / dist)
        g[:N_SECTORS] = np.clip(g[:N_SECTORS] / _PRESSURE_NORM, 0.0, 1.0)
        # wall openness: room to the left/right/bottom/top, normalized
        g[N_SECTORS + 0] = px / PIX_W
        g[N_SECTORS + 1] = (PIX_W - px) / PIX_W
        g[N_SECTORS + 2] = py / PIX_H
        g[N_SECTORS + 3] = (PIX_H - py) / PIX_H
        return np.concatenate([base, g])


if __name__ == "__main__":
    from mame_bridge import MameBridge
    from mame_obs import parse_header
    b = MameBridge(port=9966, frameskip=4, boot_timeout=90)
    bld = HybridObsBuilder(); bld.reset()
    pkt = b.reset(0)
    for _ in range(20):
        pkt = b.step(3, 3)[0]
    obs = bld(pkt); h = parse_header(pkt)
    print(f"hybrid obs shape: {obs.shape} (expect {HYBRID_DIM})  finite: {np.all(np.isfinite(obs))}")
    print(f"slot[0:5]={obs[:5]}")
    print(f"global (8 sector pressures + 4 wall): {obs[945:].round(3)}")
    print(f"player px,py: {parse_header(pkt)['player_x']},{parse_header(pkt)['player_y']}  wave={h['wave']}")
    b.close()
