"""mame_obs.py — build the 945-dim policy observation from a MAME obs packet.

The MAME server (robotron_server.lua) sends each step:
  10-byte header: [wave, lives, sc5, sc6, sc7, pX, pXsub, pY, dir, dead]
  2424-byte slot pool: 101 slots * 24 bytes starting at 6809 $98D4.

This decodes that into the (pixel_x, pixel_y, sprite_type) sprite list and runs
robotron-rl's GroundTruthPositionWrapper._extract_features to emit the same
945-dim observation the existing PPO policies were trained on — identical to
the native gym's PositionObsBuilder, just sourced from MAME bytes.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np

ROBOTRON_RL_ROOT = Path(os.environ.get("ROBOTRON_RL_ROOT", str(Path.home() / "Code" / "robotron-rl")))
if str(ROBOTRON_RL_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOTRON_RL_ROOT))

from position_wrapper import GroundTruthPositionWrapper  # noqa: E402

# 6809 state_word -> canonical entity name. EXACT-MATCH ONLY, authoritative
# per the ROM reverse-engineering in dumps/ENEMY_BEHAVIOR_ANALYSIS.md
# (2026-06-10 ground-truth pass after the Prog-on-wave-2 contradiction):
#
#   - A state_word is the entity's DEATH-HANDLER CODE ADDRESS. It is constant
#     for the species' entire life — live entities never change SW. All other
#     SWs in the pool ($3A45-$3A86, $1242-$125B, $00xx, $0390, ...) are
#     death-explosion / effect records and MUST NOT be classified as live
#     entities (an earlier range-based classifier here turned corpses into
#     phantom enemies).
#   - Progs have NO unique SW. Brain-mutated civilians show $1F1F in the slot
#     pool (same as Cruise Missile); standalone wave-data Progs use $00B6
#     (same as Hulk). Both alias to the same threat class and are therefore
#     VISIBLE to the policy as CruiseMissile/Hulk. Fine for play; revisit only
#     if Prog-specific behavior modelling is ever needed.
ENTITY_TYPES = {
    0xAADD: "PlayerIcon", 0x3A76: "Grunt", 0x3AA9: "Electrode", 0x00B6: "Hulk",
    0x1DD6: "Brain", 0x2119: "Brain (alt)", 0x1483: "Enforcer",
    0x14DC: "EnfBullet/Spark", 0x12C8: "Spheroid", 0x4BC9: "Quark",
    # Quark SW variants (live-confirmed: 1,077 deaths at waves 7/12 implicated
    # them via forensics, 2026-06-12). Already in _LIST1_SW for the obs path;
    # needed here too so kill counters award the spawner bonus and forensics
    # stop flagging them "explained-unmapped".
    0x4DF2: "Quark", 0x4FD5: "Quark",
    0x4800: "Tank", 0x1F1F: "Cruise Missile", 0x0330: "Mikey", 0x0335: "Mom",
    0x033A: "Dad", 0x7861: "PlayerBullet",
}


def classify_sw(sw: int) -> str | None:
    return ENTITY_TYPES.get(sw)


# entity name -> robotron-rl SPRITE_TYPES name (PlayerIcon / PlayerBullet dropped).
_TYPE_MAP = {
    "Grunt": "Grunt", "Electrode": "Electrode", "Hulk": "Hulk", "Brain": "Brain",
    "Brain (alt)": "Brain", "Enforcer": "Enforcer", "EnfBullet/Spark": "EnforcerBullet",
    "Spheroid": "Sphereoid", "Quark": "Quark", "Tank": "Tank",
    "Cruise Missile": "CruiseMissile", "Mikey": "Mikey", "Mom": "Mommy", "Dad": "Daddy",
}

_PLAY_PIXEL_W, _PLAY_PIXEL_H = 665.0, 492.0
_GX_MIN, _GX_RANGE = 5, 140
_GY_MIN, _GY_RANGE = 15, 215

SLOT_BASE_OFFSET = 10   # slot pool starts after the 10-byte header in the packet
SLOT_STRIDE = 24
SLOT_COUNT = 101


def _gxgy_to_pixel(gx: int, gy: int):
    return ((gx - _GX_MIN) / _GX_RANGE * _PLAY_PIXEL_W,
            (gy - _GY_MIN) / _GY_RANGE * _PLAY_PIXEL_H)


class _PlayRect:
    def __init__(self, w, h): self.width, self.height = w, h
class _EngineShim:    play_rect = _PlayRect(_PLAY_PIXEL_W, _PLAY_PIXEL_H)
class _UnwrappedShim: engine = _EngineShim()


def _build_feature_extractor() -> GroundTruthPositionWrapper:
    import gymnasium as gym
    class _Inner(gym.Env):
        observation_space = gym.spaces.Box(0, 255, (1, 1, 3), dtype=np.uint8)
        action_space = gym.spaces.MultiDiscrete([8, 8])
        unwrapped = _UnwrappedShim()
    return GroundTruthPositionWrapper(_Inner(), verbose=False)


# Prog disambiguation (movement signature, per ENEMY_BEHAVIOR_ANALYSIS.md):
#   $00B6 = Hulk OR standalone Prog. Hulk lumbers ~0.5-1 u/frame; Prog runs
#   3.5 u/frame horizontal. At frameskip 4: ~3 u/step vs ~14 u/step.
#   $1F1F = Cruise Missile OR brain-mutated Prog. Missile: X=2,Y=4 per
#   3 frames (Y-dominant, ~2.7/5.3 u/step); Prog: cardinal X-dominant fast.
# Behavioral stakes (user-flagged): Hulks are INVINCIBLE (avoid, never shoot
# to kill); Progs are shootable hunters (kill on sight). Conflating them
# inverts correct play.
_PROG_HULK_SPEED = 8.0      # u/step: above ⇒ Prog, below ⇒ Hulk
_PROG_MISSILE_DX = 8.0      # u/step horizontal: above (and X-dominant) ⇒ Prog


# Typed-entity array (appended to the packet by robotron_server.lua's
# list walk): [n] then n * 7-byte records [addr_hi, addr_lo, list_id,
# sw_hi, sw_lo, x, y]. Categories are authoritative — the game's own
# per-category linked lists; dead objects unlink, so every record is a
# LIVE entity (no corpse/effect records by construction).
ENTITY_ARRAY_OFFSET = SLOT_BASE_OFFSET + SLOT_COUNT * SLOT_STRIDE  # after pool
LIST_SPHEROID_GRP, LIST_FAMILY, LIST_GHBPCT, LIST_ELECTRODE = 1, 2, 3, 4

# Species within each mixed list, by canonical SW.
# Quark has three SW variants (legacy XBLA table, corroborated live: $4FD5
# implicated in wave-7 deaths via forensics 2026-06-10).
_LIST1_SW = {0x12C8: "Sphereoid", 0x1483: "Enforcer", 0x14DC: "EnforcerBullet",
             0x4BC9: "Quark", 0x4DF2: "Quark", 0x4FD5: "Quark"}
_LIST2_SW = {0x0330: "Mikey", 0x0335: "Mommy", 0x033A: "Daddy"}
_LIST3_SW = {0x3A76: "Grunt", 0x00B6: "Hulk", 0x1DD6: "Brain", 0x2119: "Brain",
             0x4800: "Tank", 0x1F1F: "CruiseMissile"}


def iter_entities(packet: bytes):
    """Yield (addr, list_id, sw, x, y) from the typed entity array."""
    n = packet[ENTITY_ARRAY_OFFSET]
    base = ENTITY_ARRAY_OFFSET + 1
    for i in range(n):
        off = base + i * 7
        addr = (packet[off] << 8) | packet[off + 1]
        sw = (packet[off + 3] << 8) | packet[off + 4]
        yield addr, packet[off + 2], sw, packet[off + 5], packet[off + 6]


class MameObsBuilder:
    """Build the 945-dim policy observation from the typed entity array."""

    def __init__(self):
        self._extractor = _build_feature_extractor()
        # node-address -> (sw, x, y) from the previous packet; node address is
        # a stable identity for the entity's lifetime (velocity tracking).
        self._node_prev: dict[int, tuple[int, int, int]] = {}
        # node-address -> (pixel_x, pixel_y) previous frame, for identity-stable
        # velocity (the extractor's slot-rank velocity is garbage; see
        # position_wrapper). Keyed by the same stable node address.
        self._entity_prev_pos: dict[int, tuple[float, float]] = {}

    def reset(self):
        self._extractor._prev_positions.clear()
        self._node_prev.clear()
        self._entity_prev_pos.clear()

    def __call__(self, packet: bytes) -> np.ndarray:
        return self._extractor._extract_features(self._sprites_from_packet(packet))

    def _disambiguate(self, addr: int, sw: int, x: int, y: int, name: str) -> str:
        """$00B6 = Hulk or standalone Prog; $1F1F = CruiseMissile or mutated
        Prog (same death handlers). Split by movement signature
        (ENEMY_BEHAVIOR_ANALYSIS.md): Prog ~14 u/step at frameskip 4,
        X-dominant cardinal; Hulk ~3 u/step; missile Y-dominant."""
        prev = self._node_prev.get(addr)
        if prev is None or prev[0] != sw:
            return name   # no history yet: conservative default (Hulk/missile)
        dx, dy = abs(x - prev[1]), abs(y - prev[2])
        # Teleport guard: respawn repositioning jumps entities arbitrarily far
        # in one step. Prog tops out ~16-20 u/step; anything beyond is a
        # teleport, not locomotion (caught 2026-06-10: respawning Hulks were
        # mislabeled Prog, visible as empty boxes in respawn-frame overlays).
        if max(dx, dy) > 20:
            return name
        if sw == 0x00B6 and max(dx, dy) > _PROG_HULK_SPEED:
            return "Prog"
        if sw == 0x1F1F and dx > _PROG_MISSILE_DX and dx > dy:
            return "Prog"
        return name

    def _sprites_from_packet(self, packet: bytes):
        sprites = []
        # header: [wave, lives, sc5, sc6, sc7, pX, pXsub, pY, dir, dead]
        px, py = _gxgy_to_pixel(packet[5], packet[7])
        sprites.append((px, py, "Player"))
        new_prev: dict[int, tuple[int, int, int]] = {}
        new_pos: dict[int, tuple[float, float]] = {}
        for addr, list_id, sw, x, y in iter_entities(packet):
            if list_id == LIST_ELECTRODE:
                rl_name = "Electrode"
            elif list_id == LIST_FAMILY:
                rl_name = _LIST2_SW.get(sw)
            elif list_id == LIST_SPHEROID_GRP:
                # Unknown SW on this list can only be a TankShell (the one
                # $9817 resident without a verified SW) or a same-category
                # variant — projectile threat class either way.
                rl_name = _LIST1_SW.get(sw, "TankShell")
            elif list_id == LIST_GHBPCT:
                rl_name = _LIST3_SW.get(sw)
                if sw in (0x00B6, 0x1F1F):
                    new_prev[addr] = (sw, x, y)
                    rl_name = self._disambiguate(addr, sw, x, y, rl_name)
            else:
                rl_name = None
            if rl_name is None:
                continue
            spx, spy = _gxgy_to_pixel(x, y)
            # Identity-stable velocity in pixel space, by node address.
            prev = self._entity_prev_pos.get(addr)
            vx = spx - prev[0] if prev is not None else 0.0
            vy = spy - prev[1] if prev is not None else 0.0
            new_pos[addr] = (spx, spy)
            sprites.append((spx, spy, rl_name, vx, vy))
        self._node_prev = new_prev
        self._entity_prev_pos = new_pos
        return sprites


def parse_header(packet: bytes) -> dict:
    wave, lives, s5, s6, s7, px, pxsub, py, pdir, dead = packet[:10]
    score = ((s5 >> 4) * 10 + (s5 & 0xF)) * 10000 + \
            ((s6 >> 4) * 10 + (s6 & 0xF)) * 100 + \
            ((s7 >> 4) * 10 + (s7 & 0xF))
    return {"wave": wave, "lives": lives, "score": score,
            "player_x": px, "player_y": py, "dir": pdir, "dead": dead}
