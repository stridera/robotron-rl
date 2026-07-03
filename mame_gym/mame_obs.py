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
    0x1DD6: "Brain", 0x2119: "Cruise Missile", 0x1483: "Enforcer",
    0x14DC: "EnfBullet/Spark", 0x12C8: "Spheroid", 0x4BC9: "Quark",
    # SW = collision-handler address (object $08). Verified against robomame.asm
    # (see ENEMY_MODEL.md §2), correcting a 2026-06 mislabel: $4DF2 is the TANK
    # collision handler (asm:7388,7427) and $4FD5 is SHELL_COLLISION_HANDLER
    # (asm:7863) — both were wrongly mapped to "Quark". Only $4BC9 is a Quark
    # (QUARK_COLLISION_HANDLER asm:7252). $4800 ("Tank") was a phantom — nothing
    # loads #$4800 in the ROM. This mislabel manufactured the "quarks teleport"
    # phantom: fast wall-bouncing tank shells read as teleporting quarks.
    0x4DF2: "Tank", 0x4FD5: "TankShell",
    # $2119 = CRUISE_MISSILE_COLLISION_HANDLER (asm:3175, anim $206B) — NOT a brain
    # variant; $1F1F = brain-mutated Prog (human-range anim), NOT a cruise missile.
    # The only real brain is $1DD6. Validated via anim-ptr oracle 2026-07-01.
    0x1F1F: "Prog", 0x0330: "Mikey", 0x0335: "Mom",
    0x033A: "Dad", 0x7861: "PlayerBullet",
}


def classify_sw(sw: int) -> str | None:
    return ENTITY_TYPES.get(sw)


# entity name -> robotron-rl SPRITE_TYPES name (PlayerIcon / PlayerBullet dropped).
_TYPE_MAP = {
    "Grunt": "Grunt", "Electrode": "Electrode", "Hulk": "Hulk", "Brain": "Brain",
    "Enforcer": "Enforcer", "EnfBullet/Spark": "EnforcerBullet", "Prog": "Prog",
    "Spheroid": "Sphereoid", "Quark": "Quark", "Tank": "Tank", "TankShell": "TankShell",
    "Cruise Missile": "CruiseMissile", "Mikey": "Mikey", "Mom": "Mommy", "Dad": "Daddy",
}

_PLAY_PIXEL_W, _PLAY_PIXEL_H = 665.0, 492.0
_GX_MIN, _GX_RANGE = 5, 140
_GY_MIN, _GY_RANGE = 15, 215

SLOT_BASE_OFFSET = 11   # slot pool starts after the 11-byte header (byte10 = score millions)
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
# Max believable per-step velocity in PIXEL space (frameskip 4). The fastest real
# entity (shell/spark, ~24 game-u/step) is ~114px in x / ~55px in y; 130 spares
# that but zeroes node-recycling artifacts (300px+). See _sprites_from_packet.
_VMAX_PX = 130.0


# Typed-entity array (appended to the packet by robotron_server.lua's
# list walk): [n] then n * 7-byte records [addr_hi, addr_lo, list_id,
# sw_hi, sw_lo, x, y]. Categories are authoritative — the game's own
# per-category linked lists; dead objects unlink, so every record is a
# LIVE entity (no corpse/effect records by construction).
ENTITY_ARRAY_OFFSET = SLOT_BASE_OFFSET + SLOT_COUNT * SLOT_STRIDE  # after pool
LIST_SPHEROID_GRP, LIST_FAMILY, LIST_GHBPCT, LIST_ELECTRODE = 1, 2, 3, 4

# Species within each mixed list, by canonical SW (collision-handler address at
# object $08). ROM-verified in ENEMY_MODEL.md §2. List 1 = $9817 residents:
# Quark is ONLY $4BC9; $4FD5 is the tank shell (SHELL_COLLISION_HANDLER). The
# fallback stays "TankShell" for any unverified $9817 resident.
_LIST1_SW = {0x12C8: "Sphereoid", 0x1483: "Enforcer", 0x14DC: "EnforcerBullet",
             0x4BC9: "Quark", 0x4FD5: "TankShell"}
_LIST2_SW = {0x0330: "Mikey", 0x0335: "Mommy", 0x033A: "Daddy"}
# List 3 = $9821. Tank collision handler is $4DF2 (asm:7388,7427); the old
# $4800 was a phantom, so tanks were invisible to the policy/forensics.
# $2119 = the real CruiseMissile (asm:3175); $1F1F = Prog (brain-mutated human).
# $00B6 stays Hulk-or-standalone-Prog (disambiguated by motion below).
_LIST3_SW = {0x3A76: "Grunt", 0x00B6: "Hulk", 0x1DD6: "Brain", 0x2119: "CruiseMissile",
             0x4DF2: "Tank", 0x1F1F: "Prog"}


def iter_entities(packet: bytes):
    """Yield (addr, list_id, sw, x, y) from the typed entity array."""
    n = packet[ENTITY_ARRAY_OFFSET]
    base = ENTITY_ARRAY_OFFSET + 1
    for i in range(n):
        off = base + i * 7
        addr = (packet[off] << 8) | packet[off + 1]
        sw = (packet[off + 3] << 8) | packet[off + 4]
        yield addr, packet[off + 2], sw, packet[off + 5], packet[off + 6]


# ---------------------------------------------------------------------------
# Exact-forward-model state (Stage 3). With MAME_EMIT_SIM=1 the server appends,
# after the entity records (and the anim block if MAME_EMIT_ANIM=1):
#   n*8 bytes  — per entity: $0A/$0B X whole.frac, $0C/$0D Y whole.frac,
#                $0E/$0F X-vel 8.8 signed, $10/$11 Y-vel 8.8 signed
#   n*2 bytes  — per entity: $12/$13 AI/move countdown fields
#   27 bytes   — globals: $BE5C..$BE67 (12 difficulty vars), $9884-86 (RNG),
#                $BE68..$BE71 (10 live-enemy counts), $98F0/$98F1 (spark/shell
#                live counters — the shell-budget exploit state)
# ---------------------------------------------------------------------------
_EMIT_ANIM_ENV = os.environ.get("MAME_EMIT_ANIM") == "1"
SIM_GLOBAL_NAMES = (
    "grunt_move", "grunt_move_floor", "drop_count", "enforcer_fire",
    "enforcer_spawn", "hulk_speed", "brain_fire", "brain_speed",
    "tank_fire", "shell_accuracy", "tank_spawn", "quark_move",
    "rng0", "rng1", "rng2",
    # $BE68..$BE71 per COUNT_ENEMIES_ON_SCREEN (asm:3834): cur_grunts=$BE68,
    # cur_brains=$BE6E, cur_sphereoids=$BE6F, cur_quarks=$BE70, cur_tanks=$BE71.
    # Probe-validated on wave 1 (15 grunts / 5 electrodes / 1 mom / 1 dad).
    "cnt_grunts", "cnt_electrodes", "cnt_moms", "cnt_dads", "cnt_mikeys",
    "cnt_hulks", "cnt_brains", "cnt_spheroids", "cnt_quarks", "cnt_tanks",
    # $98F0 is a score-update counter (asm:121), NOT a spark counter; $98F1 is
    # the shell live-count with the no-decrement-on-expiry budget bug.
    "score_update_ctr", "shell_live",
)


def _s16_88(hi: int, lo: int) -> float:
    """Signed 8.8 fixed-point to float."""
    v = (hi << 8) | lo
    if v >= 0x8000:
        v -= 0x10000
    return v / 256.0


def parse_sim_state(packet: bytes, emit_anim: bool | None = None):
    """Parse the MAME_EMIT_SIM trailing blocks.

    Returns (entities, globals): entities = list of dicts with addr/list/sw and
    exact 8.8 position/velocity + the two per-object countdown fields; globals =
    dict keyed by SIM_GLOBAL_NAMES. Requires the packet to have been captured
    with MAME_EMIT_SIM=1 (and pass emit_anim=True if MAME_EMIT_ANIM was also on).
    """
    if emit_anim is None:
        emit_anim = _EMIT_ANIM_ENV
    n = packet[ENTITY_ARRAY_OFFSET]
    base = ENTITY_ARRAY_OFFSET + 1
    sim_off = base + n * 7 + (n * 2 if emit_anim else 0)
    tmr_off = sim_off + n * 8
    glb_off = tmr_off + n * 2
    if len(packet) < glb_off + 27:
        raise ValueError("packet lacks MAME_EMIT_SIM blocks (server not started with it?)")
    entities = []
    for i in range(n):
        off = base + i * 7
        s = sim_off + i * 8
        t = tmr_off + i * 2
        entities.append({
            "addr": (packet[off] << 8) | packet[off + 1],
            "list": packet[off + 2],
            "sw": (packet[off + 3] << 8) | packet[off + 4],
            "x": packet[s] + packet[s + 1] / 256.0,
            "y": packet[s + 2] + packet[s + 3] / 256.0,
            "vx": _s16_88(packet[s + 4], packet[s + 5]),
            "vy": _s16_88(packet[s + 6], packet[s + 7]),
            "t12": packet[t],
            "t13": packet[t + 1],
        })
    g = packet[glb_off:glb_off + 27]
    return entities, dict(zip(SIM_GLOBAL_NAMES, g))


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
                # Only $00B6 is still ambiguous (Hulk vs standalone Prog). $1F1F
                # is unambiguously Prog and $2119 unambiguously CruiseMissile now.
                if sw == 0x00B6:
                    new_prev[addr] = (sw, x, y)
                    rl_name = self._disambiguate(addr, sw, x, y, rl_name)
            else:
                rl_name = None
            if rl_name is None:
                continue
            spx, spy = _gxgy_to_pixel(x, y)
            # Identity-stable velocity in pixel space. Keyed by (node addr, SW):
            # the 6809 allocator recycles a freed slot's address immediately with
            # no generation counter, so keying on addr alone aliases a dead entity
            # with a fresh one at the same address (validated: ~5% of deaths showed
            # 66-113 u/step "jumps"). (addr, sw) means a recycled slot with a new
            # type starts fresh (v=0); the _VMAX guard catches same-SW recycling.
            prev = self._entity_prev_pos.get((addr, sw))
            vx = spx - prev[0] if prev is not None else 0.0
            vy = spy - prev[1] if prev is not None else 0.0
            if max(abs(vx), abs(vy)) > _VMAX_PX:   # beyond any real 1-step move
                vx = vy = 0.0
            new_pos[(addr, sw)] = (spx, spy)
            sprites.append((spx, spy, rl_name, vx, vy))
        self._node_prev = new_prev
        self._entity_prev_pos = new_pos
        return sprites


def parse_header(packet: bytes) -> dict:
    # byte layout (robotron_server.lua): [wave($BDED), lives($BDEC),
    # sc5-7($BDE5-7 BCD), pX($9864), $9865(unused), pY($9866),
    # cur_player($983F), game_state($9859)]. Audited vs the annotated ASM
    # 2026-06-13: byte 8 was mislabeled "dir" (it's current_player, unused) and
    # byte 9 was mislabeled "dead" (was $9848 collision-flag; now $9859
    # game_state — 0=play, 0x1B=KILL_PLAYER death, 0x7F/0x19=transition).
    wave, lives, s5, s6, s7, px, pxsub, py, cur_player, game_state, s4 = packet[:11]
    # byte 10 = $BDE4, the MILLIONS byte: p1_score is 4 BCD bytes $BDE4-7
    # (asm:131). Added 2026-07-01 after a wave-48 game "scored 408k" — it had
    # actually scored 1,408,775 and the 3-byte read wrapped at 1M.
    score = ((s4 >> 4) * 10 + (s4 & 0xF)) * 1000000 + \
            ((s5 >> 4) * 10 + (s5 & 0xF)) * 10000 + \
            ((s6 >> 4) * 10 + (s6 & 0xF)) * 100 + \
            ((s7 >> 4) * 10 + (s7 & 0xF))
    return {"wave": wave, "lives": lives, "score": score,
            "player_x": px, "player_y": py, "game_state": game_state}
