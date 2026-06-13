"""mame_robotron_env.py — Gymnasium env over a headless MAME Robotron instance.

Drop-in replacement for train_native.py's NativeRobotronEnv, but backed by MAME
(faithful Williams emulation, all waves, full-machine save-state reset). Same
945-dim obs, same MultiDiscrete([8,8]) action surface, same reward shaping — so
the existing PPO recipe and policies transfer.

No corruption guard: MAME doesn't have the native gym's wave-transition bug.

Death forensics: every life loss is logged to MAME_DEATH_LOG_DIR (if set) as a
JSONL record with all entities near the player at the death frame and the one
before it, classified mapped/unmapped. Verdicts: 'explained' (known hostile in
kill range), 'explained-unmapped' (unmapped SW in kill range = missing sprite),
'unexplained' (nothing in range = missing condition). See death_audit.py.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box

sys.path.insert(0, str(Path(__file__).parent))
from mame_bridge import MameBridge
from mame_obs import MameObsBuilder, parse_header

# Kill-bonus counting uses the census-derived classifier so animation-frame
# SW changes don't read as kills (exact-match counters awarded phantom kill
# bonuses every time a spheroid/enforcer animated to a variant SW).
from mame_obs import classify_sw

_SPAWNER_NAMES = ("Spheroid", "Quark")
_SHOOTER_NAMES = ("Enforcer", "Tank")
_BRAIN_NAMES   = ("Brain", "Brain (alt)")
_FAMILY_NAMES  = ("Mikey", "Mom", "Dad")   # rescuable humans (the point source)
_SLOT_BASE, _SLOT_STRIDE, _SLOT_COUNT = 10, 24, 101


def _count_strategic(packet: bytes):
    spawner = shooter = brain = family = 0
    for i in range(_SLOT_COUNT):
        off = _SLOT_BASE + i * _SLOT_STRIDE
        sw = (packet[off + 4] << 8) | packet[off + 5]
        if sw == 0:
            continue
        name = classify_sw(sw)
        if name in _SPAWNER_NAMES: spawner += 1
        elif name in _SHOOTER_NAMES: shooter += 1
        elif name in _BRAIN_NAMES: brain += 1
        elif name in _FAMILY_NAMES: family += 1
    return spawner, shooter, brain, family


# ── Death forensics ────────────────────────────────────────────────────────
_KILL_RADIUS = 15      # direct-contact range
_CLOSING_RADIUS = 28   # reachable within one frameskip-4 step of mutual approach
_CONTEXT_RADIUS = 40   # log range (enforcer sparks cross ~16u/step)
# Note: Progs alias to "Hulk" ($00B6) / "Cruise Missile" ($1F1F) — covered.
_LETHAL = {"Grunt", "Electrode", "Hulk", "Sphereoid", "Quark", "Brain",
           "Brain (alt)", "Enforcer", "Tank", "Cruise Missile",
           "EnfBullet/Spark"}
_HARMLESS = {"Mom", "Dad", "Mikey", "PlayerIcon", "PlayerBullet"}


def _entities_near(packet: bytes, px: int, py: int, radius: int):
    """All slot entities within `radius` (game units, chebyshev) of (px,py)."""
    out = []
    for i in range(_SLOT_COUNT):
        off = _SLOT_BASE + i * _SLOT_STRIDE
        sw = (packet[off + 4] << 8) | packet[off + 5]
        if sw == 0:
            continue
        ex, ey = packet[off], packet[off + 1]
        d = max(abs(ex - px), abs(ey - py))
        if d <= radius:
            out.append({"sw": f"0x{sw:04X}", "name": classify_sw(sw),
                        "x": ex, "y": ey, "d": d})
    return sorted(out, key=lambda e: e["d"])


def _death_verdict(near):
    """Classify a death from the nearby-entity list (kill-radius subset)."""
    killers = [e for e in near if e["d"] <= _KILL_RADIUS]
    if any(e["name"] in _LETHAL for e in killers):
        return "explained"
    if any(e["name"] is None for e in killers):
        return "explained-unmapped"
    return "unexplained"


class MameRobotronEnv(gym.Env):
    metadata = {"render_modes": []}

    # Auto-capture states land at AUTO_CAPTURE_BASE + rank*8 + (wave - min),
    # well above hand-built ladder indices. They're harvested into the NEXT
    # link's reset pool (never the current one — avoids load/save races).
    # Override per link (MAME_AUTO_CAPTURE_BASE) so a new link's harvest never
    # overwrites the previous link's states (idx layout shifts with min_wave).
    AUTO_CAPTURE_BASE = int(os.environ.get("MAME_AUTO_CAPTURE_BASE", "100"))
    AUTO_CAPTURE_SETTLE = 30   # steps (120 frames) past wave entry

    def __init__(self, rank: int = 0, base_port: int = 9200, frameskip: int = 4,
                 reset_pool: list[int] | None = None,
                 auto_capture_min_wave: int | None = None,
                 obs_mode: str = "slot"):
        """reset_pool: list of save-state indices to sample episode starts
        from. 0 = wave-1 boot state; N>0 = 'w5_N' deep-wave state. Default
        [0] = always start at wave 1.

        auto_capture_min_wave: if set, save a full-machine state whenever an
        episode enters a wave >= this (settled past the transition). Training
        envs reach frontier waves orders of magnitude more often than the
        single-env capture script (which got 0 wave-13 entries in 400
        episodes), so this turns those moments into next-link reset seeds.

        obs_mode: 'slot' = 945-dim distance-ranked category-slot obs (MLP);
        'grid' = (11,36,24) spatial grid for a CNN policy (preserves global
        field geometry — the slot+MLP setup walled continuous play at wave ~3)."""
        super().__init__()
        self.action_space = MultiDiscrete([8, 8])
        self._obs_mode = obs_mode
        if obs_mode == "grid":
            from spatial_obs import SpatialGridObsBuilder, NUM_CHANNELS, GRID_H, GRID_W
            self.observation_space = Box(low=-np.inf, high=np.inf,
                                         shape=(NUM_CHANNELS, GRID_H, GRID_W), dtype=np.float32)
        else:
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(945,), dtype=np.float32)
        self._port = base_port + rank
        self._frameskip = frameskip
        self._reset_pool = list(reset_pool) if reset_pool else [0]
        self._rng = np.random.default_rng(1000 + rank)
        self._rank = rank
        self._auto_min_wave = auto_capture_min_wave
        self._auto_settle = -1          # countdown to save; -1 = disarmed
        self._auto_saved_waves: set[int] = set()   # per-episode
        if obs_mode == "grid":
            from spatial_obs import SpatialGridObsBuilder
            self._obs_builder = SpatialGridObsBuilder()
        else:
            self._obs_builder = MameObsBuilder()
        self._bridge: MameBridge | None = None
        self._last_score = 0
        self._last_lives = 0
        self._last_wave = 0
        self._last_spawn = self._last_shoot = self._last_brain = self._last_family = 0
        self._last_packet: bytes | None = None   # raw obs packet (diagnostics)
        # Short packet history for death forensics. Mutual-destruction kills
        # (player walks into a grunt: BOTH die) remove the killer from the
        # slot pool at the death frame — so the killer is only visible in
        # the frames BEFORE death. Keep the last 3 packets (12 frames).
        self._pkt_history: list[bytes] = []
        # Death forensics log (JSONL per env rank); enabled via env var.
        log_dir = os.environ.get("MAME_DEATH_LOG_DIR", "")
        self._death_log = None
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self._death_log = open(Path(log_dir) / f"deaths_rank{rank}.jsonl", "a")

    def _log_death(self, cur_pkt, wave, score, terminal):
        """Forensic record for a life loss. Mutual-destruction aware: the
        killer may have died WITH the player, so suspects come from the
        packet history (last 3 steps), and any suspect whose slot vanished
        at the death frame is flagged 'vanished' (prime suspect)."""
        if self._death_log is None or not self._pkt_history:
            return
        # Live SWs at the death frame, keyed by slot index, to detect vanishing.
        cur_slots = {}
        for i in range(_SLOT_COUNT):
            off = _SLOT_BASE + i * _SLOT_STRIDE
            cur_slots[i] = (cur_pkt[off + 4] << 8) | cur_pkt[off + 5]
        # Suspects: entities near the player in ANY recent packet.
        suspects = {}
        for age, pkt in enumerate(reversed(self._pkt_history)):  # age 0 = latest pre-death
            ppx, ppy = pkt[5], pkt[7]
            for e in _entities_near(pkt, ppx, ppy, _CONTEXT_RADIUS):
                key = (e["sw"], e["x"] // 8, e["y"] // 8)   # coarse identity
                if key not in suspects or e["d"] < suspects[key]["d"]:
                    e2 = dict(e); e2["age"] = age
                    suspects[key] = e2
        # Vanished flag: a suspect SW that no longer appears anywhere in the pool.
        live_sws = set(cur_slots.values())
        for s in suspects.values():
            s["vanished"] = int(s["sw"], 16) not in live_sws
        near = sorted(suspects.values(), key=lambda e: (e["d"], e["age"]))
        rec = {
            "wave": wave, "score": score, "terminal": terminal,
            "player": [self._pkt_history[-1][5], self._pkt_history[-1][7]],
            "suspects": near[:12],
        }
        # Tiered verdict (calibrated on the 2026-06-10 audit, 215 deaths):
        #   direct: lethal within contact range
        #   closing: lethal within one step of mutual approach (frameskip 4
        #            ⇒ ~12u/step combined closing speed)
        #   debris: $00xx explosion record at the site — mutual destruction,
        #           the killer died with the player
        #   unexplained: residual; pattern-matches fast sparks crossing the
        #                sampling gap, not invisible enemies
        direct  = [e for e in near if e["d"] <= _KILL_RADIUS and e["name"] in _LETHAL]
        closing = [e for e in near if e["d"] <= _CLOSING_RADIUS and e["name"] in _LETHAL]
        debris  = [e for e in near if e["d"] <= _CLOSING_RADIUS and e["name"] is None
                   and int(e["sw"], 16) <= 0x00FF]
        unmapped = [e for e in near if e["d"] <= _CLOSING_RADIUS and e["name"] is None
                    and int(e["sw"], 16) > 0x00FF]
        if direct:
            rec["verdict"] = "explained-direct"
        elif closing:
            rec["verdict"] = "explained-closing"
        elif debris:
            rec["verdict"] = "explained-debris"
        elif unmapped:
            rec["verdict"] = "explained-unmapped"   # real missing-sprite signal
        else:
            rec["verdict"] = "unexplained"
        self._death_log.write(json.dumps(rec) + "\n")
        self._death_log.flush()

    def _ensure_bridge(self):
        if self._bridge is None:
            self._bridge = MameBridge(port=self._port, frameskip=self._frameskip)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._ensure_bridge()
        state_idx = int(self._rng.choice(self._reset_pool))
        packet = self._bridge.reset(state_idx)
        self._last_packet = packet
        self._pkt_history = [packet]
        self._obs_builder.reset()
        h = parse_header(packet)
        self._last_score = h["score"]
        self._last_lives = h["lives"]
        self._last_wave = h["wave"]
        self._last_spawn, self._last_shoot, self._last_brain, self._last_family = _count_strategic(packet)
        self._auto_settle = -1
        self._auto_saved_waves = set()
        obs = self._obs_builder(packet).astype(np.float32)
        return obs, {"score": h["score"], "lives": h["lives"], "wave": h["wave"]}

    def step(self, action):
        move, fire = int(action[0]) + 1, int(action[1]) + 1   # 0..7 -> 1..8 (server: 0=none)
        packet, recovered = self._bridge.step(move, fire)
        if recovered:
            # MAME instance wedged and was relaunched: end the episode with no
            # reward signal. SB3 will call reset(), which finds a fresh game.
            self._obs_builder.reset()
            obs = self._obs_builder(packet).astype(np.float32)
            h = parse_header(packet)
            self._last_score, self._last_lives, self._last_wave = h["score"], h["lives"], h["wave"]
            self._last_spawn, self._last_shoot, self._last_brain, self._last_family = _count_strategic(packet)
            return obs, 0.0, True, False, {"score": h["score"], "lives": h["lives"],
                                           "wave": h["wave"], "recovered": True}
        h = parse_header(packet)
        score, lives, wave = h["score"], h["lives"], h["wave"]
        score_delta = score - self._last_score
        # Terminal: out of lives, OR the wave register made a non-sequential
        # move (gameplay only ever holds or increments by 1) — that means the
        # game left play mode (game over -> high-score/attract screens, where
        # $BDED/$BDEC hold garbage like wave=171 lives=2). Without this check,
        # episodes bleed into attract mode and collect spurious reward.
        terminated = (lives == 0) or not (self._last_wave <= wave <= self._last_wave + 1)

        # Death forensics: log every life loss (uses pre-death packet history).
        if lives < self._last_lives or terminated:
            self._log_death(packet, self._last_wave, self._last_score, terminated)

        # Maintain pre-death history AFTER forensics (3 most recent packets).
        self._pkt_history.append(packet)
        if len(self._pkt_history) > 3:
            self._pkt_history.pop(0)
        self._last_packet = packet

        # Reward shaping. On the terminal step the game has flipped to attract
        # mode ($BDED garbage), so terminal gets the death penalty ONLY (no
        # wave/score deltas → no spurious suicide bonus).
        # Death penalty -20: C3 tested a heavy penalty (-200/-75) and it made
        # continuous play WORSE (mean wave 3.2→2.6, more early deaths) — a big
        # negative terminal spike destabilized rather than taught caution.
        # Reverted to the C1 value (best continuous result).
        if terminated:
            reward = -20.0
        else:
            reward = score_delta / 10.0
            if lives < self._last_lives:
                reward += -20.0
            elif lives > self._last_lives:
                # Banked a bonus life (Robotron awards one at ~25k). This is the
                # SNOWBALL TRIGGER for a marathon: more lives -> survive deeper
                # -> rescue more -> more lives. C1-C4 stalled at wave ~3 because
                # the policy dies (~21k) just short of the first bonus life and
                # never starts the flywheel. Reward the 1-up directly so the
                # value function pulls play toward crossing that threshold
                # (2026-06-13).
                reward += 250.0
            else:
                reward += 0.3 * max(1, int(wave))
            # Real play advances exactly one wave at a time; a jump >1 means
            # we're reading a non-gameplay screen — no bonus.
            if wave == self._last_wave + 1:
                reward += 250.0 * wave
                if wave >= 10: reward += 8000.0
                elif wave >= 7: reward += 3000.0
                elif wave >= 5: reward += 1000.0

        spawn, shoot, brain, family = _count_strategic(packet)
        if not terminated and score_delta > 0:
            reward += 50.0 * max(0, self._last_spawn - spawn)
            reward += 20.0 * max(0, self._last_shoot - shoot)
            reward += 100.0 * max(0, self._last_brain - brain)
            # Civilian rescue: a family member vanishing WITH a score jump in the
            # rescue-bonus band (1000-5000) is a pickup, not a Hulk/Brain kill
            # (those score 0). Rescues are the dominant point source and thus the
            # 1-up engine that sustains a continuous 1->N marathon, so reward them
            # as a first-class objective rather than diffuse score_delta. (Rare
            # false positive: a family death in the same 4-frame step as a >=900
            # enemy-kill score; acceptable.)
            if score_delta >= 900:
                reward += 75.0 * max(0, self._last_family - family)

        truncated = False

        # On a terminal step the packet may hold attract-mode garbage
        # (wave=171, BCD score junk). Report the last VALID gameplay values in
        # info so metrics (highest_score / highest_wave) stay truthful.
        if terminated:
            info = {"score": self._last_score, "lives": 0, "wave": self._last_wave}
        else:
            # Auto-capture: entering a frontier wave arms a settle countdown;
            # save once the transition flash is over and the player isn't
            # mid-death (header byte 9 = $9848 being-killed flag).
            if self._auto_min_wave is not None:
                if (wave == self._last_wave + 1 and wave >= self._auto_min_wave
                        and wave not in self._auto_saved_waves):
                    self._auto_settle = self.AUTO_CAPTURE_SETTLE
                    self._auto_wave = wave
                elif wave != getattr(self, "_auto_wave", -1):
                    self._auto_settle = -1
                if self._auto_settle > 0:
                    self._auto_settle -= 1
                elif self._auto_settle == 0:
                    self._auto_settle = -1
                    if packet[9] == 0 and lives > 0:
                        idx = (self.AUTO_CAPTURE_BASE + self._rank * 8
                               + min(wave - self._auto_min_wave, 7))
                        self._bridge.save_state(idx)
                        self._auto_saved_waves.add(wave)
                        print(f"[auto-capture rank {self._rank}] saved w5_{idx}: "
                              f"wave={wave} score={score} lives={lives}", flush=True)
            info = {"score": score, "lives": lives, "wave": wave}
            self._last_score, self._last_lives, self._last_wave = score, lives, wave
            self._last_spawn, self._last_shoot, self._last_brain, self._last_family = spawn, shoot, brain, family

        obs = self._obs_builder(packet).astype(np.float32)
        return obs, reward, terminated, truncated, info

    def close(self):
        if self._bridge is not None:
            self._bridge.close()
            self._bridge = None
