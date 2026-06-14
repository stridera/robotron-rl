"""mame_bridge.py — Python side of the MAME Robotron RL bridge.

Launches a headless MAME running robotron_server.lua, connects over TCP, and
exposes a synchronous step/reset interface. One MameBridge == one MAME process
== one env.

Protocol mirrors robotron_server.lua:
  send 4 bytes [cmd, move_dir, fire_dir, frameskip]   cmd 0=step 1=reset 2=quit
  recv OBS_LEN bytes  = 10-byte header + 2424-byte slot pool
"""
from __future__ import annotations
import os
import socket
import struct
import subprocess
import time
from pathlib import Path

MAME_BIN = os.environ.get("MAME_BIN", "/usr/games/mame")
ROMPATH = os.environ.get("MAME_ROMPATH", str(Path.home() / "mame_robotron" / "roms"))
LUA_SCRIPT = str(Path(__file__).parent / "robotron_server.lua")
# Persistent — a reboot wiped /tmp/mame_states on 2026-06-11, losing the
# entire wave-ladder reset pool. Never keep save states on tmpfs.
STATE_DIR = os.environ.get("MAME_STATE_DIR",
                           str(Path.home() / "Code" / "robotron-rl" / "mame_states"))

HDR_LEN = 10
SLOT_BASE = 0x98D4
SLOT_BYTES = 101 * 24
FIXED_LEN = HDR_LEN + SLOT_BYTES + 1   # header + pool + n_entities byte
ENTITY_REC = 7                          # [addr_hi, addr_lo, list_id, sw_hi, sw_lo, x, y]

CMD_STEP, CMD_RESET, CMD_QUIT, CMD_SAVE, CMD_SNAP = 0, 1, 2, 3, 4


class MameBridge:
    def __init__(self, port: int, frameskip: int = 4, boot_timeout: float = 30.0):
        self.port = port
        self.frameskip = frameskip
        self._proc = None
        self._sock = None
        self._launch(boot_timeout)

    def _launch(self, boot_timeout: float):
        # SDL dummy drivers: with WSLg's display reachable (post-reboot),
        # "-video none" alone still opens a fullscreen black window per
        # instance. Dummy drivers guarantee true headless.
        env = dict(os.environ, MAME_RL_PORT=str(self.port),
                   SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy")
        log_path = os.environ.get("MAME_LOG", "")
        out = open(log_path, "w") if log_path else subprocess.DEVNULL
        # Headless, uncapped. cwd at rompath parent so MAME finds cfg/nvram.
        self._proc = subprocess.Popen(
            [MAME_BIN, "-rompath", ROMPATH, "robotron",
             "-video", "none", "-sound", "none", "-nothrottle",
             "-autoboot_script", LUA_SCRIPT,
             "-state_directory", STATE_DIR],
            cwd=str(Path(ROMPATH).parent),
            env=env, stdout=out, stderr=out,
        )
        # MAME opens the listen socket on its first frame; retry-connect.
        deadline = time.time() + boot_timeout
        last_err = None
        while time.time() < deadline:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(5.0)
                s.connect(("127.0.0.1", self.port))
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                # Generous I/O timeout: with many MAME instances booting
                # concurrently, the first reset (boot 900 frames + save-state)
                # can take tens of seconds under CPU contention.
                s.settimeout(120.0)
                self._sock = s
                break
            except (ConnectionRefusedError, OSError) as e:
                last_err = e
                time.sleep(0.1)
        if self._sock is None:
            self.close()
            raise RuntimeError(f"could not connect to MAME on port {self.port}: {last_err}")

    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("MAME closed the connection")
            buf += chunk
        return bytes(buf)

    def _recv_obs(self) -> bytes:
        fixed = self._recv_exact(FIXED_LEN)
        n = fixed[-1]
        tail = self._recv_exact(n * ENTITY_REC) if n else b""
        return fixed + tail

    def step(self, move_dir: int, fire_dir: int):
        """Apply action, advance frameskip frames, return raw obs bytes.
        Returns (obs, recovered): recovered=True means the MAME instance
        wedged and was relaunched — the episode is invalid past this point."""
        try:
            self._sock.sendall(bytes([CMD_STEP, move_dir & 0xFF, fire_dir & 0xFF, self.frameskip]))
            return self._recv_obs(), False
        except (TimeoutError, ConnectionError, OSError):
            self._recover()
            return self._safe_reset(), True

    def reset(self, state_idx: int = 0):
        """Load a saved state and return its observation.
        state_idx 0 = boot (wave-1) state; N>0 = saved state 'w5_N'."""
        try:
            # idx as 2 bytes (high, low). Was 1 byte → indices >=256 wrapped
            # mod 256 and aliased states in the pool (fixed 2026-06-12).
            self._sock.sendall(bytes([CMD_RESET, (state_idx >> 8) & 0xFF, state_idx & 0xFF, 0]))
            return self._recv_obs()
        except (TimeoutError, ConnectionError, OSError):
            self._recover()
            return self._safe_reset()

    def save_state(self, state_idx: int):
        """Save the current machine state as 'w5_<idx>' (2-byte idx, 1-65535)."""
        self._sock.sendall(bytes([CMD_SAVE, (state_idx >> 8) & 0xFF, state_idx & 0xFF, 0]))
        return self._recv_obs()

    def snapshot(self):
        """Take a screenshot into MAME's snapshot dir; returns the obs taken
        one frame later (pair it with the PREVIOUS obs for exact alignment)."""
        self._sock.sendall(bytes([CMD_SNAP, 0, 0, 0]))
        return self._recv_obs()

    def _safe_reset(self):
        self._sock.sendall(bytes([CMD_RESET, 0, 0, 0]))
        return self._recv_obs()

    def _recover(self):
        """Kill the wedged MAME and boot a fresh one on the same port."""
        import sys
        print(f"[mame_bridge:{self.port}] instance wedged — relaunching",
              file=sys.stderr, flush=True)
        self.recoveries = getattr(self, "recoveries", 0) + 1
        try:
            if self._sock is not None:
                self._sock.close()
        except OSError:
            pass
        self._sock = None
        if self._proc is not None:
            self._proc.kill()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            self._proc = None
        self._launch(boot_timeout=60.0)

    def close(self):
        try:
            if self._sock is not None:
                self._sock.sendall(bytes([CMD_QUIT, 0, 0, 0]))
                self._sock.close()
        except OSError:
            pass
        self._sock = None
        if self._proc is not None:
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None


def parse_obs_header(obs: bytes) -> dict:
    # byte 8 = current_player ($983F), byte 9 = game_state ($9859). See
    # mame_obs.parse_header (authoritative); kept here for standalone use.
    wave, lives, s5, s6, s7, px, pxsub, py, cur_player, game_state = obs[:HDR_LEN]
    score = ((s5 >> 4) * 10 + (s5 & 0xF)) * 10000 + \
            ((s6 >> 4) * 10 + (s6 & 0xF)) * 100 + \
            ((s7 >> 4) * 10 + (s7 & 0xF))
    return {"wave": wave, "lives": lives, "score": score,
            "player_x": px, "player_y": py, "game_state": game_state}
