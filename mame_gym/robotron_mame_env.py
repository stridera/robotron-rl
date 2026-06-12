"""robotron_mame_env — Gymnasium env that drives MAME running stock Robotron 2084
over a pair of named pipes. v0 scaffold: focus is end-to-end correctness, not yet
the final 945-dim contract.
"""
from __future__ import annotations
import os, struct, subprocess, tempfile, time
import numpy as np
import gymnasium as gym

SLOTS_REPORTED = 24
OBS_BYTES = 2 + (SLOTS_REPORTED - 1) * 4   # must match bridge.lua

MAME_BIN     = "mame"
DEFAULT_ROM  = "robotron87"
HERE         = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROMS = os.path.join(HERE, "roms")
DEFAULT_LUA  = os.path.join(HERE, "bridge.lua")


class RobotronMAMEEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, romset=DEFAULT_ROM, rompath=DEFAULT_ROMS,
                 lua_script=DEFAULT_LUA, mame_bin=MAME_BIN, verbose=False):
        # 9 discrete dirs per axis: 0=none, 1=N..8=NW
        self.action_space = gym.spaces.MultiDiscrete([9, 9])
        # raw byte observation for now; we will reshape into the 945-dim layout
        # once we have confirmed which fields are X/Y/type per slot.
        self.observation_space = gym.spaces.Box(0, 255, (OBS_BYTES,), np.uint8)

        self._tmp = tempfile.mkdtemp(prefix="robomame_")
        self._fifo_py_to_mame = os.path.join(self._tmp, "py_to_mame.fifo")
        self._fifo_mame_to_py = os.path.join(self._tmp, "mame_to_py.fifo")
        os.mkfifo(self._fifo_py_to_mame)
        os.mkfifo(self._fifo_mame_to_py)

        env = os.environ.copy()
        env["BRIDGE_IN"]  = self._fifo_py_to_mame   # mame reads this
        env["BRIDGE_OUT"] = self._fifo_mame_to_py   # mame writes this
        state_dir = os.path.join(self._tmp, "states")
        os.makedirs(state_dir, exist_ok=True)
        cmd = [mame_bin, romset, "-rompath", rompath, "-nothrottle",
               "-state_directory", state_dir,
               "-autoboot_script", lua_script, "-video", "none", "-sound", "none"]
        stdout = None if verbose else subprocess.DEVNULL
        self._proc = subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stdout)

        # IMPORTANT: open order must mirror bridge.lua's. Bridge opens IN(read)
        # then OUT(write); we open IN(write) then OUT(read) so each pair
        # rendezvous instead of both blocking on the same pipe.
        self._w = open(self._fifo_py_to_mame, "wb", buffering=0)
        self._r = open(self._fifo_mame_to_py, "rb", buffering=0)
        self._closed = False
        self._has_checkpoint = False

    # --- gym API -----------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._has_checkpoint:
            # load takes effect at the next quiesce point — burn one frame so
            # the next observation reflects the loaded state, not the pre-load one.
            self._w.write(bytes([0, 0, 0, 2]))   # flags=load
            _ = self._recv_obs()                  # discard pre-load obs
            self._w.write(bytes([0, 0, 0, 0]))   # idle step to settle
            obs = self._recv_obs()
        else:
            obs = self._recv_obs()
        return obs, {}

    def step(self, action):
        move_dir, fire_dir = int(action[0]), int(action[1])
        self._w.write(bytes([move_dir, fire_dir, 0, 0]))
        obs = self._recv_obs()
        return obs, 0.0, False, False, {}

    def save_checkpoint(self):
        """Capture the current emulator state. reset() will restore it."""
        self._w.write(bytes([0, 0, 0, 1]))    # flags=save
        _ = self._recv_obs()                  # consume the obs after save command
        self._has_checkpoint = True

    def close(self):
        if self._closed: return
        self._closed = True
        try: self._w.close()
        except Exception: pass
        try: self._r.close()
        except Exception: pass
        if self._proc.poll() is None:
            self._proc.terminate()
            try: self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired: self._proc.kill()
        for f in (self._fifo_py_to_mame, self._fifo_mame_to_py):
            try: os.unlink(f)
            except FileNotFoundError: pass

    def __del__(self): self.close()

    # --- helpers -----------------------------------------------------------
    def _recv_obs(self):
        buf = b""
        while len(buf) < OBS_BYTES:
            chunk = self._r.read(OBS_BYTES - len(buf))
            if not chunk:
                raise RuntimeError("MAME bridge closed unexpectedly "
                                   f"(proc returncode={self._proc.poll()})")
            buf += chunk
        return np.frombuffer(buf, dtype=np.uint8)

    @staticmethod
    def decode(obs):
        """Return a dict view of the raw obs for debugging."""
        px, py = int(obs[0]), int(obs[1])
        slots = []
        for i in range(1, SLOTS_REPORTED):
            o = 2 + (i - 1) * 4
            slots.append(dict(b0=int(obs[o]), b1=int(obs[o+1]),
                              sw=int(obs[o+2]) << 8 | int(obs[o+3])))
        return dict(player_x=px, player_y=py, slots=slots)
