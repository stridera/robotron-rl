"""Run Brain4 against the native robotron_native gym to verify game logic.

Bridges Brain4's strategy (committed-goal + danger-map A*) directly to the native
6809-ROM-driven gym, skipping the pixel-coordinate roundtrip the python-gym
adapter does (native already lives in game units).

If Brain4 reaches L8-9 here, the native gym matches the real game it usually
reaches L8-9 on. If it gets stuck earlier, something in the gym diverges from
the real ROM.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, '/home/strider/Code/robotron_native/core')

import io, contextlib
_silence = io.StringIO()
with contextlib.redirect_stdout(_silence), contextlib.redirect_stderr(_silence):
    from gym_env import RobotronEnv
    from robotron_core import ENTITY_TYPES

# Import Brain4 via the existing adapter (handles xenia stubbing).
from brain4_gym_adapter import Brain4, Entity, GameState, SlotAssigner

# Native state_word → Brain label (from jit_entity_reader.STATE_WORD_LABELS).
# Adds electrode variants and skips HUD/decorative SWs.
_SW_TO_LABEL = {
    0x3A76: 'G',
    0x00B6: 'H',
    0x1DD6: 'B', 0x2119: 'B',
    0x1483: 'F',
    0x14DC: 'FB',
    0x12C8: 'S',
    0x4BC9: 'Q',
    0x4800: 'T',
    0x4DF2: 'TS',
    0x1F1F: 'MS',
    0x0330: 'CC', 0x0335: 'CW', 0x033A: 'CM',
    # Electrode variants (multiple state words in real ROM):
    0x3AA9: 'E', 0x3B85: 'E', 0x3B8A: 'E', 0x3B8F: 'E',
    0x3B94: 'E', 0x3B99: 'E', 0x3BA4: 'E',
}

_ANGLE_TO_NATIVE_DIR = {
    # idx = round(angle / (pi/4)) → native dir (0=none, 1=N, 2=NE, 3=E, ...)
     0: 3, 1: 2, 2: 1, 3: 8,  4: 7,  -1: 4, -2: 5, -3: 6, -4: 7,
}


def _stick_to_native_dir(sx: float, sy: float) -> int:
    """Brain stick (sx=right+, sy=up+) → native MultiDiscrete[9] direction 0..8."""
    if abs(sx) < 0.05 and abs(sy) < 0.05:
        return 0
    angle = math.atan2(sy, sx)
    idx = int(round(angle / (math.pi / 4)))
    return _ANGLE_TO_NATIVE_DIR.get(idx, 0)


def _build_gamestate(core, slot_assigner: SlotAssigner, t: float, score: int) -> GameState:
    px = core.read8(0x9864)
    py = core.read8(0x9866)
    wave = core.read8(0xBDED)
    lives = core.read8(0xBDEC)

    by_label: dict[str, list[tuple[float, float]]] = {}
    for slot in core.slots():
        label = _SW_TO_LABEL.get(slot['sw'])
        if label is None:
            continue
        by_label.setdefault(label, []).append((float(slot['gx']), float(slot['gy'])))

    entities = slot_assigner.assign(by_label)
    return GameState(
        player_gx=float(px), player_gy=float(py),
        entities=entities, wave=int(wave),
        score=int(score), lives=int(lives), timestamp=t,
    )


def _read_score(core) -> int:
    raw = core.read_range(0xBDE5, 3)
    return ((raw[0] >> 4) * 10 + (raw[0] & 0xF)) * 10000 + \
           ((raw[1] >> 4) * 10 + (raw[1] & 0xF)) * 100 + \
           ((raw[2] >> 4) * 10 + (raw[2] & 0xF))


def main(n_steps: int = 5000) -> None:
    with contextlib.redirect_stdout(_silence), contextlib.redirect_stderr(_silence):
        env = RobotronEnv()
        obs, info = env.reset()
    core = env._core
    brain = Brain4()
    slot_assigner = SlotAssigner()
    last_wave = core.read8(0xBDED)
    last_lives = core.read8(0xBDEC)
    last_score = _read_score(core)

    # Force lives=3 so we get a fair shot (snapshot is at 2).
    core.write8(0xBDEC, 3)
    last_lives = 3

    print(f"START: wave={last_wave} lives={last_lives} score={last_score} "
          f"player=({core.read8(0x9864)},{core.read8(0x9866)})")
    print(f"entities at start: ", end='')
    counts: dict[str, int] = {}
    for slot in core.slots():
        lbl = _SW_TO_LABEL.get(slot['sw'])
        if lbl:
            counts[lbl] = counts.get(lbl, 0) + 1
    print(' '.join(f'{k}x{v}' for k, v in sorted(counts.items())))
    print()

    wave_milestones: list[tuple[int, int, int]] = []  # (step, wave, score)
    death_steps: list[tuple[int, int, int]] = []
    t = 0.0
    for step in range(n_steps):
        t += 1 / 60.0
        score = _read_score(core)
        state = _build_gamestate(core, slot_assigner, t, score)
        mx, my, sx, sy = brain.think(state)
        move_dir = _stick_to_native_dir(mx, my)
        fire_dir = _stick_to_native_dir(sx, sy)

        with contextlib.redirect_stdout(_silence), contextlib.redirect_stderr(_silence):
            obs, reward, terminated, truncated, info = env.step([move_dir, fire_dir])

        wave = core.read8(0xBDED)
        lives = core.read8(0xBDEC)

        if wave != last_wave:
            print(f"  step {step:5d}: wave {last_wave} → {wave}  (score={score})")
            wave_milestones.append((step, wave, score))
            brain.on_wave_change(wave)
            last_wave = wave
        if lives < last_lives:
            print(f"  step {step:5d}: death (lives {last_lives}→{lives}, wave={wave}, score={score})")
            death_steps.append((step, wave, score))
            last_lives = lives
        if terminated or lives == 0:
            print(f"  step {step:5d}: TERMINATED (lives=0)")
            break
        last_score = score

    final_wave = core.read8(0xBDED)
    final_score = _read_score(core)
    final_lives = core.read8(0xBDEC)
    print()
    print(f"END: steps={step+1} wave={final_wave} score={final_score} lives={final_lives}")
    print(f"  waves cleared: {len(wave_milestones)} ({[w[1] for w in wave_milestones]})")
    print(f"  deaths:        {len(death_steps)} ({[d[1] for d in death_steps]})")
    env.close()


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    main(n)
