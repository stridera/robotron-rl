# Robotron MAME gym

Train the RL agent against the **stock 1982 Williams arcade ROM** running in MAME —
the same ROM the Xbox 360 XBLA `robotron.xex` wraps, so a trained policy transfers
to the XBLA target with no domain gap. See `memory/robotron-mame-pivot.md`.

## Status / plan

1. **[blocked on you]** Install MAME + a Robotron romset (see below).
2. Run `probe.lua` to find/verify the 6809 RAM addresses for player X/Y, the object
   table (base/stride/handler-pointer format), score, wave, lives.
3. Write `bridge.lua` — reads those addresses each frame, exposes step/reset over a
   socket; `reset()` uses MAME savestates to jump to a chosen wave.
4. Write `robotron_mame_env.py` — a Gymnasium env reusing the existing 945-dim obs /
   `Discrete(64)` action contract (mirrors `position_wrapper.py`), vectorized with
   `SubprocVecEnv` (N MAME processes) for parallel training.

## What I need from you

```bash
# 1) Install MAME (Ubuntu noble has 0.264):
!sudo apt install -y mame

# 2) Provide a Robotron romset as roms/robotron.zip (you own the game via XBLA).
#    Verify the set MAME sees:
mame -rompath /home/strider/Code/robotron-rl/mame_gym/roms -verifyroms robotron
```

Once both are in place, the probe runs headless:
```bash
cd /home/strider/Code/robotron-rl/mame_gym
mame robotron -rompath roms -nothrottle -seconds_to_run 600 \
     -autoboot_script probe.lua -video none -sound none
```
