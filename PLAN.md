# Robotron RL — Fast Path to a Playable Bot

## Goal
Train an RL agent that matches then exceeds Brain3 (~W7, ~65k pts real game) and
transfers cleanly to the Xbox 360 version running in Xenia. No pixels, no CNN —
structured position observations only, zero domain gap at deploy time.

---

## Architecture

### Observation: 945-dim category-guaranteed slots
41 slots × 23 features + 2 player = 945 dims

```
SPRITE_TYPES (16, must match engine class names exactly):
  Player, Grunt, Electrode, Hulk, Sphereoid, Quark,
  Brain, Enforcer, Tank, Mommy, Daddy, Mikey,
  Prog, CruiseMissile, EnforcerBullet, TankShell

SLOT_CATEGORIES (29 category slots + 12 catch-all = 41 total):
  8  × projectiles  [CruiseMissile, TankShell, EnforcerBullet]
  4  × spawners     [Sphereoid, Quark]
  4  × shooters     [Tank, Enforcer]
  3  × brains       [Brain]
  6  × civilians    [Mommy, Daddy, Mikey]
  4  × hulks        [Hulk]
  12 × catch-all    (nearest unassigned: Grunts, Electrodes, Progs, overflow)

Per-slot features (23):
  type one-hot (16) | rel_pos (2) | dist (1) | angle (1) | valid (1) | velocity (2)
```

Player bullets excluded — agent infers position from firing actions.
Spawners/civilians always visible regardless of distance (category-guaranteed).

### Action space
`MultiDiscrete([8, 8])` — move direction × shoot direction, 0–7 (always_move=True).

### Training config
`config.yaml` — **real game waves**, not progressive curriculum.
The curriculum was a crutch for random initialization; BC warmup makes it unnecessary.

---

## Pipeline

### Step 1: Collect Brain3 demos (current)
```bash
python collect_brain3_demos.py --envs 10 --jobs 2
# Output: demos/brain3_demos.npz  (~1M transitions, real game config)
```

### Step 2: Behavioral cloning warmup
```bash
python train_bc.py
# Output: models/bc_init/  (Brain3-level policy from supervised learning)
# Expected: W3-5 behavior immediately, no RL needed yet
```

### Step 3: PPO fine-tuning from BC checkpoint
```bash
python train_progressive.py \
  --bc-checkpoint models/bc_init/final_model.zip \
  --start-level 1 \
  --num-envs 16 \
  --device cpu \
  --lives 3
# Fine-tuning HPs: lr=5e-5, clip=0.1, ent=0.005  (tight, preserves BC init)
# Switch to loose HPs (lr=2e-4, clip=0.2, ent=0.01) if policy stalls
```

### Step 4: Deploy to Xenia
```
~/win/code/robotron/real_game_player.py
  game_state.py (Xenia memory reader)
    → [(gx, gy, label), ...]
    → coordinate conversion: pixel_x = (gx-5)/140*665, pixel_y = (gy-15)/215*492
    → ObsExtractor (mirrors position_wrapper.py, no gym dependency)
    → 945-dim obs → VecNormalize (frozen) → PPO.predict()
    → (move_dir, shoot_dir) → sticks → player.py TCP → vgamepad → Xenia
```

---

## Key files

| File | Purpose |
|------|---------|
| `position_wrapper.py` | 945-dim obs wrapper; `OBS_DIM=945` |
| `train_progressive.py` | PPO training; `--bc-checkpoint`, `--start-level`, `--lr`, etc. |
| `collect_brain3_demos.py` | Brain3 demo collection on real game config |
| `train_bc.py` | Behavioral cloning from demos |
| `brain3_gym_adapter.py` | Adapts Brain3 logic to gym action/obs format |
| `~/win/code/robotron/real_game_player.py` | Xenia inference runner |

---

## HP reference

| Mode | lr | clip_range | ent_coef | vf_coef |
|------|----|-----------|---------|---------|
| BC fine-tune (tight) | 5e-5 | 0.1 | 0.005 | 0.5 |
| Loose (if stalled) | 2e-4 | 0.2 | 0.01 | 0.5 |
| Scratch | 3e-4 | 0.2 | 0.01 | 0.5 |

Stall signal: `approx_kl < 0.001` AND `clip_fraction < 0.03` sustained 2+ checks.

---

## Lessons learned

- **CruiseMissile not Cruise** — engine class is `CruiseMissile`; `Cruise` fell to catch-all silently
- **Bullet excluded** — player bullets are `Bullet` in engine but excluded entirely; agent infers from actions
- **Progressive curriculum ≠ real game** — curriculum level 15 ≈ real wave 1-2 in difficulty; all prior curriculum metrics were misleading
- **BC is essential** — scratch PPO on real game waves wastes compute; BC bootstraps Brain3 knowledge instantly
- **vf_coef=1.0 failed** — collapsed clip_fraction to 0.009, policy froze; use 0.5
- **Kill stale processes** — always verify single training process before starting a new run
