"""Diagnostic experiment: train PPO on the native gym with cap_001 ONLY.

Localizes whether snapshot rotation (cap_001/005/010/015/020 mix) was the
recipe bug responsible for the chain regression. By restricting to cap_001
(wave 1 start), every episode gets a real shot at L1→L5 progression and
the Option A deep-wave bonuses can fire.

Otherwise identical to train_native.py — warmstart from current chain head,
300k timesteps so we get a fast verdict.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Force single-snapshot rotation BEFORE train_native module load.
import train_native
train_native.SNAPSHOTS = ['/tmp/cap_001_6809.bin']

CHAIN_HEAD = "models/vm9hy8gw/checkpoints/ppo_native_checkpoint_1000000_steps.zip"
VEC_NORM   = "models/vm9hy8gw/vec_normalize.pkl"


if __name__ == "__main__":
    print(f"[DIAGNOSTIC] Snapshot pool restricted to: {train_native.SNAPSHOTS}")
    if not Path(CHAIN_HEAD).exists():
        raise SystemExit(f"Checkpoint not found: {CHAIN_HEAD}")
    train_native.main(
        num_envs=16,
        total_timesteps=300_000,
        bc_checkpoint=CHAIN_HEAD,
        vec_normalize=VEC_NORM,
        lives=3,
        lr=2e-4,
        clip_range=0.2,
        ent_coef=0.02,
        device="cpu",
    )
