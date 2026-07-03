"""Minimal-deviation multi-step analytic clearance planner (the champion's brain).

Extracted from search_value.py (2026-07-01) so the eval harness, eval_protocol
promotions, and future live deploys share ONE implementation. Behavior-identical
to the N=100-validated champion: VSEARCH_CLEAR H=6 DANGER=18 MARGIN=10, ASM-true
dynamics (VSEARCH_ASMDYN=1 default).

Config via the same env vars search_value.py used, read at import:
  VSEARCH_H, VSEARCH_CLEAR_DANGER, VSEARCH_CLEAR_MARGIN, VSEARCH_ASMDYN,
  VSEARCH_FIREPLAN, VSEARCH_FIREPLAN_HULK_R
"""
from __future__ import annotations

import math
import os

import numpy as np

CLEAR_H = int(os.environ.get("VSEARCH_H", "6"))
CLEAR_DANGER = float(os.environ.get("VSEARCH_CLEAR_DANGER", "18"))
CLEAR_MARGIN = float(os.environ.get("VSEARCH_CLEAR_MARGIN", "10"))
ASMDYN = os.environ.get("VSEARCH_ASMDYN", "1") == "1"
FIREPLAN = os.environ.get("VSEARCH_FIREPLAN", "0") == "1"
FIREPLAN_HULK_R = float(os.environ.get("VSEARCH_FIREPLAN_HULK_R", "90"))

# Per-env-step player displacement (obs px) by server move dir 1..8, measured
# 2026-06-28 at frameskip 4; scales linearly with frameskip (1px/frame per axis).
_FS_SCALE = int(os.environ.get("VSEARCH_FRAMESKIP", "4")) / 4.0
DXY = {1: (0.0, -9.2), 2: (9.5, -9.2), 3: (9.5, 0.0), 4: (9.5, 9.2),
       5: (0.0, 9.2), 6: (-9.5, 9.2), 7: (-9.5, 0.0), 8: (-9.5, -9.2)}
DXY = {d: (x * _FS_SCALE, y * _FS_SCALE) for d, (x, y) in DXY.items()}
PX_MIN, PX_MAX, PY_MIN, PY_MAX = 10.0, 655.0, 10.0, 482.0

_FIRE_DIRS = (3, 4, 5, 6, 7, 8, 1, 2)   # compass by 45deg from East, screen y-down


def _angdist(a, b):
    d = abs(a - b)
    return min(d, 8 - d)   # compass steps 0..4


def _dir_toward(px, py, tx, ty):
    """Quantize the bearing player->target to the 1..8 compass (1=N .. 3=E .. 5=S)."""
    ang = math.degrees(math.atan2(ty - py, tx - px))   # 0=E, +90=S (screen y-down)
    return _FIRE_DIRS[int(round(ang / 45.0)) % 8]


def _classify_threat(name, vx, vy):
    """(weight, is_chaser, chase_speed, reflects). weight<=0 => ignore (family).
    ASM-grounded dynamics (ENEMY_MODEL.md §4/§9 + robomame.asm); ASMDYN=0 restores
    the pre-2026-07 model (A/B control)."""
    spd = math.hypot(vx, vy)
    if not ASMDYN:                                   # pre-2026-07 baseline (A/B control)
        if name in ("Mikey", "Mommy", "Daddy"):
            return 0.0, False, 0.0, False
        if name == "Hulk":
            return 2.5, True, max(2.5, min(spd, 6.0)), False
        if name == "Quark":
            return 2.0, False, 0.0, False
        if name in ("EnforcerBullet", "Spark", "CruiseMissile", "TankShell"):
            return 1.8, False, spd, False
        if name == "Grunt":
            return 1.0, True, max(5.0, min(spd, 13.0)), False
        if name in ("Brain", "Prog"):
            return 1.2, True, max(4.0, min(spd, 12.0)), False
        if name == "Electrode":
            return 0.8, False, 0.0, False
        return 1.0, False, spd, False
    if name in ("Mikey", "Mommy", "Daddy"):
        return 0.0, False, 0.0, False                # family — not lethal
    if name == "Hulk":
        # invincible pinning obstacle (30% of deaths) — wide berth, slow 4-dir mover
        return 2.2, True, max(2.0, min(spd, 5.0)), False
    if name == "Quark":
        # slow random wanderer, NO homing, <=~9.5px/step (asm:7201-7248)
        return 1.0, False, spd, False
    if name == "TankShell":
        # straight & fast, REFLECTS off all 4 walls (asm:7807-7857); ~half are
        # deliberate wall-bank shots (asm:7694-7757)
        return 1.8, False, spd, True
    if name == "CruiseMissile":
        # homes on the player, re-aims every <=8 ticks (asm:3082-3118)
        return 1.6, True, max(6.0, min(spd, 14.0)), False
    if name == "EnforcerBullet":
        # spark: ballistic + fixed curvature, never re-homes (asm:2082-2087)
        return 1.8, False, spd, False
    if name == "Enforcer":
        # dives at the player, velocity ∝ distance (asm:1933-1964)
        return 1.3, True, max(3.0, min(spd, 22.0)), False
    if name == "Grunt":
        return 1.0, True, max(5.0, min(spd, 13.0)), False   # 4px axis beeline
    if name == "Prog":
        return 1.1, True, max(4.0, min(spd, 12.0)), False
    if name == "Brain":
        # chases the nearest HUMAN, not the player (asm:2466-2475)
        return 1.1, False, spd, False
    if name == "Electrode":
        return 0.8, False, 0.0, False
    return 1.0, False, spd, False                    # Tank/Spheroid/other


def _heading_clearance(px, py, T, d, H, return_argmin=False):
    """Min weighted clearance if the player commits to heading d for H env-steps.
    T = (tx,ty,vx,vy,w,chase,spd,refl) numpy arrays. Chasers re-aim toward the
    player each step; reflectors mirror off walls with velocity negation
    (asm:7807-7857); others clamp at the border like the game's integrator
    (asm:17308-17327). With return_argmin, also returns the BINDING threat index."""
    tx, ty, tvx, tvy, w, chase, spd, refl = T
    ppx, ppy = px, py
    dx_, dy_ = DXY[d]
    tx = tx.copy(); ty = ty.copy()
    tvx = tvx.copy(); tvy = tvy.copy()
    worst = 1e18; worst_i = 0
    for _ in range(H):
        ppx = min(PX_MAX, max(PX_MIN, ppx + dx_))
        ppy = min(PY_MAX, max(PY_MIN, ppy + dy_))
        if chase.any():
            ax = ppx - tx; ay = ppy - ty
            dist = np.sqrt(ax * ax + ay * ay) + 1e-6
            step = spd / dist
            tx = np.where(chase, tx + ax * step, tx + tvx)
            ty = np.where(chase, ty + ay * step, ty + tvy)
        else:
            tx = tx + tvx; ty = ty + tvy
        if ASMDYN:
            if refl.any():
                lo = refl & (tx < PX_MIN); hi = refl & (tx > PX_MAX)
                tvx = np.where(lo | hi, -tvx, tvx)
                tx = np.where(lo, 2 * PX_MIN - tx, np.where(hi, 2 * PX_MAX - tx, tx))
                lo = refl & (ty < PY_MIN); hi = refl & (ty > PY_MAX)
                tvy = np.where(lo | hi, -tvy, tvy)
                ty = np.where(lo, 2 * PY_MIN - ty, np.where(hi, 2 * PY_MAX - ty, ty))
            tx = np.clip(tx, PX_MIN, PX_MAX); ty = np.clip(ty, PY_MIN, PY_MAX)
        clr = np.sqrt((ppx - tx) ** 2 + (ppy - ty) ** 2) / w
        i = int(clr.argmin()); m = float(clr[i])
        if m < worst:
            worst = m; worst_i = i
    return (worst, worst_i) if return_argmin else worst


def clearance_search(sprites, fsm_mv, first_fire):
    """Guarded multi-step clearance override over pre-extracted sprites
    [(x, y, name, vx, vy), ...]. Keeps the FSM move unless its heading is in
    danger and a clearly-safer heading exists (minimal deviation)."""
    player = next((s for s in sprites if s[2] == "Player"), None)
    if player is None:
        return fsm_mv, first_fire
    px, py = player[0], player[1]
    xs, ys, vxs, vys, ws, ch, sp, rf, nm = [], [], [], [], [], [], [], [], []
    for s in sprites:
        if s[2] == "Player":
            continue
        wt, isch, speed, refl = _classify_threat(s[2], s[3], s[4])
        if wt <= 0.0:
            continue
        xs.append(s[0]); ys.append(s[1]); vxs.append(s[3]); vys.append(s[4])
        ws.append(wt); ch.append(isch); sp.append(speed); rf.append(refl)
        nm.append(s[2])
    if not xs:
        return fsm_mv, first_fire
    T = (np.array(xs), np.array(ys), np.array(vxs), np.array(vys),
         np.array(ws), np.array(ch, dtype=bool), np.array(sp),
         np.array(rf, dtype=bool))
    clr = [_heading_clearance(px, py, T, d, CLEAR_H) for d in range(1, 9)]
    v_fsm = clr[fsm_mv - 1]
    if v_fsm >= CLEAR_DANGER:                      # FSM heading is safe enough — trust it
        return fsm_mv, first_fire
    cand = [d for d in range(1, 9) if clr[d - 1] >= v_fsm + CLEAR_MARGIN]
    best_d = min(cand, key=lambda d: (_angdist(d, fsm_mv), -clr[d - 1])) if cand else fsm_mv
    if FIREPLAN:
        _, bi = _heading_clearance(px, py, T, best_d, CLEAR_H, return_argmin=True)
        bx, by = float(T[0][bi]), float(T[1][bi])
        if nm[bi] != "Hulk" or ((bx - px) ** 2 + (by - py) ** 2) <= FIREPLAN_HULK_R ** 2:
            return best_d, _dir_toward(px, py, bx, by)
    return best_d, first_fire
