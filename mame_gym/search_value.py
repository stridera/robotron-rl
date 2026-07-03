"""search_value.py — VALUE-GUIDED search teacher.

Raw H-frame-rollout search (search_validate.py) failed (wave 3-4): it scores a candidate
move by an H-frame FSM rollout, but actual play continues with SEARCH, not the FSM, so the
rollout-policy != play-policy mismatch compounds. Fix: score each candidate move's endpoint
with a learned death-risk VALUE model V(obs) ~ frames-until-death under FSM continuation
(models/value_<tag>, trained by train_value.py).

Since we pick argmax_move V(endpoint) >= V at the evolved-FSM's own move, the chosen action
is a 1-step policy-IMPROVEMENT over the evolved FSM under V, and actual search-play >= FSM
(V is a lower bound), so the mismatch that broke raw search no longer compounds.

Per decision: keep the evolved-FSM fire; for each of the 8 move dirs, save/restore-roll R
env-steps (move applied at step 0, FSM continuation after) and score the endpoint by V.

Usage: MAME_RL_RESEED=1 .venv/bin/python mame_gym/search_value.py \
       <evolved.json|default> <value_tag> [N] [PORT] [R]
"""
import sys, json, statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "mame_gym"))
import numpy as np
import torch as th
from mame_robotron_env import MameRobotronEnv
from mame_obs import MameObsBuilder, parse_header
from position_wrapper import SLOT_CATEGORIES, CATCHALL_SLOTS
import robotron_fsm as fsm
sys.path.insert(0, str(ROOT))
from train_value import ValueNet

SRC = sys.argv[1] if len(sys.argv) > 1 else "default"
VTAG = sys.argv[2] if len(sys.argv) > 2 else "v2b"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 12
PORT = int(sys.argv[4]) if len(sys.argv) > 4 else 9966
R = int(sys.argv[5]) if len(sys.argv) > 5 else 1   # env-steps rolled before V eval
# Guarded safety-override (default): keep the FSM move unless its endpoint V drops below
# DANGER (predicted imminent death) AND an alternative move is safer by >= MARGIN. This
# preserves the evolved FSM's wave-clearing aggression and only dodges death.
import os
DANGER = float(os.environ.get("VSEARCH_DANGER", "0.35"))   # V below this => consider override
MARGIN = float(os.environ.get("VSEARCH_MARGIN", "0.10"))   # alt must beat FSM move by this
PURE = os.environ.get("VSEARCH_PURE", "0") == "1"          # 1 => pure argmax-V (ablation)
FSMONLY = os.environ.get("VSEARCH_FSMONLY", "0") == "1"    # 1 => no search/override (paired baseline)
# ANALYTIC mode: score candidate moves with a 1-step analytic forward model (known player
# kinematics + obs velocity channel for enemies) and evaluate V on the SYNTHETIC next-obs.
# No emulator save/restore -> immune to MAME's lossy save-state (which broke rollout search).
ANALYTIC = os.environ.get("VSEARCH_ANALYTIC", "0") == "1"
# Per-env-step player displacement (sprite/obs px) by SERVER move dir 1..8 (measured 2026-06-28).
DXY = {1: (0.0, -9.2), 2: (9.5, -9.2), 3: (9.5, 0.0), 4: (9.5, 9.2),
       5: (0.0, 9.2), 6: (-9.5, 9.2), 7: (-9.5, 0.0), 8: (-9.5, -9.2)}
PX_MIN, PX_MAX, PY_MIN, PY_MAX = 10.0, 655.0, 10.0, 482.0
# MULTI-STEP ANALYTIC CLEARANCE PLANNER (VSEARCH_CLEAR=1). The 1-step methods + the FSM's
# nearest-flee + the potential-field all miss CONVERGENCE: grunts CHASE the moving player, so a
# move safe for 1 step can lead into a pincer 4 steps later. Here we commit to each candidate
# heading for H env-steps, re-aim chasers toward the predicted player each step (ballistic threats
# fly straight; quarks/electrodes static), and score the heading by its WORST weighted clearance
# over the horizon. Geometric (no learned V on drifting synthetic obs). Guarded: only override the
# FSM move when the FSM's own heading is in danger AND a clearly-safer heading exists.
CLEAR    = os.environ.get("VSEARCH_CLEAR", "0") == "1"
CLEAR_H  = int(os.environ.get("VSEARCH_H", "6"))                  # horizon in env-steps
CLEAR_DANGER = float(os.environ.get("VSEARCH_CLEAR_DANGER", "26"))  # FSM-heading clearance below => danger
CLEAR_MARGIN = float(os.environ.get("VSEARCH_CLEAR_MARGIN", "8"))   # alt heading must beat FSM by this
# Minimal-deviation safe-dir criterion: when overriding, penalize moves far (in compass
# steps) from the FSM's intended dir so we dodge toward the nearest safe dir instead of an
# erratic global argmax-V. dirs 1..8 are in circular order (N,NE,E,SE,S,SW,W,NW), 45 deg/step.
DEVW = float(os.environ.get("VSEARCH_DEVW", "0.0"))        # 0 => plain argmax-V (old behavior)
# ASM-true threat dynamics (2026-07-01, ENEMY_MODEL.md): shells wall-bounce, cruise
# missiles home, enforcers dive, brains chase humans (not the player), quarks are slow
# wanderers (the "static teleporter" was a decoder phantom), and off-board extrapolation
# clamps at the border like the game's integrator. Set 0 for the pre-fix baseline (A/B).
ASMDYN = os.environ.get("VSEARCH_ASMDYN", "1") == "1"
# Planned FIRE (Stage 2a): when the clearance override engages (predicted death on the
# FSM heading), redirect fire at the BINDING THREAT of the chosen escape heading — the
# entity that minimizes clearance along it. Hulk => push it open (asm:471, laser shoves
# 6-12px); anything else (all laser-killable, incl. shells/sparks) => shoot it. Unlike the
# failed reactive HULK_PUSH/HULK_NOFIRE (which fired at hulks whenever pinning), this only
# steals the FSM's shot when the planner already predicts death AND the target is what
# actually blocks the escape. Opt-in for paired A/B.
FIREPLAN = os.environ.get("VSEARCH_FIREPLAN", "0") == "1"
FIREPLAN_HULK_R = float(os.environ.get("VSEARCH_FIREPLAN_HULK_R", "90"))

def _angdist(a, b):
    d = abs(a - b)
    return min(d, 8 - d)   # compass steps 0..4

W, Hpx = 665, 492
fsm.DEBUG_LEVEL = 0
fsm.MAX_RIGHT, fsm.MAX_TOP = W, Hpx
fsm.MAX_BOTTOM, fsm.MAX_LEFT = 0, 0
fsm.Y_AXIS_INVERSION = Hpx
if SRC != "default":
    params = json.loads(Path(SRC).read_text())["best_params"]
    for name, val in params.items():
        setattr(fsm, name, val)
    print(f"loaded {len(params)} evolved params from {SRC}", flush=True)
else:
    print("using DEFAULT FSM params", flush=True)
B = float(getattr(fsm, "BORDER_ADJUST", 20))
fsm.ADJ_TOP, fsm.ADJ_BOTTOM = Hpx - 2, 0 + B + 9
fsm.ADJ_LEFT, fsm.ADJ_RIGHT = 0 + 2, W - B

builder = MameObsBuilder()        # for FSM action selection
veval = MameObsBuilder()          # dedicated builder for value-net obs (own velocity context)
SCRATCH_IDX = 60002
DEATH_V = -1.0                    # below any V in [0,1]

# --- load value model ---
VDIR = ROOT / "models" / f"value_{VTAG}"
DEVICE = "cuda" if th.cuda.is_available() else "cpu"
_norm = np.load(VDIR / "norm.npz")
VMEAN = _norm["mean"].astype(np.float32); VVAR = _norm["var"].astype(np.float32)
vnet = ValueNet().to(DEVICE)
vnet.load_state_dict(th.load(VDIR / "state_dict.pt", map_location=DEVICE))
vnet.eval()
print(f"loaded value model {VDIR} (horizon {float(_norm['horizon']):.0f})", flush=True)


def value_of(packet):
    obs = veval(packet).astype(np.float32)
    x = (obs - VMEAN) / np.sqrt(VVAR)
    with th.no_grad():
        return float(vnet(th.as_tensor(x[None]).to(DEVICE)).item())


_EXTRACT = builder._extractor._extract_features


def analytic_values(cur_packet):
    """Return (fsm_mv, first_fire, [V for move 1..8]) via the analytic 1-step forward model.
    Player moved by known kinematics; enemies extrapolated by their obs velocity. No emulator."""
    sprites = builder._sprites_from_packet(cur_packet)   # (x,y,name,vx,vy)
    player = next((s for s in sprites if s[2] == "Player"), None)
    fmv, ffr = obs_fsm_action(cur_packet)
    if player is None:
        return fmv + 1, ffr + 1, [0.0] * 8
    px, py = player[0], player[1]
    others = [s for s in sprites if s[2] != "Player"]
    # enemies extrapolated one step by their velocity (keep vxy so the obs velocity channel matches)
    ext = [(s[0] + s[3], s[1] + s[4], s[2], s[3], s[4]) for s in others]
    obs_batch = np.empty((8, 945), dtype=np.float32)
    for i, d in enumerate(range(1, 9)):
        dx, dy = DXY[d]
        npx = min(PX_MAX, max(PX_MIN, px + dx))
        npy = min(PY_MAX, max(PY_MIN, py + dy))
        synth = [(npx, npy, "Player", 0.0, 0.0)] + ext
        obs_batch[i] = _EXTRACT(synth)
    xb = (obs_batch - VMEAN) / np.sqrt(VVAR)
    with th.no_grad():
        vs = vnet(th.as_tensor(xb).to(DEVICE)).cpu().numpy().tolist()
    return fmv + 1, ffr + 1, vs


def obs_fsm_action(packet):
    sprites = builder._sprites_from_packet(packet)
    player = next((s for s in sprites if s[2] == "Player"), None)
    if player is None:
        return 0, 0
    px, py = player[0], player[1]
    others = [s for s in sprites if s[2] != "Player"]
    def d2(s):
        return (s[0] - px) ** 2 + (s[1] - py) ** 2
    used, sel = set(), []
    for cnt, types in SLOT_CATEGORIES:
        for s in sorted([s for s in others if s[2] in types and id(s) not in used], key=d2)[:cnt]:
            used.add(id(s)); sel.append(s)
    sel += sorted([s for s in others if id(s) not in used], key=d2)[:CATCHALL_SLOTS]
    data = [(px, py, "Player")] + [(s[0], s[1], s[2]) for s in sel]
    try:
        mv, fr = fsm.chooseOutputs(data)
    except Exception:
        mv, fr = 1, 1
    mi = (mv - 1) if mv >= 1 else 0
    fi = (fr - 1) if fr >= 1 else mi
    return mi, fi


def _dead(h, base_wave):
    return (h["lives"] == 0) or (h["game_state"] == 0x1B) \
        or not (base_wave <= h["wave"] <= base_wave + 1)


def _rollout_v(bridge, cur_packet, mv, first_fire, base_wave):
    """R-step save/restore rollout from scratch applying move `mv` then FSM; return endpoint V."""
    bridge.reset(SCRATCH_IDX)
    veval.reset(); veval(cur_packet)            # seed velocity context from current state
    pkt = cur_packet
    for k in range(R):
        if k == 0:
            a_mv, a_fr = mv, first_fire
        else:
            cm, cf = obs_fsm_action(pkt)
            a_mv, a_fr = cm + 1, cf + 1
        pkt, recovered = bridge.step(a_mv, a_fr)
        if recovered:
            return DEATH_V
        if _dead(parse_header(pkt), base_wave):
            return DEATH_V
    return value_of(pkt)


# Planner core extracted to clearance_planner.py (2026-07-01) so eval_protocol
# promotions and live deploys share the N=100-validated implementation. The
# VSEARCH_* env config is read there (same names/defaults as before).
from clearance_planner import clearance_search as _clearance_search_sprites


def clearance_search(cur_packet, fsm_mv, first_fire):
    """Guarded multi-step clearance override (see clearance_planner.py)."""
    return _clearance_search_sprites(
        builder._sprites_from_packet(cur_packet), fsm_mv, first_fire)


def search_move(bridge, cur_packet):
    """Guarded value-override (default) or pure argmax-V. Restores the live game on exit."""
    base_wave = parse_header(cur_packet)["wave"]
    fmv, ffr = obs_fsm_action(cur_packet)
    first_fire = ffr + 1
    fsm_mv = fmv + 1
    if FSMONLY:
        return fsm_mv, first_fire

    if CLEAR:
        return clearance_search(cur_packet, fsm_mv, first_fire)

    if ANALYTIC:
        # No emulator rollout — analytic 1-step forward model + V on synthetic obs.
        _, _, vs = analytic_values(cur_packet)
        v_fsm = vs[fsm_mv - 1]
        if PURE:
            return int(np.argmax(vs)) + 1, first_fire
        if v_fsm >= DANGER:
            return fsm_mv, first_fire
        # FSM move looks risky -> choose a safer dir. With DEVW>0, prefer the safe dir
        # nearest the FSM's intended heading (minimal-deviation), else global argmax-V.
        if DEVW > 0.0:
            scored = [(vs[d - 1] - DEVW * _angdist(d, fsm_mv), d) for d in range(1, 9)]
            best_score, best_d = max(scored)
            best_v = vs[best_d - 1]
        else:
            best_d = int(np.argmax(vs)) + 1
            best_v = vs[best_d - 1]
        if best_v >= v_fsm + MARGIN:
            return best_d, first_fire
        return fsm_mv, first_fire

    bridge.save_state(SCRATCH_IDX)

    if not PURE:
        # Fast path: trust the FSM move unless its endpoint looks lethal.
        v_fsm = _rollout_v(bridge, cur_packet, fsm_mv, first_fire, base_wave)
        if v_fsm >= DANGER:
            bridge.reset(SCRATCH_IDX)
            return fsm_mv, first_fire
        # FSM move is risky -> search all dirs for a safer one.
        best_mv, best_val = fsm_mv, v_fsm
        for mv in range(1, 9):
            if mv == fsm_mv:
                continue
            val = _rollout_v(bridge, cur_packet, mv, first_fire, base_wave)
            if val > best_val:
                best_val, best_mv = val, mv
        bridge.reset(SCRATCH_IDX)
        return (best_mv if best_val >= v_fsm + MARGIN else fsm_mv), first_fire

    # PURE ablation: argmax_move V(endpoint).
    best_mv, best_val = fsm_mv, -1e18
    for mv in range(1, 9):
        val = _rollout_v(bridge, cur_packet, mv, first_fire, base_wave)
        if val > best_val:
            best_val, best_mv = val, mv
    bridge.reset(SCRATCH_IDX)
    return best_mv, first_fire


def main():
    # Step cap: 8000 (~wave 26) truncated deep games once the ASM-dynamics planner
    # started outliving it — right-censoring the wave metric. Default now sized for
    # wave-100-scale runs (~300 steps/wave observed).
    steps_cap = int(os.environ.get("VSEARCH_STEPS", "40000"))
    # Burn K resets before playing so games start at seed K+1 (the reseed LCG
    # advances once per reset). Lets a run resume a seed sequence after a
    # deterministic-wedge seed or a prior partial run; printed game numbers
    # include the offset so paired analysis maps directly onto seed indices.
    seed_skip = int(os.environ.get("VSEARCH_SEED_SKIP", "0"))
    # frameskip 4 = the historical default; 2 halves the between-decision blind
    # window (projectiles = 55%+ of champion-v2 deaths cross it). The planner's
    # per-step kinematics scale via the same env var (clearance_planner.FS_SCALE).
    frameskip = int(os.environ.get("VSEARCH_FRAMESKIP", "4"))
    env = MameRobotronEnv(rank=0, base_port=PORT, frameskip=frameskip, reset_pool=[0], obs_mode="slot")
    waves, scores = [], []
    # Resets consumed by the CURRENT MAME process. The reseed LCG advances once
    # per reset, so game k (1-based) must be that process's k-th reset. A wedge
    # relaunch starts a FRESH process (seed sequence restarts — see the
    # 2026-07-01 pitfall in EXPERIMENT_STATE); we burn resets to realign so
    # pairing survives wedges instead of silently looping early seeds.
    process_resets = 0
    for g in range(seed_skip, seed_skip + N):
        while process_resets < g:            # burn to seed g (covers seed_skip too)
            env.reset(); process_resets += 1
        env.reset(); process_resets += 1     # game's own reset -> seed g+1
        packet = env._last_packet
        mw = env._last_wave; sc = env._last_score; steps = 0
        recovered = False
        while steps < steps_cap:
            mv, fr = search_move(env._bridge, packet)
            _, _, term, trunc, info = env.step(np.array([mv - 1, fr - 1]))
            packet = env._last_packet
            mw = max(mw, info.get("wave", mw))
            if info.get("recovered"):
                recovered = True
                process_resets = 1           # fresh process; its safe-reset consumed seed 1
            if not (term or trunc):
                sc = info.get("score", sc)
            steps += 1
            if term or trunc:
                break
        waves.append(mw); scores.append(sc)
        tag = " RECOVERED-INVALID" if recovered else ""
        print(f"game {g+1}: wave={mw} score={sc} steps={steps}{tag}", flush=True)
    env.close()
    print(f"\n=== {N} games VALUE-SEARCH teacher (R={R}, V={VTAG}, base={SRC}) ===")
    print(f"wave: min={min(waves)} max={max(waves)} mean={statistics.mean(waves):.2f} median={statistics.median(waves)}")
    print(f"score: max={max(scores)} mean={statistics.mean(scores):.0f}")
    print(f"dist: {dict(sorted(Counter(waves).items()))}")


if __name__ == "__main__":
    main()
