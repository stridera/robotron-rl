"""evolve_fsm.py — evolve the obs-limited FSM's ~19 spacing thresholds to play DEEPER.

The FSM is the CEILING of the whole BC->anchored-RL pipeline (BC clones it, anchored-RL refines
the clone). Its ~19 fight-vs-flee distance thresholds were HAND-SET, never optimized. This runs a
(mu,lambda) evolution strategy over them, fitness = mean waves reached in continuous wave-1 MAME
games, to find a config that survives deeper. We evolve the OBS-LIMITED FSM (nearest-N selection,
the exact behavior BC clones) so gains stay clonable into the existing 945-dim obs and lift the
whole 10.4 chain. Seeded from the current defaults (the known ~11.4 config) so we search around a
strong point.

Persistent MAME workers (boot once) evaluate candidates via task/result queues. Each candidate is
played over K games (env is stochastic across resets even with a deterministic policy), fitness =
mean(max_wave) + mean(score)/1e6 tiebreak. Elite re-evaluated each gen (robust to fitness noise).

Usage: .venv/bin/python3 evolve_fsm.py [generations] [lambda] [mu] [K_games] [n_workers] [base_port] [tag]
Output: logs/evolve_<tag>.log (per-gen best), models/fsm_evolved_<tag>.json (best params + history).
"""
import os
import sys
import json
import multiprocessing as mp
from pathlib import Path

ROOT = Path(__file__).parent

GENS    = int(sys.argv[1]) if len(sys.argv) > 1 else 25
LAMBDA  = int(sys.argv[2]) if len(sys.argv) > 2 else 24
MU      = int(sys.argv[3]) if len(sys.argv) > 3 else 8
KGAMES  = int(sys.argv[4]) if len(sys.argv) > 4 else 5
NWORK   = int(sys.argv[5]) if len(sys.argv) > 5 else 12
BASEPRT = int(sys.argv[6]) if len(sys.argv) > 6 else 9920
TAG     = sys.argv[7] if len(sys.argv) > 7 else "v1"
# 4000 (~wave 14) sufficed for the bare FSM; champion-level candidates need more
# headroom or every good candidate saturates the cap and the gradient vanishes.
STEPCAP = int(os.environ.get("EVOLVE_STEPCAP", "4000"))

# name -> (default, low, high, is_int)
PARAMS = [
    ("BORDER_ADJUST",            20,  5,  60, False),
    ("ADJACENT",                 40, 10, 120, False),
    ("ADJACENT_HULK",            50, 10, 120, False),
    ("CLOSE",                    75, 20, 200, False),
    ("CLOSE_FIRE_PRIORITY_ENEMY",150, 40, 350, False),
    ("CLOSE_MOVE_PRIORITY_ENEMY", 50, 10, 250, False),
    ("CLOSE_FIRE_PROJECTILE",    150, 40, 350, False),
    ("CLOSE_MOVE_PROJECTILE",    100, 10, 300, False),
    ("CLOSE_FIRE_CHASE_ENEMY",   175, 40, 350, False),
    ("CLOSE_MOVE_CHASE_ENEMY",    50, 10, 250, False),
    ("CLOSE_FIRE_ENEMY",          75, 20, 250, False),
    ("CLOSE_MOVE_ENEMY",          50, 10, 200, False),
    ("CLOSE_FIRE_HULK",           50, 10, 200, False),
    ("CLOSE_MOVE_HULK",           50, 10, 200, False),
    ("CLOSE_FIRE_OBSTACLE",       40, 10, 150, False),
    ("CLOSE_MOVE_OBSTACLE",       40, 10, 150, False),
    ("CLOSE_MOVE_COUNT_LIMIT",     5,  1,  12, True),
    ("CLOSE_FIRE_COUNT_LIMIT",     3,  1,  12, True),
    ("CLOSE_MOVE_CIVILIAN",       75, 10, 250, False),
    # STRUCTURAL constants from the 2026-06-29 forensics fixes (validated +1.92 wave combined).
    # Co-optimized here so evolution can push the teacher PAST 14.27 (threshold-only plateaued ~12.7).
    # apply_params setattrs these on the fsm module; HULK_DEFLECT (bool) stays default-ON, not evolved.
    ("ADJACENT_QUARK",            75, 30, 130, False),  # quark-kiting flee trigger distance
    ("HULK_DEFLECT_R",            60, 20, 140, False),  # hulk-deflection lookahead radius
    # Endgame hunt (2026-07-03): pursue the last killable enemy instead of wall-hiding.
    # HUNT_KILLABLE is a 0/1 toggle gene (starts ON via default; evolution can turn it
    # off if it hurts); landing in best_params makes the winner json self-contained for
    # eval_protocol/deploy (no env flag needed).
    ("HUNT_KILLABLE",              1,  0,   1, True),
    ("HUNT_STANDOFF",             90, 40, 220, False),
]
NAMES = [p[0] for p in PARAMS]
LOW = [p[1] for p in PARAMS]; DEF = [p[1] for p in PARAMS]
LO  = [p[2] for p in PARAMS]; HI = [p[3] for p in PARAMS]; ISINT = [p[4] for p in PARAMS]
DIM = len(PARAMS)


def _clip(vec):
    import numpy as np
    v = np.array(vec, dtype=float)
    v = np.clip(v, LO, HI)
    for i in range(DIM):
        if ISINT[i]:
            v[i] = round(v[i])
    return v


def worker(wid, port, task_q, result_q):
    import sys as _s
    _s.path.insert(0, str(ROOT)); _s.path.insert(0, str(ROOT / "mame_gym"))
    import numpy as np
    from mame_robotron_env import MameRobotronEnv
    from mame_obs import MameObsBuilder
    from position_wrapper import SLOT_CATEGORIES, CATCHALL_SLOTS
    import robotron_fsm as fsm

    W, H = 665, 492
    fsm.DEBUG_LEVEL = 0
    fsm.MAX_RIGHT, fsm.MAX_TOP = W, H
    fsm.MAX_BOTTOM, fsm.MAX_LEFT = 0, 0
    fsm.Y_AXIS_INVERSION = H
    builder = MameObsBuilder()
    # EVOLVE_PLANNER=1: evaluate candidates as the FULL champion (FSM + clearance
    # override; FSM_* flags like RESCUE_SEEK apply via env). The 2026-06 evolutions
    # tuned constants for the BARE FSM under BROKEN entity labels — re-evolving
    # under the deployed system targets deaths/wave directly (the wave-100 gap).
    import os as _os
    _PLANNER = _os.environ.get("EVOLVE_PLANNER", "0") == "1"
    if _PLANNER:
        from clearance_planner import clearance_search as _clear

    def apply_params(vec):
        for i, name in enumerate(NAMES):
            setattr(fsm, name, int(vec[i]) if ISINT[i] else float(vec[i]))
        # board-edge avoidance (depends on BORDER_ADJUST)
        B = float(getattr(fsm, "BORDER_ADJUST"))
        fsm.ADJ_TOP, fsm.ADJ_BOTTOM = H - 2, 0 + B + 9
        fsm.ADJ_LEFT, fsm.ADJ_RIGHT = 0 + 2, W - B

    def fsm_action(packet):
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
        mv = mv if mv >= 1 else 1
        fr = fr if fr >= 1 else mv
        if _PLANNER:
            mv, fr = _clear(sprites, mv, fr)
        return mv - 1, fr - 1

    env = MameRobotronEnv(rank=wid, base_port=port, frameskip=4, reset_pool=[0], obs_mode="slot")

    def play_one():
        env.reset(); packet = env._last_packet
        max_wave = env._last_wave; score = env._last_score; steps = 0
        lives = 0
        while steps < STEPCAP:
            mi, fi = fsm_action(packet)
            _, _, term, trunc, info = env.step(np.array([mi, fi]))
            packet = env._last_packet
            max_wave = max(max_wave, info.get("wave", max_wave))
            lives = info.get("lives", lives)
            if not (term or trunc):
                score = info.get("score", score)
            steps += 1
            if term or trunc:
                break
        # lives at exit = the life-economy MARGIN. Death -> env reports 0; reaching
        # the step cap alive -> banked lives. This is the fitness signal once the
        # hunt-era candidates saturate the cap (2026-07-03 gen-1: best dist
        # [80,80,81,81,81] = all games at cap; wave alone had no gradient left).
        return max_wave, score, lives

    while True:
        task = task_q.get()
        if task is None:
            break
        cid, vec, k = task
        apply_params(vec)
        waves, scores, livesv = [], [], []
        for _ in range(k):
            w, s, lv = play_one()
            waves.append(w); scores.append(s); livesv.append(lv)
        # mean_wave dominates below the cap; at the cap, banked lives (x0.1)
        # differentiate; score stays as the fine tiebreak.
        fit = float(np.mean(waves)) + 0.1 * float(np.mean(livesv)) \
            + float(np.mean(scores)) / 1e6
        result_q.put((cid, fit, float(np.mean(waves)), float(np.mean(scores)),
                      waves, float(np.mean(livesv))))
    env.close()


def main():
    import numpy as np
    import time
    import os
    mp.set_start_method("spawn", force=True)
    rng = np.random.default_rng(0)
    logf = ROOT / "logs" / f"evolve_{TAG}.log"
    outf = ROOT / "models" / f"fsm_evolved_{TAG}.json"
    logf.parent.mkdir(exist_ok=True); outf.parent.mkdir(exist_ok=True)

    task_q = mp.Queue(); result_q = mp.Queue()
    workers = []
    for wid in range(NWORK):
        p = mp.Process(target=worker, args=(wid, BASEPRT + wid * 2, task_q, result_q))
        p.start(); workers.append(p)

    rng_range = np.array(HI) - np.array(LO)
    warm = os.environ.get("EVOLVE_WARM", "")
    if warm and Path(warm).exists():           # warm-start mean from a prior winner's best_params
        wp = json.loads(Path(warm).read_text())["best_params"]
        # tolerate params absent from an older winner (e.g. structural consts added later) -> default
        mean = _clip([wp.get(n, DEF[i]) for i, n in enumerate(NAMES)]).astype(float)
        log_msg = f"# WARM-START mean from {warm}"
    else:
        mean = _clip(DEF).astype(float)        # seed from the known defaults
        log_msg = "# cold-start mean from defaults"
    sigma = 0.15                               # fraction of each param's range
    best_vec, best_fit, best_wave = mean.copy(), -1.0, -1.0
    history = []

    def log(msg):
        with open(logf, "a") as fh:
            fh.write(msg + "\n")
        print(msg, flush=True)

    log(f"# evolve_fsm tag={TAG} gens={GENS} lambda={LAMBDA} mu={MU} K={KGAMES} workers={NWORK} dim={DIM}")
    log(log_msg)
    for gen in range(GENS):
        # population: elite (re-eval) + children ~ N(mean, sigma*range)
        pop = [mean.copy()]
        for _ in range(LAMBDA - 1):
            child = mean + rng.normal(0, 1, DIM) * sigma * rng_range
            pop.append(_clip(child))
        pop = [_clip(v) for v in pop]
        # dispatch
        for cid, vec in enumerate(pop):
            task_q.put((cid, vec, KGAMES))
        res = [None] * len(pop)
        for _ in range(len(pop)):
            cid, fit, mw, ms, waves, ml = result_q.get()
            res[cid] = (fit, mw, ms, waves, ml)
        # rank
        order = sorted(range(len(pop)), key=lambda i: res[i][0], reverse=True)
        elites = order[:MU]
        # recombine: mean of top-mu (equal weight)
        mean = _clip(np.mean([pop[i] for i in elites], axis=0)).astype(float)
        gen_best_i = order[0]
        gbf, gbw, gbs, gbwaves, gbl = res[gen_best_i]
        if gbf > best_fit:
            best_fit, best_vec, best_wave = gbf, pop[gen_best_i].copy(), gbw
        sigma = max(0.03, sigma * 0.92)        # anneal exploration
        history.append({"gen": gen, "best_wave": gbw, "best_fit": round(gbf, 3),
                        "mean_wave_top": round(float(np.mean([res[i][1] for i in elites])), 2)})
        log(f"gen {gen+1}/{GENS}: best_wave={gbw:.2f} (dist {sorted(gbwaves)}) "
            f"lives_end={gbl:.1f} score={gbs:.0f} "
            f"| top{MU}_mean_wave={np.mean([res[i][1] for i in elites]):.2f} "
            f"| ALLTIME best_wave={best_wave:.2f} (fit {best_fit:.2f}) | sigma={sigma:.3f}")
        # checkpoint best each gen
        outf.write_text(json.dumps({
            "tag": TAG, "best_fit": best_fit, "best_wave": best_wave,
            "best_params": {NAMES[i]: (int(best_vec[i]) if ISINT[i] else round(float(best_vec[i]), 1)) for i in range(DIM)},
            "mean_params": {NAMES[i]: (int(round(mean[i])) if ISINT[i] else round(float(mean[i]), 1)) for i in range(DIM)},
            "defaults": {NAMES[i]: DEF[i] for i in range(DIM)},
            "history": history, "gen_done": gen + 1,
        }, indent=2))

    for _ in workers:
        task_q.put(None)
    for p in workers:
        p.join(timeout=10)
    log(f"DONE. best_wave={best_wave:.2f} best_fit={best_fit:.3f} -> {outf}")


if __name__ == "__main__":
    main()
