"""eval_protocol.py — the permanent promotion-gate evaluation (per the 2026-06-27 review).

Fixes the "40-game samples reverse conclusions" problem with:
  * a reserved HELD-OUT port (default 9970) whose reseed LCG sequence is disjoint from
    training ports (9940-47) and FSM-evolution ports (9920-31) — so eval seeds are never
    trained on. The reseed RNG is deterministic per port (rng_seed = PORT*2749+1, advanced
    once per reset), so a fresh server on a port replays the SAME seed sequence every run
    => two models on the same port are PAIRED by construction.
  * N>=100 continuous wave-1 games (full lives, no save-state resets).
  * survival curve P(reach wave>=k), bootstrap 95% CIs on mean & median.
  * life economy: extra lives earned, net lives, P(earn an extra life).
  * paired mode (two models): per-seed wave delta, win/tie/loss, bootstrap CI on the delta.

Promote ONLY on continuous wave-1 results here — never on deep-reset performance.

Usage:
  single: MAME_RL_RESEED=1 .venv/bin/python mame_gym/eval_protocol.py <model.zip> <vec.pkl> [N] [PORT]
  paired: MAME_RL_RESEED=1 .venv/bin/python mame_gym/eval_protocol.py <A.zip> <vecA.pkl> [N] [PORT] --vs <B.zip> <vecB.pkl>
"""
import sys, os, statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "mame_gym"))
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
from gymnasium.spaces import Box, MultiDiscrete
from mame_robotron_env import MameRobotronEnv

argv = sys.argv[1:]
VS = None
if "--vs" in argv:
    i = argv.index("--vs"); VS = argv[i + 1:i + 3]; argv = argv[:i]
MODEL_A, VEC_A = argv[0], argv[1]
N = int(argv[2]) if len(argv) > 2 else 100
PORT = int(argv[3]) if len(argv) > 3 else 9970
WAVE_MARKS = [3, 5, 8, 10, 12, 15, 20, 25]
EXTRA_LIFE_INTERVAL = 25000   # CMOS extra_man_every default ("Recommended"); robomame.asm $CC00


class _D(gym.Env):
    observation_space = Box(-np.inf, np.inf, (945,), np.float32)
    action_space = MultiDiscrete([8, 8])
    def reset(self, *, seed=None, options=None): return np.zeros(945, np.float32), {}
    def step(self, a): return np.zeros(945, np.float32), 0.0, False, False, {}


def _norm(venv, obs):
    return venv.normalize_obs(obs.reshape(1, -1).astype(np.float32))[0]


def _make_fsm_actor(json_path, with_clearance=False):
    """FSM 'model': spec 'fsm:<evolved.json>' (or 'fsm:default'). Returns actor(packet)->action,
    using the obs-limited evolved chooseOutputs (same 41-slot view as the policy obs).
    with_clearance (spec 'clear:<evolved.json>') wraps the FSM with the champion's
    minimal-deviation clearance override (clearance_planner.py, VSEARCH_* env config)."""
    import json
    from mame_obs import MameObsBuilder
    from position_wrapper import SLOT_CATEGORIES, CATCHALL_SLOTS
    import robotron_fsm as fsm
    Wpx, Hpx = 665, 492
    fsm.DEBUG_LEVEL = 0
    fsm.MAX_RIGHT, fsm.MAX_TOP = Wpx, Hpx
    fsm.MAX_BOTTOM, fsm.MAX_LEFT = 0, 0
    fsm.Y_AXIS_INVERSION = Hpx
    if json_path != "default" and Path(json_path).exists():
        for nm, vl in json.loads(Path(json_path).read_text())["best_params"].items():
            setattr(fsm, nm, vl)
    B = float(getattr(fsm, "BORDER_ADJUST", 20))
    fsm.ADJ_TOP, fsm.ADJ_BOTTOM = Hpx - 2, 0 + B + 9
    fsm.ADJ_LEFT, fsm.ADJ_RIGHT = 0 + 2, Wpx - B
    builder = MameObsBuilder()
    if with_clearance:
        from clearance_planner import clearance_search
    def actor(packet):
        sprites = builder._sprites_from_packet(packet)
        player = next((s for s in sprites if s[2] == "Player"), None)
        if player is None:
            return np.array([0, 0])
        px, py = player[0], player[1]
        others = [s for s in sprites if s[2] != "Player"]
        def d2(s): return (s[0] - px) ** 2 + (s[1] - py) ** 2
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
        if with_clearance:
            mv, fr = clearance_search(sprites, mv, fr)
        return np.array([mv - 1, fr - 1])
    return actor


def play_games(env, model_path, vec_path, n):
    """Run n continuous wave-1 games. Returns list of per-game dicts. Fresh server reset of
    the port's seed sequence happens because env was just constructed on this PORT."""
    is_fsm = str(model_path).startswith(("fsm:", "clear:"))
    if str(model_path).startswith("clear:"):
        actor = _make_fsm_actor(model_path[6:], with_clearance=True)
    elif is_fsm:
        actor = _make_fsm_actor(model_path[4:])
    else:
        venv = VecNormalize.load(vec_path, DummyVecEnv([_D])); venv.training = False
        model = PPO.load(model_path, device="cpu")
    games = []
    # Self-healing seed alignment (same scheme as search_value.py): the reseed
    # LCG advances once per reset; a wedge relaunch starts a FRESH process whose
    # sequence restarts, so we burn resets to realign — games stay seed-indexed.
    # EVAL_SEED_SKIP resumes a partial run at seed K+1.
    seed_skip = int(os.environ.get("EVAL_SEED_SKIP", "0"))
    process_resets = 0
    for g in range(seed_skip, seed_skip + n):
        while process_resets < g:
            env.reset(); process_resets += 1
        obs, info = env.reset(); process_resets += 1
        mw = info["wave"]; start_lives = info["lives"]; last_score = info["score"]
        recovered = False
        while True:
            if is_fsm:
                a = actor(env._last_packet)
            else:
                a, _ = model.predict(_norm(venv, obs), deterministic=True)
            obs, _, term, trunc, info = env.step(a)
            mw = max(mw, info.get("wave", 0))
            if info.get("recovered"):
                recovered = True
                process_resets = 1   # fresh process; its safe-reset consumed seed 1
            if not (term or trunc): last_score = info.get("score", last_score)
            if term or trunc: break
        # Extra lives from SCORE (flicker-free; the $BDEC lives counter spuriously
        # flickers 1->2->1 during the 0x7F wave-transition). Default interval 25000
        # (CMOS extra_man_every "Recommended"); robomame.asm $CC00.
        extra = last_score // EXTRA_LIFE_INTERVAL
        if not recovered:
            games.append({"wave": mw, "score": last_score, "start_lives": start_lives,
                          "extra_lives": extra, "terminated": bool(term)})
        tag = " RECOVERED-INVALID" if recovered else ""
        print(f"  game {g+1}/{seed_skip + n}: wave={mw} score={last_score} "
              f"extra_lives~{extra}{tag}", flush=True)
    return games


def boot_ci(vals, fn, iters=2000, seedbase=12345):
    arr = np.array(vals, dtype=float); m = len(arr)
    rng = np.random.default_rng(seedbase)
    stats = [fn(arr[rng.integers(0, m, m)]) for _ in range(iters)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def report(label, games):
    waves = [g["wave"] for g in games]
    scores = [g["score"] for g in games]
    earned = [g["extra_lives"] for g in games]
    mlo, mhi = boot_ci(waves, np.mean)
    dlo, dhi = boot_ci(waves, np.median)
    tot_w = sum(waves); tot_earned = sum(earned)
    # deaths in a terminated game = all lives consumed = start_lives + extra_lives earned.
    tot_deaths = sum(g["start_lives"] + g["extra_lives"] for g in games if g["terminated"])
    tot_w_term = sum(g["wave"] for g in games if g["terminated"]) or 1
    print(f"\n=== {label}: {len(games)} continuous wave-1 games (held-out port {PORT}) ===")
    print(f"wave: mean={statistics.mean(waves):.2f} [95% {mlo:.2f}-{mhi:.2f}]  "
          f"median={statistics.median(waves):.1f} [95% {dlo:.1f}-{dhi:.1f}]  "
          f"min={min(waves)} max={max(waves)}")
    print("survival: " + "  ".join(f"P(w>={k})={sum(w>=k for w in waves)/len(waves):.2f}" for k in WAVE_MARKS))
    print(f"score: mean={statistics.mean(scores):.0f} max={max(scores)}")
    # life economy (score-based, flicker-free). Sustainable toward wave 100 needs
    # life-gen/wave >= deaths/wave (else the policy bleeds lives no matter the skill).
    print(f"life-economy: score/wave={statistics.mean(scores)/ (tot_w/len(games)):.0f}  "
          f"extra-lives/game={statistics.mean(earned):.2f}  "
          f"life-gen/wave={tot_earned/max(tot_w,1):.2f}  deaths/wave={tot_deaths/tot_w_term:.2f}")
    print(f"dist: {dict(sorted(Counter(waves).items()))}")
    return waves


def main():
    env = MameRobotronEnv(rank=0, base_port=PORT, frameskip=4, reset_pool=[0], obs_mode="slot")
    # RESIDUAL_EVAL=1 -> wrap so a residual model's MultiDiscrete([2,8])=(gate,move) output
    # is applied (gate FSM movement, FSM fire). Compare to pure FSM (fsm: mode) on same port.
    if os.environ.get("RESIDUAL_EVAL") == "1":
        import sys as _s; _s.path.insert(0, str(Path(__file__).parent))
        from residual_env import ResidualWrapper
        env = ResidualWrapper(env)
        print("# RESIDUAL_EVAL: env wrapped in ResidualWrapper", flush=True)
    print(f"# model A: {MODEL_A}", flush=True)
    games_a = play_games(env, MODEL_A, VEC_A, N)
    waves_a = report(f"A = {Path(MODEL_A).parent.name}/{Path(MODEL_A).stem}", games_a)
    if VS:
        print(f"\n# model B: {VS[0]}", flush=True)
        games_b = play_games(env, VS[0], VS[1], N)   # same port => same seed sequence => PAIRED
        waves_b = report(f"B = {Path(VS[0]).parent.name}/{Path(VS[0]).stem}", games_b)
        d = np.array(waves_b) - np.array(waves_a)     # paired per-seed delta (B - A)
        wins = int((d > 0).sum()); losses = int((d < 0).sum()); ties = int((d == 0).sum())
        lo, hi = boot_ci(list(d), np.mean)
        print(f"\n=== PAIRED B-A (same {N} seeds) ===")
        print(f"mean wave delta = {d.mean():+.2f} [95% {lo:+.2f}..{hi:+.2f}]  "
              f"B wins {wins} / ties {ties} / A wins {losses}")
        verdict = ("B better" if lo > 0 else "A better" if hi < 0 else "INCONCLUSIVE (CI spans 0)")
        print(f"verdict: {verdict}")
    env.close()


if __name__ == "__main__":
    main()
