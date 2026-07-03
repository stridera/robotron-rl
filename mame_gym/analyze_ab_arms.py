"""Wedge-aware A/B analysis for search_value.py eval logs.

Handles the three log generations (see EXPERIMENT_STATE 2026-07-01):
1. Pre-self-heal runs: a "[mame_bridge] instance wedged" marker means the fresh
   MAME's reseed LCG RESTARTED — every later game in that segment replays seeds
   from 1 and is dropped (plus the recovery-artifact game itself).
2. Self-healing runs: the game line following a wedge carries a
   "RECOVERED-INVALID" tag; only that game is dropped — the harness burned
   resets to realign, so later games are seed-correct.
3. Multi-segment logs (>> appends after manual VSEARCH_SEED_SKIP resumes):
   segments split on the "loaded value model" banner; game numbers are
   seed-aligned in every segment; first occurrence of a seed wins.

Usage:
  python mame_gym/analyze_ab_arms.py BASE_LOG[,BASE_LOG2...] ARM_LOG[,ARM_LOG2..] [label]
"""
from __future__ import annotations

import math
import re
import statistics as st
import sys

GAME = re.compile(r"game (\d+): wave=(\d+) score=(\d+) steps=(\d+)( RECOVERED-INVALID)?")


def parse_log(path: str) -> dict[int, tuple[int, int, int]]:
    """Return {seed: (wave, score, steps)} honoring wedge/self-heal semantics."""
    out: dict[int, tuple[int, int, int]] = {}
    seg: list[tuple[int, int, int, int]] = []
    poisoned = False

    def flush():
        nonlocal seg
        for g, w, s, steps in seg:
            out.setdefault(g, (w, s, steps))
        seg = []

    for line in open(path):
        if "loaded value model" in line:      # new run segment (fresh process)
            flush(); poisoned = False; continue
        if "instance wedged" in line:
            flush(); poisoned = True; continue
        m = GAME.match(line)
        if not m:
            continue
        if m.group(5):                        # RECOVERED-INVALID => self-healing run:
            poisoned = False                  # drop this game, later ones realigned
            continue
        if poisoned:                          # old harness: seed sequence restarted
            continue
        seg.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))))
    flush()
    return out


def load(paths: str) -> dict[int, tuple[int, int, int]]:
    merged: dict[int, tuple[int, int, int]] = {}
    for p in paths.split(","):
        for k, v in parse_log(p).items():
            merged.setdefault(k, v)
    return merged


def main():
    base = load(sys.argv[1])
    arm = load(sys.argv[2])
    label = sys.argv[3] if len(sys.argv) > 3 else "ARM"
    seeds = sorted(set(base) & set(arm))
    if not seeds:
        print("no common seeds"); return
    bw = [base[s][0] for s in seeds]; aw = [arm[s][0] for s in seeds]
    bs = [base[s][1] for s in seeds]; asc = [arm[s][1] for s in seeds]
    dw = [a - b for a, b in zip(aw, bw)]
    se = st.stdev(dw) / math.sqrt(len(dw)) if len(dw) > 1 else float("inf")
    mb, ma = st.mean(bw), st.mean(aw)
    # per-seed correlation — r~0.18 even for near-identical configs (chaotic
    # divergence), so treat W/L as descriptive, means as the verdict
    num = sum((a - ma) * (b - mb) for a, b in zip(aw, bw))
    den = (sum((a - ma) ** 2 for a in aw) * sum((b - mb) ** 2 for b in bw)) ** 0.5
    r = num / den if den else 0.0
    print(f"{label}: n={len(seeds)} common seeds "
          f"(base has {len(base)}, arm has {len(arm)})")
    print(f"  WAVE  base {mb:.2f}/med {st.median(bw)}/max {max(bw)} | "
          f"arm {ma:.2f}/med {st.median(aw)}/max {max(aw)}")
    print(f"  SCORE base {st.mean(bs)/1000:.0f}k | arm {st.mean(asc)/1000:.0f}k")
    print(f"  delta {st.mean(dw):+.2f} 95%CI[{st.mean(dw)-1.96*se:+.2f},{st.mean(dw)+1.96*se:+.2f}] "
          f"W/L/T {sum(d>0 for d in dw)}/{sum(d<0 for d in dw)}/{sum(d==0 for d in dw)} r={r:.3f}")


if __name__ == "__main__":
    main()
