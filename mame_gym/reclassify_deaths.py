"""Re-classify an existing death-forensics log with the CORRECTED sw table.

The logged 'name' fields were produced by the buggy table (tank shells + tanks
labeled 'Quark'; tank phantom $4800). Each suspect still carries its raw `sw`,
so we re-map offline and show the killer-composition shift — no games re-run.

Usage: .venv/bin/python3 mame_gym/reclassify_deaths.py <deaths_dir>
"""
import glob, json, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from mame_obs import classify_sw   # corrected table

log_dir = sys.argv[1] if len(sys.argv) > 1 else "deaths_combteacher"
recs = []
for p in glob.glob(f"{log_dir}/deaths_rank*.jsonl"):
    for line in open(p):
        try: recs.append(json.loads(line))
        except json.JSONDecodeError: pass

print(f"{log_dir}: {len(recs)} deaths\n")

def killer(susp):
    """Nearest suspect within kill radius (15) that classifies to a name."""
    for e in susp:  # already distance-sorted
        if e["d"] <= 15:
            yield e

old = Counter(); new = Counter()
# how each OLD label re-maps under the corrected table
remap = defaultdict(Counter)
# per-wave NEW killer composition
by_wave_new = defaultdict(Counter)

for r in recs:
    ks = list(killer(r["suspects"]))
    if not ks:
        continue
    # OLD killer = first in-range with a (logged) name
    ok = next((e for e in ks if e.get("name")), None)
    # NEW killer = first in-range whose corrected classify is non-None
    def newname(e): return classify_sw(int(e["sw"], 16))
    nk = next((e for e in ks if newname(e) is not None), None)
    if ok:
        old[ok["name"]] += 1
    if nk:
        nn = newname(nk)
        new[nn] += 1
        by_wave_new[r["wave"]][nn] += 1
    # track remap for every in-range suspect
    for e in ks:
        remap[e.get("name")][newname(e)] += 1

def show(title, c):
    tot = sum(c.values())
    print(title)
    for name, n in c.most_common():
        print(f"  {str(name):<16s} {n:>5d}  ({100*n/tot:.1f}%)")
    print()

show("OLD killer composition (logged names):", old)
show("NEW killer composition (corrected table):", new)

print("Re-map of the OLD 'Quark' killers (what they REALLY were):")
q = remap.get("Quark", Counter())
tot = sum(q.values())
for name, n in q.most_common():
    print(f"  Quark(old) -> {str(name):<12s} {n:>5d}  ({100*n/tot:.1f}%)")
print()

print("Per-wave NEW killer composition (waves 5-22):")
for w in sorted(by_wave_new):
    if 5 <= w <= 22:
        c = by_wave_new[w]
        tot = sum(c.values())
        top = " ".join(f"{k}={v}" for k, v in c.most_common(4))
        print(f"  wave {w:>2d} (n={tot:>3d}): {top}")
