"""Aggregate death-forensics JSONL records into an audit report.

Reads MAME_DEATH_LOG_DIR/deaths_rank*.jsonl, reports:
  - verdict counts (explained / explained-unmapped / unexplained)
  - per-wave breakdown
  - killer composition for explained deaths (which lethal type, vanished or not)
  - top unmapped SWs implicated in explained-unmapped deaths
  - sample records for unexplained deaths (the ones to investigate)

Usage: .venv/bin/python3 mame_gym/death_audit.py <log_dir>
"""
import glob
import json
import sys
from collections import Counter, defaultdict

log_dir = sys.argv[1]
records = []
for path in glob.glob(f"{log_dir}/deaths_rank*.jsonl"):
    with open(path) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass

print(f"total deaths logged: {len(records)}")
if not records:
    sys.exit(0)

verdicts = Counter(r["verdict"] for r in records)
print("\nverdicts:")
for v, c in verdicts.most_common():
    print(f"  {v:<22s} {c:>6d}  ({100*c/len(records):.1f}%)")

print("\nper-wave verdicts:")
by_wave = defaultdict(Counter)
for r in records:
    by_wave[r["wave"]][r["verdict"]] += 1
for w in sorted(by_wave):
    c = by_wave[w]
    tot = sum(c.values())
    print(f"  wave {w}: total={tot} " + " ".join(f"{k}={v}" for k, v in c.most_common()))

print("\nkiller composition (nearest in-range suspect of explained deaths):")
killers = Counter()
for r in records:
    if r["verdict"] != "explained":
        continue
    in_range = [e for e in r["suspects"] if e["d"] <= 15 and e["name"]]
    if in_range:
        e = in_range[0]
        killers[(e["name"], "vanished" if e.get("vanished") else "present")] += 1
for (name, v), c in killers.most_common(15):
    print(f"  {name:<18s} {v:<9s} {c:>6d}")

print("\nunmapped SWs implicated (explained-unmapped deaths):")
unm = Counter()
for r in records:
    if r["verdict"] != "explained-unmapped":
        continue
    for e in r["suspects"]:
        if e["d"] <= 15 and e["name"] is None:
            unm[e["sw"]] += 1
for sw, c in unm.most_common(15):
    print(f"  {sw}: {c}")

unex = [r for r in records if r["verdict"] == "unexplained"]
print(f"\nunexplained deaths: {len(unex)}")
for r in unex[:5]:
    print(f"  wave={r['wave']} score={r['score']} player={r['player']} suspects={r['suspects'][:4]}")
